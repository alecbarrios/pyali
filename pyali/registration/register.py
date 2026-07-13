"""Cross-modal (voltage JF608 -> SBS JF608) registration — CORE (no QC).

Given the per-well SBS mosaic (:func:`mosaic.build_sbs_mosaic`) this module locates each voltage
FOV's raw-mean JF608 reference inside that mosaic and returns the affine mapping the FOV into mosaic
pixels, so voltage cells can later be matched to SBS cells (:mod:`assign`).

Three stages, mirroring the reference global-then-fine registration but with the stage<->camera
orientation recovered empirically and the cross-modal scale calibrated (never hard-coded):

* :func:`recover_dihedral` — which of the 8 flips/rotations aligns the FOV to the mosaic (recover
  ONCE per dataset and freeze — D8).
* :func:`global_register` — bootstrap: coarsely LOCALIZE the confident FOVs in the DOWNSAMPLED
  mosaic, then fit ONE similarity ``S`` : voltage stage-µm -> mosaic-px that predicts every FOV.
* :func:`fine_register` — per FOV: crop the FULL-RES mosaic at ``S``'s prediction and register the
  FOV into that crop for an accurate ``T`` : flipped-FOV -> mosaic-px.

Conventions live entirely in :mod:`coordinates` (points ``(x, y) = (col, row)``; column-vector
affines ``M @ [x, y, 1]``). Everything works in the DIHEDRAL-FLIPPED FOV frame [E2]: the dihedral op
is applied to the reference first, and recovered scale/rotation/translation refer to that flipped
image. All coarse work is on a DOWNSAMPLED mosaic [E3]. The cross-modal FOV-px->mosaic-px scale is
UNKNOWN a priori (voltage µm/px is unreliable), so it is *searched* over a grid and *calibrated*
from the first confident FOV [E9]; it is never hard-coded to the old 4.35x optical ratio.

Coarse localizer — DEVIATION FROM §4 (design decision D9, flagged for manual review)
------------------------------------------------------------------------------------
§4's pseudocode localizes each FOV with ``phasecorr.register(downscaled_FOV, whole_downscaled_
mosaic, "similarity")`` (Fourier–Mellin). On the real P-1/W-A1 well this returns **NCC = -1.0 for
all 8 dihedral ops, every FOV**: Fourier–Mellin/phase-correlation assumes the two images are the
SAME scene under a global similarity, but a single 2.3x FOV is a tiny (~0.5% area), ~3x-scaled
*sub-region* of the ~28k^2 well mosaic — FM cannot localize a small template inside a big image, and
the garbage transform warps the FOV off-frame (< 16 valid px -> the -1.0 sentinel). So the coarse
localizer here is instead a **scale-aware, variance-guarded normalized-cross-correlation template
match** (:func:`_localize`): rescale the (dihedral-applied) FOV to each hypothesized mosaic scale,
slide it over the downsampled mosaic, and take the strongest peak whose mosaic window has real
variance (the guard rejects spurious peaks in the mosaic's near-constant black gaps between sparse
tiles). This *fulfils §2.3's stated intent* ("intensity-register a few FOV references directly
against the mosaic to localize them, then fit S") with a tool that can actually localize a
sub-image; it still honours E3 (works on the downsampled mosaic). :func:`fine_register` keeps
Fourier–Mellin because there the FOV fills its crop (full overlap) — FM's correct regime.

NOTE (real data, honestly reported): even with this localizer the voltage raw-mean references do not
*robustly* lock onto the SBS JF608 mosaic (weak, dihedral-inconsistent, spatially-inconsistent
peaks). See the dev harness A4 "SCIENTIFIC CRUX" report; the cross-modal lock is an open problem for
the §10 config-notebook tuning loop (band-pass sigmas / reference / partial-overlap + arc masking).
The CODE here is verified correct on synthetic ground truth.
"""
from __future__ import annotations

import numpy as np

from . import coordinates, phasecorr
from .coordinates import (DIHEDRAL_NAMES, apply_affine, apply_dihedral_image,
                          compose, translation)

# FOV-px -> mosaic-px scale search grid (a SEARCH range, not a hard-coded scale — E9 still
# calibrates the working scale from the first confident FOV and gates the rest by scale_tol).
DEFAULT_SCALES = (2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75)
# Difference-of-Gaussians band-pass (emphasize cell-blob structure, drop shading); §10-tunable.
BAND_LOW_SIGMA = 1.0
BAND_HIGH_SIGMA = 8.0
MIN_VAR_FRAC = 0.05          # a candidate mosaic window must have >= this * global variance


# --------------------------------------------------------------------------- #
# helpers  [E11]  (§9)
# --------------------------------------------------------------------------- #
def downscale(img, f):
    """[E3] Coarse working copy: block-mean downsample by integer factor ``f``. Returns float32.

    ``downscale_local_mean`` zero-pads to a multiple of ``f`` (a thin dark strip at the far edge —
    negligible for coarse registration). ``f == 1`` is a plain float32 cast.
    """
    from skimage.transform import downscale_local_mean

    a = np.asarray(img)
    if int(f) == 1:
        return a.astype(np.float32)
    return downscale_local_mean(a.astype(float), (int(f), int(f))).astype(np.float32)


def _center_xy(img):
    """Geometric center ``(x, y) = ((W-1)/2, (H-1)/2)`` of an image, as a ``(2,)`` float array."""
    H, W = img.shape[:2]
    return np.array([(W - 1) / 2.0, (H - 1) / 2.0], float)


def _recovered_scale(M):
    """Isotropic scale of a column-vector similarity = ``sqrt(|det(linear part)|)`` (>= 0)."""
    A = np.asarray(M, float)[:2, :2]
    return float(np.sqrt(abs(np.linalg.det(A))))


def _recovered_rotation_deg(M):
    """Rotation (degrees, in ``[-180, 180]``) of a column-vector similarity from its linear part."""
    A = np.asarray(M, float)
    return float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))


def _crop_around(mosaic, cx, cy, half):
    """[E8] Crop ``mosaic`` to a window of half-extent ``half=(hr, hc)`` about center ``(cx, cy)``.

    Returns ``(crop, (col0, row0))`` — the crop and its top-left origin as ``(x0=col0, y0=row0)`` so
    a caller can ``compose(translation(col0, row0), M_ref_to_crop)`` to lift a crop-local transform
    back to mosaic coordinates. The window is clipped to the mosaic and is guaranteed non-empty even
    if the predicted center lands off the mosaic: the upper bounds are clipped to ``>= r0+1`` /
    ``>= c0+1`` so a bad ``S`` yields a thin low-NCC crop rather than an empty-array error.
    """
    hr, hc = half
    H, W = mosaic.shape[:2]
    r0 = int(np.clip(round(cy) - hr, 0, H - 1))
    r1 = int(np.clip(round(cy) + hr, r0 + 1, H))
    c0 = int(np.clip(round(cx) - hc, 0, W - 1))
    c1 = int(np.clip(round(cx) + hc, c0 + 1, W))
    return mosaic[r0:r1, c0:c1], (c0, r0)


def _band_pass(img, low=BAND_LOW_SIGMA, high=BAND_HIGH_SIGMA):
    """Difference-of-Gaussians band-pass (cross-modal cell-blob emphasis)."""
    from skimage.filters import difference_of_gaussians

    return difference_of_gaussians(np.asarray(img, float), low, high)


def _ncc_peak(fixed_bp, tmpl_bp, min_var_frac=MIN_VAR_FRAC):
    """Variance-guarded zero-mean NCC of a small template over ``fixed_bp`` (both band-passed).

    Returns ``(peak_ncc in [-1, 1], (row0, col0))`` where ``(row0, col0)`` is the template TOP-LEFT
    at the peak. Uses FFT cross-correlation; any location whose ``fixed`` window variance is below
    ``min_var_frac * fixed.var()`` is excluded so the mosaic's near-constant black gaps between
    sparse tiles cannot manufacture a spurious peak (which is what breaks ``match_template``'s own
    normalization). Returns ``(-1.0, (0, 0))`` if no valid location exists.
    """
    from scipy.signal import fftconvolve

    th, tw = tmpl_bp.shape
    t = tmpl_bp - tmpl_bp.mean()
    tnorm = float(np.sqrt((t * t).sum()))
    if tnorm == 0.0:
        return -1.0, (0, 0)
    ones = np.ones((th, tw), float)
    num = fftconvolve(fixed_bp, t[::-1, ::-1], mode="valid")        # sum(f_win * (t - t_mean))
    wsum = fftconvolve(fixed_bp, ones, mode="valid")
    wsq = fftconvolve(fixed_bp * fixed_bp, ones, mode="valid")
    n = th * tw
    wvar = np.maximum(wsq / n - (wsum / n) ** 2, 0.0)
    wnorm2 = np.maximum(wsq - wsum * wsum / n, 0.0)                 # ||f_win - f_mean||^2
    denom = np.sqrt(wnorm2) * tnorm
    resp = np.full(num.shape, -np.inf)
    ok = denom > 0
    np.divide(num, denom, out=resp, where=ok)
    resp[wvar < min_var_frac * float(fixed_bp.var())] = -np.inf
    if not np.isfinite(resp).any():
        return -1.0, (0, 0)
    r, c = np.unravel_index(int(np.argmax(resp)), resp.shape)
    return float(resp[r, c]), (int(r), int(c))


def _localize(mosaic_bp, fov_flipped, f, scales=DEFAULT_SCALES, low=BAND_LOW_SIGMA,
              high=BAND_HIGH_SIGMA, min_var_frac=MIN_VAR_FRAC):
    """Scale-aware localization of a dihedral-applied FOV in the band-passed downsampled mosaic.

    For each hypothesized FOV-px->mosaic-px ``scale``, rescale ``fov_flipped`` to the size it would
    occupy in the downsampled mosaic (``scale / f``), band-pass it, and template-match it
    (:func:`_ncc_peak`). Returns ``(best_scale, peak_ncc, center_ds_xy)`` where ``center_ds_xy`` is
    the ``(x, y)`` center of the matched template in DOWNSCALED-mosaic pixels (multiply by ``f`` for
    full-res). Rotation is not searched here (the frozen dihedral handles 90deg steps; a small
    residual rotation is recovered later by :func:`fine_register`).
    """
    from skimage.transform import rescale

    Hm, Wm = mosaic_bp.shape
    fov = np.asarray(fov_flipped, float)
    if fov.size == 0:                                          # degenerate reference -> no localization
        return float(scales[0]), -np.inf, np.zeros(2)
    best = (float(scales[0]), -np.inf, np.zeros(2))
    for s in scales:
        zoom = float(s) / f
        tmpl = rescale(fov, zoom, order=1, anti_aliasing=True, preserve_range=True)
        th, tw = tmpl.shape
        if min(th, tw) < 8 or th >= Hm or tw >= Wm:
            continue
        pk, (r, c) = _ncc_peak(mosaic_bp, _band_pass(tmpl, low, high), min_var_frac)
        if pk > best[1]:
            center = np.array([c + tw / 2.0, r + th / 2.0], float)   # (x, y) center, downscaled px
            best = (float(s), pk, center)
    return best


# --------------------------------------------------------------------------- #
# stage 1 — dihedral orientation (once per dataset)
# --------------------------------------------------------------------------- #
def recover_dihedral(fov_ref, mosaic, f=12, scales=DEFAULT_SCALES, mosaic_bp=None,
                     return_scores=False):
    """Recover the dihedral op that aligns ``fov_ref`` to ``mosaic`` (best scale-aware NCC).

    Localizes each of the 8 flips/rotations of the FOV reference in the downsampled mosaic and keeps
    the one whose best scale-match NCC is highest. D8: the voltage<->SBS orientation is a fixed
    property of the two microscopes' mounting, so recover this ONCE per dataset and pass the frozen
    op to :func:`global_register` / :func:`fine_register`.

    ``mosaic_bp`` reuses a precomputed band-passed downsampled mosaic
    (``_band_pass(downscale(mosaic, f))``) — pass it to avoid re-downsampling the ~28k^2 array in
    both this call and :func:`global_register`. With ``return_scores`` also returns ``{op: ncc}``.
    """
    mo = mosaic_bp if mosaic_bp is not None else _band_pass(downscale(mosaic, f))
    scores = {}
    for op in DIHEDRAL_NAMES:
        _s, pk, _c = _localize(mo, apply_dihedral_image(op, fov_ref), f, scales)
        scores[op] = float(pk)
    best = max(scores, key=scores.get)
    return (best, scores) if return_scores else best


# --------------------------------------------------------------------------- #
# stage 2 — global bootstrap: stage-µm -> mosaic-px similarity
# --------------------------------------------------------------------------- #
def global_register(fovs, mosaic, dihedral, f=12, min_ncc=0.30, scale_tol=0.5,
                    scales=DEFAULT_SCALES, mosaic_bp=None, return_report=False):
    """Bootstrap the global similarity ``S`` : voltage stage-µm -> full-res mosaic-px.

    For each FOV: coarsely LOCALIZE its dihedral-flipped reference in the DOWNSAMPLED mosaic [E3],
    gate on ``ncc >= min_ncc`` and on the recovered FOV-px->mosaic-px scale, then anchor the FOV's
    (flipped-frame [E2]) center to its localized mosaic position and its stage position. Finally fit
    a 4-DOF similarity through those anchors [E1] (the dihedral op has already absorbed any
    reflection, so a proper similarity is the right constraint).

    Scale gate [E9, D9]: the cross-modal scale is UNKNOWN a priori, so ``expected`` is calibrated
    from the MEDIAN scale of all FOVs that clear ``min_ncc`` (robust and ORDER-INDEPENDENT), and
    every passer — including the ones defining the fit — must fall in
    ``(expected/(1+scale_tol), expected*(1+scale_tol))`` or it is dropped. (The original "first
    confident FOV" calibration was order-dependent: an adversarial review confirmed a single
    spurious first FOV could mis-calibrate the gate and reject every valid FOV. The median is
    robust with >= 3 passers.) Nothing is hard-coded to the old 4.35x optical ratio.

    Returns ``(S, expected)`` — ``S`` is ``(3, 3)`` float64, ``expected`` the calibrated (median)
    scale. Needs >= 2 FOVs through the gates to fit ``S``; with fewer it raises ``RuntimeError``
    (or, with ``return_report=True``, returns ``(None, expected, report)`` so a caller can inspect
    why — surfacing the "do the modalities lock?" crux instead of crashing). ``return_report``
    returns a third element: a per-FOV list of ``{fov, ncc, scale, center_px, passed, reason}``.
    ``mosaic_bp`` reuses a precomputed band-passed downsampled mosaic.
    """
    mo = mosaic_bp if mosaic_bp is not None else _band_pass(downscale(mosaic, f))
    report, passers = [], []
    for k, fov in enumerate(fovs):
        di = apply_dihedral_image(dihedral, fov["reference"])                  # [E2] flipped frame
        s, c, center_ds = _localize(mo, di, f, scales)
        center_px = center_ds * f                                             # -> full-res mosaic px
        rec = dict(fov=fov.get("fov", k), ncc=float(c), scale=float(s),
                   center_px=center_px.tolist(), passed=False, reason="")
        report.append(rec)
        if c < min_ncc:
            rec["reason"] = f"ncc {c:.3f} < min_ncc {min_ncc:.2f}"
            continue
        passers.append((rec, float(s), center_px, np.asarray(fov["stage_xy"], float)))

    # [E9, D9] robust, order-independent scale calibration: median of all min_ncc passers, then gate
    # EVERY passer about it (so a lone spurious scale outlier cannot mis-calibrate and reject valid FOVs).
    expected = float(np.median([p[1] for p in passers])) if passers else None
    a_stage, a_px = [], []
    if expected is not None:
        lo, hi = expected / (1 + scale_tol), expected * (1 + scale_tol)
        for rec, s, center_px, stage in passers:
            if not (lo < s < hi):
                rec["reason"] = f"scale {s:.3f} outside ({lo:.3f}, {hi:.3f})"
                continue
            rec["passed"] = True
            a_px.append(center_px)
            a_stage.append(stage)

    n_ok = len(a_stage)
    if n_ok < 2:
        msg = (f"global_register: only {n_ok} FOV(s) cleared the NCC>={min_ncc:.2f} + scale gates; "
               f"need >= 2 to fit a similarity. Per-FOV: "
               + "; ".join(f"fov{r['fov']} ncc={r['ncc']:.3f} scale={r['scale']:.3f}"
                           + ("" if r["passed"] else f" [{r['reason']}]") for r in report))
        if return_report:
            return None, expected, report
        raise RuntimeError(msg)

    # 4-DOF similarity [E1]. RANSAC is intentionally NOT used here: coordinates.fit_similarity's
    # residual_threshold (3 px) is far tighter than coarse (f-downsampled) anchor-localization error,
    # so it would reject most anchors; the NCC + scale gates already pre-filter outliers, making a
    # plain least-squares similarity over the gated anchors the more robust choice. (D9 — flagged.)
    S = coordinates.fit_similarity(np.array(a_stage), np.array(a_px))
    if return_report:
        return S, expected, report
    return S, expected


# --------------------------------------------------------------------------- #
# stage 3 — per-FOV fine registration (full resolution)
# --------------------------------------------------------------------------- #
def _prescale_matrix(s):
    """Column-vector affine of :func:`skimage.transform.rescale` by isotropic factor ``s``.

    ``rescale`` aligns pixel-edge extents, so an input coordinate ``x`` maps to output
    ``s*x + 0.5*(s-1)`` (the half-sample edge convention). Encoding that exactly keeps ``fine_register``
    free of a ~``0.5*(s-1)`` px systematic shift.
    """
    s = float(s)
    off = 0.5 * (s - 1.0)
    return np.array([[s, 0.0, off], [0.0, s, off], [0.0, 0.0, 1.0]], float)


def fine_register(fov, mosaic, S, dihedral, scale, half=(1100, 3400)):
    """Per-FOV fine registration: return ``(T, score)`` mapping flipped-FOV -> full-res mosaic-px.

    Predicts this FOV's mosaic center from ``S`` and its stage position, crops the FULL-RES mosaic
    there ([E3]: full resolution only in this small local window, never the whole well), registers
    the dihedral-flipped FOV reference into the crop, and composes the crop origin back in [E8].

    Coarse-to-fine scale (deviation from §4, D9): the cross-modal FOV-px->mosaic-px scale is ~3-4x,
    which is beyond the validated regime of ``phasecorr.register`` (Fourier–Mellin, ~1.35x) and makes
    it fail (NCC -1). So the (dihedral-flipped) FOV is first UPSCALED by the known ``scale`` (=
    ``expected`` from :func:`global_register`) to ~mosaic resolution — where it fills the crop —
    then ``phasecorr.register`` only recovers the RESIDUAL (~1x scale + small rotation + translation).
    ``T = translation(crop_origin) @ M_residual @ prescale`` maps the original flipped-FOV pixels to
    mosaic pixels.

    ``half=(hr, hc)`` is the crop half-extent in (rows, cols); the default comfortably contains a
    312x1200 FOV upscaled ~3-4x when its long (1200) axis maps to mosaic columns after the dihedral.
    (If the frozen dihedral maps the long axis to rows instead, pass ``half=(3400, 1100)``; per-dataset
    knob, flagged.) ``score`` is the phase-correlation NCC of the fine fit.
    """
    from skimage.transform import rescale

    center = apply_affine(S, np.asarray(fov["stage_xy"], float)[None, :])[0]
    cx, cy = float(center[0]), float(center[1])
    crop, (col0, row0) = _crop_around(mosaic, cx, cy, half)
    di = np.asarray(apply_dihedral_image(dihedral, fov["reference"]), float)
    if di.size == 0:                                           # degenerate reference -> no fit
        return np.eye(3), -1.0
    di_up = rescale(di, float(scale), order=1, anti_aliasing=True,
                    preserve_range=True)                       # -> ~mosaic resolution
    M_residual, score = phasecorr.register(di_up, crop, "similarity")
    T = compose(translation(col0, row0), M_residual, _prescale_matrix(scale))   # [E8] + prescale
    return T, float(score)
