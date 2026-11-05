"""Phase-correlation image registration (a reimplementation of the reference ``imregcorr``).

Recovers a *similarity* transform (isotropic scale + rotation + translation) between two images
from the same modality/fluorophore, using the Fourier--Mellin method: the rotation and scale come
from a log-polar phase correlation of the FFT magnitudes, and the residual translation from a
final phase correlation. A ``translation`` mode (plain phase correlation) and a feature-based
fallback are also provided.

Every transform returned is a column-vector affine (see :mod:`pyreg.coordinates`) mapping
**moving -> fixed** coordinates in ``(x, y)``: a point at ``(x, y)`` in ``moving`` lands at
``M @ [x, y, 1]`` in ``fixed``. Each call also returns a confidence in ``[-1, 1]`` = the
normalized cross-correlation of ``fixed`` with ``moving`` warped by ``M`` (higher is better).

Robustness: the FFT magnitude is centrosymmetric, so the recovered rotation has a 180-degree
ambiguity, and the scale direction is easy to get backwards. Rather than track signs by hand we
try all four ``(angle, angle+180) x (scale, 1/scale)`` candidates, warp moving->fixed for each,
and keep the one with the highest cross-correlation. The data picks the right branch.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.fft import fft2, fftshift
from skimage.filters import difference_of_gaussians, window
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp, warp_polar

from .coordinates import compose, invert, similarity_matrix, translation


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_gray_float(img):
    a = np.asarray(img, float)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {a.shape}")
    # robust contrast normalization (registration is intensity-pattern based)
    lo, hi = np.percentile(a, [1, 99])
    if hi > lo:
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    return a


def _pad_to_common(a, b):
    """Center-pad both images to the elementwise-max shape."""
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])

    def pad(x):
        ph, pw = H - x.shape[0], W - x.shape[1]
        return np.pad(x, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))

    return pad(a), pad(b)


def _depad_transform(M, moving_shape, fixed_shape):
    """Re-express a padded-frame affine in ORIGINAL moving->fixed pixel coordinates.

    :func:`_pad_to_common` center-pads both images, so a transform recovered on the padded arrays
    is expressed in padded coordinates. When ``moving`` and ``fixed`` differ in size their pad
    offsets differ, and that difference leaks into the translation of ``M`` (a real bug for callers
    that register a small image against a larger one, e.g. an upscaled FOV against a mosaic crop).
    This undoes the center-pad: ``M_orig = translation(-off_f) @ M @ translation(off_m)``. It is the
    IDENTITY when ``moving`` and ``fixed`` already share a shape (all equal-size callers unchanged).
    """
    H = max(moving_shape[0], fixed_shape[0])
    W = max(moving_shape[1], fixed_shape[1])
    off_m = ((W - moving_shape[1]) // 2, (H - moving_shape[0]) // 2)      # (x, y) top-left pad of moving
    off_f = ((W - fixed_shape[1]) // 2, (H - fixed_shape[0]) // 2)        # (x, y) top-left pad of fixed
    return compose(translation(-off_f[0], -off_f[1]), np.asarray(M, float),
                   translation(off_m[0], off_m[1]))


def _fft_mag(img, low_sigma=5.0, high_sigma=20.0):
    """Band-passed, windowed, shifted FFT magnitude.

    The difference-of-Gaussians band-pass emphasizes the mid frequencies where the log-polar
    scale/rotation signal is clean; the Hann window kills edge artifacts. Translation is discarded
    (it only affects FFT phase, not magnitude), leaving rotation (angular shift) and scale (radial
    log-shift).
    """
    bp = difference_of_gaussians(img, low_sigma, high_sigma)
    w = window("hann", img.shape)
    return np.abs(fftshift(fft2(bp * w)))


def ncc(a, b, mask=None):
    """Normalized cross-correlation (Pearson) over ``mask`` (or all finite overlap)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)
    m = mask & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 16:
        return -1.0
    av, bv = a[m], b[m]
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    if denom == 0:
        return -1.0
    return float((av * bv).sum() / denom)


def _warp_moving_into_fixed(moving, M, out_shape):
    """Render ``moving`` in the fixed frame under column-vector affine ``M`` (moving->fixed).

    Returns ``(warped, valid_mask)``; pixels outside ``moving`` are NaN / False.
    """
    inv = SimilarityTransform(matrix=invert(M))            # fixed-output -> moving-input (x,y)
    warped = warp(moving, inv, output_shape=out_shape, order=1, cval=np.nan, preserve_range=True)
    valid = np.isfinite(warped)
    return warped, valid


# --------------------------------------------------------------------------- #
# rotation + scale from log-polar FFT
# --------------------------------------------------------------------------- #
def _recover_angle_scale(moving, fixed, upsample=10):
    """Return ``(angle_deg, scale)`` of moving relative to fixed from log-polar FFT phase corr."""
    fm = _fft_mag(moving)
    ff = _fft_mag(fixed)
    shape = fm.shape
    radius = shape[0] // 8                                  # low-frequency band (skimage recipe)
    wm = warp_polar(fm, radius=radius, output_shape=shape, scaling="log", order=0)
    wf = warp_polar(ff, radius=radius, output_shape=shape, scaling="log", order=0)
    half = shape[0] // 2                                    # FFT mag is centrosymmetric -> half angles
    wm, wf = wm[:half], wf[:half]
    shifts, _err, _pd = phase_cross_correlation(wf, wm, upsample_factor=upsample,
                                                normalization=None)
    shiftr, shiftc = shifts[0], shifts[1]
    angle_deg = (360.0 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    scale = np.exp(shiftc / klog)
    return angle_deg, scale


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def register_translation(moving, fixed, upsample=10):
    """Translation-only registration. Returns ``(M, confidence)`` mapping moving->fixed."""
    m = _as_gray_float(moving)
    f = _as_gray_float(fixed)
    mp, fp = _pad_to_common(m, f)
    shift, _err, _pd = phase_cross_correlation(fp, mp, upsample_factor=upsample,
                                               normalization=None)
    dy, dx = float(shift[0]), float(shift[1])
    M = similarity_matrix(1.0, 0.0, dx, dy)
    warped, valid = _warp_moving_into_fixed(mp, M, fp.shape)
    conf = ncc(fp, warped, valid)
    return _depad_transform(M, m.shape, f.shape), conf


def _score(mp, fp, M):
    """NCC of ``fp`` with ``mp`` warped by column-vector affine ``M`` (moving->fixed)."""
    warped, valid = _warp_moving_into_fixed(mp, M, fp.shape)
    return ncc(fp, warped, valid)


def _fourier_mellin_candidates(mp, fp, upsample):
    """Yield moving->fixed similarity candidates from the log-polar FFT estimate.

    Emits all four ``(angle, angle+180) x (scale, 1/scale)`` branches (the FFT-magnitude 180-degree
    ambiguity + scale-direction ambiguity), each with its residual translation solved by phase
    correlation. The caller scores them by NCC and keeps the best.
    """
    angle0, scale0 = _recover_angle_scale(mp, fp, upsample=upsample)
    cx, cy = (fp.shape[1] - 1) / 2.0, (fp.shape[0] - 1) / 2.0
    center = SimilarityTransform(translation=(cx, cy))
    uncenter = SimilarityTransform(translation=(-cx, -cy))
    for angle in (angle0, angle0 + 180.0):
        for scale in (scale0, 1.0 / scale0 if scale0 else scale0):
            if not (1e-3 < scale < 1e3):
                continue
            rot = SimilarityTransform(scale=scale, rotation=np.deg2rad(angle))
            M_rs = (center + rot + uncenter).params            # rotate+scale about center
            warped_rs, _ = _warp_moving_into_fixed(mp, M_rs, fp.shape)
            with warnings.catch_warnings():                    # a bad branch can warp to ~empty
                warnings.simplefilter("ignore")
                shift, _err, _pd = phase_cross_correlation(fp, np.nan_to_num(warped_rs),
                                                           upsample_factor=upsample,
                                                           normalization=None)
            dy, dx = float(shift[0]), float(shift[1])
            yield similarity_matrix(1.0, 0.0, dx, dy) @ M_rs


def _feature_candidate(mp, fp, min_matches=8):
    """A moving->fixed similarity from ORB keypoints + RANSAC, or ``None`` if too few matches."""
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac

    orb = ORB(n_keypoints=800, fast_threshold=0.05)
    try:
        orb.detect_and_extract(mp)
        kp_m, d_m = orb.keypoints, orb.descriptors            # (row, col)
        orb.detect_and_extract(fp)
        kp_f, d_f = orb.keypoints, orb.descriptors
    except Exception:
        return None
    matches = match_descriptors(d_m, d_f, cross_check=True)
    if len(matches) < min_matches:
        return None
    src = kp_m[matches[:, 0]][:, ::-1]                         # -> (x, y)
    dst = kp_f[matches[:, 1]][:, ::-1]
    model, inliers = ransac((src, dst), SimilarityTransform, min_samples=3,
                            residual_threshold=2, max_trials=2000)
    if model is None or inliers is None or inliers.sum() < min_matches:
        return None
    return model.params


def register_similarity(moving, fixed, upsample=10, try_features=True):
    """Similarity registration (scale+rotation+translation). Returns ``(M, confidence)``.

    Ensemble: generate candidate transforms from the log-polar FFT (Fourier--Mellin) *and* from
    ORB+RANSAC features, then return whichever maximizes the cross-correlation of ``fixed`` with
    ``moving`` warped by ``M``. FFT handles low-texture / feature-poor images; features handle
    combined scale+rotation robustly. ``M`` maps moving->fixed ``(x, y)``.
    """
    m0 = _as_gray_float(moving)
    f0 = _as_gray_float(fixed)
    mp, fp = _pad_to_common(m0, f0)
    candidates = list(_fourier_mellin_candidates(mp, fp, upsample))
    if try_features:
        Mfeat = _feature_candidate(mp, fp)
        if Mfeat is not None:
            candidates.append(Mfeat)
    best = (np.eye(3), -np.inf)
    for M in candidates:
        s = _score(mp, fp, M)
        if s > best[1]:
            best = (M, s)
    return _depad_transform(best[0], m0.shape, f0.shape), float(best[1])


def register_features(moving, fixed, min_matches=8):
    """Feature-only similarity registration (ORB + RANSAC). Returns ``(M, confidence)``."""
    m0 = _as_gray_float(moving)
    f0 = _as_gray_float(fixed)
    mp, fp = _pad_to_common(m0, f0)
    M = _feature_candidate(mp, fp, min_matches=min_matches)
    if M is None:
        return np.eye(3), -1.0
    return _depad_transform(M, m0.shape, f0.shape), _score(mp, fp, M)


def register(moving, fixed, mode="similarity", **kw):
    """Dispatch: ``mode`` in ``{'similarity', 'translation', 'features'}``. Returns ``(M, conf)``."""
    if mode == "similarity":
        return register_similarity(moving, fixed, **kw)
    if mode == "translation":
        return register_translation(moving, fixed, **kw)
    if mode == "features":
        return register_features(moving, fixed, **kw)
    raise ValueError(f"unknown mode {mode!r}")
