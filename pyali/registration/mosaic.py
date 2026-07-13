"""Build the per-well SBS JF608 mosaic (the common frame) + the shared stage->pixel transform.

Tiles are placed by microscope stage position (µm -> px via the tile pixel size). The stage-vs-camera
axis orientation is one of the 8 dihedral placements and is recovered empirically [E5] by maximizing
the overlap correlation of adjacent tiles. Overlapping regions are blended by occupation-count
averaging [E4] (float32 accumulator + uint16 count). The returned per-tile pixel offsets ``off`` are
the ONE shared placement used by BOTH the image mosaic and the SBS cell centroids (``place_cells``),
which keeps image and cells in lockstep (the correctness invariant).

``A_sbs`` (stage µm -> mosaic px, column-vector affine) is CONSTRUCTED analytically from the
recovered axes + pixel size, NOT fit: the stage->pixel map can be a reflection (a handedness flip),
which a proper similarity (``fit_similarity``) cannot represent. Construction is also exact.
"""
from __future__ import annotations

import os

import numpy as np

from . import coordinates, phasecorr
from .io import read_sbs_ref_tile

TILE = 1480
_HALF = (TILE - 1) / 2.0

# 8 candidate mappings of stage (x, y) axes onto pixel (col, row): sign flips x swap.
STAGE_AXIS_HYPS = [(sx, sy, swap) for swap in (False, True) for sy in (+1, -1) for sx in (+1, -1)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _xy(meta, t):
    r = meta.loc[meta.tile == t, ["x_pos", "y_pos"]].iloc[0]
    return float(r.x_pos), float(r.y_pos)


def _stage_rc(x, y, px, hyp):
    """Stage (x, y) µm -> un-normalized (row, col) px under an axis hypothesis (for relative use)."""
    sx, sy, swap = hyp
    X, Y = x / px, y / px
    if swap:
        X, Y = Y, X
    return sy * Y, sx * X                                    # (row, col) float


def _overlap_ncc(A, B, dr, dc, min_frac=0.03):
    """NCC of the overlap of two TILE-square images when B is offset from A by (dr, dc) px."""
    dr, dc = int(round(dr)), int(round(dc))
    r0a, r1a = max(0, dr), min(TILE, TILE + dr)
    c0a, c1a = max(0, dc), min(TILE, TILE + dc)
    oh, ow = r1a - r0a, c1a - c0a                           # per-dim overlap; guard |offset| >= TILE
    if oh <= 0 or ow <= 0 or oh * ow < min_frac * TILE * TILE:
        return -1.0
    r0b, c0b = max(0, -dr), max(0, -dc)
    subA = A[r0a:r1a, c0a:c1a]
    subB = B[r0b:r0b + oh, c0b:c0b + ow]
    return phasecorr.ncc(subA.astype(float), subB.astype(float))


def _neighbor_pairs(meta, n_pairs):
    """Up to n_pairs strongly axis-aligned nearest-neighbour tile pairs (by stage position)."""
    from scipy.spatial import cKDTree

    xy = meta[["x_pos", "y_pos"]].to_numpy(float)
    tiles = meta.tile.to_numpy()
    if len(tiles) < 2:
        return []
    d, idx = cKDTree(xy).query(xy, k=2)                      # nearest neighbour (col 1), excl self
    pitch = float(np.median(d[:, 1]))
    pairs = []
    for a in np.argsort(d[:, 1]):                            # tightest neighbours first
        b = idx[a, 1]
        if d[a, 1] > 1.3 * pitch:
            continue
        dx, dy = abs(xy[b, 0] - xy[a, 0]), abs(xy[b, 1] - xy[a, 1])
        if max(dx, dy) < 3 * min(dx, dy):                   # keep clearly axis-aligned pairs only
            continue
        pairs.append((int(tiles[a]), int(tiles[b])))
        if len(pairs) >= n_pairs:
            break
    return pairs


def _overlap_ncc_scores(meta, tiff_of, px, n_pairs):
    """Mean adjacent-tile overlap NCC per stage-axis hypothesis {hyp: score}. Local seam alignment;
    also cleanly rejects the wrong SWAP (transpose mis-tiles → NCC ~0)."""
    pairs = _neighbor_pairs(meta, n_pairs)
    if not pairs:
        return {}
    cache = {}

    def img(t):
        if t not in cache:
            cache[t] = read_sbs_ref_tile(tiff_of(t)).astype(np.float32)
        return cache[t]

    out = {}
    for hyp in STAGE_AXIS_HYPS:
        vals = []
        for a, b in pairs:
            ra, ca = _stage_rc(*_xy(meta, a), px, hyp)
            rb, cb = _stage_rc(*_xy(meta, b), px, hyp)
            v = _overlap_ncc(img(a), img(b), rb - ra, cb - ca)
            if v > -1:
                vals.append(v)
        out[hyp] = float(np.mean(vals)) if vals else -np.inf
    return out


def _lowres_well(small, meta, px, axes, ds):
    """Assemble a low-res well from pre-downsampled tiles ``small`` at the ``axes`` placement."""
    tiles = list(small)
    raw = {t: _stage_rc(*_xy(meta, t), px, axes) for t in tiles}
    rmin = min(r for r, _ in raw.values()); cmin = min(c for _, c in raw.values())
    place = {t: (int(round((r - rmin) / ds)), int(round((c - cmin) / ds))) for t, (r, c) in raw.items()}
    tds = max(s.shape[0] for s in small.values())
    H = max(r for r, _ in place.values()) + tds
    W = max(c for _, c in place.values()) + tds
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.uint16)
    for t in tiles:
        im = small[t]; r, c = place[t]; h, w = im.shape
        acc[r:r + h, c:c + w] += im; cnt[r:r + h, c:c + w] += 1
    return acc / np.maximum(cnt, 1)


def well_roughness(well, nb=180, pct=97):
    """Angular roughness of the well's bright outer boundary (numpy-only, no threshold tuning).

    Bins the brightest ``pct``-percentile pixels by angle about their centroid and takes each bin's
    outer (90th-percentile) radius; the roughness is the circular total-variation of that radial
    profile normalized by its mean radius. A cleanly-stitched circular well (continuous meniscus rim)
    gives a near-constant profile -> roughness ~0; a mis-oriented stitch spikes the rim in/out ->
    large roughness. Returns ``+inf`` if no usable boundary is found.
    """
    w = np.asarray(well, float)
    pos = w[w > 0]
    if pos.size < 100:
        return np.inf
    ys, xs = np.where(w > np.percentile(pos, pct))
    if ys.size < nb:
        return np.inf
    cy, cx = ys.mean(), xs.mean()
    r = np.hypot(ys - cy, xs - cx)
    th = np.arctan2(ys - cy, xs - cx)
    edges = np.linspace(-np.pi, np.pi, nb + 1)
    prof = []
    for i in range(nb):
        m = (th >= edges[i]) & (th < edges[i + 1])
        if m.sum() >= 2:
            prof.append(np.percentile(r[m], 90))
    if len(prof) < nb * 0.7:
        return np.inf
    prof = np.array(prof)
    tv = float(np.mean(np.abs(np.diff(np.r_[prof, prof[0]]))))
    return tv / (prof.mean() + 1e-9)


def recover_stage_axes(meta, tiff_of, n_pairs=6, ds=16, verbose=False):
    """[E5, D10] Recover (sx, sy, swap) as the placement that stitches the CLEANEST CIRCULAR WELL.

    The stage<->camera orientation is one of the 8 axis hypotheses. Selection is by **boundary
    roughness** (:func:`well_roughness`) of a low-res whole-well stitch — the geometrically-decisive
    criterion for a circular plate well: the correct axes yields a continuous circular meniscus rim,
    the wrong ones spike it in/out. Adjacent-tile **overlap NCC** is computed too and reported, but
    it is NOT the primary selector: on the dense, self-similar punctate SBS interior it is nearly
    degenerate for the axis SIGNS and previously mis-picked ``(-1,1,False)`` (a mirrored, jagged well)
    over the correct ``(1,1,False)`` — a bug caught only by looking at the stitched well (D10). A
    warning fires if the two criteria disagree.

    D8: the orientation is a FIXED microscope property — recover ONCE on a good well, then FREEZE it
    (pass ``axes=(sx,sy,swap)`` to :func:`build_sbs_mosaic`; the ``stage_axes`` config knob). Auto
    mode reads every present tile once (downsampled by ``ds``) to assemble the diagnostic wells.
    """
    px = float(meta.pixel_size_x.iloc[0])
    present = meta[meta.tile.map(lambda t: os.path.exists(tiff_of(int(t))))]   # skip missing tiffs
    tiles = [int(t) for t in present.tile]
    if len(tiles) < 4:
        if verbose:
            print("    recover_stage_axes: too few tiles; defaulting to (1,1,False)")
        return (1, 1, False)

    from skimage.transform import downscale_local_mean
    small = {t: downscale_local_mean(read_sbs_ref_tile(tiff_of(t)).astype(float), (ds, ds)).astype(np.float32)
             for t in tiles}
    rough = {hyp: well_roughness(_lowres_well(small, present, px, hyp, ds)) for hyp in STAGE_AXIS_HYPS}
    ncc = _overlap_ncc_scores(present, tiff_of, px, n_pairs)

    best = min(STAGE_AXIS_HYPS, key=lambda h: rough[h])        # cleanest circular well
    ncc_best = max(ncc, key=ncc.get) if ncc else best
    if verbose or best != ncc_best:
        order = sorted(STAGE_AXIS_HYPS, key=lambda h: rough[h])
        for h in order[:3]:
            print(f"    axes {h}  well-roughness {rough[h]:.4f}  overlap-NCC {ncc.get(h, float('nan')):.3f}")
        if best != ncc_best:
            print(f"    NOTE recover_stage_axes: overlap-NCC would pick {ncc_best} (self-similar-"
                  f"degenerate); well-geometry picks {best} (cleaner circle) — using {best}. "
                  f"Confirm via `python -m pyali.registration.viz orient` and FREEZE via stage_axes.")
    if not np.isfinite(rough[best]):
        print("    WARNING recover_stage_axes: no clear well boundary; falling back to overlap-NCC")
        return ncc_best
    return best


def _build_A_sbs(px, axes, rmin, cmin):
    """Analytic stage(x,y)µm -> mosaic(x=col, y=row)px affine (handles reflection; exact)."""
    sx, sy, swap = axes
    M = np.eye(3)
    if not swap:
        M[0, 0], M[0, 1] = sx / px, 0.0                     # col = sx * x/px
        M[1, 0], M[1, 1] = 0.0, sy / px                     # row = sy * y/px
    else:
        M[0, 0], M[0, 1] = 0.0, sx / px                     # col = sx * y/px
        M[1, 0], M[1, 1] = sy / px, 0.0                     # row = sy * x/px
    M[0, 2] = -cmin + _HALF                                 # -> mosaic x (col) of tile CENTER
    M[1, 2] = -rmin + _HALF                                 # -> mosaic y (row) of tile CENTER
    return M


def A_sbs_residual(meta, off, M):
    """Max | M @ (tile stage center) - (placed tile center) | in px, over placed tiles.

    A QC cross-check that the summary transform ``M`` reproduces the actual per-tile placement
    ``off`` (which is what truly places cells). For the analytic construction this is just the
    integer-rounding error (~0.5 px); a proper-similarity fit balloons it when the stage->pixel map
    is a reflection (see D7). ``M`` may be the constructed A_sbs, a ``fit_affine``, or a
    ``fit_similarity`` — this is the tool to compare them.
    """
    m = meta.set_index("tile")
    tiles = list(off.keys())
    src = np.array([[float(m.at[t, "x_pos"]), float(m.at[t, "y_pos"])] for t in tiles], float)
    ctr = np.array([[off[t][1] + _HALF, off[t][0] + _HALF] for t in tiles], float)  # (x=col, y=row)
    return float(np.abs(coordinates.apply_affine(M, src) - ctr).max())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_sbs_mosaic(tile_ids, tiff_of, meta, axes=None, verbose=False,
                     read_tile=None, dtype=np.uint16):
    """Return ``(mosaic [H,W] dtype, off {tile:(row0,col0)}, A_sbs (3,3) f64)``.

    ``read_tile`` (default: raw JF608 channel via :func:`read_sbs_ref_tile`) is the togglable tile
    source — pass ``demix.make_anchor_reader(method, tiff_of, sbs_info)`` (with ``dtype=np.float32``)
    to stitch a Cy5/Cy7-suppressed JF608 mosaic instead of the raw channel [3] (D11). ``dtype`` is the
    output mosaic dtype (use float32 for log/regression anchors whose values are not in uint16 range).

    ``off`` (per-tile integer pixel offsets) is the PLACEMENT SOURCE OF TRUTH — it places both the
    image tiles and the SBS cells, and captures each tile's true stage position (incl. jitter).
    Registration/assignment accuracy depends on ``off``, NOT on ``A_sbs``.

    ``A_sbs`` (stage µm -> mosaic px, x=col/y=row) is a QC/summary affine, CONSTRUCTED analytically
    from (pixel_size, axes, offsets) [D7]: exact (reproduces ``off`` to the ~0.5 px rounding), and
    it represents the reflection the stage->pixel map can have — which a proper ``fit_similarity``
    cannot. (Fitting is therefore strictly worse here and, being off the placement path, cannot
    improve registration; use :func:`A_sbs_residual` to confirm.) ``off`` includes only tiles whose
    tiff exists (missing ones skipped).
    """
    px = float(meta.pixel_size_x.iloc[0])
    if axes is None:
        axes = recover_stage_axes(meta, tiff_of, verbose=verbose)
    reader = read_tile if read_tile is not None else (lambda t: read_sbs_ref_tile(tiff_of(int(t))))

    raw = {int(t): _stage_rc(*_xy(meta, t), px, axes) for t in tile_ids}
    rmin = min(r for r, _ in raw.values())
    cmin = min(c for _, c in raw.values())
    place = {t: (int(round(r - rmin)), int(round(c - cmin))) for t, (r, c) in raw.items()}
    H = max(r for r, _ in place.values()) + TILE
    W = max(c for _, c in place.values()) + TILE

    acc = np.zeros((H, W), np.float32)                      # [E4] float32 accumulator
    cnt = np.zeros((H, W), np.uint16)                       # [E4] uint16 occupation count
    off = {}
    for t in tile_ids:
        t = int(t)
        if not os.path.exists(tiff_of(t)):                  # skip tiles with no tiff on disk
            continue
        r, c = place[t]
        acc[r:r + TILE, c:c + TILE] += reader(t)
        cnt[r:r + TILE, c:c + TILE] += 1
        off[t] = place[t]

    if verbose:
        print(f"    mosaic {(H, W)} placed={len(off)}/{len(tile_ids)} "
              f"maxcov={int(cnt.max())} overlap={float((cnt > 1).mean()) * 100:.1f}%")

    np.maximum(cnt, 1, out=cnt)                             # avoid /0; 0-coverage -> 1
    acc /= cnt                                              # in-place float32 /= uint16 -> float32
    del cnt                                                 # [review] free the count before the cast
    mosaic = acc.astype(dtype)                              # averaged overlaps
    del acc
    A_sbs = _build_A_sbs(px, axes, rmin, cmin)
    return mosaic, off, A_sbs
