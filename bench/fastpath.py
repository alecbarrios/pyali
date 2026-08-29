"""Prototype fixes for the two pyali scaling blockers, as monkey-patches over the real modules.

Both are *numerically identical* to the originals, not approximations:

1. ``movmedian_time`` — the original runs a Python loop of ``T`` calls to ``np.median`` over an
   ``(<=8, H, W)`` slab. Replaced with a strided sliding-window median evaluated in spatial
   chunks: same shrinking-window-at-the-ends definition, one vectorized median per chunk.

2. ``extract_footprints`` never touches the full ``filtered_movie`` — it only ever slices a
   ``[T, ~27, ~27]`` patch out of it (extract.py:368), and ``movie`` only for ``T``. So the
   global filtered movie is pure waste: it doubles peak RAM and filters every pixel when only
   the pixels inside region patches are ever read. ``patch_filter`` filters one patch on demand.
   Filtering along time is per-pixel independent, so a spatially-sliced filter is bit-identical
   to slicing the globally-filtered movie.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pyali import extract
from pyali.utils import round_half_away_from_zero


def movmedian_time_fast(a, window=8, axis=0, chunk_bytes=512 << 20):
    """Vectorized equivalent of :func:`pyali.utils.movmedian_time`.

    Window at index ``i`` spans ``[i - kb, i + kf]``, ``kb = window // 2``,
    ``kf = window - 1 - kb``, shrinking at both ends. Even windows average the two central
    order statistics (what ``np.median`` already does).
    """
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    kb = window // 2
    kf = window - 1 - kb
    out = np.empty_like(a)

    # --- ends: window is truncated, handle with the original per-index median ---
    for i in list(range(min(kb, n))) + list(range(max(kb, n - kf), n)):
        out[i] = np.median(a[max(0, i - kb):min(n, i + kf + 1)], axis=0)

    # --- interior: every window is exactly `window` long -> one strided median ---
    lo, hi = kb, n - kf                                   # out indices [lo, hi)
    if hi <= lo:
        return np.moveaxis(out, 0, axis)
    flat = a.reshape(n, -1)                               # [n, npix]
    out_flat = out.reshape(n, -1)
    npix = flat.shape[1]
    # sliding_window_view over axis 0 -> [n-window+1, npix, window]; window w starts at index w
    # and its output index is w + kb, so rows [lo, hi) map to windows [0, hi-lo).
    per_pix = window * a.dtype.itemsize
    step = max(1, min(npix, int(chunk_bytes // (max(1, (hi - lo)) * per_pix))))
    for c0 in range(0, npix, step):
        c1 = min(npix, c0 + step)
        sw = sliding_window_view(flat[:, c0:c1], window, axis=0)     # [n-w+1, c1-c0, w]
        out_flat[lo:hi, c0:c1] = np.median(sw[:hi - lo], axis=-1)
    return np.moveaxis(out, 0, axis)


def patch_filter(movie, r0, r1, c0, c1, window):
    """``temporal_filter`` of one spatial patch: identical to slicing the filtered movie."""
    sub = movie[:, r0:r1, c0:c1]
    filt = movmedian_time_fast(sub, window, axis=0)
    filt -= sub
    return filt


def extract_footprints_lowmem(movie, filtered_movie, regions, spatial_footprints, dog,
                              std_frames, p, height, width, verbose=False):
    """Drop-in for :func:`pyali.extract.extract_footprints` that filters per patch.

    ``filtered_movie`` is ignored (pass ``None``); everything else matches the original,
    including the order in which planes/centers are appended.
    """
    from sklearn.cluster import DBSCAN

    T = movie.shape[0]
    fw = p.filter_window
    half = int(round_half_away_from_zero(fw / 2))
    APs, COMs, planes, centers = [], [], [], []

    def _fallback(sel_full, sel_patch, pm_max, region):
        planes.append(extract.region_fallback_footprint(sel_full, sel_patch, pm_max, height, width))
        centers.append(extract.region_center_rowcol(region))

    for c, region in enumerate(regions):
        if verbose and (c % 25 == 0 or c == len(regions) - 1):
            print(f"[pyali]   region {c + 1}/{len(regions)}", flush=True)
        patch_rows, patch_cols, origin0 = extract.compute_patch(
            region["Centroid"], region["BoundingBox"], p.patch_size, height, width)
        sel_full = extract.build_selection_map(spatial_footprints[c], height, width)
        sel_patch = sel_full[np.ix_(patch_rows - 1, patch_cols - 1)]

        # the one change: filter just this patch instead of slicing a global filtered movie
        pm = patch_filter(movie, patch_rows[0] - 1, patch_rows[-1],
                          patch_cols[0] - 1, patch_cols[-1], fw)
        pm[:half] = 0.0
        pm[T - half:] = 0.0
        pm_max = pm.max(axis=0)

        region_AP, _diag = extract.detect_region_aps(pm, sel_patch, origin0, dog, std_frames,
                                                     p.threshold_factor, fw)
        region_AP = extract.dedup_close_peaks(region_AP, p.min_peak_interval)
        if len(region_AP) == 0:
            _fallback(sel_full, sel_patch, pm_max, region); continue
        APs.append(region_AP)

        region_COMs = extract.com_via_svd(pm, region_AP[:, 2].astype(int), patch_rows, patch_cols,
                                          p.svd_rank, p.com_radius, p.com_n_pixels)
        COMs.append(region_COMs)
        labels = DBSCAN(eps=p.dbscan_eps, min_samples=p.dbscan_min_pts).fit_predict(region_COMs[:, :2])
        if np.all(labels == -1):
            _fallback(sel_full, sel_patch, pm_max, region); continue

        cluster_times = [region_COMs[labels == k, 2].astype(int)
                         for k in range(labels.max() + 1) if np.any(labels == k)]
        pl, ce = extract.cluster_footprints(pm, cluster_times, sel_patch, patch_rows, patch_cols,
                                            height, width)
        planes.extend(pl); centers.extend(ce)

    footprint = np.stack(planes, axis=2) if planes else np.zeros((height, width, 0))
    footprint_center = np.array(centers, dtype=float).reshape(-1, 2)
    APs = np.vstack(APs) if APs else np.zeros((0, 4))
    COMs = np.vstack(COMs) if COMs else np.zeros((0, 3))
    return APs, COMs, footprint, footprint_center


def install(fast_median=True, lowmem_footprints=True):
    """Monkey-patch the prototypes into the live pyali modules."""
    import pyali.utils as U
    import pyali.pipeline as PL
    if fast_median:
        U.movmedian_time = movmedian_time_fast
        extract.movmedian_time = movmedian_time_fast
    if lowmem_footprints:
        extract.temporal_filter = lambda movie, window=8: None      # skip the global filter
        extract.extract_footprints = extract_footprints_lowmem
        PL.extract = extract
