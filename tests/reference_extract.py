"""Pre-E2 per-region extraction, copied verbatim from commit 1a04322, used as the oracle.

* the whole of ``pyali/extract.py`` as it was (temporal_filter_patch, read_patch, compute_patch,
  build_selection_map, median_filter_1d, mean_abs_dev, detect_region_aps, dedup_close_peaks,
  svd_denoise_frames, region_grow_brightest, _weighted_com, com_via_svd,
  region_fallback_footprint, cluster_footprints, extract_footprints, region_center_rowcol);
* ``findpeaks`` from ``pyali/utils.py`` (the loop version);
* ``old_patch`` = ``CorrectedMovie.patch`` from ``pyali/movie.py``.

The module-level ``findpeaks`` below shadows the import so the copied code runs the old loop.
Nothing here is used by the pipeline.
"""
import numpy as np

from pyali.results import SparseBuilder
from pyali.utils import movmedian_time, round_half_away_from_zero


# --------------------------------------------------------------------------- #
# Temporal filter
# --------------------------------------------------------------------------- #
def temporal_filter_patch(patch, window=8):
    """Moving-median high-pass along time of one corrected patch ``(T, Hp, Wp)``, sign-flipped so
    downward spikes become positive: ``movmedian(patch) - patch``. The filter is independent
    per pixel, so filtering a patch equals filtering the whole movie and slicing it."""
    filtered = movmedian_time(patch, window, axis=0)
    filtered -= patch
    return filtered


def read_patch(movie, r0, r1, c0, c1):
    """Corrected ``(T, r1 - r0, c1 - c0)`` block: ``movie.patch(...)`` for a
    :class:`pyali.movie.CorrectedMovie`, a plain slice for a dense ``(T, H, W)`` array."""
    if hasattr(movie, "patch"):
        return movie.patch(r0, r1, c0, c1)
    return movie[:, r0:r1, c0:c1]


# --------------------------------------------------------------------------- #
# Patch geometry
# --------------------------------------------------------------------------- #
def compute_patch(centroid, bbox, patch_size, height, width):
    """Row/column index ranges of the analysis patch for a region (a fixed patch or its bbox).

    Parameters
    ----------
    centroid : [col, row] (1-indexed)
    bbox : [x_ul, y_ul, w, h]
    patch_size : (rows, cols)
    height, width : int

    Returns
    -------
    (patch_rows, patch_cols, origin0)
        1-indexed inclusive index ranges, and the 0-indexed top-left corner ``(row0, col0)``.
    """
    cx, cy = float(centroid[0]), float(centroid[1])            # col, row
    x_ul, y_ul, w, h = (float(v) for v in bbox)
    ph, pw = patch_size
    fl = np.floor
    min_row = min(fl(cy - pw * 0.5), fl(y_ul + 0.5))          # rows
    min_col = min(fl(cx - ph * 0.5), fl(x_ul + 0.5))          # cols
    max_row = max(fl(cy + pw * 0.5), fl(y_ul + 0.5 + h))
    max_col = max(fl(cx + ph * 0.5), fl(x_ul + 0.5 + w))
    patch_rows = np.arange(max(int(min_row), 1), min(int(max_row), height) + 1)   # 1-indexed
    patch_cols = np.arange(max(int(min_col), 1), min(int(max_col), width) + 1)
    return patch_rows, patch_cols, (patch_rows[0] - 1, patch_cols[0] - 1)


def build_selection_map(pixel_list, height, width):
    """Binary map of a region's pixels. ``pixel_list`` = ``[col, row]`` (1-indexed)."""
    sel = np.zeros((height, width))
    rows = pixel_list[:, 1].astype(int) - 1
    cols = pixel_list[:, 0].astype(int) - 1
    sel[rows, cols] = 1.0
    return sel


# --------------------------------------------------------------------------- #
# AP detection helpers
# --------------------------------------------------------------------------- #
def median_filter_1d(x, n):
    """MATLAB ``medfilt1(x, n)`` with its default zero padding: output ``k`` is the median of
    ``x[k - nb : k + na]`` with ``nb = na = (n-1)/2`` for odd ``n`` and ``nb = n/2, na = n/2 - 1``
    for even ``n``; samples beyond the ends count as zeros. Even windows take the mean of the two
    middle values. Float64, same length as ``x``."""
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, np.float64).ravel()
    n = int(n)
    if n < 1:
        raise ValueError(f"window must be >= 1, got {n}")
    nb, na = ((n - 1) // 2, (n - 1) // 2) if n % 2 else (n // 2, n // 2 - 1)
    xp = np.concatenate([np.zeros(nb), x, np.zeros(na)])
    return np.median(sliding_window_view(xp, n), axis=1)


def mean_abs_dev(x):
    """MATLAB/Octave ``mad(x)`` with the default flag 0: the mean absolute deviation about the
    mean (about 0.8 sigma for Gaussian noise). Not the robust median variant."""
    x = np.asarray(x, np.float64).ravel()
    return float(np.mean(np.abs(x - x.mean()))) if x.size else 0.0


# --------------------------------------------------------------------------- #
# Per-region AP detection (Mini ALI mini_ali_patch_detect_spike_times)
# --------------------------------------------------------------------------- #
def detect_region_aps(patch_raw, selection_map_patch, pm, origin0, median_window=32,
                      edge_frames=16, threshold_factor=4.0):
    """Detect action potentials in one region from its raw patch's mean trace.

    The region trace is the sum of the in-mask pixels divided by the patch area, from the
    background-subtracted, motion-corrected movie patch (motion is already removed from the
    movie, so the original's per-trace gradient regression has no counterpart here). The
    detection signal is ``medfilt1(trace, median_window) - trace`` (spikes positive), with
    ``edge_frames`` zeroed at each end; the noise is the mean absolute deviation of that whole
    signal (MATLAB ``mad``, flag 0, ~0.8 sigma), the threshold ``threshold_factor`` times it;
    peaks are the local maxima of the thresholded signal (MATLAB ``findpeaks`` rule). No peak
    refinement, no extra end skips: with 16 zeroed edge frames there is nothing to skip.

    Parameters
    ----------
    patch_raw : (T, Hp, Wp) float array
        The movie patch (not temporally filtered).
    selection_map_patch : (Hp, Wp) array
        Region mask cropped to the patch.
    pm : (T, Hp, Wp) float array
        The per-pixel temporally filtered patch (:func:`temporal_filter_patch`), used only for
        the coarse pixel of each spike (the brightest in-mask pixel of its frame; informational,
        :func:`com_via_svd` re-finds the maximum).
    origin0 : (row0, col0)
        0-indexed patch top-left corner.
    median_window, edge_frames, threshold_factor : see ``Params``.

    Returns
    -------
    (region_AP, diag)
        ``region_AP`` : (k, 4) = ``[row, col, frame, amplitude]`` (0-indexed, pre-dedup), the
        amplitude being the detection signal at the peak; ``diag`` : ``trace, dF, sigma,
        threshold``.
    """
    mask = np.asarray(selection_map_patch) > 0
    T = patch_raw.shape[0]
    trace = np.asarray(patch_raw)[:, mask].sum(axis=1, dtype=np.float64) / float(mask.size)
    dF = median_filter_1d(trace, median_window) - trace
    e = min(int(edge_frames), T)
    dF[:e] = 0.0
    dF[max(0, T - e):] = 0.0
    sigma = mean_abs_dev(dF)
    threshold = float(threshold_factor) * sigma
    thr = np.where(dF >= threshold, dF, 0.0) if threshold > 0 else np.zeros_like(dF)
    peaks, pk = findpeaks(thr)

    region_AP = []
    for amp, t in zip(peaks, pk):
        frame = np.asarray(pm[int(t)]) * mask                            # coarse location inside the mask
        pv = frame.max()
        fi = int(np.argmax(frame.ravel(order="F") == pv))               # first occurrence, column-major
        px0, py0 = np.unravel_index(fi, frame.shape, order="F")
        region_AP.append([px0 + origin0[0], py0 + origin0[1], int(t), float(amp)])
    region_AP = np.array(region_AP, dtype=np.float64).reshape(-1, 4)
    return region_AP, dict(trace=trace, dF=dF, sigma=sigma, threshold=threshold)


def dedup_close_peaks(region_AP, min_interval=25):
    """Keep the largest-amplitude AP within any ``min_interval``-frame window.

    Rows are ``[row, col, frame, amplitude]``; the amplitude is the thresholded-trace peak value
    stored by :func:`detect_region_aps`. Frames are grouped into runs anchored at the earliest
    member (a run extends while ``frame - first_frame < min_interval``) and the largest amplitude
    of each run survives; a second greedy pass by amplitude then resolves survivors of adjacent
    runs that are still closer than ``min_interval``. The output is sorted by frame and pairwise
    ``>= min_interval`` frames apart, so a burst of spikes keeps its largest member instead of
    losing every member (the previous intersect-of-``uniquetol`` rule). Follows Mini ALI V2.2's
    minimal-interval rule with the user's "keep max in window" refinement.
    """
    if len(region_AP) == 0:
        return region_AP
    AP = region_AP[np.argsort(region_AP[:, 2], kind="stable")]
    times, amps = AP[:, 2], AP[:, 3]
    n = len(AP)
    keep = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and times[j + 1] - times[i] < min_interval:    # run anchored at i
            j += 1
        keep.append(i + int(np.argmax(amps[i:j + 1])))                   # ties -> earliest
        i = j + 1
    kept = AP[keep]
    accept = []
    for idx in np.argsort(-kept[:, 3], kind="stable"):                    # greedy by amplitude
        if all(abs(kept[idx, 2] - kept[a, 2]) >= min_interval for a in accept):
            accept.append(int(idx))
    accept.sort(key=lambda a: kept[a, 2])
    return kept[accept]


# --------------------------------------------------------------------------- #
# Center of mass via SVD + region growing
# --------------------------------------------------------------------------- #
def svd_denoise_frames(patch_movie, times, rank=10):
    """Low-rank denoise the AP frames: one filtered frame per spike (Mini ALI
    ``mini_ali_SVD_denoise`` with ``svds(pixels x spikes, rank)``).

    Parameters
    ----------
    patch_movie : (T, Hp, Wp) float array
        The per-pixel temporally filtered patch (spikes positive).
    times : array of int
        0-indexed AP frames.
    rank : int
        Truncation rank, capped at the number of spikes (a region with ``<= rank`` spikes is
        reconstructed exactly, i.e. not denoised, as in the original).

    Returns
    -------
    (Hp, Wp, nAP) float64: the rank-truncated reconstruction of each spike's frame.
    """
    times = np.asarray(times, int)
    Hp, Wp = patch_movie.shape[1:]
    nAP = len(times)
    if nAP == 0:
        return np.zeros((Hp, Wp, 0))
    X = np.asarray(patch_movie[times], np.float64).reshape(nAP, Hp * Wp).T   # pixels x spikes
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(int(rank), nAP)
    recon = (U[:, :k] * s[:k]) @ Vt[:k]                                        # pixels x spikes
    return recon.T.reshape(nAP, Hp, Wp).transpose(1, 2, 0)


def region_grow_brightest(current, radius=6.0, n_pixels=25):
    """Greedily grow up to ``n_pixels`` bright pixels outward from the maximum, within ``radius``
    of it (Mini ALI V2.2 ``mini_ali_spk_fine``).

    Parameters
    ----------
    current : (Hp, Wp) float array
        A denoised AP frame, already masked to the region (zeros outside).
    radius : float
        Euclidean radius, in pixels, measured from the seed (the global maximum).
    n_pixels : int
        Maximum number of pixels to pick.

    Returns
    -------
    (k, 2) int array of 0-indexed (row, col), ``1 <= k <= n_pixels``, seed first.

    Notes
    -----
    The seed is the global maximum (first in column-major order on ties, as MATLAB's
    ``max(x(:))``). Each step queues the 4-neighbours of the last picked pixel that lie inside
    the patch and within ``radius`` of the seed, drops already-picked pixels, and picks the
    brightest pixel *inside the queue*; an empty queue ends the growth early instead of raising
    (``spk_fine`` lines 54-56), so the result may be shorter than ``n_pixels``. The queue is a
    boolean mask (O(1) membership); on exactly equal float values it breaks ties in
    column-major order where MATLAB's coordinate list breaks them in insertion order — the one
    accepted deviation, immaterial on denoised data.
    """
    Hp, Wp = current.shape
    flat_f = current.ravel(order="F")
    seed = np.unravel_index(int(np.argmax(flat_f)), current.shape, order="F")
    picked = [seed]
    is_picked = np.zeros((Hp, Wp), bool)
    is_picked[seed] = True
    rr, cc = np.ogrid[:Hp, :Wp]
    in_radius = ((rr - seed[0]) ** 2 + (cc - seed[1]) ** 2) <= radius ** 2
    queued = np.zeros((Hp, Wp), bool)

    while len(picked) < n_pixels:
        pr, pc = picked[-1]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):           # 4-neighbours
            r, c = pr + dr, pc + dc
            if 0 <= r < Hp and 0 <= c < Wp and in_radius[r, c]:
                queued[r, c] = True
        queued &= ~is_picked                                        # drop already-picked
        idx = np.flatnonzero(queued.ravel(order="F"))               # queue in column-major order
        if idx.size == 0:
            break
        q = int(idx[int(np.argmax(flat_f[idx]))])                   # argmax INSIDE the queue
        rc = np.unravel_index(q, current.shape, order="F")
        picked.append(rc)
        is_picked[rc] = True
    return np.array(picked, int).reshape(-1, 2)


def _weighted_com(current, picked, patch_rows, patch_cols):
    """Intensity^2-weighted centroid of the picked pixels (``mini_ali_spk_fine`` lines 72-86).

    Negative values are clipped to zero before squaring; only the picked pixels themselves
    carry weight (not their row x column hull). Returns ``(nan, nan)`` when the total mass is
    zero. ``patch_rows``/``patch_cols`` are 1-indexed absolute coordinates, so the centroid is
    absolute and 1-indexed.
    """
    picked = np.asarray(picked, int).reshape(-1, 2)
    vals = current[picked[:, 0], picked[:, 1]].astype(np.float64)
    vals[vals < 0] = 0.0
    w = vals * vals
    total = w.sum()
    if total == 0:
        return np.nan, np.nan
    row_c = (w * patch_rows[picked[:, 0]]).sum() / total
    col_c = (w * patch_cols[picked[:, 1]]).sum() / total
    return row_c, col_c


def com_via_svd(patch_movie, times, patch_rows, patch_cols, sel_patch, rank=15, radius=6.0,
                n_pixels=25):
    """Center of mass per AP -> ``[k, 3]`` = ``[row, col, frame]`` (0-indexed frame), ``k <= nAP``.

    Each SVD-denoised center frame is masked with ``sel_patch`` (the region's pixels; Mini ALI's
    ``dF_AP .* patch_mask``), the brightest in-mask pixels are grown with
    :func:`region_grow_brightest` and their intensity^2 centroid taken. An AP is dropped when
    the centroid is undefined (no positive in-mask mass) or when its rounded position falls
    outside ``sel_patch`` (``run_miniali_fov.m`` lines 363-370), so the result can be shorter
    than ``times`` — even empty. ``patch_rows``/``patch_cols`` are the 1-indexed absolute patch
    coordinates and the returned ``[row, col]`` are 1-indexed absolute.
    """
    centers = svd_denoise_frames(patch_movie, times, rank)
    Hp, Wp = sel_patch.shape
    r_off, c_off = int(patch_rows[0]), int(patch_cols[0])
    coms = []
    for h in range(len(times)):
        img = centers[:, :, h] * sel_patch
        picked = region_grow_brightest(img, radius, n_pixels)
        r, c = _weighted_com(img, picked, patch_rows, patch_cols)
        if np.isnan(r) or np.isnan(c):
            continue
        pr = int(round_half_away_from_zero(r)) - r_off                 # back to patch 0-indexed
        pc = int(round_half_away_from_zero(c)) - c_off
        if not (0 <= pr < Hp and 0 <= pc < Wp) or sel_patch[pr, pc] == 0:
            continue                                                   # outside the region
        coms.append([r, c, times[h]])
    return np.array(coms, dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# Footprints
# --------------------------------------------------------------------------- #
def region_fallback_footprint(selection_map, selection_map_patch, patch_movie_max, height, width):
    """Fallback footprint (no AP / no cluster): the region's per-pixel max projection.

    The footprint is filled on the row x column hull of the selected pixels (column-major order).
    """
    fp = np.zeros((height, width))
    fi = np.where(selection_map.ravel(order="F") == 1)[0]
    x, y = np.unravel_index(fi, selection_map.shape, order="F")            # column-major order
    fip = np.where(selection_map_patch.ravel(order="F") == 1)[0]
    xp, yp = np.unravel_index(fip, selection_map_patch.shape, order="F")
    fp[np.ix_(x, y)] = patch_movie_max[np.ix_(xp, yp)]
    return fp


def cluster_footprints(patch_movie, cluster_times, selection_map_patch, patch_rows, patch_cols,
                       height, width):
    """Per-cluster footprints: mean of each cluster's AP frames, negatives clipped to zero, masked
    to an 11x11 window around the brightest in-selection pixel.

    ``cluster_times`` = list of arrays of 0-indexed frames per cluster. Returns
    ``(planes, centers)`` — each plane ``[H, W]``; each center ``[row, col]`` (1-indexed).
    """
    r0, c0 = patch_rows[0], patch_cols[0]                                  # 1-indexed origin
    planes, centers = [], []
    for times in cluster_times:
        raw = patch_movie[np.asarray(times, int)].mean(axis=0)            # [Hp, Wp]
        raw = np.maximum(raw, 0.0)                 # footprint(footprint<0)=0 (run_miniali_fov.m:486):
        #   a negative weight would subtract neighbour signal with the wrong sign in the pinv
        masked = raw * selection_map_patch
        mx = masked.max()
        fi = int(np.argmax(masked.ravel(order="F") == mx))               # first, column-major
        pkx, pky = np.unravel_index(fi, masked.shape, order="F")
        peakx, peaky = pkx + r0, pky + c0                                 # absolute (1-indexed)
        sel = np.zeros((height, width))
        sel[max(peakx - 1 - 5, 0):min(peakx - 1 + 6, height),            # 11x11 window (0-indexed)
            max(peaky - 1 - 5, 0):min(peaky - 1 + 6, width)] = 1.0
        plane = np.zeros((height, width))
        rr = patch_rows - 1
        cc = patch_cols - 1
        plane[np.ix_(rr, cc)] = raw * sel[np.ix_(rr, cc)]
        planes.append(plane)
        centers.append([peakx, peaky])
    return planes, centers


SOURCES = ("spiking", "no_ap", "no_com", "all_noise")     # cells.source vocabulary


def extract_footprints(movie, regions, spatial_footprints, p, height, width, verbose=False):
    """Per-region loop: AP detection -> center of mass -> DBSCAN clustering -> footprints.

    Parameters
    ----------
    movie : :class:`pyali.movie.CorrectedMovie` or a dense (T, H, W) float array
        Only the region patches are ever read (:func:`read_patch`), each filtered on demand by
        :func:`temporal_filter_patch`.
    regions, spatial_footprints : from :func:`pyali.segmentation.cell_segmentation`
    p : Params
    height, width : int
    verbose : bool

    Returns
    -------
    (APs, COMs, footprints, footprint_center, region_of_footprint, region_of_ap,
     region_of_spike, spike_cell_row, source_of_footprint)
        ``APs`` : (nAP, 4) = ``[row, col, frame, amplitude]`` (0-indexed);
        ``COMs`` : (nCOM, 3) = ``[row, col, frame]`` (1-indexed row/col) -- ``nCOM <= nAP``
        because :func:`com_via_svd` drops APs whose centroid is undefined or off-mask;
        ``footprints`` : :class:`pyali.results.SparseFootprints` (N cells, float64 values, packed
        as each plane is produced so no ``(H, W, N)`` stack exists);
        ``footprint_center`` : (N, 2) = ``[row, col]``;
        ``region_of_footprint`` : (N,) int, the segmentation region each footprint came from;
        ``region_of_ap`` : (nAP,) int, the region of each ``APs`` row;
        ``region_of_spike`` : (nCOM,) int, the region of each ``COMs`` row;
        ``spike_cell_row`` : (nCOM,) int, the footprint row (0-based) each localized spike was
        clustered into, or -1 (DBSCAN noise, or a region that fell back to one footprint);
        ``source_of_footprint`` : list of str, one of :data:`SOURCES` per footprint --
        ``spiking`` for a DBSCAN cluster's footprint, else why the region fell back to one
        footprint: ``no_ap`` (no AP detected), ``no_com`` (no centroid survived the mask), or
        ``all_noise`` (every centroid was DBSCAN noise).
    """
    from sklearn.cluster import DBSCAN

    T = movie.shape[0]
    fw = p.filter_window
    half = int(round_half_away_from_zero(fw / 2))
    APs, COMs, centers = [], [], []
    footprints = SparseBuilder(height, width)                 # planes packed as they are made
    region_of_footprint, region_of_ap, region_of_spike, spike_cell_row = [], [], [], []
    source_of_footprint = []

    def _fallback(sel_full, sel_patch, pm_max, region, c, reason):
        footprints.add_plane(region_fallback_footprint(sel_full, sel_patch, pm_max, height, width))
        cen = region_center_rowcol(region)
        centers.append(cen)
        region_of_footprint.append(c)
        source_of_footprint.append(reason)

    for c, region in enumerate(regions):
        if verbose and (c % 25 == 0 or c == len(regions) - 1):
            print(f"[pyali]   region {c + 1}/{len(regions)}", flush=True)
        patch_rows, patch_cols, origin0 = compute_patch(region["Centroid"], region["BoundingBox"],
                                                        p.patch_size, height, width)
        sel_full = build_selection_map(spatial_footprints[c], height, width)
        sel_patch = sel_full[np.ix_(patch_rows - 1, patch_cols - 1)]
        r0, r1 = patch_rows[0] - 1, patch_rows[-1]
        c0, c1 = patch_cols[0] - 1, patch_cols[-1]
        sub = read_patch(movie, r0, r1, c0, c1)           # corrected patch, read on demand
        pm = temporal_filter_patch(sub, fw)
        pm[:half] = 0.0
        pm[T - half:] = 0.0
        pm_max = pm.max(axis=0)

        region_AP, _diag = detect_region_aps(sub, sel_patch, pm, origin0,
                                             p.detect_median_window, p.detect_edge_frames,
                                             p.threshold_factor)
        region_AP = dedup_close_peaks(region_AP, p.min_peak_interval)
        if len(region_AP) == 0:
            _fallback(sel_full, sel_patch, pm_max, region, c, "no_ap"); continue
        APs.append(region_AP)
        region_of_ap.extend([c] * len(region_AP))

        region_COMs = com_via_svd(pm, region_AP[:, 2].astype(int), patch_rows, patch_cols,
                                  sel_patch, p.svd_rank, p.com_radius, p.com_n_pixels)
        if len(region_COMs) == 0:                         # every COM undefined or outside the mask
            _fallback(sel_full, sel_patch, pm_max, region, c, "no_com"); continue
        COMs.append(region_COMs)
        region_of_spike.extend([c] * len(region_COMs))
        labels = DBSCAN(eps=p.dbscan_eps, min_samples=p.dbscan_min_pts).fit_predict(region_COMs[:, :2])
        if np.all(labels == -1):
            spike_cell_row.extend([-1] * len(region_COMs))
            _fallback(sel_full, sel_patch, pm_max, region, c, "all_noise"); continue

        # cluster k (in label order, skipping empty labels) becomes the next appended plane
        kept = [k for k in range(labels.max() + 1) if np.any(labels == k)]
        cluster_times = [region_COMs[labels == k, 2].astype(int) for k in kept]
        pl, ce = cluster_footprints(pm, cluster_times, sel_patch, patch_rows, patch_cols,
                                    height, width)
        row_of_label = {k: footprints.n() + i for i, k in enumerate(kept)}
        spike_cell_row.extend(row_of_label.get(int(lab), -1) for lab in labels)
        for plane in pl:
            footprints.add_plane(plane)
        centers.extend(ce)
        region_of_footprint.extend([c] * len(pl))
        source_of_footprint.extend(["spiking"] * len(pl))

    footprint_center = np.array(centers, dtype=float).reshape(-1, 2)
    APs = np.vstack(APs) if APs else np.zeros((0, 4))
    COMs = np.vstack(COMs) if COMs else np.zeros((0, 3))
    return (APs, COMs, footprints.build(), footprint_center,
            np.asarray(region_of_footprint, int), np.asarray(region_of_ap, int),
            np.asarray(region_of_spike, int), np.asarray(spike_cell_row, int),
            source_of_footprint)


def region_center_rowcol(region):
    """Region center as ``[row, col]`` from the region's Centroid ``[col, row]``."""
    cx, cy = float(region["Centroid"][0]), float(region["Centroid"][1])
    return [cy, cx]


# ---- pyali/utils.py findpeaks (loop version) ----
def findpeaks(x):
    """Local maxima of a 1-D signal (values only).

    A peak is a sample strictly greater than both neighbors; endpoints are excluded and a
    flat-topped peak returns its lowest (left-edge) index.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    (peak_values, peak_indices) : ndarray, ndarray
        0-based indices in ascending order.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    idx = []
    i = 1
    while i < n - 1:
        if x[i] > x[i - 1]:                      # rising into i
            j = i
            while j < n - 1 and x[j + 1] == x[j]:   # extend across a flat top
                j += 1
            if x[j] > x[j + 1]:                  # falls after -> peak at the left edge i
                idx.append(i)
            i = j + 1
        else:
            i += 1
    idx = np.array(idx, dtype=int)
    return x[idx], idx


# ---- pyali/movie.py CorrectedMovie.patch ----
def old_patch(self, r0, r1, c0, c1, t0=0, t1=None, dtype=np.float32):
    """``CorrectedMovie.patch`` as of commit 1a04322 (``self`` = a CorrectedMovie)."""
    t0, t1 = int(t0), self.n_frames if t1 is None else min(int(t1), self.n_frames)
    r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
    n, hp, wp = t1 - t0, r1 - r0, c1 - c0
    X = np.ascontiguousarray(self.raw[t0:t1, r0:r1, c0:c1]).astype(np.float64)
    rows, cols = np.mgrid[r0:r1, c0:c1]
    pixels = (rows * self.width + cols).ravel()
    out = self.W[t0:t1] @ self.samples_flat[:, pixels]
    out += self.shift[:, t0:t1].T @ self.A[pixels].T
    X -= out.reshape(n, hp, wp)
    if self.saturation_clip is not None:
        X[X > self.saturation_clip] = 0.0
    return np.asarray(X, dtype)
