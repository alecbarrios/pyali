"""Waveform extraction: temporal filter, per-region AP detection, centers of mass,
spatial footprints, and per-cell trace extraction.

Movies are ``[T, H, W]``; patch movies ``[T, Hp, Wp]``. Frame/pixel indices in
``region_AP`` are 0-based.
"""
import numpy as np

from .utils import movmedian_time, findpeaks, round_half_away_from_zero


# --------------------------------------------------------------------------- #
# Temporal filter
# --------------------------------------------------------------------------- #
def temporal_filter(movie, window=8):
    """Moving-median high-pass along time, sign-flipped so downward spikes become positive.

    Parameters
    ----------
    movie : (T, H, W) float array (time = axis 0)
    window : int
        Moving-median window.

    Returns
    -------
    (T, H, W) ndarray

    Notes
    -----
    Materializes a second full-movie array. ``extract_footprints`` only ever reads ~27x27
    patches of the result, so prefer :func:`temporal_filter_patch` there — see the
    ``filtered_movie=None`` path.
    """
    # -(movie - movmedian) == movmedian - movie; computed in place to avoid an extra copy.
    filtered = movmedian_time(movie, window, axis=0)
    filtered -= movie
    return filtered


def temporal_filter_patch(movie, r0, r1, c0, c1, window=8):
    """:func:`temporal_filter` restricted to one spatial patch.

    The filter runs purely along time and is independent per pixel, so filtering a spatial
    sub-block is **bit-identical** to filtering the whole movie and then slicing it — while
    touching only the pixels a region actually needs.

    Parameters
    ----------
    movie : (T, H, W) float array
    r0, r1, c0, c1 : int
        0-indexed half-open patch bounds.
    window : int

    Returns
    -------
    (T, r1-r0, c1-c0) ndarray
    """
    sub = movie[:, r0:r1, c0:c1]
    filtered = movmedian_time(sub, window, axis=0)
    filtered -= sub
    return filtered


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


def extract_footprints(movie, filtered_movie, regions, spatial_footprints, p,
                       height, width, verbose=False):
    """Per-region loop: AP detection -> center of mass -> DBSCAN clustering -> footprints.

    Parameters
    ----------
    movie : (T, H, W) float array
    filtered_movie : (T, H, W) float array or None
        Pre-filtered movie. Pass ``None`` (preferred) to filter each region's patch on demand
        via :func:`temporal_filter_patch`: only region patches are ever read from it, so the
        global array is both a full extra copy of the movie and mostly wasted work. The two
        paths are bit-identical — the filter is per-pixel along time.
    regions, spatial_footprints : from :func:`pyali.segmentation.cell_segmentation`
    p : Params
    height, width : int
    verbose : bool

    Returns
    -------
    (APs, COMs, footprint, footprint_center, region_of_footprint, region_of_ap,
     region_of_spike, spike_cell_row, source_of_footprint)
        ``APs`` : (nAP, 4) = ``[row, col, frame, amplitude]`` (0-indexed);
        ``COMs`` : (nCOM, 3) = ``[row, col, frame]`` (1-indexed row/col) -- ``nCOM <= nAP``
        because :func:`com_via_svd` drops APs whose centroid is undefined or off-mask;
        ``footprint`` : (H, W, N); ``footprint_center`` : (N, 2) = ``[row, col]``;
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
    APs, COMs, planes, centers = [], [], [], []
    region_of_footprint, region_of_ap, region_of_spike, spike_cell_row = [], [], [], []
    source_of_footprint = []

    def _fallback(sel_full, sel_patch, pm_max, region, c, reason):
        planes.append(region_fallback_footprint(sel_full, sel_patch, pm_max, height, width))
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
        if filtered_movie is None:                        # filter just this patch, on demand
            pm = temporal_filter_patch(movie, r0, r1, c0, c1, fw)
        else:
            pm = filtered_movie[:, r0:r1, c0:c1].copy()
        pm[:half] = 0.0
        pm[T - half:] = 0.0
        pm_max = pm.max(axis=0)

        region_AP, _diag = detect_region_aps(movie[:, r0:r1, c0:c1], sel_patch, pm, origin0,
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
        row_of_label = {k: len(planes) + i for i, k in enumerate(kept)}
        spike_cell_row.extend(row_of_label.get(int(lab), -1) for lab in labels)
        planes.extend(pl); centers.extend(ce)
        region_of_footprint.extend([c] * len(pl))
        source_of_footprint.extend(["spiking"] * len(pl))

    footprint = np.stack(planes, axis=2) if planes else np.zeros((height, width, 0))
    footprint_center = np.array(centers, dtype=float).reshape(-1, 2)
    APs = np.vstack(APs) if APs else np.zeros((0, 4))
    COMs = np.vstack(COMs) if COMs else np.zeros((0, 3))
    return (APs, COMs, footprint, footprint_center,
            np.asarray(region_of_footprint, int), np.asarray(region_of_ap, int),
            np.asarray(region_of_spike, int), np.asarray(spike_cell_row, int),
            source_of_footprint)


def region_center_rowcol(region):
    """Region center as ``[row, col]`` from the region's Centroid ``[col, row]``."""
    cx, cy = float(region["Centroid"][0]), float(region["Centroid"][1])
    return [cy, cx]


# --------------------------------------------------------------------------- #
# Trace extraction
# --------------------------------------------------------------------------- #
_PROJECT_CHUNK_BYTES = 2 << 30       # cap on the promoted per-chunk temporary (~2 GB)


def project_movie(op, movie, chunk_bytes=_PROJECT_CHUNK_BYTES):
    """Apply a ``[N, H*W]`` spatial operator to a ``[T, H, W]`` movie -> ``[N, T]``.

    Evaluated in chunks along time. Footprint planes are built with ``np.zeros(...)`` and so are
    always float64, which makes ``op`` float64; a ``float32`` movie (``compute_dtype='float32'``)
    would then be promoted to float64 by the matmul **in one piece** — a full second copy of the
    movie, 43 GB for an 800x800x8389 acquisition, silently undoing the point of float32. Chunking
    bounds that temporary to roughly ``chunk_bytes``.

    Time is an *output* axis, not the contracted one: every output element is still the full
    ``H*W`` dot product, so the result is arithmetically identical to the unchunked form.

    Parameters
    ----------
    op : (N, H*W) float array
    movie : (T, H, W) float array
    chunk_bytes : int
        Approximate ceiling on the temporary materialized per chunk.

    Returns
    -------
    (N, T) ndarray of ``op.dtype``
    """
    T, H, W = movie.shape
    N = op.shape[0]
    flat = movie.reshape(T, H * W)                              # view, C-order
    out = np.empty((N, T), dtype=op.dtype)
    per_frame = H * W * np.dtype(op.dtype).itemsize
    step = max(1, min(T, int(chunk_bytes // max(per_frame, 1))))
    for t0 in range(0, T, step):
        t1 = min(T, t0 + step)
        out[:, t0:t1] = op @ flat[t0:t1].T                      # [N, H*W] @ [H*W, chunk]
    return out


def pinv_traces(movie, footprint):
    """Extract per-footprint temporal traces via the spatial pseudoinverse.

    ``cell_traces = pinv(footprint) @ (-movie)``.

    Parameters
    ----------
    movie : (T, H, W) float array (processed)
    footprint : (H, W, N) float array

    Returns
    -------
    (N, T) ndarray

    Notes
    -----
    Footprint and movie are flattened with a consistent C-order pixel ordering. The contraction
    runs through :func:`project_movie`, which chunks over time so a float32 movie is never
    promoted to float64 all at once.
    """
    H, W = movie.shape[1], movie.shape[2]
    flat_fp = footprint.reshape(H * W, footprint.shape[2])      # pixel = row*W + col
    rcond = max(flat_fp.shape) * np.finfo(np.float64).eps       # pseudoinverse tolerance
    pinv_fp = np.linalg.pinv(flat_fp, rcond=rcond)             # [N, H*W]
    return -project_movie(pinv_fp, movie)                       # negate the small result


# --------------------------------------------------------------------------- #
# Whitened GLS trace extraction (opt-in; Params.whiten_traces)
# --------------------------------------------------------------------------- #
def per_pixel_noise_map(filtered_movie, std_frames, floor_pct=5.0):
    """Per-pixel high-frequency noise std from the baseline (no-stimulus) frames.

    ``sigma_pix[h, w] = 1.4826 * MAD_over_time( filtered_movie[std_frames, h, w] )`` — the same
    moving-median high-pass band the SNR metric scores. Floored at the ``floor_pct`` percentile
    of the positive values so that ``1/sigma^2`` weights never explode.

    Parameters
    ----------
    filtered_movie : (T, H, W) float array (from :func:`temporal_filter`)
    std_frames : array of int (0-indexed baseline frames)
    floor_pct : float

    Returns
    -------
    (H, W) ndarray of per-pixel noise std.
    """
    base = filtered_movie[np.asarray(std_frames, int)]                 # [n_base, H, W]
    med = np.median(base, axis=0)
    sigma = 1.4826 * np.median(np.abs(base - med[None]), axis=0)       # [H, W]
    pos = sigma[sigma > 0]
    if pos.size:
        sigma = np.maximum(sigma, np.percentile(pos, floor_pct))
    else:                                                              # degenerate: uniform weights
        sigma = np.ones_like(sigma)
    return sigma


def footprint_isolated_mask(footprint):
    """Boolean ``[N]`` mask: True where a footprint's support bounding box overlaps no other.

    Uses nonzero-pixel bounding boxes (conservative: touching boxes count as overlapping), so
    cells that share any spatial support fall back to the faithful pinv row.
    """
    H, W, N = footprint.shape
    boxes = np.zeros((N, 4))                                           # rmin, rmax, cmin, cmax
    for n in range(N):
        rows = np.any(footprint[:, :, n] != 0, axis=1)
        cols = np.any(footprint[:, :, n] != 0, axis=0)
        if not rows.any():
            boxes[n] = (-1, -1, -1, -1)                               # empty footprint
            continue
        rr = np.where(rows)[0]; cc = np.where(cols)[0]
        boxes[n] = (rr[0], rr[-1], cc[0], cc[-1])
    isolated = np.ones(N, bool)
    for i in range(N):
        ri0, ri1, ci0, ci1 = boxes[i]
        if ri0 < 0:
            continue
        for j in range(N):
            if j == i or boxes[j, 0] < 0:
                continue
            rj0, rj1, cj0, cj1 = boxes[j]
            if ri0 <= rj1 and rj0 <= ri1 and ci0 <= cj1 and cj0 <= ci1:   # boxes intersect
                isolated[i] = False
                break
    return isolated


def whitened_gls_traces(movie, footprint, noise_map, ridge_frac=1e-3):
    """Noise-weighted (GLS/BLUE) trace extraction.

    ``cell_traces = (Fᵀ W F + λI)⁻¹ Fᵀ W (-movie)`` with ``W = diag(1/sigma_pix²)`` — the
    minimum-variance estimator under spatially heteroscedastic pixel noise, versus the unweighted
    :func:`pinv_traces` which implicitly assumes equal per-pixel noise.

    Parameters
    ----------
    movie : (T, H, W) float array (processed)
    footprint : (H, W, N) float array
    noise_map : (H, W) per-pixel noise std (see :func:`per_pixel_noise_map`)
    ridge_frac : float
        Ridge ``λ`` as a fraction of ``mean(diag(FᵀWF))`` (scale-invariant safety margin).

    Returns
    -------
    (N, T) ndarray

    Notes
    -----
    Same C-order pixel flattening as :func:`pinv_traces`. Uses the true single-``W`` GLS operator
    (not "divide the footprint by variance then pinv", which would apply ``W`` twice). The movie
    contraction goes through :func:`project_movie`, which chunks over time so a float32 movie is
    never promoted to float64 all at once.
    """
    H, W = movie.shape[1], movie.shape[2]
    N = footprint.shape[2]
    flat_fp = footprint.reshape(H * W, N)                             # [H*W, N]
    w = 1.0 / (noise_map.reshape(H * W) ** 2)                         # [H*W] inverse-variance
    FtW = flat_fp.T * w[None, :]                                      # [N, H*W]
    A = FtW @ flat_fp                                                 # [N, N]
    A = A + ridge_frac * np.mean(np.diag(A)) * np.eye(N)              # ridge (relative)
    B = -project_movie(FtW, movie)                                    # [N, T]
    return np.linalg.solve(A, B)                                     # [N, T]


def extract_cell_traces(movie, footprint, p, noise_map=None, verbose=False):
    """Dispatch trace extraction per ``p.whiten_traces``.

    Default (``whiten_traces=False``): the baseline unweighted :func:`pinv_traces`.
    When enabled: whitened GLS (:func:`whitened_gls_traces`); if ``p.whiten_isolated_only`` the
    GLS rows are used only for cells whose footprints overlap no other, and overlapping cells keep
    the faithful pinv row (guards against neighbor-crosstalk). Requires a precomputed
    ``noise_map`` (see :func:`per_pixel_noise_map`) built from the baseline frames.
    """
    if not getattr(p, "whiten_traces", False):
        return pinv_traces(movie, footprint)
    if noise_map is None:
        raise ValueError("whiten_traces=True requires a noise_map (per_pixel_noise_map)")
    gls = whitened_gls_traces(movie, footprint, noise_map, p.whiten_ridge)
    if not getattr(p, "whiten_isolated_only", True):
        return gls
    ols = pinv_traces(movie, footprint)                              # faithful rows for overlappers
    isolated = footprint_isolated_mask(footprint)
    if verbose:
        print(f"[pyali]   whitened {int(isolated.sum())}/{len(isolated)} isolated cells; "
              f"{int((~isolated).sum())} overlapping cells kept faithful pinv", flush=True)
    out = ols.copy()
    out[isolated] = gls[isolated]
    return out
