"""Waveform extraction: temporal filter, per-region AP detection, centers of mass and spatial
footprints (per-cell traces are solved in :mod:`pyali.traces`).

Movies are ``[T, H, W]``; patch movies ``[T, Hp, Wp]``. Frame/pixel indices in
``region_AP`` are 0-based.
"""
from dataclasses import dataclass

import numpy as np

from .results import SparseBuilder
from .utils import movmedian_time, findpeaks, round_half_away_from_zero


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
    trace = region_trace(np.asarray(patch_raw)[:, mask], mask.size)
    peaks, pk, diag = detect_peaks(trace, median_window, edge_frames, threshold_factor)
    return coarse_ap_pixels(pm, peaks, pk, mask, origin0), diag


def region_trace(mask_values, patch_area):
    """Region trace: the in-mask pixel values ``(T, n_mask)`` summed per frame (float64) and
    divided by the patch area, as Mini ALI forms it."""
    return np.asarray(mask_values).sum(axis=1, dtype=np.float64) / float(patch_area)


def detect_peaks(trace, median_window=32, edge_frames=16, threshold_factor=4.0):
    """Peaks of one region trace: ``dF = medfilt1(trace) - trace`` with ``edge_frames`` zeroed at
    each end, thresholded at ``threshold_factor`` x its mean absolute deviation, local maxima
    by the MATLAB ``findpeaks`` rule. Returns ``(peak_values, peak_frames, diag)``."""
    T = trace.shape[0]
    dF = median_filter_1d(trace, median_window) - trace
    e = min(int(edge_frames), T)
    dF[:e] = 0.0
    dF[max(0, T - e):] = 0.0
    sigma = mean_abs_dev(dF)
    threshold = float(threshold_factor) * sigma
    thr = np.where(dF >= threshold, dF, 0.0) if threshold > 0 else np.zeros_like(dF)
    peaks, pk = findpeaks(thr)
    return peaks, pk, dict(trace=trace, dF=dF, sigma=sigma, threshold=threshold)


def coarse_ap_pixels(pm, peaks, peak_frames, mask, origin0):
    """``(k, 4) = [row, col, frame, amplitude]``: for each peak, the brightest in-mask pixel of
    the filtered frame ``pm[frame]`` (first in column-major order on ties), 0-indexed absolute."""
    region_AP = []
    for amp, t in zip(peaks, peak_frames):
        frame = np.asarray(pm[int(t)]) * mask                            # coarse location inside the mask
        pv = frame.max()
        fi = int(np.argmax(frame.ravel(order="F") == pv))               # first occurrence, column-major
        px0, py0 = np.unravel_index(fi, frame.shape, order="F")
        region_AP.append([px0 + origin0[0], py0 + origin0[1], int(t), float(amp)])
    return np.array(region_AP, dtype=np.float64).reshape(-1, 4)


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
    (``spk_fine`` lines 54-56), so the result may be shorter than ``n_pixels``. On exactly equal
    float values it breaks ties in column-major order where MATLAB's coordinate list breaks them
    in insertion order — the one accepted deviation, immaterial on denoised data.

    The queue is a heap keyed by ``(-value, column-major index)``, which pops exactly the pixel
    the argmax-over-the-queue rule picks; images with non-finite values take the original
    mask-and-argmax loop (:func:`_region_grow_brightest_mask`), whose NaN handling a heap cannot
    reproduce.
    """
    import heapq
    if not np.all(np.isfinite(current)):
        return _region_grow_brightest_mask(current, radius, n_pixels)
    Hp, Wp = current.shape
    flat_f = current.ravel(order="F")
    seed = np.unravel_index(int(np.argmax(flat_f)), current.shape, order="F")
    sr, sc = int(seed[0]), int(seed[1])
    picked = [(sr, sc)]
    if n_pixels <= 1:
        return np.array(picked, int).reshape(-1, 2)
    r2 = radius ** 2
    vals = current.tolist()                         # python floats: exact for float32 / float64
    seen = {(sr, sc)}                               # picked or queued
    heap = []
    pr, pc = sr, sc
    while len(picked) < n_pixels:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):           # 4-neighbours of the last pick
            r, c = pr + dr, pc + dc
            if (0 <= r < Hp and 0 <= c < Wp and (r, c) not in seen
                    and (r - sr) ** 2 + (c - sc) ** 2 <= r2):
                seen.add((r, c))
                heapq.heappush(heap, (-vals[r][c], c * Hp + r, r, c))
        if not heap:
            break
        _, _, pr, pc = heapq.heappop(heap)                          # brightest; ties -> column-major
        picked.append((pr, pc))
    return np.array(picked, int).reshape(-1, 2)


def _region_grow_brightest_mask(current, radius=6.0, n_pixels=25):
    """:func:`region_grow_brightest` with a boolean queue and an argmax per step (the reference
    form; used for images containing non-finite values)."""
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


# --------------------------------------------------------------------------- #
# Per-region work: reads only what it uses, returns a self-contained result
# --------------------------------------------------------------------------- #
class DensePatch:
    """The :class:`pyali.movie.CorrectedPatch` interface over a dense ``(T, H, W)`` movie."""

    def __init__(self, movie, r0, r1, c0, c1):
        self.view = movie[:, r0:r1, c0:c1]
        self.shape = self.view.shape

    def full(self):
        return self.view

    def frames(self, frames):
        return self.view[np.asarray(frames, dtype=np.int64)]

    def columns(self, mask):
        return np.asarray(self.view)[:, np.asarray(mask, bool)]


def region_source(movie, r0, r1, c0, c1):
    """The corrected block ``[r0:r1, c0:c1]`` of ``movie`` as a lazily read patch: a
    :class:`pyali.movie.CorrectedPatch` for a :class:`pyali.movie.CorrectedMovie`, a
    :class:`DensePatch` for a dense array."""
    if hasattr(movie, "region"):
        return movie.region(r0, r1, c0, c1)
    return DensePatch(movie, r0, r1, c0, c1)


class FrameSubset:
    """Rows of a ``(T, Hp, Wp)`` filtered patch at a few frames, indexable by frame number like
    the full array (``fs[t]``, ``fs[array_of_t]``, ``fs.shape``); asking for a frame that was not
    computed raises ``KeyError``."""

    def __init__(self, frames, stack, n_frames):
        self.frames = np.asarray(frames, dtype=np.int64)
        self.stack = stack
        self.shape = (int(n_frames),) + tuple(stack.shape[1:])

    def _rows(self, t):
        t = np.asarray(t, dtype=np.int64)
        i = np.searchsorted(self.frames, t)
        if np.any(i >= self.frames.size) or np.any(self.frames[np.minimum(i, self.frames.size - 1)] != t):
            raise KeyError("frame not in this FrameSubset")
        return i

    def __getitem__(self, t):
        if np.ndim(t) == 0:
            return self.stack[int(self._rows(int(t)))]
        return self.stack[self._rows(t)]


def filtered_frames(src, frames, window, half):
    """The temporally filtered patch (:func:`temporal_filter_patch`, then ``half`` frames zeroed
    at each end) at ``frames`` only, reading just their ``window``-frame neighbourhoods from
    ``src``. Bit-identical to the corresponding rows of the full filtered patch."""
    from .utils import movmedian_at, movmedian_support
    T = src.shape[0]
    want = np.unique(np.asarray(frames, dtype=np.int64))
    rows = movmedian_support(want, T, window)
    stack = src.frames(rows)
    if not np.issubdtype(stack.dtype, np.floating):
        stack = stack.astype(np.float64)
    pm = movmedian_at(stack, rows, want, T, window)
    pm -= stack[np.searchsorted(rows, want)]                     # movmedian(a) - a, in a's dtype
    pm[(want < half) | (want >= T - half)] = 0.0
    return FrameSubset(want, pm, T)


def _full_filtered_max(src, window, half):
    """Per-pixel maximum over time of the full filtered patch (the fallback footprint's input)."""
    pm = temporal_filter_patch(src.full(), window)
    T = pm.shape[0]
    pm[:half] = 0.0
    pm[T - half:] = 0.0
    return pm.max(axis=0)


def _sparse_plane(plane):
    """``(pixel, values)`` of an ``(H, W)`` plane's nonzeros in C order (what
    :meth:`pyali.results.SparseBuilder.add_plane` stores)."""
    flat = np.ascontiguousarray(plane).ravel()
    nz = np.flatnonzero(flat)
    return nz.astype(np.uint32), flat[nz]


@dataclass
class RegionResult:
    """Everything one region contributes, with cell rows local to the region (``-1`` = none)."""
    aps: np.ndarray                     # (k, 4) [row, col, frame, amplitude], after dedup
    coms: np.ndarray                    # (m, 3) [row, col, frame]
    spike_rows: np.ndarray              # (m,) local footprint row of each COM, or -1
    planes: list                        # [(pixel uint32, values float64)] in footprint order
    centers: list                       # [[row, col]] per footprint (1-indexed)
    sources: list                       # SOURCES entry per footprint


def process_region(movie, region, pixel_list, p, height, width):
    """AP detection -> centers of mass -> DBSCAN clustering -> footprints for one region.

    Pure: depends only on its arguments and returns a :class:`RegionResult`, so regions can run
    in any order or in parallel. Reads the region trace from the in-mask pixels and the filtered
    patch only at the detected peak frames (their 8-frame neighbourhoods); the full filtered
    patch is built only when the region falls back to its max-projection footprint.
    """
    from sklearn.cluster import DBSCAN

    fw = p.filter_window
    half = int(round_half_away_from_zero(fw / 2))
    patch_rows, patch_cols, origin0 = compute_patch(region["Centroid"], region["BoundingBox"],
                                                    p.patch_size, height, width)
    sel_full = build_selection_map(pixel_list, height, width)
    sel_patch = sel_full[np.ix_(patch_rows - 1, patch_cols - 1)]
    r0, r1 = patch_rows[0] - 1, patch_rows[-1]
    c0, c1 = patch_cols[0] - 1, patch_cols[-1]
    src = region_source(movie, r0, r1, c0, c1)
    mask = np.asarray(sel_patch) > 0

    aps = np.zeros((0, 4))
    coms = np.zeros((0, 3))
    spike_rows = np.zeros(0, dtype=np.int64)

    def fallback(reason):
        plane = region_fallback_footprint(sel_full, sel_patch, _full_filtered_max(src, fw, half),
                                          height, width)
        return RegionResult(aps, coms, spike_rows, [_sparse_plane(plane)],
                            [region_center_rowcol(region)], [reason])

    trace = region_trace(src.columns(mask), mask.size)
    peaks, pk, _diag = detect_peaks(trace, p.detect_median_window, p.detect_edge_frames,
                                    p.threshold_factor)
    if len(pk) == 0:
        return fallback("no_ap")
    pm = filtered_frames(src, pk, fw, half)
    region_AP = dedup_close_peaks(coarse_ap_pixels(pm, peaks, pk, mask, origin0),
                                  p.min_peak_interval)
    if len(region_AP) == 0:
        return fallback("no_ap")
    aps = region_AP

    region_COMs = com_via_svd(pm, region_AP[:, 2].astype(int), patch_rows, patch_cols,
                              sel_patch, p.svd_rank, p.com_radius, p.com_n_pixels)
    if len(region_COMs) == 0:
        return fallback("no_com")
    coms = region_COMs
    labels = DBSCAN(eps=p.dbscan_eps, min_samples=p.dbscan_min_pts).fit_predict(region_COMs[:, :2])
    if np.all(labels == -1):
        spike_rows = np.full(len(region_COMs), -1, dtype=np.int64)
        return fallback("all_noise")

    # cluster k (in label order, skipping empty labels) becomes the next footprint
    kept = [k for k in range(labels.max() + 1) if np.any(labels == k)]
    cluster_times = [region_COMs[labels == k, 2].astype(int) for k in kept]
    pl, ce = cluster_footprints(pm, cluster_times, sel_patch, patch_rows, patch_cols,
                                height, width)
    row_of_label = {k: i for i, k in enumerate(kept)}
    spike_rows = np.array([row_of_label.get(int(lab), -1) for lab in labels], dtype=np.int64)
    return RegionResult(aps, coms, spike_rows, [_sparse_plane(pl_) for pl_ in pl],
                        [list(c) for c in ce], ["spiking"] * len(pl))


# --------------------------------------------------------------------------- #
# The region loop: sequential or a fork process pool, assembled in region order
# --------------------------------------------------------------------------- #
_POOL_CONTEXT = None                   # (movie, regions, spatial_footprints, p, H, W) for workers


def _pool_init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)        # one region per process: no nested BLAS threads


def _pool_task(c):
    movie, regions, spatial_footprints, p, height, width = _POOL_CONTEXT
    return process_region(movie, regions[c], spatial_footprints[c], p, height, width)


def resolve_workers(workers):
    """``workers`` for :func:`extract_footprints`: ``None`` or ``0`` -> ``os.cpu_count()``."""
    import os
    return int(workers) if workers else (os.cpu_count() or 1)


def iter_region_results(movie, regions, spatial_footprints, p, height, width, workers=1):
    """Yield ``(c, RegionResult)`` in region order, computed inline (``workers == 1``) or by a
    fork-based process pool: the movie and the other inputs are inherited copy-on-write, so
    nothing large is pickled; results come back in order (``imap``)."""
    n = len(regions)
    workers = max(1, min(resolve_workers(workers), n))
    if workers == 1:
        for c in range(n):
            yield c, process_region(movie, regions[c], spatial_footprints[c], p, height, width)
        return
    import multiprocessing as mp
    global _POOL_CONTEXT
    _POOL_CONTEXT = (movie, regions, spatial_footprints, p, height, width)
    try:
        with mp.get_context("fork").Pool(workers, initializer=_pool_init) as pool:
            chunk = max(1, n // (workers * 8))
            for c, res in enumerate(pool.imap(_pool_task, range(n), chunksize=chunk)):
                yield c, res
    finally:
        _POOL_CONTEXT = None


def extract_footprints(movie, regions, spatial_footprints, p, height, width, verbose=False,
                       workers=1):
    """Per-region loop: AP detection -> center of mass -> DBSCAN clustering -> footprints.

    Parameters
    ----------
    movie : :class:`pyali.movie.CorrectedMovie` or a dense (T, H, W) float array
        Read per region through :func:`region_source` (in-mask pixels for the trace, peak-frame
        neighbourhoods for the filtered patch).
    regions, spatial_footprints : from :func:`pyali.segmentation.cell_segmentation`
    p : Params
    height, width : int
    verbose : bool
    workers : int
        Region processes (1 = inline; ``None``/0 = all CPUs). Each region is a pure
        :func:`process_region` task, so the output does not depend on ``workers``.

    Returns
    -------
    (APs, COMs, footprints, footprint_center, region_of_footprint, region_of_ap,
     region_of_spike, spike_cell_row, source_of_footprint)
        ``APs`` : (nAP, 4) = ``[row, col, frame, amplitude]`` (0-indexed);
        ``COMs`` : (nCOM, 3) = ``[row, col, frame]`` (1-indexed row/col) -- ``nCOM <= nAP``
        because :func:`com_via_svd` drops APs whose centroid is undefined or off-mask;
        ``footprints`` : :class:`pyali.results.SparseFootprints` (N cells, float64 values, packed
        as each region's planes arrive, so no ``(H, W, N)`` stack exists);
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
    APs, COMs, centers = [], [], []
    footprints = SparseBuilder(height, width)
    region_of_footprint, region_of_ap, region_of_spike, spike_cell_row = [], [], [], []
    source_of_footprint = []
    n = len(regions)
    for c, res in iter_region_results(movie, regions, spatial_footprints, p, height, width,
                                      workers):
        if verbose and (c % 100 == 0 or c == n - 1):
            print(f"[pyali]   region {c + 1}/{n}", flush=True)
        if len(res.aps):
            APs.append(res.aps)
            region_of_ap.extend([c] * len(res.aps))
        if len(res.coms):
            COMs.append(res.coms)
            region_of_spike.extend([c] * len(res.coms))
        offset = footprints.n()
        spike_cell_row.extend(int(r) + offset if r >= 0 else -1 for r in res.spike_rows)
        for pix, vals in res.planes:
            footprints.add_sparse(pix, vals)
        centers.extend(res.centers)
        region_of_footprint.extend([c] * len(res.planes))
        source_of_footprint.extend(res.sources)

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
