"""Waveform extraction: per-region AP detection and centers of mass, field-wide spike assignment
(:mod:`pyali.assign`), per-cluster spatial footprints (per-cell traces are solved in
:mod:`pyali.traces`).

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


SOURCES = ("spiking", "no_ap", "no_com", "unclustered")   # cells.source vocabulary


# --------------------------------------------------------------------------- #
# Pass 1: per region, spikes only (detection -> dedup -> centres of mass)
# --------------------------------------------------------------------------- #
@dataclass
class RegionSpikes:
    """One region's detected APs and localized spikes."""
    aps: np.ndarray                     # (k, 4) [row, col, frame, amplitude], after dedup
    coms: np.ndarray                    # (m, 4) [row, col, frame, peak_df], 1-indexed row/col
    reason: str                         # "spiking" (m > 0), "no_ap" or "no_com"


def region_spikes(movie, region, pixel_list, p, height, width):
    """AP detection and sub-pixel localization for one region (pure; runs in a worker or inline).

    Reads the region trace from the in-mask pixels and the filtered patch only at the detected
    peak frames. Each localized spike carries ``peak_df``, the largest in-mask value of its
    filtered frame (the coarse spike amplitude Mini ALI keeps when merging duplicates).
    """
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
    none4 = np.zeros((0, 4))

    trace = region_trace(src.columns(mask), mask.size)
    peaks, pk, _diag = detect_peaks(trace, p.detect_median_window, p.detect_edge_frames,
                                    p.threshold_factor)
    if len(pk) == 0:
        return RegionSpikes(none4, none4, "no_ap")
    pm = filtered_frames(src, pk, fw, half)
    region_AP = dedup_close_peaks(coarse_ap_pixels(pm, peaks, pk, mask, origin0),
                                  p.min_peak_interval)
    if len(region_AP) == 0:
        return RegionSpikes(none4, none4, "no_ap")
    coms = com_via_svd(pm, region_AP[:, 2].astype(int), patch_rows, patch_cols,
                       sel_patch, p.svd_rank, p.com_radius, p.com_n_pixels)
    if len(coms) == 0:
        return RegionSpikes(region_AP, none4, "no_com")
    peak_df = np.array([float(np.asarray(pm[int(t)])[mask].max()) for t in coms[:, 2]])
    return RegionSpikes(region_AP, np.column_stack([coms, peak_df]), "spiking")


# --------------------------------------------------------------------------- #
# Pass 2: per cluster, the footprint from the cluster's own spike frames
# --------------------------------------------------------------------------- #
def cluster_footprint(movie, rc, frames, p, height, width):
    """Footprint of one spike cluster (Mini ALI's per-cluster loop): the mean of the filtered
    frames at the cluster's spikes over a ``(2r+1)^2`` window (``r = p.footprint_radius_px``)
    centred on the rounded mean spike COM, negatives clipped to zero.

    ``rc`` (k, 2) 1-indexed spike COMs and ``frames`` (k,) 0-based frames, in the cluster's
    spike order. Returns ``((pixel, values), centre)`` with the window's nonzero pixels in C
    order and ``centre`` = the mean COM ``[row, col]`` (1-indexed).
    """
    fw = p.filter_window
    half = int(round_half_away_from_zero(fw / 2))
    r = int(p.footprint_radius_px)
    rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
    frames = np.asarray(frames, dtype=np.int64).ravel()
    centre = rc.mean(axis=0)
    cr = round_half_away_from_zero(centre).astype(int)            # 1-indexed
    r0, r1 = max(cr[0] - r, 1) - 1, min(cr[0] + r, height)        # 0-indexed half-open
    c0, c1 = max(cr[1] - r, 1) - 1, min(cr[1] + r, width)
    src = region_source(movie, r0, r1, c0, c1)
    stack = filtered_frames(src, frames, fw, half)[frames]        # (k, hp, wp), spike order
    fp = stack.mean(axis=0)
    fp[fp < 0] = 0
    rows, cols = np.nonzero(fp)
    pixel = ((rows + r0) * width + (cols + c0)).astype(np.uint32)
    return (pixel, fp[rows, cols].astype(np.float64)), [float(centre[0]), float(centre[1])]


# --------------------------------------------------------------------------- #
# Pass 3: per region without a cluster, the max-projection fallback footprint
# --------------------------------------------------------------------------- #
def region_fallback(movie, region, pixel_list, p, height, width):
    """The region's max-projection footprint over the whole filtered patch and its centre, for
    regions that end up in no spike cluster."""
    fw = p.filter_window
    half = int(round_half_away_from_zero(fw / 2))
    patch_rows, patch_cols, _ = compute_patch(region["Centroid"], region["BoundingBox"],
                                              p.patch_size, height, width)
    sel_full = build_selection_map(pixel_list, height, width)
    sel_patch = sel_full[np.ix_(patch_rows - 1, patch_cols - 1)]
    src = region_source(movie, patch_rows[0] - 1, patch_rows[-1], patch_cols[0] - 1, patch_cols[-1])
    plane = region_fallback_footprint(sel_full, sel_patch, _full_filtered_max(src, fw, half),
                                      height, width)
    return _sparse_plane(plane), region_center_rowcol(region)


# --------------------------------------------------------------------------- #
# Tasks: inline or a fork process pool, results in order
# --------------------------------------------------------------------------- #
_POOL_CONTEXT = None                   # (fn, shared args) inherited by forked workers


def _pool_init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)        # one task per process: no nested BLAS threads


def _pool_task(item):
    fn, shared = _POOL_CONTEXT
    return fn(*shared, item)


def resolve_workers(workers):
    """``workers`` for :func:`run_tasks`: ``None`` or ``0`` -> ``os.cpu_count()``."""
    import os
    return int(workers) if workers else (os.cpu_count() or 1)


def run_tasks(fn, items, shared, workers=1):
    """Yield ``fn(*shared, item)`` for each item, in order: inline (``workers == 1``) or from a
    fork-based process pool, where ``shared`` (the movie, regions, params) is inherited
    copy-on-write and only the small ``items`` and results are pickled."""
    items = list(items)
    n = len(items)
    workers = max(1, min(resolve_workers(workers), n)) if n else 1
    if workers == 1:
        for it in items:
            yield fn(*shared, it)
        return
    import multiprocessing as mp
    global _POOL_CONTEXT
    _POOL_CONTEXT = (fn, tuple(shared))
    try:
        with mp.get_context("fork").Pool(workers, initializer=_pool_init) as pool:
            yield from pool.imap(_pool_task, items, chunksize=max(1, n // (workers * 8)))
    finally:
        _POOL_CONTEXT = None


def _region_spikes_task(movie, regions, spatial_footprints, p, height, width, c):
    return region_spikes(movie, regions[c], spatial_footprints[c], p, height, width)


def _cluster_task(movie, p, height, width, item):
    rc, frames = item
    return cluster_footprint(movie, rc, frames, p, height, width)


def _fallback_task(movie, regions, spatial_footprints, p, height, width, c):
    return region_fallback(movie, regions[c], spatial_footprints[c], p, height, width)


# --------------------------------------------------------------------------- #
# The extraction: spikes per region -> field-wide merge and clustering -> footprints
# --------------------------------------------------------------------------- #
def extract_footprints(movie, regions, spatial_footprints, p, height, width, verbose=False,
                       workers=1):
    """Spikes per region, assigned to cells across the whole field, and one footprint per cell.

    1. Per region (parallel): AP detection, dedup, sub-pixel centres of mass (:func:`region_spikes`).
    2. Field-wide: same-frame spikes within ``p.spike_merge_distance_px`` merged, keeping the
       largest ``peak_df`` (:func:`pyali.assign.merge_close_spikes_keep_max`); the survivors
       clustered by spike density (:func:`pyali.assign.ali_clustering`). Cells packed into one
       segmentation region separate here when their spike COM clouds form separate density peaks.
    3. Per cluster (parallel): footprint from the cluster's own spike frames (:func:`cluster_footprint`).
    4. Regions none of whose spikes reached a cluster keep a max-projection footprint (parallel):
       ``no_ap`` (no AP), ``no_com`` (no valid centre), ``unclustered`` (spikes, but none -- after
       following merged spikes to their survivor -- in any cluster).

    Parameters
    ----------
    movie : :class:`pyali.movie.CorrectedMovie` or a dense (T, H, W) float array
    regions, spatial_footprints : from :func:`pyali.segmentation.cell_segmentation`
    p : Params
    height, width : int
    verbose : bool
    workers : int
        Processes for the per-region and per-cluster tasks (1 = inline; ``None``/0 = all CPUs).
        The output does not depend on it.

    Returns
    -------
    (APs, COMs, footprints, footprint_center, region_of_footprint, region_of_ap,
     region_of_spike, spike_cell_row, source_of_footprint, spike_info)
        ``APs`` : (nAP, 4) ``[row, col, frame, amplitude]`` (0-indexed), every region's APs;
        ``COMs`` : (nCOM, 3) ``[row, col, frame]`` (1-indexed row/col), every localized spike in
        region order (before the merge);
        ``footprints`` : :class:`pyali.results.SparseFootprints`, cluster cells (in cluster label
        order) then fallback cells (in region order);
        ``footprint_center`` : (N, 2) ``[row, col]`` -- a cluster cell's mean spike COM;
        ``region_of_footprint`` : (N,) int -- the region most of a cluster's spikes came from;
        ``region_of_ap`` / ``region_of_spike`` : region of each ``APs`` / ``COMs`` row;
        ``spike_cell_row`` : (nCOM,) footprint row of each spike, -1 when merged away, unclustered
        or in a fallback region;
        ``source_of_footprint`` : list of :data:`SOURCES` entries;
        ``spike_info`` : dict of per-spike arrays ``peak_df``, ``kept`` (survived the merge),
        ``cluster`` (density-cluster label, 0 = none) and counts ``n_spikes_before_merge``,
        ``n_spikes_after_merge``, ``n_spikes_clustered``, ``n_clusters``.
    """
    from . import assign
    n = len(regions)
    shared_regions = (movie, regions, spatial_footprints, p, height, width)

    # ---- 1. per-region spikes ----
    per_region = []
    for c, res in enumerate(run_tasks(_region_spikes_task, range(n), shared_regions, workers)):
        if verbose and (c % 200 == 0 or c == n - 1):
            print(f"[pyali]   region {c + 1}/{n}", flush=True)
        per_region.append(res)
    APs = [r.aps for r in per_region if len(r.aps)]
    region_of_ap = np.concatenate([[c] * len(r.aps) for c, r in enumerate(per_region)] + [[]]).astype(int)
    S = np.vstack([r.coms for r in per_region if len(r.coms)] or [np.zeros((0, 4))])
    region_of_spike = np.concatenate([[c] * len(r.coms) for c, r in enumerate(per_region)] + [[]]).astype(int)

    # ---- 2. field-wide merge and density clustering ----
    survivor, kept_order = assign.merge_close_spikes_keep_max(S, p.spike_merge_distance_px)
    kept = survivor == np.arange(len(S))
    cluster = np.zeros(len(S), dtype=int)
    if len(kept_order):
        labels, _, _, _ = assign.ali_clustering(S[kept_order, :2], p.ali_cluster_factor, height,
                                                width, p.ali_cluster_threshold, p.ali_flood_margin)
        cluster[kept_order] = labels
    labels_kept = cluster[kept_order]
    cluster_ids = np.unique(labels_kept[labels_kept > 0])
    members_of = [kept_order[labels_kept == L] for L in cluster_ids]       # merged (time) order
    if verbose:
        print(f"[pyali]   spikes: {len(S)} localized, {int(kept.sum())} after the same-frame merge, "
              f"{int((labels_kept > 0).sum())} in {len(cluster_ids)} density clusters", flush=True)

    # ---- 3. per-cluster footprints ----
    footprints = SparseBuilder(height, width)
    centers, region_of_footprint, source_of_footprint = [], [], []
    spike_cell_row = np.full(len(S), -1, dtype=int)
    items = [(S[m, :2], S[m, 2].astype(int)) for m in members_of]
    for k, ((pix, vals), centre) in enumerate(run_tasks(_cluster_task, items,
                                                        (movie, p, height, width), workers)):
        m = members_of[k]
        footprints.add_sparse(pix, vals)
        centers.append(centre)
        counts = np.bincount(region_of_spike[m])
        region_of_footprint.append(int(np.argmax(counts)))                # majority; ties -> lowest
        source_of_footprint.append("spiking")
        spike_cell_row[m] = k

    # ---- 4. fallback footprints for regions without a cluster ----
    in_cluster = cluster[survivor] > 0                                     # via the survivor
    has_cluster = np.zeros(n, dtype=bool)
    has_cluster[np.unique(region_of_spike[in_cluster])] = True
    fallback = [c for c in range(n) if not has_cluster[c]]
    reasons = [per_region[c].reason if per_region[c].reason != "spiking" else "unclustered"
               for c in fallback]
    for c, reason, ((pix, vals), centre) in zip(
            fallback, reasons, run_tasks(_fallback_task, fallback, shared_regions, workers)):
        footprints.add_sparse(pix, vals)
        centers.append(centre)
        region_of_footprint.append(c)
        source_of_footprint.append(reason)

    footprint_center = np.array(centers, dtype=float).reshape(-1, 2)
    APs = np.vstack(APs) if APs else np.zeros((0, 4))
    spike_info = dict(peak_df=S[:, 3].copy(), kept=kept, cluster=cluster,
                      n_spikes_before_merge=int(len(S)), n_spikes_after_merge=int(kept.sum()),
                      n_spikes_clustered=int((labels_kept > 0).sum()),
                      n_clusters=int(len(cluster_ids)))
    return (APs, S[:, :3].copy(), footprints.build(), footprint_center,
            np.asarray(region_of_footprint, int), region_of_ap, region_of_spike, spike_cell_row,
            source_of_footprint, spike_info)


def region_center_rowcol(region):
    """Region center as ``[row, col]`` from the region's Centroid ``[col, row]``."""
    cx, cy = float(region["Centroid"][0]), float(region["Centroid"][1])
    return [cy, cx]
