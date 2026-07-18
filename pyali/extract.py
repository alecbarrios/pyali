"""Waveform extraction: temporal filter, per-region AP detection, centers of mass,
spatial footprints, and per-cell trace extraction.

Movies are ``[T, H, W]``; patch movies ``[T, Hp, Wp]``. Frame/pixel indices in
``region_AP`` are 0-based.
"""
import numpy as np

from .utils import movmedian_time, findpeaks, uniquetol_reps, round_half_away_from_zero


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
    """
    # -(movie - movmedian) == movmedian - movie; computed in place to avoid an extra copy.
    filtered = movmedian_time(movie, window, axis=0)
    filtered -= movie
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
def apply_dog_gain(trace, dog):
    """Scale a 1-D trace by the DOG kernel's center value (a unit gain for the normalized
    kernel). Used only to shape the noise estimate below."""
    return trace * dog[len(dog) // 2]


def forward_rolling_min_subtract(trace, look=5):
    """Subtract, from each sample, the minimum of the next ``look`` samples.

    ``out[i] = trace[i] - min(trace[i+1 : i+1+look])``; the last sample is left at 0.
    """
    n = len(trace)
    out = np.zeros(n)
    for i in range(n - 1):
        hi = min(i + look, n - 1)
        out[i] = trace[i] - np.min(trace[i + 1:hi + 1])
    return out


# --------------------------------------------------------------------------- #
# Per-region AP detection
# --------------------------------------------------------------------------- #
def detect_region_aps(patch_movie, selection_map_patch, origin0, dog, std_frames,
                      threshold_factor=3.25, filter_window=8):
    """Detect action potentials in one region's filtered patch movie.

    Parameters
    ----------
    patch_movie : (T, Hp, Wp) float array (from :func:`temporal_filter`)
    selection_map_patch : (Hp, Wp) array
        Region mask cropped to the patch.
    origin0 : (row0, col0)
        0-indexed patch top-left corner.
    dog : (K,) array
        DOG kernel for the noise-estimate gain.
    std_frames : array of int
        0-indexed baseline frame indices for the noise std.
    threshold_factor : float
    filter_window : int

    Returns
    -------
    (region_AP, diag)
        ``region_AP`` : (k, 4) = ``[row, col, frame, amplitude]`` (0-indexed, pre-dedup);
        ``diag`` : dict of intermediate traces/thresholds.
    """
    pm = np.array(patch_movie, dtype=np.float64, copy=True)
    T = pm.shape[0]
    half = int(round_half_away_from_zero(filter_window / 2))   # round(8/2) = 4
    pm[:half] = 0.0                                            # zero first/last filter_window/2 frames
    pm[T - half:] = 0.0

    trace = np.nanmean(pm * selection_map_patch[None], axis=(1, 2))     # spatial mean incl. zeros
    dog_filtered = apply_dog_gain(trace, dog)
    dog_filtered = forward_rolling_min_subtract(dog_filtered, look=5)
    std_noise = np.nanstd(dog_filtered[std_frames], ddof=1)             # sample std (N-1)
    threshold = threshold_factor * std_noise

    thr = trace.copy()
    thr[thr < threshold] = 0.0
    peaks, pk = findpeaks(thr)                                          # amplitudes + 0-based idx
    peak_time = pk.astype(float).copy()                                # refined below (diag)

    region_AP = []
    for f in range(len(pk)):
        t0 = int(pk[f])
        if t0 <= 2 or t0 >= T - 4:                                     # skip peaks near the ends
            continue
        shift = int(np.argmax(thr[t0 - 3:t0 + 3])) - 3                 # refine within [t0-3, t0+2]
        t0r = t0 + shift
        peak_time[f] = t0r
        frame = pm[t0r]
        pv = frame.max()
        fi = int(np.argmax(frame.ravel(order="F") == pv))             # first occurrence, column-major
        px0, py0 = np.unravel_index(fi, frame.shape, order="F")
        region_AP.append([px0 + origin0[0], py0 + origin0[1], t0r, peaks[f]])

    region_AP = np.array(region_AP, dtype=np.float64).reshape(-1, 4)
    diag = dict(patch_traces=trace, dog_filtered_patch_traces=dog_filtered,
                std_noise=std_noise, threshold=threshold, peaks=peaks, peak_time=peak_time)
    return region_AP, diag


def dedup_close_peaks(region_AP, min_interval=15):
    """Remove APs whose frame is within ``min_interval`` of another.

    Keeps only frames that are their own representative under both the 'highest' and 'lowest'
    tolerance groupings (i.e. isolated frames).
    """
    if len(region_AP) == 0:
        return region_AP
    times = region_AP[:, 2]
    keep = np.intersect1d(uniquetol_reps(times, min_interval, "highest"),
                          uniquetol_reps(times, min_interval, "lowest"))
    return region_AP[np.isin(times, keep)]


# --------------------------------------------------------------------------- #
# Center of mass via SVD + region growing
# --------------------------------------------------------------------------- #
def svd_denoise_ap_stack(patch_movie, times, rank=15):
    """Build a 7-frame (+/-3) stack per AP, low-rank denoise it, and keep the center frames.

    Parameters
    ----------
    patch_movie : (T, Hp, Wp) float array
    times : array of int
        0-indexed AP frames.
    rank : int
        Truncation rank (capped at the number of stacked frames).

    Returns
    -------
    (recon, centers)
        ``recon`` : (Hp, Wp, 7*nAP) rank-truncated reconstruction;
        ``centers`` : (Hp, Wp, nAP) denoised center frame of each AP.
    """
    Hp, Wp = patch_movie.shape[1:]
    nAP = len(times)
    stack = np.zeros((Hp, Wp, 7 * nAP))
    for i, t in enumerate(times):
        stack[:, :, 7 * i:7 * i + 7] = np.transpose(patch_movie[t - 3:t + 4], (1, 2, 0))
    X = stack.reshape(Hp * Wp, 7 * nAP, order="F")             # pixels x frames
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(rank, X.shape[1])                                  # rank capped at the number of frames
    recon = ((U[:, :k] * s[:k]) @ Vt[:k]).reshape(Hp, Wp, 7 * nAP, order="F")
    sa = recon.copy()
    for i in range(nAP):                                       # zero all but the center frame
        drop = list(range(7 * i, 7 * i + 3)) + list(range(7 * i + 4, 7 * i + 7))
        sa[:, :, drop] = 0.0
    keep = ~((sa[-1, -1, :] == 0) & (sa[9, 9, :] == 0))       # drop the zeroed frames (probe pixels)
    return recon, sa[:, :, keep]


def region_grow_brightest(current, radius=8.5, n_pixels=50):
    """Greedily grow ``n_pixels`` bright pixels outward from the max, within ``radius``.

    Parameters
    ----------
    current : (Hp, Wp) float array
    radius : float
    n_pixels : int

    Returns
    -------
    (n_pixels, 2) int array of 0-indexed (row, col)

    Notes
    -----
    Each step adds a pixel's 4-neighbors to a queue and picks the brightest queued pixel
    (first occurrence in column-major order).
    """
    Hp, Wp = current.shape
    picked = np.zeros((n_pixels, 2), int)
    mx = current.max()
    fi = int(np.argmax(current.ravel(order="F") == mx))
    picked[0] = np.unravel_index(fi, current.shape, order="F")
    queue = np.zeros((0, 2), int)
    pc = 1
    while pc < n_pixels:
        pix = picked[pc - 1]
        sc = pix + np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])     # 4-neighbors (row, col)
        d = sc - picked[0]
        sc = sc[d[:, 0] ** 2 + d[:, 1] ** 2 <= radius ** 2]          # within search radius
        sc = sc[(sc[:, 0] >= 0) & (sc[:, 0] < Hp) & (sc[:, 1] >= 0) & (sc[:, 1] < Wp)]
        queue = np.vstack([queue, sc])
        queue = queue[[i for i in range(len(queue))
                       if not (queue[i] == picked[:pc]).all(1).any()]]     # drop already-picked
        mval = current[queue[:, 0], queue[:, 1]].max()             # brightest queued pixel
        fi = int(np.argmax(current.ravel(order="F") == mval))
        picked[pc] = np.unravel_index(fi, current.shape, order="F")
        pc += 1
    return picked


def _weighted_com(current, picked, patch_rows, patch_cols):
    """Intensity^2-weighted centroid over the row x column hull of the picked pixels.

    The picked pixels define a row x column hull (via ``np.ix_``); the centroid is computed
    over it in absolute coordinates. ``patch_rows``/``patch_cols`` are 1-indexed.
    """
    ap = np.zeros(current.shape)
    ap[np.ix_(picked[:, 0], picked[:, 1])] = 1.0
    ap[ap == 0] = np.nan
    cp = current * ap
    cp[cp < 0] = np.nan
    total = np.nansum(cp * cp)
    row_c = np.nansum(cp * cp * patch_rows[:, None]) / total
    col_c = np.nansum(cp * cp * patch_cols[None, :]) / total
    return row_c, col_c


def com_via_svd(patch_movie, times, patch_rows, patch_cols, rank=15, radius=8.5, n_pixels=50):
    """Center of mass per AP -> ``[nAP, 3]`` = ``[row, col, frame]`` (0-indexed frame).

    ``patch_rows``/``patch_cols`` are the 1-indexed absolute patch coordinates.
    """
    _recon, centers = svd_denoise_ap_stack(patch_movie, times, rank)
    n = len(times)
    coms = np.zeros((n, 3))
    coms[:, 2] = times
    for h in range(n):
        picked = region_grow_brightest(centers[:, :, h], radius, n_pixels)
        coms[h, 0], coms[h, 1] = _weighted_com(centers[:, :, h], picked, patch_rows, patch_cols)
    return coms


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
    """Per-cluster footprints: mean of each cluster's AP frames, masked to an 11x11 window
    around the brightest in-selection pixel.

    ``cluster_times`` = list of arrays of 0-indexed frames per cluster. Returns
    ``(planes, centers)`` — each plane ``[H, W]``; each center ``[row, col]`` (1-indexed).
    """
    r0, c0 = patch_rows[0], patch_cols[0]                                  # 1-indexed origin
    planes, centers = [], []
    for times in cluster_times:
        raw = patch_movie[np.asarray(times, int)].mean(axis=0)            # [Hp, Wp]
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


def extract_footprints(movie, filtered_movie, regions, spatial_footprints, dog, std_frames, p,
                       height, width, verbose=False):
    """Per-region loop: AP detection -> center of mass -> DBSCAN clustering -> footprints.

    Parameters
    ----------
    movie, filtered_movie : (T, H, W) float arrays
    regions, spatial_footprints : from :func:`pyali.segmentation.cell_segmentation`
    dog : (K,) DOG kernel
    std_frames : array of int (0-indexed baseline frames)
    p : Params
    height, width : int
    verbose : bool

    Returns
    -------
    (APs, COMs, footprint, footprint_center)
        ``footprint`` : (H, W, N); ``footprint_center`` : (N, 2) = ``[row, col]``.
    """
    from sklearn.cluster import DBSCAN

    T = movie.shape[0]
    fw = p.filter_window
    half = int(round_half_away_from_zero(fw / 2))
    APs, COMs, planes, centers = [], [], [], []

    def _fallback(sel_full, sel_patch, pm_max, region):
        planes.append(region_fallback_footprint(sel_full, sel_patch, pm_max, height, width))
        cen = region_center_rowcol(region)
        centers.append(cen)

    for c, region in enumerate(regions):
        if verbose and (c % 25 == 0 or c == len(regions) - 1):
            print(f"[pyali]   region {c + 1}/{len(regions)}", flush=True)
        patch_rows, patch_cols, origin0 = compute_patch(region["Centroid"], region["BoundingBox"],
                                                        p.patch_size, height, width)
        sel_full = build_selection_map(spatial_footprints[c], height, width)
        sel_patch = sel_full[np.ix_(patch_rows - 1, patch_cols - 1)]
        pm = filtered_movie[:, patch_rows[0] - 1:patch_rows[-1], patch_cols[0] - 1:patch_cols[-1]].copy()
        pm[:half] = 0.0
        pm[T - half:] = 0.0
        pm_max = pm.max(axis=0)

        region_AP, _diag = detect_region_aps(pm, sel_patch, origin0, dog, std_frames,
                                             p.threshold_factor, fw)
        region_AP = dedup_close_peaks(region_AP, p.min_peak_interval)
        if len(region_AP) == 0:
            _fallback(sel_full, sel_patch, pm_max, region); continue
        APs.append(region_AP)

        region_COMs = com_via_svd(pm, region_AP[:, 2].astype(int), patch_rows, patch_cols,
                                  p.svd_rank, p.com_radius, p.com_n_pixels)
        COMs.append(region_COMs)
        labels = DBSCAN(eps=p.dbscan_eps, min_samples=p.dbscan_min_pts).fit_predict(region_COMs[:, :2])
        if np.all(labels == -1):
            _fallback(sel_full, sel_patch, pm_max, region); continue

        cluster_times = [region_COMs[labels == k, 2].astype(int)
                         for k in range(labels.max() + 1) if np.any(labels == k)]
        pl, ce = cluster_footprints(pm, cluster_times, sel_patch, patch_rows, patch_cols,
                                    height, width)
        planes.extend(pl); centers.extend(ce)

    footprint = np.stack(planes, axis=2) if planes else np.zeros((height, width, 0))
    footprint_center = np.array(centers, dtype=float).reshape(-1, 2)
    APs = np.vstack(APs) if APs else np.zeros((0, 4))
    COMs = np.vstack(COMs) if COMs else np.zeros((0, 3))
    return APs, COMs, footprint, footprint_center


def region_center_rowcol(region):
    """Region center as ``[row, col]`` from the region's Centroid ``[col, row]``."""
    cx, cy = float(region["Centroid"][0]), float(region["Centroid"][1])
    return [cy, cx]


# --------------------------------------------------------------------------- #
# Trace extraction
# --------------------------------------------------------------------------- #
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
    Footprint and movie are flattened with a consistent C-order pixel ordering.
    """
    T, H, W = movie.shape
    N = footprint.shape[2]
    flat_fp = footprint.reshape(H * W, N)                       # pixel = row*W + col
    rcond = max(flat_fp.shape) * np.finfo(np.float64).eps       # pseudoinverse tolerance
    pinv_fp = np.linalg.pinv(flat_fp, rcond=rcond)             # [N, H*W]
    flat_mov = movie.reshape(T, H * W).T                        # [H*W, T] view (no copy)
    return -(pinv_fp @ flat_mov)                                # [N, T]; negate small result,
    #                                                             not the full movie (saves a copy)


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
    contraction is done as ``movie_flat @ FtW.T`` to avoid materializing the large ``[H*W, T]``
    transpose.
    """
    T, H, W = movie.shape
    N = footprint.shape[2]
    flat_fp = footprint.reshape(H * W, N)                             # [H*W, N]
    w = 1.0 / (noise_map.reshape(H * W) ** 2)                         # [H*W] inverse-variance
    FtW = flat_fp.T * w[None, :]                                      # [N, H*W]
    A = FtW @ flat_fp                                                 # [N, N]
    A = A + ridge_frac * np.mean(np.diag(A)) * np.eye(N)              # ridge (relative)
    movie_flat = movie.reshape(T, H * W)                             # view, C-order
    B = -(movie_flat @ FtW.T).T                                      # [N, T] == FtW @ (-movie_flat.T)
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
