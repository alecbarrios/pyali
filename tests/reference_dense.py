"""Pre-E1 dense reference implementations, copied verbatim from commit e8acf8e, used as oracles.

* ``adaptive_background`` / ``motion_correct``: the in-place float corrections of the full movie
  (``pyali/preprocess.py``), against which :class:`pyali.movie.CorrectedMovie` is checked.
* ``temporal_filter``: the global moving-median high-pass (``pyali/extract.py``).
* ``project_movie`` / ``pinv_traces`` / ``per_pixel_noise_map`` / ``footprint_isolated_mask`` /
  ``whitened_gls_traces`` / ``extract_cell_traces``: the dense ``(H, W, N)`` trace solve
  (``pyali/extract.py``), against which :mod:`pyali.traces` is checked.

Nothing here is used by the pipeline.
"""
import numpy as np

from pyali.preprocess import _interp_weight, background_samples, motion_operator
from pyali.utils import movmedian_time


def adaptive_background(movie, ranges, disk_radius=15):
    """Subtract a per-frame, linearly-interpolated morphological background, in place.

    Parameters
    ----------
    movie : (T, H, W) float array (modified in place)
    ranges : list of (start, end)
        1-indexed inclusive background frame ranges.
    disk_radius : int

    Returns
    -------
    (movie, samples, anchors)

    Notes
    -----
    The interpolated background is applied frame by frame (never materialized as a full
    ``[T, H, W]`` array) to keep memory low.
    """
    samples, anchors = background_samples(movie, ranges, disk_radius)
    T = movie.shape[0]
    for i in range(T):
        ib, ia, w = _interp_weight(i + 1, anchors)
        bkg_f = samples[ib] if ib == ia else (1.0 - w) * samples[ib] + w * samples[ia]
        movie[i] -= bkg_f
    return movie, samples, anchors



def motion_correct(movie, truncate_last=10):
    """Estimate and remove a rigid 2-parameter (x, y) shift per frame, in place.

    A reference image ``f0`` (mean of all but the last ``truncate_last`` frames) defines the
    two spatial-gradient regressors; each frame's shift is the least-squares fit and the
    reconstructed motion is subtracted. Solved per frame with the pseudoinverse of ``A``,
    streaming over frames so the full movie is never copied.

    Parameters
    ----------
    movie : (T, H, W) float array (modified in place)
    truncate_last : int

    Returns
    -------
    (movie, shift, A, f0) : with ``shift`` shape ``(2, T)``
    """
    T, H, W = movie.shape
    f0 = movie[:T - truncate_last].mean(axis=0)               # reference from all but last frames
    A = motion_operator(f0)                                   # [H*W, 2]
    pinvA = np.linalg.pinv(A)                                 # [2, H*W]
    shift = np.zeros((2, T))
    for t in range(T):
        b = movie[t].ravel()
        s = pinvA @ b                                         # least-squares [x; y] shift
        shift[:, t] = s
        movie[t] = (b - A @ s).reshape(H, W)                 # remove reconstructed motion
    return movie, shift, A, f0


def clip_saturation(movie, threshold=25000.0):
    """Zero out pixels above ``threshold`` (saturated), in place."""
    movie[movie > threshold] = 0.0


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
