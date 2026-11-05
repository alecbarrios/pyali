"""Preprocessing: reference/correlation image, sharpening, background subtraction, motion.

All 2-D images are ``[H, W]``; movie frame stacks are ``[T, H, W]``. scipy / scikit-image
are imported lazily inside the functions that need them, so this module imports even when
those packages are absent (the reference/correlation image needs only NumPy).
"""
import numpy as np

from .utils import fspecial_laplacian, strel_disk, round_half_away_from_zero


# --------------------------------------------------------------------------- #
# Reference image + correlation image
# --------------------------------------------------------------------------- #
def reference_and_correlation_image(ref_frames, eps=None):
    """Mean image of the reference frames, weighted by a local-correlation image.

    Parameters
    ----------
    ref_frames : (n_ref, H, W) float64
        Stack of frames used to build the reference (e.g. the first ~600 frames).
    eps : float, optional
        Small value guarding zero-variance pixels (default machine epsilon).

    Returns
    -------
    (reference_image, corr_image) : (H, W) ndarray, (H, W) ndarray
        ``reference_image = mean(ref_frames) * corr``; ``corr`` is normalized to [0, 1].
    """
    if eps is None:
        eps = np.finfo(np.float64).eps
    reference_image = ref_frames.mean(axis=0)                 # brightness
    corr = _correlation_image(ref_frames, eps)                # cell specificity
    corr = np.maximum(corr, 0.0)
    corr = corr / corr.max()
    return reference_image * corr, corr


def _correlation_image(ref, eps):
    """Mean Pearson correlation of each interior pixel's time trace with its 4 orthogonal
    neighbors (vectorized). The ``+eps`` guards zero norms; border and zero-variance pixels
    stay 0."""
    n, H, W = ref.shape
    c = ref - ref.mean(axis=0, keepdims=True)                 # per-pixel centered traces
    norm = np.sqrt(np.einsum("thw,thw->hw", c, c))            # L2 norm of each trace
    total = np.zeros((H, W))
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):         # up, down, left, right
        nb = np.roll(np.roll(c, -dr, axis=1), -dc, axis=2)    # neighbor centered trace at (r+dr,c+dc)
        nnorm = np.roll(np.roll(norm, -dr, axis=0), -dc, axis=1)
        dot = np.einsum("thw,thw->hw", c, nb)
        total += dot / (norm * nnorm + eps)
    corr_full = total / 4.0
    out = np.zeros((H, W))
    interior = np.zeros((H, W), dtype=bool)
    interior[1:-1, 1:-1] = True
    interior &= (norm != 0)                                   # skip zero-variance pixels
    out[interior] = corr_full[interior]
    return out


def _correlation_image_literal(ref, eps=None):
    """Straightforward per-pixel reference implementation of :func:`_correlation_image`.

    Slow; kept as a readable equivalent of the vectorized version.
    """
    if eps is None:
        eps = np.finfo(np.float64).eps
    n, H, W = ref.shape
    corr = np.zeros((H, W))
    for r in range(1, H - 1):
        for cc in range(1, W - 1):
            t = ref[:, r, cc] - ref[:, r, cc].mean()
            npix = np.linalg.norm(t)
            if npix == 0:
                continue
            nb = np.stack([ref[:, r - 1, cc], ref[:, r + 1, cc],
                           ref[:, r, cc - 1], ref[:, r, cc + 1]], axis=1)
            nb = nb - nb.mean(axis=0, keepdims=True)
            rvals = (t @ nb) / (npix * np.linalg.norm(nb, axis=0) + eps)
            corr[r, cc] = rvals.mean()
    return corr


# --------------------------------------------------------------------------- #
# Sharpening chain
# --------------------------------------------------------------------------- #
def gaussian_kernel(sigma):
    """Normalized 2-D Gaussian kernel of radius ``ceil(2*sigma)`` (size ``2*ceil(2*sigma)+1``).

    Parameters
    ----------
    sigma : float

    Returns
    -------
    (k, k) ndarray of float64 summing to 1
    """
    rad = int(np.ceil(2 * sigma))
    ax = np.arange(-rad, rad + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def sharpen(reference_image, disk_radius=15, sigma=1.75, lap_alpha=0.2, k=2.25):
    """Sharpen the reference image via a Laplacian-of-Gaussian unsharp mask.

    Chain: morphological-opening background ``bkg`` -> ``I2 = image - bkg`` -> Gaussian blur ->
    Laplacian -> normalized LoG -> ``sharpened = I2 - k * LoG_normalized``.

    Parameters
    ----------
    reference_image : (H, W) float array
    disk_radius : int
        Radius of the opening structuring element.
    sigma : float
        Gaussian blur width.
    lap_alpha : float
        Laplacian shape parameter.
    k : float
        Sharpening strength.

    Returns
    -------
    (bkg, I2, blurred, h_lap, LoG_image, LoG_normalized, sharpened) : tuple of (H, W) arrays
        (``h_lap`` is the 3x3 Laplacian kernel.)
    """
    from scipy import ndimage

    se = strel_disk(disk_radius)                                    # octagonal disk
    bkg = ndimage.grey_opening(reference_image, footprint=se, mode="reflect")
    I2 = reference_image - bkg
    gk = gaussian_kernel(sigma)
    blurred = ndimage.correlate(I2, gk, mode="nearest")             # Gaussian blur, replicate edges
    h_lap = fspecial_laplacian(lap_alpha)
    LoG_image = ndimage.correlate(blurred, h_lap, mode="nearest")   # Laplacian, replicate edges
    LoG_normalized = LoG_image / np.max(np.abs(LoG_image)) * np.std(I2, ddof=1)  # sample std (N-1)
    sharpened = I2 - k * LoG_normalized
    return bkg, I2, blurred, h_lap, LoG_image, LoG_normalized, sharpened


# --------------------------------------------------------------------------- #
# Adaptive background subtraction
# --------------------------------------------------------------------------- #
def background_samples(movie, ranges, disk_radius=15):
    """Morphological background estimate for each specified frame range.

    Parameters
    ----------
    movie : (T, H, W) float array
    ranges : list of (start, end)
        1-indexed inclusive frame ranges, e.g. ``[(1, 750), (4100, 4200), (6100, 6350)]``.
    disk_radius : int

    Returns
    -------
    (samples, anchors) : (n, H, W) ndarray, (n,) ndarray
        ``samples[i]`` = opening of the temporal mean over range ``i``;
        ``anchors[i]`` = round(mean(range i)) as a 1-indexed frame number.
    """
    from scipy import ndimage
    se = strel_disk(disk_radius)
    n = len(ranges)
    H, W = movie.shape[1], movie.shape[2]
    samples = np.zeros((n, H, W))
    anchors = np.zeros(n)
    for i, (a, b) in enumerate(ranges):
        local_avg = movie[a - 1:b].mean(axis=0)               # 1-indexed inclusive -> python slice
        samples[i] = ndimage.grey_opening(local_avg, footprint=se, mode="reflect")
        anchors[i] = round_half_away_from_zero((a + b) / 2.0)  # center frame (1-indexed)
    return samples, anchors


def _interp_weight(frame_1idx, anchors):
    """Linear-interpolation bracket for a 1-indexed frame: ``(idx_before, idx_after, w)``.

    Frames before the first anchor or after the last are clamped (``w=0``).
    """
    f = frame_1idx
    if f <= anchors[0]:
        return 0, 0, 0.0
    if f >= anchors[-1]:
        return len(anchors) - 1, len(anchors) - 1, 0.0
    ia = next(k for k in range(len(anchors)) if anchors[k] > f)   # first anchor > f
    ib = ia - 1
    w = (f - anchors[ib]) / (anchors[ia] - anchors[ib])
    return ib, ia, w


def background_pixel_trace(samples, anchors, n_frames, r, c):
    """Interpolated background over time at pixel ``(r, c)`` (0-indexed) -> ``(n_frames,)``.

    Computed without materializing the full 3-D interpolated background.
    """
    vals = samples[:, r, c]
    out = np.empty(n_frames)
    for i in range(n_frames):
        ib, ia, w = _interp_weight(i + 1, anchors)
        out[i] = (1.0 - w) * vals[ib] + w * vals[ia]
    return out


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


# --------------------------------------------------------------------------- #
# Rigid motion correction
# --------------------------------------------------------------------------- #
def _edge_gradients(f0):
    """Central-difference spatial gradients of ``f0`` [H,W], edge-replicated to full size."""
    dfdx = (f0[:, 2:] - f0[:, :-2]) / 2.0                     # d/dcol
    dfdy = (f0[2:, :] - f0[:-2, :]) / 2.0                     # d/drow
    dfdx = np.column_stack([dfdx[:, [0]], dfdx, dfdx[:, [-1]]])   # replicate first/last col -> [H,W]
    dfdy = np.vstack([dfdy[[0], :], dfdy, dfdy[[-1], :]])         # replicate first/last row -> [H,W]
    return dfdx, dfdy


def motion_operator(f0):
    """Least-squares design matrix ``A = [dfdx(:), dfdy(:)]`` -> ``[H*W, 2]``.

    Uses C-order ravel, consistent with ``movie[t].ravel()``.
    """
    dfdx, dfdy = _edge_gradients(f0)
    return np.column_stack([dfdx.ravel(), dfdy.ravel()])


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
    return movie
