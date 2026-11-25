"""Preprocessing: reference/correlation image, large-scale background, time-varying background
subtraction, motion correction.

All 2-D images are ``[H, W]``; movie frame stacks are ``[T, H, W]``. scipy / scikit-image
are imported lazily inside the functions that need them, so this module imports even when
those packages are absent (the reference/correlation image needs only NumPy).
"""
import numpy as np

from .utils import strel_disk, round_half_away_from_zero


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


# --------------------------------------------------------------------------- #
# MATLAB-style Gaussian (imgaussfilt): kernel 2*ceil(2*sigma)+1, replicate padding, separable
# --------------------------------------------------------------------------- #
def gaussian_matlab(image, sigma):
    """``imgaussfilt(image, sigma)``: separable Gaussian with a ``2*ceil(2*sigma)+1`` kernel,
    normalized to sum 1, replicate ("nearest") padding. Float64 output."""
    from scipy import ndimage
    half = int(np.ceil(2.0 * float(sigma)))
    x = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-(x * x) / (2.0 * float(sigma) ** 2))
    g /= g.sum()
    out = ndimage.correlate1d(np.asarray(image, np.float64), g, axis=0, mode="nearest")
    return ndimage.correlate1d(out, g, axis=1, mode="nearest")


# --------------------------------------------------------------------------- #
# Large-scale, mask-aware background (the Mini ALI bgMap) and the raw mean reference
# --------------------------------------------------------------------------- #
def masked_grey_opening(image, footprint, mask):
    """Flat grey opening in which only ``mask`` pixels contribute: the erosion sees masked-out
    pixels as +inf (they can never be a local minimum) and the dilation sees them as -inf.
    Masked-out output pixels are NaN. Any non-finite result inside the mask falls back to the
    input value."""
    from scipy import ndimage
    img = np.asarray(image, np.float64)
    mask = np.asarray(mask, bool)
    out = np.full(img.shape, np.nan)
    if not mask.any():
        return out
    eroded = ndimage.grey_erosion(np.where(mask, img, np.inf), footprint=footprint, mode="nearest")
    dilated = ndimage.grey_dilation(np.where(mask, eroded, -np.inf), footprint=footprint, mode="nearest")
    out[mask] = dilated[mask]
    bad = mask & ~np.isfinite(out)
    out[bad] = img[bad]
    return out


def large_scale_background(image, radius=50, sigma=40.0, mask=None, min_weight=1e-6):
    """Illumination-scale background of a reference image: mask-aware grey opening with an
    exact disk of ``radius`` px, then a normalized Gaussian of ``sigma`` px
    (``G(bg * mask) / max(G(mask), min_weight)``, so only valid pixels are averaged).
    NaN outside ``mask``. ``mask=None`` means every finite pixel. Float32 output.

    The opening removes everything narrower than the disk (cells), keeping only the slowly
    varying illumination; the Gaussian smooths the opening's blockiness. The exact-disk opening
    costs ~15 s on an 800x800 image."""
    from .utils import exact_disk
    img = np.asarray(image, np.float64)
    valid = np.isfinite(img) if mask is None else (np.asarray(mask, bool) & np.isfinite(img))
    opened = masked_grey_opening(img, exact_disk(radius), valid)
    if sigma and sigma > 0:
        num = gaussian_matlab(np.where(valid, opened, 0.0), sigma)
        den = gaussian_matlab(valid.astype(np.float64), sigma)
        bg = num / np.maximum(den, float(min_weight))
    else:
        bg = opened
    bg = np.where(valid, bg, np.nan)
    return bg.astype(np.float32)


def raw_mean_reference(movie, n_frames=300):
    """Mean of the first ``n_frames`` frames (clipped to ``T``) of the **raw** movie, in
    acquisition units; float64 ``(H, W)``. ``movie`` is not modified."""
    T = movie.shape[0]
    return np.asarray(movie[:min(int(n_frames), T)]).mean(axis=0, dtype=np.float64)


def focus_score(image, sigma=1.0):
    """Focus QC for a mean image: variance of the Laplacian of the Gaussian(``sigma``)-smoothed
    image, divided by the squared mean intensity so it is independent of exposure. Sharp
    800x800 fields of this preparation score ~1e-3 to 3.5e-3; a defocused field scores ~1e-4 or
    less (P-4 W-A2 burst45: 9e-5, with its sharp neighbours at 1.7e-3 to 3.5e-3). Compare
    within an acquisition; the absolute scale depends on magnification and cell density."""
    from scipy import ndimage
    img = np.asarray(image, np.float64)
    if img.size == 0 or not np.isfinite(img).all():
        return float("nan")
    lap = ndimage.laplace(ndimage.gaussian_filter(img, float(sigma)))
    return float(lap.var() / max(img.mean(), 1e-9) ** 2)

