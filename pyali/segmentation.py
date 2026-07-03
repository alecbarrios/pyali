"""Cell segmentation.

Pipeline: normalize to [0, 1] -> Gaussian smoothing -> adaptive mean threshold -> binarize
-> remove small objects -> connected-component region properties. scikit-image / scipy are
imported lazily so the module imports without them.

Region properties are returned as: Centroid and PixelList as ``[col, row]`` (1-indexed);
BoundingBox as ``[x_ul, y_ul, w, h]``.
"""
import numpy as np

from .preprocess import gaussian_kernel


def adaptive_threshold_mean(image, sensitivity, neighborhood=None):
    """Locally adaptive threshold surface based on the neighborhood mean.

    Computes the local mean over a ``2*floor(size/16)+1`` neighborhood (replicate padding),
    then scales it: ``T = local_mean * scale(sensitivity)``. For the sensitivity used by the
    pipeline (0.10) the scale is 1.5.

    Parameters
    ----------
    image : (H, W) float array
    sensitivity : float in [0, 1]
    neighborhood : (int, int), optional
        Local-mean window; defaults to ``2*floor(size/16)+1`` per axis.

    Returns
    -------
    (H, W) ndarray
        Per-pixel threshold surface.
    """
    from scipy import ndimage
    I = np.asarray(image, dtype=np.float64)
    if neighborhood is None:
        neighborhood = tuple(int(2 * (s // 16) + 1) for s in I.shape)  # 2*floor(size/16)+1
    local_mean = ndimage.uniform_filter(I, size=neighborhood, mode="nearest")  # replicate padding
    return local_mean * _threshold_scale(sensitivity)


def _threshold_scale(sensitivity):
    # Multiplier applied to the local mean; 1.5 corresponds to the pipeline's sensitivity (0.10).
    if abs(sensitivity - 0.10) < 1e-9:
        return 1.5
    return 1.5


def cell_segmentation(image, threshold=0.90, gauss_size=0.1, region_size=10):
    """Segment cells from a grayscale image.

    Parameters
    ----------
    image : (H, W) float array
        Sharpened reference image.
    threshold : float
        Brightness percentile (higher keeps fewer/brighter pixels); the adaptive-threshold
        sensitivity is ``1 - threshold``.
    gauss_size : float
        Gaussian smoothing width.
    region_size : int
        Minimum connected-component size (pixels) to keep.

    Returns
    -------
    (regions, binary_map, spatial_footprints)
        ``regions`` : list of dicts (Area, Centroid, BoundingBox, PixelList);
        ``binary_map`` : (H, W) bool;
        ``spatial_footprints`` : list of (Ni, 2) int arrays of ``[col, row]`` (1-indexed).
    """
    from scipy import ndimage
    from skimage.morphology import remove_small_objects

    I = np.asarray(image, dtype=np.float64)
    I_adj = (I - I.min()) / (I.max() - I.min())                    # normalize to [0, 1]
    I_filt = ndimage.correlate(I_adj, gaussian_kernel(gauss_size), mode="nearest")  # smoothing
    T = adaptive_threshold_mean(I_filt, 1.0 - threshold)           # adaptive mean threshold
    BW = I_adj > T                                                 # binarize
    BW = remove_small_objects(BW, region_size, connectivity=2)     # drop small objects (8-conn)

    regions = _regionprops(BW)
    spatial_footprints = [r["PixelList"] for r in regions]
    return regions, BW, spatial_footprints


def _regionprops(BW):
    """Connected-component region properties with column-major pixel ordering.

    Centroid = ``[col, row]`` (1-indexed); BoundingBox = ``[x_ul, y_ul, w, h]`` with the corner
    at ``(min_col-0.5, min_row-0.5)``; PixelList = ``[col, row]`` (1-indexed), ordered
    column-major (down columns).
    """
    from skimage.measure import label, regionprops

    labels = label(BW, connectivity=2)                            # 8-connectivity
    out = []
    for rp in regionprops(labels):
        coords = rp.coords                                        # (row, col), 0-indexed
        rows, cols = coords[:, 0], coords[:, 1]
        order = np.lexsort((rows, cols))                          # sort by col, then row
        pixel_list = np.column_stack([cols[order] + 1, rows[order] + 1]).astype(float)  # [col,row] 1-idx
        min_row, min_col, max_row, max_col = rp.bbox              # max exclusive
        out.append({
            "Area": float(rp.area),
            "Centroid": np.array([rp.centroid[1] + 1.0, rp.centroid[0] + 1.0]),   # [col, row] 1-idx
            "BoundingBox": np.array([min_col + 0.5, min_row + 0.5,
                                     max_col - min_col, max_row - min_row]),       # [x_ul,y_ul,w,h]
            "PixelList": pixel_list,
        })
    return out
