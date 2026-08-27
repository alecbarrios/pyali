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


def drop_oversized_regions(regions, binary_map, max_bbox_frac):
    """Drop regions whose **bounding box** covers more than ``max_bbox_frac`` of the frame.

    The earliest and latest bursts of a well image partially outside it, leaving a band of
    un-illuminated inter-well area. Those dark pixels share the same absence of signal, so they
    correlate with one another and segmentation merges them into one enormous object — up to 84%
    of the frame. Such an object is never a cell, and because :func:`pyali.extract.compute_patch`
    sizes each region's patch from its bounding box, one of them alone can hold ~24 GB in
    :func:`pyali.extract.detect_region_aps`.

    The cut is on bounding-box area, not pixel area: these artifacts are only ~1-2% of the frame
    by pixel count but span most of it, so an area-based threshold misses them entirely. It is
    also not ``area/bbox_area`` (extent), which describes shape and cannot bound the allocation.

    Parameters
    ----------
    regions : list of dict
        From :func:`cell_segmentation`; ``BoundingBox`` is ``[x_ul, y_ul, w, h]``.
    binary_map : (H, W) bool ndarray
    max_bbox_frac : float or None
        Fraction of the frame above which a region is rejected. ``None``/0 disables the guard.

    Returns
    -------
    (kept, binary_map, dropped)
        ``binary_map`` has the dropped regions' pixels cleared, so the mask matches what was
        actually analyzed. It is copied only when something is dropped.
    """
    if not max_bbox_frac:
        return regions, binary_map, []
    height, width = binary_map.shape
    limit = float(max_bbox_frac) * height * width
    kept, dropped = [], []
    for r in regions:
        _x_ul, _y_ul, w, h = (float(v) for v in r["BoundingBox"])
        (dropped if w * h > limit else kept).append(r)
    if dropped:
        binary_map = binary_map.copy()
        for r in dropped:
            px = r["PixelList"].astype(int)                 # [col, row], 1-indexed
            binary_map[px[:, 1] - 1, px[:, 0] - 1] = False
    return kept, binary_map, dropped


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
