"""Cell segmentation.

Pipeline: normalize to [0, 1] -> Gaussian smoothing -> adaptive mean threshold -> binarize
-> remove small objects -> connected-component region properties. scikit-image / scipy are
imported lazily so the module imports without them.

Region properties are returned as: Centroid and PixelList as ``[col, row]`` (1-indexed);
BoundingBox as ``[x_ul, y_ul, w, h]``.
"""
import numpy as np

from .preprocess import gaussian_kernel


def adaptive_threshold_mean(image, sensitivity, neighborhood=None, multiplier=None):
    """Locally adaptive threshold surface based on the neighborhood mean.

    Computes the local mean over a ``2*floor(size/16)+1`` neighborhood (replicate padding),
    then scales it: ``T = local_mean * scale(sensitivity)``.

    Parameters
    ----------
    image : (H, W) float array
    sensitivity : float in [0, 1)
        MATLAB-style sensitivity; see :func:`_threshold_scale`.
    neighborhood : (int, int), optional
        Local-mean window; defaults to ``2*floor(size/16)+1`` per axis. This sets the spatial
        scale of "local": a window much larger than a cell makes the threshold nearly global,
        which merges neighbouring cells.
    multiplier : float, optional
        Use this multiplier directly and ignore ``sensitivity``.

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
    m = _threshold_scale(sensitivity) if multiplier is None else float(multiplier)
    return local_mean * m


def _threshold_scale(sensitivity):
    """Multiplier applied to the local mean, from a sensitivity in [0, 1).

    Higher sensitivity => lower threshold => more foreground, matching MATLAB ``adaptthresh``.
    Calibrated so the pipeline's historical sensitivity of 0.10 returns **exactly 1.5** — the
    value the previous implementation returned for every input, so defaults are unchanged.

    Raising the multiplier is the knob that separates touching cells: a higher bar shrinks each
    blob and pulls clumps apart at the necks. Lowering it fuses them.
    """
    s = float(np.clip(sensitivity, 0.0, 0.99))
    return 1.5 * (1.0 - s) / 0.9


def cell_segmentation(image, threshold=0.90, gauss_size=0.1, region_size=10,
                      threshold_mult=None, neighborhood=None):
    """Segment cells from a grayscale image.

    Parameters
    ----------
    image : (H, W) float array
        Sharpened reference image.
    threshold : float
        Brightness percentile (higher keeps fewer/brighter pixels); the adaptive-threshold
        sensitivity is ``1 - threshold``. The default 0.90 gives a multiplier of 1.5.
    gauss_size : float
        Gaussian smoothing width. NB: at the default 0.1 the 3x3 kernel is a delta function
        (centre 1.0, off-centre ~1e-22), i.e. no smoothing. *Increasing* it blurs cells
        together, so it is not a knob for splitting clumps.
    region_size : int
        Minimum connected-component size (pixels) to keep.
    threshold_mult : float, optional
        Local-mean multiplier, bypassing ``threshold``. Higher separates touching cells.
    neighborhood : (int, int), optional
        Local-mean window for the adaptive threshold.

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
    T = adaptive_threshold_mean(I_filt, 1.0 - threshold, neighborhood,   # adaptive mean threshold
                                multiplier=threshold_mult)
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
