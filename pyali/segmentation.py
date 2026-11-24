"""Cell segmentation (Mini ALI V2.2 "better segmentation", reproduced faithfully).

Two masks, ANDed:

* :func:`hysteresis_mask` -- *what is signal*: on the background-corrected raw-mean reference,
  pixels >= background + 3 sigma that are 8-connected to a seed >= background + 6 sigma, where
  background and sigma are measured per field (histogram mode, left-side MAD). Calibrated to the
  field's noise, so an empty field yields nothing.
* :func:`soma_watershed_mask` -- *how many somata, cut where*: the top 5% of a soma-scale
  top-hat image, partitioned by a watershed from its local peaks with the ridge lines removed.

Then one minimum-area filter and connected-component region properties. scikit-image / scipy are
imported lazily so the module imports without them.

Region properties are returned as: Centroid and PixelList as ``[col, row]`` (1-indexed);
BoundingBox as ``[x_ul, y_ul, w, h]``.
"""
import numpy as np

from .preprocess import gaussian_matlab

_EIGHT = np.ones((3, 3), bool)                  # 8-connectivity structuring element


# --------------------------------------------------------------------------- #
# Noise-calibrated hysteresis mask (Mini ALI mini_ali_brightness_threshold)
# --------------------------------------------------------------------------- #
def hysteresis_mask(ref_corrected, mask=None, smooth_sigma=2.0, high_sigma=6.0, low_sigma=3.0,
                    min_area=15):
    """Foreground = pixels at least ``low_sigma`` background noise units above the background
    that are 8-connected to a seed pixel at least ``high_sigma`` units above it.

    ``ref_corrected`` is the background-corrected mean reference (raw units); negatives are
    clipped to 0 and the image is smoothed with ``smooth_sigma`` first. The background level is
    the mode of a 256-bin histogram over the valid pixels; the noise is 1.4826 x the median
    absolute deviation of the pixels at or below the mode (bright cells only add a right tail),
    falling back to the MAD of all pixels when fewer than 21 pixels lie left of the mode.

    Returns ``(BW, info)`` with ``info`` = ``bg_center, sigma_bg, th_high, th_low, n_seed, n_low``.
    """
    from skimage.morphology import remove_small_objects
    from scipy import ndimage

    I = np.maximum(np.asarray(ref_corrected, np.float64), 0.0)
    Iseg = gaussian_matlab(I, smooth_sigma).astype(np.float32)
    valid = np.isfinite(Iseg) if mask is None else (np.asarray(mask, bool) & np.isfinite(Iseg))
    x = Iseg[valid]
    info = dict(bg_center=float("nan"), sigma_bg=float("nan"), th_high=float("nan"),
                th_low=float("nan"), n_seed=0, n_low=0)
    if x.size == 0:
        return np.zeros(Iseg.shape, bool), info

    counts, edges = np.histogram(x, bins=256)
    i = int(np.argmax(counts))
    bg = 0.5 * (edges[i] + edges[i + 1])
    left = x[x <= bg]
    if left.size > 20:
        sigma = 1.4826 * float(np.median(np.abs(left - bg)))
    else:
        sigma = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    sigma = max(sigma, float(np.finfo(np.float32).eps))
    th_high = bg + high_sigma * sigma
    th_low = bg + low_sigma * sigma

    seed = valid & (Iseg >= th_high)
    low = valid & (Iseg >= th_low)
    labels, _n = ndimage.label(low, structure=_EIGHT)          # imreconstruct(seed, low): keep
    keep = np.unique(labels[seed])                             # low components holding a seed
    keep = keep[keep > 0]
    BW = np.isin(labels, keep) if keep.size else np.zeros(Iseg.shape, bool)
    BW = remove_small_objects(BW, int(min_area), connectivity=2)
    info.update(bg_center=float(bg), sigma_bg=float(sigma), th_high=float(th_high),
                th_low=float(th_low), n_seed=int(seed.sum()), n_low=int(low.sum()))
    return BW, info


# --------------------------------------------------------------------------- #
# Soma watershed mask (Mini ALI main script, "watershed of neuron soma")
# --------------------------------------------------------------------------- #
def soma_watershed_mask(ref_raw, tophat_radius=3, background_radius=50, bg_sigma=20.0,
                        percentile=95.0, close_radius=2, h=0.03):
    """Soma-scale mask split into one basin per local peak.

    On the raw mean reference: Gaussian sigma 1; top-hat with the MATLAB ``N=4`` octagon of
    ``tophat_radius`` (keeps structures narrower than the element, i.e. somata); minus the
    Gaussian(``bg_sigma``) of its ``background_radius`` octagon opening; min-max scaled; the top
    ``100 - percentile`` percent of pixels (MATLAB ``prctile``), area-open 8, closed with the
    exact ``close_radius`` disk. Markers are the h-maxima (``h``) over the whole image; the
    watershed floods from every marker over a flat surface across the whole frame (the original
    ``imimposemin`` with -inf collapses the distance transform, so split lines fall midway
    between markers), ridge lines are removed, pixels outside the mask are cleared, area-open 2.
    """
    from scipy import ndimage
    from skimage.morphology import h_maxima, remove_small_objects
    from skimage.segmentation import watershed
    from .utils import matlab_disk

    I = gaussian_matlab(np.asarray(ref_raw, np.float64), 1.0)
    I_soma = I - ndimage.grey_opening(I, footprint=matlab_disk(tophat_radius), mode="nearest")
    I_bg = gaussian_matlab(ndimage.grey_opening(I_soma, footprint=matlab_disk(background_radius),
                                                mode="nearest"), bg_sigma)
    I_corr = I_soma - I_bg
    span = float(I_corr.max() - I_corr.min())
    I_corr = (I_corr - I_corr.min()) / span if span > 0 else np.zeros_like(I_corr)   # mat2gray
    if span == 0:
        return np.zeros(I_corr.shape, bool)
    th = float(np.percentile(I_corr, percentile, method="hazen"))                     # MATLAB prctile
    BW = I_corr > th
    BW = remove_small_objects(BW, 8, connectivity=2)
    BW = ndimage.binary_closing(BW, structure=matlab_disk(close_radius))
    markers = h_maxima(I_corr, float(h), footprint=_EIGHT)                             # imextendedmax
    marker_labels, n_markers = ndimage.label(markers, structure=_EIGHT)
    if n_markers == 0:
        return np.zeros(I_corr.shape, bool)
    L = watershed(np.zeros(I_corr.shape, np.float64), markers=marker_labels, connectivity=2,
                  watershed_line=True)                                                 # flat flood, all markers
    L[~BW] = 0
    return remove_small_objects(L > 0, 2, connectivity=2)


# --------------------------------------------------------------------------- #
# Combined segmentation
# --------------------------------------------------------------------------- #
def cell_segmentation(ref_raw, ref_corrected, p, mask=None):
    """Segment cells: ``hysteresis_mask(ref_corrected) & soma_watershed_mask(ref_raw)``, then one
    minimum-area filter (``p.seg_min_area``; the original has none and keeps 1-px fragments).

    Parameters
    ----------
    ref_raw : (H, W) float array
        Mean of the first frames of the raw movie (:func:`pyali.preprocess.raw_mean_reference`).
    ref_corrected : (H, W) float array
        ``ref_raw`` minus its large-scale background (:func:`pyali.preprocess.large_scale_background`).
    p : Params
        ``seg_smooth_sigma, seg_high_sigma, seg_low_sigma, seg_min_area, ws_tophat_radius,
        background_radius, ws_bg_sigma, ws_percentile, ws_close_radius, ws_h``.
    mask : (H, W) bool, optional
        Valid-pixel mask (a well mask); ``None`` = the whole frame.

    Returns
    -------
    (regions, binary_map, spatial_footprints, info)
        ``regions`` : list of dicts (Area, Centroid, BoundingBox, PixelList);
        ``binary_map`` : (H, W) bool;
        ``spatial_footprints`` : list of (Ni, 2) arrays of ``[col, row]`` (1-indexed);
        ``info`` : the hysteresis calibration plus the pixel count of each mask.
    """
    from skimage.morphology import remove_small_objects

    BW_int, info = hysteresis_mask(ref_corrected, mask, p.seg_smooth_sigma, p.seg_high_sigma,
                                   p.seg_low_sigma, p.seg_min_area)
    BW_ws = soma_watershed_mask(ref_raw, p.ws_tophat_radius, p.background_radius, p.ws_bg_sigma,
                                p.ws_percentile, p.ws_close_radius, p.ws_h)
    BW = remove_small_objects(BW_int & BW_ws, int(p.seg_min_area), connectivity=2)
    regions = _regionprops(BW)
    info.update(n_px_intensity=int(BW_int.sum()), n_px_watershed=int(BW_ws.sum()),
                n_px_final=int(BW.sum()))
    return regions, BW, [r["PixelList"] for r in regions], info


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
