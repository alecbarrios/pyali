"""Spike assignment across the whole field: which localized spikes are the same event, and which
cell each spike belongs to (Mini ALI V2.2 ``mini_ali_mergeCloseSpikesKeepMax`` and
``mini_ali_ALI_clustering``).

A cell's spikes are found region by region, so one action potential can be detected by two
neighbouring regions in the same frame; and cells packed into one segmentation region cannot be
told apart spatially in the reference image, only by where (to sub-pixel precision) and when they
fire. The two steps here resolve both on the full set of spike centres of mass (COMs):

* :func:`merge_close_spikes_keep_max` -- spikes in the same frame whose COMs are connected by
  links of at most ``dist`` px are one event; the one with the largest coarse dF survives.
* :func:`ali_clustering` -- a spike-density ("ALI") map at ``factor`` x the pixel resolution,
  Gaussian-smoothed, thresholded; its regional maxima seed clusters and a flat-surface watershed
  splits the field between them (the original passes ``-Inf`` background to ``imimposemin``,
  which collapses the surface to the markers alone); each spike takes the label of its bin,
  0 when it lies below threshold or on a watershed line.

Coordinates are 1-indexed ``[row, col]`` as everywhere in pyali's spike tables.
"""
import numpy as np

from .preprocess import gaussian_matlab
from .utils import round_half_away_from_zero


# --------------------------------------------------------------------------- #
# Same-frame merge
# --------------------------------------------------------------------------- #
def merge_close_spikes_keep_max(spikes, dist=8.0):
    """Merge same-frame spikes whose COMs are within ``dist`` px (single linkage), keeping the
    largest ``peak_df``.

    Parameters
    ----------
    spikes : (n, 4) array ``[row, col, frame, peak_df]``
    dist : float
        Link distance in px (inclusive, as ``rangesearch``).

    Returns
    -------
    (survivor, kept_order)
        ``survivor`` (n,) int: index of the spike each spike merged into (itself if it survived);
        ``kept_order`` (k,) int: the surviving spikes in the original's output order -- by frame
        (stable), then by component in order of its first member.

    Notes
    -----
    Follows the original exactly: a stable sort by frame; within a frame, connected components
    of the ``<= dist`` neighbour graph numbered by their first member (the union-find shim's
    ``parent(max) = min`` roots); the first maximum of ``peak_df`` in member order wins.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    S = np.asarray(spikes, dtype=np.float64).reshape(-1, 4)
    n = len(S)
    survivor = np.arange(n)
    if n == 0:
        return survivor, np.zeros(0, dtype=int)
    order = np.argsort(S[:, 2], kind="stable")
    t = S[order, 2]
    starts = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    ends = np.r_[starts[1:], n]
    kept = []
    for a, b in zip(starts, ends):
        idx = order[a:b]
        m = b - a
        if m == 1:
            kept.append(int(idx[0]))
            continue
        pairs = cKDTree(S[idx, :2]).query_pairs(r=float(dist), output_type="ndarray")
        if len(pairs) == 0:
            kept.extend(int(i) for i in idx)
            continue
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
        _, comp = connected_components(g, directed=False)
        first = np.full(comp.max() + 1, m)
        np.minimum.at(first, comp, np.arange(m))          # each component's first member
        for c in np.argsort(first, kind="stable"):        # components in first-member order
            members = np.flatnonzero(comp == c)
            vals = S[idx[members], 3]
            best = members[int(np.nanargmax(vals))] if np.any(np.isfinite(vals)) else members[0]
            survivor[idx[members]] = idx[best]
            kept.append(int(idx[best]))
    return survivor, np.asarray(kept, dtype=int)


# --------------------------------------------------------------------------- #
# Spike-density clustering
# --------------------------------------------------------------------------- #
def density_map(rc, shape, factor=4):
    """Spike counts on a grid ``factor`` x finer than the pixels (Mini ALI
    ``mini_ali_density_map``): bin edges ``linspace(0.5, n + 0.5, n * factor + 1)`` per axis,
    bin ``k`` holding ``edge_k <= x < edge_{k+1}``. Returns ``(counts, (row_centres, col_centres))``.
    """
    nrow, ncol = int(shape[0]), int(shape[1])
    e1 = np.linspace(0.5, nrow + 0.5, nrow * int(factor) + 1)
    e2 = np.linspace(0.5, ncol + 0.5, ncol * int(factor) + 1)
    rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
    # hist3 puts a value equal to the last edge into an overflow bin the original drops
    keep = (rc[:, 0] < e1[-1]) & (rc[:, 1] < e2[-1])
    cnt, _, _ = np.histogram2d(rc[keep, 0], rc[keep, 1], bins=[e1, e2])
    return cnt, ((e1[:-1] + e1[1:]) / 2.0, (e2[:-1] + e2[1:]) / 2.0)


def ali_clustering(rc, factor=4, nrow=800, ncol=800, threshold=0.15, flood_margin=16):
    """Cluster spike COMs on the whole field by spike density (Mini ALI ``mini_ali_ALI_clustering``).

    Parameters
    ----------
    rc : (m, 2) array of 1-indexed ``[row, col]`` spike COMs
    factor : int
        Density-map bins per pixel along each axis.
    nrow, ncol : int
        Frame size.
    threshold : float
        Smoothed density below which no cluster exists.
    flood_margin : int or None
        The watershed floods only bins within this many bins of the above-threshold area (the
        labels outside it are cleared anyway). ``None`` floods the whole grid as the original;
        16 bins (4 px) gives the same spike labels on the reference fields in 3 s instead of 18 s.

    Returns
    -------
    (labels, seeds_rc, alimap, label_map)
        ``labels`` (m,) int: cluster label of each spike, 0 = unclustered; ``seeds_rc`` (s, 2):
        bin-centre coordinates of the regional-maximum pixels (the original's ``clust_cen``);
        ``alimap``: the smoothed density map; ``label_map``: the cluster label of every bin.
    """
    from scipy import ndimage
    from skimage.morphology import local_maxima
    from skimage.segmentation import watershed

    rc = np.asarray(rc, dtype=np.float64).reshape(-1, 2)
    cnt, (cen_r, cen_c) = density_map(rc, (nrow, ncol), factor)
    alimap = gaussian_matlab(cnt, 2.0)                             # imgaussfilt(cnt, 2)
    above = alimap >= threshold
    if not above.any():
        return (np.zeros(len(rc), dtype=int), np.zeros((0, 2)), alimap,
                np.zeros(alimap.shape, dtype=np.int32))
    map_th = np.where(above, alimap, 0.0)
    seeds = local_maxima(map_th, connectivity=2, allow_borders=True)          # imregionalmax(., 8)
    markers, _ = ndimage.label(seeds, structure=np.ones((3, 3), dtype=int))
    mask = None
    if flood_margin is not None:
        mask = ndimage.binary_dilation(above, structure=np.ones((3, 3), bool),
                                       iterations=int(flood_margin)) | (markers > 0)
    label_map = watershed(np.zeros(alimap.shape), markers=markers, mask=mask, connectivity=2,
                          watershed_line=True).astype(np.int32)                # flat flood
    label_map[~above] = 0
    si, sj = np.nonzero(seeds.T)                                               # column-major, as find()
    seeds_rc = np.column_stack([cen_r[sj], cen_c[si]])
    nb_r, nb_c = alimap.shape
    ri = np.clip(round_half_away_from_zero((rc[:, 0] - 0.5) * factor + 0.5), 1, nb_r).astype(int) - 1
    ci = np.clip(round_half_away_from_zero((rc[:, 1] - 0.5) * factor + 0.5), 1, nb_c).astype(int) - 1
    return label_map[ri, ci].astype(int), seeds_rc, alimap, label_map
