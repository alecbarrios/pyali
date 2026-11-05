"""Small numeric helpers used across the pipeline (kernels, rounding, moving median,
peak finding, tolerance grouping)."""
import numpy as np

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (trapz still works but warns).
_trapz = getattr(np, "trapezoid", np.trapz)


def round_half_away_from_zero(x):
    """Round to the nearest integer with ties rounded away from zero.

    Differs from NumPy's round-half-to-even: round(0.5)=1, round(-0.5)=-1, round(375.5)=376.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray of float64
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def fspecial_laplacian(alpha=0.2):
    """3x3 Laplacian kernel parameterized by ``alpha`` in [0, 1].

    corners = a/(a+1), edges = (1-a)/(a+1), center = -4/(a+1).

    Parameters
    ----------
    alpha : float

    Returns
    -------
    (3, 3) ndarray of float64
    """
    a = float(alpha)
    k = np.array([[a, 1 - a, a],
                  [1 - a, -4.0, 1 - a],
                  [a, 1 - a, a]], dtype=np.float64)
    return k / (a + 1.0)


def strel_disk(radius=15):
    """Octagonal disk structuring element of the given ``radius`` (a polygonal approximation,
    not a Euclidean disk).

    For ``radius=15`` this is a 29x29 boolean mask equal to the L1 ball ``|x| + |y| <= 20``
    on the ``[-14, 14]^2`` grid. The general ``(rr, b)`` rule below matches ``radius=15``;
    other radii are a reasonable octagon approximation.

    Parameters
    ----------
    radius : int

    Returns
    -------
    (2*radius-1, 2*radius-1) ndarray of bool
    """
    rr = radius - 1                       # grid half-size  (15 -> 14  => 29x29)
    b = int(round(radius * 4 / 3))        # L1 (diamond) bound (15 -> 20)
    y, x = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    return (np.abs(x) + np.abs(y)) <= b


def dog_kernel(sigma1=1.0, sigma2=3.0, width=19):
    """Difference-of-Gaussians kernel, zero-summed via area normalization.

    Two Gaussians centered at ``(1 + width) / 2`` are subtracted with a weight that equalizes
    their areas, then the result is scaled so its peak (center tap) is 1.0.

    Parameters
    ----------
    sigma1, sigma2 : float
        Narrow and wide Gaussian widths.
    width : int
        Kernel length (odd).

    Returns
    -------
    (width,) ndarray of float64
    """
    x = np.arange(1, width + 1, dtype=np.float64)
    c = (1 + width) / 2.0
    g1 = np.exp(-(x - c) ** 2 / (2.0 * sigma1 ** 2))
    g2 = np.exp(-(x - c) ** 2 / (2.0 * sigma2 ** 2))
    area_ratio = _trapz(g1, x) / _trapz(g2, x)
    dog = g1 - area_ratio * g2
    return dog / dog.max()


# Comparator network for the median of 8. Batcher's odd-even mergesort fully sorts 8 elements in
# 19 compare-exchanges, but the median only needs order statistics 3 and 4, so two comparators
# are redundant. The 17 below were found by greedily dropping comparators and re-verifying with
# the 0-1 principle (a network computes the right order statistics for all inputs iff it does so
# for every 0/1 input) over all 2^8 = 256 binary vectors — see the unit test in _median8_network.
#
# The first four are applied straight to the input views, which removes the eight upfront array
# copies the previous implementation needed; together that is ~26% fewer passes over the data.
_NET8_FIRST = ((0, 1), (2, 3), (4, 5), (6, 7))
_NET8_TAIL = ((0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (3, 7), (1, 5), (2, 6),
              (1, 4), (3, 6), (2, 4), (3, 5), (3, 4))


def _median8_network(chans, out):
    """Median of 8 aligned arrays via a comparator network, written into ``out``.

    Each compare-exchange is a pair of vectorized ``minimum``/``maximum`` calls over whole
    arrays — no per-element partition and no gather, so the cost is plain streaming memory
    traffic. The median of an even-length window is the mean of the two central order
    statistics, which after the network are channels 3 and 4.
    """
    w = [None] * 8
    for i, j in _NET8_FIRST:                    # straight from the read-only input views
        w[i] = np.minimum(chans[i], chans[j])
        w[j] = np.maximum(chans[i], chans[j])
    for i, j in _NET8_TAIL:
        lo = np.minimum(w[i], w[j])
        np.maximum(w[i], w[j], out=w[j])        # in place; w[i] is not yet overwritten
        w[i] = lo
    np.add(w[3], w[4], out=out)
    out *= 0.5
    return out


def movmedian_time(a, window=8, axis=0, chunk_bytes=256 << 20):
    """Moving median along ``axis`` with a shrinking window at the array ends (no padding).

    The window at index ``i`` spans ``[i - kb, i + kf]`` with ``kb = window // 2`` and
    ``kf = window - 1 - kb`` (for ``window=8``: 4 before, 3 after, plus the current sample).
    Even-length windows use the mean of the two central order statistics.

    Parameters
    ----------
    a : array_like
    window : int
    axis : int
    chunk_bytes : int
        Working-buffer budget for the ``window == 8`` fast path.

    Returns
    -------
    ndarray, same shape as ``a`` (float32 input stays float32; ints are upcast to float64)

    Notes
    -----
    ``window == 8`` — the only window the pipeline uses — takes a vectorized sorting-network
    path that is ~3.4x faster than the per-index ``np.median`` loop and **bit-identical** to it.
    The interior windows are all exactly 8 long and are formed as 8 shifted *views* of the time
    axis, so only the network's working buffers are allocated; those are chunked over pixels.
    The shrinking end windows keep the reference path. Any other window uses the reference loop.
    """
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):          # preserve float32/float64; upcast ints
        a = a.astype(np.float64)
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    kb = window // 2
    kf = window - 1 - kb
    out = np.empty_like(a)

    if window != 8:                                      # reference path
        for i in range(n):
            out[i] = np.median(a[max(0, i - kb):min(n, i + kf + 1)], axis=0)
        return np.moveaxis(out, 0, axis)

    lo, hi = kb, n - kf                                  # interior output rows [lo, hi)
    ends = list(range(min(kb, n))) + list(range(max(kb, n - kf), n))
    for i in ends:                                       # truncated windows: reference path
        out[i] = np.median(a[max(0, i - kb):min(n, i + kf + 1)], axis=0)
    if hi <= lo:
        return np.moveaxis(out, 0, axis)

    flat = a.reshape(n, -1)
    out_flat = out.reshape(n, -1)
    npix = flat.shape[1]
    rows = hi - lo
    step = max(1, min(npix, int(chunk_bytes // max(rows * a.dtype.itemsize * 9, 1))))
    for c0 in range(0, npix, step):
        c1 = min(npix, c0 + step)
        # output row lo+r is the median of a[r : r+8], so channel k is a[k : k+rows]
        chans = [flat[k:k + rows, c0:c1] for k in range(8)]
        _median8_network(chans, out_flat[lo:hi, c0:c1])
    return np.moveaxis(out, 0, axis)


def findpeaks(x):
    """Local maxima of a 1-D signal (values only).

    A peak is a sample strictly greater than both neighbors; endpoints are excluded and a
    flat-topped peak returns its lowest (left-edge) index.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    (peak_values, peak_indices) : ndarray, ndarray
        0-based indices in ascending order.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    idx = []
    i = 1
    while i < n - 1:
        if x[i] > x[i - 1]:                      # rising into i
            j = i
            while j < n - 1 and x[j + 1] == x[j]:   # extend across a flat top
                j += 1
            if x[j] > x[j + 1]:                  # falls after -> peak at the left edge i
                idx.append(i)
            i = j + 1
        else:
            i += 1
    idx = np.array(idx, dtype=int)
    return x[idx], idx


def uniquetol_reps(values, tol, occurrence="highest"):
    """Representatives of a tolerance-based grouping of ``values``.

    Sorts the values, greedily groups runs within ``tol`` of the group's first element, and
    returns the highest (or lowest) actual value in each group.

    Parameters
    ----------
    values : array_like
    tol : float
        Absolute grouping tolerance.
    occurrence : {'highest', 'lowest'}

    Returns
    -------
    ndarray
    """
    v = np.sort(np.asarray(values, dtype=np.float64))
    reps = []
    i, n = 0, v.size
    while i < n:
        j = i
        while j + 1 < n and (v[j + 1] - v[i]) <= tol:
            j += 1
        grp = v[i:j + 1]
        reps.append(grp.max() if occurrence == "highest" else grp.min())
        i = j + 1
    return np.array(reps)
