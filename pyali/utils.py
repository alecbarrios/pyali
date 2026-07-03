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


def movmedian_time(a, window=8, axis=0):
    """Moving median along ``axis`` with a shrinking window at the array ends (no padding).

    The window at index ``i`` spans ``[i - kb, i + kf]`` with ``kb = window // 2`` and
    ``kf = window - 1 - kb`` (for ``window=8``: 4 before, 3 after, plus the current sample).
    Even-length windows use the mean of the two central order statistics.

    Parameters
    ----------
    a : array_like
    window : int
    axis : int

    Returns
    -------
    ndarray of float64, same shape as ``a``

    Notes
    -----
    Clear O(n)-medians reference implementation; slow for very large arrays.
    """
    a = np.asarray(a, dtype=np.float64)
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    kb = window // 2
    kf = window - 1 - kb
    out = np.empty_like(a)
    for i in range(n):
        lo = max(0, i - kb)
        hi = min(n, i + kf + 1)                 # shrink at the ends
        out[i] = np.median(a[lo:hi], axis=0)
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
