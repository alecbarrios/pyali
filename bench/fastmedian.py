"""Sorting-network moving median along time — bit-identical to pyali.utils.movmedian_time.

The pipeline's window is 8, so every interior output is the median of exactly 8 samples =
the mean of the 4th and 5th order statistics. Batcher's odd-even network sorts 8 values in
19 compare-exchanges, each a pair of vectorized ``np.minimum``/``np.maximum`` calls over
whole arrays — no per-element partition, no gather, pure streaming memory traffic.

The 8 window channels are *views* into the time axis (``a[k : n-7+k]``), so only the network's
working buffers are allocated, and those are chunked over pixels to bound peak memory.
"""
import numpy as np

# Batcher odd-even mergesort network for n=8 (19 comparators).
_NET8 = [(0, 1), (2, 3), (4, 5), (6, 7),
         (0, 2), (1, 3), (4, 6), (5, 7),
         (1, 2), (5, 6), (0, 4), (3, 7),
         (1, 5), (2, 6),
         (1, 4), (3, 6),
         (2, 4), (3, 5),
         (3, 4)]


def _median8_network(chans, out):
    """Median of 8 aligned arrays via a sorting network. ``chans`` is a list of 8 views."""
    w = [np.array(c, copy=True) for c in chans]          # working copies (views are read-only inputs)
    for i, j in _NET8:
        lo = np.minimum(w[i], w[j])
        hi = np.maximum(w[i], w[j])
        w[i], w[j] = lo, hi
    np.add(w[3], w[4], out=out)
    out *= 0.5
    return out


def movmedian_time_net(a, window=8, axis=0, chunk_bytes=256 << 20):
    """Drop-in replacement for :func:`pyali.utils.movmedian_time` (window=8 fast path)."""
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    kb = window // 2                                     # 4
    kf = window - 1 - kb                                 # 3
    out = np.empty_like(a)

    # ends: shrinking window, cheap (kb + kf = 7 rows) -> keep the original definition
    for i in list(range(min(kb, n))) + list(range(max(kb, n - kf), n)):
        out[i] = np.median(a[max(0, i - kb):min(n, i + kf + 1)], axis=0)

    lo, hi = kb, n - kf                                  # interior output rows [lo, hi)
    if hi <= lo or window != 8:
        if window != 8:                                  # fall back for any other window
            from pyali.utils import movmedian_time as _slow
            return _slow(np.moveaxis(a, 0, axis), window, axis)
        return np.moveaxis(out, 0, axis)

    flat = a.reshape(n, -1)
    out_flat = out.reshape(n, -1)
    npix = flat.shape[1]
    rows = hi - lo
    # 8 working buffers + output, all [rows, step]
    step = max(1, min(npix, int(chunk_bytes // (rows * a.dtype.itemsize * 9))))
    for c0 in range(0, npix, step):
        c1 = min(npix, c0 + step)
        chans = [flat[k:k + rows, c0:c1] for k in range(8)]   # window for out row lo+r is a[r:r+8]
        _median8_network(chans, out_flat[lo:hi, c0:c1])
    return np.moveaxis(out, 0, axis)
