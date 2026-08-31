"""Per-cell waveform features computed from extracted traces.

Everything here is post hoc: it reads ``cell_traces`` (``[N, T]``) and never touches the movie.
Spike detection mirrors :func:`pyali.metrics.per_cell_snr` -- zero-phase high-pass at ``hp`` Hz,
robust sigma of the filtered trace, peaks above ``k * sigma`` at least 10 ms apart -- so the two
modules agree on what a spike is.

* :func:`spike_triggered_average` -- each cell's mean waveform around its spikes, in units of that
  cell's own noise sigma, with sub-sample alignment. Sigma units are what make cells and
  acquisitions comparable: absolute trace units depend on bit depth, frame size and footprint
  weighting; sigma-normalised amplitude does not.
"""
import numpy as np
from scipy.signal import find_peaks

from .metrics import highpass, robust_sigma


def _cubic_shift(seg4, d, n):
    """Resample ``n`` points at fractional offset ``d`` using a Catmull-Rom cubic.

    ``seg4`` carries 2 guard samples on each side. **Linear** interpolation is not usable here:
    an action potential is only ~2 samples wide at 800 Hz, and linear resampling broadens the
    averaged FWHM while *lowering* the peak. A cubic keeps the width and the peak. A `scipy`
    spline object per snippet would be exact but costs ~40 us per spike; this closed form is the
    same cubic family, vectorised.
    """
    i0 = int(np.floor(d))
    t = d - i0
    idx = np.arange(n) + 2 + i0
    pm1, p0, p1, p2 = seg4[idx - 1], seg4[idx], seg4[idx + 1], seg4[idx + 2]
    return (p0 + 0.5 * t * (p1 - pm1)
            + 0.5 * t * t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1 - p2)
            + 0.5 * t * t * t * (-pm1 + 3.0 * p0 - 3.0 * p1 + p2))


def cell_sta(x, xhp, sig, pk, half=20, align=True, min_spikes=3, shoulder=6):
    """Mean waveform around one cell's spikes, in sigma units.

    Snippets are cut from ``x`` (the raw trace) while the peaks ``pk`` came from ``xhp`` (the
    high-passed trace used for detection). Averaging the *filtered* trace would stamp the
    zero-phase filter's symmetric negative lobes onto the waveform as a spurious pre-spike dip
    and an inflated undershoot; the raw trace shows the physiological rising ramp instead.

    Parameters
    ----------
    x, xhp : (T,) float arrays
        Raw trace and its high-passed version.
    sig : float
        Robust noise sigma of ``xhp``; amplitudes are returned in units of it.
    pk : (K,) int array
        Sorted peak sample indices from ``find_peaks`` on ``xhp``.
    half : int
        Samples on each side of the peak; the window is ``2 * half + 1`` samples.
    align : bool
        Sub-sample alignment: the parabolic vertex of the filtered peak gives the offset, and
        the snippet is resampled with :func:`_cubic_shift`. Without it, nearest-sample
        alignment smears an average of ~2-sample-wide spikes by up to half a sample.
    min_spikes : int
        A cell needs at least this many clean snippets to contribute an STA.
    shoulder : int
        Samples of the *leading* shoulder used as the baseline. Using both shoulders would
        force the post-spike tail to zero and hide any genuine afterpotential.

    Returns
    -------
    (sta, n_used)
        ``sta`` is a ``(2 * half + 1,)`` float array, or ``None`` when fewer than ``min_spikes``
        clean snippets exist; ``n_used`` is the number of snippets averaged.
    """
    T = x.size
    pad = 2 if align else 0                # guard samples the cubic needs
    usable = []
    for j, p in enumerate(pk):
        if p < half + pad + 1 or p >= T - half - pad - 1:
            continue
        # Neighbour exclusion: another detected spike inside the window would add a second bump.
        if (j and p - pk[j - 1] <= half) or (j + 1 < pk.size and pk[j + 1] - p <= half):
            continue
        usable.append(p)
    if len(usable) < min_spikes:
        return None, 0
    n = 2 * half + 1
    acc = np.zeros(n)
    for p in usable:
        if align:
            ym1, y0, yp1 = xhp[p - 1], xhp[p], xhp[p + 1]
            den = ym1 - 2.0 * y0 + yp1
            d = 0.5 * (ym1 - yp1) / den if den != 0 else 0.0
            d = float(np.clip(d, -0.5, 0.5))
            w = _cubic_shift(x[p - half - 2:p + half + 3].astype(float), d, n)
        else:
            w = x[p - half:p + half + 1].astype(float)
        acc += w - np.median(w[:shoulder])
    return acc / len(usable) / sig, len(usable)


def spike_triggered_average(traces, fps, hp=20.0, k=4.0, half=20, align=True, min_spikes=3,
                            shoulder=6):
    """Spike-triggered average of every cell in ``traces`` (``[N, T]``).

    Detection parameters ``hp`` and ``k`` default to the same values as
    :func:`pyali.metrics.per_cell_snr`; see :func:`cell_sta` for the rest.

    Returns
    -------
    dict
        ``sta`` -- ``(M, 2 * half + 1)`` array of the M cells with an STA, in sigma units;
        ``cell_index`` -- ``(M,)`` row index of each into ``traces``;
        ``n_spikes_used`` -- ``(M,)`` snippets averaged per cell;
        ``n_spikes_total`` -- ``(N,)`` detected peaks per cell, all cells;
        ``t_ms`` -- ``(2 * half + 1,)`` time axis in milliseconds, 0 at the peak.
    """
    traces = np.asarray(traces, float)
    N = traces.shape[0]
    dist = max(1, int(round(0.01 * fps)))
    t_ms = np.arange(-half, half + 1) * (1000.0 / fps)
    curves, idx, used = [], [], []
    n_total = np.zeros(N, int)
    for i in range(N):
        x = traces[i]
        xhp = highpass(x, hp, fps)
        sig = robust_sigma(xhp)
        if sig <= 0:
            continue
        pk, _ = find_peaks(xhp, height=k * sig, distance=dist)
        n_total[i] = pk.size
        c, nu = cell_sta(x, xhp, sig, pk, half, align, min_spikes, shoulder)
        if c is not None:
            curves.append(c)
            idx.append(i)
            used.append(nu)
    sta = np.asarray(curves, float).reshape(len(curves), 2 * half + 1)
    return dict(sta=sta, cell_index=np.asarray(idx, int), n_spikes_used=np.asarray(used, int),
                n_spikes_total=n_total, t_ms=t_ms)
