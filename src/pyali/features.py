"""Per-cell waveform features computed from extracted traces.

Everything here is post hoc: it reads ``cell_traces`` (``[N, T]``) and never touches the movie.
Spike detection mirrors :func:`pyali.metrics.per_cell_snr` -- zero-phase high-pass at ``hp`` Hz,
robust sigma of the filtered trace, peaks above ``k * sigma`` at least 10 ms apart -- so the two
modules agree on what a spike is.

* :func:`spike_triggered_average` -- each cell's mean waveform around its spikes, in units of that
  cell's own noise sigma, with sub-sample alignment. Sigma units are what make cells and
  acquisitions comparable: absolute trace units depend on bit depth, frame size and footprint
  weighting; sigma-normalised amplitude does not.
* :func:`sta_variance_decomposition` -- how much of the across-cell spread of the STA is real
  (between cells) versus the finite-snippet measurement variance.
* :func:`cell_features` -- per-cell shape features (peak, decay ratio, FWHM) as arrays, and
  :func:`decay_noise_sd`, the decay-ratio spread a pure noise crossing would show.
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


def sta_variance_decomposition(A, n_used, t_ms, t_late_ms=2.5):
    """Between-cell vs measurement variance of the per-cell STA at the peak and a late sample.

    Cells in one field can be genuinely heterogeneous (for example a pooled perturbation library,
    where every cell may carry a different perturbation), and the question is whether the spread
    of their waveforms is real. Each snippet carries ~1 sigma of noise by construction, so a
    cell's STA point has measurement variance ``1 / n_spikes_used``; averaged over cells that is
    ``mean(1 / n_used)`` (``var_meas``, the one formula that is easy to get wrong -- it is not
    ``1 / median(n)``). Subtracting it from the observed across-cell variance leaves the part that
    is genuinely between cells. ``ratio = var_obs / var_meas`` near 1 is the signature of cells
    whose "spikes" are threshold noise crossings.

    Parameters
    ----------
    A : (M, L) float array
        Per-cell STAs in sigma units, e.g. ``spike_triggered_average(...)["sta"]``.
    n_used : (M,) array
        Snippets averaged per cell.
    t_ms : (L,) array
        Time axis of ``A`` in milliseconds.
    t_late_ms : float
        The late sample, by default 2.5 ms after the peak.

    Returns
    -------
    dict of floats, keyed ``pk_*`` (peak) and ``t25_*`` (late sample): ``median``, ``q25``,
    ``q75``, ``p5``, ``p95``, ``var_obs``, ``var_meas``, ``var_bio``, ``sd_bio``, ``ratio``;
    plus ``n_cells`` and ``n_spikes_used_median``.
    """
    A = np.asarray(A, float)
    n_used = np.asarray(n_used, float)
    out = {}
    i0 = int(np.argmin(np.abs(t_ms)))
    i25 = int(np.argmin(np.abs(t_ms - t_late_ms)))
    vm = float(np.mean(1.0 / np.clip(n_used, 1, None))) if n_used.size else np.nan
    for tag, i in (("pk", i0), ("t25", i25)):
        v = A[:, i] if A.size else np.zeros(0)
        vo = float(np.var(v, ddof=1)) if v.size > 1 else 0.0
        q = np.percentile(v, [5, 25, 50, 75, 95]) if v.size else [np.nan] * 5
        out.update({f"{tag}_median": float(q[2]), f"{tag}_q25": float(q[1]),
                    f"{tag}_q75": float(q[3]), f"{tag}_p5": float(q[0]), f"{tag}_p95": float(q[4]),
                    f"{tag}_var_obs": vo, f"{tag}_var_meas": vm,
                    f"{tag}_var_bio": max(vo - vm, 0.0) if np.isfinite(vm) else np.nan,
                    f"{tag}_sd_bio": float(np.sqrt(max(vo - vm, 0.0))) if np.isfinite(vm) else np.nan,
                    f"{tag}_ratio": vo / vm if vm and vm > 0 else np.nan})
    out["n_cells"] = int(A.shape[0])
    out["n_spikes_used_median"] = float(np.median(n_used)) if n_used.size else np.nan
    return out


def cell_features(A, n_used, cell_index, t_ms, t_decay_ms=2.5, t_decay_end_ms=5.0):
    """Per-cell waveform shape features from the STA matrix. One entry per cell, as arrays.

    **decay** is the amplitude at ``t_decay_ms`` after the peak divided by the peak. A real action
    potential decays over a few milliseconds, so a genuine AP lands well above zero, whereas a
    threshold noise crossing is a single-sample excursion with nothing after it and lands near 0.
    It is a pure SHAPE measure -- independent of amplitude and of the sigma normalisation -- so
    filtering on it does not manufacture the amplitude result being interpreted, and it is immune
    to the ``k * sigma`` detection floor that compresses the amplitude distribution.

    **fwhm_ms** is quantised to the sample interval (an AP is only ~2 samples wide at 800 Hz), so
    it is reported for completeness, not as a continuous shape variable.

    Parameters
    ----------
    A : (M, L) float array
        Per-cell STAs in sigma units.
    n_used : (M,) array
        Snippets averaged per cell.
    cell_index : (M,) array
        Row of each cell in the original ``cell_traces``.
    t_ms : (L,) array
        Time axis of ``A`` in milliseconds.
    t_decay_ms, t_decay_end_ms : float
        The decay sample, and the end of the window averaged for ``decay_5ms``.

    Returns
    -------
    dict of equal-length arrays: ``cell_index`` (int32), ``peak_sigma``, ``decay``,
    ``decay_5ms``, ``fwhm_ms`` (float32), ``n_spikes_used`` (int32).
    """
    A = np.asarray(A, float)
    t_ms = np.asarray(t_ms, float)
    dt = float(t_ms[1] - t_ms[0]) if t_ms.size > 1 else np.nan
    i0 = int(np.argmin(np.abs(t_ms)))
    i25 = int(np.argmin(np.abs(t_ms - t_decay_ms)))
    i50 = int(np.argmin(np.abs(t_ms - t_decay_end_ms)))
    pk = A[:, i0].astype(float) if A.size else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(pk > 0, A[:, i25] / pk, np.nan) if A.size else np.zeros(0)
        decay5 = (np.where(pk > 0, A[:, i25:i50 + 1].mean(axis=1) / pk, np.nan)
                  if A.size else np.zeros(0))
    fwhm = np.full(A.shape[0], np.nan)
    for j in range(A.shape[0]):
        if pk[j] <= 0:
            continue
        above = np.flatnonzero(A[j] >= pk[j] / 2.0)
        if above.size:
            lo = hi = i0                                  # contiguous run containing the peak
            while lo - 1 in above:
                lo -= 1
            while hi + 1 in above:
                hi += 1
            fwhm[j] = (hi - lo + 1) * dt
    return dict(cell_index=np.asarray(cell_index, np.int32),
                peak_sigma=pk.astype(np.float32),
                decay=np.asarray(decay, np.float32),
                decay_5ms=np.asarray(decay5, np.float32),
                fwhm_ms=fwhm.astype(np.float32),
                n_spikes_used=np.asarray(n_used, np.int32))


def decay_noise_sd(peak_sigma, n_used):
    """Expected decay-ratio spread for a cell whose "spike" is a threshold noise crossing.

    The sample after a noise crossing is uncorrelated noise. Averaged over ``n`` snippets it has SD
    ~ ``1/sqrt(n)`` in sigma units; dividing by the peak gives ``1/(peak*sqrt(n))``. This is why the
    noise population is a broad shoulder centred on 0 rather than a spike at 0, and why it can
    overlap the AP population instead of separating from it.
    """
    return 1.0 / (np.clip(np.asarray(peak_sigma, float), 1e-6, None)
                  * np.sqrt(np.clip(np.asarray(n_used, float), 1, None)))
