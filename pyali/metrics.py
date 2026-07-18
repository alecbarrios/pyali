"""Per-cell high-frequency SNR metrics.

Used by the interactive cell explorer (:mod:`pyali.figures`) to label and filter cells. The same
definitions are used by the standalone benchmark ``snr_analysis/snr/snr_compare.py``:

  * ``noise_sigma``      - robust HF noise floor, ``1.4826 * MAD`` of the 20 Hz high-pass trace
  * ``snr_median``       - median detected-spike amplitude (> ``k`` sigma) divided by the floor
  * ``spectral_hf_snr``  - excess Welch-PSD power in the 20-150 Hz band over the white floor
  * ``n_spikes``         - number of detected peaks
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch


def highpass(x, cutoff, fs, order=4):
    """Zero-phase Butterworth high-pass (removes slow drift, keeps the spike band)."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, x)


def robust_sigma(x):
    """Robust noise std via MAD (insensitive to spikes)."""
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def per_cell_snr(traces, fps, hp=20.0, k=3.0, sig_hi=150.0, floor_lo=300.0):
    """Return dict of per-cell arrays: noise_sigma, snr_median, spectral_hf_snr, n_spikes.

    ``traces`` is ``[N, T]``. NaN where a metric is undefined (e.g. no spikes / silent floor).
    """
    traces = np.asarray(traces, float)
    N = traces.shape[0]
    ns = np.full(N, np.nan)
    sm = np.full(N, np.nan)
    sh = np.full(N, np.nan)
    nsp = np.zeros(N, int)
    dist = max(1, int(round(0.01 * fps)))
    for i, x in enumerate(traces):
        xhp = highpass(x, hp, fps)
        sig = robust_sigma(xhp)
        ns[i] = sig
        if sig > 0:
            pk, pr = find_peaks(xhp, height=k * sig, distance=dist)
            nsp[i] = pk.size
            if pk.size:
                sm[i] = float(np.median(pr["peak_heights"]) / sig)
        f, P = welch(x - x.mean(), fs=fps, nperseg=int(min(4096, len(x))), window="hann")
        fl = f >= floor_lo
        sb = (f >= hp) & (f <= sig_hi)
        floor = float(np.median(P[fl])) if np.any(fl) else np.nan
        sigb = float(np.mean(P[sb])) if np.any(sb) else np.nan
        if floor and np.isfinite(floor) and floor > 0:
            sh[i] = (sigb - floor) / floor
    return dict(noise_sigma=ns, snr_median=sm, spectral_hf_snr=sh, n_spikes=nsp)
