#!/usr/bin/env python3
"""Measure POST-HOC (no pipeline re-run) SNR improvements on already-extracted cell_traces.

The full pipeline's big SNR levers (a real temporal band-pass instead of the DOG no-op,
denoised/optimally-weighted footprints, regularized trace unmixing) require re-running the
extraction on the raw movie.  This script demonstrates and *quantifies* the improvements that
can be applied to the final ``cell_traces`` array alone, and — crucially — checks that each one
raises SNR WITHOUT distorting spike amplitude (an adversarial guard against "SNR cheating").

Currently demonstrated:
  * mains-hum removal   — narrow IIR notch at 60/120/180 Hz (the line noise visible in the PSD,
                          present identically in both pyali and MATLAB output).
  * slow-drift removal  — per-cell high-pass detrend (removes residual bleaching/network drift).

For each cell it reports the robust HF noise floor and spike SNR before/after, plus the change
in detected-spike amplitude (the distortion check: a legitimate denoiser leaves spike amplitude
≈ unchanged while lowering the noise floor).

Usage
-----
    python snr_postproc_demo.py PY_DIR_OR_MAT [--fps 800] [--hp 20] [--notch 60,120,180]
                                [--out postproc_demo]
"""
import argparse
import os

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def _load(path, var):
    import h5py
    if os.path.isdir(path):
        for c in (os.path.join(path, "ALI_Result.mat"),
                  os.path.join(path, "analysis", "ALI_Result.mat")):
            if os.path.isfile(c):
                path = c
                break
    with h5py.File(path, "r") as f:
        a = np.array(f[var])
        return (a.T if a.ndim >= 2 else a.squeeze()), path


def highpass(x, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, x)


def notch_chain(x, freqs, fs, q=30.0):
    y = x.copy()
    for f0 in freqs:
        if f0 >= 0.5 * fs:
            continue
        b, a = iirnotch(f0, q, fs)
        y = filtfilt(b, a, y)
    return y


def robust_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def snr_of(trace, fs, hp, k=4.0):
    """Robust noise floor + spike SNR + spike amplitudes for one trace."""
    xhp = highpass(trace, hp, fs)
    sigma = robust_sigma(xhp)
    if sigma <= 0:
        return dict(sigma=np.nan, snr=np.nan, n=0, amps=np.array([]), peaks=np.array([]))
    pk, pr = find_peaks(xhp, height=k * sigma, distance=max(1, int(0.01 * fs)))
    amps = pr["peak_heights"] if pk.size else np.array([])
    snr = float(np.median(amps) / sigma) if amps.size else np.nan
    return dict(sigma=float(sigma), snr=snr, n=int(pk.size), amps=amps, peaks=pk)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("py")
    ap.add_argument("--fps", type=float, default=800.0)
    ap.add_argument("--hp", type=float, default=20.0)
    ap.add_argument("--notch", default="60,120,180")
    ap.add_argument("--q", type=float, default=300.0,
                    help="notch quality factor; high Q (narrow) preserves broadband spikes. "
                         "Low Q (e.g. 30) removes more hum but clips spike amplitude.")
    ap.add_argument("--out", default="postproc_demo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    freqs = [float(f) for f in args.notch.split(",") if f.strip()]

    ct, path = _load(args.py, "cell_traces")
    ct = ct.astype(np.float64)
    if ct.shape[0] > ct.shape[1]:            # ensure [N, T]
        ct = ct.T
    N, T = ct.shape
    print(f"[postproc] {path}\n[postproc] {N} cells x {T} frames, fps={args.fps}")

    variants = {
        "baseline": lambda x: x,
        "notch": lambda x: notch_chain(x, freqs, args.fps, args.q),
    }
    res = {k: dict(sigma=[], snr=[], amp=[]) for k in variants}
    # track paired amplitude preservation on matched spikes (baseline peaks)
    amp_ratio = []
    for i in range(N):
        base = snr_of(ct[i], args.fps, args.hp)
        for name, fn in variants.items():
            m = snr_of(fn(ct[i]), args.fps, args.hp)
            res[name]["sigma"].append(m["sigma"])
            res[name]["snr"].append(m["snr"])
            res[name]["amp"].append(np.median(m["amps"]) if m["amps"].size else np.nan)
        # distortion check: amplitude at baseline peak locations, notched vs baseline
        if base["peaks"].size:
            xhp_b = highpass(ct[i], args.hp, args.fps)
            xhp_n = highpass(notch_chain(ct[i], freqs, args.fps, args.q), args.hp, args.fps)
            pk = base["peaks"]
            ratio = np.median(xhp_n[pk] / xhp_b[pk])
            if np.isfinite(ratio):
                amp_ratio.append(ratio)

    def med(a):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else np.nan

    print("\n  variant     med noise_sigma   med spike_SNR   med spike_amp")
    for name in variants:
        print(f"  {name:10s}  {med(res[name]['sigma']):14.5f}  {med(res[name]['snr']):13.3f}  "
              f"{med(res[name]['amp']):13.4f}")
    d_sigma = 100 * (med(res["notch"]["sigma"]) / med(res["baseline"]["sigma"]) - 1)
    d_snr = 100 * (med(res["notch"]["snr"]) / med(res["baseline"]["snr"]) - 1)
    print(f"\n  notch effect: noise floor {d_sigma:+.1f}%   spike SNR {d_snr:+.1f}%")
    print(f"  DISTORTION CHECK — median spike-amplitude preserved after notch: "
          f"{med(amp_ratio):.4f}  (1.0 = no distortion; want >~0.97)")
    print("\n  NOTE: these are post-hoc gains on the final traces only. The larger SNR gains")
    print("  (real temporal band-pass, denoised footprints, regularized unmixing) require")
    print("  re-running the pipeline on the raw movie; validate those with snr_compare.py.")

    if HAVE_MPL:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name in variants:
            s = np.asarray(res[name]["snr"], float); s = s[np.isfinite(s)]
            ax[0].hist(s, bins=40, histtype="step", label=name, lw=1.5)
        ax[0].set_xlabel("spike SNR"); ax[0].set_ylabel("cells"); ax[0].legend()
        ax[0].set_title("Spike SNR before/after notch")
        ar = np.asarray(amp_ratio, float); ar = ar[np.isfinite(ar)]
        ax[1].hist(ar, bins=40); ax[1].axvline(1.0, color="r", ls="--")
        ax[1].set_xlabel("spike amp ratio (notch / baseline)"); ax[1].set_ylabel("cells")
        ax[1].set_title(f"Distortion check (median {med(amp_ratio):.3f})")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "postproc_snr.png"), dpi=140)
        plt.close(fig)
        print(f"\n[postproc] figure -> {args.out}/postproc_snr.png")


if __name__ == "__main__":
    main()
