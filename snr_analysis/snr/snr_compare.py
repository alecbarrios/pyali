#!/usr/bin/env python3
"""Quantitative high-frequency SNR comparison of TWO pyali runs on the SAME movie.

This is the benchmarking core: run the pipeline two ways (e.g. a new iteration vs the previous
one, or ``--whiten-traces`` vs the default) and measure whether the change actually improved the
signal-to-noise ratio of the extracted spike waveforms -- rather than just changing the output.

Each run saves a ``cell_traces`` array ``[N, T]`` and a ``footprint_center`` array ``[N, 2]``
(row, col) inside ``ALI_Result.mat``. This script:

1. Loads ``cell_traces`` + ``footprint_center`` from run A and run B.
2. Matches cells one-to-one by footprint-center proximity (Hungarian assignment) so we compare
   the *same physical cell* across runs (orderings and counts can differ slightly).
3. Computes a suite of high-frequency SNR metrics per cell for each run:
     * robust HF noise floor  (1.4826 * MAD of the high-pass-filtered trace)
     * spike SNR              (detected-peak amplitude / noise floor; median / p90 / max)
     * spectral HF-SNR        (excess Welch PSD power in the spike band over the white
                               shot-noise floor near Nyquist)
4. Paired diagnostics on matched cells: Pearson correlation (raw + high-pass) and
   magnitude-squared coherence vs frequency (where in the spectrum the two runs agree).
5. Aggregates, runs a Wilcoxon signed-rank test on the matched pairs, writes a CSV of per-cell
   metrics, a text/JSON summary, and diagnostic figures.

Interpreting a benchmark: a genuine improvement in run A over run B shows up as LOWER
``noise_sigma``/``psd_floor`` and HIGHER ``snr_*``/``spectral_hf_snr``, while ``corr_hp`` stays
~1 on cells the change should not have touched (confirming the waveform was not distorted).

Usage
-----
    python snr_compare.py RUN_A RUN_B [--out OUTDIR] [--fps 800] [--hp 20] [--k 4]
                          [--label-a new] [--label-b baseline]

Each positional argument may be either an ``ALI_Result.mat`` file, or a directory that contains
one (searched directly, then in an ``analysis/`` subdir). Example (whitened vs baseline A/B):

    python snr_compare.py /path/to/whitened /path/to/baseline \
        --label-a whitened --label-b baseline --out report_whiten_vs_baseline
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, filtfilt, find_peaks, welch, coherence

# ---- Optional plotting (script still emits CSV/summary if matplotlib is missing) ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:                                                       # pragma: no cover
    HAVE_MPL = False


# =========================================================================================
# I/O
# =========================================================================================
def _load_v73(path, var):
    """Read one variable from a v7.3 (HDF5) .mat, transposing 2-D+ to native orientation."""
    import h5py
    with h5py.File(path, "r") as f:
        a = np.array(f[var])
        return a.T if a.ndim >= 2 else a.squeeze()


def _resolve_mat(path):
    """Accept an ALI_Result.mat file or a directory containing one (also under analysis/)."""
    if os.path.isfile(path):
        return path
    for cand in (os.path.join(path, "ALI_Result.mat"),
                 os.path.join(path, "analysis", "ALI_Result.mat")):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"No ALI_Result.mat found at or under {path!r}")


def load_result(path):
    """Return (cell_traces [N,T], footprint_center [N,2]=row,col) from an ALI_Result.mat."""
    mat = _resolve_mat(path)
    ct = _load_v73(mat, "cell_traces").astype(np.float64)
    fc = _load_v73(mat, "footprint_center").astype(np.float64)
    if ct.shape[0] != fc.shape[0]:          # ensure [N,T] and [N,2] agree on N
        if ct.shape[1] == fc.shape[0]:
            ct = ct.T
    if fc.shape[1] != 2 and fc.shape[0] == 2:
        fc = fc.T
    return ct, fc, mat


# =========================================================================================
# Cell matching
# =========================================================================================
def match_cells(fc_a, fc_b, max_dist):
    """Hungarian one-to-one match of cells by footprint-center Euclidean distance.

    Returns (pairs, dists): pairs = list of (i_a, j_b); dists = matched center distances.
    Only pairs with distance <= max_dist are kept (confident same-cell matches).
    """
    D = np.linalg.norm(fc_a[:, None, :] - fc_b[None, :, :], axis=2)     # [Na, Nb]
    ri, ci = linear_sum_assignment(D)
    pairs, dists = [], []
    for i, j in zip(ri, ci):
        if D[i, j] <= max_dist:
            pairs.append((int(i), int(j)))
            dists.append(float(D[i, j]))
    return pairs, np.asarray(dists)


# =========================================================================================
# Signal conditioning + SNR metrics
# =========================================================================================
def highpass(x, cutoff, fs, order=4):
    """Zero-phase Butterworth high-pass; removes DC + slow drift, keeps the spike band."""
    b, a = butter(order, cutoff / (0.5 * fs), btype="high")
    return filtfilt(b, a, x)


def robust_sigma(x):
    """Robust noise std via MAD (spike-insensitive)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


def spike_snr_metrics(x_hp, sigma, fs, k=4.0, refractory_s=0.010):
    """Detect positive spikes on the high-passed trace and summarise amplitude / sigma."""
    if sigma <= 0 or not np.isfinite(sigma):
        return dict(n_spikes=0, snr_median=np.nan, snr_p90=np.nan, snr_max=np.nan)
    dist = max(1, int(round(refractory_s * fs)))
    peaks, props = find_peaks(x_hp, height=k * sigma, distance=dist)
    if peaks.size == 0:
        return dict(n_spikes=0, snr_median=np.nan, snr_p90=np.nan, snr_max=np.nan)
    amps = props["peak_heights"]
    return dict(n_spikes=int(peaks.size),
                snr_median=float(np.median(amps) / sigma),
                snr_p90=float(np.percentile(amps, 90) / sigma),
                snr_max=float(amps.max() / sigma))


def spectral_hf_snr(x, fs, hp, sig_hi=150.0, floor_lo=300.0, nperseg=4096):
    """Welch-PSD high-frequency SNR.

    Signal band = [hp, sig_hi] Hz (where AP transients concentrate excess power).
    White noise floor = median PSD in [floor_lo, Nyquist] Hz (shot noise ~ flat there).
    Returns (excess_ratio, floor, signal_power, freqs, Pxx) where excess_ratio =
    (mean signal-band PSD - floor) / floor.
    """
    nperseg = int(min(nperseg, len(x)))
    f, P = welch(x - np.mean(x), fs=fs, nperseg=nperseg, window="hann")
    ny = 0.5 * fs
    floor_hi = min(floor_lo + (ny - floor_lo), ny)
    fl = (f >= floor_lo) & (f <= floor_hi)
    sb = (f >= hp) & (f <= sig_hi)
    floor = float(np.median(P[fl])) if np.any(fl) else np.nan
    sig = float(np.mean(P[sb])) if np.any(sb) else np.nan
    excess = (sig - floor) / floor if (floor and np.isfinite(floor) and floor > 0) else np.nan
    return excess, floor, sig, f, P


def per_cell_metrics(traces, fs, hp, k):
    """Compute the full metric dict for every trace in ``traces`` [N,T]."""
    out = []
    for x in traces:
        x = np.asarray(x, float)
        xhp = highpass(x, hp, fs)
        sigma = robust_sigma(xhp)
        m = dict(noise_sigma=float(sigma))
        m.update(spike_snr_metrics(xhp, sigma, fs, k=k))
        excess, floor, sig, _f, _P = spectral_hf_snr(x, fs, hp)
        m["spectral_hf_snr"] = float(excess)
        m["psd_floor"] = float(floor)
        out.append(m)
    return out


# =========================================================================================
# Paired diagnostics
# =========================================================================================
def paired_diagnostics(x_a, x_b, fs, hp):
    """Pearson corr (raw + high-pass) and mean high-band coherence for one matched pair."""
    xa, xb = np.asarray(x_a, float), np.asarray(x_b, float)
    r_raw = float(np.corrcoef(xa, xb)[0, 1])
    hp_a, hp_b = highpass(xa, hp, fs), highpass(xb, hp, fs)
    r_hp = float(np.corrcoef(hp_a, hp_b)[0, 1])
    nperseg = int(min(2048, len(xa)))
    f, Cxy = coherence(hp_a, hp_b, fs=fs, nperseg=nperseg)
    hi = (f >= 100) & (f <= 300)
    coh_hi = float(np.mean(Cxy[hi])) if np.any(hi) else np.nan
    return dict(corr_raw=r_raw, corr_hp=r_hp, coh_hi=coh_hi), (f, Cxy)


# =========================================================================================
# Reporting
# =========================================================================================
def _summ(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, median=np.nan, mean=np.nan, iqr=np.nan)
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(np.mean(v)),
                iqr=float(np.percentile(v, 75) - np.percentile(v, 25)))


def write_csv(path, pairs, ma, mb, pdiag, dists):
    keys = ["noise_sigma", "n_spikes", "snr_median", "snr_p90", "snr_max",
            "spectral_hf_snr", "psd_floor"]
    with open(path, "w") as fh:
        hdr = (["pair", "i_a", "j_b", "center_dist"]
               + [f"a_{k}" for k in keys] + [f"b_{k}" for k in keys]
               + ["corr_raw", "corr_hp", "coh_hi"])
        fh.write(",".join(hdr) + "\n")
        for n, (i, j) in enumerate(pairs):
            row = [n, i, j, f"{dists[n]:.3f}"]
            row += [f"{ma[i][k]:.6g}" for k in keys]
            row += [f"{mb[j][k]:.6g}" for k in keys]
            row += [f"{pdiag[n][k]:.6g}" for k in ("corr_raw", "corr_hp", "coh_hi")]
            fh.write(",".join(str(v) for v in row) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="run A: ALI_Result.mat or its (analysis) directory")
    ap.add_argument("b", help="run B: ALI_Result.mat or its directory (the reference/baseline)")
    ap.add_argument("--out", default="snr_report", help="output directory")
    ap.add_argument("--fps", type=float, default=800.0)
    ap.add_argument("--hp", type=float, default=20.0, help="high-pass cutoff (Hz)")
    ap.add_argument("--k", type=float, default=4.0, help="spike threshold in sigma")
    ap.add_argument("--max-dist", type=float, default=6.0,
                    help="max footprint-center distance (px) to accept a cell match")
    ap.add_argument("--label-a", default="run_a")
    ap.add_argument("--label-b", default="run_b")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[snr] loading {args.label_a} ...")
    ct_a, fc_a, mat_a = load_result(args.a)
    print(f"[snr] loading {args.label_b} ...")
    ct_b, fc_b, mat_b = load_result(args.b)
    print(f"[snr] {args.label_a}: {ct_a.shape[0]} cells x {ct_a.shape[1]} frames  ({mat_a})")
    print(f"[snr] {args.label_b}: {ct_b.shape[0]} cells x {ct_b.shape[1]} frames  ({mat_b})")

    pairs, dists = match_cells(fc_a, fc_b, args.max_dist)
    print(f"[snr] matched {len(pairs)} / min({ct_a.shape[0]},{ct_b.shape[0]}) cells "
          f"(median center dist {np.median(dists):.2f}px)" if len(pairs) else "[snr] NO matches")

    print("[snr] computing per-cell metrics ...")
    ma = per_cell_metrics(ct_a, args.fps, args.hp, args.k)
    mb = per_cell_metrics(ct_b, args.fps, args.hp, args.k)

    print("[snr] paired diagnostics ...")
    pdiag, coh_curves = [], []
    for (i, j) in pairs:
        d, coh = paired_diagnostics(ct_a[i], ct_b[j], args.fps, args.hp)
        pdiag.append(d)
        coh_curves.append(coh)

    # -------- aggregate + paired stats --------
    from scipy.stats import wilcoxon
    metrics = ["noise_sigma", "snr_median", "snr_p90", "snr_max", "spectral_hf_snr", "n_spikes"]
    summary = {"files": {"a": mat_a, "b": mat_b},
               "labels": {"a": args.label_a, "b": args.label_b},
               "params": vars(args),
               "counts": {"a_cells": ct_a.shape[0], "b_cells": ct_b.shape[0],
                          "matched": len(pairs)},
               "aggregate": {}, "paired": {}}
    for m in metrics:
        summary["aggregate"][m] = {args.label_a: _summ([d[m] for d in ma]),
                                   args.label_b: _summ([d[m] for d in mb])}
    # paired test on matched cells (higher = better, except noise_sigma where lower=better)
    for m in metrics:
        a = np.array([ma[i][m] for (i, _j) in pairs], float)
        b = np.array([mb[j][m] for (_i, j) in pairs], float)
        ok = np.isfinite(a) & np.isfinite(b)
        entry = {"n": int(ok.sum()),
                 f"{args.label_a}_median": float(np.median(a[ok])) if ok.any() else None,
                 f"{args.label_b}_median": float(np.median(b[ok])) if ok.any() else None,
                 "a_gt_b_frac": float(np.mean(a[ok] > b[ok])) if ok.any() else None}
        if ok.sum() >= 8 and np.any(a[ok] != b[ok]):
            try:
                w, pval = wilcoxon(a[ok], b[ok])
                entry["wilcoxon_p"] = float(pval)
            except Exception:
                entry["wilcoxon_p"] = None
        summary["paired"][m] = entry
    for key in ("corr_raw", "corr_hp", "coh_hi"):
        summary["paired"][key] = _summ([d[key] for d in pdiag])

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    write_csv(os.path.join(args.out, "per_cell_metrics.csv"), pairs, ma, mb, pdiag, dists)
    _write_text_summary(os.path.join(args.out, "summary.txt"), summary, args)

    if HAVE_MPL and len(pairs):
        print("[snr] figures ...")
        make_figures(args, ct_a, ct_b, pairs, dists, ma, mb, pdiag, coh_curves)
    print(f"[snr] done -> {args.out}/  (summary.txt, summary.json, per_cell_metrics.csv, *.png)")


def _write_text_summary(path, s, args):
    la, lb = args.label_a, args.label_b
    L = []
    L.append("HIGH-FREQUENCY SNR COMPARISON  (run A vs run B)")
    L.append("=" * 60)
    L.append(f"{la:>10} (A): {s['files']['a']}")
    L.append(f"{lb:>10} (B): {s['files']['b']}")
    L.append(f"cells: A={s['counts']['a_cells']}  B={s['counts']['b_cells']}  "
             f"matched={s['counts']['matched']}")
    L.append(f"params: fps={args.fps} high-pass={args.hp}Hz spike_k={args.k}sigma "
             f"match<= {args.max_dist}px")
    L.append("")
    L.append("PAIRED SIMILARITY (matched cells; how alike are the two runs)")
    for k in ("corr_raw", "corr_hp", "coh_hi"):
        d = s["paired"][k]
        L.append(f"  {k:9s}: median={d['median']:.4f}  mean={d['mean']:.4f}  n={d['n']}")
    L.append("")
    L.append(f"HIGH-FREQUENCY SNR METRICS (paired medians; 'A>B frac' = fraction of matched "
             f"cells where A wins)")
    L.append(f"  {'metric':16s} {la+' med':>12s} {lb+' med':>12s} {'A>B frac':>10s} "
             f"{'wilcoxon p':>11s}   note")
    notes = {"noise_sigma": "lower=better (less HF noise)",
             "snr_median": "higher=better",
             "snr_p90": "higher=better",
             "snr_max": "higher=better",
             "spectral_hf_snr": "higher=better (excess HF power over noise floor)",
             "n_spikes": "detected events"}
    for m in ("noise_sigma", "snr_median", "snr_p90", "snr_max", "spectral_hf_snr", "n_spikes"):
        d = s["paired"][m]
        pv = d.get("wilcoxon_p")
        pv_s = f"{pv:.2e}" if isinstance(pv, float) else "n/a"
        L.append(f"  {m:16s} {d[la+'_median']:>12.4g} {d[lb+'_median']:>12.4g} "
                 f"{d['a_gt_b_frac']:>10.3f} {pv_s:>11s}   {notes[m]}")
    L.append("")
    L.append("INTERPRETATION GUIDE")
    L.append("  * corr_hp ~ 1 and coh_hi ~ 1  => the two runs extract the SAME high-freq")
    L.append("    waveform; any SNR gap is then a real algorithmic difference, not noise.")
    L.append("  * A change IMPROVES SNR if: noise_sigma drops AND snr_* rise, while corr_hp stays")
    L.append("    ~1 on cells the change should not touch (no distortion / no fabricated spikes).")
    L.append("  * If A>B frac ~ 0.5 with large wilcoxon p, the two runs are equivalent (no change).")
    L.append("  * Inspect per_cell_metrics.csv rows with lowest corr_hp for cells that changed most.")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def make_figures(args, ct_a, ct_b, pairs, dists, ma, mb, pdiag, coh_curves):
    la, lb = args.label_a, args.label_b
    od = args.out

    # 1) paired SNR scatter (A vs B) for snr_median and spectral_hf_snr and noise_sigma
    for metric, fname in [("snr_median", "scatter_spike_snr.png"),
                          ("spectral_hf_snr", "scatter_spectral_snr.png"),
                          ("noise_sigma", "scatter_noise_sigma.png")]:
        a = np.array([ma[i][metric] for (i, _j) in pairs], float)
        b = np.array([mb[j][metric] for (_i, j) in pairs], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(b[ok], a[ok], s=14, alpha=0.6, edgecolors="k", linewidths=0.3)
        lim = [min(a[ok].min(), b[ok].min()), max(a[ok].max(), b[ok].max())]
        ax.plot(lim, lim, "r--", lw=1, label="y = x")
        ax.set_xlabel(f"{lb}  {metric}")
        ax.set_ylabel(f"{la}  {metric}")
        ax.set_title(f"{metric}: {la} vs {lb} (matched cells)")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(od, fname), dpi=140); plt.close(fig)

    # 2) distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["noise_sigma", "snr_median", "spectral_hf_snr"]):
        pa = np.array([d[metric] for d in ma], float); pa = pa[np.isfinite(pa)]
        pb = np.array([d[metric] for d in mb], float); pb = pb[np.isfinite(pb)]
        ax.boxplot([pa, pb], tick_labels=[la, lb], showfliers=False)
        ax.set_title(metric)
    fig.suptitle("Per-cell metric distributions (all cells)")
    fig.tight_layout(); fig.savefig(os.path.join(od, "distributions.png"), dpi=140); plt.close(fig)

    # 3) mean PSD comparison (log-log)
    def mean_psd(traces):
        Ps = []
        for x in traces:
            f, P = welch(x - x.mean(), fs=args.fps, nperseg=int(min(4096, len(x))), window="hann")
            Ps.append(P)
        return f, np.mean(Ps, axis=0)
    fa, Pa = mean_psd(ct_a)
    fb, Pb = mean_psd(ct_b)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(fa[1:], Pa[1:], label=la, lw=1.4)
    ax.loglog(fb[1:], Pb[1:], label=lb, lw=1.4)
    ax.axvline(args.hp, color="gray", ls=":", lw=1, label=f"high-pass {args.hp} Hz")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Mean PSD (a.u.)")
    ax.set_title("Mean power spectral density across cells")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(od, "mean_psd.png"), dpi=140); plt.close(fig)

    # 4) mean coherence (A vs B) across matched cells
    if coh_curves:
        f = coh_curves[0][0]
        C = np.mean([c[1] for c in coh_curves], axis=0)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(f, C, lw=1.5)
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(f"mean coherence  ({la} vs {lb})")
        ax.set_ylim(0, 1.02)
        ax.set_title("Cross-run coherence of matched high-pass traces\n"
                     "(1 = identical HF content; drop = divergence)")
        fig.tight_layout(); fig.savefig(os.path.join(od, "coherence.png"), dpi=140); plt.close(fig)

    # 5) example overlay: best-matched, high-SNR cell
    score = np.array([pdiag[n]["corr_hp"] * (ma[pairs[n][0]]["snr_max"] or 0)
                      for n in range(len(pairs))], float)
    if np.any(np.isfinite(score)):
        n = int(np.nanargmax(score))
        i, j = pairs[n]
        xa, xb = ct_a[i], ct_b[j]
        t = np.arange(len(xa)) / args.fps
        hp_a = highpass(xa, args.hp, args.fps); hp_b = highpass(xb, args.hp, args.fps)
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=False)
        axes[0].plot(t, xa, lw=0.6, label=la); axes[0].plot(t, xb, lw=0.6, alpha=0.7, label=lb)
        axes[0].set_title(f"matched cell (A#{i}, B#{j}, corr_hp={pdiag[n]['corr_hp']:.3f}) "
                          f"- raw traces"); axes[0].legend()
        axes[1].plot(t, hp_a, lw=0.6, label=la); axes[1].plot(t, hp_b, lw=0.6, alpha=0.7, label=lb)
        axes[1].set_title(f"high-pass > {args.hp} Hz"); axes[1].legend()
        pk = int(np.argmax(hp_a))
        lo, hi = max(0, pk - 200), min(len(xa), pk + 200)
        axes[2].plot(t[lo:hi], hp_a[lo:hi], lw=1.0, marker=".", ms=2, label=la)
        axes[2].plot(t[lo:hi], hp_b[lo:hi], lw=1.0, marker=".", ms=2, alpha=0.7, label=lb)
        axes[2].set_title("zoom on largest high-pass event"); axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        fig.tight_layout(); fig.savefig(os.path.join(od, "example_matched_cell.png"), dpi=140)
        plt.close(fig)

    # 6) histogram of paired high-pass correlations
    r = np.array([d["corr_hp"] for d in pdiag], float); r = r[np.isfinite(r)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=40)
    ax.set_xlabel(f"high-pass trace correlation ({la} vs {lb})"); ax.set_ylabel("matched cells")
    ax.set_title(f"Cross-run agreement (median r={np.median(r):.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(od, "corr_hist.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
