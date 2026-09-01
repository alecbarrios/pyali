#!/usr/bin/env python3
"""Representative single-cell traces at the 5th / 50th / 95th percentile of each SNR metric.

One PNG per cell, for a slide. For each day and each of three metrics, the cell whose value sits
closest to the 5th, 50th and 95th percentile is plotted as a plain trace -- raw ``cell_traces`` in
a.u. against time in seconds -- with all three metrics annotated on the figure.

Metric names on the figures deliberately use the **presentation terminology**, not the column names:

| figure label | column | definition |
|---|---|---|
| Spike Amplitude SNR | ``snr_median`` | median detected-spike height / noise floor |
| Band-Limited SNR | ``spectral_hf_snr`` | excess 20-150 Hz PSD power over the 300-400 Hz floor |
| Firing Rate (Hz) | derived | ``n_spikes / (n_frames_analyzed / fps)`` |

Cells are drawn only from **active wells** -- those whose median decay ratio is >= 0.10 (README
section 11.2). The quiescent wells' detections are dominated by threshold noise, so their
percentiles would describe the detector rather than the cells.

    python scripts/representative_traces.py
    python scripts/representative_traces.py --days 20260715

Reads nothing but ``cells_all.parquet``, ``sta_cell_features.parquet`` and the one
``ALI_Result.mat`` per selected FOV. Changes no existing output.
"""
import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from pyali.io import load_v73                                            # noqa: E402
from snr_aggregate import OUT_DEFAULT                                    # noqa: E402
from snr_corpus import READ_ROOT_DEFAULT, RESULT                         # noqa: E402
from snr_figures import FONT_STACK, THEME, style                         # noqa: E402

OUT_DIR_DEFAULT = ("/home/jovyan/spatial-technology-platform/AB/pyali_3fc51c7_outputs/"
                   "representative_traces")

# (column, figure label, filename stem, value format)
METRICS = [
    ("snr_median", "Spike Amplitude SNR", "spike_amplitude_snr", "{:.2f}"),
    ("spectral_hf_snr", "Band-Limited SNR", "band_limited_snr", "{:.2f}"),
    ("firing_rate", "Firing Rate (Hz)", "firing_rate", "{:.2f}"),
]
PCTS = (5, 50, 95)
JOINT_PCT = 95        # the 'high on all three metrics at once' selection
DECAY_MIN = 0.10          # well-level QC gate; see README section 11.2


def active_wells(summary_dir):
    """(day, plate, well) whose median per-cell decay ratio clears the QC gate."""
    f = pd.read_parquet(os.path.join(summary_dir, "sta_cell_features.parquet"),
                        columns=["day", "plate", "well", "decay"])
    med = f.groupby(["day", "plate", "well"])["decay"].median()
    return set(med[med >= DECAY_MIN].index), med


def pick(df, col, pct):
    """The row whose ``col`` is closest to the ``pct`` percentile of ``col``."""
    v = df[col].to_numpy(float)
    target = float(np.nanpercentile(v, pct))
    j = int(np.nanargmin(np.abs(v - target)))
    return df.iloc[j], target


def plot_trace(tr, fps, info, out_path, dpi):
    """Plain trace: raw amplitude in a.u. against time in seconds, metrics annotated."""
    th = THEME["light"]
    t = np.arange(tr.size) / float(fps)
    # Wide and tall so the slow envelope and the individual spikes are both legible; the whole
    # point is that dynamics are not squished.
    fig, ax = plt.subplots(1, 1, figsize=(19.0, 6.4), facecolor=th["surface"])
    ax.plot(t, tr, "-", color=th["series"][0], linewidth=0.5, zorder=3)
    style(ax, th)
    ax.grid(axis="both", color=th["grid"], linewidth=0.5, alpha=0.85)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time (s)", color=th["secondary"], fontsize=13)
    ax.set_ylabel("trace (a.u.)", color=th["secondary"], fontsize=13)
    ax.tick_params(colors=th["secondary"], labelsize=11)

    ax.set_title(info["title"], color=th["primary"], fontsize=15.5, loc="left", pad=12)
    txt = "\n".join([
        f"Spike Amplitude SNR = {info['snr_median']:.2f}",
        f"Band-Limited SNR = {info['spectral_hf_snr']:.2f}",
        f"Firing Rate = {info['firing_rate']:.2f} Hz",
    ])
    ax.text(0.012, 0.975, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=13.5, color=th["primary"], linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=th["surface"],
                      edgecolor=th["grid"], linewidth=0.9, alpha=0.94), zorder=6)
    ax.text(0.012, 0.035, info["sub"], transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, color=th["secondary"], zorder=6)
    fig.tight_layout(rect=(0, 0.01, 1, 0.99))
    fig.savefig(out_path, dpi=dpi, facecolor=th["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", default=OUT_DEFAULT)
    ap.add_argument("--read-root", default=READ_ROOT_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = FONT_STACK
    os.makedirs(a.out_dir, exist_ok=True)

    keep, med = active_wells(a.summary_dir)
    cells = pd.read_parquet(
        os.path.join(a.summary_dir, "cells_all.parquet"),
        columns=["day", "dir_name", "cell_index", "plate", "well", "burst", "noise_sigma",
                 "snr_median", "spectral_hf_snr", "n_spikes", "fps", "n_frames_analyzed"])
    cells["firing_rate"] = cells["n_spikes"] / (cells["n_frames_analyzed"] / cells["fps"])
    cells["_wk"] = list(zip(cells["day"], cells["plate"], cells["well"]))
    elig = cells[cells["_wk"].isin(keep) & (cells["n_spikes"] > 0)].copy()
    days = sorted(a.days) if a.days else sorted(elig["day"].unique())
    print(f"[trace] {len(keep)} active wells of {len(med)}; "
          f"{len(elig)} eligible cells (active well, n_spikes>0)", flush=True)
    for d in days:
        w = sorted({f"{p}/{wl}" for (dd, p, wl) in keep if dd == d})
        print(f"        {d:<16} wells used: {', '.join(w)}", flush=True)

    # choose first, then read each FOV once
    sel, need = [], defaultdict(list)
    for day in days:
        g = elig[elig["day"] == day]
        if not len(g):
            continue
        for col, label, stem, _fmt in METRICS:
            gg = g.dropna(subset=[col])
            for pct in PCTS:
                row, target = pick(gg, col, pct)
                item = dict(day=day, dir_name=row["dir_name"], cell_index=int(row["cell_index"]),
                            plate=row["plate"], well=row["well"], burst=row["burst"],
                            fps=float(row["fps"]), metric=label, stem=stem, pct=pct,
                            target=target, noise_sigma=float(row["noise_sigma"]),
                            snr_median=float(row["snr_median"]),
                            spectral_hf_snr=float(row["spectral_hf_snr"]),
                            firing_rate=float(row["firing_rate"]),
                            n_spikes=int(row["n_spikes"]))
                sel.append(item)
                need[(day, row["dir_name"])].append(len(sel) - 1)
    # One extra per day: the cell that clears the 95th percentile on ALL THREE metrics at once.
    # Among those, take the LEAST extreme -- closest to (p95, p95, p95) in rank space -- so it
    # represents that corner rather than being the single most extreme cell in the day.
    for day in days:
        g = elig[elig["day"] == day].dropna(subset=["snr_median", "spectral_hf_snr"])
        if not len(g):
            continue
        cols = ["snr_median", "spectral_hf_snr", "firing_rate"]
        r = pd.DataFrame({m: g[m].rank(pct=True) * 100 for m in cols}, index=g.index)
        ok = r.min(axis=1) >= JOINT_PCT
        if not ok.any():
            print(f"[trace] {day}: no cell clears p{JOINT_PCT} on all three", flush=True)
            continue
        dist = ((r[ok] - JOINT_PCT) ** 2).sum(axis=1)
        row = g.loc[dist.idxmin()]
        rr = r.loc[dist.idxmin()]
        item = dict(day=day, dir_name=row["dir_name"], cell_index=int(row["cell_index"]),
                    plate=row["plate"], well=row["well"], burst=row["burst"],
                    fps=float(row["fps"]), metric="All three metrics", stem="all_three",
                    pct=JOINT_PCT, target=float(JOINT_PCT),
                    noise_sigma=float(row["noise_sigma"]),
                    snr_median=float(row["snr_median"]),
                    spectral_hf_snr=float(row["spectral_hf_snr"]),
                    firing_rate=float(row["firing_rate"]), n_spikes=int(row["n_spikes"]),
                    ranks=f"ranks p{rr['snr_median']:.1f} / p{rr['spectral_hf_snr']:.1f} / "
                           f"p{rr['firing_rate']:.1f}",
                    n_qualifying=int(ok.sum()))
        sel.append(item)
        need[(day, row["dir_name"])].append(len(sel) - 1)

    print(f"[trace] {len(sel)} traces from {len(need)} distinct FOVs", flush=True)

    made = 0
    for (day, dir_name), idxs in need.items():
        path = os.path.join(a.read_root, day, dir_name, RESULT)
        try:
            ct = np.asarray(load_v73(path, "cell_traces"))
        except Exception as e:                                # noqa: BLE001
            print(f"[trace] READ FAIL {day}/{dir_name}: {e}", flush=True)
            continue
        for i in idxs:
            it = sel[i]
            ci = it["cell_index"]
            if ci >= ct.shape[0]:
                print(f"[trace] cell {ci} out of range in {dir_name}", flush=True)
                continue
            burst = "" if pd.isna(it["burst"]) else f" burst {int(it['burst'])}"
            it["title"] = (f"{it['metric']} — {it['pct']}th percentile   |   {day}  "
                           f"{it['plate']}/{it['well']}{burst}  cell {ci}")
            tail = (f"{it['ranks']}   ·   {it['n_qualifying']} cells in this day clear "
                    f"p{JOINT_PCT} on all three" if "ranks" in it
                    else f"{it['metric']} {it['pct']}th pct target = {it['target']:.2f}")
            it["sub"] = (f"day {day} {it['plate']}/{it['well']}{burst}, cell {ci}   ·   "
                         f"spikes = {it['n_spikes']}   ·   noise_sigma = "
                         f"{it['noise_sigma']:.4f}   ·   {tail}")
            fn = f"trace_{day}_{it['stem']}_p{it['pct']:02d}.png"
            plot_trace(np.asarray(ct[ci], float), it["fps"], it, os.path.join(a.out_dir, fn),
                       a.dpi)
            made += 1
            print(f"[trace] {fn}   {it['metric']} = {it['target']:.2f} (target)  "
                  f"actual: spike {it['snr_median']:.2f}  band {it['spectral_hf_snr']:.2f}  "
                  f"rate {it['firing_rate']:.2f} Hz", flush=True)
    print(f"\n[trace] {made} PNGs -> {a.out_dir}", flush=True)
    pd.DataFrame(sel).drop(columns=["title", "sub"], errors="ignore").to_csv(
        os.path.join(a.out_dir, "selected_cells.csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
