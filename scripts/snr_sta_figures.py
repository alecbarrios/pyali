#!/usr/bin/env python3
"""Aggregate and plot the spike-triggered averages -- README section 10f, stage 3b.

Reads every ``sta.csv`` written by :mod:`snr_sta`, plus ``fov_summary.csv`` for the metadata, and
writes into ``snr_summary/``:

* ``sta_by_fov.parquet`` -- every FOV's curve with day / plate / well / burst
* ``sta_by_well.csv``    -- per well, aggregated over the per-FOV curves
* four figures (light, 450 dpi)

**Two figure types, answering different questions.**

* ``sta_fov_grid`` -- one **representative FOV per well**; the band is the spread **across cells**,
  i.e. cell-to-cell waveform heterogeneity.
* ``sta_by_well`` -- the per-FOV curves aggregated per well; the band is the spread **across FOVs**,
  i.e. how reproducible the waveform is between fields.

Each comes in a median+IQR version and a mean+/-SEM version.

Amplitudes are in units of each cell's own ``noise_sigma``, which is what makes days comparable --
absolute trace units are not, since bit depth, frame size and footprint weighting all differ.

    python scripts/snr_sta_figures.py
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.ticker import MaxNLocator                                # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from snr_aggregate import OUT_DEFAULT                                     # noqa: E402
from snr_corpus import META, REQUIRED, list_fovs, S3_DEFAULT             # noqa: E402
from snr_corpus import READ_ROOT_DEFAULT                                 # noqa: E402
from snr_figures import (DPI_DEFAULT, THEME, _color_of, day_label,        # noqa: E402
                         legend, style)
from snr_sta import STA                                                  # noqa: E402

# Restrict the representative FOV to the interior of each well's burst sequence. Firing rate at the
# first/last 8% of bursts is 0.42x the interior (README section 9) because those bursts image partly
# outside the well -- so an edge FOV would show acquisition geometry, not biology.
INTERIOR = (0.2, 0.8)


def read_sta(args):
    day, dir_name, read_root = args
    try:
        d = pd.read_csv(os.path.join(read_root, day, dir_name, STA),
                        float_precision="round_trip")
    except Exception as e:                                   # noqa: BLE001
        return dir_name, None, f"{type(e).__name__}: {e}"
    if not len(d):
        return dir_name, None, None                          # header-only: no qualifying cell
    d["day"] = day
    d["dir_name"] = dir_name
    return dir_name, d, None


def draw_band(ax, t, lo, mid, hi, color, th, label=None):
    ax.axhline(0.0, color=th["grid"], linewidth=0.6, zorder=1)
    ax.axvline(0.0, color=th["grid"], linewidth=0.6, zorder=1)
    ax.fill_between(t, lo, hi, color=color, alpha=0.30, linewidth=0, zorder=2)
    ax.plot(t, mid, "-", color=color, linewidth=2.0, zorder=3, label=label)


def grid_dpi(n_panels, base=DPI_DEFAULT, floor=600, cap=900):
    """Resolution for a small-multiples grid, rising with panel count.

    Panel size on the page is fixed, so a 27-panel grid shows each panel small when the figure is
    fitted to a screen -- the only way to zoom into one during a talk is to have real pixels there.
    Scaling as sqrt(n_panels) keeps per-panel resolution climbing without the total pixel count
    running away: at 27 panels this yields ~2600 x 1935 px per panel.
    """
    return int(min(cap, max(floor, base * (n_panels / 6.0) ** 0.5)))


def grid_figure(panels, out_dir, stem, title, band_note, labels, days, dpi=None, ncols=5):
    """One small-multiple panel per entry in ``panels`` = [(label, day, t, lo, mid, hi, n)]."""
    th = THEME["light"]
    color_of = _color_of(days, th)
    dpi = dpi or grid_dpi(len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols + 1.0, 2.15 * nrows + 1.8),
                             facecolor=th["surface"], sharex=True, sharey=True)
    axf = np.atleast_1d(axes).ravel()
    for ax, (lab, day, t, lo, mid, hi, n) in zip(axf, panels):
        draw_band(ax, t, lo, mid, hi, color_of(day), th)
        ax.set_title(f"{lab}  (n={n})", color=th["primary"], fontsize=8.5, loc="left", pad=4)
        style(ax, th)
    for ax in axf[len(panels):]:
        ax.set_visible(False)
    grid = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in grid[:, 0]:
        ax.set_ylabel("amplitude (sigma)", color=th["secondary"], fontsize=8)
    for c in range(ncols):
        used = [r for r in range(nrows) if r * ncols + c < len(panels)]
        if used:
            ax = grid[used[-1], c]
            ax.set_xlabel("time from spike peak (ms)", color=th["secondary"], fontsize=7.5)
            ax.tick_params(labelbottom=True)
    # Clamp x to the real window and place ticks inside it. Autoscale padded out to +/-30 ms on a
    # +/-25 ms window, putting labels hard against the panel edge where the neighbouring column's
    # first label sits -- they collided.
    t0 = min(float(p[2].min()) for p in panels)
    t1 = max(float(p[2].max()) for p in panels)
    axf[0].set_xlim(t0, t1)
    step = 10.0 if (t1 - t0) <= 80 else 20.0
    cand = np.arange(np.ceil(t0 / step) * step, t1 + 1e-9, step)
    margin = 0.06 * (t1 - t0)
    axf[0].set_xticks(cand[(cand - t0 > margin) & (t1 - cand > margin)])
    # prune="both" drops the outermost y labels. Rows stack vertically on a shared axis, so the
    # bottom label of one panel otherwise lands on the top label of the panel beneath it -- which
    # it did, 8 times. Amplitude genuinely goes negative (the undershoot), so unlike firing rate
    # this cannot be fixed by clamping the limit to zero.
    axf[0].yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    fig.suptitle(title, color=th["primary"], fontsize=12.5, x=0.008, ha="left", y=0.997)
    fig.text(0.008, 0.962, band_note, color=th["secondary"], fontsize=8.5, ha="left")
    legend(fig, th, days, labels, color_of)
    fig.tight_layout(rect=(0, 0.035, 1, 0.952), h_pad=1.9, w_pad=1.4)
    w, h = fig.get_size_inches()
    fig.savefig(os.path.join(out_dir, f"{stem}__light.png"), dpi=dpi, facecolor=th["surface"])
    plt.close(fig)
    return dpi, int(w * dpi), int(h * dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-prefix", default=S3_DEFAULT)
    ap.add_argument("--read-root", default=READ_ROOT_DEFAULT)
    ap.add_argument("--summary-dir", default=OUT_DEFAULT)
    ap.add_argument("--out-dir", default=None, help="defaults to <summary-dir>/figures")
    ap.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--dpi", type=int, default=None,
                    help="override the automatic per-grid resolution")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.summary_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    fovs = list_fovs(a.s3_prefix)
    keys = sorted(k for k, v in fovs.items() if set(REQUIRED).issubset(v) and STA in v)
    if a.days:
        keys = [k for k in keys if k[0] in a.days]
    print(f"[sta-fig] {len(keys)} FOVs with {STA}", flush=True)

    frames, empty, errors = [], 0, []
    with ThreadPoolExecutor(a.threads) as ex:
        for dir_name, d, err in ex.map(read_sta, [(x, y, a.read_root) for x, y in keys]):
            if err:
                errors.append((dir_name, err))
            elif d is None:
                empty += 1
            else:
                frames.append(d)
    for n, e in errors:
        print(f"[sta-fig] READ FAIL {n}: {e}", flush=True)
    sta = pd.concat(frames, ignore_index=True)
    print(f"[sta-fig] {len(frames)} curves ({empty} FOVs had no qualifying cell)", flush=True)

    meta = pd.read_csv(os.path.join(a.summary_dir, "fov_summary.csv"),
                       float_precision="round_trip")
    keep = ["day", "dir_name", "plate", "plate_num", "well", "burst", "n_cells", "bit", "dims"]
    sta = sta.merge(meta[keep], on=["day", "dir_name"], how="left", suffixes=("", "_fov"))
    sta.to_parquet(os.path.join(a.summary_dir, "sta_by_fov.parquet"), index=False)

    days = sorted(sta["day"].unique())
    labels = {d: day_label(d, g) for d, g in meta.groupby("day")}

    # ---- per well, aggregated over FOV curves (band = across FOVs) ----
    rows = []
    for (day, plate, pn, wl), g in sta.groupby(["day", "plate", "plate_num", "well"]):
        p = g.pivot_table(index="t_ms", columns="dir_name", values="median").sort_index()
        pm = g.pivot_table(index="t_ms", columns="dir_name", values="mean").sort_index()
        A, B = p.values, pm.values
        n = A.shape[1]
        rows.append(pd.DataFrame({
            "day": day, "plate": plate, "plate_num": pn, "well": wl, "n_fovs": n,
            "t_ms": p.index.values,
            "median": np.median(A, axis=1),
            "q25": np.percentile(A, 25, axis=1), "q75": np.percentile(A, 75, axis=1),
            "mean": B.mean(axis=1),
            "sem": B.std(axis=1, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(B.shape[0]),
        }))
    wells = pd.concat(rows, ignore_index=True)
    wells.to_csv(os.path.join(a.summary_dir, "sta_by_well.csv"), index=False)

    # ---- one representative interior FOV per well ----
    m = meta.copy()
    m["bw"] = m["burst"] - m.groupby(["day", "plate_num", "well"])["burst"].transform("min")
    span = m.groupby(["day", "plate_num", "well"])["bw"].transform("max").replace(0, np.nan)
    m["frac"] = m["bw"] / span
    have = set(zip(sta["day"], sta["dir_name"]))
    reps = []
    for (day, plate, pn, wl), g in m.groupby(["day", "plate", "plate_num", "well"]):
        g = g[[(d, n) in have for d, n in zip(g["day"], g["dir_name"])]]
        inner = g[(g["frac"] >= INTERIOR[0]) & (g["frac"] <= INTERIOR[1])]
        pick_from = inner if len(inner) else g
        if not len(pick_from):
            continue
        target = pick_from["n_cells"].median()
        r = pick_from.iloc[(pick_from["n_cells"] - target).abs().argsort().iloc[0]]
        reps.append((day, plate, wl, r["dir_name"], int(r["n_cells"])))

    def panel_rows(kind):
        out = []
        for day, plate, wl, dir_name, ncell in reps:
            g = sta[(sta["day"] == day) & (sta["dir_name"] == dir_name)].sort_values("t_ms")
            t = g["t_ms"].values
            if kind == "iqr":
                out.append((f"{plate}/{wl}", day, t, g["q25"].values, g["median"].values,
                            g["q75"].values, ncell))
            else:
                mu, sd = g["mean"].values, g["sd"].values
                sem = sd / max(np.sqrt(ncell), 1.0)
                out.append((f"{plate}/{wl}", day, t, mu - sem, mu, mu + sem, ncell))
        return out

    def well_rows(kind):
        out = []
        for (day, plate, _pn, wl), g in wells.groupby(["day", "plate", "plate_num", "well"]):
            g = g.sort_values("t_ms"); t = g["t_ms"].values; n = int(g["n_fovs"].iloc[0])
            if kind == "iqr":
                out.append((f"{plate}/{wl}", day, t, g["q25"].values, g["median"].values,
                            g["q75"].values, n))
            else:
                mu, se = g["mean"].values, g["sem"].values
                out.append((f"{plate}/{wl}", day, t, mu - se, mu, mu + se, n))
        return out

    jobs = [
        (panel_rows("iqr"), "sta_fov_grid", "Spike-triggered average, representative FOV per well",
         "line = median across cells, band = IQR across cells (cell-to-cell heterogeneity); "
         "n = cells. Interior bursts only.", "cells"),
        (panel_rows("sem"), "sta_fov_grid_sem", "Spike-triggered average, representative FOV per well",
         "line = mean across cells, band = +/- SEM across cells; n = cells. Interior bursts only.",
         "cells"),
        (well_rows("iqr"), "sta_by_well", "Spike-triggered average by well",
         "line = median of per-FOV curves, band = IQR across FOVs (field-to-field "
         "reproducibility); n = FOVs.", "fovs"),
        (well_rows("sem"), "sta_by_well_sem", "Spike-triggered average by well",
         "line = mean of per-FOV curves, band = +/- SEM across FOVs; n = FOVs.", "fovs"),
    ]
    for panels, stem, title, note, _unit in jobs:
        d, px, py = grid_figure(panels, out_dir, stem, title, note, labels, days, a.dpi)
        mb = os.path.getsize(os.path.join(out_dir, f"{stem}__light.png")) / 1e6
        print(f"[sta-fig] {stem}__light.png  {len(panels)} panels  {d} dpi  "
              f"{px}x{py} px  {mb:.1f} MB", flush=True)

    pk = sta.groupby("day")["median"].max()
    print(f"\n[sta-fig] {a.summary_dir}")
    print("[sta-fig] peak of per-FOV median curve, by day (sigma):")
    for d, v in pk.items():
        print(f"            {d:<16} {v:.2f}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
