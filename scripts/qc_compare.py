#!/usr/bin/env python3
"""Before / after QC around chosen cells: does a lumped trace come apart into several cells?

    python scripts/qc_compare.py <before_dir> <after_dir> --cells C0713,C0316,C0331,C1216 [--n 6]
                                 [--radius 10] [--crop 32] [--label before]

For every *before* cell (the named ones, then ``--n`` picks across the before result's
spike-amplitude-SNR percentiles), one row:

* before map  -- the reference crop with the before cell's footprint contour and spike COMs (yellow)
  and the other spikes in the crop (gray; ``x`` merged duplicates, ``o`` in no cell);
* before trace -- its full trace, spike ticks on top;
* after map   -- the same crop with every *after* cell whose centre lies within ``--radius`` px of
  the before cell: one colour per cell for its spike COMs and footprint contour;
* after traces -- those cells' traces stacked in the same colours on the same time axis (each
  scaled to its own robust sigma), spike ticks per cell.

A lumped before trace that splits shows several coloured clouds and several traces whose spikes
interleave; a clean cell shows one cloud and one trace that matches the before trace.

Writes ``<after_dir>/QC/compare_<label>.png`` and ``.csv`` (before cell -> after cells with their
distance, spike count and snr_median).
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from pyali.results import load_results                                   # noqa: E402
from qc_footprints import (OWN_COLOUR, NEIGHBOUR_COLOURS, QC_DIR, cell_table, crop_window,  # noqa: E402
                           draw_spike_map, pick_named, select_cells)

CSV_COLUMNS = ["before_cell", "before_snr", "before_spikes", "after_cell", "distance_px",
               "after_snr", "after_spikes", "after_source"]


def after_cells_near(after, centre_rc, radius):
    """Trace rows of ``after`` cells within ``radius`` px of ``centre_rc`` (1-indexed), nearest
    first, with their distances."""
    d = np.hypot(after.centers[:, 0] - centre_rc[0], after.centers[:, 1] - centre_rc[1])
    rows = np.flatnonzero(d <= radius)
    rows = rows[np.argsort(d[rows], kind="stable")]
    return [int(k) for k in rows], d[rows]


def _robust_scale(x):
    from pyali.metrics import highpass, robust_sigma
    s = robust_sigma(highpass(np.asarray(x, float), 20.0, 800.0)) or 1.0
    return (x - np.median(x)) / s


def render(before, after, rows_b, out_png, after_snr, radius=10.0, crop=32, max_after=6, dpi=170,
           header=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = float(before.attrs["fps"])
    H, W = before.footprints.height, before.footprints.width
    n = max(len(rows_b), 1)
    fig_h = 2.3 * n + 1.0
    fig, axes = plt.subplots(n, 4, figsize=(21, fig_h), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 3.3, 1, 3.3], wspace=0.1, hspace=0.4))
    fig.subplots_adjust(top=1 - 0.8 / fig_h, bottom=0.4 / fig_h, left=0.07, right=0.99)
    t_b = np.arange(before.traces.shape[1]) / fps
    t_a = np.arange(after.traces.shape[1]) / fps
    records = []
    for i, (_, row) in enumerate(rows_b.iterrows()):
        k = int(row["trace_row"])
        ax_bm, ax_bt, ax_am, ax_at = axes[i]
        r0, c0 = crop_window(before.centers[k], H, W, crop)
        S = int(min(crop, H, W))
        draw_spike_map(ax_bm, before, [k], r0, c0, S, neighbours=[],
                       title=f"before {row['cell_id'].split('_')[-1]}")
        tag = f"P{row['percentile']:.0f}" if np.isfinite(row["percentile"]) else "named"
        ax_bm.set_ylabel(f"{tag}\nSNR {row['snr_median']:.1f}\nspikes {int(row['n_spikes'])}",
                         rotation=0, ha="right", va="center", fontsize=8, labelpad=6)
        tr = _robust_scale(before.traces[k])
        ax_bt.plot(t_b, tr, "-", color="#1f3b73", linewidth=0.4)
        fr = before.spikes.loc[before.spikes["cell_row"] == k, "frame"].to_numpy()
        top = np.percentile(tr, 99.9)
        ax_bt.vlines(fr / fps, top * 1.08, top * 1.25, color=OWN_COLOUR, linewidth=0.6)
        ax_bt.set_xlim(0, t_b[-1])
        ax_bt.set_title("before trace (robust-sigma units)", fontsize=8, loc="left")
        ax_bt.tick_params(labelsize=7)

        near, dist = after_cells_near(after, before.centers[k], radius)
        near, dist = near[:max_after], dist[:max_after]
        colours = {kk: NEIGHBOUR_COLOURS[j % len(NEIGHBOUR_COLOURS)] for j, kk in enumerate(near)}
        draw_spike_map(ax_am, after, near, r0, c0, S, neighbours=[], colours=colours,
                       title=f"after: {len(near)} cell(s) within {radius:g} px")
        offset = 0.0
        a_tab = after.cells
        for kk, dd in zip(near, dist):
            x = _robust_scale(after.traces[kk])
            span = np.percentile(x, 99.9) - np.percentile(x, 0.1)
            ax_at.plot(t_a, x - offset, "-", color=colours[kk], linewidth=0.4)
            frk = after.spikes.loc[after.spikes["cell_row"] == kk, "frame"].to_numpy()
            hi = np.percentile(x, 99.9) - offset
            ax_at.vlines(frk / fps, hi + 0.05 * span, hi + 0.18 * span, color=colours[kk], linewidth=0.6)
            ax_at.text(t_a[-1], np.median(x) - offset, f" {a_tab['cell_id'].iloc[kk].split('_')[-1]}",
                       fontsize=7, color=colours[kk], va="center")
            records.append(dict(before_cell=row["cell_id"], before_snr=float(row["snr_median"]),
                                before_spikes=int(row["n_spikes"]),
                                after_cell=a_tab["cell_id"].iloc[kk], distance_px=float(dd),
                                after_snr=float(after_snr[kk]), after_spikes=int(a_tab["n_spikes"].iloc[kk]),
                                after_source=a_tab["source"].iloc[kk]))
            offset += span * 1.3
        if not near:
            ax_at.text(0.5, 0.5, "no after cell nearby", ha="center", va="center", fontsize=8,
                       transform=ax_at.transAxes)
            records.append(dict(before_cell=row["cell_id"], before_snr=float(row["snr_median"]),
                                before_spikes=int(row["n_spikes"]), after_cell="", distance_px=np.nan,
                                after_snr=np.nan, after_spikes=0, after_source=""))
        ax_at.set_xlim(0, t_a[-1] * 1.06)
        ax_at.set_yticks([])
        ax_at.set_title("after traces (each in robust-sigma units, stacked)", fontsize=8, loc="left")
        ax_at.tick_params(labelsize=7)
        if i == n - 1:
            ax_bt.set_xlabel("time (s)", fontsize=8)
            ax_at.set_xlabel("time (s)", fontsize=8)
    fig.suptitle(header, fontsize=9, y=1 - 0.08 / fig_h)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return records


def run(before_dir, after_dir, cells=None, n=6, radius=10.0, crop=32, label="before", dpi=170,
        min_spikes=3):
    import pandas as pd
    before, after = load_results(before_dir), load_results(after_dir)
    db = cell_table(before)
    da = cell_table(after)
    after_snr = da["snr_median"].to_numpy(float)
    parts = []
    if cells:
        parts.append(pick_named(db, cells))
    if n:
        parts.append(select_cells(db, n=n, min_spikes=min_spikes))
    rows_b = pd.concat(parts, ignore_index=True).drop_duplicates("cell_id") if parts else db.iloc[:0]
    header = (f"before: {before.attrs.get('fov_id', '')} ({len(db)} cells, pyali "
              f"{str(before.attrs.get('pyali_git', ''))[:7]})   after: {len(da)} cells, pyali "
              f"{str(after.attrs.get('pyali_git', ''))[:7]}\n"
              "maps: yellow = the before cell's spike COMs; colours = after cells within the radius "
              "(dots + footprint contour); gray x = merged same-frame duplicate, gray o = in no cell")
    out_dir = os.path.join(after_dir, QC_DIR)
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"compare_{label}.png")
    records = render(before, after, rows_b, png, after_snr, radius=radius, crop=crop, dpi=dpi,
                     header=header)
    tab = pd.DataFrame(records, columns=CSV_COLUMNS)
    tab.to_csv(os.path.join(out_dir, f"compare_{label}.csv"), index=False)
    print(f"[qc_compare] {len(rows_b)} before cells -> {png}")
    for cid, g in tab.groupby("before_cell", sort=False):
        print(f"  {cid.split('_')[-1]}: {len(g[g['after_cell'] != ''])} after cell(s) "
              + ", ".join(f"{a.split('_')[-1]} ({d:.1f} px, {s} spk)" for a, d, s in
                          zip(g["after_cell"], g["distance_px"], g["after_spikes"]) if a))
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before_dir")
    ap.add_argument("after_dir")
    ap.add_argument("--cells", default="", help="comma-separated before cell ids / C0713 suffixes")
    ap.add_argument("--n", type=int, default=6, help="extra before cells across the SNR percentiles")
    ap.add_argument("--radius", type=float, default=10.0, help="after cells within this many px")
    ap.add_argument("--crop", type=int, default=32)
    ap.add_argument("--label", default="before", help="output name: QC/compare_<label>.png")
    ap.add_argument("--dpi", type=int, default=170)
    a = ap.parse_args(argv)
    cells = [c for c in a.cells.split(",") if c.strip()]
    run(a.before_dir, a.after_dir, cells, a.n, a.radius, a.crop, a.label, a.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
