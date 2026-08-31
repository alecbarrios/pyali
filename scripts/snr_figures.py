#!/usr/bin/env python3
"""Violin figures over the SNR summary tables -- README section 9, stage 3 of 3.

Violins of ``noise_sigma``, ``snr_median`` and ``spectral_hf_snr`` at **well, plate, day and
whole-set** level, plus cell-count distributions. Each violin is a distribution over **per-FOV
medians** -- one point is one FOV -- which is what section 9's "summarize at FOV level first"
rule means for a figure: a 1240-cell FOV must not outweigh an 8-cell one inside the same well.
Cell-count panels are the exception; there the per-FOV count *is* the quantity.

    python scripts/snr_figures.py
    python scripts/snr_figures.py --summary-dir /tmp/snr

Reads only what :mod:`snr_aggregate` wrote; re-run it after a refresh to redraw.

Design notes, so the choices are not re-litigated later:

* **Color encodes the acquisition day**, never rank or position -- days differ in bit depth and
  frame size, which is exactly what drives these metrics, so the legend carries both (section 9).
  Hues are the first three categorical slots, assigned in fixed order and never cycled.
* Both palettes were checked with the data-viz validator (``--pairs all``, since a reader compares
  any two violins, not just neighbours): light passes every gate with worst-pair CVD dE 9.2 and
  normal-vision dE 24.0; dark passes with 9.4 / 20.9. Aqua sits at 2.74:1 on the light surface,
  below the 3:1 bar, so the **relief rule** applies -- every violin carries a visible ``n`` label
  and the summary CSVs are the table view, so identity is never colour-alone.
* Dark mode is a **selected** second render against the dark surface, not an automatic inversion;
  ``index.html`` swaps on ``prefers-color-scheme``.
* Titles are nouns -- "Noise floor", "Spike SNR" -- never a sentence.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

# Arial first, Calibri second, then metric-compatible substitutes. matplotlib walks this list and
# takes the first family actually installed, so a machine with Arial renders Arial while this one
# falls through to Liberation Sans -- which is metrically identical to Arial (same advance widths),
# so layout and line breaks are unchanged either way. DejaVu Sans is the last-resort backstop.
FONT_STACK = ["Arial", "Calibri", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = FONT_STACK
# Vector text stays text (not outlines) if these are ever re-saved as PDF/SVG for a slide deck.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

# 450 dpi: these are projected at full-wall size in talks, where 150 dpi visibly softens.
DPI_DEFAULT = 450

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from snr_aggregate import METRIC_COLS, OUT_DEFAULT                       # noqa: E402

# Nouns, not sentences.
METRIC_TITLE = {"noise_sigma": "Noise floor",
                "snr_median": "Spike SNR",
                "spectral_hf_snr": "Spectral HF SNR"}
METRIC_UNIT = {"noise_sigma": "1.4826 x MAD of the 20 Hz high-pass trace  (log scale)",
               "snr_median": "median spike amplitude / noise floor",
               "spectral_hf_snr": "excess 20-150 Hz PSD power over the white floor  (asinh scale)"}

# Slots 1-3 of the validated categorical theme, in fixed order. Assigned to days by sort order,
# so a day keeps its hue no matter which subset is plotted -- colour follows the entity.
THEME = {
    "light": dict(series=["#2a78d6", "#eb6834", "#1baf7a"], surface="#fcfcfb",
                  primary="#0b0b0b", secondary="#52514e", grid="#dededa"),
    "dark": dict(series=["#3987e5", "#d95926", "#199e70"], surface="#1a1a19",
                 primary="#ffffff", secondary="#c3c2b7", grid="#3a3a38"),
}


def day_label(day, g):
    """Legend text: the day plus the bit depth and frame size that drive its metrics."""
    return f"{day} - {g['bit'].iloc[0]}, {g['dims'].iloc[0].replace('x', ' x ')}"


def style(ax, th, ylabel=None):
    """Recessive axes: horizontal grid only, no top/right spines, text in ink not series colour."""
    ax.set_facecolor(th["surface"])
    ax.grid(axis="y", color=th["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["grid"])
    ax.tick_params(colors=th["secondary"], labelsize=8, length=3)
    if ylabel:
        ax.set_ylabel(ylabel, color=th["secondary"], fontsize=8)


# A violin's KDE is estimated in the coordinates it is handed. Handing it raw values and then
# log-scaling the axis distorts the density -- one July FOV at spectral_hf_snr 300 (p99 is 2.1)
# flattened every other violin into a featureless trapezoid. So the values are transformed FIRST,
# the density is estimated in that space, and the ticks are relabelled with the original units.
# asinh rather than log for spectral_hf_snr: it is an excess ratio and may be negative.
TRANSFORM = {"log": (np.log10, lambda x: np.power(10.0, x)),
             "asinh": (np.arcsinh, np.sinh)}


def _nice_ticks(scale, vmin, vmax):
    """Round original-unit tick values spanning [vmin, vmax], placed at transformed positions."""
    cand = []
    if scale == "log":
        for k in range(int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax))) + 1):
            cand += [m * 10.0 ** k for m in (1, 2, 5)]
    else:
        cand = [0.0]
        for k in range(-3, 4):
            for m in (1, 3):
                cand += [m * 10.0 ** k, -m * 10.0 ** k]
    return sorted({c for c in cand if vmin <= c <= vmax})


def draw_violins(ax, groups, th, color_of, scale=None):
    """One violin per group. ``groups`` is a list of ``(label, values, day, note)``."""
    kept = [(lab, np.asarray(v, float)[np.isfinite(v)], day, note)
            for lab, v, day, note in groups]
    if scale == "log":                       # log is undefined at and below zero
        kept = [(lab, v[v > 0], day, note) for lab, v, day, note in kept]
    kept = [(lab, v, day, note) for lab, v, day, note in kept if v.size]
    if not kept:
        return
    tf, _inv = TRANSFORM.get(scale, (None, None))
    allv = np.concatenate([v for _l, v, _d, _n in kept])
    pos = np.arange(len(kept))
    parts = ax.violinplot([(tf(v) if tf else v) for _l, v, _d, _n in kept],
                          positions=pos, widths=0.72,
                          showextrema=False, showmedians=False)
    for body, (_lab, _v, day, _n) in zip(parts["bodies"], kept):
        body.set_facecolor(color_of(day))
        body.set_alpha(0.55)
        body.set_edgecolor(color_of(day))
        body.set_linewidth(1.0)
    for i, (_lab, v, day, _n) in enumerate(kept):
        # Percentiles commute with a monotone transform, so these are the true quartiles of the
        # data, simply drawn at their transformed positions.
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        if tf:
            q25, med, q75 = tf(q25), tf(med), tf(q75)
        ax.vlines(i, q25, q75, color=th["primary"], linewidth=2.0, zorder=3)  # IQR
        # Median marker carries a surface ring so it stays legible over the body.
        ax.plot(i, med, "o", markersize=4.5, color=th["primary"],
                markeredgecolor=th["surface"], markeredgewidth=1.2, zorder=4)
    ax.set_xticks(pos)
    ax.set_xticklabels([lab for lab, _v, _d, _n in kept], rotation=45, ha="right",
                       fontsize=7.5, color=th["secondary"])
    if tf:
        ticks = _nice_ticks(scale, float(allv.min()), float(allv.max()))
        if ticks:
            ax.set_yticks([tf(t) for t in ticks])
            ax.set_yticklabels([f"{t:g}" for t in ticks])
    # Visible n labels: the relief rule for the low-contrast slot, and useful regardless.
    # Drawn in axes coordinates above the frame, clear of the panel title (see title pad).
    for i, (_lab, _v, _d, note) in enumerate(kept):
        ax.annotate(note, xy=(i, 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, 4), ha="center", va="bottom",
                    fontsize=6.2, color=th["secondary"], annotation_clip=False)


def legend(fig, th, days, labels, color_of):
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=7,
                          markerfacecolor=color_of(d), markeredgecolor=color_of(d))
               for d in days]
    leg = fig.legend(handles, [labels[d] for d in days], loc="lower center",
                     ncol=min(3, len(days)), frameon=False, fontsize=8,
                     bbox_to_anchor=(0.5, 0.005))
    for t in leg.get_texts():
        t.set_color(th["secondary"])           # text wears ink, never the series colour


def metric_figure(fovs, by, level_name, fname, out_dir, mode, labels, days, scales=None,
                  dpi=DPI_DEFAULT):
    """Three stacked panels -- one per metric -- of per-FOV medians grouped by ``by``."""
    th = THEME[mode]
    scales = scales or {}
    groups_meta = _group(fovs, by, level_name)
    n = max(1, len(groups_meta))
    fig, axes = plt.subplots(3, 1, figsize=(max(6.5, 0.62 * n + 3.2), 10.6),
                             facecolor=th["surface"])
    color_of = _color_of(days, th)
    for ax, m in zip(axes, METRIC_COLS):
        groups = [(lab, g[f"{m}_median"].values, day,
                   f"n={len(g)}" + (f"\n{g[f'{m}_n_nan'].sum() / max(g['n_cells'].sum(), 1) * 100:.1f}% NaN"
                                    if g[f"{m}_n_nan"].sum() else ""))
                  for lab, g, day in groups_meta]
        draw_violins(ax, groups, th, color_of, scale=scales.get(m))
        # pad clears the two-line n / NaN annotations drawn just above the frame.
        ax.set_title(METRIC_TITLE[m], color=th["primary"], fontsize=11, loc="left", pad=30)
        style(ax, th, METRIC_UNIT[m])
    fig.suptitle(f"Per-FOV medians by {level_name}", color=th["primary"], fontsize=12.5,
                 x=0.012, ha="left", y=0.995)
    legend(fig, th, days, labels, color_of)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975), h_pad=2.4)
    fig.savefig(os.path.join(out_dir, fname), dpi=dpi, facecolor=th["surface"])
    plt.close(fig)


def cells_figure(fovs, out_dir, mode, labels, days, dpi=DPI_DEFAULT):
    """Cells per FOV, grouped at well / plate / day level."""
    th = THEME[mode]
    color_of = _color_of(days, th)
    levels = [(["day", "plate", "well"], "well"), (["day", "plate"], "plate"), (["day"], "day")]
    fig, axes = plt.subplots(3, 1, figsize=(max(6.5, 0.62 * fovs.groupby(
        ["day", "plate", "well"]).ngroups + 3.2), 9.6), facecolor=th["surface"])
    for ax, (by, name) in zip(axes, levels):
        groups = [(lab, g["n_cells"].values, day, f"n={len(g)}")
                  for lab, g, day in _group(fovs, by, name)]
        draw_violins(ax, groups, th, color_of)
        ax.set_title(f"Cells per FOV, by {name}", color=th["primary"], fontsize=11,
                     loc="left", pad=30)
        style(ax, th, "cells per FOV")
    fig.suptitle("Cell counts", color=th["primary"], fontsize=12.5, x=0.012, ha="left", y=0.995)
    legend(fig, th, days, labels, color_of)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975), h_pad=2.4)
    fig.savefig(os.path.join(out_dir, f"cells_per_fov__{mode}.png"), dpi=dpi,
                facecolor=th["surface"])
    plt.close(fig)


def spikes_figure(fovs, out_dir, mode, labels, days, dpi=DPI_DEFAULT):
    """Spikes per cell, as a raw count and as a duration-normalised rate.

    Both panels are shown because **raw counts are not comparable across days**: recording length
    differs (8389 frames = 10.49 s for July vs 6389 = 7.99 s for March/June at 800 Hz), so July's
    count is inflated ~31% before any biology enters. The rate panel divides that out and is the
    one to quote when comparing days; the count panel maps directly onto ``n_spikes`` in the tables.
    """
    th = THEME[mode]
    color_of = _color_of(days, th)
    f = fovs.copy()
    f["duration_s"] = f["n_frames_analyzed"] / f["fps"]
    f["rate_hz"] = f["n_spikes_median"] / f["duration_s"]
    levels = [(["day", "plate", "well"], "well"), (["day", "plate"], "plate"),
              (["day"], "day"), ([], "whole set")]
    panels = [("n_spikes_median", "Spikes per cell", "median spikes per cell in the FOV"),
              ("rate_hz", "Firing rate", "median spikes per cell per second (Hz)")]
    for by, name in levels:
        groups_meta = _group(f, by, name)
        n = max(1, len(groups_meta))
        fig, axes = plt.subplots(2, 1, figsize=(max(6.5, 0.62 * n + 3.2), 7.4),
                                 facecolor=th["surface"])
        for ax, (col, title, ylab) in zip(np.atleast_1d(axes), panels):
            groups = [(lab, g[col].values, day, f"n={len(g)}") for lab, g, day in groups_meta]
            draw_violins(ax, groups, th, color_of)
            ax.set_title(f"{title}, by {name}" if by else title, color=th["primary"],
                         fontsize=11, loc="left", pad=30)
            style(ax, th, ylab)
        fig.suptitle("Spiking activity", color=th["primary"], fontsize=12.5, x=0.012,
                     ha="left", y=0.995)
        legend(fig, th, days, labels, color_of)
        fig.tight_layout(rect=(0, 0.06, 1, 0.972), h_pad=2.4)
        stem = name.replace(" ", "_")
        fig.savefig(os.path.join(out_dir, f"spikes_by_{stem}__{mode}.png"), dpi=dpi,
                    facecolor=th["surface"])
        plt.close(fig)


def _color_of(days, th):
    order = {d: i for i, d in enumerate(days)}
    return lambda d: th["series"][order[d] % len(th["series"])]


def _group(fovs, by, level_name):
    """``[(label, frame, day)]`` in stable order; partial wells are labelled ``(have/expected)``."""
    if not by:
        return [("all FOVs", fovs, fovs["day"].iloc[0])]
    out = []
    for key, g in fovs.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        lab = "/".join(str(k) for k in key[1:]) or str(key[0])
        if level_name == "well":
            exp = g["n_fovs_expected_in_well"].iloc[0]
            if not pd.isna(exp) and len(g) < exp:
                lab += f" ({len(g)}/{int(exp)})"          # partial wells labelled inline
        out.append((lab, g, key[0]))
    return out


INDEX = """<!doctype html><meta charset="utf-8"><title>SNR summary</title>
<style>
:root{{color-scheme:light dark}}
body{{background:#fcfcfb;color:#0b0b0b;font:14px/1.55 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,sans-serif;margin:0;padding:32px 28px 64px;max-width:1180px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:34px 0 10px;font-weight:600}}
p,li{{color:#52514e;max-width:74ch}} code{{font-size:12px}}
img{{max-width:100%;border:1px solid #dededa;border-radius:6px;background:#fcfcfb}}
.dk{{display:none}} table{{border-collapse:collapse;font-size:12.5px;margin:8px 0 4px}}
th,td{{text-align:left;padding:3px 14px 3px 0;color:#52514e}} th{{color:#0b0b0b}}
@media (prefers-color-scheme:dark){{
 body{{background:#1a1a19;color:#fff}} p,li,th,td{{color:#c3c2b7}} th{{color:#fff}}
 img{{border-color:#3a3a38;background:#1a1a19}}
 .lt{{display:none}} .dk{{display:block}}}}
</style>
<h1>SNR summary statistics</h1>
<p>Per-cell SNR metrics over the pyali extraction corpus, post hoc &mdash; the extraction
pipeline was not re-run. Each violin is a distribution over <b>per-FOV medians</b> (one point =
one FOV), with the interquartile range as a vertical bar and the median as a ringed dot.
Colour encodes the acquisition day; bit depth and frame size are in the legend because they
differ across days and drive the metrics. Partial wells are labelled
<code>(have/expected)</code>.</p>
<p><b>NaN is a statistic, not missing data.</b> <code>per_cell_snr</code> returns NaN where a
metric is undefined &mdash; <code>snr_median</code> when no spike clears k&middot;&sigma; &mdash;
so the NaN rate is the fraction of segmented cells with no detectable activity. It is annotated
on each violin and carried in every table.</p>
{coverage}
{sections}
<h2>Tables</h2>
<p>The numbers behind these figures, one file per level:
<code>cells_all.parquet</code> (one row per cell) and
<code>fov_summary.csv</code> / <code>well_summary.csv</code> / <code>plate_summary.csv</code> /
<code>day_summary.csv</code> / <code>overall_summary.csv</code>, each carrying median, q25, q75,
IQR, <code>n_cells</code> and <code>n_nan</code> per metric. <code>manifest.json</code> records
which pyali outputs this version covered.</p>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", default=OUT_DEFAULT)
    ap.add_argument("--out-dir", default=None, help="defaults to <summary-dir>/figures")
    ap.add_argument("--dpi", type=int, default=DPI_DEFAULT,
                    help="raster resolution; the default suits projection at full-wall size")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.summary_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    fovs = pd.read_csv(os.path.join(a.summary_dir, "fov_summary.csv"),
                       float_precision="round_trip")
    days = sorted(fovs["day"].unique())
    labels = {d: day_label(d, g) for d, g in fovs.groupby("day")}
    # noise_sigma spans an order of magnitude across days (16-bit vs 8-bit), and
    # spectral_hf_snr has a long upper tail; both would otherwise flatten most violins into
    # lines. spectral_hf_snr is an excess ratio and can be negative, hence symlog not log.
    scales = {"noise_sigma": "log", "spectral_hf_snr": "asinh"}

    figs = [(["day", "plate", "well"], "well", "by_well"),
            (["day", "plate"], "plate", "by_plate"),
            (["day"], "day", "by_day"),
            ([], "whole set", "overall")]
    made = []
    for mode in ("light", "dark"):
        for by, level, stem in figs:
            fname = f"snr_{stem}__{mode}.png"
            metric_figure(fovs, by, level, fname, out_dir, mode, labels, days, scales, a.dpi)
            made.append(fname)
        cells_figure(fovs, out_dir, mode, labels, days, a.dpi)
        made.append(f"cells_per_fov__{mode}.png")
        spikes_figure(fovs, out_dir, mode, labels, days, a.dpi)
        made += [f"spikes_by_{n}__{mode}.png" for n in ("well", "plate", "day", "whole_set")]

    sections = []
    for stem, heading in [("by_well", "By well"), ("by_plate", "By plate"),
                          ("by_day", "By day"), ("overall", "Whole set")]:
        sections.append(
            f'<h2>{heading}</h2>\n'
            f'<img class="lt" src="figures/snr_{stem}__light.png" alt="{heading}">\n'
            f'<img class="dk" src="figures/snr_{stem}__dark.png" alt="{heading}">')
    sections.append(
        '<h2>Cell counts</h2>\n'
        '<img class="lt" src="figures/cells_per_fov__light.png" alt="Cells per FOV">\n'
        '<img class="dk" src="figures/cells_per_fov__dark.png" alt="Cells per FOV">')
    sections.append(
        '<h2>Spiking activity</h2>\n<p>Two panels per level. <b>Spikes per cell</b> is the raw '
        'count; <b>firing rate</b> divides by recording duration, which differs by day '
        '(10.49 s for 20260715 vs 7.99 s for 20260331_dir1 and 20260612, both at 800 Hz). '
        'Raw counts are therefore ~31% higher for July before any biology &mdash; <b>quote the '
        'rate when comparing days.</b></p>\n' +
        "\n".join(f'<img class="lt" src="figures/spikes_by_{n}__light.png" alt="Spikes by {n}">\n'
                   f'<img class="dk" src="figures/spikes_by_{n}__dark.png" alt="Spikes by {n}">'
                   for n in ("well", "plate", "day", "whole_set")))

    mpath = os.path.join(a.summary_dir, "manifest.json")
    cov = ""
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        rows = "".join(
            f"<tr><td>{d['day']}</td><td>{d['bit']}, {d['dims']}</td><td>{d['n_fovs']}</td>"
            f"<td>{d['n_cells']}</td>"
            f"<td>{sum(1 for w in d['wells'] if w['complete'])} complete, "
            f"{sum(1 for w in d['wells'] if not w['complete'])} partial</td></tr>"
            for d in man["days"])
        cov = (f"<h2>Coverage</h2><table><tr><th>day</th><th>acquisition</th><th>FOVs</th>"
               f"<th>cells</th><th>wells</th></tr>{rows}</table>"
               f"<p>pyali <code>{man['pyali']['describe']}</code> &middot; generated "
               f"{man['generated_utc']} &middot; {man['totals']['n_fovs']} FOVs, "
               f"{man['totals']['n_cells']} cells.</p>")

    with open(os.path.join(a.summary_dir, "index.html"), "w") as f:
        f.write(INDEX.format(coverage=cov, sections="\n".join(sections)))
    print(f"[fig] {len(made)} figures -> {out_dir}")
    print(f"[fig] {os.path.join(a.summary_dir, 'index.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
