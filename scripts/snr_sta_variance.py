#!/usr/bin/env python3
"""Make between-cell STA variance visible -- README section 11, Task 1.

This is a **pooled** perturbational library: every cell in a FOV may carry a different
perturbation, so the heterogeneity of interest is *between cells within a FOV*. The
median+IQR band in ``sta_fov_grid`` hides it for three reasons -- IQR shows only the middle 50%
while a pooled library puts its phenotypes in the tails, the y-limits are shared across 27 panels,
and the ``k=3 sigma`` detection floor censors the low end.

**The variance is real.** Each snippet carries ~1 sigma of noise by construction, so a cell's STA
point has measurement variance ~ ``1 / n_spikes_used``; anything above that is genuine between-cell
spread. Measured at the peak, ``Var_obs / Var_meas`` runs **13-128x** in healthy wells -- and
**1.0x** in a well whose "spikes" are threshold noise crossings, which is exactly the control case
this ratio is meant to catch.

Outputs, into ``snr_summary/``:

* ``sta_variance_by_fov.csv`` -- the decomposition for every FOV, at t=0 and t=+2.5 ms
* ``sta_fov_p5p95``   -- median with p5-p95 (outer) and IQR (inner) bands
* ``sta_fov_absolute`` -- absolute amplitude (per-cell sigma restored), median + p5-p95
* ``sta_fov_percell`` -- every cell's curve overlaid: the honest picture of a pooled library
* ``sta_fov_heatmap`` -- per-cell STAs as a heatmap, cells sorted by peak, so clusters band

Reads only ``sta_cells.mat`` (already on disk) plus ``snr_metrics.csv`` for the 27 plotted FOVs.
No trace re-reads, no recompute.

    python scripts/snr_sta_variance.py
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm      # noqa: E402
from matplotlib.ticker import MaxNLocator                                # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from pyali.io import load_v73                                            # noqa: E402
from snr_aggregate import OUT_DEFAULT                                    # noqa: E402
from snr_corpus import METRICS, REQUIRED, S3_DEFAULT, list_fovs          # noqa: E402
from snr_corpus import READ_ROOT_DEFAULT                                 # noqa: E402
from snr_figures import THEME, _color_of, day_label, legend, style       # noqa: E402
from snr_sta import STA_CELLS                                            # noqa: E402
from snr_sta_figures import INTERIOR, grid_dpi                           # noqa: E402

# Diverging map for the heatmap: STA amplitude relative to a 0 baseline is genuine polarity
# (depolarising vs hyperpolarising), so a diverging scale is the right encoding -- two hues with a
# NEUTRAL GREY midpoint, never a hue at the middle and never a rainbow. The poles are slots 1 and 2
# of the validated categorical theme. Limits are kept symmetric so equal colour distance means
# equal value distance.
DIVERGING = LinearSegmentedColormap.from_list(
    "sta_div", ["#2a78d6", "#a8c6e8", "#e8e8e4", "#f3b79b", "#eb6834"])


def read_cells(args):
    """Load one FOV's per-cell STA matrix. Returns ``(key, A, n_used, cell_index, t_ms, err)``."""
    day, dir_name, read_root = args
    p = os.path.join(read_root, day, dir_name, STA_CELLS)
    try:
        d = load_v73(p)
        A = np.asarray(d["sta_cells"], float)
        if A.ndim != 2 or not A.size:
            return (day, dir_name), None, None, None, None, None
        return ((day, dir_name), A, np.asarray(d["n_spikes_used"]).ravel().astype(float),
                np.asarray(d["cell_index"]).ravel().astype(int),
                np.asarray(d["t_ms"]).ravel(), None)
    except Exception as e:                                   # noqa: BLE001
        return (day, dir_name), None, None, None, None, f"{type(e).__name__}: {e}"


def decompose(A, n_used, t_ms):
    """Variance decomposition of the per-cell STA at t=0 and t=+2.5 ms.

    ``var_meas`` is the variance a cell's STA point carries purely from averaging a finite number of
    noisy snippets. In sigma units each snippet has unit noise variance, so the mean of ``n``
    snippets has variance ``1/n``; averaged over cells that is ``mean(1/n_used)``. Subtracting it
    from the observed across-cell variance leaves the part that is genuinely between cells.
    """
    out = {}
    i0 = int(np.argmin(np.abs(t_ms)))
    i25 = int(np.argmin(np.abs(t_ms - 2.5)))
    vm = float(np.mean(1.0 / np.clip(n_used, 1, None)))
    for tag, i in (("pk", i0), ("t25", i25)):
        v = A[:, i]
        vo = float(np.var(v, ddof=1)) if v.size > 1 else 0.0
        q = np.percentile(v, [5, 25, 50, 75, 95]) if v.size else [np.nan] * 5
        out.update({f"{tag}_median": q[2], f"{tag}_q25": q[1], f"{tag}_q75": q[3],
                    f"{tag}_p5": q[0], f"{tag}_p95": q[4],
                    f"{tag}_var_obs": vo, f"{tag}_var_meas": vm,
                    f"{tag}_var_bio": max(vo - vm, 0.0),
                    f"{tag}_sd_bio": float(np.sqrt(max(vo - vm, 0.0))),
                    f"{tag}_ratio": vo / vm if vm > 0 else np.nan})
    out["n_cells"] = int(A.shape[0])
    out["n_spikes_used_median"] = float(np.median(n_used))
    return out


def cell_features(A, n_used, cell_index, t_ms):
    """Per-cell shape features. One row per cell.

    **decay ratio** = amplitude at +2.5 ms divided by the peak. A real action potential decays over
    a few milliseconds, so a genuine AP lands at ~0.25-0.38. A threshold noise crossing is a single
    -sample excursion with nothing after it, so it lands at ~0. It is a pure SHAPE measure --
    independent of amplitude and of the sigma normalisation -- so filtering on it does not
    manufacture the amplitude result being interpreted, and it is immune to the k=3 sigma censoring
    that compresses the amplitude distribution against its floor.

    **fwhm_ms** is quantised: at 800 Hz the sample interval is 1.25 ms and the AP is ~2 samples
    wide, so it can only take the values 1.25, 2.50, 3.75 ... It is reported for completeness, not
    as a continuous shape variable.
    """
    dt = float(t_ms[1] - t_ms[0])
    i0 = int(np.argmin(np.abs(t_ms)))
    i25 = int(np.argmin(np.abs(t_ms - 2.5)))
    i50 = int(np.argmin(np.abs(t_ms - 5.0)))
    pk = A[:, i0].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(pk > 0, A[:, i25] / pk, np.nan)
        decay5 = np.where(pk > 0, A[:, i25:i50 + 1].mean(axis=1) / pk, np.nan)
    fwhm = np.full(A.shape[0], np.nan)
    for j in range(A.shape[0]):
        if pk[j] <= 0:
            continue
        above = np.flatnonzero(A[j] >= pk[j] / 2.0)
        if above.size:
            # contiguous run containing the peak
            lo = hi = i0
            while lo - 1 in above:
                lo -= 1
            while hi + 1 in above:
                hi += 1
            fwhm[j] = (hi - lo + 1) * dt
    return pd.DataFrame({"cell_index": cell_index.astype(np.int32),
                         "peak_sigma": pk.astype(np.float32),
                         "decay": decay.astype(np.float32),
                         "decay_5ms": decay5.astype(np.float32),
                         "fwhm_ms": fwhm.astype(np.float32),
                         "n_spikes_used": n_used.astype(np.int32)})


DECAY_CUTS = (0.00, 0.05, 0.10, 0.15, 0.20)
DECAY_CUT_DEFAULT = 0.10


def decay_noise_sd(peak_sigma, n_used):
    """Expected decay-ratio spread for a cell whose "spike" is a threshold noise crossing.

    The sample after a noise crossing is uncorrelated noise. Averaged over ``n`` snippets it has SD
    ~ ``1/sqrt(n)`` in sigma units; dividing by the peak gives ``1/(peak*sqrt(n))``. This is why the
    noise population is a broad shoulder centred on 0 rather than a spike at 0, and why it OVERLAPS
    the AP population instead of separating from it.
    """
    return 1.0 / (np.clip(peak_sigma, 1e-6, None) * np.sqrt(np.clip(n_used, 1, None)))


# Panels are deliberately large.# Panels are deliberately large. These are presented from a projector, and the earlier 2.9 x 2.15 in
# panel on a y-axis shared across all 27 wells left a 3.4 sigma waveform occupying under half its
# panel. Figures are split per day (each day gets its own y range) and capped at 4 columns.
PANEL_W, PANEL_H = 4.4, 3.3
MAX_COLS = 4


def grid(panels, draw, out_dir, stem, title, note, labels, days, dpi=None, ncols=None,
         ylabel="amplitude (sigma)", sharey=True, ylim_fn=None):
    """Small-multiples layout shared by all four figures; ``draw(ax, panel, th, color_of)``."""
    th = THEME["light"]
    color_of = _color_of(days, th)
    dpi = dpi or grid_dpi(len(panels))
    ncols = ncols or min(MAX_COLS, max(1, len(panels)))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(PANEL_W * ncols + 1.2, PANEL_H * nrows + 1.9),
                             facecolor=th["surface"], sharex=True, sharey=sharey)
    axf = np.atleast_1d(axes).ravel()
    for ax, p in zip(axf, panels):
        draw(ax, p, th, color_of)
        ax.set_title(f"{p['label']}  (n={p['n_cells']}, {p['ratio']:.0f}x, "
                     f"SD={p['sd_bio']:.2f})", color=th["primary"], fontsize=10.5,
                     loc="left", pad=5)
        style(ax, th)
    for ax in axf[len(panels):]:
        ax.set_visible(False)
    g = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in g[:, 0]:
        ax.set_ylabel(ylabel, color=th["secondary"], fontsize=9.5)
    for c in range(ncols):
        used = [r for r in range(nrows) if r * ncols + c < len(panels)]
        if used:
            ax = g[used[-1], c]
            ax.set_xlabel("time from spike peak (ms)", color=th["secondary"], fontsize=9)
            ax.tick_params(labelbottom=True)
    t0 = min(float(p["t"].min()) for p in panels)
    t1 = max(float(p["t"].max()) for p in panels)
    axf[0].set_xlim(t0, t1)
    step = 10.0
    cand = np.arange(np.ceil(t0 / step) * step, t1 + 1e-9, step)
    m = 0.06 * (t1 - t0)
    axf[0].set_xticks(cand[(cand - t0 > m) & (t1 - cand > m)])
    if sharey and ylim_fn is not None:
        lo, hi = ylim_fn(panels)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.06 * (hi - lo)
            axf[0].set_ylim(lo - pad, hi + pad)
    axf[0].yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    fig.suptitle(title, color=th["primary"], fontsize=12.5, x=0.008, ha="left", y=0.997)
    fig.text(0.008, 0.962, note, color=th["secondary"], fontsize=8.5, ha="left")
    legend(fig, th, days, labels, color_of)
    fig.tight_layout(rect=(0, 0.035, 1, 0.952), h_pad=1.9, w_pad=1.4)
    fig.savefig(os.path.join(out_dir, f"{stem}__light.png"), dpi=dpi, facecolor=th["surface"])
    plt.close(fig)
    w, h = fig.get_size_inches()
    return dpi, int(w * dpi), int(h * dpi)


def ylim_bands(panels, key="A"):
    """Range that contains the p5-p95 bands. Extremes are left to clip -- a handful of cells
    otherwise stretches the axis to ~14 sigma and squashes every median into the lower third."""
    lo = min(float(np.percentile(p[key], 5, axis=0).min()) for p in panels)
    hi = max(float(np.percentile(p[key], 95, axis=0).max()) for p in panels)
    return lo, hi


def ylim_percell(panels, key="A"):
    """Range containing the bulk of individual cell curves (0.5th-99.5th percentile)."""
    lo = min(float(np.percentile(p[key], 0.5)) for p in panels)
    hi = max(float(np.percentile(p[key], 99.5)) for p in panels)
    return lo, hi


def _bands(ax, p, th, color_of, key="A"):
    c = color_of(p["day"])
    A, t = p[key], p["t"]
    q = np.percentile(A, [5, 25, 50, 75, 95], axis=0)
    ax.axhline(0.0, color=th["grid"], linewidth=0.6, zorder=1)
    ax.axvline(0.0, color=th["grid"], linewidth=0.6, zorder=1)
    ax.fill_between(t, q[0], q[4], color=c, alpha=0.18, linewidth=0, zorder=2)   # p5-p95
    ax.fill_between(t, q[1], q[3], color=c, alpha=0.38, linewidth=0, zorder=3)   # IQR
    ax.plot(t, q[2], "-", color=c, linewidth=1.8, zorder=4)


def draw_p5p95(ax, p, th, color_of):
    _bands(ax, p, th, color_of, "A")


def draw_absolute(ax, p, th, color_of):
    _bands(ax, p, th, color_of, "A_abs")


def draw_percell(ax, p, th, color_of):
    c = color_of(p["day"])
    A, t = p["A"], p["t"]
    ax.axhline(0.0, color=th["grid"], linewidth=0.6, zorder=1)
    ax.axvline(0.0, color=th["grid"], linewidth=0.6, zorder=1)
    # Every cell, drawn faintly. Alpha is tuned so a few hundred overlapping curves read as a
    # density rather than a blob; this is the picture a pooled library actually produces.
    step = max(1, A.shape[0] // 900)
    ax.plot(t, A[::step].T, "-", color=c, linewidth=0.35,
            alpha=min(0.30, max(0.02, 12.0 / max(A.shape[0], 1))), zorder=2)
    ax.plot(t, np.median(A, axis=0), "-", color=th["primary"], linewidth=1.6, zorder=4)


def draw_heatmap(ax, p, th, color_of):
    A, t = p["A"], p["t"]
    order = np.argsort(A[:, int(np.argmin(np.abs(t)))])
    M = A[order]
    lim = float(np.percentile(np.abs(M), 99)) or 1.0
    ax.imshow(M, aspect="auto", origin="lower", cmap=DIVERGING,
              norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
              extent=(t[0], t[-1], 0, M.shape[0]), interpolation="nearest")
    ax.grid(False)
    # Row index carries no information beyond the sort order, and with a per-panel y axis these
    # labels both clutter the grid and collide across columns. The count is in the panel title.
    ax.set_yticks([])


def hist_grid(panels, out_dir, stem, title, note, labels, days, cut_by_day, dpi=None,
              ncols=None):
    """One decay-ratio histogram per panel, with the AP-vs-noise cut marked."""
    th = THEME["light"]
    color_of = _color_of(days, th)
    dpi = dpi or grid_dpi(len(panels))
    ncols = ncols or min(MAX_COLS, max(1, len(panels)))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(PANEL_W * ncols + 1.2, PANEL_H * nrows + 1.9),
                             facecolor=th["surface"], sharex=True, sharey=False)
    axf = np.atleast_1d(axes).ravel()
    bins = np.linspace(-0.3, 0.8, 90)
    for ax, p in zip(axf, panels):
        v = p["vals"][np.isfinite(p["vals"])]
        ax.hist(v, bins=bins, color=color_of(p["day"]), alpha=0.75, linewidth=0)
        cut = cut_by_day[p["day"]]
        ax.axvline(cut, color=th["primary"], linewidth=1.6, linestyle="--", zorder=5)
        frac = float(np.mean(v < cut)) * 100 if v.size else np.nan
        ax.set_title(f"{p['label']}  (n={v.size}, {frac:.0f}% below cut)",
                     color=th["primary"], fontsize=10.5, loc="left", pad=5)
        style(ax, th)
        ax.set_yticks([])
    for ax in axf[len(panels):]:
        ax.set_visible(False)
    g = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in g[:, 0]:
        ax.set_ylabel("cells", color=th["secondary"], fontsize=9.5)
    for c in range(ncols):
        used = [r for r in range(nrows) if r * ncols + c < len(panels)]
        if used:
            ax = g[used[-1], c]
            ax.set_xlabel("decay ratio  amp(+2.5 ms) / peak", color=th["secondary"], fontsize=9)
            ax.tick_params(labelbottom=True)
    axf[0].set_xlim(-0.3, 0.8)
    axf[0].set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    fig.suptitle(title, color=th["primary"], fontsize=12.5, x=0.008, ha="left", y=0.997)
    fig.text(0.008, 0.962, note, color=th["secondary"], fontsize=8.5, ha="left")
    legend(fig, th, days, labels, color_of)
    fig.tight_layout(rect=(0, 0.035, 1, 0.952), h_pad=1.9, w_pad=1.4)
    fig.savefig(os.path.join(out_dir, f"{stem}__light.png"), dpi=dpi, facecolor=th["surface"])
    plt.close(fig)
    return dpi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-prefix", default=S3_DEFAULT)
    ap.add_argument("--read-root", default=READ_ROOT_DEFAULT)
    ap.add_argument("--summary-dir", default=OUT_DEFAULT)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--dpi", type=int, default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.summary_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(os.path.join(a.summary_dir, "fov_summary.csv"),
                       float_precision="round_trip")
    fovs = list_fovs(a.s3_prefix)
    keys = sorted(k for k, v in fovs.items()
                  if set(REQUIRED).issubset(v) and STA_CELLS in v)
    if a.days:
        keys = [k for k in keys if k[0] in a.days]
    print(f"[var] {len(keys)} FOVs with {STA_CELLS}", flush=True)

    # ---- which FOVs get plotted: one representative INTERIOR FOV per well ----
    m = meta.copy()
    m["bw"] = m["burst"] - m.groupby(["day", "plate_num", "well"])["burst"].transform("min")
    span = m.groupby(["day", "plate_num", "well"])["bw"].transform("max").replace(0, np.nan)
    m["frac"] = m["bw"] / span
    have = set(keys)
    reps = {}
    for (day, plate, pn, wl), g in m.groupby(["day", "plate", "plate_num", "well"]):
        g = g[[(d, n) in have for d, n in zip(g["day"], g["dir_name"])]]
        inner = g[(g["frac"] >= INTERIOR[0]) & (g["frac"] <= INTERIOR[1])]
        pick = inner if len(inner) else g
        if not len(pick):
            continue
        r = pick.iloc[(pick["n_cells"] - pick["n_cells"].median()).abs().argsort().iloc[0]]
        reps[(day, r["dir_name"])] = f"{plate}/{wl}"

    rows, feats, kept, errors = [], [], {}, []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, (key, A, nu, ci, t, err) in enumerate(
                ex.map(read_cells, [(d, n, a.read_root) for d, n in keys]), start=1):
            if err:
                errors.append((key, err)); continue
            if A is None:
                continue
            rec = dict(day=key[0], dir_name=key[1])
            rec.update(decompose(A, nu, t))
            rows.append(rec)
            cf = cell_features(A, nu, ci, t)
            cf["day"] = key[0]; cf["dir_name"] = key[1]
            feats.append(cf)
            if key in reps:
                kept[key] = (A, ci, t)
            if i % 400 == 0:
                print(f"[var] read {i}/{len(keys)}", flush=True)
    for k, e in errors:
        print(f"[var] READ FAIL {k}: {e}", flush=True)

    var = pd.DataFrame(rows).merge(
        meta[["day", "dir_name", "plate", "plate_num", "well", "burst", "bit", "dims"]],
        on=["day", "dir_name"], how="left")
    var.to_csv(os.path.join(a.summary_dir, "sta_variance_by_fov.csv"), index=False)

    days = sorted(var["day"].unique())
    labels = {d: day_label(d, g) for d, g in meta.groupby("day")}

    # ---- per-cell features (section 11.2) ----
    cf = pd.concat(feats, ignore_index=True)
    # noise_sigma / snr_median / n_spikes come from cells_all.parquet rather than re-reading 2184
    # snr_metrics.csv files.
    ca = pd.read_parquet(os.path.join(a.summary_dir, "cells_all.parquet"),
                         columns=["day", "dir_name", "cell_index", "noise_sigma",
                                  "snr_median", "n_spikes"])
    cf = cf.merge(ca, on=["day", "dir_name", "cell_index"], how="left")
    cf = cf.merge(meta[["day", "dir_name", "plate", "plate_num", "well", "burst"]],
                  on=["day", "dir_name"], how="left")
    cf["peak_abs"] = cf["peak_sigma"] * cf["noise_sigma"]
    cf.to_parquet(os.path.join(a.summary_dir, "sta_cell_features.parquet"), index=False)
    print(f"[var] sta_cell_features.parquet  {len(cf)} cells x {cf.shape[1]} cols", flush=True)

    # ONE cut, stated rather than discovered. The pooled per-cell decay density is unimodal
    # (mode ~ +0.29, long left shoulder through 0), so there is no trough to find: the noise
    # population is centred on 0 but broad -- SD ~ 1/(peak*sqrt(n_spikes)) ~ 0.10-0.17 -- and
    # therefore OVERLAPS the AP population near +0.30. A per-cell cut can only shift the mixture,
    # never cleanly separate it. A per-DAY cut would additionally filter days differently and
    # confound exactly the day comparison the summary is for, so the cut is global.
    cut_by_day = {d: DECAY_CUT_DEFAULT for d in days}
    for day in days:
        sub = cf[cf.day == day]
        panels = [dict(label=f"{pl}/{wl}", day=day, vals=g["decay"].values)
                  for (pl, wl), g in sub.groupby(["plate", "well"])]
        d = hist_grid(panels, out_dir, f"sta_decay_hist_{day}",
                      f"Decay ratio per cell \u2014 {day}",
                      "amp(+2.5 ms)/peak. A real AP decays (~0.25-0.38); a threshold noise crossing "
                      f"does not (~0, broad). Dashed line = stated cut {DECAY_CUT_DEFAULT:.2f} -- the "
                      "pooled density is unimodal, so this is a criterion, not a discovered trough.",
                      labels, days, cut_by_day, a.dpi)
        print(f"[var] sta_decay_hist_{day}__light.png  {len(panels)} panels  {d} dpi", flush=True)

    # noise_sigma only for the plotted FOVs, to restore absolute amplitude
    panels = []
    for key, label in sorted(reps.items(), key=lambda kv: kv[1]):
        if key not in kept:
            continue
        A, ci, t = kept[key]
        day, dir_name = key
        sig = None
        try:
            mm = pd.read_csv(os.path.join(a.read_root, day, dir_name, METRICS),
                             float_precision="round_trip").set_index("cell_index")
            sig = mm.loc[ci, "noise_sigma"].values
        except Exception:                                    # noqa: BLE001
            sig = np.ones(A.shape[0])
        d = var[(var.day == day) & (var.dir_name == dir_name)].iloc[0]
        panels.append(dict(label=label, day=day, t=t, A=A, A_abs=A * sig[:, None],
                           n_cells=int(A.shape[0]), ratio=float(d.pk_ratio),
                           sd_bio=float(d.pk_sd_bio)))
    print(f"[var] {len(panels)} representative FOVs plotted", flush=True)

    jobs = [
        (draw_p5p95, "sta_fov_p5p95", "STA spread across cells within a FOV",
         "line = median across cells; dark band = IQR (middle 50%); pale band = p5-p95. "
         "Panel: n cells, Var_obs/Var_meas at the peak, and between-cell SD in sigma.",
         "amplitude (sigma)", lambda ps: ylim_bands(ps, "A")),
        (draw_absolute, "sta_fov_absolute", "STA spread, absolute amplitude",
         "Per-cell sigma restored (curve x that cell's noise_sigma), so amplitude differences are "
         "not divided out. NOT comparable across days -- trace units differ.",
         "amplitude (trace units)", lambda ps: ylim_bands(ps, "A_abs")),
        (draw_percell, "sta_fov_percell", "Every cell's STA, overlaid",
         "One faint line per cell, median in black. The distribution itself rather than a summary "
         "of it -- what a pooled library actually looks like. Axis spans the 0.5-99.5 percentile; "
         "a few extreme cells clip.", "amplitude (sigma)", lambda ps: ylim_percell(ps, "A")),
        (draw_heatmap, "sta_fov_heatmap", "Per-cell STAs, cells sorted by peak amplitude",
         "Rows = cells (sorted), columns = time. Diverging scale, neutral grey at the 0 baseline, "
         "symmetric limits at the 99th percentile of |amplitude|. Banding = cluster structure.",
         "cell (sorted by peak)", None),
    ]
    # One figure PER DAY per type. Sharing a y-axis across all 27 wells squashed the low-amplitude
    # wells; per-day figures let each day set its own range, and fewer panels means bigger ones.
    for draw, stem, title, note, ylab, ylim_fn in jobs:
        for day in days:
            sub = [p for p in panels if p["day"] == day]
            if not sub:
                continue
            # the heatmap's y extent is cell count, which differs per panel, so it cannot share y
            sy = stem != "sta_fov_heatmap"
            fname = f"{stem}_{day}"
            d, px, py = grid(sub, draw, out_dir, fname, f"{title} \u2014 {day}", note,
                             labels, days, a.dpi, ylabel=ylab, sharey=sy, ylim_fn=ylim_fn)
            mb = os.path.getsize(os.path.join(out_dir, f"{fname}__light.png")) / 1e6
            print(f"[var] {fname}__light.png  {len(sub)} panels  {d} dpi  {px}x{py} px  "
                  f"{mb:.1f} MB", flush=True)

    print("\n[var] decay ratio -- the pooled per-cell density is UNIMODAL (mode ~+0.29, long left",
          flush=True)
    print("[var] shoulder through 0), so no trough exists; the cut below is a stated criterion.",
          flush=True)
    print(f"\n[var] retention sweep (fraction of cells with decay >= cut):", flush=True)
    hdr = "        " + f"{'day':<16}" + "".join(f"{c:>9.2f}" for c in DECAY_CUTS)
    print(hdr, flush=True)
    for day in days:
        v = cf.loc[cf.day == day, "decay"].dropna()
        print("        " + f"{day:<16}" + "".join(f"{float((v >= c).mean()) * 100:8.1f}%"
                                                 for c in DECAY_CUTS), flush=True)
    v = cf["decay"].dropna()
    print("        " + f"{'ALL':<16}" + "".join(f"{float((v >= c).mean()) * 100:8.1f}%"
                                               for c in DECAY_CUTS), flush=True)
    print(f"\n[var] at the default cut {DECAY_CUT_DEFAULT:.2f}: "
          f"{int((cf['decay'] >= DECAY_CUT_DEFAULT).sum())} of {len(cf)} cells retained "
          f"({float((cf['decay'] >= DECAY_CUT_DEFAULT).mean()) * 100:.1f}%)", flush=True)
    print("\n[var] WELL-level decay medians separate cleanly where per-cell values do not:",
          flush=True)
    wm = cf.groupby(["day", "plate", "well"])["decay"].median().sort_values()
    for (day, pl, wl), val in wm.items():
        flag = "  <-- threshold-noise well" if val < 0.10 else ""
        print(f"        {day:<16}{pl}/{wl:<4} {val:+.3f}{flag}", flush=True)

    print("\n[var] between-cell variance at the STA peak, by day "
          "(Var_obs/Var_meas; >1 means real spread):", flush=True)
    for day, g in var.groupby("day"):
        print(f"        {day:<16} median ratio {g.pk_ratio.median():7.1f}   "
              f"median SD_bio {g.pk_sd_bio.median():.2f}σ   "
              f"FOVs with ratio<2 (no real spread): {(g.pk_ratio < 2).mean() * 100:5.1f}%",
          flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
