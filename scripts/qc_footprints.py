#!/usr/bin/env python3
"""Footprint-on-pixels + trace QC for one result directory: 10 cells across the spike-amplitude-SNR
percentiles (or the cells named with ``--cells``), each as [reference crop with footprint overlay |
spike map | spike-triggered average | trace | largest spike].

    python scripts/qc_footprints.py <results_dir> [--fov <fov_dir>] [--n 10] [--crop 48] [--no-sta]
    python scripts/qc_footprints.py <results_dir> --cells C0713,C0316,C0331,C1216

Writes ``<results_dir>/QC/footprints_traces.png`` and ``footprints_traces.csv`` so each FOV's QC
travels with its results. Per picked cell:

* column 1 -- ``brightness_reference`` (raw-unit background-corrected 300-frame mean, the image
  segmentation saw) with the footprint as an alpha fill, a contour at half its maximum and a ``+``
  at the cell centre;
* column 2 -- spike map: the reference crop with every localized spike's COM -- this cell's as
  yellow dots, each neighbouring cell's in its own colour with that cell's footprint contour,
  spikes merged into a stronger same-frame duplicate as gray ``x`` and spikes in no cell as hollow
  gray circles. Several clouds under one footprint are cells lumped into one trace; one cloud
  per contour is a clean separation;
* column 3 -- spatial spike-triggered average read crop-by-crop from ``frames1.bin`` at the cell's
  own spike frames: ``mean(F[t - lag]) - mean(F[t])``, the fluorescence *drop* at the spike (the
  pipeline's sign convention: spikes are negative-going in F and positive in the traces), so the
  pixels that actually flash are bright. The frames are the raw, motion-uncorrected ones (shifts on
  these fields are <= ~1 px);
* column 4 -- the full trace with ticks at the pipeline's spikes; column 5 -- +/-25 ms around the
  largest one.

``eff_px = (sum w)^2 / sum w^2`` is the footprint's effective pixel count (the number of equally
weighted pixels that would give the same noise averaging); ``n_px`` counts every nonzero weight.

Cells are drawn from the spiking cells with at least ``--min-spikes`` QC spikes; the picks sit at
the 5th, 15th, ..., 95th percentile of ``snr_median`` (:func:`pyali.metrics.per_cell_snr`: median
spike height / 1.4826 x MAD of the 20 Hz high-passed trace), i.e. 2-3 per quartile.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pyali.io import open_bin_memmap                                      # noqa: E402
from pyali.metrics import per_cell_snr                                   # noqa: E402
from pyali.results import load_results                                   # noqa: E402

QC_DIR = "QC"
PNG = "footprints_traces.png"
CSV = "footprints_traces.csv"
CSV_COLUMNS = ["cell_id", "percentile", "snr_median", "noise_sigma", "qc_spikes", "n_spikes",
               "n_px", "eff_px", "x_px", "y_px", "region_id", "source", "trace_row"]


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def cell_table(R):
    """``R.cells`` plus ``snr_median`` / ``noise_sigma`` / ``qc_spikes`` (from
    :func:`per_cell_snr`), ``n_px`` (footprint nonzeros), ``eff_px`` (``(sum w)^2 / sum w^2``)
    and ``trace_row`` = row index."""
    df = R.cells.copy().reset_index(drop=True)
    m = per_cell_snr(R.traces, float(R.attrs["fps"]))
    df["snr_median"] = m["snr_median"]
    df["noise_sigma"] = m["noise_sigma"]
    df["qc_spikes"] = m["n_spikes"]
    off = np.asarray(R.footprints.offsets, np.int64)
    df["n_px"] = (off[1:] - off[:-1]).astype(int)
    w = np.asarray(R.footprints.values, np.float64)
    s1 = np.add.reduceat(w, off[:-1]) if w.size else np.zeros(len(df))
    s2 = np.add.reduceat(w * w, off[:-1]) if w.size else np.zeros(len(df))
    empty = df["n_px"].to_numpy() == 0                                   # reduceat misreads empty runs
    s1[empty] = 0.0; s2[empty] = 0.0
    df["eff_px"] = np.where(s2 > 0, s1 * s1 / np.where(s2 > 0, s2, 1.0), 0.0).astype(np.float32)
    df["trace_row"] = np.arange(len(df))
    return df


def percentiles(n):
    """``n`` percentiles evenly spread over (0, 100): 5, 15, ..., 95 for ``n = 10``."""
    return [100.0 * (i + 0.5) / n for i in range(n)]


def select_cells(df, n=10, min_spikes=3):
    """The rows nearest the :func:`percentiles` of ``snr_median`` among spiking cells with at
    least ``min_spikes`` QC spikes; duplicates (small pools) are dropped. Adds ``percentile``."""
    pool = df[(df["source"] == "spiking") & (df["qc_spikes"] >= min_spikes)
              & np.isfinite(df["snr_median"].to_numpy(float))]
    if len(pool) == 0:
        return pool.assign(percentile=np.zeros(0))
    v = pool["snr_median"].to_numpy(float)
    rows, pcts, seen = [], [], set()
    for p in percentiles(n):
        j = int(np.argmin(np.abs(v - np.percentile(v, p))))
        if j in seen:
            continue
        seen.add(j)
        rows.append(j)
        pcts.append(p)
    out = pool.iloc[rows].copy()
    out["percentile"] = pcts
    return out.reset_index(drop=True)


def pick_named(df, names):
    """Rows of ``df`` for ``names`` -- full cell ids, ``C0713``-style suffixes or trace rows --
    in the given order, with ``percentile`` = NaN. Unknown names raise ``KeyError``."""
    rows = []
    ids = df["cell_id"].astype(str).tolist()
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if name.isdigit() and int(name) < len(df):
            j = int(name)
        else:
            hits = [i for i, cid in enumerate(ids) if cid == name or cid.endswith("_" + name)]
            if not hits:
                raise KeyError(f"no cell {name!r} in this result")
            j = hits[0]
        rows.append(j)
    out = df.iloc[rows].copy()
    out["percentile"] = np.nan
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Crops
# --------------------------------------------------------------------------- #
def crop_window(center_rc, H, W, size):
    """0-indexed top-left ``(r0, c0)`` of a ``size x size`` crop centred on the 1-indexed
    ``center_rc``, clamped so the crop stays inside an ``(H, W)`` frame."""
    size = int(min(size, H, W))
    r = int(round(float(center_rc[0]) - 1)) - size // 2
    c = int(round(float(center_rc[1]) - 1)) - size // 2
    return min(max(r, 0), H - size), min(max(c, 0), W - size)


def sta_crop(movie, frames, r0, c0, size, lag=8, max_frames=100, amplitudes=None, sign=-1.0):
    """``sign * (mean(F[t]) - mean(F[t - lag]))`` over the crop at the given spike frames
    (0-based); the default ``sign=-1`` shows the fluorescence drop at the spike as positive, the
    pipeline's convention. Frames before ``lag`` or past the movie are dropped; when more than
    ``max_frames`` remain the largest ``amplitudes`` win (or the first ones). ``None`` when no
    frame is usable."""
    frames = np.asarray(frames, np.int64).ravel()
    T = movie.shape[0]
    keep = (frames >= lag) & (frames < T)
    frames = frames[keep]
    if frames.size == 0:
        return None
    if frames.size > max_frames:
        if amplitudes is not None:
            amp = np.asarray(amplitudes, float).ravel()[keep]
            frames = frames[np.argsort(-amp, kind="stable")[:max_frames]]
        else:
            frames = frames[:max_frames]
    frames = np.sort(frames)
    on = np.asarray(movie[frames, r0:r0 + size, c0:c0 + size], np.float64).mean(axis=0)
    off = np.asarray(movie[frames - lag, r0:r0 + size, c0:c0 + size], np.float64).mean(axis=0)
    return float(sign) * (on - off)


def find_movie(results_dir, meta, fov_dir=None):
    """Memmap of ``frames1.bin`` for this result (``fov_dir`` overrides the recorded one);
    ``None`` when the movie or the frame dimensions are unavailable."""
    fov_dir = fov_dir or meta.get("fov_dir")
    params = meta.get("params") or {}
    if not fov_dir or not all(k in params for k in ("nrow", "ncol")):
        return None
    path = os.path.join(fov_dir, "frames1.bin")
    if not os.path.isfile(path):
        return None
    return open_bin_memmap(path, int(params["nrow"]), int(params["ncol"]),
                           str(params.get("read_dtype", "uint16")))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
NEIGHBOUR_COLOURS = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2", "#8c564b", "#bcbd22",
                     "#1f77b4", "#7f7f7f"]                       # no cyan: that is the cell's own contour
OWN_COLOUR = "#ffd400"


def _stretch(img, lo=1.0, hi=99.0):
    v = np.asarray(img, float)
    a, b = np.nanpercentile(v, lo), np.nanpercentile(v, hi)
    return (a, b) if b > a else (float(np.nanmin(v)), float(np.nanmax(v)) + 1e-9)


def cells_in_crop(R, r0, c0, size, exclude=()):
    """Trace rows of the cells whose centre lies inside the crop, nearest the crop centre first."""
    rc = R.centers - 1.0
    inside = (rc[:, 0] >= r0) & (rc[:, 0] < r0 + size) & (rc[:, 1] >= c0) & (rc[:, 1] < c0 + size)
    rows = [int(k) for k in np.flatnonzero(inside) if int(k) not in set(exclude)]
    mid = np.array([r0 + size / 2.0, c0 + size / 2.0])
    return sorted(rows, key=lambda k: float(np.hypot(*(rc[k] - mid))))


def spikes_in_crop(R, r0, c0, size):
    """The spikes table restricted to COMs inside the crop, with crop coordinates ``y``/``x`` and
    ``kept`` / ``cluster`` filled in for results written before those columns existed."""
    sp = R.spikes
    y = sp["row"].to_numpy(float) - 1.0 - r0
    x = sp["col"].to_numpy(float) - 1.0 - c0
    m = (y >= -0.5) & (y < size - 0.5) & (x >= -0.5) & (x < size - 0.5)
    out = sp[m].copy()
    out["y"], out["x"] = y[m], x[m]
    if "kept" not in out:
        out["kept"] = True
    if "cluster" not in out:
        out["cluster"] = np.where(out["cell_row"] >= 0, 1, 0)
    return out


def _half_max_contour(ax, plane, colour, lw):
    if plane.max() > 0:
        ax.contour(plane / plane.max(), levels=[0.5], colors=colour, linewidths=lw)


def draw_spike_map(ax, R, own_rows, r0, c0, size, neighbours=None, colours=None, title=None):
    """Reference crop with spike COMs and footprint contours.

    ``own_rows`` are drawn in ``OWN_COLOUR`` (dots + cyan contour); ``neighbours`` (default: every
    other cell centred in the crop) each get a colour from ``colours`` for their dots and contour;
    merged-away spikes are gray ``x``, spikes in no cell hollow gray circles. Returns the
    ``{trace_row: colour}`` map used.
    """
    own_rows = [int(k) for k in np.atleast_1d(own_rows)]
    ref = R.images["brightness_reference"][r0:r0 + size, c0:c0 + size]
    vmin, vmax = _stretch(ref)
    ax.imshow(ref, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if neighbours is None:
        neighbours = cells_in_crop(R, r0, c0, size, exclude=own_rows)
    colours = colours or {}
    cmap = {}
    for i, k in enumerate(neighbours):
        cmap[int(k)] = colours.get(int(k), NEIGHBOUR_COLOURS[i % len(NEIGHBOUR_COLOURS)])
    for k in own_rows:
        cmap.setdefault(k, colours.get(k, OWN_COLOUR))
    sp = spikes_in_crop(R, r0, c0, size)
    merged = sp[~sp["kept"].astype(bool)]
    loose = sp[sp["kept"].astype(bool) & (sp["cell_row"] < 0)]
    ax.scatter(merged["x"], merged["y"], s=7, marker="x", c="#9a9a9a", linewidths=0.6)
    ax.scatter(loose["x"], loose["y"], s=9, facecolors="none", edgecolors="#9a9a9a", linewidths=0.6)
    for k, col in cmap.items():
        mine = sp[sp["cell_row"] == k]
        ax.scatter(mine["x"], mine["y"], s=6 if k not in own_rows else 8, c=col, linewidths=0)
        _half_max_contour(ax, R.footprints.plane(k)[r0:r0 + size, c0:c0 + size],
                          "cyan" if (k in own_rows and k not in colours) else col,
                          1.0 if k in own_rows else 0.7)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return cmap


def render(R, picked, movie, out_png, crop=48, lag=8, dpi=200, header=""):
    """Draw the ``len(picked) x 3`` panel figure and save it to ``out_png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps = float(R.attrs["fps"])
    H, W = R.footprints.height, R.footprints.width
    n = len(picked)
    fig_h = 1.9 * max(n, 1) + 1.2
    fig, axes = plt.subplots(max(n, 1), 5, figsize=(20, fig_h), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 4, 1.1], wspace=0.12, hspace=0.45))
    fig.subplots_adjust(top=1 - 0.9 / fig_h, bottom=0.45 / fig_h, left=0.12, right=0.99)
    half = int(round(0.025 * fps))                                      # +/-25 ms zoom
    t_all = np.arange(R.traces.shape[1]) / fps
    for i, (_, row) in enumerate(picked.iterrows()):
        k = int(row["trace_row"])
        ax_ref, ax_map, ax_sta, ax_tr, ax_zoom = axes[i]
        r0, c0 = crop_window(R.centers[k], H, W, crop)
        S = int(min(crop, H, W))
        fp = R.footprints.plane(k)[r0:r0 + S, c0:c0 + S]
        fpn = fp / fp.max() if fp.max() > 0 else fp
        ref = R.images["brightness_reference"][r0:r0 + S, c0:c0 + S]
        spk = R.spikes[R.spikes["cell_row"] == k]
        frames = spk["frame"].to_numpy(np.int64)
        amps = spk["amplitude"].to_numpy(float)
        cr, cc = float(R.centers[k][0]) - 1 - r0, float(R.centers[k][1]) - 1 - c0

        vmin, vmax = _stretch(ref)
        ax_ref.imshow(ref, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax_ref.imshow(fpn, cmap="hot", vmin=0, vmax=1, alpha=0.55 * fpn, interpolation="nearest")
        if fp.max() > 0:
            ax_ref.contour(fpn, levels=[0.5], colors="cyan", linewidths=0.8)
        ax_ref.plot(cc, cr, "+", color="yellow", markersize=7, markeredgewidth=1.0)
        ax_ref.set_title(f"ref  n_px={int(row['n_px'])}  eff={row['eff_px']:.0f}", fontsize=8)
        tag = f"P{row['percentile']:.0f}" if np.isfinite(row["percentile"]) else "named"
        ax_ref.set_ylabel(f"{tag}  SNR {row['snr_median']:.1f}\n"
                          f"σ {row['noise_sigma']:.2f}  spikes {int(row['n_spikes'])}\n"
                          f"{row['cell_id']}", rotation=0, ha="right", va="center", fontsize=7.5,
                          labelpad=6)

        n_nb = len(cells_in_crop(R, r0, c0, S, exclude=[k]))
        draw_spike_map(ax_map, R, [k], r0, c0, S,
                       title=f"spike map  {len(frames)} own, {n_nb} neighbour cell(s)")

        sta = sta_crop(movie, frames, r0, c0, S, lag=lag, amplitudes=amps) if movie is not None else None
        if sta is None:
            ax_sta.text(0.5, 0.5, "no movie" if movie is None else "no usable spike frames",
                        ha="center", va="center", fontsize=8, transform=ax_sta.transAxes)
            ax_sta.set_title("STA", fontsize=8)
        else:
            vmin, vmax = _stretch(sta, 1.0, 99.5)
            ax_sta.imshow(sta, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            if fp.max() > 0:
                ax_sta.contour(fpn, levels=[0.5], colors="cyan", linewidths=0.8)
            ax_sta.plot(cc, cr, "+", color="yellow", markersize=7, markeredgewidth=1.0)
            ax_sta.set_title(f"STA of {min(len(frames), 100)} spikes", fontsize=8)
        for ax in (ax_ref, ax_sta):
            ax.set_xticks([]); ax.set_yticks([])

        tr = R.traces[k].astype(float)
        ax_tr.plot(t_all, tr, "-", color="#1f3b73", linewidth=0.4)
        lo, hi = np.percentile(tr, [0.1, 99.9])
        span = max(hi - lo, 1e-9)
        if frames.size:
            ax_tr.vlines(frames / fps, hi + 0.08 * span, hi + 0.2 * span, color="#d62728",
                         linewidth=0.5)
        ax_tr.set_xlim(0, t_all[-1])
        ax_tr.set_ylim(lo - 0.1 * span, hi + 0.28 * span)
        ax_tr.tick_params(labelsize=7)
        ax_tr.set_ylabel("a.u.", fontsize=7)
        if i == n - 1:
            ax_tr.set_xlabel("time (s)", fontsize=8)
        if frames.size:
            j = int(frames[np.argmax(amps)]) if np.all(np.isfinite(amps)) else int(frames[0])
            a, b = max(j - half, 0), min(j + half + 1, tr.size)
            ax_zoom.plot((np.arange(a, b) - j) * 1000.0 / fps, tr[a:b], "-", color="#1f3b73",
                         linewidth=0.8)
            ax_zoom.plot(0, tr[j], "o", color="#d62728", markersize=3)
            ax_zoom.set_title(f"largest spike, t={j / fps:.2f} s", fontsize=7)
        else:
            ax_zoom.text(0.5, 0.5, "no spikes", ha="center", va="center", fontsize=7,
                         transform=ax_zoom.transAxes)
        ax_zoom.tick_params(labelsize=6)
        if i == n - 1:
            ax_zoom.set_xlabel("ms", fontsize=7)
    fig.suptitle(header, fontsize=9, y=1 - 0.08 / fig_h)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(results_dir, fov_dir=None, n=10, crop=48, min_spikes=3, lag=8, sta=True, dpi=200,
        cells=None, png=PNG, csv=CSV):
    """Build the QC for ``results_dir``; returns the picked-cell table (also written as CSV).
    ``cells`` (list of ids / suffixes / trace rows) replaces the percentile picks."""
    R = load_results(results_dir)
    meta_path = os.path.join(results_dir, "fov_metadata.json")
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    df = cell_table(R)
    picked = pick_named(df, cells) if cells else select_cells(df, n=n, min_spikes=min_spikes)
    movie = find_movie(results_dir, meta, fov_dir) if sta else None
    pool = int(((df["source"] == "spiking") & (df["qc_spikes"] >= min_spikes)
                & np.isfinite(df["snr_median"].to_numpy(float))).sum())
    focus = R.attrs.get("focus_score", meta.get("focus_score"))
    git = str(R.attrs.get("pyali_git", ""))[:7]
    header = (f"{R.attrs.get('fov_id', os.path.basename(os.path.abspath(results_dir)))}   "
              f"focus {focus:.1e}   " if focus is not None else
              f"{R.attrs.get('fov_id', os.path.basename(os.path.abspath(results_dir)))}   ")
    header += (f"pyali {git}   cells {len(df)} (pool: {pool} spiking cells with >= {min_spikes} QC spikes)\n"
               "spike-amplitude SNR = median peak / 1.4826·MAD of the 20 Hz high-passed trace;  "
               "eff = (Σw)²/Σw² effective pixels of the footprint;  "
               "cyan = footprint at half max;  + = cell centre\n"
               f"STA = mean(F[t−{lag}]) − mean(F[t]) = fluorescence drop at the spike, on raw frames "
               "(motion-uncorrected, <= ~1 px);  red ticks = the pipeline's spikes for this cell\n"
               "spike map: yellow = this cell's spike COMs, colours = neighbour cells (dots + footprint "
               "contour), x = merged into a same-frame duplicate, o = in no cell")

    out_dir = os.path.join(results_dir, QC_DIR)
    os.makedirs(out_dir, exist_ok=True)
    render(R, picked, movie, os.path.join(out_dir, png), crop=crop, lag=lag, dpi=dpi, header=header)
    picked[CSV_COLUMNS].to_csv(os.path.join(out_dir, csv), index=False)
    print(f"[qc] {results_dir}: {len(df)} cells, pool {pool}, picked {len(picked)}, "
          f"movie {'yes' if movie is not None else 'no'} -> {out_dir}/{{{png},{csv}}}")
    for _, r in picked.iterrows():
        print(f"  {('P%.0f' % r['percentile']) if np.isfinite(r['percentile']) else 'named':>5}  snr {r['snr_median']:5.2f}  sigma {r['noise_sigma']:.3f}  "
              f"spikes {int(r['n_spikes']):4d}  px {int(r['n_px']):4d}  eff {r['eff_px']:5.1f}  {r['cell_id']}")
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="a pyali result directory (cells.parquet, spikes.parquet, signals.h5)")
    ap.add_argument("--fov", default=None, help="FOV directory holding frames1.bin (default: the one in fov_metadata.json)")
    ap.add_argument("--n", type=int, default=10, help="cells to show, spread over the SNR percentiles")
    ap.add_argument("--crop", type=int, default=48, help="crop size in pixels")
    ap.add_argument("--min-spikes", type=int, default=3, help="QC spikes a cell needs to enter the pool")
    ap.add_argument("--lag", type=int, default=8, help="STA baseline: frames before the spike")
    ap.add_argument("--no-sta", action="store_true", help="skip the spike-triggered average column")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell ids, C0713-style suffixes or trace rows to show "
                         "instead of the percentile picks (written as footprints_traces_cells.png)")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)
    cells = [c for c in (a.cells or "").split(",") if c.strip()] or None
    run(a.results_dir, a.fov, a.n, a.crop, a.min_spikes, a.lag, not a.no_sta, a.dpi, cells=cells,
        png="footprints_traces_cells.png" if cells else PNG,
        csv="footprints_traces_cells.csv" if cells else CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
