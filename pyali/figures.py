"""Presentation-quality result figures for a processed field of view.

Produces four static plots (PNG) plus one interactive plot (HTML):

  * detected_regions.png       - segmentation mask with region centroids + bounding boxes
  * coms.png                   - action-potential centers of mass on the reference image
  * cell_traces.png            - normalized per-cell traces, stacked
  * center_of_cell_regions.png - footprint centers on the reference image
  * cell_traces.html           - interactive cell traces (zoom/pan, click a legend entry to
                                 hide/isolate a trace, hover for values); needs ``plotly``

Region centroids/bounding boxes are 1-indexed; 1 is subtracted when drawing on the
0-indexed image axes.
"""
import os

import matplotlib
matplotlib.use("Agg")                                 # render to files, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np


def _rescale(x):
    """Linearly map an array to [0, 1] (a constant array maps to zeros)."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _jet_rgb(frac):
    """'rgb(r,g,b)' string for a value in [0, 1] on the jet colormap."""
    r, g, b, _ = plt.cm.jet(float(frac))
    return f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"


def _title(ax, text):
    ax.set_title(text, fontsize=15, fontweight="bold")


def fig_detected_regions(binary_map, regions, path, dpi=150):
    """Binary segmentation mask with red centroids, green bounding boxes, and region labels.

    binary_map : (H, W) bool/uint8 array
    regions    : list of dicts with 'Centroid' [col, row] and 'BoundingBox' [x, y, w, h]
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(np.asarray(binary_map), cmap="gray", interpolation="nearest")
    for b, r in enumerate(regions, 1):
        cx, cy = float(r["Centroid"][0]) - 1, float(r["Centroid"][1]) - 1
        x_ul, y_ul, w, h = (float(v) for v in r["BoundingBox"])
        ax.plot(cx, cy, "r*", ms=6)
        ax.add_patch(Rectangle((x_ul - 1, y_ul - 1), w, h, ec="lime", fc="none", lw=1))
        ax.text(cx + 5, cy + 5, str(b), color="r", fontsize=7)
    _title(ax, "Detected Regions with Centroids and Bounding Boxes")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_coms(reference_image, COMs, path, dpi=150):
    """Action-potential centers of mass scattered on the reference image.

    reference_image : (H, W) float array
    COMs            : (K, >=2) array; column 0 = row, column 1 = column (1-indexed)
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(np.asarray(reference_image), cmap="gray", interpolation="nearest")
    if len(COMs):
        C = np.asarray(COMs)
        ax.scatter(C[:, 1] - 1, C[:, 0] - 1, s=15, c="r", edgecolors="k", linewidths=0.5)
    _title(ax, "COMs")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_cell_traces(cell_traces, path, fps=800.0, drop_last=100, dpi=150):
    """Per-cell traces, each rescaled to [0, 1] and offset by its index (jet colormap).

    cell_traces : (N, T) float array
    fps         : frames per second, for the time axis
    drop_last   : number of final frames to omit from the plot
    """
    ct = np.asarray(cell_traces)
    n = ct.shape[0]
    fig, ax = plt.subplots(figsize=(13, max(4.0, min(0.16 * n + 2, 40))))
    if n:
        T = ct.shape[1]
        time = np.arange(T) / fps
        cmap = plt.cm.jet(np.linspace(0, 1, n))
        end = T - drop_last if T > drop_last else T
        for i in range(n):
            ax.plot(time[:end], _rescale(ct[i, :end]) + i - 0.5, color=cmap[i], lw=1.2)
        ax.set_xlim(0, time.max()); ax.set_ylim(0, n + 1)
        ax.set_yticks(np.arange(0.5, n, 5))
        ax.set_yticklabels([str(k) for k in range(1, n + 1, 5)])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cell Traces")
    _title(ax, "Normalized Cell Traces from Original Video")
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_cell_traces_interactive(cell_traces, path, fps=800.0, drop_last=100, stride=1):
    """Interactive stacked cell traces as a self-contained HTML page (via plotly).

    Zoom/pan with the mouse; click a legend entry to hide a trace, double-click to isolate it;
    hover to read values. Returns the path, or None if plotly is not installed.

    cell_traces : (N, T) float array
    fps         : frames per second, for the time axis
    drop_last   : number of final frames to omit
    stride      : plot every ``stride``-th sample (>1 shrinks the file for very long movies)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[pyali] plotly not installed; skipping interactive cell_traces.html "
              "(`pip install plotly`). The static PNG was still written.")
        return None
    ct = np.asarray(cell_traces)
    n = ct.shape[0]
    if n == 0:
        return None
    T = ct.shape[1]
    end = T - drop_last if T > drop_last else T
    time = (np.arange(end) / fps)[::stride]
    fig = go.Figure()
    for i in range(n):
        y = (_rescale(ct[i, :end]) + i - 0.5)[::stride]
        fig.add_trace(go.Scattergl(
            x=time, y=y, mode="lines", name=f"cell {i + 1}",
            line=dict(width=1, color=_jet_rgb(i / max(n - 1, 1))),
            hovertemplate=f"cell {i + 1}<br>t=%{{x:.3f}} s<br>%{{y:.3f}}<extra></extra>"))
    fig.update_layout(
        title="Normalized Cell Traces (interactive)",
        xaxis_title="Time (s)", yaxis_title="Cell Traces (stacked, rescaled)",
        height=max(700, min(16 * n, 4000)), hovermode="closest", template="plotly_white",
        legend=dict(title="click to toggle", itemsizing="constant"))
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def fig_center_of_regions(reference_image, footprint_center, path, dpi=150):
    """Footprint centers as numbered circles (jet colormap) on the reference image.

    footprint_center : (N, 2) array of [row, column] (1-indexed)
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(np.asarray(reference_image), cmap="gray", interpolation="nearest")
    fc = np.asarray(footprint_center)
    if len(fc):
        cmap = plt.cm.jet(np.linspace(0, 1, len(fc)))
        for t, (row, col) in enumerate(fc):
            ax.add_patch(Circle((col - 1, row - 1), 3, ec=cmap[t], fc="none", lw=1.5))
            ax.text(col - 1 + 5, row - 1 + 5, str(t + 1), color=cmap[t], fontsize=7)
    _title(ax, "Center of cell regions")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def save_result_figures(out_dir, reference_image, regions, binary_map, COMs,
                        footprint_center, cell_traces, fps=800.0, dpi=150):
    """Write all result figures into ``out_dir``; returns the list of output paths.

    Produces the four PNGs and (if plotly is available) the interactive cell_traces.html.
    """
    os.makedirs(out_dir, exist_ok=True)
    j = lambda name: os.path.join(out_dir, name)
    paths = [
        fig_detected_regions(binary_map, regions, j("detected_regions.png"), dpi),
        fig_coms(reference_image, COMs, j("coms.png"), dpi),
        fig_cell_traces(cell_traces, j("cell_traces.png"), fps, dpi=dpi),
        fig_center_of_regions(reference_image, footprint_center, j("center_of_cell_regions.png"), dpi),
    ]
    html = fig_cell_traces_interactive(cell_traces, j("cell_traces.html"), fps)
    if html:
        paths.append(html)
    return paths
