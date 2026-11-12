#!/usr/bin/env python3
"""Real-FOV helpers for the review oracles: a z-projection to identify the field of view, and a
spatial crop that keeps every frame so the full 443screen2 protocol (baseline windows up to frame
8241) still applies while the movie fits a small machine.

    python tests/real_fov.py zproject FOV_DIR --out OUT_DIR [--nrow 800 --ncol 800 --dtype uint8]
    python tests/real_fov.py crop FOV_DIR --out CROP_DIR --size 256 [--row R --col C]

``zproject`` streams the movie once and writes ``zproj_mean.png``, ``zproj_std.png``, a combined
``zproj.png`` with the crop window drawn on it, and ``zproj.npz`` (mean, std, max). ``crop``
picks the ``size``x``size`` window with the most temporal activity (highest summed std
projection) unless ``--row/--col`` give its top-left corner, and writes ``frames1.bin`` plus a
``crop.json`` recording the window.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from pyali.io import open_bin_memmap                                     # noqa: E402

CHUNK = 256                                                               # frames per pass


def project(fov_dir, nrow, ncol, dtype):
    mm = open_bin_memmap(os.path.join(fov_dir, "frames1.bin"), nrow, ncol, dtype)
    T = mm.shape[0]
    s = np.zeros((nrow, ncol)); s2 = np.zeros((nrow, ncol)); mx = np.zeros((nrow, ncol))
    for t0 in range(0, T, CHUNK):
        blk = np.asarray(mm[t0:t0 + CHUNK], dtype=np.float64)
        s += blk.sum(0); s2 += (blk * blk).sum(0); mx = np.maximum(mx, blk.max(0))
    mean = s / T
    std = np.sqrt(np.maximum(s2 / T - mean * mean, 0.0))
    return dict(mean=mean, std=std, max=mx, T=T)


def best_window(std, size):
    """Top-left corner of the ``size``x``size`` window with the largest summed std projection."""
    ii = np.pad(std, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    H, W = std.shape
    sums = (ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size])
    r, c = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(r), int(c)


def _stretch(img, lo=1.0, hi=99.7):
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / max(b - a, 1e-9), 0, 1)


def cmd_zproject(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    os.makedirs(a.out, exist_ok=True)
    pr = project(a.fov_dir, a.nrow, a.ncol, a.dtype)
    np.savez_compressed(os.path.join(a.out, "zproj.npz"), mean=pr["mean"], std=pr["std"], max=pr["max"])
    r, c = best_window(pr["std"], a.size) if a.row is None or a.col is None else (a.row, a.col)
    name = os.path.basename(os.path.normpath(a.fov_dir))
    for key in ("mean", "std"):
        plt.imsave(os.path.join(a.out, f"zproj_{key}.png"), _stretch(pr[key]), cmap="gray")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    for ax, key, title in zip(axes, ("mean", "std"), ("mean projection", "std projection (activity)")):
        ax.imshow(_stretch(pr[key]), cmap="gray", interpolation="nearest")
        ax.add_patch(Rectangle((c, r), a.size, a.size, ec="#ff7f0e", fc="none", lw=1.5))
        ax.set_title(f"{title}  ·  {pr['T']} frames", fontsize=11)
        ax.set_xlabel("column (px)"); ax.set_ylabel("row (px)")
    fig.suptitle(f"{name}  ·  {a.nrow}x{a.ncol} {a.dtype}  ·  box = {a.size}x{a.size} crop at row {r}, col {c}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "zproj.png"), dpi=110)
    print(f"[real_fov] {name}: T={pr['T']}  mean {pr['mean'].min():.1f}-{pr['mean'].max():.1f}  "
          f"best {a.size}x{a.size} window at row {r}, col {c}  ->  {a.out}/zproj.png")
    return 0


def cmd_crop(a):
    mm = open_bin_memmap(os.path.join(a.fov_dir, "frames1.bin"), a.nrow, a.ncol, a.dtype)
    if a.row is None or a.col is None:
        pr = project(a.fov_dir, a.nrow, a.ncol, a.dtype)
        r, c = best_window(pr["std"], a.size)
    else:
        r, c = a.row, a.col
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "frames1.bin")
    with open(out, "wb") as f:
        for t0 in range(0, mm.shape[0], CHUNK):
            np.ascontiguousarray(mm[t0:t0 + CHUNK, r:r + a.size, c:c + a.size]).tofile(f)
    meta = dict(source=os.path.abspath(a.fov_dir), row=r, col=c, size=a.size, T=int(mm.shape[0]),
                dtype=a.dtype)
    with open(os.path.join(a.out, "crop.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[real_fov] crop {a.size}x{a.size} at row {r}, col {c}, {mm.shape[0]} frames -> {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("zproject", "crop"):
        s = sub.add_parser(name)
        s.add_argument("fov_dir")
        s.add_argument("--out", required=True)
        s.add_argument("--nrow", type=int, default=800)
        s.add_argument("--ncol", type=int, default=800)
        s.add_argument("--dtype", default="uint8")
        s.add_argument("--size", type=int, default=256)
        s.add_argument("--row", type=int, default=None, help="window top-left row (default: most active)")
        s.add_argument("--col", type=int, default=None, help="window top-left column")
    a = ap.parse_args(argv)
    return cmd_zproject(a) if a.cmd == "zproject" else cmd_crop(a)


if __name__ == "__main__":
    sys.exit(main())
