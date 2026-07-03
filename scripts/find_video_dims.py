#!/usr/bin/env python3
"""Recover a raw movie's frame dimensions (nrow, ncol) from a headerless .bin.

Method: the correct (nrow, ncol) is the factor pair of the pixels-per-frame that makes
each frame a smooth, coherent image — i.e. it maximizes the average correlation between
horizontally- and vertically-adjacent pixels. Wrong factorizations scramble the rows and
destroy that correlation.

Pixels-per-frame is detected AUTOMATICALLY (you no longer need to know it):
  1. Sidecar text file in the same folder (Luminos/Hamamatsu writes e.g.
     `frames1_dropped_frames.txt` containing `exposure_bytes_per_frame=748800`).
  2. If no sidecar, a signal-based search over divisors of the total pixel count.
You can still override with --pixels-per-frame / --bytes-per-frame.

Usage:
    python find_video_dims.py FRAMES.bin                      # fully automatic
    python find_video_dims.py FRAMES.bin --pixels-per-frame 374400
    python find_video_dims.py FRAMES.bin --bytes-per-frame 748800 --dtype u2
"""
import argparse
import os
import re
import sys

import numpy as np

_ITEMSIZE = {"u1": 1, "u2": 2, "i2": 2, "u4": 4, "i4": 4, "f4": 4, "f8": 8}

# sidecar keys -> how to turn the value into pixels-per-frame (given itemsize)
_BYTE_KEYS = ("exposure_bytes_per_frame", "bytes_per_frame", "frame_bytes", "bytes/frame")
_PIXEL_KEYS = ("pixels_per_frame", "frame_pixels", "pixels/frame")


# --------------------------------------------------------------------------- #
# pixels-per-frame detection
# --------------------------------------------------------------------------- #
def _scan_sidecar(bin_path, itemsize):
    """Look for a sidecar .txt in the .bin's folder and parse a per-frame size key."""
    folder = os.path.dirname(os.path.abspath(bin_path))
    stem = os.path.splitext(os.path.basename(bin_path))[0]
    txts = [f for f in os.listdir(folder) if f.lower().endswith(".txt")]
    # prefer sidecars whose name shares the .bin stem (e.g. frames1_*.txt for frames1.bin)
    txts.sort(key=lambda f: (not f.startswith(stem), f))
    for fn in txts:
        try:
            text = open(os.path.join(folder, fn), "r", errors="ignore").read()
        except OSError:
            continue
        low = text.lower()
        for key in _BYTE_KEYS:
            m = re.search(re.escape(key) + r"\s*[=:]\s*(\d+)", low)
            if m:
                b = int(m.group(1))
                if b % itemsize == 0:
                    return b // itemsize, f"sidecar '{fn}': {key}={b} bytes / {itemsize}"
        for key in _PIXEL_KEYS:
            m = re.search(re.escape(key) + r"\s*[=:]\s*(\d+)", low)
            if m:
                return int(m.group(1)), f"sidecar '{fn}': {key}={m.group(1)}"
    return None, None


def _divisors(P):
    small, large = [], []
    i = 1
    while i * i <= P:
        if P % i == 0:
            small.append(i)
            if i != P // i:
                large.append(P // i)
        i += 1
    return small + large[::-1]


def _signal_search(bin_path, itemsize, dim_lo, dim_hi):
    """Fallback: pick pixels-per-frame + (nrow,ncol) that maximize adjacent-pixel
    correlation, scanning divisors of the total pixel count."""
    P = os.path.getsize(bin_path) // itemsize
    best = None                                        # (score, ppf, nrow, ncol)
    for ppf in _divisors(P):
        if not (dim_lo * dim_lo <= ppf <= dim_hi * dim_hi):
            continue
        frames = P // ppf
        if frames < 2 or frames > 1_000_000:
            continue
        raw = np.fromfile(bin_path, dtype="<u2" if itemsize == 2 else "<u1",
                          count=ppf * 2).astype(np.float64)
        if raw.size < ppf:
            continue
        fr = [raw[i * ppf:(i + 1) * ppf] for i in range(raw.size // ppf)]
        for nr, nc in factor_pairs(ppf, dim_lo, dim_hi):
            s = smoothness(fr, nr, nc)
            if best is None or s > best[0]:
                best = (s, ppf, nr, nc)
    if best is None:
        return None, None
    return best[1], (f"signal search: best adjacency score {best[0]:.3f} at "
                     f"{best[2]}x{best[3]}")


def detect_pixels_per_frame(bin_path, itemsize, dim_lo, dim_hi):
    ppf, how = _scan_sidecar(bin_path, itemsize)
    if ppf is not None:
        return ppf, how
    return _signal_search(bin_path, itemsize, dim_lo, dim_hi)


# --------------------------------------------------------------------------- #
# dimension ranking
# --------------------------------------------------------------------------- #
def factor_pairs(pxpf, lo, hi):
    pairs = set()
    d = 1
    while d * d <= pxpf:
        if pxpf % d == 0:
            a, b = d, pxpf // d
            if lo <= a <= hi and lo <= b <= hi:
                pairs.add((a, b)); pairs.add((b, a))
        d += 1
    return sorted(pairs)


def smoothness(frames, nrow, ncol):
    scores = []
    for fr in frames:
        img = fr.reshape(nrow, ncol)                    # C-order: ncol is the fast axis
        h = np.corrcoef(img[:, :-1].ravel(), img[:, 1:].ravel())[0, 1]
        v = np.corrcoef(img[:-1, :].ravel(), img[1:, :].ravel())[0, 1]
        scores.append((h + v) / 2.0)
    return float(np.mean(scores))


def recover_dims(path, pixels_per_frame, dtype="u2", n_probe=5, lo=100, hi=2100):
    raw = np.fromfile(path, dtype="<" + dtype, count=pixels_per_frame * n_probe)
    n_have = raw.size // pixels_per_frame
    if n_have == 0:
        raise ValueError("file smaller than one frame; check pixels-per-frame/dtype")
    frames = [raw[i * pixels_per_frame:(i + 1) * pixels_per_frame].astype(np.float64)
              for i in range(n_have)]
    pairs = factor_pairs(pixels_per_frame, lo, hi)
    if not pairs:
        raise ValueError(f"no factor pairs of {pixels_per_frame} in [{lo},{hi}]")
    return sorted(((smoothness(frames, nr, nc), nr, nc) for nr, nc in pairs), reverse=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bin", help="path to the raw .bin movie")
    ap.add_argument("--pixels-per-frame", type=int, default=None)
    ap.add_argument("--bytes-per-frame", type=int, default=None)
    ap.add_argument("--dtype", default="u2", choices=list(_ITEMSIZE),
                    help="sample dtype (default u2 = uint16 little-endian)")
    ap.add_argument("--probe-frames", type=int, default=5)
    ap.add_argument("--min", type=int, default=100, help="min plausible dimension")
    ap.add_argument("--max", type=int, default=2100, help="max plausible dimension")
    a = ap.parse_args(argv)

    itemsize = _ITEMSIZE[a.dtype]
    if a.pixels_per_frame is not None:
        pxpf, how = a.pixels_per_frame, "user --pixels-per-frame"
    elif a.bytes_per_frame is not None:
        pxpf, how = a.bytes_per_frame // itemsize, "user --bytes-per-frame"
    else:
        pxpf, how = detect_pixels_per_frame(a.bin, itemsize, a.min, a.max)
        if pxpf is None:
            ap.error("could not auto-detect pixels-per-frame; pass --pixels-per-frame "
                     "or --bytes-per-frame (check a sidecar .txt for exposure_bytes_per_frame)")

    print(f"pixels/frame = {pxpf}   [{how}]   dtype = {a.dtype}\n")
    ranked = recover_dims(a.bin, pxpf, a.dtype, a.probe_frames, a.min, a.max)
    print(f"  {'nrow':>6} {'ncol':>6}   score")
    for s, nr, nc in ranked[:8]:
        print(f"  {nr:6d} {nc:6d}   {s:.4f}")
    best = ranked[0]
    print(f"\nBEST -> nrow={best[1]}, ncol={best[2]}")
    if len(ranked) > 1 and best[0] - ranked[1][0] < 0.05:
        print("WARNING: top two candidates are close; verify by eye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
