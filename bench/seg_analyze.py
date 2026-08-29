#!/usr/bin/env python3
"""Turn the segmentation survey into a guard-threshold decision.

Reports: the region-size distribution, whether "cells" and "giant blobs" separate cleanly,
what each candidate threshold rejects, and whether giant regions really do cluster at the
first/last bursts of a well (the partially-outside-the-well hypothesis).
"""
import argparse
import csv
from collections import Counter, defaultdict

import numpy as np

CANDIDATES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
RANK_ORDER = ["first", "second", "q25", "mid", "q75", "penultimate", "last"]


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("area", "bbox_h", "bbox_w", "bbox_area", "bbox_frac", "patch_area",
                  "patch_frac", "extent", "frame_px", "n_regions", "burst"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
    return rows


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return np.percentile(a, q) if a.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/jovyan/bench/seg_survey.csv")
    a = ap.parse_args()
    rows = load(a.csv)
    bf = np.array([r["bbox_frac"] for r in rows])
    fovs = {(r["day"], r["dir_name"]) for r in rows}
    print(f"{len(rows):,} regions from {len(fovs):,} FOVs\n")

    # ---------- 1. distribution ----------
    print("=== bbox area as % of frame — distribution over ALL regions")
    for q in (50, 75, 90, 95, 99, 99.5, 99.9, 100):
        print(f"   p{q:<5} {100*pct(bf,q):8.3f}%")

    # ---------- 2. separation ----------
    s = np.sort(bf)
    print("\n=== largest 15 regions in the whole survey (% of frame)")
    print("   " + "  ".join(f"{100*v:.2f}" for v in s[-15:]))
    # biggest multiplicative gap in the top tail = natural cut point
    tail = s[s > 0.001]
    if tail.size > 2:
        ratios = tail[1:] / np.maximum(tail[:-1], 1e-12)
        i = int(np.argmax(ratios))
        print(f"\n   largest gap in the tail: {100*tail[i]:.3f}%  ->  {100*tail[i+1]:.3f}%  "
              f"({ratios[i]:.1f}x jump)")

    # ---------- 3. candidate thresholds ----------
    print("\n=== effect of each candidate guard (on bbox area)")
    print(f"   {'thresh':>7}  {'regions cut':>12}  {'% of regions':>12}  {'FOVs touched':>13}  "
          f"{'max patch kept':>15}")
    for t in CANDIDATES:
        cut = bf > t
        touched = {(r["day"], r["dir_name"]) for r, c in zip(rows, cut) if c}
        kept = np.array([r["patch_frac"] for r, c in zip(rows, cut) if not c])
        print(f"   {100*t:6.1f}%  {cut.sum():12,}  {100*cut.mean():11.3f}%  "
              f"{len(touched):6,} / {len(fovs):<6,}  {100*np.max(kept) if kept.size else 0:14.2f}%")

    # ---------- 4. what gets lost ----------
    print("\n=== biggest region that each threshold KEEPS (is it plausibly a cell?)")
    for t in CANDIDATES:
        keep = [r for r in rows if r["bbox_frac"] <= t]
        if not keep:
            continue
        b = max(keep, key=lambda r: r["bbox_frac"])
        print(f"   {100*t:5.1f}%: bbox {b['bbox_h']:.0f}x{b['bbox_w']:.0f}, area {b['area']:.0f} px, "
              f"extent {b['extent']:.2f}  ({b['day']})")

    # ---------- 5. hypothesis: giant regions at well edges ----------
    print("\n=== giant regions (>3% of frame) by burst position within the well")
    tot = Counter(); big = Counter()
    fov_rank = {}
    for r in rows:
        k = (r["day"], r["dir_name"])
        fov_rank[k] = r["burst_rank"]
    for k, rank in fov_rank.items():
        tot[rank] += 1
    seen = set()
    for r in rows:
        k = (r["day"], r["dir_name"])
        if r["bbox_frac"] > 0.03 and k not in seen:
            seen.add(k); big[r["burst_rank"]] += 1
    print(f"   {'position':>13}  {'FOVs':>6}  {'with giant':>11}  {'rate':>8}")
    for rank in RANK_ORDER:
        if tot[rank]:
            print(f"   {rank:>13}  {tot[rank]:6,}  {big[rank]:11,}  {100*big[rank]/tot[rank]:7.1f}%")
    edge = sum(big[r] for r in ("first", "second", "penultimate", "last"))
    edge_n = sum(tot[r] for r in ("first", "second", "penultimate", "last"))
    inter = sum(big[r] for r in ("q25", "mid", "q75"))
    inter_n = sum(tot[r] for r in ("q25", "mid", "q75"))
    if edge_n and inter_n:
        print(f"\n   edge-of-sequence: {100*edge/edge_n:.1f}%   interior: {100*inter/inter_n:.1f}%")

    # ---------- 6. per-day ----------
    print("\n=== per day: FOVs with a >3% region, and the worst region seen")
    byday = defaultdict(list)
    for r in rows:
        byday[(r["day"], r["bit"])].append(r)
    print(f"   {'day':<16} {'bit':>7} {'FOVs':>6} {'w/ giant':>9} {'worst bbox%':>12} {'worst patch%':>13}")
    for k in sorted(byday):
        rs = byday[k]
        fv = {r["dir_name"] for r in rs}
        gf = {r["dir_name"] for r in rs if r["bbox_frac"] > 0.03}
        print(f"   {k[0]:<16} {k[1]:>7} {len(fv):6,} {len(gf):9,} "
              f"{100*max(r['bbox_frac'] for r in rs):11.2f}% {100*max(r['patch_frac'] for r in rs):12.2f}%")

    # ---------- 7. RAM implication ----------
    print("\n=== RAM: patch-movie allocation driven by the worst region")
    for t in [None] + CANDIDATES:
        keep = rows if t is None else [r for r in rows if r["bbox_frac"] <= t]
        if not keep:
            continue
        w = max(keep, key=lambda r: r["patch_frac"])
        # detect_region_aps holds a float32 patch copy AND a float64 copy at once
        px = w["patch_area"]
        T = 8389 if w["frame_px"] == 640000 else 6389
        gb = px * T * (4 + 8) / 1e9
        lab = "no guard" if t is None else f"{100*t:.1f}%"
        print(f"   {lab:>9}: worst patch {100*w['patch_frac']:6.2f}% of frame -> "
              f"{gb:6.2f} GB held during detect_region_aps")


if __name__ == "__main__":
    main()
