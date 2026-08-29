#!/usr/bin/env python3
"""Segmentation-region size survey across every in-scope acquisition day.

Segmentation depends ONLY on the first ``n_ref`` frames: ``pipeline.py`` builds
``reference_image`` from ``movie[:n_ref]``, sharpens it, and segments the sharpened image —
all before ``adaptive_background`` touches anything. So this reproduces the exact regions the
full pipeline would find while reading ~1% of each movie.

Emits one row per region with its area, bounding box, and the patch that ``compute_patch``
would allocate, so the guard threshold can be chosen on the real distribution.
"""
import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/jovyan/workbench/pyali")

import numpy as np

from pyali import extract, preprocess, segmentation
from pyali.params import Params

KEEP = "/home/jovyan/spatial-technology-platform/AB/pyali_c27fc46_outputs/keep.csv"
# Fast Mountpoint roots (~4 GB/s) instead of the s3fs mounts (~120 MB/s).
ROOT_AB = "/mnt/s3ab/AB"
ROOT_WB = "/mnt/s3wb/data"

# §5 of the outputs README: profile per day/shape. Only geometry + dtype matter for
# segmentation (seg_threshold/seg_gauss/seg_region_size are identical across all profiles).
GROUPS = {
    ("20260331_dir1", "16-bit"): dict(nrow=312, ncol=1200, read="uint8" if False else "uint16", profile="6GP002"),
    ("20260331_dir2", "16-bit"): dict(nrow=312, ncol=1200, read="uint16", profile="6GP002"),
    ("20260401", "16-bit"):      dict(nrow=312, ncol=1200, read="uint16", profile="6GP002"),
    ("20260401", "8-bit"):       dict(nrow=1080, ncol=1080, read="uint8", profile="6GP002"),
    ("20260611", "8-bit"):       dict(nrow=1000, ncol=1000, read="uint8", profile="443screen1"),
    ("20260612", "8-bit"):       dict(nrow=1000, ncol=1000, read="uint8", profile="443screen1"),
    ("20260715", "8-bit"):       dict(nrow=800, ncol=800, read="uint8", profile="443screen2"),
    ("20260716", "8-bit"):       dict(nrow=800, ncol=800, read="uint8", profile="443screen2"),
    ("20260717", "8-bit"):       dict(nrow=800, ncol=800, read="uint8", profile="443screen2"),
    ("20260718", "8-bit"):       dict(nrow=800, ncol=800, read="uint8", profile="443screen2"),
}

_RAW = {"uint8": "<u1", "uint16": "<u2"}
N_REF = 600


def fov_root(day):
    return ROOT_WB if day == "20260331_dir1" else ROOT_AB


def read_ref_frames(path, nrow, ncol, read_dtype, n=N_REF):
    """Read only the first ``n`` frames — segmentation needs nothing else."""
    raw = np.fromfile(path, dtype=_RAW[read_dtype], count=n * nrow * ncol)
    t = raw.size // (nrow * ncol)
    return raw[:t * nrow * ncol].reshape(t, nrow, ncol).astype(np.float32)


def survey_one(job):
    day, bit, rel, dir_name, plate, well, burst, rank, cfg = job
    H, W = cfg["nrow"], cfg["ncol"]
    # keep.csv fov_path already starts with the day for both trees, so join it whole:
    #   AB tree -> /mnt/s3ab/AB/<day>/.../frames1.bin
    #   dir1    -> /mnt/s3wb/data/20260331_dir1/.../frames1.bin
    path = os.path.join(fov_root(day), rel, "frames1.bin")
    try:
        t0 = time.perf_counter()
        ref = read_ref_frames(path, H, W, cfg["read"])
        if ref.shape[0] < 50:
            return [], dict(day=day, rel=rel, error=f"only {ref.shape[0]} frames")
        p = Params(nrow=H, ncol=W)
        reference_image, _corr = preprocess.reference_and_correlation_image(ref)
        del ref                      # ~11 GB/worker at 1080x1080; drop it before sharpening
        _b, _i, _bl, _h, _l, _ln, sharpened = preprocess.sharpen(
            reference_image, p.disk_radius, p.gauss_sigma, p.lap_alpha, p.sharpen_k)
        regions, _bw, _sf = segmentation.cell_segmentation(
            sharpened, p.seg_threshold, p.seg_gauss, p.seg_region_size)
        dt = time.perf_counter() - t0

        frame_px = H * W
        out = []
        for r in regions:
            x_ul, y_ul, w, h = (float(v) for v in r["BoundingBox"])
            bbox_area = w * h
            pr, pc, _o = extract.compute_patch(r["Centroid"], r["BoundingBox"], p.patch_size, H, W)
            patch_area = len(pr) * len(pc)
            out.append(dict(
                day=day, bit=bit, plate=plate, well=well, burst=burst, burst_rank=rank,
                dir_name=dir_name, n_regions=len(regions), frame_px=frame_px,
                area=r["Area"], bbox_h=h, bbox_w=w, bbox_area=bbox_area,
                bbox_frac=bbox_area / frame_px, patch_area=patch_area,
                patch_frac=patch_area / frame_px,
                extent=(r["Area"] / bbox_area) if bbox_area else np.nan))
        return out, dict(day=day, rel=rel, n_regions=len(regions), seconds=dt)
    except Exception as e:
        return [], dict(day=day, rel=rel, error=repr(e))


def pick_jobs(per_group):
    with open(KEEP) as f:
        rows = list(csv.DictReader(f))
    jobs = []
    for (day, bit), cfg in GROUPS.items():
        sel = [r for r in rows if r["day"] == day and r["bit"] == bit]
        # group by (plate, well) and sample burst positions within each well's own sequence
        byw = {}
        for r in sel:
            m = re.search(r"burst(\d+)", r["dir_name"])
            if not m:
                continue
            byw.setdefault((r["plate"], r["well"]), []).append((int(m.group(1)), r))
        for key, lst in byw.items():
            lst.sort(key=lambda t: t[0])
            n = len(lst)
            if n == 0:
                continue
            # first two, three interior quantiles, last two -> tests the edge-of-well hypothesis
            idx = {0: "first", 1: "second", n - 2: "penultimate", n - 1: "last",
                   n // 4: "q25", n // 2: "mid", (3 * n) // 4: "q75"}
            for i, rank in sorted(idx.items()):
                if not (0 <= i < n):
                    continue
                burst, r = lst[i]
                jobs.append((day, bit, r["fov_path"], r["dir_name"], r["plate"], r["well"],
                             burst, rank, cfg))
    # de-duplicate (short wells can map several ranks to one FOV)
    seen, uniq = set(), []
    for j in jobs:
        k = (j[0], j[2])
        if k not in seen:
            seen.add(k); uniq.append(j)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-group", type=int, default=7)
    ap.add_argument("--out", default="/home/jovyan/bench/seg_survey.csv")
    a = ap.parse_args()

    jobs = pick_jobs(a.per_group)
    print(f"[survey] {len(jobs)} FOVs across {len(GROUPS)} day/shape groups", flush=True)
    est = sum(N_REF * j[8]["nrow"] * j[8]["ncol"] * (1 if j[8]["read"] == "uint8" else 2)
              for j in jobs) / 1e9
    print(f"[survey] ~{est:.0f} GB to read ({N_REF} frames per FOV)", flush=True)

    rows, logs = [], []
    t0 = time.perf_counter()
    # submit/as_completed rather than map: one worker death (e.g. an OOM kill) then loses only
    # that FOV instead of aborting the whole sweep with BrokenProcessPool.
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(survey_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            j = futs[fut]
            try:
                rs, log = fut.result()
            except Exception as e:
                rs, log = [], dict(day=j[0], rel=j[2], error=repr(e))
            rows.extend(rs); logs.append(log)
            if (i + 1) % 50 == 0 or i == len(jobs) - 1:
                errs = sum("error" in l for l in logs)
                print(f"[survey] {i+1}/{len(jobs)}  regions={len(rows)}  errors={errs}  "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)

    if rows:
        with open(a.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader(); wtr.writerows(rows)
    with open(a.out.replace(".csv", "_fovs.csv"), "w", newline="") as f:
        keys = sorted({k for l in logs for k in l})
        wtr = csv.DictWriter(f, fieldnames=keys); wtr.writeheader(); wtr.writerows(logs)
    bad = [l for l in logs if "error" in l]
    print(f"[survey] done in {time.perf_counter()-t0:.0f}s: {len(rows)} regions from "
          f"{len(logs)-len(bad)} FOVs, {len(bad)} errors -> {a.out}", flush=True)
    for l in bad[:5]:
        print("   ERROR", l, flush=True)


if __name__ == "__main__":
    main()
