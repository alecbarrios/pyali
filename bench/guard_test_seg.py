#!/usr/bin/env python3
"""Guard verification, segmentation stage only.

Reproduces exactly the segmentation path ``pipeline.process_fov`` runs — reference image ->
sharpen -> ``cell_segmentation`` -> ``drop_oversized_regions`` -> rebuild ``spatial_footprints``
— on the selected FOVs, using the production functions. Segmentation depends only on the first
``n_ref`` frames, so this needs ~1% of each movie and none of the (currently slow) extraction.

Checks per FOV:
  * region count before the guard matches the survey exactly (segmentation is deterministic);
  * the guard drops exactly the predicted number of regions;
  * every surviving region's bbox is within the limit, and every dropped one exceeded it;
  * ``spatial_footprints`` stays index-aligned with ``regions`` (``extract_footprints`` does
    ``spatial_footprints[c]``, so a misalignment would silently pair a region with another's
    pixels);
  * ``binary_map`` loses exactly the dropped regions' pixels and nothing else;
  * the worst patch ``compute_patch`` would now allocate.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/jovyan/workbench/pyali")

import numpy as np

from pyali import extract, preprocess, segmentation
from pyali.params import Params

ROOT_AB = "/mnt/s3ab/AB"
ROOT_WB = "/mnt/s3wb/data"
KEEP = "/home/jovyan/spatial-technology-platform/AB/pyali_c27fc46_outputs/keep.csv"
_RAW = {"uint8": "<u1", "uint16": "<u2"}


def make_params(day, bit):
    """The §5 profile for this day (compute_dtype float32 throughout, as at scale)."""
    if day in ("20260331_dir1", "20260331_dir2") or (day == "20260401" and bit == "16-bit"):
        return Params.profile_6GP002(compute_dtype="float32")
    if day == "20260401" and bit == "8-bit":
        return Params.profile_6GP002(nrow=1080, ncol=1080, read_dtype="uint8",
                                     compute_dtype="float32", saturation_clip=None)
    if day in ("20260611", "20260612"):
        return Params.profile_443screen1()
    return Params.profile_443screen2()


def run_one(job):
    day, bit, fov_path, dir_name, expect_before, expect_dropped, tag = job
    rec = dict(tag=tag, day=day, bit=bit, dir_name=dir_name,
               expect_before=expect_before, expect_dropped=expect_dropped)
    try:
        p = make_params(day, bit)
        H, W = p.nrow, p.ncol
        root = ROOT_WB if day == "20260331_dir1" else ROOT_AB
        path = os.path.join(root, fov_path, "frames1.bin")

        t0 = time.perf_counter()
        raw = np.fromfile(path, dtype=_RAW[p.read_dtype], count=p.n_ref * H * W)
        t = raw.size // (H * W)
        ref = raw[:t * H * W].reshape(t, H, W).astype(p.compute_dtype)

        # ---- exactly pipeline.process_fov's segmentation path ----
        reference_image, _corr = preprocess.reference_and_correlation_image(ref[:p.n_ref])
        del ref
        *_, sharpened = preprocess.sharpen(reference_image, p.disk_radius, p.gauss_sigma,
                                           p.lap_alpha, p.sharpen_k)
        regions, binary_map, spatial_footprints = segmentation.cell_segmentation(
            sharpened, p.seg_threshold, p.seg_gauss, p.seg_region_size)
        before, bm_before = len(regions), int(binary_map.sum())
        pre_ids = [id(r["PixelList"]) for r in regions]

        regions, binary_map2, dropped = segmentation.drop_oversized_regions(
            regions, binary_map, p.max_region_bbox_frac)
        if dropped:
            spatial_footprints = [r["PixelList"] for r in regions]
        rec["seconds"] = time.perf_counter() - t0

        limit = p.max_region_bbox_frac * H * W
        bbox = lambda r: float(r["BoundingBox"][2]) * float(r["BoundingBox"][3])

        # ---- invariants ----
        rec["regions_before"] = before
        rec["dropped"] = len(dropped)
        rec["regions_after"] = len(regions)
        rec["dropped_bbox_pct"] = sorted(round(100 * bbox(r) / (H * W), 2) for r in dropped)
        rec["max_kept_bbox_pct"] = round(100 * max([bbox(r) for r in regions], default=0) / (H * W), 3)
        rec["matches_survey"] = (before == expect_before and len(dropped) == expect_dropped)
        rec["all_kept_within_limit"] = all(bbox(r) <= limit for r in regions)
        rec["all_dropped_over_limit"] = all(bbox(r) > limit for r in dropped)
        # index alignment: spatial_footprints[c] must be region c's own PixelList
        rec["footprints_aligned"] = (len(spatial_footprints) == len(regions) and
                                     all(spatial_footprints[c] is regions[c]["PixelList"]
                                         for c in range(len(regions))))
        # mask bookkeeping: exactly the dropped regions' pixels cleared, original untouched
        dropped_px = sum(len(r["PixelList"]) for r in dropped)
        rec["mask_true_before"] = bm_before
        rec["mask_true_after"] = int(binary_map2.sum())
        rec["mask_delta_ok"] = (bm_before - int(binary_map2.sum()) == dropped_px)
        rec["input_mask_untouched"] = (int(binary_map.sum()) == bm_before)
        rec["pixellists_unchanged"] = all(
            id(r["PixelList"]) in pre_ids for r in regions)
        # what extract would now allocate for the worst surviving region
        worst_patch = 0
        for r in regions:
            pr, pc, _o = extract.compute_patch(r["Centroid"], r["BoundingBox"], p.patch_size, H, W)
            worst_patch = max(worst_patch, len(pr) * len(pc))
        rec["worst_patch_pct"] = round(100 * worst_patch / (H * W), 3)
        T = 8389 if p.nrow == 800 else 6389
        rec["worst_patch_gb"] = round(worst_patch * T * (4 + 8) / 1e9, 2)
        rec["ok"] = True
    except Exception:
        rec["ok"] = False
        rec["error"] = traceback.format_exc()[-1200:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="/home/jovyan/bench/guard_seg_results.json")
    a = ap.parse_args()

    sel = json.load(open("/home/jovyan/bench/guard_test_fovs.json"))
    keep = {(r["day"], r["dir_name"]): r["fov_path"] for r in csv.DictReader(open(KEEP))}
    jobs = []
    for tag in ("valley", "worst"):
        for i in sel[tag]:
            fp = keep.get((i["day"], i["dir_name"]))
            if not fp:
                print(f"[guard] MISSING keep.csv row: {i['day']}/{i['dir_name']}", flush=True)
                continue
            jobs.append((i["day"], i["bit"], fp, i["dir_name"],
                         i["n_regions"], i["n_over_1pct"], tag))
    print(f"[guard] {len(jobs)} FOVs, segmentation stage only, guard = 1% of frame bbox area\n",
          flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    order = {("valley", 0): 0}
    results.sort(key=lambda r: (r["tag"] != "valley", -(r.get("dropped_bbox_pct") or [0])[-1]))

    hdr = f"{'tag':>6} {'day':<15} {'regions':>16} {'dropped bbox %':>22} {'max kept':>9} {'worst patch':>12}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        if not r.get("ok"):
            print(f"{r['tag']:>6} {r['day']:<15}  FAILED"); print(r.get("error")); continue
        print(f"{r['tag']:>6} {r['day']:<15} "
              f"{r['regions_before']:>6} -> {r['regions_after']:<6} "
              f"{str(r['dropped_bbox_pct']):>22} "
              f"{r['max_kept_bbox_pct']:>8.3f}% "
              f"{r['worst_patch_pct']:>7.2f}% / {r['worst_patch_gb']}GB")

    json.dump(results, open(a.out, "w"), indent=2)
    checks = ["matches_survey", "all_kept_within_limit", "all_dropped_over_limit",
              "footprints_aligned", "mask_delta_ok", "input_mask_untouched",
              "pixellists_unchanged"]
    print("\n=== invariants across all FOVs")
    for c in checks:
        vals = [r.get(c) for r in results if r.get("ok")]
        bad = [r["dir_name"][:40] for r in results if r.get("ok") and not r.get(c)]
        print(f"   {c:<26} {sum(bool(v) for v in vals)}/{len(vals)}"
              + (f"   FAILED: {bad}" if bad else ""))
    zero = [r for r in results if r.get("regions_after") == 0]
    print(f"\n   FOVs left with 0 regions: {len(zero)}"
          + (f"  ({', '.join(r['day'] for r in zero)})" if zero else ""))
    print(f"   completed: {sum(r.get('ok', False) for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
