#!/usr/bin/env python3
"""Per-stage profile of a HIGH-region FOV — the regime the corpus actually lives in.

Every previous per-stage measurement came from FOVs with 15-40 cells, which turned out to be
the unrepresentative `burst1` fields that image partly outside the well. The median 20260715
FOV segments into ~578 regions. This profiles one of those, with timers inside
``extract_footprints`` so the per-region inner loops are visible rather than lumped together.

Timings are INCLUSIVE (a wrapper around each function), so nested entries double-count their
children; the report marks the nesting and derives self-time where it matters.
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/home/jovyan/workbench/pyali")

import numpy as np

from pyali import extract, io, preprocess, segmentation
from pyali.params import Params
from pyali.pipeline import process_fov

FOV = "/mnt/s3ab/AB/20260715/144039_P-1_W-C4_WT_443screen2_DIV34__burst65"
OUT = "/home/jovyan/bench/profile_out"

T = defaultdict(float)
C = defaultdict(int)


def wrap(mod, name, tag):
    fn = getattr(mod, name)

    def w(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            T[tag] += time.perf_counter() - t0
            C[tag] += 1
    setattr(mod, name, w)
    return fn


def main():
    os.makedirs(OUT, exist_ok=True)
    # top-level stages
    for mod, name, tag in [
        (io, "read_bin_mov", "load"),
        (preprocess, "reference_and_correlation_image", "reference+corr"),
        (preprocess, "sharpen", "sharpen"),
        (preprocess, "adaptive_background", "adaptive_background"),
        (preprocess, "motion_correct", "motion_correct"),
        (segmentation, "cell_segmentation", "segmentation"),
        (extract, "extract_footprints", "TOTAL extract_footprints"),
        (extract, "extract_cell_traces", "TOTAL trace_extraction"),
        (io, "save_mat_v73", "save_mat (gzip)"),
    ]:
        wrap(mod, name, tag)
    # inside extract_footprints, per region
    for name, tag in [
        ("temporal_filter_patch", "  ├ temporal_filter_patch"),
        ("detect_region_aps", "  ├ detect_region_aps"),
        ("dedup_close_peaks", "  ├ dedup_close_peaks"),
        ("com_via_svd", "  ├ com_via_svd"),
        ("cluster_footprints", "  ├ cluster_footprints"),
        ("region_fallback_footprint", "  ├ region_fallback_footprint"),
        ("build_selection_map", "  ├ build_selection_map"),
    ]:
        wrap(extract, name, tag)
    # the suspected Python-loop hot spots (nested inside the above)
    for name, tag in [
        ("movmedian_time", "      · movmedian_time (in patch filter)"),
        ("findpeaks", "      · findpeaks (in detect_region_aps)"),
        ("forward_rolling_min_subtract", "      · forward_rolling_min_subtract (in detect)"),
        ("uniquetol_reps", "      · uniquetol_reps (in dedup)"),
        ("svd_denoise_ap_stack", "      · svd_denoise_ap_stack (in com_via_svd)"),
        ("region_grow_brightest", "      · region_grow_brightest (in com_via_svd)"),
        ("_weighted_com", "      · _weighted_com (in com_via_svd)"),
        ("project_movie", "      · project_movie (in pinv)"),
    ]:
        wrap(extract, name, tag)
    import pyali.pipeline as PL
    PL.io, PL.preprocess, PL.segmentation, PL.extract = io, preprocess, segmentation, extract

    p = Params.profile_443screen2()
    itemsize = 1
    n = os.path.getsize(os.path.join(FOV, "frames1.bin")) // (p.nrow * p.ncol * itemsize)
    n -= p.truncate_last
    p.bkg_ranges = [(max(1, a), min(b, n)) for a, b in p.bkg_ranges if max(1, a) <= min(b, n)]
    p.std_ranges = [(max(1, a), min(b, n)) for a, b in p.std_ranges if max(1, a) <= min(b, n)]

    print(f"[prof] {FOV.split('/')[-1]}  {p.nrow}x{p.ncol} T={n}", flush=True)
    t0 = time.perf_counter()
    out = process_fov(FOV, out_dir=OUT, p=p, save=True, verbose=False)
    total = time.perf_counter() - t0

    nreg = len(out["regions"])
    ncell = int(out["cell_traces"].shape[0])
    naps = int(out["APs"].shape[0])
    print(f"[prof] total {total:.1f}s   regions={nreg}  cells={ncell}  APs={naps}\n", flush=True)

    print(f"{'stage':<46}{'sec':>9}{'% total':>9}{'calls':>9}{'ms/call':>10}")
    print("-" * 83)
    order = ["load", "reference+corr", "sharpen", "adaptive_background", "motion_correct",
             "segmentation", "TOTAL extract_footprints",
             "  ├ temporal_filter_patch", "      · movmedian_time (in patch filter)",
             "  ├ build_selection_map",
             "  ├ detect_region_aps", "      · findpeaks (in detect_region_aps)",
             "      · forward_rolling_min_subtract (in detect)",
             "  ├ dedup_close_peaks", "      · uniquetol_reps (in dedup)",
             "  ├ com_via_svd", "      · svd_denoise_ap_stack (in com_via_svd)",
             "      · region_grow_brightest (in com_via_svd)",
             "      · _weighted_com (in com_via_svd)",
             "  ├ cluster_footprints", "  ├ region_fallback_footprint",
             "TOTAL trace_extraction", "      · project_movie (in pinv)", "save_mat (gzip)"]
    for k in order:
        if C[k]:
            print(f"{k:<46}{T[k]:9.1f}{100*T[k]/total:8.1f}%{C[k]:9,}{1000*T[k]/C[k]:10.2f}")
    acct = sum(T[k] for k in ("load", "reference+corr", "sharpen", "adaptive_background",
                              "motion_correct", "segmentation", "TOTAL extract_footprints",
                              "TOTAL trace_extraction", "save_mat (gzip)"))
    print("-" * 83)
    print(f"{'accounted for':<46}{acct:9.1f}{100*acct/total:8.1f}%")
    ef = T["TOTAL extract_footprints"]
    kids = sum(T[k] for k in ("  ├ temporal_filter_patch", "  ├ detect_region_aps",
                              "  ├ dedup_close_peaks", "  ├ com_via_svd",
                              "  ├ cluster_footprints", "  ├ region_fallback_footprint",
                              "  ├ build_selection_map"))
    print(f"\nextract_footprints {ef:.1f}s = {kids:.1f}s in the timed children "
          f"+ {ef-kids:.1f}s elsewhere in the loop")
    gil = sum(T[k] for k in ("      · findpeaks (in detect_region_aps)",
                             "      · forward_rolling_min_subtract (in detect)",
                             "      · region_grow_brightest (in com_via_svd)",
                             "      · uniquetol_reps (in dedup)"))
    print(f"pure-Python inner loops (findpeaks + rolling-min + region_grow + uniquetol): "
          f"{gil:.1f}s = {100*gil/total:.1f}% of the FOV")
    json.dump(dict(fov=FOV, total_s=total, n_regions=nreg, n_cells=ncell, n_aps=naps,
                   seconds={k: round(v, 2) for k, v in T.items()}, calls=dict(C)),
              open("/home/jovyan/bench/profile_highN.json", "w"), indent=2)
    print("\n[prof] saved /home/jovyan/bench/profile_highN.json", flush=True)


if __name__ == "__main__":
    main()
