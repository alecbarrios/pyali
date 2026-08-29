#!/usr/bin/env python3
"""Post-pinv-fix timing + peak RSS, one FOV per bit-depth x HxW combination.

Each shape runs at its profile's OWN default ``compute_dtype`` (16-bit stays float64), so this
measures the corpus as it would actually run today and leaves the float32 question open.

The 312x1200 16-bit / float64 case is the control: with the movie already float64 there was
never a promotion to remove, so its peak RSS should be unchanged by the fix.
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/jovyan/workbench/pyali")

import numpy as np

ROOT_AB = "/mnt/s3ab/AB"
OUT = "/home/jovyan/bench/pinv_out"

# one representative FOV per shape; all confirmed present earlier
SHAPES = [
    ("312x1200 16-bit f64", "20260331_dir2", "6GP002",
     "20260331_dir2/Data_B4/123951_P01_12w_B4_6GP002_DIV36_burst1"),
    ("800x800 8-bit f32", "20260715", "443screen2",
     "20260715/114944P-1_W-A1_443_443screen2_DIV34__burst1"),
    ("1000x1000 8-bit f32", "20260611", "443screen1",
     "20260611/122253_P06_0xSM_C2_DIV55_443GP_burst1"),
    ("1080x1080 8-bit f32", "20260401", "6GP002-8bit",
     "20260401/Data_B1/161507_P02_6w_B1_JF608_6GP002_DIV37_8bit_burst1"),
]


def peak_rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def run_one(job):
    label, day, profile, rel = job
    from pyali import extract, io, preprocess, segmentation
    from pyali.metrics import per_cell_snr
    from pyali.params import Params
    from pyali.pipeline import process_fov
    import pyali.pipeline as PL

    stages = []

    def timed(mod, name, tag):
        fn = getattr(mod, name)

        def w(*a, **kw):
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            stages.append(dict(stage=tag, seconds=round(time.perf_counter() - t0, 1),
                               rss_gb=round(rss_gb(), 1), peak_gb=round(peak_rss_gb(), 1)))
            return r
        setattr(mod, name, w)

    for mod, name, tag in [(io, "read_bin_mov", "load"),
                           (preprocess, "reference_and_correlation_image", "reference"),
                           (preprocess, "sharpen", "sharpen"),
                           (preprocess, "adaptive_background", "adaptive_background"),
                           (preprocess, "motion_correct", "motion_correct"),
                           (segmentation, "cell_segmentation", "segmentation"),
                           (extract, "temporal_filter", "temporal_filter"),
                           (extract, "extract_footprints", "extract_footprints"),
                           (extract, "extract_cell_traces", "trace_extraction")]:
        timed(mod, name, tag)
    PL.io, PL.preprocess, PL.segmentation, PL.extract = io, preprocess, segmentation, extract

    rec = dict(label=label, day=day, profile=profile)
    out_dir = os.path.join(OUT, label.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    try:
        # exactly what run_pyali.py --profile <name> builds
        factory = {"6GP002": Params.profile_6GP002, "6GP002-8bit": Params.profile_6GP002_8bit,
                   "443screen1": Params.profile_443screen1,
                   "443screen2": Params.profile_443screen2}[profile]
        p = factory()
        fov = os.path.join(ROOT_AB, rel)
        itemsize = 1 if p.read_dtype == "uint8" else 2
        T_disk = os.path.getsize(os.path.join(fov, "frames1.bin")) // (p.nrow * p.ncol * itemsize)
        T = T_disk - p.truncate_last
        p.bkg_ranges = [(max(1, x), min(y, T)) for x, y in p.bkg_ranges if max(1, x) <= min(y, T)]
        p.std_ranges = [(max(1, x), min(y, T)) for x, y in p.std_ranges if max(1, x) <= min(y, T)]

        itemf = 4 if p.compute_dtype == "float32" else 8
        rec.update(nrow=p.nrow, ncol=p.ncol, read_dtype=p.read_dtype,
                   compute_dtype=p.compute_dtype, T=int(T),
                   one_array_gb=round(T * p.nrow * p.ncol * itemf / 1e9, 1),
                   guard=p.max_region_bbox_frac)

        t0 = time.perf_counter()
        out = process_fov(fov, out_dir=out_dir, p=p, save=True, verbose=False)
        rec["pipeline_s"] = round(time.perf_counter() - t0, 1)

        traces = out["cell_traces"]
        t0 = time.perf_counter()
        m = per_cell_snr(traces, p.fps)
        rec["snr_s"] = round(time.perf_counter() - t0, 2)

        def med(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            return round(float(np.median(x)), 4) if x.size else None
        rec.update(n_cells=int(traces.shape[0]), stages=stages,
                   peak_rss_gb=round(peak_rss_gb(), 1),
                   total_s=round(rec["pipeline_s"] + rec["snr_s"], 1),
                   snr=dict(noise_sigma=med(m["noise_sigma"]), snr_median=med(m["snr_median"]),
                            spectral_hf_snr=med(m["spectral_hf_snr"])),
                   ok=True)
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-1500:],
                   peak_rss_gb=round(peak_rss_gb(), 1), stages=stages)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="/home/jovyan/bench/pinv_shape_results.json")
    ap.add_argument("--only", default=None, help="run only shapes whose label contains this")
    a = ap.parse_args()
    shapes = [s for s in SHAPES if not a.only or a.only in s[0]]
    print(f"[pinv] {len(shapes)} shapes, {a.workers} at a time\n", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, s): s for s in shapes}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            json.dump(results, open(a.out, "w"), indent=2)
            if r.get("ok"):
                print(f"[pinv] OK   {r['label']:<22} total={r['total_s']:7.1f}s  "
                      f"peak={r['peak_rss_gb']:5.1f}GB  cells={r['n_cells']}", flush=True)
            else:
                print(f"[pinv] FAIL {r['label']}\n{r.get('error')}", flush=True)

    order = {s[0]: i for i, s in enumerate(shapes)}
    results.sort(key=lambda r: order[r["label"]])
    print(f"\n{'shape':<22} {'1x arr':>7} {'peak RSS':>9} {'trace_s':>8} {'tfilt_s':>8} "
          f"{'total':>8} {'cells':>6}")
    print("-" * 76)
    for r in results:
        if not r.get("ok"):
            print(f"{r['label']:<22}  FAILED"); continue
        st = {s["stage"]: s for s in r["stages"]}
        print(f"{r['label']:<22} {r['one_array_gb']:6.1f}G {r['peak_rss_gb']:8.1f}G "
              f"{st.get('trace_extraction',{}).get('seconds',0):7.1f}s "
              f"{st.get('temporal_filter',{}).get('seconds',0):7.1f}s "
              f"{r['total_s']:7.1f}s {r['n_cells']:6}")
    print("\npeak RSS vs the 2x-array floor (temporal_filter holds movie + filtered):")
    for r in results:
        if r.get("ok"):
            print(f"   {r['label']:<22} peak {r['peak_rss_gb']:5.1f} GB   vs 2x = "
                  f"{2*r['one_array_gb']:5.1f} GB   -> ratio {r['peak_rss_gb']/(2*r['one_array_gb']):.2f}")


if __name__ == "__main__":
    main()
