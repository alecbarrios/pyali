#!/usr/bin/env python3
"""Fastpath A/B: patch-local filtering + sorting-network median, vs the saved baseline runs.

Baseline = the commit-e90596b runs already on disk under pinv_out/ (global temporal_filter,
per-index np.median). This re-runs the same FOVs with patch_local_filter=True and compares
cell_traces / footprint / footprint_center element-for-element.

Both changes are meant to be exactly value-preserving:
  * the moving median takes a sorting-network path for window=8 (verified bit-identical);
  * the temporal filter runs per patch instead of globally, and the filter is per-pixel along
    time, so a spatially-sliced filter equals slicing the globally-filtered movie.
So anything short of array_equal on cell_traces is a bug, not a tolerance question.
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
BASE = "/home/jovyan/bench/pinv_out"
OUT = "/home/jovyan/bench/fastpath_out"

SHAPES = [
    ("312x1200 16-bit f64", "6GP002",
     "20260331_dir2/Data_B4/123951_P01_12w_B4_6GP002_DIV36_burst1"),
    ("800x800 8-bit f32", "443screen2",
     "20260715/114944P-1_W-A1_443_443screen2_DIV34__burst1"),
    ("1000x1000 8-bit f32", "443screen1",
     "20260611/122253_P06_0xSM_C2_DIV55_443GP_burst1"),
    ("1080x1080 8-bit f32", "6GP002-8bit",
     "20260401/Data_B1/161507_P02_6w_B1_JF608_6GP002_DIV37_8bit_burst1"),
]


def peak_rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024 / 1024
    return float("nan")


def run_one(job):
    label, profile, rel = job
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
            stages.append(dict(stage=tag, seconds=round(time.perf_counter() - t0, 1)))
            return r
        setattr(mod, name, w)

    for mod, name, tag in [(extract, "temporal_filter", "temporal_filter_global"),
                           (extract, "extract_footprints", "extract_footprints"),
                           (extract, "extract_cell_traces", "trace_extraction")]:
        timed(mod, name, tag)
    PL.extract = extract

    rec = dict(label=label, profile=profile)
    out_dir = os.path.join(OUT, label.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)
    try:
        factory = {"6GP002": Params.profile_6GP002, "6GP002-8bit": Params.profile_6GP002_8bit,
                   "443screen1": Params.profile_443screen1,
                   "443screen2": Params.profile_443screen2}[profile]
        p = factory()
        assert p.patch_local_filter is True
        fov = os.path.join(ROOT_AB, rel)
        itemsize = 1 if p.read_dtype == "uint8" else 2
        T = os.path.getsize(os.path.join(fov, "frames1.bin")) // (p.nrow * p.ncol * itemsize)
        T -= p.truncate_last
        p.bkg_ranges = [(max(1, x), min(y, T)) for x, y in p.bkg_ranges if max(1, x) <= min(y, T)]
        p.std_ranges = [(max(1, x), min(y, T)) for x, y in p.std_ranges if max(1, x) <= min(y, T)]

        t0 = time.perf_counter()
        out = process_fov(fov, out_dir=out_dir, p=p, save=True, verbose=False)
        rec["pipeline_s"] = round(time.perf_counter() - t0, 1)
        rec["peak_rss_gb"] = round(peak_rss_gb(), 1)
        rec["stages"] = stages
        rec["n_cells"] = int(out["cell_traces"].shape[0])
        itemf = 4 if p.compute_dtype == "float32" else 8
        rec["one_array_gb"] = round(T * p.nrow * p.ncol * itemf / 1e9, 1)

        # ---- compare against the saved baseline ----
        b = io.load_v73(os.path.join(BASE, label.replace(" ", "_"), "ALI_Result.mat"))
        cmp = {}
        for k, new in (("cell_traces", out["cell_traces"]),
                       ("footprint", out["footprint"]),
                       ("footprint_center", out["footprint_center"])):
            old = b[k]
            same_shape = tuple(old.shape) == tuple(np.asarray(new).shape)
            cmp[k] = dict(shape_old=list(old.shape), shape_new=list(np.asarray(new).shape),
                          same_shape=bool(same_shape))
            if same_shape and old.size:
                d = np.abs(np.asarray(old, float) - np.asarray(new, float))
                cmp[k]["identical"] = bool(np.array_equal(np.asarray(old), np.asarray(new)))
                cmp[k]["max_abs_diff"] = float(d.max())
                den = np.abs(np.asarray(old, float)).max() or 1.0
                cmp[k]["max_rel_diff"] = float(d.max() / den)
            elif same_shape:
                cmp[k]["identical"] = True
                cmp[k]["max_abs_diff"] = 0.0
                cmp[k]["max_rel_diff"] = 0.0
        rec["compare"] = cmp

        m = per_cell_snr(out["cell_traces"], p.fps)

        def med(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)]
            return round(float(np.median(x)), 4) if x.size else None
        rec["snr"] = dict(noise_sigma=med(m["noise_sigma"]), snr_median=med(m["snr_median"]),
                          spectral_hf_snr=med(m["spectral_hf_snr"]))
        rec["ok"] = True
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-1500:])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="/home/jovyan/bench/fastpath_ab_results.json")
    a = ap.parse_args()
    print(f"[ab] {len(SHAPES)} shapes, {a.workers} at a time\n", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, s): s for s in SHAPES}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            json.dump(results, open(a.out, "w"), indent=2)
            if r.get("ok"):
                ct = r["compare"]["cell_traces"]
                print(f"[ab] OK   {r['label']:<22} {r['pipeline_s']:7.1f}s peak={r['peak_rss_gb']:5.1f}GB "
                      f"cells={r['n_cells']:3} traces_identical={ct.get('identical')}", flush=True)
            else:
                print(f"[ab] FAIL {r['label']}\n{r.get('error')}", flush=True)
    print("\n[ab] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
