#!/usr/bin/env python3
"""Instrumented end-to-end pyali benchmark: raw movie -> traces -> SNR metrics.

Wraps every pipeline stage with a timer so the real ``process_fov`` runs unmodified,
then computes the SNR metrics on the extracted traces. Reports per-stage wall time and
peak RSS (VmHWM, the kernel's own high-water mark).

    python bench_pipeline.py FOV_DIR --profile july8bit|mar16bit [--out DIR]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/home/jovyan/workbench/pyali")

import numpy as np

from pyali import extract, io, preprocess, segmentation
from pyali.metrics import per_cell_snr
from pyali.params import Params
from pyali.pipeline import process_fov

TIMINGS = []


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


def timed(mod, name, label):
    """Wrap ``mod.name`` so each call appends (label, seconds, rss_after, peak) to TIMINGS."""
    fn = getattr(mod, name)

    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        dt = time.perf_counter() - t0
        TIMINGS.append(dict(stage=label, seconds=dt, rss_gb=rss_gb(), peak_rss_gb=peak_rss_gb()))
        print(f"    [bench] {label:<28s} {dt:8.1f} s   rss={rss_gb():6.1f} GB  peak={peak_rss_gb():6.1f} GB",
              flush=True)
        return r

    setattr(mod, name, wrapper)


def instrument():
    timed(io, "read_bin_mov", "load+cast")
    timed(preprocess, "reference_and_correlation_image", "reference+corr_image")
    timed(preprocess, "sharpen", "sharpen")
    timed(preprocess, "adaptive_background", "adaptive_background")
    timed(preprocess, "motion_correct", "motion_correct")
    timed(segmentation, "cell_segmentation", "segmentation")
    timed(extract, "temporal_filter", "temporal_filter")
    timed(extract, "extract_footprints", "extract_footprints")
    timed(extract, "extract_cell_traces", "trace_extraction(pinv)")
    timed(io, "save_mat_v73", "save_mat")
    # pipeline.py imported these names into its own namespace at import time; rebind there too.
    import pyali.pipeline as pl
    pl.io, pl.preprocess, pl.segmentation, pl.extract = io, preprocess, segmentation, extract


PROFILES = {
    "july8bit": lambda: Params.profile_443screen2(),
    "june8bit": lambda: Params.profile_443screen2(nrow=1000, ncol=1000),
    "apr8bit": lambda: Params.profile_443screen2(nrow=1080, ncol=1080),
    "mar16bit": lambda: Params.profile_6GP002(compute_dtype="float32"),
    "mar16bit64": lambda: Params.profile_6GP002(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fov_dir")
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    p = PROFILES[a.profile]()
    out_dir = a.out or os.path.join(a.fov_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    nbytes = os.path.getsize(os.path.join(a.fov_dir, "frames1.bin"))
    itemsize = 1 if p.read_dtype == "uint8" else 2
    T_disk = nbytes // (p.nrow * p.ncol * itemsize)

    # Same clamping run_pyali.py applies, so protocol ranges fit this movie.
    T = T_disk - p.truncate_last
    p.bkg_ranges = [(max(1, x), min(y, T)) for x, y in p.bkg_ranges if max(1, x) <= min(y, T)]
    p.std_ranges = [(max(1, x), min(y, T)) for x, y in p.std_ranges if max(1, x) <= min(y, T)]

    print(f"[bench] {a.tag or a.profile}: {p.nrow}x{p.ncol} {p.read_dtype} -> {p.compute_dtype}, "
          f"{T_disk} frames on disk ({nbytes/1e9:.2f} GB), T={T} after truncation", flush=True)
    print(f"[bench] one float array = {T*p.nrow*p.ncol*(4 if p.compute_dtype=='float32' else 8)/1e9:.1f} GB",
          flush=True)

    instrument()
    t0 = time.perf_counter()
    out = process_fov(a.fov_dir, out_dir=out_dir, p=p, save=True, verbose=False)
    pipeline_s = time.perf_counter() - t0

    traces = out["cell_traces"]
    print(f"[bench] pipeline done in {pipeline_s:.1f} s -> {traces.shape[0]} cells x {traces.shape[1]} frames",
          flush=True)

    t0 = time.perf_counter()
    m = per_cell_snr(traces, p.fps)
    snr_s = time.perf_counter() - t0
    TIMINGS.append(dict(stage="snr_metrics", seconds=snr_s, rss_gb=rss_gb(), peak_rss_gb=peak_rss_gb()))
    print(f"    [bench] {'snr_metrics':<28s} {snr_s:8.1f} s   rss={rss_gb():6.1f} GB  peak={peak_rss_gb():6.1f} GB",
          flush=True)

    def med(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size else float("nan")

    result = dict(
        tag=a.tag or a.profile, profile=a.profile, fov_dir=a.fov_dir,
        nrow=p.nrow, ncol=p.ncol, read_dtype=p.read_dtype, compute_dtype=p.compute_dtype,
        n_frames_disk=int(T_disk), T=int(T), bin_gb=nbytes / 1e9,
        n_regions=len(out["regions"]), n_cells=int(traces.shape[0]),
        pipeline_seconds=pipeline_s, snr_seconds=snr_s, total_seconds=pipeline_s + snr_s,
        peak_rss_gb=peak_rss_gb(), stages=TIMINGS,
        snr_summary=dict(median_noise_sigma=med(m["noise_sigma"]),
                         median_snr_median=med(m["snr_median"]),
                         median_spectral_hf_snr=med(m["spectral_hf_snr"]),
                         median_n_spikes=med(m["n_spikes"])),
    )
    with open(os.path.join(out_dir, "bench.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez(os.path.join(out_dir, "snr_metrics.npz"), **m)

    print(f"\n[bench] TOTAL {pipeline_s + snr_s:.1f} s  peak RSS {peak_rss_gb():.1f} GB  "
          f"cells={traces.shape[0]}", flush=True)
    print(f"[bench] SNR medians: noise_sigma={result['snr_summary']['median_noise_sigma']:.4g}  "
          f"snr_median={result['snr_summary']['median_snr_median']:.3g}  "
          f"spectral_hf_snr={result['snr_summary']['median_spectral_hf_snr']:.3g}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
