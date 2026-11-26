#!/usr/bin/env python3
"""A/B of two pyali result directories (written for cycle E1; reused by later cycles).

    python e1_ab.py <new_dir> <ref_dir> --out report.json [--label-a E1 --label-b D1]

Reports: whether cells.parquet / spikes.parquet / footprints are byte-identical (and, if not,
how they differ: region counts, spike counts, matched cells); per-cell trace differences in units
of each cell's noise sigma (max, p99.9, fraction of float32-identical samples); per_cell_snr
agreement; stage timings and memory from both fov_metadata.json files.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from pyali.results import load_results                                   # noqa: E402
from pyali.metrics import per_cell_snr                                   # noqa: E402


def meta(d):
    p = os.path.join(d, "fov_metadata.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def match_cells(ca, cb, tol=0.01):
    """Nearest-neighbour match of cell centres (x_px, y_px) within ``tol`` px -> (idx_a, idx_b)."""
    from scipy.spatial import cKDTree
    A = np.column_stack([ca.y_px, ca.x_px]).astype(float)
    B = np.column_stack([cb.y_px, cb.x_px]).astype(float)
    if len(A) == 0 or len(B) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d, j = cKDTree(B).query(A, distance_upper_bound=tol)
    ok = np.isfinite(d)
    ia = np.flatnonzero(ok)
    jb = j[ok]
    _, first = np.unique(jb, return_index=True)             # one-to-one
    return ia[first], jb[first]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir"); ap.add_argument("ref_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="new"); ap.add_argument("--label-b", default="ref")
    a = ap.parse_args(argv)
    A, B = load_results(a.new_dir), load_results(a.ref_dir)
    ma, mb = meta(a.new_dir), meta(a.ref_dir)
    rep = dict(new=a.new_dir, ref=a.ref_dir, label_a=a.label_a, label_b=a.label_b)

    # ---- upstream files ----
    ca, cb = A.cells, B.cells
    rep["n_cells"] = [int(len(ca)), int(len(cb))]
    rep["n_spikes"] = [int(len(A.spikes)), int(len(B.spikes))]
    rep["n_regions"] = [ma.get("n_regions_kept"), mb.get("n_regions_kept")]
    rep["cells_identical"] = bool(len(ca) == len(cb) and ca.reset_index(drop=True).equals(cb.reset_index(drop=True)))
    rep["spikes_identical"] = bool(len(A.spikes) == len(B.spikes)
                                   and A.spikes.reset_index(drop=True).equals(B.spikes.reset_index(drop=True)))
    fa, fb = A.footprints, B.footprints
    rep["footprints_identical"] = bool(fa.n() == fb.n() and np.array_equal(fa.values, fb.values)
                                       and np.array_equal(fa.pixel, fb.pixel) and np.array_equal(fa.offsets, fb.offsets))
    if not rep["cells_identical"]:
        ia, ib = match_cells(ca, cb)
        rep["cells_matched_0p01px"] = int(len(ia))
        if len(ia):
            sa = ca.n_spikes.to_numpy()[ia]; sb = cb.n_spikes.to_numpy()[ib]
            rep["matched_cells_same_n_spikes"] = int((sa == sb).sum())
            rep["matched_cells_same_source"] = int((ca.source.to_numpy()[ia] == cb.source.to_numpy()[ib]).sum())
    else:
        ia = ib = np.arange(len(ca))

    # ---- traces on matched cells ----
    fps = float(A.attrs["fps"])
    if len(ia):
        ta = A.traces[ia].astype(np.float64); tb = B.traces[ib].astype(np.float64)
        mb_ = per_cell_snr(tb, fps); ma_ = per_cell_snr(ta, fps)
        sig = mb_["noise_sigma"]
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
        d = np.abs(ta - tb)
        per_cell_max = d.max(axis=1) / sig
        rep["trace_diff_over_sigma"] = dict(
            max=float(np.nanmax(per_cell_max)), p99_9=float(np.nanpercentile(per_cell_max, 99.9)),
            median_cell_max=float(np.nanmedian(per_cell_max)),
            n_cells_over_1e_3=int(np.nansum(per_cell_max > 1e-3)),
            float32_identical_fraction=float(np.mean(A.traces[ia] == B.traces[ib])),
            n_cells=int(len(ia)))
        for k in ("snr_median", "noise_sigma", "spectral_hf_snr"):
            x, y = ma_[k], mb_[k]
            ok = np.isfinite(x) & np.isfinite(y)
            rel = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1e-12)
            rep[f"{k}_max_rel_diff"] = float(rel.max()) if ok.any() else None
        rep["qc_spikes_identical_fraction"] = float(np.mean(ma_["n_spikes"] == mb_["n_spikes"]))

    # ---- telemetry ----
    for lab, m in ((a.label_a, ma), (a.label_b, mb)):
        rep[f"timing_s_{lab}"] = m.get("timing_s")
        rep[f"peak_rss_gb_{lab}"] = m.get("peak_rss_gb")
        rep[f"stage_peak_gb_{lab}"] = {k: v.get("peak") for k, v in (m.get("memory_gb") or {}).items()}
        rep[f"cpu_s_{lab}"] = m.get("cpu_s")
        rep[f"trace_solve_{lab}"] = m.get("trace_solve")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2, default=str)
    for k in ("n_cells", "n_spikes", "n_regions", "cells_identical", "spikes_identical",
              "footprints_identical", "cells_matched_0p01px", "trace_diff_over_sigma",
              "snr_median_max_rel_diff", "noise_sigma_max_rel_diff", "qc_spikes_identical_fraction",
              f"peak_rss_gb_{a.label_a}", f"peak_rss_gb_{a.label_b}", f"trace_solve_{a.label_a}"):
        if k in rep:
            print(f"{k}: {rep[k]}")
    ta_, tb_ = rep.get(f"timing_s_{a.label_a}") or {}, rep.get(f"timing_s_{a.label_b}") or {}
    for st in ("load", "adaptive_background", "motion_correct", "extract_footprints", "trace_extraction", "total"):
        print(f"  {st:<22} {ta_.get(st, float('nan')):8.1f} s   vs {tb_.get(st, float('nan')):8.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
