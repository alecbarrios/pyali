#!/usr/bin/env python3
"""Aggregate the per-FOV SNR metrics into one table plus roll-ups -- README section 9, stage 2 of 3.

Joins ``fov_metadata.json`` to ``snr_metrics.csv`` for every FOV under the analysis prefix and
writes a single ``cells_all.parquet`` (one row per cell, carrying day / timestamp / plate / well /
DIV / batch / burst / bit depth / dims / profile), then summarises it.

**Median + IQR, at FOV level first, then rolled up**: FOV -> well -> plate -> calendar day ->
whole set. Summarising per FOV before aggregating stops FOVs with many cells from dominating a
well -- cell counts here range from 8 to 1240 in the same well.

Every level above the FOV is computed from the **FOV medians**, not by re-taking a median of the
level below it. So a well is the median over its FOVs and a plate is the median over *its* FOVs,
rather than a median of well-medians; each level therefore reports a directly comparable statistic
with an honest ``n_fovs``, and no level is a median of medians of medians.

**NaN is reported as its own statistic, never dropped.** ``per_cell_snr`` returns NaN where a
metric is undefined -- ``snr_median`` when no spike clears k*sigma, ``spectral_hf_snr`` on a silent
floor -- so the NaN rate *is* a result (the fraction of segmented cells with no detectable
activity). ``n_cells`` and ``n_nan`` accompany the median and IQR at every level, and NaN counts
are always taken from the raw cell rows, never from the summarised ones.

    python scripts/snr_aggregate.py
    python scripts/snr_aggregate.py --days 20260715 --out-dir /tmp/snr

Re-runnable: as further days finish, re-running picks up every FOV then present and refreshes the
tables. ``manifest.json`` records exactly what each version covered.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from pyali import metrics as _metrics                                    # noqa: E402
from snr_corpus import META, METRICS, REQUIRED, list_fovs                # noqa: E402
from snr_corpus import READ_ROOT_DEFAULT, S3_DEFAULT                     # noqa: E402

OUT_DEFAULT = "/home/jovyan/spatial-technology-platform/AB/pyali_3fc51c7_outputs/snr_summary"
KEEP_DEFAULT = "/home/jovyan/spatial-technology-platform/AB/pyali_3fc51c7_outputs/keep.csv"

METRIC_COLS = ("noise_sigma", "snr_median", "spectral_hf_snr")
# Metadata carried through to every cell row, so the parquet is self-contained.
FOV_FIELDS = ("day", "dir_name", "timestamp", "time_of_day", "plate", "plate_num", "well",
              "well_source", "burst", "div", "batch_id", "cell_line", "dose", "genotype",
              "bit", "dims", "n_frames", "fov_path")
ANALYSIS_FIELDS = ("profile", "fps", "nrow", "ncol", "compute_dtype")


def expected_per_well(keep_csv):
    """``(day, plate_num, well) -> n_FOVs`` the corpus should eventually hold, from ``keep.csv``.

    Lets a partially-processed well be labelled as such rather than sitting silently beside a
    complete one. ``keep.csv`` stores the plate as a float-like string (``"1.0"``), whereas
    ``fov_metadata.json`` carries both ``plate`` (``"P-1"``) and an integer ``plate_num``; the
    integer is the reliable join key.
    """
    exp = {}
    if not os.path.exists(keep_csv):
        return exp
    keep = pd.read_csv(keep_csv, usecols=["day", "plate", "well"], dtype=str)
    for day, plate, well in keep.itertuples(index=False):
        try:
            pn = int(float(plate))
        except (TypeError, ValueError):
            continue
        exp[(day, pn, str(well))] = exp.get((day, pn, str(well)), 0) + 1
    return exp


def read_fov(args):
    """Read one FOV's metadata + metrics, returning a per-cell DataFrame (or ``None``)."""
    day, dir_name, read_root = args
    d = os.path.join(read_root, day, dir_name)
    try:
        with open(os.path.join(d, META)) as f:
            meta = json.load(f)
        # float_precision="round_trip" is required: snr_corpus writes shortest-round-tripping
        # reprs, and pandas' default parser drifts up to 50 ULP on small noise_sigma values.
        df = pd.read_csv(os.path.join(d, METRICS), float_precision="round_trip")
    except Exception as e:                                   # noqa: BLE001 - reported, not raised
        return dir_name, None, None, f"{type(e).__name__}: {e}"
    ana = meta.get("analysis", {})
    for k in FOV_FIELDS:
        df[k] = meta.get(k)
    for k in ANALYSIS_FIELDS:
        df[k] = ana.get(k)
    df["n_frames_analyzed"] = meta.get("n_frames_analyzed")
    df["n_regions_kept"] = meta.get("n_regions_kept")
    df["n_regions_dropped"] = meta.get("n_regions_dropped")
    # `meta` is returned alongside the frame because a zero-cell FOV yields an EMPTY frame, and
    # an empty frame carries no metadata rows -- the FOV would then disappear from every summary.
    return dir_name, df, meta, None


def _q(v, p):
    return float(np.nanpercentile(v, p)) if len(v) else float("nan")


def _stats(values, prefix, out):
    """median / q25 / q75 / IQR of the finite entries of ``values`` into dict ``out``."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    out[f"{prefix}_median"] = float(np.median(v)) if v.size else float("nan")
    out[f"{prefix}_q25"] = _q(v, 25)
    out[f"{prefix}_q75"] = _q(v, 75)
    out[f"{prefix}_iqr"] = out[f"{prefix}_q75"] - out[f"{prefix}_q25"]
    return out


def fov_summary(cells, metas):
    """One row per FOV: median + IQR over that FOV's cells, plus n_cells / n_nan per metric.

    Driven by ``metas`` -- one entry per FOV read -- rather than by grouping ``cells``. A FOV whose
    segmentation yielded **zero cells** contributes no cell rows, so a groupby over ``cells`` would
    drop it silently; section 5 predicts exactly such FOVs (their only region was the oversized
    artifact). Zero cells is a *result*, so the FOV appears with ``n_cells=0`` and NaN metrics.
    """
    rows = []
    groups = dict(list(cells.groupby("dir_name", sort=False))) if len(cells) else {}
    empty = cells.iloc[0:0]
    for meta in metas:
        dir_name = meta.get("dir_name")
        g = groups.get(dir_name, empty)
        ana = meta.get("analysis", {})
        r = {"dir_name": dir_name}
        for k in FOV_FIELDS:
            if k != "dir_name":
                r[k] = meta.get(k)
        for k in ANALYSIS_FIELDS:
            r[k] = ana.get(k)
        for k in ("n_frames_analyzed", "n_regions_kept", "n_regions_dropped"):
            r[k] = meta.get(k)
        r["n_cells"] = int(len(g))
        for m in METRIC_COLS:
            _stats(g[m].values, m, r)
            n_nan = int(g[m].isna().sum())
            r[f"{m}_n_nan"] = n_nan
            r[f"{m}_nan_frac"] = n_nan / len(g) if len(g) else float("nan")
        r["n_spikes_total"] = int(g["n_spikes"].sum())
        r["n_spikes_median"] = float(g["n_spikes"].median())
        rows.append(r)
    return pd.DataFrame(rows)


def roll_up(fovs, cells, by, label):
    """Summarise FOV medians over ``by``; NaN counts come from the raw cell rows.

    ``by`` is a list of columns present in both frames. Every level uses the same recipe, so
    well / plate / day / overall are directly comparable.
    """
    rows = []
    cell_groups = cells.groupby(by, sort=False) if by else [((), cells)]
    cell_map = {k if isinstance(k, tuple) else (k,): g for k, g in cell_groups}
    fov_groups = fovs.groupby(by, sort=False) if by else [((), fovs)]
    for key, g in fov_groups:
        key = key if isinstance(key, tuple) else (key,)
        r = {"level": label}
        for col, val in zip(by, key):
            r[col] = val
        cg = cell_map.get(key, g.iloc[0:0])
        r["n_fovs"] = int(len(g))
        r["n_cells"] = int(g["n_cells"].sum())
        _stats(g["n_cells"].values, "cells_per_fov", r)
        for m in METRIC_COLS:
            _stats(g[f"{m}_median"].values, m, r)          # over FOV medians
            n_nan = int(cg[m].isna().sum())                # from raw cells
            r[f"{m}_n_nan"] = n_nan
            r[f"{m}_nan_frac"] = n_nan / len(cg) if len(cg) else float("nan")
        rows.append(r)
    df = pd.DataFrame(rows)
    return df.sort_values(by).reset_index(drop=True) if by else df


def git_state(repo):
    def _g(*a):
        try:
            return subprocess.run(["git", "-C", repo] + list(a), capture_output=True,
                                  text=True).stdout.strip() or None
        except Exception:                                    # noqa: BLE001
            return None
    return {"describe": _g("describe", "--tags", "--dirty"),
            "commit": _g("rev-parse", "HEAD"),
            "branch": _g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_g("status", "--porcelain"))}


def snr_defaults():
    """The ``per_cell_snr`` keyword defaults, read from the function rather than hardcoded."""
    import inspect
    sig = inspect.signature(_metrics.per_cell_snr)
    return {k: v.default for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-prefix", default=S3_DEFAULT)
    ap.add_argument("--read-root", default=READ_ROOT_DEFAULT)
    ap.add_argument("--keep-csv", default=KEEP_DEFAULT, help="for expected FOVs per well")
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()

    t0 = time.perf_counter()
    fovs_s3 = list_fovs(a.s3_prefix)
    keys = sorted(k for k, v in fovs_s3.items() if set(REQUIRED).issubset(v) and METRICS in v)
    missing = sorted(k for k, v in fovs_s3.items() if set(REQUIRED).issubset(v)
                     and METRICS not in v)
    if a.days:
        # --days is a scope, so it must filter `missing` too. Otherwise a day deliberately left
        # out -- e.g. another extraction run in flight on a day this summary does not cover --
        # is reported as an unmeasured gap in this summary's own coverage, which it is not.
        keys = [k for k in keys if k[0] in a.days]
        missing = [k for k in missing if k[0] in a.days]
    print(f"[agg] {len(keys)} FOVs with {METRICS}; {len(missing)} extracted but not yet measured "
          f"(run snr_corpus.py)", flush=True)
    if a.days:
        print(f"[agg] scope pinned to days: {' '.join(sorted(a.days))}", flush=True)
    if not keys:
        return 1

    frames, metas, errors = [], [], []
    with ThreadPoolExecutor(a.threads) as ex:
        for i, (dir_name, df, meta, err) in enumerate(
                ex.map(read_fov, [(d, n, a.read_root) for d, n in keys]), start=1):
            if err:
                errors.append((dir_name, err))
            else:
                frames.append(df)
                metas.append(meta)
            if i % 250 == 0:
                print(f"[agg] read {i}/{len(keys)}", flush=True)
    for dir_name, err in errors:
        print(f"[agg] READ FAIL {dir_name}: {err}", flush=True)

    # Concatenate only non-empty frames: an all-empty entry contributes no rows anyway and makes
    # pandas warn about dtype inference. Zero-cell FOVs are carried by `metas` instead.
    nonempty = [f for f in frames if len(f)]
    cells = pd.concat(nonempty, ignore_index=True) if nonempty else frames[0]
    n_zero = len(frames) - len(nonempty)
    print(f"[agg] {len(cells)} cell rows from {len(frames)} FOVs "
          f"({n_zero} with zero cells) ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Label well completeness. Wells are keyed (day, plate_num, well) -- 20260331_dir1/P01/A1 and
    # 20260715/P-1/A1 are different wells and must never merge. Observed counts come from `metas`,
    # not from `cells`, so a zero-cell FOV still counts toward its well's n_fovs.
    exp = expected_per_well(a.keep_csv)
    obs = {}
    for m in metas:
        k = (m.get("day"), m.get("plate_num"), m.get("well"))
        obs[k] = obs.get(k, 0) + 1
    idx = list(zip(cells["day"], cells["plate_num"], cells["well"]))
    cells["n_fovs_in_well"] = [obs.get(k, 0) for k in idx]
    cells["n_fovs_expected_in_well"] = [exp.get(k) for k in idx]
    cells["well_complete"] = [None if exp.get(k) is None else obs.get(k, 0) >= exp[k]
                              for k in idx]

    os.makedirs(a.out_dir, exist_ok=True)
    cells.to_parquet(os.path.join(a.out_dir, "cells_all.parquet"), index=False)

    # Built from `metas` so a well whose only FOVs are zero-cell still appears.
    per_well = pd.DataFrame([
        {"day": k[0], "plate_num": k[1], "well": k[2], "n_fovs_in_well": v,
         "n_fovs_expected_in_well": exp.get(k),
         "well_complete": None if exp.get(k) is None else v >= exp[k]}
        for k, v in obs.items()])

    fovs = fov_summary(cells, metas).merge(per_well, on=["day", "plate_num", "well"], how="left")
    fovs = fovs.sort_values(["day", "plate_num", "well", "burst"]).reset_index(drop=True)
    fovs.to_csv(os.path.join(a.out_dir, "fov_summary.csv"), index=False)

    levels = [(["day", "plate", "plate_num", "well"], "well", "well_summary.csv"),
              (["day", "plate", "plate_num"], "plate", "plate_summary.csv"),
              (["day"], "day", "day_summary.csv"),
              ([], "overall", "overall_summary.csv")]
    tables = {}
    for by, label, fname in levels:
        t = roll_up(fovs, cells, by, label)
        if label == "well":                      # carry completeness onto the well table itself
            t = t.merge(per_well.drop(columns=["n_fovs_in_well"]),
                        on=["day", "plate_num", "well"], how="left")
        t.to_csv(os.path.join(a.out_dir, fname), index=False)
        tables[label] = t
        print(f"[agg] {fname:22} {len(t):4} rows", flush=True)

    wells = tables["well"]
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": os.path.basename(__file__),
        "s3_prefix": a.s3_prefix,
        "keep_csv": a.keep_csv,
        "days_requested": sorted(a.days) if a.days else None,
        "pyali": git_state(os.path.dirname(_HERE)),
        "per_cell_snr_params": snr_defaults(),
        "totals": {"n_fovs": int(len(fovs)), "n_cells": int(len(cells)),
                   "n_wells": int(len(wells)), "n_days": int(fovs["day"].nunique()),
                   "n_fovs_zero_cell": int((fovs["n_cells"] == 0).sum()),
                   "n_fovs_extracted_not_measured": len(missing),
                   "n_read_errors": len(errors)},
        "nan": {m: {"n_nan": int(cells[m].isna().sum()),
                    "nan_frac": float(cells[m].isna().mean())} for m in METRIC_COLS},
        "days": [], "outputs": ["cells_all.parquet", "fov_summary.csv", "well_summary.csv",
                                "plate_summary.csv", "day_summary.csv", "overall_summary.csv",
                                "manifest.json"],
    }
    # Iterate `fovs`, not `cells`: a zero-cell FOV has no cell rows but is still a measured FOV.
    for day, fg in fovs.groupby("day", sort=True):
        wd = wells[wells["day"] == day]
        manifest["days"].append({
            "day": day, "n_fovs": int(len(fg)), "n_cells": int(fg["n_cells"].sum()),
            "n_fovs_zero_cell": int((fg["n_cells"] == 0).sum()),
            "bit": fg["bit"].iloc[0], "dims": fg["dims"].iloc[0],
            "profile": fg["profile"].iloc[0], "fps": float(fg["fps"].iloc[0]),
            "wells": [{"plate": r["plate"], "well": r["well"], "n_fovs": int(r["n_fovs"]),
                       "n_fovs_expected": (None if pd.isna(r["n_fovs_expected_in_well"])
                                           else int(r["n_fovs_expected_in_well"])),
                       "complete": (None if pd.isna(r["well_complete"])
                                    else bool(r["well_complete"])),
                       "n_cells": int(r["n_cells"])}
                      for _, r in wd.iterrows()],
        })
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n[agg] {a.out_dir}", flush=True)
    print(f"[agg] {len(cells)} cells / {len(fovs)} FOVs / {len(wells)} wells in "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    for m in METRIC_COLS:
        n = manifest["nan"][m]["n_nan"]
        print(f"[agg]   NaN {m:18} {n:7} ({manifest['nan'][m]['nan_frac'] * 100:.2f}%)",
              flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
