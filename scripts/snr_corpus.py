#!/usr/bin/env python3
"""Per-cell SNR metrics over the pyali corpus outputs -- README section 9, stage 1 of 3.

**Post hoc, not in the pipeline.** This reads ``cell_traces`` from each ``ALI_Result.mat``
already in S3 and applies :func:`pyali.metrics.per_cell_snr` unchanged. The extraction pipeline
is never re-run, so a refresh costs minutes rather than days and never reprocesses a movie.

One ``snr_metrics.csv`` is written per FOV, **into the same directory as the ``.mat`` outputs** --
the same depth rule ``fov_metadata.json`` follows -- so downstream aggregation stays a single glob:

    <out_root>/<day>/<fov_dirname>/
        ALI_Int_Result.mat   ALI_Result.mat   fov_metadata.json   snr_metrics.csv

Columns: ``cell_index, noise_sigma, snr_median, spectral_hf_snr, n_spikes``.

    python scripts/snr_corpus.py
    python scripts/snr_corpus.py --days 20260612 --limit 3 --dry-run
    python scripts/snr_corpus.py --workers 8 --force

Resumable and re-runnable: FOVs that already carry ``snr_metrics.csv`` are skipped, so re-running
as further days finish picks up only what is new. Safe to run while an extraction run is still
uploading -- a FOV is eligible only once all three of its output files are present, and S3 objects
appear atomically, so there are no torn reads.

Reads and writes only the analysis prefix. It never touches the raw movies, so the Mountpoint
roots (``/mnt/s3ab``, ``/mnt/s3wb``) are not needed; ~33 GB over s3fs is a few minutes.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pyali.io import load_v73                                            # noqa: E402
from pyali.metrics import per_cell_snr                                   # noqa: E402

S3_DEFAULT = "s3://spatial-technology-platform/AB/pyali_analysis"
READ_ROOT_DEFAULT = "/home/jovyan/spatial-technology-platform/AB/pyali_analysis"

RESULT = "ALI_Result.mat"
INT_RESULT = "ALI_Int_Result.mat"
META = "fov_metadata.json"
METRICS = "snr_metrics.csv"
REQUIRED = (RESULT, INT_RESULT, META)
COLUMNS = ("cell_index", "noise_sigma", "snr_median", "spectral_hf_snr", "n_spikes")


def list_fovs(s3_prefix):
    """Map ``(day, dir_name) -> set(filenames)`` from one recursive listing of the prefix.

    Listing S3 rather than walking the s3fs mount is deliberate: it is one call for the whole
    corpus (a few seconds for thousands of keys) instead of thousands of FUSE round trips.
    """
    out = subprocess.run(["aws", "s3", "ls", s3_prefix.rstrip("/") + "/", "--recursive"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"listing failed: {out.stderr.strip()[:300]}")
    base = s3_prefix.split("/", 3)[-1].rstrip("/")          # key prefix inside the bucket
    fovs = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        rel = parts[-1][len(base):].strip("/")
        bits = rel.split("/")
        if len(bits) != 3:                                  # day/dir_name/file
            continue
        fovs.setdefault((bits[0], bits[1]), set()).add(bits[2])
    return fovs


def _fmt(v):
    """Full-precision, round-trippable float; ``nan`` stays ``nan``.

    NaN is a *result* here, not missing data -- ``per_cell_snr`` returns it where a metric is
    undefined (``snr_median`` with no spike over k*sigma, ``spectral_hf_snr`` on a silent floor) --
    so it is written explicitly rather than blanked, and pandas reads it straight back as NaN.

    Values are written with ``repr``, i.e. the shortest string that round-trips exactly.
    **Readers must ask for that precision back**: ``pd.read_csv(..., float_precision="round_trip")``.
    pandas' default parser is a fast approximate one and was measured drifting up to 50 ULP on
    small ``noise_sigma`` values here -- harmless for a median, but it would make any bit-exact
    A/B against a recomputation fail for no real reason.
    """
    v = float(v)
    return "nan" if not math.isfinite(v) else repr(v)


def build_csv(metrics, n_cells):
    """Render the per-cell metrics as CSV text. ``n_cells == 0`` yields a header-only file."""
    lines = [",".join(COLUMNS)]
    for i in range(n_cells):
        lines.append(",".join((str(i),
                               _fmt(metrics["noise_sigma"][i]),
                               _fmt(metrics["snr_median"][i]),
                               _fmt(metrics["spectral_hf_snr"][i]),
                               str(int(metrics["n_spikes"][i])))))
    return "\n".join(lines) + "\n"


def _write(text, dest_dir, s3_dest, mode):
    """Write ``snr_metrics.csv`` either through the s3fs mount or via the aws CLI.

    s3fs is the default: these are ~30 KB files and one per FOV, so ~1200 ``aws`` invocations
    would cost more in process startup than the whole computation. The file is built in memory
    and written in a single open/write/close, which s3fs turns into one PUT.
    """
    if mode == "s3fs":
        with open(os.path.join(dest_dir, METRICS), "w") as f:
            f.write(text)
        return
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, METRICS)
        with open(tmp, "w") as f:
            f.write(text)
        cp = subprocess.run(["aws", "s3", "cp", tmp, s3_dest, "--only-show-errors"],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"upload failed: {cp.stderr.strip()[:300]}")


def _run_one(job):
    """Compute and write one FOV's metrics. Never raises."""
    day, dir_name, read_root, s3_prefix, mode = job
    rec = dict(day=day, dir_name=dir_name)
    t0 = time.perf_counter()
    try:
        fov_dir = os.path.join(read_root, day, dir_name)
        with open(os.path.join(fov_dir, META)) as f:
            meta = json.load(f)
        fps = meta.get("analysis", {}).get("fps")
        if not fps:
            raise RuntimeError(f"no analysis.fps in {META}")

        traces = np.asarray(load_v73(os.path.join(fov_dir, RESULT), "cell_traces"))
        if traces.ndim != 2:                                # (0,) for a zero-cell FOV
            traces = traces.reshape(0, 0)
        n_cells = int(traces.shape[0])
        rec.update(n_cells=n_cells, n_frames=int(traces.shape[1]), fps=float(fps),
                   n_cells_meta=meta.get("n_cells"), n_frames_meta=meta.get("n_frames_analyzed"))

        # per_cell_snr is applied unchanged, at its documented defaults
        # (hp=20 Hz, k=3 sigma, sig_hi=150 Hz, floor_lo=300 Hz).
        metrics = per_cell_snr(traces, float(fps))
        for m in ("noise_sigma", "snr_median", "spectral_hf_snr"):
            rec[f"n_nan_{m}"] = int(np.count_nonzero(~np.isfinite(metrics[m])))
        rec["n_spikes_total"] = int(np.sum(metrics["n_spikes"]))

        # A zero-cell FOV still gets a header-only CSV, so it is provably done rather than
        # merely skipped -- section 5 notes three FOVs legitimately segment to zero regions.
        _write(build_csv(metrics, n_cells), fov_dir,
               f"{s3_prefix.rstrip('/')}/{day}/{dir_name}/{METRICS}", mode)
        rec.update(ok=True, total_s=round(time.perf_counter() - t0, 2))
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-1200:],
                   total_s=round(time.perf_counter() - t0, 2))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-prefix", default=S3_DEFAULT)
    ap.add_argument("--read-root", default=READ_ROOT_DEFAULT,
                    help="s3fs mount mirroring --s3-prefix (reads, and writes when --upload s3fs)")
    ap.add_argument("--manifest", default="/home/jovyan/snr_corpus_manifest.jsonl")
    ap.add_argument("--upload", choices=["s3fs", "awscli"], default="s3fs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--days", nargs="+", default=None, help="restrict to these days")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of FOVs")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth FOV (spreads a sample)")
    ap.add_argument("--force", action="store_true", help="recompute FOVs that already have a CSV")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(a.read_root):
        ap.error(f"{a.read_root} is not a directory -- pass --read-root")

    fovs = list_fovs(a.s3_prefix)
    keys = sorted(fovs)
    if a.days:
        keys = [k for k in keys if k[0] in a.days]

    # A FOV is eligible only once all three extraction outputs are present. An extraction run
    # uploads them with one `aws s3 cp --recursive`, so they land at slightly different moments;
    # requiring all three is what makes this safe to run against a live run.
    partial = [k for k in keys if not set(REQUIRED).issubset(fovs[k])]
    keys = [k for k in keys if set(REQUIRED).issubset(fovs[k])]

    done = [k for k in keys if METRICS in fovs[k]]
    if not a.force:
        keys = [k for k in keys if METRICS not in fovs[k]]
    if a.stride > 1:
        keys = keys[::a.stride]
    if a.limit:
        keys = keys[:a.limit]

    byday = {}
    for d, _n in keys:
        byday[d] = byday.get(d, 0) + 1
    print(f"[snr] {len(keys)} FOVs to process "
          f"({len(done)} already have {METRICS}{', recomputing' if a.force else ', skipped'}; "
          f"{len(partial)} still uploading, deferred)", flush=True)
    print(f"[snr] prefix {a.s3_prefix}   read {a.read_root}   upload={a.upload}   "
          f"workers {a.workers}", flush=True)
    for d in sorted(byday):
        print(f"        {d:<16} {byday[d]:5}", flush=True)
    if a.dry_run:
        print("[snr] dry run, stopping here", flush=True)
        return 0
    if not keys:
        return 0

    jobs = [(d, n, a.read_root, a.s3_prefix, a.upload) for d, n in keys]
    t0 = time.perf_counter()
    ok = fail = cells = 0
    mf = open(a.manifest, "a")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), start=1):
            rec = fut.result()
            rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if rec.get("ok"):
                ok += 1
                cells += rec.get("n_cells", 0)
            else:
                fail += 1
                print(f"[snr] FAIL {rec['day']}/{rec['dir_name']}\n{rec.get('error', '')}",
                      flush=True)
            if i % 50 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                print(f"[snr] {i}/{len(jobs)}  ok={ok} fail={fail}  {cells} cells  "
                      f"{i / el * 3600:.0f} FOV/h  eta "
                      f"{(len(jobs) - i) / max(i / el, 1e-9) / 60:.1f} min", flush=True)
    mf.close()
    print(f"\n[snr] done in {(time.perf_counter() - t0) / 60:.1f} min: {ok} ok, {fail} failed, "
          f"{cells} cells", flush=True)
    print(f"[snr] manifest: {a.manifest}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
