#!/usr/bin/env python3
"""Run pyali across the corpus, streaming each FOV's outputs to S3 and freeing the local copy.

Outputs are ~114-364 MB per FOV (``footprint`` is a dense ``(H, W, N)`` float64 array, and
``ALI_Int_Result.mat`` holds 7 full-frame images), so the full corpus is roughly **1 TB**
against 466 GB of local NVMe. Rather than shrink the outputs, each FOV is written to local
scratch, uploaded, and deleted immediately — peak local usage is then a few FOVs, not the run.

Reads go through the Mountpoint mounts (~4 GB/s); the s3fs mounts are ~32x slower and should
not be used as the read root. Writes go to local scratch because Mountpoint is read-only here
and, even writable, supports only sequential writes of new files.

    python scripts/run_corpus.py --keep-csv KEEP.csv --workers 8
    python scripts/run_corpus.py --keep-csv KEEP.csv --days 20260715 --limit 20 --figures
    python scripts/run_corpus.py --keep-csv KEEP.csv --dry-run

Resumable: FOVs whose ``ALI_Result.mat`` already exists under the destination prefix are
skipped, so an interrupted run picks up where it stopped.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pyali.params import Params                                          # noqa: E402
from pyali.pipeline import process_fov                                   # noqa: E402

ROOT_AB = "/mnt/s3ab/AB"                       # Mountpoint; all days except 20260331_dir1
ROOT_WB = "/mnt/s3wb/data"                     # Mountpoint; 20260331_dir1 only
S3_DEFAULT = "s3://spatial-technology-platform/AB/pyali_analysis"

PROFILE = {("20260331_dir1", "16-bit"): "6GP002", ("20260331_dir2", "16-bit"): "6GP002",
           ("20260401", "16-bit"): "6GP002", ("20260401", "8-bit"): "6GP002-8bit",
           ("20260611", "8-bit"): "443screen1", ("20260612", "8-bit"): "443screen1",
           ("20260715", "8-bit"): "443screen2", ("20260716", "8-bit"): "443screen2",
           ("20260717", "8-bit"): "443screen2", ("20260718", "8-bit"): "443screen2"}
FACTORY = {"6GP002": Params.profile_6GP002, "6GP002-8bit": Params.profile_6GP002_8bit,
           "443screen1": Params.profile_443screen1, "443screen2": Params.profile_443screen2}
RESULT = "ALI_Result.mat"


def read_root(day):
    return ROOT_WB if day == "20260331_dir1" else ROOT_AB


def dir_size(path):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _d, fs in os.walk(path) for f in fs)


def _run_one(job):
    """Process one FOV, upload it, delete the local copy. Never raises."""
    day, bit, fov_path, dir_name, keep_row, s3_prefix, scratch, figures, compute_dtype = job
    rec = dict(day=day, bit=bit, dir_name=dir_name, fov_path=fov_path)
    local = os.path.join(scratch, day, dir_name)
    t0 = time.perf_counter()
    try:
        profile = PROFILE[(day, bit)]
        p = FACTORY[profile]()
        if compute_dtype:
            p.compute_dtype = compute_dtype
        fov_dir = os.path.join(read_root(day), fov_path)

        # clamp the protocol ranges to this movie, exactly as run_pyali.py does
        itemsize = 1 if p.read_dtype == "uint8" else 2
        T = os.path.getsize(os.path.join(fov_dir, "frames1.bin")) // (p.nrow * p.ncol * itemsize)
        T -= p.truncate_last
        p.bkg_ranges = [(max(1, a), min(b, T)) for a, b in p.bkg_ranges if max(1, a) <= min(b, T)]
        p.std_ranges = [(max(1, a), min(b, T)) for a, b in p.std_ranges if max(1, a) <= min(b, T)]

        shutil.rmtree(local, ignore_errors=True)                 # drop any partial earlier attempt
        out = process_fov(fov_dir, out_dir=local, p=p, save=True, verbose=False,
                          make_figures=figures, keep_row=keep_row, profile=profile, day=day)
        rec["n_cells"] = int(out["cell_traces"].shape[0])
        rec["n_regions"] = len(out["regions"])
        rec["process_s"] = round(time.perf_counter() - t0, 1)

        if not os.path.exists(os.path.join(local, RESULT)):
            raise RuntimeError(f"{RESULT} missing after process_fov")
        rec["bytes"] = dir_size(local)

        t1 = time.perf_counter()
        dest = f"{s3_prefix.rstrip('/')}/{day}/{dir_name}/"
        cp = subprocess.run(["aws", "s3", "cp", local, dest, "--recursive", "--only-show-errors"],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"upload failed: {cp.stderr.strip()[:300]}")
        rec["upload_s"] = round(time.perf_counter() - t1, 1)
        rec["dest"] = dest

        shutil.rmtree(local, ignore_errors=True)                 # only after a clean upload
        rec.update(ok=True, total_s=round(time.perf_counter() - t0, 1))
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-1200:],
                   total_s=round(time.perf_counter() - t0, 1))
        shutil.rmtree(local, ignore_errors=True)
    return rec


def existing_keys(s3_prefix):
    """Set of ``(day, dir_name)`` already carrying a RESULT under the prefix, for --resume."""
    out = subprocess.run(["aws", "s3", "ls", s3_prefix.rstrip("/") + "/", "--recursive"],
                         capture_output=True, text=True)
    done = set()
    if out.returncode != 0:
        return done                                              # nothing there yet
    base = s3_prefix.split("/", 3)[-1].rstrip("/")               # key prefix inside the bucket
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4 or not parts[-1].endswith(RESULT):
            continue
        rel = parts[-1][len(base):].strip("/")
        bits = rel.split("/")
        if len(bits) >= 3:
            done.add((bits[0], bits[1]))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-csv", required=True, help="the 6504-FOV keep.csv")
    ap.add_argument("--s3-prefix", default=S3_DEFAULT)
    ap.add_argument("--scratch", default="/home/jovyan/corpus_scratch")
    ap.add_argument("--manifest", default="/home/jovyan/corpus_manifest.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--days", nargs="+", default=None, help="restrict to these days")
    ap.add_argument("--bit", choices=["16-bit", "8-bit"], default=None,
                    help="restrict to one bit depth. 20260401 holds BOTH a 312x1200 16-bit set "
                         "and a 1080x1080 8-bit set, so --days alone cannot separate them")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of FOVs")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth FOV (spreads a sample)")
    ap.add_argument("--figures", action="store_true", help="also write the QC figures per FOV")
    ap.add_argument("--compute-dtype", choices=["float64", "float32"], default=None)
    ap.add_argument("--no-resume", action="store_true", help="reprocess FOVs already in S3")
    ap.add_argument("--min-free-gb", type=float, default=60.0,
                    help="abort if local scratch free space drops below this")
    ap.add_argument("--max-restarts", type=int, default=3,
                    help="rebuild the worker pool this many times after an OOM kill before "
                         "giving up on the remaining FOVs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for root in (ROOT_AB, ROOT_WB):
        if not os.path.isdir(root):
            ap.error(f"{root} is not mounted — see the cold-start section of the outputs README "
                     f"(sudo mount-s3 ...). Reading via s3fs instead would be ~32x slower.")

    with open(a.keep_csv) as f:
        rows = [r for r in csv.DictReader(f)]
    if a.days:
        rows = [r for r in rows if r["day"] in a.days]
    if a.bit:
        rows = [r for r in rows if r["bit"] == a.bit]
    rows.sort(key=lambda r: (r["day"], r["fov_path"]))
    if a.stride > 1:
        rows = rows[::a.stride]

    skipped = 0
    if not a.no_resume:
        done = existing_keys(a.s3_prefix)
        if done:
            before = len(rows)
            rows = [r for r in rows if (r["day"], r["dir_name"]) not in done]
            skipped = before - len(rows)
    if a.limit:
        rows = rows[:a.limit]

    os.makedirs(a.scratch, exist_ok=True)
    free_gb = shutil.disk_usage(a.scratch).free / 1e9
    print(f"[corpus] {len(rows)} FOVs to run ({skipped} already in S3, skipped)", flush=True)
    print(f"[corpus] dest {a.s3_prefix}   scratch {a.scratch} ({free_gb:.0f} GB free)   "
          f"workers {a.workers}   figures={a.figures}", flush=True)
    byday = {}
    for r in rows:
        byday[r["day"]] = byday.get(r["day"], 0) + 1
    for d in sorted(byday):
        print(f"           {d:<16} {byday[d]:5}", flush=True)
    if a.dry_run:
        print("[corpus] dry run, stopping here", flush=True)
        return 0
    if free_gb < a.min_free_gb:
        print(f"[corpus] ABORT: only {free_gb:.0f} GB free, need {a.min_free_gb:.0f}", flush=True)
        return 1

    jobs = [(r["day"], r["bit"], r["fov_path"], r["dir_name"], r, a.s3_prefix, a.scratch,
             a.figures, a.compute_dtype) for r in rows]

    t0 = time.perf_counter()
    ok = fail = 0
    total_bytes = 0
    done_total = 0
    mf = open(a.manifest, "a")
    remaining = list(jobs)
    stop = False
    # A worker killed by the OOM reaper breaks the whole pool, and `fut.result()` re-raises it,
    # which would otherwise abandon the entire day. Catch it and rebuild the pool over whatever
    # is still outstanding; completed FOVs are already uploaded, so no work is redone.
    for attempt in range(1, a.max_restarts + 2):
        if not remaining or stop:
            break
        if attempt > 1:
            print(f"[corpus] restarting pool (attempt {attempt}) for {len(remaining)} "
                  f"remaining FOVs", flush=True)
        finished = set()
        try:
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(_run_one, j): j for j in remaining}
                for fut in as_completed(futs):
                    rec = fut.result()
                    rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    mf.write(json.dumps(rec) + "\n"); mf.flush()
                    finished.add((rec["day"], rec["dir_name"]))
                    done_total += 1
                    if rec.get("ok"):
                        ok += 1
                        total_bytes += rec.get("bytes", 0)
                    else:
                        fail += 1
                        print(f"[corpus] FAIL {rec['day']}/{rec['dir_name']}\n"
                              f"{rec.get('error', '')}", flush=True)
                    if done_total % 10 == 0 or done_total == len(jobs):
                        el = time.perf_counter() - t0
                        rate = done_total / el * 3600
                        free = shutil.disk_usage(a.scratch).free / 1e9
                        print(f"[corpus] {done_total}/{len(jobs)}  ok={ok} fail={fail}  "
                              f"{rate:.0f} FOV/h  eta {(len(jobs)-done_total)/max(rate,1e-9):.1f} h"
                              f"  uploaded {total_bytes/1e9:.1f} GB  free {free:.0f} GB", flush=True)
                        if free < a.min_free_gb:
                            print("[corpus] ABORT: local scratch running out", flush=True)
                            stop = True
                            break
        except BrokenProcessPool:
            print(f"[corpus] WORKER DIED (pool broken — almost certainly an OOM kill). "
                  f"{len(finished)} FOVs completed this round.", flush=True)
        remaining = [j for j in remaining if (j[0], j[3]) not in finished]
    if remaining and not stop:
        print(f"[corpus] GAVE UP with {len(remaining)} FOVs unprocessed after "
              f"{a.max_restarts + 1} attempts — lower --workers and re-run (resume will skip "
              f"what is already in S3)", flush=True)
    mf.close()
    el = time.perf_counter() - t0
    print(f"\n[corpus] done in {el/3600:.2f} h: {ok} ok, {fail} failed, "
          f"{total_bytes/1e9:.1f} GB uploaded", flush=True)
    if ok:
        print(f"[corpus] mean {total_bytes/ok/1e6:.0f} MB per FOV  ->  "
              f"{total_bytes/ok*6504/1e12:.2f} TB projected for all 6504", flush=True)
    print(f"[corpus] manifest: {a.manifest}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
