#!/usr/bin/env python3
"""Run pyali over one run-id of an inventory, streaming each FOV's outputs to S3.

    python scripts/build_inventory.py --root /mnt/stp/20260716 --experiment-id 443screen2 --out inv.csv
    python scripts/run_corpus.py --inventory inv.csv --run-id P4_A2 \\
        --s3-prefix s3://spatial-technology-platform/AB/pyali-443screen2 --dry-run
    python scripts/run_corpus.py --inventory inv.csv --run-id P4_A2 --s3-prefix s3://... --workers 8

One subprocess per FOV (``scripts/run_pyali.py``, which owns profile/bit-depth/range logic), in a
thread pool: wait for RAM and scratch disk -> run with a per-process RSS watchdog -> upload the
output directory -> read the done-marker back from S3 -> delete the local copy. Each FOV lands at
``<s3_prefix>/Plate_<plate>/Well_<well>/<fov_name>/``; the run's manifest, FOV index and logs go
to ``<s3_prefix>/_runs/<run_id>/``. Resume: FOVs whose ``cells.parquet`` is already in S3 are
skipped. Everything after ``--`` on the command line is passed to ``run_pyali.py`` unchanged.

A Snakefile could wrap the same per-FOV unit for a cluster run; this script is the single-machine
version.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pyali.metadata import Identity                                      # noqa: E402
from pyali.telemetry import avail_ram_gb, rss_gb                         # noqa: E402

MARKER = "cells.parquet"                       # the done-marker, see pyali/results.py
METADATA = "fov_metadata.json"
RUN_PYALI = os.path.join(_HERE, "run_pyali.py")
_TRUE = {"true", "1", "yes", "y"}


# --------------------------------------------------------------------------- #
# Inventory -> FOV table
# --------------------------------------------------------------------------- #
def reroot(src_dir, day, src_root=None):
    """Re-root ``src_dir`` under ``src_root`` by the path from the ``day`` component onward
    (``/mnt/stp/20260716/<fov>`` -> ``<src_root>/20260716/<fov>``). ``None`` -> unchanged."""
    if not src_root:
        return src_dir
    parts = src_dir.rstrip("/").split("/")
    if day not in parts:
        raise ValueError(f"day {day!r} is not a component of {src_dir!r}; cannot re-root")
    i = len(parts) - 1 - parts[::-1].index(day)              # last occurrence (nested twins)
    return os.path.join(src_root, *parts[i:])


def load_run(inventory_csv, run_id, src_root=None, experiment_id=None):
    """Rows of ``inventory_csv`` with ``run_id`` (included and excluded), sorted by
    (plate, well, burst, fov_name), each with ``fov`` (= burst), 1-based ``fov_index``,
    ``include`` (bool), ``identity`` and ``src`` (re-rooted ``src_dir``).

    Excluded rows keep their slot in ``fov_index`` so numbering is stable across re-runs.
    ``experiment_id`` defaults to the batch token in each directory name (``443screen2``).
    """
    from pyali.metadata import parse_dir_name
    with open(inventory_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["run_id"] == run_id]
    if not rows:
        raise ValueError(f"no rows with run_id {run_id!r} in {inventory_csv}")
    rows.sort(key=lambda r: (int(r["plate"]), r["well"], int(r["burst"] or 0), r["fov_name"]))
    for i, r in enumerate(rows, 1):
        r["plate"] = int(r["plate"])
        r["fov"] = int(r["burst"] or 0)
        r["fov_index"] = i
        r["include"] = str(r.get("include", "")).strip().lower() in _TRUE
        r["src"] = reroot(r["src_dir"], r["day"], src_root)
        exp = experiment_id or parse_dir_name(r["fov_name"])["batch_id"] or "unknown"
        r["identity"] = Identity(experiment_id=exp, plate=r["plate"], well=r["well"], fov=r["fov"],
                                 fov_index=i, batch=r.get("batch") or "")
        r["fov_id"] = r["identity"].fov_id
    return rows


def dest_prefix(s3_prefix, plate, well):
    return f"{s3_prefix.rstrip('/')}/Plate_{int(plate)}/Well_{well}"


# --------------------------------------------------------------------------- #
# S3 state and the per-FOV command
# --------------------------------------------------------------------------- #
def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def existing_done(s3_prefix, plate, well):
    """FOV names under ``<s3_prefix>/Plate_<plate>/Well_<well>/`` whose directory holds the
    done-marker (``<fov_name>/cells.parquet``). A failed listing means nothing is there yet."""
    out = sh(["aws", "s3", "ls", dest_prefix(s3_prefix, plate, well) + "/", "--recursive"])
    done = set()
    if out.returncode != 0:
        return done
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1].endswith("/" + MARKER):
            done.add(parts[-1].split("/")[-2])
    return done


def fov_command(python, src, out_dir, identity, profile=None, figures=False, extra=()):
    """The ``run_pyali.py`` command line for one FOV."""
    cmd = [python, RUN_PYALI, src, "--out", out_dir,
           "--experiment-id", identity.experiment_id, "--plate", str(identity.plate),
           "--well", identity.well, "--fov", str(identity.fov), "--fov-index", str(identity.fov_index)]
    if profile:
        cmd += ["--profile", profile]
    if figures:
        cmd.append("--figures")
    return cmd + list(extra)


# --------------------------------------------------------------------------- #
# Resources: RSS watchdog and launch gate (rss_gb / avail_ram_gb live in pyali.telemetry)
# --------------------------------------------------------------------------- #
def watch_rss(proc, max_gb, state, interval=10.0):
    """Poll ``proc``'s RSS; record the peak in ``state['peak_gb']``; kill it above ``max_gb``
    (``state['killed'] = True``). Runs until the process exits."""
    while proc.poll() is None:
        gb = rss_gb(proc.pid)
        state["peak_gb"] = max(state.get("peak_gb", 0.0), gb)
        if max_gb and gb > max_gb:
            state["killed"] = True
            proc.kill()
            return
        time.sleep(interval)


def wait_for_slot(scratch, min_avail_gb, min_free_disk_gb, log, tag, poll_s=60.0, halt=None):
    """Block until RAM and scratch disk are above the thresholds. ``False`` if halted."""
    while halt is None or not halt.is_set():
        avail = avail_ram_gb()
        free = shutil.disk_usage(scratch).free / 1e9
        if avail >= min_avail_gb and free >= min_free_disk_gb:
            return True
        log(f"{tag}: waiting for resources (avail_ram={avail:.0f} GB, free_disk={free:.0f} GB)")
        time.sleep(poll_s)
    return False


# --------------------------------------------------------------------------- #
# One FOV end to end
# --------------------------------------------------------------------------- #
def run_one(r, cfg):
    """Process one inventory row: run_pyali subprocess -> upload -> verify -> delete local.
    Returns the manifest record; never raises. ``cfg`` is the namespace from :func:`main`
    (``python, analysis_dir, log_dir, s3_prefix, threads, max_fov_rss_gb, min_avail_gb,
    min_free_disk_gb, profile, figures, extra, git_sha, run_id, log, halt, launch_lock``)."""
    ident = r["identity"]
    tag = f"fov{r['fov']:03d}_P{r['plate']}_{r['well']}"
    local = os.path.join(cfg.analysis_dir, f"Plate_{r['plate']}", f"Well_{r['well']}", r["fov_name"])
    dest = f"{dest_prefix(cfg.s3_prefix, r['plate'], r['well'])}/{r['fov_name']}"
    rec = dict(run_id=cfg.run_id, fov_index=r["fov_index"], fov=r["fov"], fov_id=r["fov_id"],
               fov_name=r["fov_name"], plate=r["plate"], well=r["well"], batch=ident.batch,
               src=r["src"], dest=dest, git_sha=cfg.git_sha,
               started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    t0 = time.perf_counter()
    state = {}
    try:
        with cfg.launch_lock:
            if not wait_for_slot(cfg.analysis_dir, cfg.min_avail_gb, cfg.min_free_disk_gb, cfg.log,
                                 tag, halt=cfg.halt):
                raise RuntimeError("halted before launch")
            time.sleep(cfg.stagger_s)                             # stagger the big reads
        shutil.rmtree(local, ignore_errors=True)                  # drop any partial attempt
        os.makedirs(local, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(cfg.threads), OPENBLAS_NUM_THREADS=str(cfg.threads),
                   MKL_NUM_THREADS=str(cfg.threads))
        cmd = fov_command(cfg.python, r["src"], local, ident, cfg.profile, cfg.figures, cfg.extra)
        log_path = os.path.join(cfg.log_dir, f"{tag}.log")
        t1 = time.perf_counter()
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            t = threading.Thread(target=watch_rss, args=(proc, cfg.max_fov_rss_gb, state),
                                 kwargs=dict(interval=cfg.watch_s), daemon=True)
            t.start()
            rc = proc.wait()
            t.join()
        rec.update(rc=rc, process_s=round(time.perf_counter() - t1, 1),
                   peak_rss_gb=round(state.get("peak_gb", 0.0), 2), killed=bool(state.get("killed")))
        if rc != 0 or not os.path.isfile(os.path.join(local, MARKER)):
            tail = ""
            try:
                with open(log_path) as lf:
                    tail = "\n".join(lf.read().splitlines()[-15:])
            except OSError:
                pass
            raise RuntimeError(f"run_pyali rc={rc} marker={os.path.isfile(os.path.join(local, MARKER))} "
                               f"killed={rec['killed']}\n{tail}")
        try:
            with open(os.path.join(local, METADATA)) as f:
                meta = json.load(f)
            rec["n_cells"] = meta.get("n_cells")
            rec["n_cells_by_source"] = meta.get("n_cells_by_source")
            rec["n_regions"] = meta.get("n_regions_kept")
            rec["focus_score"] = meta.get("focus_score")             # QC: ~1e-4 or less = defocused
            rec["timing_s"] = meta.get("timing_s")                   # per-stage wall seconds
            rec["stage_peak_gb"] = {k: v.get("peak") for k, v in (meta.get("memory_gb") or {}).items()}
            rec["process_peak_gb"] = meta.get("peak_rss_gb")         # in-process sampler (peak_rss_gb = watchdog)
        except (OSError, ValueError):
            pass
        fs = rec.get("focus_score")
        cfg.log(f"{tag}: processed in {rec['process_s']}s, peak {rec['peak_rss_gb']} GB, "
                f"{rec.get('n_cells')} cells, focus {fs:.2e}" if isinstance(fs, (int, float)) else
                f"{tag}: processed in {rec['process_s']}s, peak {rec['peak_rss_gb']} GB, "
                f"{rec.get('n_cells')} cells")

        t2 = time.perf_counter()
        up = sh(["aws", "s3", "cp", local, dest + "/", "--recursive", "--only-show-errors"])
        if up.returncode != 0:
            raise RuntimeError(f"upload failed: {up.stderr.strip()[-500:]}")
        chk = sh(["aws", "s3", "ls", f"{dest}/{MARKER}"])
        if chk.returncode != 0 or MARKER not in chk.stdout:
            raise RuntimeError(f"upload verification failed: {MARKER} not listed at {dest}")
        rec["upload_s"] = round(time.perf_counter() - t2, 1)
        shutil.rmtree(local, ignore_errors=True)                  # only after the read-back
        rec.update(ok=True, total_s=round(time.perf_counter() - t0, 1))
        cfg.log(f"{tag}: DONE in {rec['total_s']}s -> {dest}")
    except Exception as e:                                        # noqa: BLE001 - recorded, not raised
        rec.update(ok=False, error=str(e)[-2000:], total_s=round(time.perf_counter() - t0, 1))
        cfg.log(f"{tag}: FAILED after {rec['total_s']}s: {str(e)[:300]}")
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rec


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def git_sha():
    out = sh(["git", "-C", os.path.dirname(_HERE), "rev-parse", "HEAD"])
    sha = out.stdout.strip()[:12] if out.returncode == 0 else "unknown"
    dirty = sh(["git", "-C", os.path.dirname(_HERE), "status", "--porcelain"]).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def write_fov_index(path, rows):
    cols = ["fov_index", "fov", "fov_id", "plate", "well", "batch", "fov_name", "include",
            "exclude_reason", "src"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["fov_index"], r["fov"], r["fov_id"], r["plate"], r["well"],
                        r["identity"].batch, r["fov_name"], r["include"], r.get("exclude_reason", ""),
                        r["src"]])


def archive_run(analysis_dir, s3_prefix, run_id, log):
    """Copy manifest.jsonl, fov_index.csv and logs/ to ``<s3_prefix>/_runs/<run_id>/``."""
    dest = f"{s3_prefix.rstrip('/')}/_runs/{run_id}"
    for name in ("manifest.jsonl", "fov_index.csv"):
        p = os.path.join(analysis_dir, name)
        if os.path.isfile(p):
            out = sh(["aws", "s3", "cp", p, f"{dest}/{name}", "--only-show-errors"])
            if out.returncode != 0:
                log(f"archive: failed to copy {name}: {out.stderr.strip()[-300:]}")
    logs = os.path.join(analysis_dir, "logs")
    if os.path.isdir(logs):
        out = sh(["aws", "s3", "cp", logs, f"{dest}/logs/", "--recursive", "--only-show-errors"])
        if out.returncode != 0:
            log(f"archive: failed to copy logs: {out.stderr.strip()[-300:]}")
    log(f"archived manifest, fov_index and logs to {dest}/")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:                                             # everything after -- goes to run_pyali
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True, help="inventory CSV from scripts/build_inventory.py")
    ap.add_argument("--run-id", required=True, help="run_id column value to process, e.g. P4_A2")
    ap.add_argument("--s3-prefix", required=True, help="s3://bucket/prefix the Plate_N/Well_X tree goes under")
    ap.add_argument("--scratch", default="/home/jovyan/pyali_scratch",
                    help="local scratch; outputs live under <scratch>/<run_id>/Analysis until uploaded")
    ap.add_argument("--src-root", default=None,
                    help="re-root the inventory's src_dir from its day component onward (e.g. the s3fs "
                         "mount when the Mountpoint path in the inventory is absent)")
    ap.add_argument("--experiment-id", default=None, help="override the experiment id in fov_id (default: "
                    "the batch token in each directory name)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for run_pyali.py")
    ap.add_argument("--profile", default=None, help="run_pyali --profile (default: auto-detect)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4, help="BLAS/OpenMP threads per FOV process")
    ap.add_argument("--min-avail-gb", type=float, default=25.0, help="wait to launch below this free RAM")
    ap.add_argument("--min-free-disk-gb", type=float, default=40.0, help="wait to launch below this free scratch")
    ap.add_argument("--max-fov-rss-gb", type=float, default=40.0, help="kill a FOV process above this RSS (0 = off)")
    ap.add_argument("--expected-n", type=int, default=None,
                    help="refuse to run unless the inventory has exactly this many included FOVs")
    ap.add_argument("--force", action="store_true", help="run despite an --expected-n mismatch")
    ap.add_argument("--only", type=int, default=None, help="run only this FOV number (burst)")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of FOVs to run")
    ap.add_argument("--no-resume", action="store_true", help="re-run FOVs already in S3")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--stagger-s", type=float, default=2.0, help="pause between launches")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    analysis_dir = os.path.join(a.scratch, a.run_id, "Analysis")
    log_dir = os.path.join(analysis_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    lock = threading.Lock()

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with lock:
            print(line, flush=True)
            with open(os.path.join(log_dir, "run_corpus.log"), "a") as f:
                f.write(line + "\n")

    rows = load_run(a.inventory, a.run_id, a.src_root, a.experiment_id)
    included = [r for r in rows if r["include"]]
    excluded = [r for r in rows if not r["include"]]
    write_fov_index(os.path.join(analysis_dir, "fov_index.csv"), rows)

    done = set()
    if not a.no_resume:
        done = existing_done(a.s3_prefix, rows[0]["plate"], rows[0]["well"])
    todo = [r for r in included if r["fov_name"] not in done]
    if a.only is not None:
        todo = [r for r in todo if r["fov"] == a.only]
    if a.limit:
        todo = todo[:a.limit]
    sha = git_sha()
    gate = ""
    if a.expected_n is not None:
        gate = " expected OK" if len(included) == a.expected_n else f" expected {a.expected_n} MISMATCH"
    log(f"{a.run_id}: {len(rows)} in inventory ({len(excluded)} excluded), {len(done)} done in S3, "
        f"{len(todo)} to run{gate}  [git {sha}]")
    for r in excluded:
        log(f"  excluded fov {r['fov']:03d} {r['fov_name']}: {r.get('exclude_reason', '')}")
    if a.expected_n is not None and len(included) != a.expected_n and not a.force:
        log(f"refusing to run: {len(included)} included FOVs != --expected-n {a.expected_n} (use --force)")
        return 2
    if a.dry_run:
        for r in todo[:3]:
            log(f"  would run fov {r['fov']:03d} {r['fov_id']} {r['src']}")
        if len(todo) > 3:
            log(f"  ... and {len(todo) - 3} more")
        return 0
    if not todo:
        log("nothing to do")
        return 0

    import types
    cfg = types.SimpleNamespace(
        python=a.python, analysis_dir=analysis_dir, log_dir=log_dir, s3_prefix=a.s3_prefix,
        threads=a.threads, max_fov_rss_gb=a.max_fov_rss_gb, min_avail_gb=a.min_avail_gb,
        min_free_disk_gb=a.min_free_disk_gb, profile=a.profile, figures=a.figures, extra=extra,
        git_sha=sha, run_id=a.run_id, log=log, halt=threading.Event(), launch_lock=threading.Lock(),
        stagger_s=a.stagger_s, watch_s=10.0)
    log(f"workers={a.workers} threads={a.threads} s3={a.s3_prefix} scratch={analysis_dir}")
    manifest = os.path.join(analysis_dir, "manifest.jsonl")
    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, r, cfg) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            with lock:
                with open(manifest, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            n_ok = sum(1 for x in results if x.get("ok"))
            el = time.perf_counter() - t0
            rate = n_ok / el * 3600 if el else 0.0
            eta = (len(todo) - i) / rate if rate else 0.0
            log(f"progress {i}/{len(todo)}  ok={n_ok} fail={i - n_ok}  {rate:.1f} FOV/h  eta {eta:.1f} h")
    failed = [x for x in results if not x.get("ok")]
    log(f"ALL DONE {a.run_id} in {(time.perf_counter() - t0) / 3600:.2f} h: "
        f"{len(results) - len(failed)} ok, {len(failed)} failed")
    for x in failed:
        log(f"  FAILED {x['fov_name']}: {x.get('error', '')[:200]}")
    archive_run(analysis_dir, a.s3_prefix, a.run_id, log)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
