#!/usr/bin/env python3
"""Run every well of one plate, then verify the plate is complete before anything else starts.

    python scripts/run_plate.py --inventory inventory_443screen2.csv --plate 3 \\
        --out-root s3://spatial-technology-platform/AB/pyali/443screen2/analysis_outputs \\
        --profile 443screen2 --workers 8 --threads 8 [--dry-run | --verify-only] [-- <run_pyali args>]

For each well of the plate (inventory ``run_id`` P<plate>_<well>, in well order) it runs
``scripts/run_corpus.py`` with ``--layout well`` into ``<out_root>/plate_<plate>/``, so a FOV lands
at ``plate_<plate>/Well_<well>/<fov_name>/`` and each well's manifest and logs at
``plate_<plate>/_runs/<run_id>/``. Then it lists the plate folder once and checks that every
included inventory row has its ``cells.parquet``, and writes to ``plate_<plate>/_runs/``:

* ``plate_summary.json`` -- git commit, command, timing, per-well included / done / exit code,
  every missing FOV, the focus-flag count;
* ``inventory_plate_<plate>.csv`` -- the plate's inventory rows (the input record);
* ``focus_report.csv`` -- every FOV's focus score (from its ``fov_metadata.json``) with
  ``suspect_defocus`` = score < ``--focus-abs`` or < ``--focus-rel`` x its well's median. Flags are
  for review only; nothing is excluded on focus grounds.

Exit code 0 only when every included FOV is present; the wells' local scratch folders (logs
already archived to ``_runs/``) are then removed. Re-running resumes (done FOVs are skipped).
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import run_corpus                                                         # noqa: E402

MARKER = "cells.parquet"
METADATA = "fov_metadata.json"
_TRUE = {"true", "1", "yes"}


def sh(cmd, stdin=None):
    return subprocess.run(cmd, capture_output=True, text=True, input=stdin)


def plate_rows(inventory_csv, plate):
    """The plate's inventory rows (all, included or not), sorted by well, burst, name."""
    with open(inventory_csv) as f:
        rows = [r for r in csv.DictReader(f) if int(r["plate"] or 0) == int(plate)]
    rows.sort(key=lambda r: (r["well"], int(r["burst"] or 0), r["fov_name"]))
    return rows


def included(r):
    return str(r.get("include", "")).strip().lower() in _TRUE


def done_fovs(prefix):
    """``{(well, fov_name)}`` with a done-marker under ``<prefix>/Well_<well>/<fov_name>/``."""
    out = sh(["aws", "s3", "ls", prefix.rstrip("/") + "/", "--recursive"])
    done = set()
    if out.returncode != 0:
        return done
    for line in out.stdout.splitlines():
        key = line.split()[-1] if line.split() else ""
        parts = key.split("/")
        if len(parts) >= 3 and parts[-1] == MARKER and parts[-3].startswith("Well_"):
            done.add((parts[-3][len("Well_"):], parts[-2]))
    return done


def read_focus(prefix, rows, threads=16):
    """``{fov_name: focus_score or None}`` from each FOV's ``fov_metadata.json``."""
    def one(r):
        out = sh(["aws", "s3", "cp", f"{prefix}/Well_{r['well']}/{r['fov_name']}/{METADATA}", "-"])
        try:
            return r["fov_name"], json.loads(out.stdout).get("focus_score")
        except (ValueError, AttributeError):
            return r["fov_name"], None
    with ThreadPoolExecutor(max_workers=threads) as ex:
        return dict(ex.map(one, rows))


def focus_report(rows, focus, abs_thr=3e-4, rel_thr=1 / 3):
    """Rows ``(well, burst, fov_name, focus, well_median, suspect_defocus)``."""
    import statistics
    by_well = {}
    for r in rows:
        v = focus.get(r["fov_name"])
        if isinstance(v, (int, float)):
            by_well.setdefault(r["well"], []).append(float(v))
    med = {w: statistics.median(v) for w, v in by_well.items()}
    out = []
    for r in rows:
        v = focus.get(r["fov_name"])
        m = med.get(r["well"])
        if isinstance(v, (int, float)):
            suspect = v < abs_thr or (m is not None and v < rel_thr * m)
        else:
            suspect = None
        out.append(dict(well=r["well"], burst=r["burst"], fov_name=r["fov_name"],
                        focus=v if v is not None else "", well_median=m if m is not None else "",
                        suspect_defocus="" if suspect is None else bool(suspect)))
    return out


def upload_text(text, dest):
    r = sh(["aws", "s3", "cp", "-", dest, "--only-show-errors"], stdin=text)
    if r.returncode != 0:
        raise RuntimeError(f"upload to {dest} failed: {r.stderr.strip()[-300:]}")


def _csv_text(rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--plate", type=int, required=True)
    ap.add_argument("--out-root", required=True, help="s3://... folder holding plate_<N>/")
    ap.add_argument("--profile", default=None, help="run_pyali --profile (e.g. 443screen2)")
    ap.add_argument("--workers", type=int, default=8, help="FOVs processed at once")
    ap.add_argument("--threads", type=int, default=8, help="CPU budget per FOV (BLAS threads and region workers)")
    ap.add_argument("--scratch", default="/home/jovyan/pyali_scratch", help="local scratch (emptied per FOV)")
    ap.add_argument("--max-fov-rss-gb", type=float, default=40.0)
    ap.add_argument("--min-avail-gb", type=float, default=25.0)
    ap.add_argument("--focus-abs", type=float, default=3e-4, help="flag focus scores below this")
    ap.add_argument("--focus-rel", type=float, default=1 / 3, help="flag focus below this x the well median")
    ap.add_argument("--dry-run", action="store_true", help="show what each well would run")
    ap.add_argument("--verify-only", action="store_true", help="skip running; verify and report")
    a = ap.parse_args(argv)

    from pyali.results import git_sha
    prefix = f"{a.out_root.rstrip('/')}/plate_{a.plate}"
    rows = plate_rows(a.inventory, a.plate)
    if not rows:
        print(f"[plate {a.plate}] no inventory rows", flush=True)
        return 2
    inc = [r for r in rows if included(r)]
    run_ids = sorted({r["run_id"] for r in inc})
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()
    print(f"[plate {a.plate}] {len(rows)} inventory rows, {len(inc)} included, wells "
          f"{', '.join(run_ids)} -> {prefix}/", flush=True)
    rc_of = {}
    if not a.verify_only:
        for rid in run_ids:
            args = ["--inventory", a.inventory, "--run-id", rid, "--s3-prefix", prefix,
                    "--layout", "well", "--workers", str(a.workers), "--threads", str(a.threads),
                    "--scratch", a.scratch, "--max-fov-rss-gb", str(a.max_fov_rss_gb),
                    "--min-avail-gb", str(a.min_avail_gb)]
            if a.profile:
                args += ["--profile", a.profile]
            if a.dry_run:
                args.append("--dry-run")
            print(f"[plate {a.plate}] well {rid}: run_corpus {' '.join(args)}"
                  + (f" -- {' '.join(extra)}" if extra else ""), flush=True)
            rc_of[rid] = run_corpus.main(args + (["--"] + extra if extra else []))
    if a.dry_run:
        return 0

    done = done_fovs(prefix)
    missing = [r for r in inc if (r["well"], r["fov_name"]) not in done]
    focus = read_focus(prefix, [r for r in inc if (r["well"], r["fov_name"]) in done])
    report = focus_report(inc, focus, a.focus_abs, a.focus_rel)
    n_flag = sum(1 for x in report if x["suspect_defocus"] is True)
    wells = {}
    for rid in sorted({r["run_id"] for r in rows}):
        w_rows = [r for r in rows if r["run_id"] == rid]
        w_inc = [r for r in w_rows if included(r)]
        wells[rid] = dict(rows=len(w_rows), included=len(w_inc),
                          done=sum((r["well"], r["fov_name"]) in done for r in w_inc),
                          run_corpus_rc=rc_of.get(rid))
    summary = dict(plate=a.plate, out=prefix, pyali_git=git_sha(), started=started,
                   finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   hours=round((time.perf_counter() - t0) / 3600, 3), command=sys.argv, profile=a.profile,
                   run_pyali_extra=extra, workers=a.workers, threads=a.threads, wells=wells,
                   n_included=len(inc), n_done=len(inc) - len(missing),
                   missing=[f"{r['well']}/{r['fov_name']}" for r in missing],
                   focus_thresholds=dict(abs=a.focus_abs, rel_to_well_median=a.focus_rel),
                   n_focus_suspects=n_flag, complete=not missing)
    upload_text(json.dumps(summary, indent=2, default=str), f"{prefix}/_runs/plate_summary.json")
    upload_text(_csv_text(rows, list(rows[0].keys())), f"{prefix}/_runs/inventory_plate_{a.plate}.csv")
    upload_text(_csv_text(report, ["well", "burst", "fov_name", "focus", "well_median", "suspect_defocus"]),
                f"{prefix}/_runs/focus_report.csv")
    if not missing and not a.verify_only:                # logs are archived in _runs/: leave no local copy
        import shutil
        for rid in run_ids:
            local = os.path.join(a.scratch, rid)
            if os.path.isdir(local) and os.path.realpath(local).startswith(os.path.realpath(a.scratch) + os.sep):
                shutil.rmtree(local, ignore_errors=True)
    print(f"[plate {a.plate}] {summary['n_done']}/{len(inc)} FOVs present, {len(missing)} missing, "
          f"{n_flag} focus suspects -> {prefix}/_runs/plate_summary.json", flush=True)
    for m in summary["missing"][:20]:
        print(f"  MISSING {m}", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
