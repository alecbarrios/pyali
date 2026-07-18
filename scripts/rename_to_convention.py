#!/usr/bin/env python3
"""Rename FOV directories in ONE day directory to the canonical pyali naming convention.

Future acquisitions may name FOV directories slightly differently. Point this at a day directory
and it proposes a canonical rename for every FOV sub-dir (one that contains ``frames1.bin``):

    <ts>_P-<plate>_W-<well>_<genotype>_<batch>_DIV<div>__burst<n>   (or _manual[_burst<n>])

so ``build_manifest.py`` parses everything uniformly afterwards. Parsing/canonicalization is shared
with build_manifest.py (``parse_fov_name`` / ``canonical_name``).

SAFE BY DEFAULT: with no ``--apply`` it only WRITES A PLAN CSV (old_name, new_name, action) and
renames nothing. Review the plan, then re-run with ``--apply``. Directories whose names can't be
parsed (no plate/well) are listed as ``action=SKIP_unparsed`` for you to handle by hand.

    python scripts/rename_to_convention.py DAY_DIR                 # dry-run -> rename_plan.csv
    python scripts/rename_to_convention.py DAY_DIR --plan p.csv     # dry-run, custom plan path
    python scripts/rename_to_convention.py DAY_DIR --apply          # actually rename

NOTE (s3fs): if DAY_DIR is on an s3fs/FUSE mount, ``os.rename`` of a directory re-keys every object
under it on S3 (a server-side copy+delete per file) -- slow/costly for multi-GB frames1.bin. Prefer
renaming a LOCAL copy, or renaming before upload, when you can.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # so build_manifest imports
from build_manifest import parse_fov_name, canonical_name            # noqa: E402


def build_plan(day_dir, default_batch):
    """Return list of (old_name, new_name, action) for every FOV dir (has frames1.bin) in day_dir."""
    try:
        names = sorted(e.name for e in os.scandir(day_dir) if e.is_dir())
    except OSError as e:
        sys.exit(f"cannot list {day_dir}: {e}")
    plan, taken = [], set()
    for name in names:
        if not os.path.isfile(os.path.join(day_dir, name, "frames1.bin")):
            continue                                                 # not a FOV dir
        d = parse_fov_name(name)
        if not d["parse_ok"]:
            plan.append((name, "", "SKIP_unparsed"))
            continue
        new = canonical_name(d, default_batch=default_batch)
        if new == name:
            action = "ok_already"
        elif new in taken or os.path.exists(os.path.join(day_dir, new)):
            action = "SKIP_collision"
        else:
            action = "rename"
            taken.add(new)
        plan.append((name, new, action))
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day_dir", help="day directory whose FOV sub-dirs to rename")
    ap.add_argument("--apply", action="store_true", help="perform the renames (default: dry-run)")
    ap.add_argument("--plan", default=None, help="plan CSV path (default: <day_dir>/rename_plan.csv)")
    ap.add_argument("--batch", default="443screen2", help="fallback batch id if a name omits it")
    a = ap.parse_args(argv)

    plan = build_plan(a.day_dir, a.batch)
    plan_path = a.plan or os.path.join(a.day_dir, "rename_plan.csv")
    with open(plan_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["old_name", "new_name", "action"])
        w.writerows(plan)

    counts = {}
    for _o, _n, act in plan:
        counts[act] = counts.get(act, 0) + 1
    print(f"[rename] {len(plan)} FOV dirs in {a.day_dir}")
    for act, c in sorted(counts.items()):
        print(f"[rename]   {act}: {c}")
    print(f"[rename] plan written -> {plan_path}")

    if not a.apply:
        print("[rename] DRY-RUN: nothing renamed. Review the plan, then re-run with --apply.")
        return 0

    n_done, n_err = 0, 0
    for old, new, act in plan:
        if act != "rename":
            continue
        try:
            os.rename(os.path.join(a.day_dir, old), os.path.join(a.day_dir, new))
            n_done += 1
        except OSError as e:
            print(f"[rename]   FAILED {old} -> {new}: {e}", file=sys.stderr)
            n_err += 1
    print(f"[rename] APPLIED: {n_done} renamed, {n_err} failed.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
