#!/usr/bin/env python3
"""Build a corpus inventory: one CSV row per FOV directory (a directory holding ``frames1.bin``).

The inventory is the contract between a dataset and ``scripts/run_corpus.py``: nothing about a
dataset is hard-coded anywhere else. Point it at any mounted root (Mountpoint or s3fs), e.g.::

    python scripts/build_inventory.py --root /mnt/stp/20260716 --experiment-id 443screen2 \\
        --out inv_20260716.csv
    python scripts/build_inventory.py --root /mnt/stp/20260715 --plate-map 20260715:P-1=5 \\
        --wells A2 C3 --experiment-id 443screen2 --out inv_20260715.csv

Columns (row order = plate, well, burst, fov_name)::

    run_id, fov_name, day, plate, plate_token, well, batch, burst, src_dir, bin_bytes,
    bytes_per_frame, frames_in_file, frames_requested, include, exclude_reason

``run_id = P<plate>_<well>`` groups the FOVs ``run_corpus`` processes together. ``day`` is the
first path component under ``--root`` (or the root's own name when the FOV directories sit
directly in it). Identity fields come from the directory name via ``pyali.metadata``; the plate
can be remapped per day with ``--plate-map DAY:TOKEN=PLATE`` for relabelled acquisitions.
Excluded rows (zero-byte or partial movies, no ``burstN`` token, unparseable names) stay in the
inventory with a reason, so they keep their place in the numbering and the record.
"""
import argparse
import csv
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pyali.metadata import parse_dir_name                                 # noqa: E402

COLUMNS = ["run_id", "fov_name", "day", "plate", "plate_token", "well", "batch", "burst",
           "src_dir", "bin_bytes", "bytes_per_frame", "frames_in_file", "frames_requested",
           "include", "exclude_reason", "focus"]
FOCUS_FRAMES = 300
SIDECAR = "frames1_dropped_frames.txt"
MOVIE = "frames1.bin"
ABORTED_FRACTION = 0.5          # a movie shorter than this fraction of target_frames is an aborted run


def read_sidecar(fov_dir):
    """``key=value`` lines of the acquisition sidecar as a dict; integers are converted.
    Missing sidecar -> ``{}``."""
    path = os.path.join(fov_dir, SIDECAR)
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            try:
                out[k.strip()] = int(v)
            except ValueError:
                out[k.strip()] = v
    return out


def scan_fovs(root):
    """Every directory under ``root`` (inclusive) that holds a ``frames1.bin``, sorted."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if MOVIE in filenames:
            found.append(dirpath)
    return sorted(found)


def parse_plate_map(items):
    """``["20260715:P-1=5", ...]`` -> ``{("20260715", "P-1"): 5}``."""
    out = {}
    for it in items or ():
        m = re.fullmatch(r"([^:]+):([^=]+)=(\d+)", it.strip())
        if not m:
            raise ValueError(f"--plate-map expects DAY:TOKEN=PLATE, got {it!r}")
        out[(m.group(1), m.group(2))] = int(m.group(3))
    return out


def focus_of(fov_dir, nrow, ncol, bytes_per_frame):
    """Focus QC score of the mean of the first ``FOCUS_FRAMES`` frames (8-bit when
    ``bytes_per_frame == nrow * ncol``, else 16-bit); ``''`` when the file is too short."""
    import numpy as np
    from pyali.preprocess import focus_score
    dtype = np.uint8 if bytes_per_frame == nrow * ncol else np.uint16
    n = nrow * ncol * FOCUS_FRAMES
    raw = np.fromfile(os.path.join(fov_dir, MOVIE), dtype=dtype, count=n)
    if raw.size < n:
        return ""
    return round(focus_score(raw.reshape(FOCUS_FRAMES, nrow, ncol).mean(axis=0, dtype=np.float64)), 8)


def inventory_row(fov_dir, root, experiment_id, plate_map=None, focus_dims=None):
    """One inventory row for ``fov_dir`` (see the module docstring for the columns).
    ``focus_dims=(nrow, ncol)`` fills the ``focus`` QC column from the first 300 frames."""
    plate_map = plate_map or {}
    fov_dir = os.path.abspath(fov_dir)
    name = os.path.basename(fov_dir)
    rel = os.path.relpath(fov_dir, os.path.abspath(root)).split(os.sep)
    day = rel[0] if len(rel) > 1 else os.path.basename(os.path.abspath(root))

    d = parse_dir_name(name)
    token = d["plate"] or ""
    digits = re.sub(r"\D", "", token)
    plate = plate_map.get((day, token), int(digits) if digits else 0)
    well = d["well"] or ""
    burst = d["burst"]
    batch = d["cell_line"] or ""

    side = read_sidecar(fov_dir)
    bin_bytes = os.path.getsize(os.path.join(fov_dir, MOVIE))
    bpf = int(side.get("exposure_bytes_per_frame") or 0)
    frames_in_file = bin_bytes // bpf if bpf else ""
    target = side.get("target_frames")

    if bin_bytes == 0:
        reason = "zero-byte frames1.bin"
    elif burst is None:
        reason = "no burstN token (manual/wash acquisition)"
    elif not plate or not well:
        reason = "plate/well not parseable from the directory name"
    elif bpf and bin_bytes % bpf:
        reason = "frames1.bin is not a whole number of frames"
    elif bpf and isinstance(target, int) and target > 0 and frames_in_file < ABORTED_FRACTION * target:
        reason = f"aborted acquisition ({frames_in_file} of {target} frames)"
    else:
        reason = ""

    focus = ""
    if focus_dims and reason == "" and bpf:
        focus = focus_of(fov_dir, int(focus_dims[0]), int(focus_dims[1]), bpf)
    return dict(run_id=f"P{plate}_{well}", fov_name=name, day=day, plate=plate, plate_token=token,
                well=well, batch=batch, burst=burst if burst is not None else 0, src_dir=fov_dir,
                bin_bytes=bin_bytes, bytes_per_frame=bpf, frames_in_file=frames_in_file,
                frames_requested=target if target is not None else "",
                include=reason == "", exclude_reason=reason, focus=focus)


def build_inventory(roots, out_csv, experiment_id, plate_map=None, wells=None, focus_dims=None):
    """Scan ``roots``, write ``out_csv`` and return the rows (sorted by plate, well, burst, name)."""
    rows = [inventory_row(f, root, experiment_id, plate_map, focus_dims)
            for root in roots for f in scan_fovs(root)]
    if wells:
        keep = {w.upper() for w in wells}
        rows = [r for r in rows if r["well"].upper() in keep]
    rows.sort(key=lambda r: (r["plate"], r["well"], int(r["burst"]), r["fov_name"]))
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(rows):
    """``[(run_id, n_included, n_excluded)]`` in run order."""
    out = {}
    for r in rows:
        inc, exc = out.get(r["run_id"], (0, 0))
        out[r["run_id"]] = (inc + 1, exc) if r["include"] else (inc, exc + 1)
    return [(k, v[0], v[1]) for k, v in sorted(out.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="directory to scan (repeatable); a day directory or a parent of day directories")
    ap.add_argument("--out", required=True, help="inventory CSV to write")
    ap.add_argument("--experiment-id", required=True, help="experiment id used in fov_id / cell_id")
    ap.add_argument("--plate-map", action="append", default=[],
                    help="DAY:TOKEN=PLATE, repeatable, e.g. 20260715:P-1=5 (the folder token P-1 is plate 5)")
    ap.add_argument("--wells", nargs="+", default=None, help="keep only these wells, e.g. A2 C3")
    ap.add_argument("--focus-dims", nargs=2, type=int, metavar=("NROW", "NCOL"), default=None,
                    help="fill the focus QC column from the first 300 frames of each included movie "
                         "(e.g. 800 800 for 443screen2); ~1e-4 or less means defocused")
    a = ap.parse_args(argv)
    rows = build_inventory(a.root, a.out, a.experiment_id, parse_plate_map(a.plate_map), a.wells,
                           a.focus_dims)
    print(f"[inventory] {len(rows)} FOV directories -> {a.out}")
    for run_id, inc, exc in summarize(rows):
        print(f"  {run_id:<10} include={inc:<4} excluded={exc}")
    if a.focus_dims:
        scored = [r for r in rows if r["focus"] != ""]
        if scored:
            import statistics
            med = statistics.median(float(r["focus"]) for r in scored)
            low = [r for r in scored if float(r["focus"]) < med / 3]
            print(f"[inventory] focus: median {med:.2e}; {len(low)} of {len(scored)} fields below 1/3 of it"
                  + (": " + ", ".join(f"{r['run_id']} b{r['burst']} ({float(r['focus']):.1e})" for r in low) if low else ""))
    for r in rows:
        if not r["include"]:
            print(f"  EXCLUDED {r['day']}/{r['fov_name']}: {r['exclude_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
