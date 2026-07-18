#!/usr/bin/env python3
"""Build the pyali samples manifest by scanning one or more "day" directories.

Each day directory (e.g. ``.../AB/20260715``) holds one sub-directory per field of view (FOV).
A real FOV directory contains ``frames1.bin``; anything else (``Snaps``, ``ScanLayout.fig``,
``Session_Notes.txt``, manual metadata) is ignored. Every FOV directory name is parsed into the
grouping keys used downstream (extraction / QC / aggregation):

    day, timestamp, plate, well, genotype, diff_id, line, batch_id, div, burst, is_manual

Naming convention parsed (tolerant to the ``114944P-1`` vs ``163925_P-1`` timestamp variants):
    <ts>_P-<plate>_W-<well>_<genotype>_<batch>_DIV<div>__burst<n>        (or _manual[_burst<n>])
where ``genotype`` is e.g. ``443-2`` (line 443, differentiation 2), ``443`` or ``WT``, and
``batch`` is e.g. ``443screen2``. Directories whose names don't parse are still listed with
``parse_ok=False`` so you can fix them first with ``scripts/rename_to_convention.py``.

Outputs a manifest as parquet (+ CSV) with one row per FOV, plus a printed summary.

Usage:
    python scripts/build_manifest.py DAY_DIR [DAY_DIR ...] --out manifest.parquet
    python scripts/build_manifest.py ~/spatial-technology-platform/AB/20260715 \
           ~/spatial-technology-platform/AB/20260716 ~/spatial-technology-platform/AB/20260717 \
           --out ~/workbench/voltage/pyali_output/manifest.parquet
"""
import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------------------------------------------------------------- #
# Name parsing (shared with rename_to_convention.py)
# --------------------------------------------------------------------------- #
def parse_fov_name(name):
    """Parse a FOV directory name into fields. ``parse_ok`` is True iff plate AND well were found.

    Returns a dict with keys: timestamp, plate (int), well (str), genotype (str), line (str),
    diff_id (int|None), batch_id (str), div (int), burst (int), is_manual (bool), parse_ok (bool).
    """
    d = dict(timestamp=None, plate=None, well=None, genotype=None, line=None,
             diff_id=None, batch_id=None, div=None, burst=None, is_manual=False, parse_ok=False)

    m = re.match(r'(\d{6})', name)                       # leading HHMMSS timestamp
    if m:
        d['timestamp'] = m.group(1)
    m = re.search(r'P-?0*(\d+)', name)                   # plate: P-1 / P-01 / P06
    if m:
        d['plate'] = int(m.group(1))
    m = re.search(r'W-([A-Za-z]\d+)', name)              # well: W-A1
    if m:
        d['well'] = m.group(1).upper()
    m = re.search(r'([A-Za-z0-9]+screen\d+)', name)      # batch: 443screen2
    if m:
        d['batch_id'] = m.group(1)
    if d['well'] and d['batch_id']:                      # genotype sits between W-<well>_ and _<batch>
        mm = re.search(re.escape('W-' + d['well']) + r'_(.+?)_' + re.escape(d['batch_id']), name)
        if mm:
            d['genotype'] = mm.group(1)
    m = re.search(r'DIV(\d+)', name)
    if m:
        d['div'] = int(m.group(1))
    d['is_manual'] = 'manual' in name.lower()
    m = re.search(r'burst(\d+)', name, re.IGNORECASE)
    if m:
        d['burst'] = int(m.group(1))

    if d['genotype']:                                    # differentiation id: 443-2 -> line 443, diff 2
        gm = re.match(r'([A-Za-z0-9]+?)-(\d+)$', d['genotype'])
        if gm:
            d['line'], d['diff_id'] = gm.group(1), int(gm.group(2))
        else:
            d['line'] = d['genotype']                    # WT / 443 (no differentiation suffix)

    d['parse_ok'] = d['plate'] is not None and d['well'] is not None
    return d


def canonical_name(d, default_batch='443screen2'):
    """Canonical FOV directory name from parsed fields (needs plate + well). See module docstring."""
    ts = d.get('timestamp') or '000000'
    geno = d.get('genotype') or 'NA'
    batch = d.get('batch_id') or default_batch
    parts = [ts, f"P-{d['plate']}", f"W-{d['well']}", geno, batch]
    if d.get('div') is not None:
        parts.append(f"DIV{d['div']}")
    stem = "_".join(parts)
    if d.get('is_manual'):
        stem += "_manual" + (f"_burst{d['burst']}" if d.get('burst') is not None else "")
    elif d.get('burst') is not None:
        stem += f"__burst{d['burst']}"
    return stem


# --------------------------------------------------------------------------- #
# Sidecar / file metadata
# --------------------------------------------------------------------------- #
def _read_sidecar(fov_path):
    """Return (bytes_per_frame, target_frames) from frames1_dropped_frames.txt, or (None, None)."""
    txt = os.path.join(fov_path, "frames1_dropped_frames.txt")
    bpf = tf = None
    try:
        with open(txt, "r", errors="ignore") as fh:
            s = fh.read()
        m = re.search(r'exposure_bytes_per_frame\s*[=:]\s*(\d+)', s)
        if m:
            bpf = int(m.group(1))
        m = re.search(r'target_frames\s*[=:]\s*(\d+)', s)
        if m:
            tf = int(m.group(1))
    except OSError:
        pass
    return bpf, tf


def _row_for(day_dir, name):
    """Build a manifest row for one candidate FOV dir, or None if it has no frames1.bin."""
    fov_path = os.path.join(day_dir, name)
    bin_path = os.path.join(fov_path, "frames1.bin")
    try:
        file_bytes = os.path.getsize(bin_path)          # HEAD over s3fs; raises if absent
    except OSError:
        return None                                     # not a real FOV (Snaps/, metadata, ...)
    bpf, tf = _read_sidecar(fov_path)
    d = parse_fov_name(name)
    d.update(day=os.path.basename(os.path.normpath(day_dir)), dir_name=name,
             fov_path=os.path.abspath(fov_path), file_bytes=file_bytes,
             bytes_per_frame=bpf, target_frames=tf,
             n_frames_est=(file_bytes // bpf if bpf else None))
    return d


def scan_day(day_dir, workers=16):
    """Return (rows, n_skipped) for one day directory (only dirs containing frames1.bin become rows)."""
    try:
        names = sorted(e.name for e in os.scandir(day_dir) if e.is_dir())
    except OSError as e:
        print(f"  !! cannot list {day_dir}: {e}", file=sys.stderr)
        return [], 0
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_row_for, day_dir, n): n for n in names}
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                rows.append(r)
    n_skipped = len(names) - len(rows)
    return rows, n_skipped


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("day_dirs", nargs="+", help="one or more day directories to scan")
    ap.add_argument("--out", default="manifest.parquet", help="output parquet path")
    ap.add_argument("--csv", default=None, help="also write CSV here (default: <out>.csv)")
    ap.add_argument("--workers", type=int, default=16, help="parallel stat workers (s3fs latency)")
    a = ap.parse_args(argv)

    import pandas as pd

    all_rows, total_skipped = [], 0
    for day in a.day_dirs:
        print(f"[manifest] scanning {day} ...", flush=True)
        rows, skipped = scan_day(day, a.workers)
        total_skipped += skipped
        all_rows.extend(rows)
        print(f"[manifest]   {len(rows)} FOVs (+{skipped} non-FOV entries skipped)", flush=True)

    cols = ["day", "dir_name", "timestamp", "plate", "well", "genotype", "line", "diff_id",
            "batch_id", "div", "burst", "is_manual", "parse_ok",
            "file_bytes", "bytes_per_frame", "target_frames", "n_frames_est", "fov_path"]
    df = pd.DataFrame(all_rows, columns=cols).sort_values(["day", "plate", "well", "burst"],
                                                          na_position="last").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    try:
        df.to_parquet(a.out, index=False)
        wrote = a.out
    except Exception as e:                               # pyarrow missing / parquet issue -> CSV only
        print(f"[manifest] parquet write failed ({e}); writing CSV only.", file=sys.stderr)
        wrote = None
    csv_path = a.csv or (os.path.splitext(a.out)[0] + ".csv")
    df.to_csv(csv_path, index=False)

    # ---- summary ----
    n = len(df)
    n_bad = int((~df["parse_ok"]).sum())
    print(f"\n[manifest] {n} FOVs written -> {wrote or '(no parquet)'} + {csv_path}")
    print(f"[manifest] unparseable names: {n_bad}"
          + ("  (fix with scripts/rename_to_convention.py)" if n_bad else ""))
    if n:
        print("[manifest] FOVs per day:")
        for day, cnt in df.groupby("day").size().items():
            print(f"           {day}: {cnt}")
        print("[manifest] FOVs per (day, plate, well)  [top 12]:")
        g = df.groupby(["day", "plate", "well"]).size().sort_values(ascending=False)
        for (day, plate, well), cnt in list(g.items())[:12]:
            print(f"           {day}  P-{plate}  W-{well}: {cnt}")
        bpf = df["bytes_per_frame"].dropna().unique()
        print(f"[manifest] distinct bytes_per_frame seen: {sorted(bpf.tolist())} "
              f"(640000 => 800x800 uint8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
