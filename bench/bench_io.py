#!/usr/bin/env python3
"""s3fs read-throughput scaling test.

Reads N *distinct, never-touched* movies concurrently (so nothing is served from page
cache) and reports aggregate MB/s. Each worker reads only the first ``--gb`` GB of its
file, which is enough to reach steady state without spending an hour per point.

    python bench_io.py --day 20260715 --streams 1 2 4 8 16 32 --gb 2
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

KEEP = "/home/jovyan/spatial-technology-platform/AB/pyali_c27fc46_outputs/keep.csv"
ROOT = "/home/jovyan/spatial-technology-platform/AB"

CHUNK = 8 << 20


def read_head(path, nbytes):
    got = 0
    with open(path, "rb", buffering=0) as f:
        while got < nbytes:
            b = f.read(min(CHUNK, nbytes - got))
            if not b:
                break
            got += len(b)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="20260715")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--gb", type=float, default=2.0)
    ap.add_argument("--out", default="/home/jovyan/bench/io_scaling.json")
    a = ap.parse_args()

    with open(KEEP) as f:
        rows = [r for r in csv.DictReader(f) if r["day"] == a.day]
    rows.sort(key=lambda r: r["fov_path"])
    paths = [os.path.join(ROOT, r["fov_path"], "frames1.bin") for r in rows]

    nbytes = int(a.gb * 1e9)
    need = sum(a.streams)
    if need > len(paths):
        raise SystemExit(f"need {need} distinct movies, day has {len(paths)}")

    cursor = 0
    results = []
    for n in a.streams:
        batch = paths[cursor:cursor + n]          # fresh files every point -> always cold
        cursor += n
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            got = sum(ex.map(lambda p: read_head(p, nbytes), batch))
        dt = time.perf_counter() - t0
        agg = got / 1e6 / dt
        results.append(dict(streams=n, seconds=dt, gb=got / 1e9, agg_mb_s=agg,
                            per_stream_mb_s=agg / n))
        print(f"  {n:3d} streams: {agg:8.1f} MB/s aggregate  ({agg/n:6.1f} MB/s per stream, "
              f"{got/1e9:.1f} GB in {dt:.1f} s)", flush=True)

    with open(a.out, "w") as f:
        json.dump(dict(day=a.day, gb_per_stream=a.gb, results=results), f, indent=2)
    best = max(results, key=lambda r: r["agg_mb_s"])
    print(f"\n  peak aggregate: {best['agg_mb_s']:.0f} MB/s at {best['streams']} streams")
    print(f"  -> 35.62 TB at that rate = {35.62e6/best['agg_mb_s']/3600:.1f} h "
          f"({35.62e6/best['agg_mb_s']/86400:.2f} days) of pure reading")


if __name__ == "__main__":
    main()
