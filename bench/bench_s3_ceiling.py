#!/usr/bin/env python3
"""True S3 -> instance read ceiling: parallel ranged GETs, no filesystem in the path.

Bypasses s3fs entirely so the number reflects only the network + S3 + HTTP client. Every
point reads fresh byte ranges from cold objects, and nothing is written to disk.
"""
import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

BUCKET = "spatial-technology-platform"
KEEP = "/home/jovyan/spatial-technology-platform/AB/pyali_c27fc46_outputs/keep.csv"
READ_CHUNK = 8 << 20


def make_client(pool):
    return boto3.client("s3", config=Config(
        max_pool_connections=pool, retries={"max_attempts": 3, "mode": "standard"},
        tcp_keepalive=True))


def get_range(client, key, start, length):
    r = client.get_object(Bucket=BUCKET, Key=key,
                          Range=f"bytes={start}-{start + length - 1}")
    body = r["Body"]
    got = 0
    while True:
        b = body.read(READ_CHUNK)
        if not b:
            break
        got += len(b)
    body.close()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", default=["20260718"])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--gb-per-point", type=float, default=4.0)
    ap.add_argument("--part-mb", type=int, default=64)
    ap.add_argument("--out", default="/home/jovyan/bench/s3_ceiling.json")
    a = ap.parse_args()

    with open(KEEP) as f:
        rows = [r for r in csv.DictReader(f) if r["day"] in a.days]
    rows.sort(key=lambda r: r["fov_path"])
    objs = [("AB/" + r["fov_path"] + "/frames1.bin", int(float(r["bin_bytes"]))) for r in rows]
    print(f"[s3] {len(objs)} cold objects available in {a.days}", flush=True)

    part = a.part_mb << 20
    # Build one global list of distinct (key, offset) parts; each point consumes a fresh slice,
    # so no byte is ever fetched twice across the whole sweep.
    parts = []
    for key, size in objs:
        for off in range(0, size - part, part):
            parts.append((key, off))
    print(f"[s3] {len(parts)} distinct {a.part_mb} MB parts available "
          f"({len(parts)*part/1e12:.2f} TB)", flush=True)

    cursor = 0
    results = []
    total_read = 0
    for n in a.concurrency:
        want = int(a.gb_per_point * 1e9 / part)
        batch = parts[cursor:cursor + want]
        cursor += want
        if len(batch) < want:
            print("[s3] ran out of cold parts", flush=True)
            break
        client = make_client(max(n * 2, 32))
        local = threading.local()

        def work(item):
            if not hasattr(local, "c"):
                local.c = client
            return get_range(local.c, item[0], item[1], part)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            got = sum(ex.map(work, batch))
        dt = time.perf_counter() - t0
        total_read += got
        mb = got / 1e6 / dt
        results.append(dict(concurrency=n, seconds=dt, gb=got / 1e9, mb_s=mb,
                            gbps=mb * 8 / 1000))
        print(f"  {n:4d} threads: {mb:8.0f} MB/s  ({mb*8/1000:5.2f} Gbps)  "
              f"{got/1e9:.1f} GB in {dt:.1f} s", flush=True)

    with open(a.out, "w") as f:
        json.dump(dict(bucket=BUCKET, part_mb=a.part_mb, results=results,
                       total_gb_read=total_read / 1e9), f, indent=2)
    if results:
        best = max(results, key=lambda r: r["mb_s"])
        print(f"\n  peak: {best['mb_s']:.0f} MB/s ({best['gbps']:.2f} Gbps) "
              f"at {best['concurrency']} threads", flush=True)
        print(f"  vs s3fs ~120 MB/s  ->  {best['mb_s']/120:.1f}x", flush=True)
        print(f"  35.62 TB at peak = {35.62e6/best['mb_s']/3600:.1f} h", flush=True)
        print(f"  total read this sweep: {total_read/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
