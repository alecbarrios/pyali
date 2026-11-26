#!/usr/bin/env python3
"""E1 before/after table from fov_metadata.json telemetry.

    python e1_table.py <fov_dir_under_experiments/E1> [variant ...]     (default: baseline solver lazy)

Prints, per stage: wall s, cpu/wall (effective threads) and RSS peak for each variant, then the
totals, peak RSS and the throughput estimate
    fields_per_hour = min(floor(cores / threads), floor(0.85 * MemAvail / peak_rss_gb)) * 3600 / total_s
for the machine's cores / MemAvailable (from /proc) at 16 and 8 threads per field.
"""
import json
import os
import sys

STAGES = ["load", "reference", "background", "adaptive_background", "motion_correct",
          "segmentation", "extract_footprints", "trace_extraction", "write"]


def avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def main(argv):
    root = argv[0]
    variants = argv[1:] or ["baseline", "solver", "lazy"]
    metas = {}
    for v in variants:
        p = os.path.join(root, v, "fov_metadata.json")
        if os.path.isfile(p):
            metas[v] = json.load(open(p))
    cores = os.cpu_count()
    avail = avail_gb()
    print(f"### {os.path.basename(root.rstrip('/'))}  (cores {cores}, MemAvailable {avail:.0f} GB)\n")
    hdr = "| stage | " + " | ".join(f"{v} wall s | {v} cpu/wall | {v} RSS peak GB" for v in metas) + " |"
    print(hdr)
    print("|" + "---|" * (1 + 3 * len(metas)))
    for st in STAGES:
        cells = []
        for v, m in metas.items():
            t = (m.get("timing_s") or {}).get(st)
            c = (m.get("cpu_s") or {}).get(st)
            pk = ((m.get("memory_gb") or {}).get(st) or {}).get("peak")
            cells.append(f"{t:.1f}" if t is not None else "")
            cells.append(f"{c / t:.1f}x" if (t and c is not None) else "")
            cells.append(f"{pk:.1f}" if pk is not None else "")
        print(f"| {st} | " + " | ".join(cells) + " |")
    cells = []
    for v, m in metas.items():
        cells += [f"**{(m.get('timing_s') or {}).get('total', float('nan')):.1f}**", "",
                  f"**{m.get('peak_rss_gb', float('nan')):.1f}**"]
    print("| total / peak | " + " | ".join(cells) + " |")
    print()
    print("| variant | threads | total s | peak GB | cpu-bound fields | mem-bound fields | fields / hour |")
    print("|---|---|---|---|---|---|---|")
    for v, m in metas.items():
        total = (m.get("timing_s") or {}).get("total")
        peak = m.get("peak_rss_gb")
        thr = m.get("threads") or {}
        omp = int(thr.get("omp_num_threads") or 16)
        if not total or not peak:
            continue
        for threads in sorted({omp, 8}):
            cpu_b = cores // threads
            mem_b = int(0.85 * avail // peak)
            n = min(cpu_b, mem_b)
            # total_s is measured at `omp` threads; at other thread counts it is an estimate only
            est = "" if threads == omp else " (est.)"
            print(f"| {v} | {threads}{est} | {total:.0f} | {peak:.1f} | {cpu_b} | {mem_b} | "
                  f"{n * 3600 / total:.1f} |")
    print()
    for v, m in metas.items():
        ts = m.get("trace_solve")
        if ts:
            print(f"- {v} solve: {ts.get('method')} on {ts.get('n_union')} union px, cond {ts.get('cond'):.3g}, "
                  f"{ts.get('n_dropped')} dropped, {ts.get('n_unresolved')} unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
