#!/usr/bin/env python3
"""N-concurrency throughput test on high-region (stress) FOVs.

Runs the SAME 8 FOVs at several concurrency levels so the levels are directly comparable, and
samples system RAM throughout. FOVs are drawn mid-sequence from 20260715 (800x800 — 52% of the
corpus by count, and mid-sequence bursts are the dense ones: the seg survey puts the median at
~578 regions and the tail past 1100).

The throughput set is deliberately homogeneous — no 0-region or 8-region FOVs — because mixing
trivial FOVs into a FOV/h measurement distorts it. Edge cases belong in the correctness gates.
"""
import json
import os
import subprocess
import sys
import threading
import time

KEEP = "/home/jovyan/spatial-technology-platform/AB/pyali_6b59e79_outputs/keep.csv"
S3 = "s3://spatial-technology-platform/AB/pyali_nconc_TEST"
RUNNER = "/home/jovyan/workbench/pyali/scripts/run_corpus.py"
PY = "/home/jovyan/.venvs/pyali/bin/python"
LEVELS = [8, 4, 2]                      # most decision-relevant first
N_FOV = 8

_stop = threading.Event()
ram = []


def sample_ram():
    while not _stop.is_set():
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}
        ram.append((time.time(), (m["MemTotal"] - m["MemAvailable"]) / 1024 / 1024))
        _stop.wait(5)


def main():
    threading.Thread(target=sample_ram, daemon=True).start()
    results = []
    for n in LEVELS:
        mf = f"/home/jovyan/bench/nconc_manifest_N{n}.jsonl"
        if os.path.exists(mf):
            os.remove(mf)
        ram_lo = len(ram)
        print(f"\n{'='*70}\n[nconc] concurrency N={n}, {N_FOV} stress FOVs\n{'='*70}", flush=True)
        t0 = time.perf_counter()
        cp = subprocess.run(
            [PY, RUNNER, "--keep-csv", KEEP, "--days", "20260715", "--stride", "150",
             "--limit", str(N_FOV), "--workers", str(n), "--no-resume",
             "--s3-prefix", S3, "--manifest", mf, "--scratch", "/home/jovyan/corpus_scratch"],
            capture_output=True, text=True)
        wall = time.perf_counter() - t0
        sys.stdout.write(cp.stdout[-2500:])
        if cp.returncode != 0:
            sys.stdout.write("STDERR:\n" + cp.stderr[-1500:])
        recs = [json.loads(l) for l in open(mf)] if os.path.exists(mf) else []
        ok = [r for r in recs if r.get("ok")]
        peak = max((u for _ts, u in ram[ram_lo:]), default=float("nan"))
        per = [r["process_s"] for r in ok if "process_s" in r]
        regs = [r.get("n_regions", 0) for r in ok]
        cells = [r.get("n_cells", 0) for r in ok]
        byt = [r.get("bytes", 0) for r in ok]
        res = dict(N=n, wall_s=round(wall, 1), n_ok=len(ok), n_fail=len(recs) - len(ok),
                   fov_per_h=round(3600 * len(ok) / wall, 1) if wall else 0,
                   mean_process_s=round(sum(per) / len(per), 1) if per else 0,
                   max_process_s=round(max(per), 1) if per else 0,
                   peak_ram_gb=round(peak, 1),
                   regions_min=min(regs) if regs else 0, regions_max=max(regs) if regs else 0,
                   regions_mean=round(sum(regs) / len(regs)) if regs else 0,
                   cells_mean=round(sum(cells) / len(cells)) if cells else 0,
                   mean_mb=round(sum(byt) / len(byt) / 1e6, 1) if byt else 0)
        results.append(res)
        print(f"\n[nconc] N={n}: wall {wall:.0f}s  {res['fov_per_h']} FOV/h  "
              f"peak RAM {res['peak_ram_gb']} GB  regions {res['regions_min']}-{res['regions_max']} "
              f"(mean {res['regions_mean']})  mean {res['mean_process_s']}s/FOV  "
              f"{res['n_fail']} failures", flush=True)
        json.dump(results, open("/home/jovyan/bench/nconc_results.json", "w"), indent=2)

    _stop.set()
    print(f"\n\n{'='*70}\n[nconc] SUMMARY — same {N_FOV} FOVs at each level\n{'='*70}")
    print(f"{'N':>3}{'wall s':>9}{'FOV/h':>9}{'speedup':>9}{'mean s/FOV':>12}{'peak RAM':>10}"
          f"{'MB/FOV':>9}{'fail':>6}")
    base = next((r for r in results if r["N"] == min(LEVELS)), results[-1])
    for r in sorted(results, key=lambda r: r["N"]):
        print(f"{r['N']:>3}{r['wall_s']:>9.0f}{r['fov_per_h']:>9.1f}"
              f"{r['fov_per_h']/base['fov_per_h']:>8.2f}x{r['mean_process_s']:>12.0f}"
              f"{r['peak_ram_gb']:>9.0f}G{r['mean_mb']:>9.0f}{r['n_fail']:>6}")
    r8 = next((r for r in results if r["N"] == 8), None)
    if r8 and r8["fov_per_h"]:
        print(f"\ncorpus projection at N=8: 6504 FOVs / {r8['fov_per_h']:.1f} FOV/h = "
              f"{6504/r8['fov_per_h']:.1f} h = {6504/r8['fov_per_h']/24:.2f} days")
        print(f"output projection: {r8['mean_mb']:.0f} MB/FOV x 6504 = "
              f"{r8['mean_mb']*6504/1e6:.2f} TB")
    print("[nconc] done", flush=True)


if __name__ == "__main__":
    main()
