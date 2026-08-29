# Benchmark, profiling and parameter-sweep scripts

Every script that produced a number or a default in the AB batch-analysis README
(`s3://spatial-technology-platform/AB/pyali_3fc51c7_outputs/README.md`). They are committed here
**verbatim**, exactly as they ran, so the published numbers and the tuned parameters are traceable
to code rather than to prose.

Run with `/home/jovyan/.venvs/pyali/bin/python`. The system conda (`/opt/conda`, numpy 2.2.6) is
broken for this work — `skimage`, `h5py`, `sklearn` and `pandas` there were built against numpy 1.x
and fail to import.

**Results are not in git.** Logs, JSON, CSV and the HTML contact sheets stay under
`s3://spatial-technology-platform/AB/pyali_3fc51c7_outputs/` — the `S3 results` column below gives
the path relative to that prefix.

## I/O and throughput — §4

| script | what it measured | claim it supports | S3 results |
|---|---|---|---|
| `bench_io.py` | s3fs read throughput vs stream count | flat ~120 MB/s across a 64× concurrency sweep — 1.9% of 50 Gbps | `bench/io_scaling.json`, `.log` |
| `bench_s3_ceiling.py` | boto3 parallel ranged GETs, no filesystem | 658 MB/s, 5.5× s3fs | `bench/s3_ceiling.json`, `.log` |
| `bench_mount_ab.py` | s3fs vs Mountpoint head-to-head | 3278 MB/s at 16 streams, 3962 MB/s on the production pattern — 32–38× | `bench/mount_ab.json`, `.log`, `mount_sustained.log` |

## Pipeline profiling and RAM — §4, §7

| script | what it measured | claim it supports | S3 results |
|---|---|---|---|
| `bench_pipeline.py` | first per-stage end-to-end profile | the pre-fix baseline the later profiles are compared against | `bench/mar16bit.log`, `july8bit.log` |
| `profile_highN.py` | per-stage profile of a representative high-region FOV | the 387 s / 536-region stage table (`extract_footprints` 53.9%) | `bench/profile_highN.json`, `profile_highN_before.json`, `.log`, `profile_after.log` |
| `pinv_shape_test.py` | per-shape timing + peak RSS after the `pinv_traces` fix | RAM amplifier 1 — float64 promotion of the whole movie | `bench/pinv_shape_results.json`, `pinv_shape.log`, `pinv_1080*.json/.log` |
| `fastpath.py`, `fastmedian.py` | prototypes: patch-local filtering, sorting-network median | RAM amplifier 3 — `temporal_filter` materialising a second movie | — (prototypes) |
| `fastpath_ab.py` | fastpath A/B against saved baselines | 9.5×, bit-identical on 4 real FOVs (`cell_traces`, `footprint`, `footprint_center` all `array_equal`) | `bench/fastpath_ab_results.json`, `.log` |
| `nconc_test.py` | N-concurrency throughput and peak RAM | N=8 is the ceiling: 195.3 GB of 249, 58.6 FOV/h, 4.62 days | `bench/nconc_results.json`, `nconc.log`, `corpus_test.log` |

## Segmentation survey, guard and parameter sweeps — §5

These are the ones that set current defaults, so treat them as the provenance for
`Params.sharpen_k`, `Params.seg_neighborhood` and `Params.max_region_bbox_frac`.

| script | what it measured | claim it supports | S3 results |
|---|---|---|---|
| `seg_survey.py` | 525-FOV region-size survey at 7 burst positions per (plate, well) | 194,564 regions, 0 errors, 608 s; 94.0% of edge-of-sequence FOVs carry a >3% region vs 1.8% interior — a 52× enrichment | `bench/seg_survey.csv`, `seg_survey_fovs.csv`, `.log` |
| `seg_analyze.py` | analysis of that survey | the cleanly bimodal size distribution and the extent table → **bbox guard at 1%, not 3%** (84 extra regions of 194,564) | reads `bench/seg_survey.csv` |
| `guard_test_seg.py` | guard verification on 10 FOVs, 7 invariants each | kept/dropped all on the right side of the limit, footprints index-aligned, worst patch 24.13 GB → 0.38 GB | `bench/guard_seg_results.json`, `guard_test_fovs.json` |
| `seg_qc.py` | 260-FOV segmentation QC sweep | the QC sweep in §8: done, 0 errors | `seg_qc/index.html` + 10 contact sheets + 260 overlays, `bench/seg_qc.log` |
| `seg_grid.py` | multiplier × `sharpen_k` grid, 25 FOVs | **multiplier is the wrong knob** — 1.5 → 2.25 cut regions to 0.55× and median area to 0.68×; rejected. `sharpen_k=3.0` | `seg_grid/index.html` (25 FOVs × zoom+full), `bench/seg_grid.log` |
| `seg_nbhd.py` | `seg_neighborhood` sweep, 25 FOVs × 6 neighbourhoods | **neighbourhood is the right knob** — 51×51 gives +4% regions at 97% of median area; 21×21 over-fragments (1.18× count, 0.77× area) | `seg_nbhd/index.html`, `bench/seg_nbhd.log` |

## Known caveats — these scripts do not run unmodified today

Committed verbatim deliberately, so what is in git is provably the code that produced the published
numbers. That means the environmental assumptions came along with them:

1. **Outputs are hardcoded to `/home/jovyan/bench/...`** — local NVMe, not persisted. The directory
   is gone after a reboot; the archived results in S3 are the surviving copies.
2. **Inputs reference superseded `keep.csv` snapshots** — `pyali_c27fc46_outputs/keep.csv`
   (`bench_io`, `bench_mount_ab`, `bench_s3_ceiling`, `guard_test_seg`, `seg_survey`) and
   `pyali_6b59e79_outputs/keep.csv` (`nconc_test`, `seg_qc`, `seg_grid`). **Neither prefix still
   exists** — only `pyali_3fc51c7_outputs/` remains. Repoint at
   `pyali_3fc51c7_outputs/keep.csv` before re-running; the 6504-FOV row set is the same.
3. **Read roots assume the Mountpoint mounts** (`/mnt/s3ab`, `/mnt/s3wb`), which are not persisted
   either — see the cold-start section of the outputs README to rebuild them. The s3fs paths work
   but are ~32× slower.
4. `seg_grid.py` reads `/home/jovyan/seg_qc/seg_qc.json`, i.e. it depends on `seg_qc.py` having been
   run first into a local scratch path.

Fixing 1–4 is a **tidy pass, deliberately deferred** so that this first commit stays a faithful
record. Do it as separate commits once the corpus extraction run has finished.

## Deliberately not archived

Regenerable and large:

- `data/` — 9.5 GB of raw `frames1.bin` copies, byte-identical to the source movies in S3
- `out/`, `pinv_out/`, `fastpath_out/`, `profile_out/`, `guard_out/` — 2.1 GB of `.mat` A/B
  baselines; re-create by re-running the script that produced them.
