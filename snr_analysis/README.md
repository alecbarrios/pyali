# snr_analysis — benchmarking waveform SNR across pipeline iterations

This folder is a **regression/benchmarking harness for the extracted waveforms**. When you change
the pipeline (a new filter, a new footprint rule, a new trace-extraction method), the question is
not "did the output change?" but "did the change actually make the extracted spike waveforms
*better* — higher signal-to-noise — without distorting them?" This harness answers that
quantitatively by comparing two pipeline runs on the **same movie**.

## The idea in one sentence

Run the pipeline twice (iteration A vs iteration B), then measure the high-frequency SNR of the
extracted `cell_traces` on both and test, cell by cell, whether A improved on B.

## What it measures

Each cell's trace is high-pass filtered (default 20 Hz) to isolate the fast spike band from slow
bleaching drift, then:

| metric | meaning | better |
|---|---|---|
| `noise_sigma` | robust noise floor, `1.4826 × MAD` of the high-pass trace | lower |
| `snr_median` / `snr_p90` / `snr_max` | detected-spike amplitude ÷ `noise_sigma` (typical / strong / best) | higher |
| `spectral_hf_snr` | excess Welch-PSD power in the 20–150 Hz spike band over the white noise floor near Nyquist | higher |
| `psd_floor` | the white (shot-noise) power floor, measured 300–400 Hz | lower |
| `corr_raw` / `corr_hp` | correlation of the two runs' raw / high-pass traces per matched cell | ~1 = same waveform |
| `coh_hi` | mean coherence (per-frequency agreement) of the two runs, 100–300 Hz | ~1 = no HF loss |

Cells are matched one-to-one between runs by footprint-center proximity (Hungarian assignment),
so we always compare the same physical cell. A genuine improvement shows **`noise_sigma`/`psd_floor`
down and `snr_*` up**, while `corr_hp ≈ 1` on cells the change should not touch confirms no
distortion or fabricated spikes.

## Layout

```
snr_analysis/
├── snr/
│   ├── snr_compare.py        # the benchmark: compare run A vs run B, emit CSV + summary + figures
│   └── snr_postproc_demo.py  # measure post-hoc trace tweaks (e.g. mains-hum notch) with a
│                             #   built-in distortion guard
├── report_whiten_vs_baseline/  # worked example (see below): whitened-GLS vs baseline A/B
└── README.md
```

## How to run an A/B (two iterations)

From the repo root, with the project's Python:

```bash
PY="pyali/bin/python"                       # the venv merged into the package dir
FOV="/absolute/path/to/your/video_dir"      # folder containing frames1.bin

# 1) produce the two runs (into separate output dirs):
$PY scripts/run_pyali.py "$FOV" --out /tmp/run_baseline
$PY scripts/run_pyali.py "$FOV" --out /tmp/run_new --whiten-traces   # or whatever the change is

# 2) benchmark A (new) against B (baseline):
$PY snr_analysis/snr/snr_compare.py /tmp/run_new /tmp/run_baseline \
    --label-a new --label-b baseline --out snr_analysis/report_new_vs_baseline
```

`snr_compare.py` prints the paired summary and writes `summary.txt`, `summary.json`,
`per_cell_metrics.csv`, and diagnostic figures (`coherence.png`, `mean_psd.png`,
`example_matched_cell.png`, `corr_hist.png`, `distributions.png`, three `scatter_*.png`).
Each positional argument may be an `ALI_Result.mat` or a directory containing one.

## Worked example: the `--whiten-traces` experiment (benchmarked, not adopted)

`report_whiten_vs_baseline/` is a real use of this harness. The default trace extractor is an
unweighted spatial pseudoinverse that implicitly trusts every footprint pixel equally. The
`--whiten-traces` option instead solves a noise-weighted (generalized-least-squares) problem that
down-weights noisier pixels — in theory the minimum-variance estimator.

The benchmark verdict on the test movie: **it did not improve HF-SNR.** On the cells it modified,
the noise floor dropped ~3.2% but spike amplitude dropped ~4.6%, so net spike-SNR fell ~1.5%
(`snr_median` 4.70 vs 4.75, Wilcoxon p≈8e-7; `spectral_hf_snr` 1.16 vs 1.25). The reason is
physical: in shot-noise-limited imaging the bright pixels carry the signal *and* the most photon
noise, so weighting by `1/noise²` discards signal faster than noise. The intensity-weighted
footprint with a plain pseudoinverse is already near the right (matched-filter) weighting.

So `whiten_traces` stays **off by default** — the harness caught a plausible-looking change that
does not help this data. That is exactly what the harness is for: only adopt a change once the
numbers here show `noise_sigma` down and `snr_*` up.
