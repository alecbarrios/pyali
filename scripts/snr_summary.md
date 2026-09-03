# SNR summary statistics — the three-stage post-hoc pass

Implements §9 of the AB batch-analysis README
(`s3://spatial-technology-platform/AB/pyali_3fc51c7_outputs/README.md`).

**Post hoc, not in the pipeline.** These scripts read `cell_traces` from `ALI_Result.mat` files
already in S3 and apply `pyali.metrics.per_cell_snr` **unchanged**. The extraction pipeline is
never modified and no movie is ever reprocessed, so the metrics can be recomputed or extended
without touching 0.48 TB of results. A full refresh is minutes, not days.

Run with `/home/jovyan/.venvs/pyali/bin/python`, in this order:

| stage | script | writes |
|---|---|---|
| 1 | `snr_corpus.py` | one `snr_metrics.csv` per FOV, beside its `.mat` outputs in S3 |
| 2 | `snr_aggregate.py` | `cells_all.parquet` + five summary tables + `manifest.json` |
| 3 | `snr_figures.py` | 10 PNGs (light + dark) + `index.html` |

```bash
python scripts/snr_corpus.py --workers 8      # ~9 min for 1200 FOVs / 530k cells
python scripts/snr_aggregate.py               # ~1.5 min
python scripts/snr_figures.py                 # ~1 min
```

Stages 2 and 3 default to
`/home/jovyan/spatial-technology-platform/AB/pyali_3fc51c7_outputs/snr_summary/`.

## Stage 1 — `snr_corpus.py`

One `snr_metrics.csv` per FOV, written **into the same directory as the `.mat` outputs** — the
depth rule `fov_metadata.json` already follows, so aggregation stays a single glob:

```
<out_root>/<day>/<fov_dirname>/
    ALI_Int_Result.mat   ALI_Result.mat   fov_metadata.json   snr_metrics.csv
```

Columns: `cell_index, noise_sigma, snr_median, spectral_hf_snr, n_spikes`. `fps` comes from
`fov_metadata.json` → `analysis.fps`; `per_cell_snr` runs at its documented defaults
(`hp=20 Hz, k=3σ, sig_hi=150 Hz, floor_lo=300 Hz`).

- **Safe against a live extraction run.** A FOV is eligible only once *all three* extraction
  outputs are present. `run_corpus.py` uploads them with one `aws s3 cp --recursive`, so they land
  at slightly different moments; S3 objects appear atomically, so there are no torn reads.
- **Resumable and incremental.** FOVs already carrying a CSV are skipped, so re-running as further
  days finish processes only what is new.
- **Zero-cell FOVs get a header-only CSV**, so they are provably done rather than merely skipped
  (§5 notes three FOVs legitimately segment to zero regions).
- **Writes through the s3fs mount** by default rather than shelling out per FOV: these are ~30 KB
  files, and ~1200 `aws` invocations would cost more in process startup than the whole
  computation. `--upload awscli` keeps the other path available.

> **Read `snr_metrics.csv` with `float_precision="round_trip"`.** Floats are written with `repr`,
> i.e. shortest-round-tripping. pandas' *default* parser is a fast approximate one and was measured
> drifting **up to 50 ULP** on small `noise_sigma` values — harmless for a median, but it silently
> breaks any bit-exact A/B against a recomputation.

## Stage 2 — `snr_aggregate.py`

`cells_all.parquet` is one row per cell, carrying day / timestamp / plate / well / DIV / batch /
burst / bit depth / dims / profile, so it is self-contained.

**Median + IQR, at FOV level first, then rolled up**: FOV → well → plate → calendar day → whole
set. Summarising per FOV before aggregating is what stops FOVs with many cells from dominating a
well — counts range from 8 to 1240 within a single well here.

**Every level above the FOV is computed from the FOV medians**, not by re-taking a median of the
level below. A plate is the median over *its own* FOVs, not a median of well-medians. Each level
therefore reports a directly comparable statistic with an honest `n_fovs`, and no level is a
median of medians of medians. §9 lists the levels but does not settle this; the choice is recorded
here so it is not silently reinterpreted later.

**NaN is reported, never dropped.** `n_cells`, `n_nan` and `nan_frac` accompany the median and IQR
at every level, and NaN counts always come from the raw cell rows rather than from summarised
ones. `per_cell_snr` returns NaN where a metric is genuinely undefined, so the NaN rate is the
fraction of segmented cells with no detectable activity — a result in its own right.

**Wells are keyed `(day, plate_num, well)`.** `20260331_dir1/P01/A1` and `20260715/P-1/A1` are
different wells and must never merge. `keep.csv` stores the plate as a float-like string (`"1.0"`)
while `fov_metadata.json` carries an integer `plate_num`, so the integer is the join key. Expected
FOV counts from `keep.csv` label each well complete or partial, so a 17-of-56 well is never
mistaken for a finished one.

`manifest.json` records what each refresh covered: per-day FOV and cell counts, every well with
`n_fovs` / `n_fovs_expected` / `complete`, the corpus-wide NaN rates, the S3 prefix, and pyali's
`git describe` / commit / branch / dirty flag. `per_cell_snr`'s parameters are captured by
inspecting the function signature rather than hardcoded, so they cannot drift out of sync.

## Stage 3 — `snr_figures.py`

Violins of the three metrics at **well / plate / day / whole-set** level, plus cells-per-FOV at
well / plate / day. Light and dark renders, and an `index.html` gallery matching the
`seg_qc/` · `seg_grid/` · `seg_nbhd/` convention.

**Each violin is a distribution over per-FOV medians** — one point is one FOV. That is what §9's
"summarize at FOV level first" rule means for a figure: a 1240-cell FOV must not outweigh an
8-cell one inside the same well. IQR is a vertical bar, median a ringed dot. Cell-count panels are
the exception, since there the per-FOV count *is* the quantity.

- **Colour encodes the acquisition day**, never rank or position — days differ in bit depth and
  frame size, which is exactly what drives these metrics, so the legend carries both (§9). Hues
  are the first three categorical slots in fixed order, so a day keeps its hue in any subset.
- **Palettes were validated, not eyeballed.** Light passes every gate under `--pairs all`
  (worst-pair CVD ΔE 9.2, normal-vision 24.0); dark passes at 9.4 / 20.9. Aqua sits at 2.74:1 on
  the light surface, below the 3:1 bar, so the **relief rule** applies: every violin carries a
  visible `n` label and the summary CSVs are the table view, so identity is never colour-alone.
  Dark mode is a *selected* second render against the dark surface, not an automatic inversion.
- **Axis scales.** `noise_sigma` is log — it spans an order of magnitude between the 16-bit March
  day and the 8-bit July one. `spectral_hf_snr` is **symlog**: it is an excess ratio that can go
  negative, so plain log is unavailable, and its long upper tail otherwise flattens every violin
  into a line.
- **450 dpi in an Arial-first font stack**, for projection at full-wall size. `--dpi` adjusts.
  Neither Arial nor Calibri is installed on the analysis box, so it falls through to Liberation
  Sans — metrically identical to Arial, so layout is unchanged. `pdf.fonttype=42` /
  `svg.fonttype="none"` keep text selectable if re-saved as vector for a deck.
- Partial wells label inline as `(have/expected)`; NaN % is annotated wherever non-zero. Titles
  are nouns — "Noise floor", "Spike SNR", "Cells per FOV".

## Future work — not implemented

Recorded here so the design decisions are not re-derived. Full rationale in §10 of the outputs
README.

**Per-cell noise reference by spike-window excision (§10a).** The floor is currently the mean PSD
over 300–400 Hz, which assumes the noise is white from 20 Hz up and that no spike power reaches
300 Hz. The second assumption is doubtful: the spike-triggered average has FWHM ≈ 2.5 ms — only 2
samples at 800 Hz — so AP content extends to roughly Nyquist, and anything above 400 Hz aliases
back into the upper band where the floor is measured. Replace it by excising each spike *and its
afterpotential* (the STA undershoot is still −0.96σ at +10 ms) from that cell's own trace and
estimating the noise from the residual. Same cell, so brightness, expression and shot-noise scale
match by construction — which a silent-cell population cannot offer, since silent cells are dim or
poorly expressing and shot noise scales with brightness.

**Post-hoc single-cell filter (§10b) — a segmentation-quality tool, NOT a noise reference.**
Separates true single-cell segmentations from merged clumps, debris and fragments. Two criteria on
`cells_all.parquet` plus per-cell footprint geometry; no movie is re-read:

| criterion | rule |
|---|---|
| activity | `n_spikes == 0` — *nominates* candidates only |
| morphology | footprint `extent` (`area / bbox_area`) and `area_frac`, against **sweepable** thresholds — suggested `min_extent` 0.2–0.5, `max_area_frac` 0.1–0.5% |

§5 measured the separation these exploit: compact real regions have median extent 0.483 vs 0.139
for sprawling artifacts, and real regions sit below ~0.3% of frame area while artifacts run 1–84%.
Sweep as `seg_neighborhood` and `sharpen_k` were swept, reporting cell yield and the three SNR
medians per setting; the operating point is where yield still falls slowly but the medians stop
improving.

**Morphology must carry the decision, never power.** Selecting on a power statistic and then
computing power statistics on the survivors is circular, and it selects the wrong objects: merged
clumps mix several sources and so have inflated variance, so a "high band-limited power, no spikes"
rule would preferentially *admit* clumps. Footprint geometry comes from the segmentation,
independent of the trace, so it cannot feed back into the metric it cleans. Note the two error
modes are asymmetric — a genuinely silent *real* cell (no activity, compact footprint) is a
biological result worth counting, whereas a clump is an artifact; a filter tuned only to raise
median SNR deletes both.

**Deferred: a 5–20 Hz subthreshold band (§10c).** Cheap (the PSD is already computed on the raw,
not high-passed, trace) but it would not measure subthreshold depolarization: active-vs-quiet power
ratio is 6.76× at 1–20 Hz against 6.73× at 20–50 Hz, because the spike-train envelope and the AP
afterpotential both land there. Requires §10a first.

**Free metrics not yet visualized (§10e).** Already columns in `fov_summary.csv` — visualization
only, ~15 min each: silent fraction (`snr_median_nan_frac`, the most direct activity readout),
cells per region (`n_cells / n_regions_kept`, a segmentation-quality readout), regions dropped
(`n_regions_dropped`, the 1% guard's per-FOV footprint), and per-well/per-plate acquisition-order
trends.

**Spike-triggered average (§10f) — scoped, not implemented.** Two figures: **per FOV** with the
band showing **spread across cells within that FOV** (cell-to-cell waveform heterogeneity), and
**per well** aggregating the per-FOV STAs. Per-plate/day/whole-set deferred. **Units: σ** — divide
each cell's snippet by that cell's own `noise_sigma` before averaging, for line and band alike,
since absolute trace units are not comparable across days. Cost: `APs`/`COMs` are computed but
never saved, so spike times must be re-detected — a full 32.7 GB trace re-read (~22 min), or ~1%
of that on a stratified sample. Build in a ±10–15 ms window (the undershoot is −0.96σ at +10 ms)
and sub-sample peak alignment (the AP is 2 samples wide at 800 Hz, so nearest-sample alignment
broadens the average).

**Fold STA aggregation into `snr_aggregate.py` (§10f-ii) — deferred.** It currently lives in
`snr_sta_figures.py` because its output is *curves* rather than scalars, and keeping it separate
avoided touching validated code mid-run. Later, produce `sta_by_fov.parquet` and `sta_by_well.csv`
alongside `cells_all.parquet` and the roll-up tables, recorded in the same `manifest.json`, leaving
`snr_sta_figures.py` as rendering only.

**Second per-FOV STA set from the seg_nbhd FOVs (§10f-i) — planned, ~30 min.** 7 of the 25 FOVs in
`seg_nbhd/seg_nbhd.json` fall in this run's three days and already have `sta.csv`, so no trace
re-reads. New stems `sta_nbhd_grid` / `sta_nbhd_grid_sem`; must not clobber `sta_fov_grid*`.

**Decided: band edges stay fixed corpus-wide (§10d).** Per-cell floor *value* with fixed edges is
correct and already implemented. Per-FOV adaptive *edges* are rejected — each FOV would integrate a
different frequency range, so a well-to-well difference could be a band difference rather than a
biological one.

## Var_meas — the one formula that is easy to get wrong

`Var_meas = mean(1 / n_spikes_used)`, averaged over **every** cell. It is the spread you would still
see if all cells had an identical true waveform, since each cell's STA averages a finite noisy
sample of its own spikes.

**Do not substitute `1/median(n)`.** `1/n` is convex, so the few very-low-spike cells dominate the
average: on a 1398-cell FOV, `mean(1/n) = 0.0369` against `1/median(n) = 0.0286` — a **1.29×**
difference, because the 3.4% of cells with n ≤ 10 contribute 11.5% of `Σ(1/n)`. Using the median
understates the noise floor ~30% and inflates the variance ratio.

**Units**: every snippet is divided by that cell's `noise_sigma`, so one normalised snippet has
variance **1** by construction and the mean of `n` of them has variance `1/n`. `Var_obs` and
`Var_meas` are therefore in the same (dimensionless) units — writing "σ²" is shorthand for
"variance of an amplitude in units of the noise σ" — which is why their ratio is meaningful.

## Pitfalls worth knowing before touching this code

Full catalogue in §12 of the outputs README. The ones that bite hardest here:

- **Read `snr_metrics.csv` and `sta.csv` with `float_precision="round_trip"`** — pandas' default
  parser drifts up to 50 ULP on small `noise_sigma`.
- **Any shared write helper must take its filename from the caller.** `snr_corpus._write` once
  hardcoded `snr_metrics.csv` and overwrote three FOVs' metrics with STA content.
- **Drive per-FOV summaries off the FOV list, not a groupby of the cell table** — a zero-cell FOV
  contributes no rows and silently vanishes.
- **Re-base `burst` per well** before any acquisition-order analysis: it is within-well for
  20260612/20260715 but global for 20260331_dir1.
- **Detect on the 20 Hz high-passed trace, measure waveform shape on the raw one** — the filter is
  zero-phase and stamps a symmetric −0.45σ pre-spike dip on anything measured from it.
- **Resolve `dpi` explicitly in every figure function** — `savefig(dpi=None)` silently falls back to
  100 dpi.
- **Never poll with `pgrep -f <pattern>`** — it matches its own command line and the loop never exits.
- **The repo is on shared s3fs.** While an extraction run is live, only add files and commit; never
  `checkout`/`stash`/`reset`/`merge`, which rewrite the working tree another instance is importing.

## Known issue — stale CSVs after a `--no-resume` re-run

`run_corpus.py --no-resume` re-processes a FOV and re-uploads with `aws s3 cp`, which **does not
delete**. The old `snr_metrics.csv` therefore survives beside the *new* `ALI_Result.mat`, and
`snr_corpus.py`'s resume check would treat that FOV as already measured — leaving metrics silently
attributed to traces that no longer exist.

Not fixed here, and deliberately recorded rather than patched. The obvious remedies are to compare
the CSV's mtime against `ALI_Result.mat`'s, or to have `run_corpus.py` delete the CSV when it
re-processes a FOV. Until then, **pass `--force` after any `--no-resume` extraction re-run.**
