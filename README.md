# pyali

A voltage-imaging waveform-extraction pipeline. Given a raw movie of a field of view, pyali:

1. builds a reference image (mean weighted by local temporal correlation),
2. subtracts a non-uniform, time-varying background and corrects rigid motion,
3. segments cells,
4. detects action potentials per region,
5. extracts per-cell spatial footprints (center-of-mass + clustering), and
6. recovers per-cell temporal traces via the spatial pseudoinverse.

Arrays are row-major C-order; movies are `[T, H, W]` float64.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies: numpy, scipy, scikit-image, scikit-learn, h5py, matplotlib, and (for the
interactive figure) plotly.

## Analyze a movie

Point the CLI at a folder containing `frames1.bin` (plus the usual sidecars
`output_data.mat`, `frames1_dropped_frames.txt`, `frames1_ROI_mean_stdev.txt`). Frame
dimensions are auto-detected; `--figures` also saves the result plots.

```bash
python scripts/run_pyali.py "/path/to/your/video_dir" --figures
```

Writes to `/path/to/your/video_dir/analysis/` (the layout is defined in `pyali/results.py`,
the only writer and loader of these files):

- `cells.parquet` (+ `cells.csv` for eyeballing) — one row per cell:
  `cell_id` = `<fov_id>_C####`, `experiment_id`, `plate`, `well`, `fov`, `fov_id` =
  `<experiment>_P<plate:02d>_<well>_F<fov:03d>`, `fov_index` (position in a corpus run; 0 when
  unknown), `local_cell_id` (1..N), `trace_row` (0-based row in `signals.h5`), `x_px`/`y_px`
  (footprint centre, 1-indexed), `region_id` (segmentation region), `n_spikes` (APs assigned
  to the cell), `source` (`spiking` for a clustered footprint; `no_ap` / `no_com` / `all_noise`
  for a region that fell back to one footprint because no AP was detected / no centroid
  survived the mask / every centroid was DBSCAN noise), `weighted_brightness` (footprint-
  weighted mean of the brightness reference; NaN when the cell has no positive weight)
- `spikes.parquet` — one row per localized AP: `region_id`, `frame` (0-based), `row`/`col`
  (centre of mass, 1-indexed), `amplitude`, `cell_row` (-1 = not assigned to a cell)
- `signals.h5` — `/traces/raw` float32 `[T, N]` on disk (gzip; MATLAB/Octave read it as
  `[N, T]`), `/footprints/{values,pixel,offsets}` (sparse COO per cell, C-order pixel index
  `row * W + col`), `/images/{reference,binary_map,brightness_reference}` (the last is the raw-
  unit mean of the first 300 frames minus a radius-50 grey opening, the `x` in
  `weighted_brightness`), and attributes `height`, `width`, `n_frames`, `fps`, `fov_id`,
  `pyali_git`, `schema_version`
- `fov_metadata.json` — acquisition metadata, `identity`, every `Params` field under `params`,
  per-stage wall seconds under `timing_s`, cell counts by `source`, the git commit

Identity defaults come from the directory name (`P-4` → plate 4, `W-A2` → well, `burstN` → fov,
the batch token → experiment id); override with `--experiment-id --plate --well --fov --fov-index`,
e.g. `--plate 5` for the 20260715 folders that are named `P-1` but are plate 5.
- with `--figures`: `detected_regions.png`, `coms.png`, `cell_traces.png`,
  `center_of_cell_regions.png`, and **`cell_traces.html`** — an interactive plot
  (zoom/pan, hover for values, click a legend entry to hide/isolate a trace)

Read a result directory back with `pyali.results.load_results`:

```python
from pyali.results import load_results
r = load_results("/path/to/your/video_dir/analysis")
r.traces            # (N, T) float32, spikes positive
r.centers           # (N, 2) [row, col], 1-indexed
r.cells, r.spikes   # DataFrames
r.footprints.plane(k)   # dense (H, W) footprint of cell k; .dense() rebuilds the (H, W, N) array
```

`ALI_Result.mat` / `ALI_Int_Result.mat` are no longer written; `pyali.io.load_v73` stays for
reading archived runs.

A full field of view takes several minutes; progress is printed per stage.

## Python API

```python
from pyali.pipeline import process_fov
from pyali.params import Params

# frame dimensions and background/baseline frame ranges are configurable
p = Params(nrow=312, ncol=1200)
out = process_fov("/path/to/video_dir", out_dir="/path/to/analysis", p=p, make_figures=True)
footprint, traces = out["footprint"], out["cell_traces"]
```

`Params` holds all tunables (frame size, frame rate, background/baseline ranges, sharpening,
AP threshold, clustering, etc.). The background and baseline frame ranges are acquisition-
specific — set `Params.bkg_ranges` / `Params.std_ranges` for your protocol.

## Benchmarking a pipeline change (SNR)

`snr_analysis/` is a harness for testing whether a change to the pipeline actually improves the
signal-to-noise ratio of the extracted waveforms, by comparing two runs on the same movie
(see [`snr_analysis/README.md`](snr_analysis/README.md)). Run the pipeline two ways, then:

```bash
python snr_analysis/snr/snr_compare.py RUN_A_DIR RUN_B_DIR --label-a new --label-b baseline --out report
```

It reports per-cell high-frequency noise floor, spike SNR, spectral HF-SNR, and cross-run
correlation/coherence, so an improvement is a measurable drop in noise floor and rise in spike
SNR — not just a changed output.

### Optional: whitened GLS trace extraction

`--whiten-traces` swaps the default unweighted pseudoinverse for a noise-weighted
(generalized-least-squares) trace extractor (`Params.whiten_traces`, off by default):

```bash
python scripts/run_pyali.py "/path/to/video_dir" --whiten-traces
```

This is an opt-in experiment — on shot-noise-limited data the benchmark above shows it does not
improve SNR (the bright signal pixels are also the noisiest, so inverse-variance weighting
discards signal); it is left in as a worked example of the benchmarking workflow. Default off
reproduces the standard pseudoinverse exactly.

## Helper: recover a movie's frame dimensions

`scripts/find_video_dims.py` recovers `(nrow, ncol)` from a headerless `.bin` by choosing the
factor pair of the pixels-per-frame that maximizes adjacent-pixel correlation. Pixels-per-frame
is auto-detected from the sidecar `.txt`:

```bash
python scripts/find_video_dims.py FRAMES.bin
```

## Architecture

`scripts/arch_graph.py` derives the module / function / dataflow graph from the source with the
standard-library `ast` module and renders two figures: the core voltage-extraction dataflow through
`process_fov`, and the post-hoc QC pass as script → file → script. Outputs live in
`docs/architecture/` (`architecture.html` is the self-contained page; the figures are also written as
standalone SVG and as Mermaid text, and `arch.json` is the full graph). They are committed, so
`--check` fails when they no longer match the source.

```bash
python scripts/arch_graph.py            # regenerate docs/architecture/
python scripts/arch_graph.py --check    # exit 1 if the committed outputs are stale
```

## References

This algorithm is modeled after the voltage extraction pipelines in Zhang, J., Gong, D., Barrios, A., et al. 2026 (unpublished), and Chen, TW, et al. Nature Methods (2025). 
