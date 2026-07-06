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

Writes to `/path/to/your/video_dir/analysis/`:

- `ALI_Int_Result.mat` — reference image and the sharpening chain
- `ALI_Result.mat` — `footprint`, `footprint_center`, `cell_traces`
- with `--figures`: `detected_regions.png`, `coms.png`, `cell_traces.png`,
  `center_of_cell_regions.png`, and **`cell_traces.html`** — an interactive plot
  (zoom/pan, hover for values, click a legend entry to hide/isolate a trace)

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

## References

This algorithm is modeled after the voltage extraction pipelines in Zhang, J., Gong, D., Barrios, A., et al. 2026 (unpublished), and Chen, TW, et al. Nature Methods (2025). 
