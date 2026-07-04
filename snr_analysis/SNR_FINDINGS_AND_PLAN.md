# pyali vs MATLAB miniALI — High-Frequency SNR: findings & improvement plan

_Analysis date: 2026-07-04. Dataset: `101034_P02_6w_A1_JF608_6GP002_DIV37_burst98`._

## TL;DR

1. **The Python port is NOT lossy. Its extracted-waveform high-frequency SNR is numerically
   identical to MATLAB's.** On the test FOV, 350 of 351 matched cells have high-pass trace
   correlation > 0.9999; cross-pipeline coherence is flat at 1.0 to Nyquist; spike-SNR, noise
   floor and spectral HF-SNR are statistically indistinguishable (Wilcoxon p 0.19–0.71).
2. **To beat MATLAB you must improve the shared algorithm.** An exhaustive, adversarially-verified
   search of the codebase found that **23 of 24** candidate improvements do **not** raise
   extracted-trace HF-SNR (with evidence). **One** survived: **whitened GLS trace extraction**,
   now implemented behind `Params.whiten_traces` (default OFF, so fidelity is preserved).

---

## Q1 — Is MATLAB's HF-SNR higher? No.

Method (`scripts/snr_compare.py`): load `cell_traces [N,T]` + `footprint_center [N,2]` from both
`ALI_Result.mat` files, match cells one-to-one by footprint center (Hungarian), then per matched
cell compute — on a >20 Hz high-pass of each trace — the robust noise floor (1.4826·MAD), spike
SNR (peak/floor), and a Welch spectral HF-SNR (excess spike-band power over the white floor near
Nyquist). Cross-pipeline Pearson correlation and magnitude-squared coherence quantify agreement.

Result (`snr_report_101034/summary.txt`):

| metric | pyali | MATLAB | pyali-wins frac | Wilcoxon p |
|---|---|---|---|---|
| HF noise floor (lower=better) | 0.05123 | 0.05123 | 0.52 | 0.71 |
| spike SNR (median) | 4.736 | 4.736 | 0.47 | 0.21 |
| spike SNR (p90) | 5.833 | 5.833 | 0.47 | 0.41 |
| spectral HF-SNR | 1.254 | 1.250 | 0.52 | 0.19 |

- Median paired high-pass correlation = **1.00000 at every cutoff 10–150 Hz**.
- **Coherence flat at 1.0 across the whole spectrum** (`coherence.png`) — no HF loss at any band.
- 1/351 cells diverges (a data-dependent DBSCAN/AP-set branch); there Python found *more* events.
- Cross-check: the MATLAB `.fig` "Normalized Cell Traces" is exactly the `.mat` `cell_traces`
  plotted with per-trace min/max `rescale` (SNR-invariant) and last 100 frames dropped
  (fig line 0 ↔ `.mat` row 0 correlate at 1.000000). So the compared objects are correct.

**Conclusion:** the perceived difference between the Python `cell_traces.png` and the MATLAB
`.fig` is cosmetic (per-trace rescale + static-vs-interactive rendering), not a data difference.

---

## Q2 — How to extract higher-SNR waveforms

### The governing insight

A given cell's extracted-trace HF-SNR is set almost entirely by **(its spatial footprint template)
× (the `pinv` projection onto the movie)**. Everything upstream — AP detection, the DOG temporal
filter, segmentation, the bleaching model, the temporal median filter — only changes *which*
cells/clusters get a footprint, **not the noise floor of an already-extracted trace**. And the SNR
metric high-passes each trace *after* extraction, so low-frequency fixes move a band it discards.

### Debunked (verified NOT to raise HF-SNR)

| Proposal | Why it fails (evidence) |
|---|---|
| Fix the DOG temporal-filter no-op | The DOG/AP-detection path feeds only COM→DBSCAN clustering; it never enters `cell_traces = pinv(footprint)@(-movie)`. Symmetrizing the median window measured *slightly worse*; changing the threshold statistic risks false positives (effective threshold already ~3.9σ). |
| Ridge-regularize `pinv` | Footprint Gram is **well-conditioned (cond ≈ 15**, no tiny singular values), so `pinv` isn't amplifying noise. Small λ → ~0% gain; large λ → injects ~5.7% neighbor-spike crosstalk (false events). |
| SVD-denoised / perimeter-clipped footprints | Bias toward the region's dominant spatial mode; *increase* crosstalk between overlapping footprints. |
| NNLS unmixing | Half-wave-rectifies the residual → apparent noise drop is a rectification artifact with upward baseline bias. |
| Detrend / better bleaching / motion / segmentation | Target bands the metric discards (≈0 gain), or feed noisier footprints into the shared `pinv` and *raise* crosstalk on already-clean cells. |
| Post-hoc 60 Hz notch on final traces | Real but small (≈1–1.4% floor reduction at high Q); a wide notch clips broadband spikes (`snr_postproc_demo.py`). |

### The one survivor — whitened GLS trace extraction (implemented, opt-in)

Replace the unweighted pseudoinverse in `pinv_traces` with the minimum-variance (BLUE) estimator
under spatially heteroscedastic pixel noise:

```
cell_traces = (Fᵀ W F + λI)⁻¹ Fᵀ W (−movie),   W = diag(1 / σ_pix²)
```

`σ_pix` = per-pixel high-frequency noise std, from `1.4826·MAD` over the baseline (`std_ranges`)
frames of the moving-median-filtered movie. The current OLS `pinv` implicitly assumes equal
per-pixel noise and so over-weights bright-but-noisy pixels; GLS down-weights them.

- **Validated (synthetic, `tests/test_whiten.py`):** GLS==OLS to 2e-15 under uniform noise;
  ~38% lower residual noise under a 4× noise gradient with MAD-estimated σ working as well as
  true σ; spike amplitude unbiased (0.995). On real data expect a smaller gain (~single-digit to
  ~20%, set by the actual per-pixel noise heterogeneity).
- **Risk / guards:** on *overlapping* footprints whitening can raise neighbor crosstalk ~10–13%.
  Guards: percentile floor on σ_pix + relative ridge λ (`whiten_ridge`, default 1e-3), and
  `whiten_isolated_only=True` — only cells whose footprint overlaps no other are whitened;
  overlapping cells keep the faithful `pinv` row. Linear operator ⇒ cannot fabricate spikes.
- **Flag:** `Params.whiten_traces=False` by default ⇒ byte-for-byte the original pinv (all 41
  tests green, MATLAB fidelity intact). CLI: `run_pyali.py --whiten-traces [--whiten-all-cells]`.

### How to validate on your data (A/B)

```bash
PY="pyali/pyali/bin/python"
# baseline (== MATLAB-faithful) and whitened runs of the SAME movie:
$PY pyali/scripts/run_pyali.py /path/to/FOV --out /path/to/analysis_baseline
$PY pyali/scripts/run_pyali.py /path/to/FOV --out /path/to/analysis_whitened --whiten-traces
# compare (py = whitened, ml = baseline):
$PY scripts/snr_compare.py /path/to/analysis_whitened /path/to/analysis_baseline \
    --label-py gls --label-ml pinv --out snr_analysis/gls_vs_pinv
```

**Keep the change only if:** paired `noise_sigma` drops **and** `corr_hp ≈ 1` on isolated cells
(waveform unchanged apart from noise reduction). If `coh_hi`/`corr_hp` degrade on the overlapping
subset, keep `whiten_isolated_only=True` (the default). A full-FOV run needs the raw `frames1.bin`
and ~18 GB RAM and takes minutes.

---

## Files

- `scripts/snr_compare.py` — HF-SNR comparison of two `ALI_Result.mat` dirs (the Q1 tool).
- `scripts/snr_postproc_demo.py` — post-hoc trace improvements with a built-in distortion guard.
- `snr_analysis/snr_report_101034/` — the Q1 report (summary, CSV, figures).
- Pipeline changes: `pyali/pyali/params.py` (`whiten_*` fields), `pyali/pyali/extract.py`
  (`per_pixel_noise_map`, `whitened_gls_traces`, `footprint_isolated_mask`, `extract_cell_traces`),
  `pyali/pyali/pipeline.py` (wiring), `pyali/scripts/run_pyali.py` (`--whiten-traces`),
  `pyali/tests/test_whiten.py` (regression tests).
