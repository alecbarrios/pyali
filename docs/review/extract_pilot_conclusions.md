## Calibration conclusions

Written by hand after reading the findings, the verdicts and the oracle runs; everything above
this section is generated from the JSON files.

### What the pilot measured

* **Precision of a rubric-guided reviewer is high when the evidence standard is enforced.** All
  ten findings reproduced exactly; the verifier rejected none. Requiring a concrete failing input
  per finding, with an oracle available for probes, filtered the speculation out before it was
  reported. The reviewer's own non-findings list covers all 22 functions.
* **The verifier's value was triage, not rejection.** It moved two findings down a severity
  (EXT-03 S1→S3, EXT-05 S2→S3) and sorted all ten by reachability: four are reachable from
  `process_fov` with production parameters, six need a non-default tunable or an input the
  pipeline never produces. That ordering decides what gets fixed first, and a reviewer cannot be
  trusted to make it about its own findings.
* **Production exposure is narrower than the raw count.** Of the four pipeline-reachable
  findings, one is a crash on a short movie through the library API (EXT-04; the CLI clamps
  the ranges, the corpus runner and the API do not), one is a silent-wrong-result whose fix
  depends on the MATLAB original (EXT-02), one fired zero times on 459 real spikes (EXT-01), and
  one is a memory ceiling that matters only at 1080x1080 (EXT-08). The honest headline is: one
  bug to fix now, one decision that needs the `.m` source, two robustness items.

### Categories: which earned their place on this file

| category | reported → confirmed | reachable in production | verdict for the full review |
|---|---|---|---|
| 3 degenerate inputs | 3 → 3 | 2 | keep; the most productive category, and the oracle's edge cases fed it directly |
| 4 parameter-coupled invariants | 2 → 2 | 0 | keep as the robustness tier; findings are real but need a non-default parameter, so they are assertions to add, not bugs to fix |
| 8 memory amplification | 2 → 2 | 1 | keep, and require a measurement (the verifier's 3.00x and 3.03x figures are what make these actionable) |
| 2 MATLAB parity and numerics | 2 → 2 | 1 | keep only if the MATLAB source is available to the reviewer; without it, parity claims cannot be closed and collapse into category 9 |
| 9 doc and code drift | 1 → 1 | 0 | keep; cheap and it found a real swap hidden by a square default |
| 1 index and convention crossings | 0 | — | not falsified: thirteen column-major sites and eighteen index conversions were all checked and found sound. A single-file reviewer cannot see a boundary bug, so this stays with the cross-module lens |
| 5 shared mutable state | 0 | — | the known instance (`Params()` evaluated once as a default argument) lives in `pipeline.py`, out of scope here; keep for group C |
| 6 silent degradation | 0 | — | the mechanical pass already proved every `getattr` name valid and found no broad `except`; move this category to tier 0 and stop spending agent attention on it |
| 7 cross-module schema contracts | 0 | — | cross-module by definition; lens only |

Two leads surfaced outside the file under review and go to the next wave rather than being lost:
a noise-free blank movie with only a gradient and bleaching segments into four border regions
(`tests/test_extract_oracle.py::test_drifting_blank_movie_has_no_regions`, xfail), and the dark
real crop outside the illuminated well yields 152 regions, more than the cell field's 86. Both
belong to preprocess and segmentation, and the second also bears on the corpus QC's
cells-per-region metric. The reviewer's notes add three MATLAB-parity questions and two
`utils`/`preprocess` items.

### Budget: what one review costs and how far it stretches

| stage | agents | wall | tokens | output |
|---|---|---|---|---|
| mechanical | 0 | 1 min | 0 | 4 style hits, 0 unknown names, the column-major and index-conversion map |
| reviewer | 1 | 20 min | ~180k | 10 findings, 22 non-findings, 22 probes |
| verifier | 1 | 17 min | ~160k | 10 reproducers, 3 instrumented real runs, 2 severity changes, reachability for all |

* **646 lines of dense numeric code is within one reviewer's reach** with full coverage and
  probes. Groups of roughly 600 to 900 lines are the right unit. Group D in the preliminary
  outline (`metrics` plus seven SNR scripts, over 3000 lines) is too large and splits into
  D1 (`metrics`, `snr_corpus`, `snr_aggregate`) and D2 (the STA and figure scripts).
* **The oracle, not the agents, was the expensive part.** Three tuning iterations were needed
  before the synthetic movie passed under production defaults, because the production
  segmentation admits speckle on anything realistic. Each group needs its own oracle budgeted
  explicitly: B can reuse the synthetic movie at the segmentation level (the border-regions lead
  is already a test), C needs file round-trip oracles (`.mat` v7.3 save/load, metadata parsing
  against the 6504-name corpus), D needs trace-level oracles (`per_cell_snr` on traces of known
  SNR).
* **Real data runs as 256x256 crops.** This machine has 30 GB, so a full 800x800 field of view
  (about 43 GB at float32) cannot run here; a crop with every frame kept runs in 45 to 65 s at
  4.7 GB and reproduced identically under instrumentation. The cell-field and dark crops of
  `133603_P-4_W-C2_443-2_443screen2_DIV35__burst47` are the standing real oracle.
* **Verifier mandate changes.** "Try to reject" produced no rejections; what it produced was
  reachability and severity. The full-review verifier prompt should ask for exactly that, plus
  a measurement for every memory or performance claim.

### Revised outline for the full review (to be detailed in the next planning cycle)

| wave | agents | scope |
|---|---|---|
| 1 | 5 | B `preprocess`+`segmentation`+`utils` (reviewer + verifier); C `io`+`_h5read`+`params`+`pipeline`+`metadata` (reviewer + verifier); one convention lens across the A–B–C boundaries, whose findings go to the matching group's verifier |
| 2 | 6 | D1 and D2 (reviewer each, one shared verifier); E CLI and corpus runners (reviewer + verifier); the degenerate-input and memory lens is dropped, since both categories were productive inside a single-file review |
| 3 | 2 | `pyali/registration` (reviewer + verifier), once a registration oracle exists |
| fixes | — | after each wave, not inside it: every confirmed finding gets a regression test in `tests/` with the fix |

The MATLAB-parity lens runs only if the `.m` source is obtained first. Expected cost at the
pilot's rates: about 340k tokens and 37 agent-minutes per group (reviewer plus verifier), so
roughly 2.5M tokens and four to five hours of agent time for the seven groups and two lenses,
before fixes.

### Fix order proposed for `extract.py` (not done in the pilot)

1. EXT-04: clamp or reject out-of-range baseline frames inside the library path, not only in the CLI.
2. EXT-02: settle against the MATLAB source, then either fix the code or the docstring.
3. EXT-10: swap `patch_size` to `(rows, cols)` as documented; add a non-square test.
4. EXT-01: collapse identical `[row, col, frame]` rows after refinement.
5. EXT-06 and EXT-07: assert the invariants on `com_radius`/`com_n_pixels` and `filter_window` at `Params` construction.
6. EXT-08: replace the dense pseudoinverse with a least-squares solve or a chunked product; EXT-09: in-place MAD.
7. EXT-05: explicit index bookkeeping instead of probe pixels; EXT-03: `np.ascontiguousarray` guard in the filters.
