# Code-review rubric for pyali

How a module is reviewed for bugs, what counts as a finding, and how a finding is verified.
Calibrated on the pilot review of `pyali/extract.py` (`docs/review/extract_pilot.md`).

## What we are looking for

pyali is a Python port of a MATLAB voltage-extraction pipeline, run unattended over thousands of
fields of view. Its own fixed-bug history (a silently non-functional parameter, a float32 movie
promoted to float64 in one piece, a shared write helper overwriting another stage's file, a PSD
estimator mismatch, three RAM amplifiers) contains no crashes. The bugs that matter here produce
a plausible wrong number quietly. The categories below are weighted accordingly.

| # | category | definition | example in pyali |
|---|---|---|---|
| 1 | **Index and convention crossings** | 0- vs 1-indexing, `[row, col]` vs `[col, row]`, inclusive vs half-open bounds, C- vs column-major flattening, converted at a boundary wrongly or not at all | region `PixelList` is `[col, row]` 1-indexed; `compute_patch` returns 1-indexed inclusive ranges; `ravel(order="F")` picks the first maximum column-major |
| 2 | **MATLAB parity and numerics** | ddof, rounding mode, pseudoinverse tolerance, even-window medians, padding modes, epsilon guards, integer arithmetic before the cast, dtype promotion, chunk-order dependence | `np.nanstd(..., ddof=1)`; `round_half_away_from_zero`; `pinv(rcond=max(shape)*eps)`; `project_movie` chunking |
| 3 | **Degenerate inputs** | zero regions, zero cells, one spike, spikes at the movie ends, patches clipped at the border, movies shorter than the protocol ranges, empty arrays into `max`/`argmax`/`svd` | `np.zeros((H, W, 0))` into `pinv_traces`; `patch_movie[t-3:t+4]` at the ends |
| 4 | **Parameter-coupled invariants** | code that is correct only while two tunables keep a relation nothing asserts | `region_grow_brightest` needs ≥ `com_n_pixels` pixels inside `com_radius` of the maximum |
| 5 | **Shared mutable state and aliasing** | in-place mutation of arguments, mutable default arguments, views returned where copies are expected, dict schemas shared across modules | `process_fov(..., p=Params())` evaluated once; movies modified in place |
| 6 | **Silent degradation** | `getattr(obj, "name", default)` where a misspelt name selects the default, broad `except`, fallbacks that change results without a message | `getattr(p, "whiten_traces", False)` |
| 7 | **Cross-module schema contracts** | keys, column names, index alignment and units one module writes and another assumes, documented nowhere but in the consumer | `regions[c]` must align with `spatial_footprints[c]`; `fov_metadata.json → analysis.fps` |
| 8 | **Memory amplification** | transient copies or masks the size of the movie, dtype promotion, per-region copies that scale with region count | `movie[movie > thr] = 0` builds a full-size mask |
| 9 | **Doc and code drift** | a docstring states a shape, order or semantics the code does not implement | shapes in `Returns` blocks; "bit-identical" claims |

Not in scope: security (inputs are trusted local files) and performance (the `bench/` suite owns it).

## Severity

| level | meaning |
|---|---|
| **S1** | silent wrong result on valid input: the pipeline completes and the numbers are wrong |
| **S2** | crash or exception on valid input |
| **S3** | fragile: correct today by coincidence (an unasserted invariant, a guard that depends on another guard) |
| **S4** | maintainability or documentation only |

## Evidence standard

A finding names a **concrete input** and the **wrong output or exception** it produces. "This
looks off" is not a finding. A reviewer may run probes but must not edit repository files.

Finding schema (one JSON object per finding):

```json
{"id": "EXT-01", "file": "pyali/extract.py", "line": 123, "function": "detect_region_aps",
 "category": 3, "severity": "S2", "claim": "one sentence",
 "failure_scenario": "input -> observed wrong output / exception",
 "reproducer_sketch": "python snippet or steps", "confidence": "high|medium|low"}
```

Reviewers also return a **non-findings** list: `{"function": "...", "checked": "what was examined
and found sound"}` for every function, so the next review does not repeat the work.

## Verification

A second agent receives each finding's `claim` and `failure_scenario` only, never the reviewer's
reasoning, plus the code and the oracles. It writes a minimal reproducer, runs it, and returns:

| verdict | meaning |
|---|---|
| **CONFIRMED** | the reproducer shows the wrong output or exception |
| **PLAUSIBLE** | the reasoning holds but no trigger was found inside the documented input domain |
| **REJECTED** | the claim is false, with the guard or fact that makes it so |

Each verdict carries the command run and an excerpt of its output. The verifier's job is to try
to reject.

## Oracles

* `tests/synthetic.py`: a generated movie with planted cells, spike times, footprints and traces,
  under the production defaults; `tests/test_extract_oracle.py` states the invariants.
* A real FOV, or a spatial crop of one, pointed at by `PYALI_REAL_FOV`, for behaviour the
  synthetic movie cannot show (acquisition artefacts, real region counts, timing and memory).
* The repository's own claims: patch-local filtering is bit-identical to global filtering; the
  float32 path reproduces float64 to tolerance; results are independent of worker count.
