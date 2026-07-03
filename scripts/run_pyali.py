#!/usr/bin/env python3
"""Run the pyali waveform-extraction pipeline on ONE field-of-view directory.

The FOV directory must contain the raw movie ``frames1.bin`` (plus the usual sidecars
``output_data.mat``, ``frames1_dropped_frames.txt``, ``frames1_ROI_mean_stdev.txt``).
Frame dimensions (nrow, ncol) are auto-detected from the sidecar + pixel correlation unless
you pass ``--nrow``/``--ncol``.

Outputs are written to an ``analysis`` folder inside the FOV directory (or ``--out``):
    ALI_Int_Result.mat, ALI_Result.mat
and, with ``--figures``, the presentation result figures:
    detected_regions.png, coms.png, cell_traces.png, center_of_cell_regions.png,
    and cell_traces.html (interactive; zoom/pan and click a legend entry to isolate a trace)

Usage:
    python scripts/run_pyali.py /path/to/fov_dir --figures
    python scripts/run_pyali.py /path/to/fov_dir --figures --out /path/to/analysis
    python scripts/run_pyali.py /path/to/fov_dir --nrow 312 --ncol 1200
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)                        # the pyali repo root (contains the package)
sys.path.insert(0, _REPO)                             # so `import pyali` works
sys.path.insert(0, _HERE)                             # so `import find_video_dims` works

import find_video_dims as fvd                          # noqa: E402
from pyali.params import Params                        # noqa: E402
from pyali.pipeline import process_fov                 # noqa: E402


def _clamp_ranges(ranges, T):
    """Clamp 1-indexed inclusive frame ranges to [1, T]; drop empties. Returns (ranges, changed)."""
    out, changed = [], False
    for a, b in ranges:
        na, nb = max(1, a), min(b, T)
        if na > nb:
            changed = True
            continue
        if (na, nb) != (a, b):
            changed = True
        out.append((na, nb))
    return out, changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fov_dir", help="directory containing frames1.bin (+ sidecars)")
    ap.add_argument("--figures", action="store_true", help="also save the 4 result figures")
    ap.add_argument("--out", default=None, help="output dir (default: <fov_dir>/analysis)")
    ap.add_argument("--nrow", type=int, default=None, help="force frame height (else auto-detect)")
    ap.add_argument("--ncol", type=int, default=None, help="force frame width (else auto-detect)")
    a = ap.parse_args(argv)

    bin_path = os.path.join(a.fov_dir, "frames1.bin")
    if not os.path.isfile(bin_path):
        ap.error(f"no frames1.bin found in {a.fov_dir}")

    # ---- frame dimensions: auto-detect unless given ----
    if a.nrow and a.ncol:
        nrow, ncol = a.nrow, a.ncol
        print(f"[pyali] using given dimensions: nrow={nrow}, ncol={ncol}")
    else:
        ppf, how = fvd.detect_pixels_per_frame(bin_path, itemsize=2, dim_lo=100, dim_hi=2100)
        if ppf is None:
            ap.error("could not auto-detect frame size; pass --nrow/--ncol")
        _score, nrow, ncol = fvd.recover_dims(bin_path, ppf, dtype="u2")[0]
        print(f"[pyali] auto-detected dimensions: nrow={nrow}, ncol={ncol}   [{how}]")

    p = Params(nrow=nrow, ncol=ncol)

    # ---- clamp protocol frame-ranges to this video's length ----
    nframes = os.path.getsize(bin_path) // (nrow * ncol * 2)
    T = nframes - p.truncate_last
    p.bkg_ranges, c1 = _clamp_ranges(p.bkg_ranges, T)
    p.std_ranges, c2 = _clamp_ranges(p.std_ranges, T)
    if c1 or c2:
        print(f"[pyali] WARNING: background/std frame ranges are protocol-specific and were "
              f"clamped to this video's {T} frames. Edit Params.bkg_ranges/std_ranges if your "
              f"acquisition protocol differs.")

    out_dir = a.out or os.path.join(a.fov_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[pyali] frames={T}  ->  outputs in {out_dir}\n")

    process_fov(a.fov_dir, out_dir=out_dir, p=p, save=True, verbose=True, make_figures=a.figures)

    print(f"\n[pyali] wrote ALI_Int_Result.mat, ALI_Result.mat" +
          (" + result figures (detected_regions/coms/cell_traces/center_of_cell_regions .png "
           "and interactive cell_traces.html)" if a.figures else "") + f" to:\n  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
