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


# Acquisition profiles, keyed by name. ``auto`` resolves via _AUTO below.
_PROFILES = {
    "6GP002": Params.profile_6GP002,             # 312x1200 16-bit  (20260331 dir1/dir2, 20260401)
    "6GP002-8bit": Params.profile_6GP002_8bit,   # 1080x1080 8-bit  (20260401 *_8bit_*)
    "443screen1": Params.profile_443screen1,     # 1000x1000 8-bit  (20260611/12; dirs say 443GP)
    "443screen2": Params.profile_443screen2,     # 800x800 8-bit    (20260715-18)
}

# (nrow, ncol, read_dtype) -> profile name
_AUTO = {
    (312, 1200, "uint16"): "6GP002",
    (1080, 1080, "uint8"): "6GP002-8bit",
    (1000, 1000, "uint8"): "443screen1",
    (800, 800, "uint8"): "443screen2",
}


def detect_bit_depth(bin_path, n_bytes=1 << 22):
    """Return ``"uint8"`` or ``"uint16"`` from a byte-plane adjacency test.

    Under a uint16 little-endian hypothesis the even bytes are the sample *low* halves — close to
    uniform noise carrying no image structure — while the odd bytes are the high halves and hold a
    /256 copy of the image. In genuine uint8 data both planes are just alternating pixels of the
    same image, so both stay spatially smooth. The discriminator is each plane's lag-1
    correlation: uint16 gives ~0.02 (even) vs ~0.48 (odd); uint8 gives ~0.99 for both.

    Reads from the middle of the file to avoid any header/leading blank frames.
    """
    import numpy as np
    size = os.path.getsize(bin_path)
    off = (size // 2) & ~1                                   # even offset keeps plane parity
    with open(bin_path, "rb") as f:
        f.seek(off)
        buf = np.frombuffer(f.read(min(n_bytes, size - off)), dtype=np.uint8)
    buf = buf[:(buf.size // 2) * 2]
    if buf.size < 1024:
        return "uint16"                                      # too small to judge; keep the default
    even = buf[0::2].astype(np.float64)
    odd = buf[1::2].astype(np.float64)

    def lag1(x):
        a, b = x[:-1], x[1:]
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            return 0.0
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

    ce, co = lag1(even), lag1(odd)
    depth = "uint8" if ce > 0.5 else "uint16"
    print(f"[pyali] bit depth: {depth}  (byte-plane lag-1 corr: even={ce:.3f}, odd={co:.3f})")
    return depth


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
    ap.add_argument("--whiten-traces", action="store_true",
                    help="use whitened GLS trace extraction (noise-weighted, opt-in) instead of "
                         "the baseline unweighted pinv; see snr_analysis/ to A/B it")
    ap.add_argument("--whiten-all-cells", action="store_true",
                    help="with --whiten-traces, whiten overlapping cells too (default: only "
                         "isolated footprints are whitened; overlapping ones keep the faithful pinv)")
    ap.add_argument("--read-dtype", choices=["uint16", "uint8"], default=None,
                    help="on-disk sample dtype (default: auto-detect by byte-plane test)")
    ap.add_argument("--compute-dtype", choices=["float64", "float32"], default=None,
                    help="in-RAM movie dtype (default: the profile's; float32 ~halves peak RAM)")
    ap.add_argument("--profile", choices=["auto"] + sorted(_PROFILES), default="auto",
                    help="acquisition profile: sets frame size, dtypes, and the background/"
                         "baseline frame ranges for that batch. 'auto' picks it from the "
                         "detected (nrow, ncol, bit depth)")
    ap.add_argument("--keep-csv", default=None,
                    help="keep.csv to join per-FOV metadata against (authoritative plate/well)")
    ap.add_argument("--day", default=None,
                    help="acquisition day for the metadata sidecar (default: parent dir name)")
    ap.add_argument("--max-region-bbox-frac", type=float, default=None,
                    help="reject segmentation regions whose bounding box exceeds this fraction "
                         "of the frame (default 0.01; pass 0 to disable the guard)")
    a = ap.parse_args(argv)

    bin_path = os.path.join(a.fov_dir, "frames1.bin")
    if not os.path.isfile(bin_path):
        ap.error(f"no frames1.bin found in {a.fov_dir}")

    # ---- bit depth: given, or from the profile, or auto-detected ----
    if a.read_dtype:
        read_dtype = a.read_dtype
    elif a.profile != "auto":
        read_dtype = _PROFILES[a.profile]().read_dtype
    else:
        read_dtype = detect_bit_depth(bin_path)
    itemsize = 1 if read_dtype == "uint8" else 2
    raw_code = "u1" if read_dtype == "uint8" else "u2"

    # ---- frame dimensions: given, or from the profile, or auto-detected ----
    if a.nrow and a.ncol:
        nrow, ncol = a.nrow, a.ncol
        print(f"[pyali] using given dimensions: nrow={nrow}, ncol={ncol}")
    elif a.profile != "auto":
        base = _PROFILES[a.profile]()
        nrow, ncol = base.nrow, base.ncol
        print(f"[pyali] dimensions from profile {a.profile}: nrow={nrow}, ncol={ncol}")
    else:
        ppf, how = fvd.detect_pixels_per_frame(bin_path, itemsize=itemsize, dim_lo=100, dim_hi=2100)
        if ppf is None:
            ap.error("could not auto-detect frame size; pass --nrow/--ncol or --profile")
        _score, nrow, ncol = fvd.recover_dims(bin_path, ppf, dtype=raw_code)[0]
        print(f"[pyali] auto-detected dimensions: nrow={nrow}, ncol={ncol}   [{how}]")

    # ---- profile: the background/baseline ranges are acquisition-specific, so this matters ----
    name = a.profile
    if name == "auto":
        name = _AUTO.get((nrow, ncol, read_dtype))
        if name is None:
            ap.error(f"no profile for {nrow}x{ncol} {read_dtype}; known shapes are "
                     f"{sorted(_AUTO)}. Pass --profile explicitly.")
    overrides = dict(nrow=nrow, ncol=ncol, read_dtype=read_dtype)
    if a.compute_dtype:
        overrides["compute_dtype"] = a.compute_dtype
    p = _PROFILES[name](**overrides)
    if a.max_region_bbox_frac is not None:
        p.max_region_bbox_frac = a.max_region_bbox_frac or None
    print(f"[pyali] profile {name}: {p.nrow}x{p.ncol} {p.read_dtype} -> {p.compute_dtype}, "
          f"bkg/std ranges {p.bkg_ranges}, region bbox guard "
          f"{'off' if not p.max_region_bbox_frac else f'{100*p.max_region_bbox_frac:g}%'}")

    if a.whiten_traces:
        p.whiten_traces = True
        p.whiten_isolated_only = not a.whiten_all_cells
        print(f"[pyali] whitened GLS trace extraction ON "
              f"(isolated_only={p.whiten_isolated_only})")

    # ---- clamp protocol frame-ranges to this video's length ----
    nframes = os.path.getsize(bin_path) // (nrow * ncol * itemsize)
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

    keep_row = None
    if a.keep_csv:
        from pyali.metadata import load_keep_index
        dir_name = os.path.basename(os.path.normpath(a.fov_dir))
        idx = load_keep_index(a.keep_csv)
        day = a.day or os.path.basename(os.path.dirname(os.path.normpath(a.fov_dir)))
        keep_row = idx.get((day, dir_name))
        if keep_row is None:                       # nested trees e.g. 20260401/Data_B1/<fov>
            hits = [v for (d, dn), v in idx.items() if dn == dir_name]
            keep_row = hits[0] if len(hits) == 1 else None
        if keep_row is None:
            print(f"[pyali] WARNING: no keep.csv row for {dir_name}; metadata will come from "
                  f"the directory name alone")

    process_fov(a.fov_dir, out_dir=out_dir, p=p, save=True, verbose=True, make_figures=a.figures,
                keep_row=keep_row, profile=name, day=a.day)

    print(f"\n[pyali] wrote ALI_Int_Result.mat, ALI_Result.mat" +
          (" + result figures (detected_regions/coms/cell_traces/center_of_cell_regions .png "
           "and interactive cell_traces.html)" if a.figures else "") + f" to:\n  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
