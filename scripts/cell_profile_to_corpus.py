#!/usr/bin/env python3
"""Split a MATLAB ``cell_profile_<ID>.mat`` into pyali's per-FOV corpus layout.

The ``cell_profile_*.mat`` files (HDF5 v7.3, written by the MATLAB ALI pipeline) hold one
*perturbation's* worth of extracted traces, pooled over all of its fields of view::

    ALI_traces            (N_cells, T) single    raw fluorescence, one row per cell
    FOV                   (N_cells, 1) uint16    1-based FOV index of each cell
    COM                   (N_cells, 2) single    cell centre of mass
    reference_image_base  (H, W, n_FOV) uint16   one reference image per FOV

No plate / well / batch / timestamp metadata comes with them. This script re-shapes each file
into the layout every post-hoc SNR stage already consumes, so **none of those stages change**::

    <out_root>/<ID>/<ID>_fovNN/
        ALI_Result.mat        cell_traces (N, T), footprint_center (N, 2)
        ALI_Int_Result.mat    reference_image (H, W)
        fov_metadata.json     day = plate = well = <ID>; burst = FOV index; analysis.fps

with the whole perturbation standing in for one day / plate / well and the FOV index playing the
part of ``burst`` (acquisition order). ``dir_name`` carries the ID because
``snr_aggregate.fov_summary`` groups on ``dir_name`` alone -- a bare ``fov01`` shared by two
perturbations would merge their cells.

**The traces are negated.** These are raw-fluorescence traces of a negative-going indicator: the
action potential is a *dip*. Checked on JZ043 (600 cells): 3-sigma troughs outnumber 3-sigma peaks
30839 to 8277, the trough-triggered average is a clean -6.9-sigma waveform with a rising ramp and
decay, and the peak-triggered one is a 1-sample +2.7-sigma noise crossing. pyali's own
``cell_traces`` are already sign-flipped at extraction (``pinv_traces`` returns
``-project_movie(...)``, see ``pyali/extract.py``), and ``pyali.metrics.per_cell_snr`` detects
*positive* peaks. Negating here puts the two conventions in the same frame, so the metrics are
applied to these traces exactly as they are to the corpus. ``--no-invert`` keeps the raw sign.

Frame rate and bit depth are not in the file and must be given; both are recorded in
``fov_metadata.json`` as supplied values.

    python scripts/cell_profile_to_corpus.py cell_profile_JZ043.mat cell_profile_JZ045.mat \\
        --out-root /path/to/pyali_analysis --fps 800 --bit 8-bit

Then the unchanged stages, pointing ``--read-root`` at ``out_root`` and ``--s3-prefix`` at
wherever it was synced (the stages discover FOVs by listing S3, so sync first)::

    aws s3 sync OUT_ROOT S3_PREFIX
    python scripts/snr_corpus.py    --s3-prefix S3_PREFIX --read-root OUT_ROOT --manifest M1
    python scripts/snr_sta.py       --s3-prefix S3_PREFIX --read-root OUT_ROOT --manifest M2
    aws s3 sync OUT_ROOT S3_PREFIX
    python scripts/snr_aggregate.py --s3-prefix S3_PREFIX --read-root OUT_ROOT \\
        --keep-csv /nonexistent --out-dir OUT_ROOT/snr_summary
    python scripts/snr_figures.py      --summary-dir OUT_ROOT/snr_summary
    python scripts/snr_sta_figures.py  --s3-prefix S3_PREFIX --read-root OUT_ROOT --summary-dir OUT_ROOT/snr_summary
    python scripts/snr_sta_variance.py --s3-prefix S3_PREFIX --read-root OUT_ROOT --summary-dir OUT_ROOT/snr_summary
    python scripts/representative_traces.py --summary-dir OUT_ROOT/snr_summary --read-root OUT_ROOT \\
        --out-dir OUT_ROOT/representative_traces
    aws s3 sync OUT_ROOT S3_PREFIX
"""
import argparse
import json
import os
import re
import sys
import time

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from pyali.io import save_mat_v73                                        # noqa: E402
from pyali.metadata import write_metadata                                # noqa: E402

_ID = re.compile(r"cell_profile_([A-Za-z0-9-]+)\.mat$")


def perturbation_id(path):
    m = _ID.search(os.path.basename(path))
    if not m:
        raise ValueError(f"cannot read a perturbation ID from {os.path.basename(path)!r}; "
                         f"expected cell_profile_<ID>.mat")
    return m.group(1)


def convert(path, out_root, fps, bit, invert=True, source_uri=None):
    """Write one perturbation's FOV directories under ``out_root``. Returns a summary dict."""
    pid = perturbation_id(path)
    digits = re.sub(r"\D", "", pid)
    plate_num = int(digits) if digits else None
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        fov = np.asarray(f["FOV"]).ravel().astype(int)          # 1-based, one per cell
        com = np.asarray(f["COM"]).T.astype(np.float32)          # (N, 2) as stored by MATLAB
        ref = f["reference_image_base"]                          # HDF5 (n_fov, W, H)
        n_fov = ref.shape[0]
        # One chunk per cell in the file, so the whole array reads in a few seconds; the largest
        # file (22916 cells) is ~770 MB as float32, well within RAM.
        traces = np.asarray(f["ALI_traces"]).T                   # (N, T) float32
        N, T = traces.shape
        if fov.size != N or com.shape[0] != N:
            raise ValueError(f"{pid}: FOV/COM/ALI_traces cell counts disagree "
                             f"({fov.size}, {com.shape[0]}, {N})")
        if invert:
            traces = -traces                                     # dips -> positive spikes
        H, W = ref.shape[2], ref.shape[1]
        rec = dict(perturbation=pid, source=os.path.basename(path), n_cells=int(N),
                   n_frames=int(T), n_fov=int(n_fov), fovs=[])
        for k in range(1, n_fov + 1):
            sel = np.flatnonzero(fov == k)
            dir_name = f"{pid}_fov{k:02d}"
            out_dir = os.path.join(out_root, pid, dir_name)
            os.makedirs(out_dir, exist_ok=True)
            ct = np.ascontiguousarray(traces[sel]) if sel.size else np.zeros((0, T), np.float32)
            save_mat_v73(os.path.join(out_dir, "ALI_Result.mat"),
                         cell_traces=ct, footprint_center=com[sel])
            # HDF5 stores MATLAB's (H, W, n) as (n, W, H); transpose recovers the image as MATLAB
            # would show it (row, col).
            save_mat_v73(os.path.join(out_dir, "ALI_Int_Result.mat"),
                         reference_image=np.asarray(ref[k - 1]).T)
            meta = dict(
                dir_name=dir_name, day=pid, timestamp=None, time_of_day=None,
                plate=pid, plate_num=plate_num, well=pid,
                well_source="perturbation ID stands in for plate/well/batch (no such metadata "
                            "was supplied with the file)",
                burst=int(k), div=None, batch_id=pid, cell_line=None, dose=None, genotype=None,
                perturbation=pid, bit=bit, dims=f"{H}x{W}", n_frames=int(T),
                fov_path=f"{source_uri or path}#FOV={k}",
                source_file=os.path.basename(path), source_variable="ALI_traces",
                trace_sign_inverted=bool(invert),
                trace_note=("cell_traces = -ALI_traces: the source is raw fluorescence of a "
                            "negative-going indicator, negated to match pyali's positive-spike "
                            "convention" if invert else "cell_traces = ALI_traces (raw sign)"),
                n_cells=int(sel.size), n_frames_analyzed=int(T),
                n_regions_kept=None, n_regions_dropped=None,
                analysis=dict(profile="cell_profile_mat", fps=float(fps), nrow=int(H),
                              ncol=int(W), read_dtype="uint8" if bit.startswith("8") else None,
                              compute_dtype="float32",
                              fps_source="user-supplied", bit_source="user-supplied",
                              extraction="MATLAB ALI pipeline (traces imported, not re-extracted)"),
            )
            write_metadata(out_dir, meta)
            rec["fovs"].append(dict(dir_name=dir_name, fov=int(k), n_cells=int(sel.size)))
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mat", nargs="+", help="cell_profile_<ID>.mat files")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--fps", type=float, required=True, help="frame rate in Hz (not in the file)")
    ap.add_argument("--bit", default="8-bit", help="bit depth label for the legends (not in the file)")
    ap.add_argument("--no-invert", action="store_true", help="keep the raw trace sign")
    ap.add_argument("--source-uri", default=None,
                    help="where the .mat files live (e.g. an s3:// prefix), recorded in metadata")
    a = ap.parse_args()

    os.makedirs(a.out_root, exist_ok=True)
    summary = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   script=os.path.basename(__file__), fps=a.fps, bit=a.bit,
                   trace_sign_inverted=not a.no_invert, perturbations=[])
    for p in a.mat:
        uri = f"{a.source_uri.rstrip('/')}/{os.path.basename(p)}" if a.source_uri else None
        rec = convert(p, a.out_root, a.fps, a.bit, invert=not a.no_invert, source_uri=uri)
        summary["perturbations"].append(rec)
        print(f"[convert] {rec['perturbation']:<8} {rec['n_fov']:3} FOVs  {rec['n_cells']:6} cells  "
              f"{rec['n_frames']} frames  {rec['seconds']}s", flush=True)
    with open(os.path.join(a.out_root, "conversion_manifest.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[convert] {a.out_root}/conversion_manifest.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
