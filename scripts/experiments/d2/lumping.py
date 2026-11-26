#!/usr/bin/env python3
"""Neighbour-pair crosstalk ("lumping") of one or more result directories.

    python lumping.py <results_dir> [<results_dir> ...] [--labels before,after] [--radius 8] --out report.json

For every pair of cells whose centres are within ``--radius`` px: the fraction of shared trace
spikes (20 Hz high-pass, peaks above 4 robust sigma, matched within +/-1 frame, over the smaller
spike count) and the correlation of the high-passed traces. Lumped cells -- one trace carrying a
neighbour's spikes -- show up as pairs with a high shared fraction; a clean separation keeps both
low. Also reports the spike-amplitude SNR distribution (cells with snr_median >= 4 / 5 / 6).
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

from pyali.metrics import highpass, per_cell_snr, robust_sigma   # noqa: E402
from pyali.results import load_results                            # noqa: E402


def trace_spikes(x, fps, k=4.0):
    from scipy.signal import find_peaks
    h = highpass(np.asarray(x, float), 20.0, fps)
    pk, _ = find_peaks(h, height=k * robust_sigma(h), distance=max(1, int(0.01 * fps)))
    return pk, h


def lumping(results_dir, radius=8.0):
    from scipy.spatial import cKDTree
    R = load_results(results_dir)
    fps = float(R.attrs["fps"])
    spiking = (R.cells["source"] == "spiking").to_numpy()
    idx = np.flatnonzero(spiking)
    pk, hp = {}, {}
    for k in idx:
        pk[k], hp[k] = trace_spikes(R.traces[k], fps)
    pairs = cKDTree(R.centers[idx]).query_pairs(r=radius, output_type="ndarray")
    shared, corr = [], []
    for a, b in pairs:
        i, j = idx[a], idx[b]
        si, sj = pk[i], pk[j]
        if len(si) and len(sj):
            d = np.abs(si[:, None] - sj[None, :]).min(axis=1)
            shared.append(float((d <= 1).sum()) / min(len(si), len(sj)))
        else:
            shared.append(0.0)
        corr.append(float(np.corrcoef(hp[i], hp[j])[0, 1]))
    shared, corr = np.array(shared), np.array(corr)
    snr = per_cell_snr(R.traces[idx], fps)["snr_median"]
    s = snr[np.isfinite(snr)]
    return dict(results=results_dir, n_cells=int(len(R.cells)), n_spiking=int(spiking.sum()),
                n_pairs=int(len(pairs)),
                shared_median=float(np.median(shared)) if len(shared) else None,
                pairs_shared_gt_0p5=int((shared > 0.5).sum()),
                pairs_shared_gt_0p8=int((shared > 0.8).sum()),
                corr_median=float(np.median(corr)) if len(corr) else None,
                pairs_corr_gt_0p5=int((corr > 0.5).sum()),
                snr_median_of_cells=float(np.median(s)) if len(s) else None,
                cells_snr_ge_4=int((s >= 4).sum()), cells_snr_ge_5=int((s >= 5).sum()),
                cells_snr_ge_6=int((s >= 6).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    labels = a.labels.split(",") if a.labels else [os.path.basename(r.rstrip("/")) for r in a.results]
    rep = {lab: lumping(r, a.radius) for lab, r in zip(labels, a.results)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2)
    keys = ["n_cells", "n_spiking", "n_pairs", "shared_median", "pairs_shared_gt_0p5", "pairs_shared_gt_0p8",
            "corr_median", "pairs_corr_gt_0p5", "snr_median_of_cells", "cells_snr_ge_4", "cells_snr_ge_5",
            "cells_snr_ge_6"]
    print("| metric | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels))
    for k in keys:
        vals = [rep[lab][k] for lab in labels]
        print(f"| {k} | " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
