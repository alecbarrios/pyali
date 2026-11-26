#!/usr/bin/env python3
"""Field-level comparison of a pyali result with the stored Octave (Mini ALI V2.2) result.

    python oracle_compare.py <pyali_results_dir> <octave_fov_dir> --out report.json

Spike counts before / after the same-frame merge and in clusters, cluster and cell counts, and
how many cell centres coincide (pyali spiking cells within 1 / 2 px of an Octave footprint centre
and the reverse).
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _ROOT)

from pyali.results import load_results                            # noqa: E402


def main(argv=None):
    import scipy.io as sio
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pyali_dir")
    ap.add_argument("octave_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = load_results(a.pyali_dir)
    meta = json.load(open(os.path.join(a.pyali_dir, "fov_metadata.json")))
    m = sio.loadmat(os.path.join(a.octave_dir, "octave_extras.mat"), squeeze_me=True, struct_as_record=False)
    rs = m["run_summary"]
    oc = np.asarray(m["footprint_center"], float).reshape(-1, 2)
    spk = (R.cells["source"] == "spiking").to_numpy()
    pc = R.centers[spk]
    d_po = cKDTree(oc).query(pc)[0] if len(oc) and len(pc) else np.zeros(0)
    d_op = cKDTree(pc).query(oc)[0] if len(oc) and len(pc) else np.zeros(0)
    rep = dict(
        pyali=dict(n_spikes_before_merge=meta.get("n_spikes_before_merge"),
                   n_spikes_after_merge=meta.get("n_spikes_after_merge"),
                   n_spikes_clustered=meta.get("n_spikes_clustered"), n_clusters=meta.get("n_clusters"),
                   n_cells=int(len(R.cells)), n_cells_by_source=meta.get("n_cells_by_source"),
                   n_regions=meta.get("n_regions_kept")),
        octave=dict(n_spikes_before_merge=float(rs.n_spikes_before_merge),
                    n_spikes_after_merge=float(rs.n_spikes_after_merge),
                    n_spikes_clustered=float(rs.n_spikes_clustered), n_clusters=float(rs.n_clusters),
                    n_cells=float(rs.n_cells), n_nonspiking=float(rs.n_nonspiking),
                    n_regions=float(rs.n_regions)),
        centres=dict(pyali_spiking_within_1px=int((d_po <= 1).sum()), pyali_spiking_within_2px=int((d_po <= 2).sum()),
                     octave_within_1px=int((d_op <= 1).sum()), octave_within_2px=int((d_op <= 2).sum()),
                     n_pyali_spiking=int(spk.sum()), n_octave=int(len(oc))))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2, default=float)
    print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
