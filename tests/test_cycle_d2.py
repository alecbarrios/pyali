"""Cycle D2: spike assignment across the field (same-frame merge, spike-density clustering,
per-cluster footprints) -- unit tests, Octave oracles, and the synthetic temporal-separation test.

Oracles run only where their inputs exist: ``PYALI_OCTAVE`` (an Octave binary with the image and
statistics packages, e.g. the miniali-octave env) for the merge, and the stored Octave outputs of
the two reference fields under the AB mount for the clustering.

Run: ``PYALI_OCTAVE=/home/jovyan/.conda/envs/miniali-octave/bin/octave \\
      /home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_cycle_d2.py -q``
"""
import os
import subprocess

import numpy as np
import pytest

from pyali import assign

MINIALI = "/home/jovyan/workbench/miniali/pipeline"
OCTAVE_OUT = "/home/jovyan/spatial-technology-platform/AB/miniali-443screen2/Plate_4"
REF_FIELDS = [("Well_A2", "164512_P-4_W-A2_443-1_443screen2_DIV35__burst41"),
              ("Well_C3", "140913_P-4_W-C3_443-2_443screen2_DIV35__burst45")]


# --------------------------------------------------------------------------- #
# 1. merge_close_spikes_keep_max
# --------------------------------------------------------------------------- #
def test_merge_chains_keep_max_and_never_mixes_frames():
    S = np.array([
        [10.0, 10.0, 5, 1.0],     # 0 \
        [10.0, 17.0, 5, 3.0],     # 1  } chain 0-1-2 (7 px links) in frame 5 -> keep 1
        [10.0, 24.0, 5, 2.0],     # 2 /
        [10.0, 40.0, 5, 9.0],     # 3   alone in frame 5 (16 px away)
        [10.0, 10.0, 6, 0.5],     # 4   frame 6: never merged with frame 5
        [30.0, 30.0, 7, 4.0],     # 5 \ exactly 8 px apart: linked (inclusive)
        [30.0, 38.0, 7, 4.0],     # 6 / tie -> first member (5) wins
    ])
    surv, kept = assign.merge_close_spikes_keep_max(S, 8.0)
    assert surv.tolist() == [1, 1, 1, 3, 4, 5, 5]
    assert kept.tolist() == [1, 3, 4, 5]                    # by frame, components by first member
    s0, k0 = assign.merge_close_spikes_keep_max(np.zeros((0, 4)))
    assert s0.size == 0 and k0.size == 0
    s1, k1 = assign.merge_close_spikes_keep_max(S[:1])
    assert s1.tolist() == [0] and k1.tolist() == [0]


def test_merge_is_order_stable_within_frames():
    rng = np.random.default_rng(0)
    S = np.column_stack([rng.uniform(0, 30, (200, 2)), rng.integers(0, 5, 200), rng.normal(size=200)])
    surv, kept = assign.merge_close_spikes_keep_max(S, 8.0)
    assert np.all(S[surv, 2] == S[:, 2])                    # survivors share their frame
    assert np.all(surv[kept] == kept)                       # survivors survive themselves
    assert np.all(np.diff(S[kept, 2]) >= 0)                 # output sorted by frame
    for i in range(len(S)):                                 # the survivor is its group's maximum
        g = surv == surv[i]
        assert S[surv[i], 3] == S[g, 3].max()


@pytest.mark.skipif(not os.environ.get("PYALI_OCTAVE"), reason="set PYALI_OCTAVE to an Octave binary")
def test_merge_matches_octave_function(tmp_path):
    import scipy.io as sio
    octave = os.environ["PYALI_OCTAVE"]
    env_root = os.path.dirname(os.path.dirname(octave))
    rng = np.random.default_rng(1)
    n = 2000
    centers = rng.uniform(1, 80, (20, 2))
    S = np.column_stack([centers[rng.integers(0, 20, n)] + rng.normal(0, 3, (n, 2)),
                         rng.integers(1, 50, n).astype(float), np.round(rng.normal(5, 2, n), 1)])
    sio.savemat(tmp_path / "spikes.mat", {"S": S})
    cmd = (f"pkg load image statistics signal; warning('off','Octave:shadowed-function'); "
           f"addpath('{MINIALI}/core'); addpath('{MINIALI}/shims'); load('spikes.mat'); "
           f"M = mini_ali_mergeCloseSpikesKeepMax(S, 8); save('-v7', 'merged.mat', 'M');")
    env = dict(os.environ, OCTAVE_HOME=env_root, CONDA_PREFIX=env_root,
               PATH=f"{env_root}/bin:{os.environ.get('PATH', '')}")
    subprocess.run([octave, "--no-gui", "-q", "--eval", cmd], cwd=tmp_path, env=env, check=True,
                   capture_output=True, timeout=300)
    M = sio.loadmat(tmp_path / "merged.mat")["M"]
    _, kept = assign.merge_close_spikes_keep_max(S, 8.0)
    assert np.array_equal(S[kept], M)                       # same rows, same order


# --------------------------------------------------------------------------- #
# 2. density_map / ali_clustering
# --------------------------------------------------------------------------- #
def test_density_map_bins_and_edges():
    rc = np.array([[1.0, 1.0], [1.24, 1.0], [1.25, 1.0], [10.5, 3.0], [10.6, 3.0], [0.4, 2.0]])
    cnt, (cr, cc) = assign.density_map(rc, (10, 4), 4)
    assert cnt.shape == (40, 16) and cnt.sum() == 3         # 10.5 (last edge), 10.6 and 0.4 dropped
    assert cnt[2, 2] == 2 and cnt[3, 2] == 1                # 1.25 sits on an edge: upper bin
    assert np.isclose(cr[0], 0.625) and np.isclose(cc[-1], 4.375)


def _clouds(rng, centres, n=60, sd=0.35):
    return np.vstack([c + rng.normal(0, sd, (n, 2)) for c in centres])


def test_ali_clustering_splits_two_close_clouds_and_drops_sparse_spikes():
    rng = np.random.default_rng(2)
    rc = _clouds(rng, [(20.0, 20.0), (20.0, 24.0), (60.0, 60.0)])
    sparse = rng.uniform(70, 99, (5, 2))                     # isolated spikes: below threshold
    lab, seeds, alimap, L = assign.ali_clustering(np.vstack([rc, sparse]), 4, 100, 100, 0.15)
    a, b, c = lab[:60], lab[60:120], lab[120:180]
    for g in (a, b, c):                                      # each cloud one dominant label
        vals, cnt = np.unique(g[g > 0], return_counts=True)
        assert cnt.max() >= 0.9 * len(g)
    tops = [np.bincount(g[g > 0]).argmax() for g in (a, b, c)]
    assert len(set(tops)) == 3                               # the two touching clouds split
    assert np.all(lab[180:] == 0)
    assert L.shape == alimap.shape == (400, 400) and len(seeds) >= 3


def test_ali_clustering_degenerate_inputs_and_flood_margin():
    lab, seeds, alimap, L = assign.ali_clustering(np.zeros((0, 2)), 4, 50, 50, 0.15)
    assert lab.size == 0 and len(seeds) == 0 and not L.any()
    lab, seeds, _, _ = assign.ali_clustering(np.array([[10.0, 10.0]]), 4, 50, 50, 0.15)
    assert lab.tolist() == [0]                               # one spike never reaches threshold
    rng = np.random.default_rng(3)
    rc = _clouds(rng, [(5.0, 5.0), (5.0, 9.0), (30.0, 12.0), (1.2, 44.0)], n=40)   # incl. a border cloud
    full = assign.ali_clustering(rc, 4, 50, 50, 0.15, flood_margin=None)[0]
    fast = assign.ali_clustering(rc, 4, 50, 50, 0.15, flood_margin=16)[0]
    assert np.array_equal(full, fast)


def _octave_extras(well, fov):
    path = os.path.join(OCTAVE_OUT, well, fov, "octave_extras.mat")
    if not os.path.isfile(path):
        pytest.skip(f"Octave outputs not mounted: {path}")
    import scipy.io as sio
    return sio.loadmat(path, squeeze_me=True)


@pytest.mark.parametrize("well, fov", REF_FIELDS)
def test_ali_clustering_matches_octave_on_its_own_spikes(well, fov):
    m = _octave_extras(well, fov)
    C = np.asarray(m["COMs"], float)
    oc = np.asarray(m["clust_idx"]).astype(int)
    lab, seeds, _, _ = assign.ali_clustering(C[:, :2], 4, 800, 800, 0.15)
    # identical seeds (regional maxima of the smoothed density map)
    assert set(map(tuple, np.round(seeds, 6))) == set(map(tuple, np.round(np.asarray(m["clust_cen"]), 6)))
    assert len(np.unique(lab[lab > 0])) == len(np.unique(oc[oc > 0]))
    # labels agree up to numbering (matched by maximum overlap) on >= 99.8% of spikes
    both = (lab > 0) & (oc > 0)
    pairs, cnt = np.unique(np.stack([lab[both], oc[both]]), axis=1, return_counts=True)
    best = {}
    for (x, y), k in zip(pairs.T, cnt):
        if k > best.get(x, (0, 0))[0]:
            best[x] = (k, y)
    mapped = np.array([best.get(x, (0, -1))[1] if x > 0 else 0 for x in lab])
    assert np.mean(mapped == oc) >= 0.998
    full = assign.ali_clustering(C[:, :2], 4, 800, 800, 0.15, flood_margin=None)[0]
    assert np.array_equal(full, lab)                         # the restricted flood changes nothing here


# --------------------------------------------------------------------------- #
# 3. temporal separation of two cells that share one segmentation region
# --------------------------------------------------------------------------- #
def _trace_spikes(trace, fps=800.0, k=4.0):
    """Spike frames of a trace: 20 Hz high-pass, peaks above k robust sigma (as the SNR metric)."""
    from scipy.signal import find_peaks
    from pyali.metrics import highpass, robust_sigma
    x = highpass(np.asarray(trace, float), 20.0, fps)
    pk, _ = find_peaks(x, height=k * robust_sigma(x), distance=8)
    return pk


def test_two_cells_in_one_region_are_separated_in_time(tmp_path):
    """Two cells 4 px apart share one segmentation region and fire in alternation. The field-wide
    density clustering gives each its own cluster, footprint and trace; the pre-D2 per-region loop
    lumps both spike trains into one cell (asserted too, as the regression this fixes)."""
    import dataclasses
    from pyali import traces as tr
    from pyali.movie import corrected_movie
    from pyali.pipeline import process_fov
    from tests import reference_extract as ref
    from tests.synthetic import make_movie, spike_recall_precision, write_fov
    a_times = np.arange(60, 760, 60)                     # 12 spikes each, 30 frames apart overall
    b_times = a_times + 30
    cells = [(40, 40), (40, 44)]
    data = make_movie(seed=5, cells=cells, spike_times=[a_times, b_times], cell_amp=900.0,
                      spike_frac=0.3)
    write_fov(tmp_path, data["movie"])
    p = data["params"]
    out = process_fov(str(tmp_path), out_dir=None, p=p, save=False, verbose=False)
    lab = np.zeros((96, 96), int)
    for i, r in enumerate(out["regions"]):
        lab[r["PixelList"][:, 1].astype(int) - 1, r["PixelList"][:, 0].astype(int) - 1] = i + 1
    assert lab[40, 40] > 0 and lab[40, 40] == lab[40, 44]                    # one region for both

    def score(centres1, traces):
        c0 = np.asarray(centres1, float) - 1.0
        res = []
        for (r, c), own, other in ((cells[0], a_times, b_times), (cells[1], b_times, a_times)):
            k = int(np.argmin(np.hypot(c0[:, 0] - r, c0[:, 1] - c)))
            pk = _trace_spikes(traces[k]) - 1                                # kernel peaks a frame late
            res.append((k, spike_recall_precision(pk, own, tol=2)[0],
                        spike_recall_precision(pk, other, tol=2)[0]))
        return res

    new = score(out["footprint_center"], np.asarray(out["cell_traces"]))
    assert new[0][0] != new[1][0]                                            # two cells
    for k, rec_own, rec_other in new:
        assert out["source_of_footprint"][k] == "spiking"
        assert rec_own >= 0.9 and rec_other <= 0.1, new
    # the pre-D2 loop on the same regions: one cell carrying both trains
    raw = np.asarray(data["movie"]).astype("<u2")
    cm, _ = corrected_movie(raw, p, n_frames=raw.shape[0] - p.truncate_last)
    q = type("Q", (), {**{k: getattr(p, k) for k in p.__dataclass_fields__},
                       "dbscan_eps": 1.5, "dbscan_min_pts": 3})()
    o = ref.extract_footprints(cm, out["regions"], [r["PixelList"] for r in out["regions"]], q,
                               p.nrow, p.ncol)
    old = score(o[3], tr.extract_cell_traces(cm, o[2], p))
    assert old[0][0] == old[1][0] and old[0][2] >= 0.8, old                 # lumped
    assert dataclasses.is_dataclass(p)
