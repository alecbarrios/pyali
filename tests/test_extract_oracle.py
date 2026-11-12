"""Oracle invariants for the extraction pipeline on synthetic movies with planted ground truth.

Each test is a claim the verifier of a code review can reuse. Thresholds are deliberately loose
where the synthetic movie is noisy by design (trace correlation) and tight where the repository
itself makes a claim (bit-identical patch-local filtering).

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_extract_oracle.py -q``
"""
import dataclasses

import numpy as np
import pytest

from pyali.pipeline import process_fov
from tests.synthetic import (make_movie, write_fov, match_centers, spike_recall_precision,
                             spike_zscore)


def run(fov_dir, p, **overrides):
    """``process_fov`` on ``fov_dir`` with a *copy* of ``p`` carrying ``overrides``."""
    q = dataclasses.replace(p, **overrides)
    return process_fov(str(fov_dir), out_dir=None, p=q, save=False, verbose=False)


@pytest.fixture(scope="module")
def fov(tmp_path_factory):
    d = tmp_path_factory.mktemp("fov_default")
    data = make_movie(seed=0)
    write_fov(d, data["movie"])
    out = run(d, data["params"])
    return dict(dir=d, truth=data["truth"], params=data["params"], out=out)


def _per_cell_ap_frames(out, centers_rc0, tol=6.0):
    """Detected AP frames grouped by the nearest planted centre (``APs`` rows are
    ``[row, col, frame, amp]``, 0-indexed)."""
    aps = np.asarray(out["APs"]).reshape(-1, 4)
    groups = [[] for _ in range(len(centers_rc0))]
    if len(aps):
        j, _ = match_centers(aps[:, :2], centers_rc0, tol=tol)
        for k, row in enumerate(aps):
            if j[k] >= 0:
                groups[j[k]].append(int(row[2]))
    return groups


# --------------------------------------------------------------------------- #
# 1. happy path
# --------------------------------------------------------------------------- #
def test_segments_every_planted_cell_once(fov):
    """Every planted cell gets exactly one footprint centre within 2 px. Extra (speckle) regions
    are allowed, as on real data, but a blow-up beyond 10x the planted count is not."""
    out, truth = fov["out"], fov["truth"]
    n = len(truth["centers"])
    cen = np.array([[r["Centroid"][1] - 1, r["Centroid"][0] - 1] for r in out["regions"]]).reshape(-1, 2)
    jr, _ = match_centers(cen, truth["centers"], tol=3.0)
    assert len(set(jr[jr >= 0].tolist())) == n, "a planted cell has no segmentation region"
    assert len(out["regions"]) <= 10 * n, f"{len(out['regions'])} regions for {n} cells"
    fc = np.asarray(out["footprint_center"], float).reshape(-1, 2) - 1.0     # 1-indexed -> 0
    j, dist = match_centers(fc, truth["centers"], tol=2.0)
    hits = j[j >= 0]
    assert len(set(hits.tolist())) == n, f"planted cells matched: {sorted(set(hits.tolist()))}; dist {dist}"
    assert len(hits) == n, "two footprints on one planted cell"
    print(f"\n[oracle] regions={len(out['regions'])} footprints={len(fc)} planted={n} "
          f"extra_regions={len(out['regions']) - n}")


def test_detects_planted_spikes(fov):
    out, truth = fov["out"], fov["truth"]
    groups = _per_cell_ap_frames(out, truth["centers"])
    for i, planted in enumerate(truth["spike_times"]):
        recall, precision = spike_recall_precision(groups[i], planted, tol=1)
        assert recall >= 0.9, f"cell {i}: recall {recall:.2f}"
        assert precision >= 0.9, f"cell {i}: precision {precision:.2f}"


def test_traces_carry_the_planted_spikes(fov):
    out, truth = fov["out"], fov["truth"]
    fc = np.asarray(out["footprint_center"], float).reshape(-1, 2) - 1.0
    j, _ = match_centers(fc, truth["centers"], tol=2.0)
    ct = np.asarray(out["cell_traces"])
    assert ct.shape == (len(fc), truth["traces"].shape[1])
    assert (j >= 0).sum() == len(truth["centers"])
    for k, i in enumerate(j):
        if i < 0:                                       # a speckle region's trace: not scored
            continue
        z = spike_zscore(ct[k], truth["spike_times"][i])
        assert z >= 4.0, f"cell {k}: spike z-score {z:.1f}"
        r = np.corrcoef(ct[k], truth["traces"][i])[0, 1]
        assert r >= 0.5, f"cell {k}: trace correlation {r:.2f}"


# --------------------------------------------------------------------------- #
# 2-3. the repository's own claims
# --------------------------------------------------------------------------- #
def test_patch_local_filter_is_bit_identical(fov):
    ref = fov["out"]
    alt = run(fov["dir"], fov["params"], patch_local_filter=False)
    for key in ("footprint", "footprint_center", "cell_traces"):
        assert np.array_equal(np.asarray(ref[key]), np.asarray(alt[key])), key


def test_float32_path_matches_float64(fov):
    ref = fov["out"]
    alt = run(fov["dir"], fov["params"], compute_dtype="float32")
    assert np.asarray(alt["footprint_center"]).shape == np.asarray(ref["footprint_center"]).shape
    assert np.abs(np.asarray(alt["footprint_center"]) - np.asarray(ref["footprint_center"])).max() <= 1.0
    a, b = np.asarray(alt["cell_traces"], float), np.asarray(ref["cell_traces"], float)
    assert a.shape == b.shape
    scale = np.abs(b).max() or 1.0
    assert np.allclose(a, b, rtol=1e-2, atol=1e-2 * scale)


# --------------------------------------------------------------------------- #
# 4-8. degenerate and boundary inputs
# --------------------------------------------------------------------------- #
def test_constant_movie_reaches_the_zero_cell_path(tmp_path):
    """A perfectly constant movie: the only synthetic input that yields zero regions. With shot
    noise the production segmentation always admits speckle, and even a noise-free *drifting*
    blank yields regions (see the next test), so this is the way to exercise the empty-output
    contract (a dark real crop outside the well is the other; see test_real_fov)."""
    data = make_movie(seed=1, blank=True, noise=False, drift=False)
    write_fov(tmp_path, data["movie"])
    out = run(tmp_path, data["params"])
    H, W = data["params"].nrow, data["params"].ncol
    assert len(out["regions"]) == 0
    assert np.asarray(out["footprint"]).shape == (H, W, 0)
    assert np.asarray(out["cell_traces"]).shape == (0, data["movie"].shape[0] - data["params"].truncate_last)


@pytest.mark.xfail(strict=False, reason="candidate finding: a noise-free blank movie with only a "
                   "smooth gradient and bleaching segments into several regions near the frame "
                   "border (pilot ledger: blank-movie regions)")
def test_drifting_blank_movie_has_no_regions(tmp_path):
    data = make_movie(seed=1, blank=True, noise=False, drift=True)
    write_fov(tmp_path, data["movie"])
    out = run(tmp_path, data["params"])
    cen = [tuple(np.round(r["Centroid"], 1)) for r in out["regions"]]
    assert len(out["regions"]) == 0, f"{len(cen)} regions on a blank movie at [col,row] {cen}"


def _planted_found(out, truth, tol=2.0):
    fc = np.asarray(out["footprint_center"], float).reshape(-1, 2) - 1.0
    j, dist = match_centers(fc, truth["centers"], tol=tol)
    return sorted(set(j[j >= 0].tolist())), fc, dist


def test_cell_one_diameter_from_corner_is_found(tmp_path):
    data = make_movie(seed=2, cells=[(6, 6), (48, 48)])
    write_fov(tmp_path, data["movie"])
    out = run(tmp_path, data["params"])
    found, fc, dist = _planted_found(out, data["truth"])
    assert found == [0, 1], f"planted cells found: {found}; centres {fc}; dist {dist}"


def test_cell_three_px_from_corner_is_found(tmp_path):
    data = make_movie(seed=2, cells=[(3, 3), (48, 48)])
    write_fov(tmp_path, data["movie"])
    out = run(tmp_path, data["params"])
    found, fc, dist = _planted_found(out, data["truth"])
    assert found == [0, 1], f"planted cells found: {found}; centres {fc}; dist {dist}"


def test_single_spike_cell_uses_fallback_footprint(tmp_path):
    data = make_movie(seed=3, cells=[(48, 48)], spike_times=[[300]])
    write_fov(tmp_path, data["movie"])
    out = run(tmp_path, data["params"])
    found, fc, dist = _planted_found(out, data["truth"])
    assert found == [0], f"planted cell found: {found}; centres {fc}; dist {dist}"


def test_spikes_at_the_movie_ends_do_not_crash(tmp_path):
    T = 800
    data = make_movie(seed=4, cells=[(48, 48)], spike_times=[[5, 400, T - 6]])
    write_fov(tmp_path, data["movie"])
    out = run(tmp_path, data["params"])
    assert np.isfinite(np.asarray(out["cell_traces"], float)).all()


def test_whitened_gls_runs_and_keeps_shapes(fov):
    ref = fov["out"]
    alt = run(fov["dir"], fov["params"], whiten_traces=True)
    assert np.asarray(alt["cell_traces"]).shape == np.asarray(ref["cell_traces"]).shape
    assert np.isfinite(np.asarray(alt["cell_traces"], float)).all()
