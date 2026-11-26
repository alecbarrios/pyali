"""Oracle invariants for the extraction pipeline on synthetic movies with planted ground truth.

Each test is a claim the verifier of a code review can reuse. Thresholds are deliberately loose
where the synthetic movie is noisy by design (trace correlation) and tight where the repository
itself makes a claim (bit-identical patch-local filtering).

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_extract_oracle.py -q``
"""
import dataclasses
import os

import numpy as np
import pytest

from pyali.pipeline import process_fov
from pyali.results import load_results
from tests.synthetic import (make_movie, write_fov, match_centers, spike_recall_precision,
                             spike_zscore)


def run(fov_dir, p, out_dir=None, **overrides):
    """``process_fov`` on ``fov_dir`` with a *copy* of ``p`` carrying ``overrides``; results are
    written only when ``out_dir`` is given."""
    q = dataclasses.replace(p, **overrides)
    return process_fov(str(fov_dir), out_dir=out_dir, p=q, save=out_dir is not None,
                       verbose=False)


@pytest.fixture(scope="module")
def fov(tmp_path_factory):
    d = tmp_path_factory.mktemp("fov_default")
    data = make_movie(seed=0)
    write_fov(d, data["movie"])
    out_dir = str(d / "analysis")
    out = run(d, data["params"], out_dir=out_dir)
    return dict(dir=d, truth=data["truth"], params=data["params"], out=out, out_dir=out_dir)


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


@pytest.fixture(scope="module")
def burst_fov(tmp_path_factory):
    d = tmp_path_factory.mktemp("fov_burst")
    data = make_movie(seed=0, burst=True)
    write_fov(d, data["movie"])
    out = run(d, data["params"])
    return dict(dir=d, truth=data["truth"], params=data["params"], out=out)


def test_bursts_keep_one_spike_per_burst_at_min_interval(burst_fov):
    """Spikes planted in 4-spike bursts at 10-frame spacing (plus isolated spikes): the per-region
    dedup keeps the largest member of each burst instead of discarding the whole burst. Every
    detected frame is a planted spike (precision), at least one spike per burst is recalled, and
    a cell's detected frames are pairwise >= ``min_peak_interval`` apart."""
    out, truth, p = burst_fov["out"], burst_fov["truth"], burst_fov["params"]
    groups = _per_cell_ap_frames(out, truth["centers"])
    burst_hits, isolated_hits = [], []
    for i, planted in enumerate(truth["spike_times"]):
        found = np.array(sorted(set(groups[i])), int)
        assert len(found), f"cell {i}: no APs"
        _, precision = spike_recall_precision(found, planted, tol=1)
        assert precision >= 0.9, f"cell {i}: precision {precision:.2f}"
        if len(found) > 1:
            assert np.diff(found).min() >= p.min_peak_interval, f"cell {i}: frames {found}"
        for g in truth["spike_groups"][i]:
            hit = bool(np.any(np.abs(found[:, None] - np.asarray(g)[None, :]) <= 1))
            (burst_hits if len(g) > 1 else isolated_hits).append(hit)
    assert len(burst_hits) >= 4
    assert np.mean(burst_hits) >= 0.9, f"burst recall {np.mean(burst_hits):.2f}"
    assert np.mean(isolated_hits) >= 0.9, f"isolated-spike recall {np.mean(isolated_hits):.2f}"


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
    contract (a dark real crop outside the well is the other; see test_real_fov). The written
    results must load back with every field empty."""
    data = make_movie(seed=1, blank=True, noise=False, drift=False)
    write_fov(tmp_path, data["movie"])
    out_dir = str(tmp_path / "analysis")
    out = run(tmp_path, data["params"], out_dir=out_dir)
    H, W = data["params"].nrow, data["params"].ncol
    T = data["movie"].shape[0] - data["params"].truncate_last
    assert len(out["regions"]) == 0
    assert np.asarray(out["footprint"]).shape == (H, W, 0)
    assert np.asarray(out["cell_traces"]).shape == (0, T)
    r = load_results(out_dir)
    assert len(r.cells) == 0 and len(r.spikes) == 0
    assert r.traces.shape == (0, T) and r.footprints.n() == 0 and r.centers.shape == (0, 2)
    assert r.footprints.dense().shape == (H, W, 0)
    assert r.attrs["n_frames"] == T and r.attrs["n_ap_detected"] == 0


@pytest.mark.parametrize("noise", [True, False])
def test_drifting_blank_movie_has_no_soma_sized_region(tmp_path, noise):
    """A blank movie with a smooth gradient and bleaching (with or without shot noise) yields no
    soma-sized region. The hysteresis threshold is calibrated to the field's own noise and the
    large-scale background removes the gradient; what survives is a few pixels of watershed
    residue, because the soma mask's fixed top-5% rule is never empty and nothing is filtered
    after the AND (a minimum area there deleted real somata on real fields). The real dark crop
    outside the well yields exactly zero pixels (tests/test_real_fov.py)."""
    data = make_movie(seed=1, blank=True, noise=noise, drift=True)
    write_fov(tmp_path, data["movie"])
    out = run(tmp_path, data["params"])
    areas = sorted(int(r["Area"]) for r in out["regions"])
    assert len(out["regions"]) <= 3 and sum(areas) <= 12, f"blank movie regions with areas {areas}"
    assert all(a < 8 for a in areas), f"soma-sized region on a blank movie: areas {areas}"


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


# --------------------------------------------------------------------------- #
# 9. per-footprint region ids and per-spike cell assignment (Cycle B1)
# --------------------------------------------------------------------------- #
def test_region_and_spike_bookkeeping(fov):
    out, p = fov["out"], fov["params"]
    fc = np.asarray(out["footprint_center"], float).reshape(-1, 2)
    aps = np.asarray(out["APs"]).reshape(-1, 4)
    coms = np.asarray(out["COMs"]).reshape(-1, 3)
    rof = np.asarray(out["region_of_footprint"])
    roa = np.asarray(out["region_of_ap"])
    ros = np.asarray(out["region_of_spike"])
    scr = np.asarray(out["spike_cell_row"])
    N, n_regions = len(fc), len(out["regions"])
    assert rof.shape == (N,) and roa.shape == (len(aps),) and ros.shape == (len(coms),)
    assert scr.shape == (len(coms),)
    assert len(coms) <= len(aps) and len(aps) > 0
    assert set(rof.tolist()) == set(range(n_regions)), "every region yields >= 1 footprint"
    assert rof.min() >= 0 and rof.max() < n_regions and roa.max() < n_regions
    assert np.all((scr == -1) | ((scr >= 0) & (scr < N)))
    # a localized spike belongs to the region of the AP it came from, and that region's APs
    # contain its frame
    for r in set(ros.tolist()):
        assert set(coms[ros == r, 2].astype(int)) <= set(aps[roa == r, 2].astype(int))
    # an assigned spike's cell lies in the spike's region, and sits within 2 px of its centre
    n_spikes = np.bincount(scr[scr >= 0], minlength=N)
    for i in np.flatnonzero(scr >= 0):
        k = scr[i]
        assert rof[k] == ros[i], f"spike {i}: cell {k} is in region {rof[k]}, spike in {ros[i]}"
        assert np.hypot(coms[i, 0] - fc[k, 0], coms[i, 1] - fc[k, 1]) <= 2.0, (
            f"spike {i} at {coms[i, :2]} is far from cell {k} centre {fc[k]}")
    # cluster cells carry at least a DBSCAN cluster's worth of spikes; fallback cells carry none
    assert np.all((n_spikes == 0) | (n_spikes >= p.dbscan_min_pts))
    assert (n_spikes > 0).sum() >= len(fov["truth"]["centers"]), "a planted cell has no spikes"
    print(f"\n[oracle] cells={N} spiking={(n_spikes > 0).sum()} fallback={(n_spikes == 0).sum()} "
          f"APs={len(aps)} localized={len(coms)} assigned={(scr >= 0).sum()}")


# --------------------------------------------------------------------------- #
# 10. the written results are the in-memory results (Cycle B1)
# --------------------------------------------------------------------------- #
def test_written_results_round_trip(fov):
    out, out_dir = fov["out"], fov["out_dir"]
    for legacy in ("ALI_Result.mat", "ALI_Int_Result.mat"):
        assert not os.path.exists(os.path.join(out_dir, legacy)), legacy
    for name in ("cells.parquet", "cells.csv", "spikes.parquet", "signals.h5", "fov_metadata.json"):
        assert os.path.isfile(os.path.join(out_dir, name)), name
    r = load_results(out_dir)
    assert np.array_equal(r.traces, np.asarray(out["cell_traces"]).astype(np.float32))
    # values are stored float32 (exact for float32-computed FOVs)
    assert np.array_equal(r.footprints.dense(), np.asarray(out["footprint"]).astype(np.float32))
    fc = np.asarray(out["footprint_center"], float).reshape(-1, 2)
    assert np.array_equal(r.centers, fc.astype(np.float32).astype(np.float64))
    N = len(fc)
    assert r.cells["local_cell_id"].tolist() == list(range(1, N + 1))
    assert r.cells["trace_row"].tolist() == list(range(N))
    assert r.cells["region_id"].tolist() == np.asarray(out["region_of_footprint"]).tolist()
    scr = np.asarray(out["spike_cell_row"])
    assert r.cells["n_spikes"].tolist() == np.bincount(scr[scr >= 0], minlength=N).tolist()
    coms = np.asarray(out["COMs"]).reshape(-1, 3)
    assert len(r.spikes) == len(coms)
    assert r.spikes["frame"].tolist() == coms[:, 2].astype(int).tolist()
    assert r.spikes["cell_row"].tolist() == scr.tolist()
    assert np.isfinite(r.spikes["amplitude"]).all() and (r.spikes["amplitude"] > 0).all()
    assert r.attrs["n_ap_detected"] == len(out["APs"]) and r.attrs["n_ap_localized"] == len(coms)
    assert r.attrs["fps"] == fov["params"].fps and r.attrs["schema_version"] == 2
    assert np.array_equal(r.images["reference"], np.asarray(out["reference_image"], np.float32))
    assert np.array_equal(r.images["binary_map"], np.asarray(out["binary_map"], np.uint8))
    assert np.array_equal(r.images["brightness_reference"], out["reference_corrected"])
    # Cycle B2 columns
    ident = out["identity"]
    assert r.cells["source"].tolist() == out["source_of_footprint"]
    assert r.cells["fov_id"].tolist() == [ident.fov_id] * N and r.attrs["fov_id"] == ident.fov_id
    assert r.cells["cell_id"].tolist() == [ident.cell_id(i) for i in range(1, N + 1)]
    b = r.cells["weighted_brightness"].to_numpy()
    assert np.array_equal(b, np.asarray(out["brightness"], np.float32), equal_nan=True)
    assert np.isfinite(b[r.cells["source"] == "spiking"]).all()
