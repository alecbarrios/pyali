"""Unit tests for Cycle B2: identity, source flag, brightness reference, weighted brightness."""
import numpy as np
import pytest

import pyali.extract as extract
from pyali.extract import SOURCES
from pyali.metadata import Identity, identity_from_dir_name
from pyali.pipeline import process_fov
from tests.synthetic import make_movie, write_fov


@pytest.fixture(scope="module")
def synthetic_fov(tmp_path_factory):
    d = tmp_path_factory.mktemp("b2_fov")
    data = make_movie(seed=0)
    write_fov(d, data["movie"])
    return d, data


def _run(fov_dir, params):
    return process_fov(str(fov_dir), out_dir=None, p=params, save=False, verbose=False)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_identity_id_formatting():
    i = Identity("443screen2", 4, "C2", 47, fov_index=12, batch="443-2")
    assert i.fov_id == "443screen2_P04_C2_F047"
    assert i.cell_id(7) == "443screen2_P04_C2_F047_C0007"
    assert i.cell_id(1234) == "443screen2_P04_C2_F047_C1234"
    d = i.as_dict()
    assert d["fov_id"] == i.fov_id and d["batch"] == "443-2" and d["fov_index"] == 12


def test_identity_from_20260716_name():
    i = identity_from_dir_name("/x/20260716/163532_P-4_W-A2_443-1_443screen2_DIV35__burst1")
    assert (i.experiment_id, i.plate, i.well, i.fov, i.fov_index, i.batch) == \
        ("443screen2", 4, "A2", 1, 0, "443-1")
    assert i.fov_id == "443screen2_P04_A2_F001"


def test_identity_from_20260718_name_without_separator():
    i = identity_from_dir_name("154200P-1_W-A2_443-1_443screen2_DIV37__burst12")
    assert (i.plate, i.well, i.fov) == (1, "A2", 12)


def test_identity_manual_prewash_without_burst_token():
    i = identity_from_dir_name("122502_P-4_W-C1_443-2_443screen2_DIV35_manual_prewash_")
    assert (i.plate, i.well, i.fov) == (4, "C1", 0)
    assert i.fov_id == "443screen2_P04_C1_F000"


def test_identity_overrides_win():
    i = identity_from_dir_name("114944P-1_W-A1_443_443screen2_DIV34__burst3",
                               plate=5, fov_index=9, experiment_id="exp", well="B9", fov=77)
    assert (i.experiment_id, i.plate, i.well, i.fov, i.fov_index) == ("exp", 5, "B9", 77, 9)
    assert i.fov_id == "exp_P05_B9_F077"


def test_identity_arbitrary_name_never_raises():
    i = identity_from_dir_name("/tmp/pytest/fov0")
    assert (i.experiment_id, i.plate, i.well, i.fov, i.batch) == ("unknown", 0, "", 0, "")
    assert i.fov_id == "unknown_P00__F000"


# --------------------------------------------------------------------------- #
# source flag
# --------------------------------------------------------------------------- #
def test_source_planted_cells_are_spiking(synthetic_fov):
    d, data = synthetic_fov
    out = _run(d, data["params"])
    src = out["source_of_footprint"]
    N = np.asarray(out["footprint"]).shape[2]
    assert len(src) == N and set(src) <= set(SOURCES)
    n_spk = np.bincount(out["spike_cell_row"][out["spike_cell_row"] >= 0], minlength=N)
    # a footprint has spikes assigned iff it is a DBSCAN cluster footprint
    assert all((s == "spiking") == (n > 0) for s, n in zip(src, n_spk))
    assert src.count("spiking") >= len(data["truth"]["centers"])


def test_source_no_ap_when_detection_finds_nothing(synthetic_fov, monkeypatch):
    d, data = synthetic_fov
    monkeypatch.setattr(extract, "detect_region_aps",
                        lambda *a, **k: (np.zeros((0, 4)), {}))
    out = _run(d, data["params"])
    assert out["source_of_footprint"] == ["no_ap"] * len(out["regions"])
    assert np.asarray(out["APs"]).shape == (0, 4)


def test_source_no_com_when_every_centroid_is_rejected(synthetic_fov, monkeypatch):
    d, data = synthetic_fov
    monkeypatch.setattr(extract, "com_via_svd", lambda *a, **k: np.zeros((0, 3)))
    out = _run(d, data["params"])
    had_ap = set(out["region_of_ap"].tolist())                 # speckle regions may have no AP
    expected = ["no_com" if c in had_ap else "no_ap" for c in range(len(out["regions"]))]
    assert out["source_of_footprint"] == expected
    assert "no_com" in expected


def test_source_all_noise_when_dbscan_rejects_everything(synthetic_fov, monkeypatch):
    d, data = synthetic_fov
    import sklearn.cluster

    class _NoiseDBSCAN:
        def __init__(self, *a, **k):
            pass

        def fit_predict(self, X):
            return -np.ones(len(X), int)

    monkeypatch.setattr(sklearn.cluster, "DBSCAN", _NoiseDBSCAN)
    out = _run(d, data["params"])
    src = out["source_of_footprint"]
    assert len(src) == len(out["regions"])
    assert set(src) <= {"all_noise", "no_ap", "no_com"} and "all_noise" in src
    assert (out["spike_cell_row"] == -1).all()


# --------------------------------------------------------------------------- #
# brightness reference
# --------------------------------------------------------------------------- #
def _blob(H, W, r, c, sigma):
    rr, cc = np.mgrid[:H, :W]
    return np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2.0 * sigma ** 2))


def test_brightness_reference_flat_movie_is_zero():
    from pyali.preprocess import brightness_reference
    movie = np.full((40, 32, 32), 17.0, np.float32)
    ref = brightness_reference(movie, n_frames=300, radius=5)
    assert ref.shape == (32, 32) and ref.dtype == np.float32
    assert np.abs(ref).max() < 1e-5


def test_brightness_reference_removes_gradient_keeps_blob_and_uses_first_frames():
    from pyali.preprocess import brightness_reference
    H = W = 64
    rr, cc = np.mgrid[:H, :W]
    gradient = 100.0 + 0.5 * cc + 0.25 * rr                   # smooth illumination
    blob = 40.0 * _blob(H, W, 30, 30, 2.0)                     # a cell-sized bright spot
    frame = (gradient + blob).astype(np.float32)
    movie = np.repeat(frame[None], 20, axis=0)
    movie[10:] += 1000.0                                        # frames beyond n_frames must be ignored
    ref = brightness_reference(movie, n_frames=10, radius=8)
    peak = ref[30, 30]
    assert abs(peak - 40.0) < 0.05 * 40.0                       # blob retained within 5%
    # background pixels, excluding the border band where any opening of a ramp undershoots
    # by slope * radius (nearest padding); that band is the price of the operator, not a bug
    interior = (rr >= 8) & (rr < H - 8) & (cc >= 8) & (cc < W - 8)
    far = ref[(np.hypot(rr - 30, cc - 30) > 12) & interior]
    assert np.abs(far).max() < 0.01 * 40.0                      # gradient removed


# --------------------------------------------------------------------------- #
# weighted brightness
# --------------------------------------------------------------------------- #
def _sparse(planes):
    from pyali.results import SparseFootprints
    return SparseFootprints.from_dense(np.stack(planes, axis=2))


def test_weighted_brightness_hand_computed_two_cells():
    from pyali.results import weighted_brightness
    H = W = 6
    p0 = np.zeros((H, W)); p0[1, 1] = 1.0; p0[1, 2] = 3.0           # cell 0: weights 1, 3
    p1 = np.zeros((H, W)); p1[4, 4] = 2.0                            # cell 1: single pixel
    ref = np.arange(H * W, dtype=float).reshape(H, W)                # x = row * W + col
    b = weighted_brightness(_sparse([p0, p1]), ref)
    assert b.dtype == np.float32 and b.shape == (2,)
    assert abs(b[0] - (1 * 7 + 3 * 8) / 4.0) < 1e-6                  # pixels (1,1)=7, (1,2)=8
    assert abs(b[1] - 28.0) < 1e-6                                   # (4,4) = 28


def test_weighted_brightness_ignores_non_positive_weights_and_nan_pixels():
    from pyali.results import weighted_brightness
    H = W = 5
    p = np.zeros((H, W)); p[2, 2] = 4.0; p[2, 3] = -1.0; p[3, 3] = 2.0
    ref = np.full((H, W), 10.0); ref[2, 2] = 20.0; ref[2, 3] = 1e6; ref[3, 3] = np.nan
    b = weighted_brightness(_sparse([p]), ref)
    assert abs(b[0] - 20.0) < 1e-6                                   # only (2,2) counts


def test_weighted_brightness_nan_when_no_positive_weight_and_shape_check():
    from pyali.results import weighted_brightness, SparseFootprints
    H = W = 4
    p = np.zeros((H, W)); p[1, 1] = -5.0
    b = weighted_brightness(_sparse([p, np.zeros((H, W))]), np.ones((H, W)))
    assert np.isnan(b).all()
    with pytest.raises(ValueError):
        weighted_brightness(_sparse([p]), np.ones((H, W + 1)))
    assert weighted_brightness(SparseFootprints.from_dense(np.zeros((H, W, 0))),
                               np.ones((H, W))).shape == (0,)


# --------------------------------------------------------------------------- #
# pipeline wiring: cells columns, brightness, metadata
# --------------------------------------------------------------------------- #
def test_pipeline_writes_identity_source_brightness_and_metadata(synthetic_fov, tmp_path):
    import json
    import os
    from pyali.results import load_results, weighted_brightness
    d, data = synthetic_fov
    out_dir = str(tmp_path / "analysis")
    ident = Identity("exp", 5, "A2", 12, fov_index=7, batch="443-1")
    out = process_fov(str(d), out_dir=out_dir, p=data["params"], save=True, verbose=False,
                      identity=ident)
    r = load_results(out_dir)
    N = len(r.cells)
    assert N == np.asarray(out["footprint"]).shape[2]
    assert r.cells["cell_id"].tolist() == [f"exp_P05_A2_F012_C{i:04d}" for i in range(1, N + 1)]
    assert set(r.cells["fov_index"]) == {7} and set(r.cells["plate"]) == {5}
    assert set(r.cells["source"]) <= set(SOURCES) and "spiking" in set(r.cells["source"])
    # brightness is recomputable from the written files alone
    recomputed = weighted_brightness(r.footprints, r.images["brightness_reference"])
    assert np.array_equal(r.cells["weighted_brightness"].to_numpy(), recomputed, equal_nan=True)
    # the brightness reference is in raw units: planted cells are brighter than background
    br = r.images["brightness_reference"]
    for rc in data["truth"]["centers"]:
        assert br[int(round(rc[0])), int(round(rc[1]))] > 5 * np.median(np.abs(br))
    with open(os.path.join(out_dir, "fov_metadata.json")) as f:
        meta = json.load(f)
    assert meta["fov_id"] == "exp_P05_A2_F012" and meta["identity"]["batch"] == "443-1"
    assert meta["params"]["brightness_radius"] == data["params"].brightness_radius
    assert meta["params"]["min_peak_interval"] == data["params"].min_peak_interval
    assert sum(meta["n_cells_by_source"].values()) == N == meta["n_cells"]
    for stage in ("load", "brightness_reference", "extract_footprints", "trace_extraction",
                  "write", "total"):
        assert stage in meta["timing_s"] and meta["timing_s"][stage] >= 0
    assert len(meta["pyali_git"]) in (0, 40)


def test_pipeline_default_identity_comes_from_the_directory_name(tmp_path):
    from pyali.results import load_results
    d = tmp_path / "163532_P-4_W-A2_443-1_443screen2_DIV35__burst9"
    d.mkdir()
    data = make_movie(seed=3)
    write_fov(d, data["movie"])
    out_dir = str(d / "analysis")
    process_fov(str(d), out_dir=out_dir, p=data["params"], save=True, verbose=False)
    r = load_results(out_dir)
    assert set(r.cells["fov_id"]) == {"443screen2_P04_A2_F009"}
    assert r.attrs["fov_id"] == "443screen2_P04_A2_F009"


def test_cli_identity_flags_reach_the_written_cells(tmp_path, monkeypatch):
    """run_pyali's --experiment-id/--plate/--well/--fov/--fov-index land in cells.parquet."""
    import sys
    import pandas as pd
    repo = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
    sys.path.insert(0, __import__("os").path.join(repo, "scripts"))
    import run_pyali
    import pyali.pipeline as pipeline
    captured = {}

    def fake_process_fov(fov_dir, **kw):
        captured["identity"] = kw["identity"]
        return {}

    monkeypatch.setattr(run_pyali, "process_fov", fake_process_fov)
    d = tmp_path / "154200P-1_W-A2_443-1_443screen2_DIV37__burst3"
    d.mkdir()
    data = make_movie(seed=5)
    write_fov(d, data["movie"])
    rc = run_pyali.main([str(d), "--profile", "443screen2", "--nrow", "96", "--ncol", "96",
                         "--plate", "5", "--fov-index", "42"])
    assert rc == 0
    ident = captured["identity"]
    assert ident.fov_id == "443screen2_P05_A2_F003" and ident.fov_index == 42
    assert ident.batch == "443-1"
    del pd, pipeline


def test_brightness_reference_leaves_movie_untouched():
    from pyali.preprocess import brightness_reference
    movie = np.random.default_rng(1).random((12, 16, 16)).astype(np.float32)
    before = movie.copy()
    brightness_reference(movie, n_frames=5, radius=3)
    assert np.array_equal(movie, before)
