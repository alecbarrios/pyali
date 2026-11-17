"""Smoke tests for every consumer of the per-FOV results layout (Cycle B1): each reader is
exercised on a synthetic FOV written by ``process_fov`` through :mod:`pyali.results`.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_readers.py -q``
"""
import importlib.util
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

from pyali.io import save_mat_v73
from pyali.pipeline import process_fov
from pyali.results import load_results
from tests.synthetic import make_movie, write_fov

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO, "scripts")
_SNR_COMPARE = os.path.join(_REPO, "snr_analysis", "snr", "snr_compare.py")
PY = sys.executable
DAY, DIR_NAME = "20990101", "120000_P-1_W-A1_443-2_443screen2_DIV35_burst1"


def _import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """``read_root/<day>/<dir_name>/`` holding one written synthetic FOV (the corpus layout)."""
    root = tmp_path_factory.mktemp("corpus")
    movie_dir = root / "movie"
    data = make_movie(seed=0)
    write_fov(movie_dir, data["movie"])
    read_root = root / "read_root"
    out_dir = read_root / DAY / DIR_NAME
    out = process_fov(str(movie_dir), out_dir=str(out_dir), p=data["params"], save=True,
                      verbose=False, profile="synthetic", day=DAY)
    return dict(read_root=str(read_root), out_dir=str(out_dir), out=out, params=data["params"])


# --------------------------------------------------------------------------- #
# snr_compare: new layout, legacy fallback, A/B self-compare
# --------------------------------------------------------------------------- #
def test_snr_compare_loads_new_layout(corpus):
    sc = _import_path("snr_compare", _SNR_COMPARE)
    r = load_results(corpus["out_dir"])
    for arg in (corpus["out_dir"], os.path.join(corpus["out_dir"], "signals.h5")):
        ct, fc, src = sc.load_result(arg)
        assert src == corpus["out_dir"]
        assert ct.dtype == np.float64 and np.array_equal(ct, r.traces.astype(np.float64))
        assert np.array_equal(fc, r.centers)
    # analysis/ subdir is searched too
    parent = os.path.dirname(corpus["out_dir"])
    link = os.path.join(parent, "with_analysis")
    os.makedirs(link, exist_ok=True)
    if not os.path.exists(os.path.join(link, "analysis")):
        os.symlink(corpus["out_dir"], os.path.join(link, "analysis"))
    ct, fc, src = sc.load_result(link)
    assert ct.shape == r.traces.shape and src.endswith("analysis")


def test_snr_compare_legacy_mat_fallback(corpus, tmp_path):
    sc = _import_path("snr_compare", _SNR_COMPARE)
    out = corpus["out"]
    ct0 = np.asarray(out["cell_traces"])
    fc0 = np.asarray(out["footprint_center"], float).reshape(-1, 2)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    mat = str(legacy / "ALI_Result.mat")
    save_mat_v73(mat, footprint=np.asarray(out["footprint"]), footprint_center=fc0, cell_traces=ct0)
    for arg in (str(legacy), mat):                       # dir with the .mat and no signals.h5; the file
        ct, fc, src = sc.load_result(arg)
        assert src == mat
        assert np.array_equal(ct, ct0.astype(np.float64)) and np.array_equal(fc, fc0)
    # a directory holding BOTH prefers the new layout
    both = tmp_path / "both"
    both.mkdir()
    for name in os.listdir(corpus["out_dir"]):
        os.symlink(os.path.join(corpus["out_dir"], name), str(both / name))
    os.symlink(mat, str(both / "ALI_Result.mat"))
    _ct, _fc, src = sc.load_result(str(both))
    assert src == str(both)
    with pytest.raises(FileNotFoundError):
        sc.load_result(str(tmp_path / "nothing_here"))


def test_snr_compare_self_ab_has_unit_correlation(corpus, tmp_path):
    out = str(tmp_path / "report")
    cp = subprocess.run([PY, _SNR_COMPARE, corpus["out_dir"], corpus["out_dir"], "--out", out,
                         "--fps", str(corpus["params"].fps)], capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr[-2000:]
    with open(os.path.join(out, "summary.json")) as f:
        s = json.load(f)
    N = len(load_results(corpus["out_dir"]).cells)
    assert s["counts"]["a_cells"] == N and s["counts"]["matched"] == N
    assert s["paired"]["corr_raw"]["median"] == pytest.approx(1.0)
    assert s["paired"]["corr_hp"]["median"] == pytest.approx(1.0)
    per_cell = pd.read_csv(os.path.join(out, "per_cell_metrics.csv"))
    assert len(per_cell) == N and np.allclose(per_cell["corr_raw"], 1.0)
    assert np.allclose(per_cell["center_dist"], 0.0)


# --------------------------------------------------------------------------- #
# corpus stages
# --------------------------------------------------------------------------- #
def test_snr_corpus_stage_writes_metrics(corpus):
    sys.path.insert(0, _SCRIPTS)
    import snr_corpus
    assert snr_corpus.RESULT == "signals.h5" and "cells.parquet" in snr_corpus.REQUIRED
    rec = snr_corpus._run_one((DAY, DIR_NAME, corpus["read_root"], "s3://unused/prefix", "s3fs"))
    assert rec["ok"], rec.get("error")
    N = len(load_results(corpus["out_dir"]).cells)
    assert rec["n_cells"] == N and rec["n_cells_meta"] == N
    csv = pd.read_csv(os.path.join(corpus["out_dir"], snr_corpus.METRICS))
    assert list(csv.columns) == list(snr_corpus.COLUMNS) and len(csv) == N


def test_snr_sta_stage_writes_sta(corpus):
    sys.path.insert(0, _SCRIPTS)
    import snr_sta
    rec = snr_sta._run_one((DAY, DIR_NAME, corpus["read_root"], "s3://unused/prefix", "s3fs",
                            True, True))
    assert rec["ok"], rec.get("error")
    N = len(load_results(corpus["out_dir"]).cells)
    assert rec["n_cells_total"] == N
    assert os.path.isfile(os.path.join(corpus["out_dir"], snr_sta.STA))


def test_representative_traces_imports():
    sys.path.insert(0, _SCRIPTS)
    import representative_traces
    assert representative_traces.load_results is load_results


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def test_registration_reads_cells_from_results(corpus):
    from pyali.registration.io import read_voltage_cells
    out, p = corpus["out"], corpus["params"]
    v = read_voltage_cells(corpus["out_dir"], p.fps)
    assert np.array_equal(v["footprint"], np.asarray(out["footprint"]).astype(np.float32))
    fc = np.asarray(out["footprint_center"], float).reshape(-1, 2)
    assert np.array_equal(v["center_rc"], fc.astype(np.float32).astype(np.float64) - 1.0)
    assert v["traces"].dtype == np.float64 and v["traces"].shape == np.asarray(out["cell_traces"]).shape
    assert set(v["snr"]) >= {"noise_sigma", "snr_median", "spectral_hf_snr", "n_spikes"}
    assert len(v["snr"]["n_spikes"]) == len(fc)


# --------------------------------------------------------------------------- #
# cell_profile_to_corpus
# --------------------------------------------------------------------------- #
def test_cell_profile_to_corpus_writes_new_layout(tmp_path):
    sys.path.insert(0, _SCRIPTS)
    import cell_profile_to_corpus as cpc
    rng = np.random.default_rng(0)
    N, T, H, W, n_fov = 7, 150, 20, 30, 2
    traces = rng.normal(size=(N, T)).astype(np.float32)
    fov = np.array([1, 1, 2, 1, 2, 2, 2], np.uint16)
    com = rng.uniform(1, 20, size=(N, 2)).astype(np.float32)
    ref = rng.integers(0, 255, size=(H, W, n_fov)).astype(np.uint16)
    mat = str(tmp_path / "cell_profile_JZ999.mat")
    with h5py.File(mat, "w") as f:                      # MATLAB v7.3: dimensions reversed
        f["ALI_traces"] = traces.T
        f["FOV"] = fov.reshape(1, N)
        f["COM"] = com.T
        f["reference_image_base"] = np.ascontiguousarray(ref.T)      # (n_fov, W, H)
    out_root = str(tmp_path / "out")
    rec = cpc.convert(mat, out_root, fps=800.0, bit="8-bit", invert=True)
    assert rec["n_cells"] == N and rec["n_fov"] == n_fov
    for k in range(1, n_fov + 1):
        d = os.path.join(out_root, "JZ999", f"JZ999_fov{k:02d}")
        for legacy in ("ALI_Result.mat", "ALI_Int_Result.mat"):
            assert not os.path.exists(os.path.join(d, legacy))
        assert os.path.isfile(os.path.join(d, "fov_metadata.json"))
        r = load_results(d)
        sel = np.flatnonzero(fov == k)
        assert np.array_equal(r.traces, -traces[sel])
        assert np.array_equal(r.centers, com[sel].astype(np.float64))
        assert len(r.cells) == len(sel) and len(r.spikes) == 0
        assert r.footprints.n() == len(sel) and r.footprints.dense().shape == (H, W, len(sel))
        assert np.array_equal(r.images["reference"], ref[:, :, k - 1].astype(np.float32))
        assert r.attrs["height"] == H and r.attrs["width"] == W and r.attrs["fps"] == 800.0
