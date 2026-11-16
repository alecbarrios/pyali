"""Unit tests for :mod:`pyali.results`: the sparse footprint packing and the on-disk layout.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_results.py -q``
"""
import numpy as np

from pyali.results import SparseFootprints


def _random_planes(rng, H, W, N, density=0.02):
    fp = np.zeros((H, W, N))
    for k in range(N):
        n = max(1, int(density * H * W))
        idx = rng.choice(H * W, size=n, replace=False)
        fp[:, :, k].ravel()[idx] = rng.uniform(0.1, 5.0, size=n)        # float64, positive
        # a few negative values too: the packing must not assume a sign
        fp[:, :, k].ravel()[idx[:3]] *= -1.0
    return fp


# --------------------------------------------------------------------------- #
# SparseFootprints
# --------------------------------------------------------------------------- #
def test_sparse_round_trip_is_exact():
    rng = np.random.default_rng(0)
    H, W, N = 37, 53, 9
    fp = _random_planes(rng, H, W, N).astype(np.float32).astype(np.float64)   # float32-representable
    sp = SparseFootprints.from_dense(fp)
    assert sp.n() == N and sp.height == H and sp.width == W
    assert sp.values.dtype == np.float32 and sp.pixel.dtype == np.uint32 and sp.offsets.dtype == np.uint32
    assert sp.offsets[0] == 0 and sp.offsets[-1] == len(sp.values) == len(sp.pixel)
    assert np.array_equal(sp.dense(), fp)
    for k in range(N):
        assert np.array_equal(sp.plane(k), fp[:, :, k].astype(np.float32))


def test_sparse_pixel_index_is_c_order_row_major():
    H, W = 5, 7
    fp = np.zeros((H, W, 1))
    fp[2, 3, 0] = 1.5                                   # row 2, col 3 -> 2 * 7 + 3 = 17
    fp[4, 6, 0] = 2.5                                   # row 4, col 6 -> 34
    sp = SparseFootprints.from_dense(fp)
    assert sp.pixel.tolist() == [17, 34]
    assert sp.values.tolist() == [1.5, 2.5]
    assert sp.offsets.tolist() == [0, 2]
    assert np.array_equal(sp.plane(0), fp[:, :, 0])


def test_sparse_cell_with_no_nonzero_pixels():
    H, W = 8, 8
    fp = np.zeros((H, W, 3))
    fp[1, 1, 0] = 1.0
    fp[2, 2, 2] = 3.0                                   # cell 1 is entirely zero
    sp = SparseFootprints.from_dense(fp)
    assert sp.offsets.tolist() == [0, 1, 1, 2]
    assert np.array_equal(sp.plane(1), np.zeros((H, W), np.float32))
    assert np.array_equal(sp.dense(), fp)


def test_sparse_n_equals_zero():
    H, W = 6, 9
    sp = SparseFootprints.from_dense(np.zeros((H, W, 0)))
    assert sp.n() == 0
    assert sp.values.shape == (0,) and sp.pixel.shape == (0,) and sp.offsets.tolist() == [0]
    assert sp.dense().shape == (H, W, 0)
    assert sp.height == H and sp.width == W


# --------------------------------------------------------------------------- #
# write_results / load_results
# --------------------------------------------------------------------------- #
import os                                                                  # noqa: E402

import h5py                                                                # noqa: E402
import pandas as pd                                                        # noqa: E402

from pyali.metadata import Identity                                        # noqa: E402
from pyali.results import (CELLS_DTYPES, SCHEMA_VERSION, SPIKES_DTYPES,    # noqa: E402
                           load_results, make_cells, pack_spikes, write_results)

IDENT = Identity("exp", 4, "C2", 47, fov_index=3, batch="443-2")


def _cells(centers, region, n_spikes, source=None, brightness=None, identity=IDENT):
    n = len(np.asarray(centers).reshape(-1, 2))
    source = ["spiking"] * n if source is None else source
    brightness = np.arange(n, dtype=np.float32) if brightness is None else brightness
    return make_cells(centers, region, n_spikes, source, brightness, identity)


def _synthetic_fov(rng, H=40, W=50, N=5, T=300):
    fp = _random_planes(rng, H, W, N).astype(np.float32).astype(np.float64)
    traces = rng.normal(size=(N, T)).astype(np.float32).astype(np.float64)
    centers = np.column_stack([rng.integers(1, H + 1, N), rng.integers(1, W + 1, N)]).astype(float)
    region = np.array([0, 0, 1, 2, 3])
    # APs: 3 in region 0 (one of them loses its COM), 1 in region 1, 2 in region 3
    APs = np.array([[3, 4, 100, 2.5], [3, 4, 150, 1.5], [3, 4, 200, 3.0],
                    [10, 10, 100, 4.0], [20, 20, 50, 0.5], [20, 20, 90, 0.7]])
    region_of_ap = np.array([0, 0, 0, 1, 3, 3])
    COMs = np.array([[3.2, 4.1, 100], [3.1, 4.4, 200], [10.5, 10.5, 100], [20.1, 20.2, 50],
                     [20.3, 20.4, 90]])
    region_of_spike = np.array([0, 0, 1, 3, 3])
    spike_cell_row = np.array([0, 1, 2, -1, 4])
    n_spikes = np.bincount(spike_cell_row[spike_cell_row >= 0], minlength=N)
    cells = _cells(centers, region, n_spikes, source=["spiking", "spiking", "spiking", "no_ap", "spiking"],
                   brightness=np.array([1.5, np.nan, 3.0, 4.0, 5.0], np.float32))
    spikes = pack_spikes(APs, COMs, spike_cell_row, region_of_spike, region_of_ap)
    images = dict(reference=rng.uniform(size=(H, W)).astype(np.float32),
                  binary_map=(rng.uniform(size=(H, W)) > 0.7).astype(np.uint8),
                  brightness_reference=rng.normal(size=(H, W)).astype(np.float32))
    attrs = dict(fps=800.0, profile="synthetic", n_ap_detected=len(APs), n_ap_localized=len(COMs),
                 nothing=None)
    return dict(fp=fp, traces=traces, centers=centers, region=region, n_spikes=n_spikes,
                cells=cells, spikes=spikes, images=images, attrs=attrs, APs=APs, COMs=COMs,
                spike_cell_row=spike_cell_row, region_of_spike=region_of_spike)


def test_make_cells_schema():
    cells = _cells([[10.0, 20.0], [30.5, 40.25]], [3, 7], [2, 0], source=["spiking", "no_com"],
                   brightness=[12.5, np.nan])
    assert list(cells.columns) == list(CELLS_DTYPES)
    assert {c: str(t) for c, t in cells.dtypes.items()} == CELLS_DTYPES
    assert cells["local_cell_id"].tolist() == [1, 2] and cells["trace_row"].tolist() == [0, 1]
    assert cells["y_px"].tolist() == [10.0, 30.5] and cells["x_px"].tolist() == [20.0, 40.25]
    assert cells["region_id"].tolist() == [3, 7] and cells["n_spikes"].tolist() == [2, 0]
    assert cells["cell_id"].tolist() == ["exp_P04_C2_F047_C0001", "exp_P04_C2_F047_C0002"]
    assert cells["fov_id"].tolist() == ["exp_P04_C2_F047"] * 2
    assert cells["experiment_id"].tolist() == ["exp"] * 2 and cells["well"].tolist() == ["C2"] * 2
    assert cells["plate"].tolist() == [4, 4] and cells["fov"].tolist() == [47, 47]
    assert cells["fov_index"].tolist() == [3, 3]
    assert cells["source"].tolist() == ["spiking", "no_com"]
    assert cells["weighted_brightness"][0] == np.float32(12.5) and np.isnan(cells["weighted_brightness"][1])


def test_make_cells_rejects_bad_source_and_lengths():
    import pytest
    with pytest.raises(ValueError):
        _cells([[1.0, 1.0]], [0], [0], source=["bogus"])
    with pytest.raises(ValueError):
        _cells([[1.0, 1.0], [2.0, 2.0]], [0, 1], [0, 0], source=["spiking"])
    with pytest.raises(ValueError):
        _cells([[1.0, 1.0]], [0], [0], brightness=[1.0, 2.0])


def test_pack_spikes_joins_amplitude_by_region_and_frame():
    s = _synthetic_fov(np.random.default_rng(1))
    sp = s["spikes"]
    assert list(sp.columns) == list(SPIKES_DTYPES)
    assert {c: str(t) for c, t in sp.dtypes.items()} == SPIKES_DTYPES
    assert len(sp) == len(s["COMs"])
    # region 0 frame 100 -> 2.5; region 1 frame 100 -> 4.0 (same frame, different region)
    assert np.array_equal(sp["amplitude"], np.array([2.5, 3.0, 4.0, 0.5, 0.7], np.float32))
    assert sp["frame"].tolist() == [100, 200, 100, 50, 90]
    assert sp["cell_row"].tolist() == [0, 1, 2, -1, 4]
    assert np.allclose(sp["row"], s["COMs"][:, 0]) and np.allclose(sp["col"], s["COMs"][:, 1])


def test_write_and_load_round_trip_every_field(tmp_path):
    s = _synthetic_fov(np.random.default_rng(2))
    out = str(tmp_path / "fov")
    write_results(out, cells_df=s["cells"], traces=s["traces"], footprints=s["fp"],
                  spikes_df=s["spikes"], images=s["images"], attrs=s["attrs"])
    for name in ("cells.parquet", "cells.csv", "spikes.parquet", "signals.h5"):
        assert os.path.isfile(os.path.join(out, name)), name
    for p in (out, os.path.join(out, "signals.h5")):           # dir or the h5 path
        r = load_results(p)
        pd.testing.assert_frame_equal(r.cells, s["cells"])
        pd.testing.assert_frame_equal(r.spikes, s["spikes"])
        assert r.traces.dtype == np.float32 and r.traces.shape == s["traces"].shape
        assert np.array_equal(r.traces, s["traces"].astype(np.float32))
        assert np.array_equal(r.centers, s["centers"]) and r.centers.dtype == np.float64
        assert np.array_equal(r.footprints.dense(), s["fp"])
        assert r.footprints.n() == len(s["cells"])
        assert np.array_equal(r.images["reference"], s["images"]["reference"])
        assert np.array_equal(r.images["binary_map"], s["images"]["binary_map"])
        assert np.array_equal(r.images["brightness_reference"], s["images"]["brightness_reference"])
        assert r.images["reference"].dtype == np.float32 and r.images["binary_map"].dtype == np.uint8
        assert r.images["brightness_reference"].dtype == np.float32
        assert r.cells["source"].tolist() == s["cells"]["source"].tolist()
        assert r.cells["cell_id"].tolist() == s["cells"]["cell_id"].tolist()
    csv = pd.read_csv(os.path.join(out, "cells.csv"))
    assert list(csv.columns) == list(CELLS_DTYPES) and len(csv) == len(s["cells"])


def test_signals_h5_layout_chunks_and_attrs(tmp_path):
    s = _synthetic_fov(np.random.default_rng(3))
    out = str(tmp_path / "fov")
    write_results(out, cells_df=s["cells"], traces=s["traces"], footprints=s["fp"],
                  spikes_df=s["spikes"], images=s["images"], attrs=s["attrs"])
    N, T = s["traces"].shape
    H, W, _ = s["fp"].shape
    with h5py.File(os.path.join(out, "signals.h5"), "r") as f:
        d = f["/traces/raw"]
        assert d.shape == (T, N) and d.dtype == np.float32           # [T, N] on disk
        assert d.chunks == (min(2048, T), min(256, N))
        assert d.compression == "gzip" and d.compression_opts == 4
        assert np.array_equal(np.asarray(d), s["traces"].T.astype(np.float32))
        assert f["/footprints/values"].dtype == np.float32
        assert f["/footprints/pixel"].dtype == np.uint32
        assert f["/footprints/offsets"].dtype == np.uint32 and f["/footprints/offsets"].shape == (N + 1,)
        assert f["/images/reference"].shape == (H, W) and f["/images/reference"].dtype == np.float32
        assert f["/images/binary_map"].shape == (H, W) and f["/images/binary_map"].dtype == np.uint8
        assert f["/images/brightness_reference"].shape == (H, W)
        assert f["/images/brightness_reference"].dtype == np.float32
        a = dict(f.attrs)
        assert a["height"] == H and a["width"] == W and a["n_frames"] == T and a["fps"] == 800.0
        assert a["schema_version"] == SCHEMA_VERSION
        assert a["profile"] == "synthetic" and a["nothing"] == ""
        assert a["n_ap_detected"] == 6 and a["n_ap_localized"] == 5
        assert isinstance(a["pyali_git"], str)                     # '' if git is unavailable
    r = load_results(out)
    assert len(r.attrs["pyali_git"]) in (0, 40)
    assert r.attrs["schema_version"] == SCHEMA_VERSION and r.attrs["profile"] == "synthetic"


def test_zero_cell_fov_writes_every_file_and_loads(tmp_path):
    H, W, T = 32, 48, 120
    out = str(tmp_path / "empty")
    cells = _cells(np.zeros((0, 2)), [], [])
    spikes = pack_spikes(np.zeros((0, 4)), np.zeros((0, 3)), [], [], [])
    images = dict(reference=np.zeros((H, W), np.float32), binary_map=np.zeros((H, W), np.uint8),
                  brightness_reference=np.zeros((H, W), np.float32))
    write_results(out, cells_df=cells, traces=np.zeros((0, T)), footprints=np.zeros((H, W, 0)),
                  spikes_df=spikes, images=images, attrs=dict(fps=800.0))
    for name in ("cells.parquet", "cells.csv", "spikes.parquet", "signals.h5"):
        assert os.path.isfile(os.path.join(out, name)), name
    with h5py.File(os.path.join(out, "signals.h5"), "r") as f:
        assert f["/traces/raw"].shape == (T, 0)
        assert f["/footprints/offsets"].shape == (1,) and f["/footprints/values"].shape == (0,)
        assert f.attrs["n_frames"] == T and f.attrs["height"] == H and f.attrs["width"] == W
    r = load_results(out)
    assert len(r.cells) == 0 and {c: str(t) for c, t in r.cells.dtypes.items()} == CELLS_DTYPES
    assert len(r.spikes) == 0 and {c: str(t) for c, t in r.spikes.dtypes.items()} == SPIKES_DTYPES
    assert r.traces.shape == (0, T) and r.centers.shape == (0, 2)
    assert r.footprints.n() == 0 and r.footprints.dense().shape == (H, W, 0)
    assert r.images["reference"].shape == (H, W)


def test_write_results_rejects_inconsistent_inputs(tmp_path):
    import pytest
    s = _synthetic_fov(np.random.default_rng(4))
    kw = dict(cells_df=s["cells"], traces=s["traces"], footprints=s["fp"],
              spikes_df=s["spikes"], images=s["images"], attrs=s["attrs"])
    with pytest.raises(ValueError):
        write_results(str(tmp_path / "a"), **dict(kw, traces=s["traces"][:-1]))
    with pytest.raises(ValueError):
        write_results(str(tmp_path / "b"), **dict(kw, cells_df=s["cells"].drop(columns="n_spikes")))
