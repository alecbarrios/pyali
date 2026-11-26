"""``scripts/qc_footprints.py``: footprint-overlay + trace QC across SNR percentiles.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_qc_footprints.py -q``
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _ROOT)
import qc_footprints as qc                                                 # noqa: E402
from pyali.metadata import Identity                                        # noqa: E402
from pyali.results import load_results, make_cells, pack_spikes, write_results   # noqa: E402

H, W, T, FPS = 40, 50, 2000, 800.0
AMPS = np.linspace(5.0, 40.0, 12)            # spike amplitudes in noise-sigma units, one per cell


def _blob(r, c, sigma=1.5):
    rr, cc = np.mgrid[:H, :W]
    return np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma ** 2))


def _results_dir(tmp_path, with_movie=True, silent_last=True):
    """Synthetic result directory: 12 cells of increasing spike amplitude (the last one silent
    when ``silent_last``), Gaussian footprints, a spikes table, and an 8-bit movie whose blobs
    brighten at the spike frames."""
    rng = np.random.default_rng(0)
    N = len(AMPS)
    centers0 = [(5 + 3 * k, 4 + 4 * k) for k in range(N)]                 # 0-indexed, spread
    traces = rng.normal(size=(N, T))
    fp = np.zeros((H, W, N))
    movie = np.full((T, H, W), 40, np.float64) + rng.normal(0, 1, (T, H, W))
    APs, region_of_ap, COMs, region_of_spike, cell_row = [], [], [], [], []
    n_spikes = np.zeros(N, int)
    for k, (r, c) in enumerate(centers0):
        fp[:, :, k] = _blob(r, c)
        if silent_last and k == N - 1:
            continue
        frames = np.arange(60 + 17 * k, T - 60, 160)                      # >= 25 apart, >= lag
        traces[k, frames] += AMPS[k]
        movie[frames] += 30.0 * _blob(r, c)[None]
        for f in frames:
            APs.append([r + 1, c + 1, f, AMPS[k]]); region_of_ap.append(k)
            COMs.append([r + 1, c + 1, f]); region_of_spike.append(k); cell_row.append(k)
        n_spikes[k] = len(frames)
    centers = np.array(centers0, float) + 1.0                              # 1-indexed
    source = ["spiking"] * N
    if silent_last:
        source[-1] = "no_ap"
    cells = make_cells(centers, np.arange(N), n_spikes, source, np.ones(N, np.float32),
                       Identity("exp", 4, "A2", 41, fov_index=0, batch="443-1"))
    spikes = pack_spikes(np.array(APs), np.array(COMs), np.array(cell_row),
                         np.array(region_of_spike), np.array(region_of_ap))
    ref = (40 + 30 * fp.sum(axis=2)).astype(np.float32)
    images = dict(reference=ref, binary_map=(fp.sum(axis=2) > 0.3).astype(np.uint8),
                  brightness_reference=ref)
    out = str(tmp_path / "res")
    write_results(out, cells_df=cells, traces=traces.astype(np.float32), footprints=fp,
                  spikes_df=spikes, images=images,
                  attrs=dict(fps=FPS, profile="synthetic", fov_id="exp_P04_A2_F041",
                             focus_score=2.0e-3))
    fov_dir = str(tmp_path / "fov")
    meta = dict(fov_dir=fov_dir, params=dict(nrow=H, ncol=W, read_dtype="uint8"))
    if with_movie:
        os.makedirs(fov_dir)
        np.clip(movie, 0, 255).astype("<u1").tofile(os.path.join(fov_dir, "frames1.bin"))
    with open(os.path.join(out, "fov_metadata.json"), "w") as f:
        json.dump(meta, f)
    return out, np.array(centers0), movie


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_percentiles_spread():
    assert qc.percentiles(10) == pytest.approx([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    assert qc.percentiles(1) == [50.0]


def test_cell_table_and_select_cells_follow_spike_amplitude(tmp_path):
    out, _, _ = _results_dir(tmp_path)
    R = load_results(out)
    df = qc.cell_table(R)
    assert list(df["trace_row"]) == list(range(len(AMPS)))
    assert (df["n_px"] > 0).all()
    assert (df["eff_px"] > 1).all() and (df["eff_px"] <= df["n_px"]).all()   # Gaussian blobs: 1 < eff < n_px
    snr = df["snr_median"].to_numpy(float)[:-1]                            # the 11 spiking cells
    assert np.all(np.diff(snr) > 0)                                         # monotone in amplitude
    assert (df["qc_spikes"].to_numpy()[:-1] >= 3).all()
    picked = qc.select_cells(df, n=10, min_spikes=3)
    assert 1 <= len(picked) <= 10
    assert picked["cell_id"].is_unique
    assert np.all(np.diff(picked["percentile"].to_numpy()) > 0)
    assert np.all(np.diff(picked["snr_median"].to_numpy()) >= 0)           # ordered with percentile
    assert "no_ap" not in set(picked["source"])                            # the silent cell is out
    assert picked["percentile"].iloc[0] == 5 and picked["percentile"].iloc[-1] == 95
    # a pool smaller than n gives fewer, still unique, picks
    small = qc.select_cells(df.iloc[:3], n=10)
    assert len(small) == 3 and small["cell_id"].is_unique
    assert len(qc.select_cells(df.iloc[-1:], n=10)) == 0                   # empty pool


# --------------------------------------------------------------------------- #
# crops
# --------------------------------------------------------------------------- #
def test_crop_window_centres_and_clamps():
    assert qc.crop_window((21.0, 26.0), H, W, 10) == (15, 20)               # 1-indexed centre -> 0-indexed
    assert qc.crop_window((1.0, 1.0), H, W, 10) == (0, 0)                   # top-left corner
    assert qc.crop_window((40.0, 50.0), H, W, 10) == (30, 40)               # bottom-right corner
    assert qc.crop_window((20.0, 25.0), H, W, 200) == (0, 4)                # oversize -> 40 x 40, centred in the wide axis


def test_sta_crop_peaks_at_the_blinking_blob(tmp_path):
    out, centers0, movie = _results_dir(tmp_path)
    R = load_results(out)
    k = 5
    r, c = centers0[k]
    frames = R.spikes[R.spikes["cell_row"] == k]["frame"].to_numpy()
    r0, c0 = qc.crop_window(R.centers[k], H, W, 16)
    sta = qc.sta_crop(movie, frames, r0, c0, 16, lag=8, sign=1.0)          # the synthetic blob brightens
    assert sta.shape == (16, 16)
    pr, pc = np.unravel_index(np.argmax(sta), sta.shape)
    assert (pr + r0, pc + c0) == (r, c)
    assert sta.max() > 20                                                   # ~30 planted minus noise
    # the default sign shows a fluorescence *drop* as positive (the pipeline's convention)
    assert np.allclose(qc.sta_crop(movie, frames, r0, c0, 16, lag=8), -sta)
    # frames before ``lag`` or past the end are dropped; none usable -> None
    assert qc.sta_crop(movie, [3, T + 5], r0, c0, 16, lag=8) is None
    # max_frames keeps the largest amplitudes
    amps = np.arange(len(frames), dtype=float)
    few = qc.sta_crop(movie, frames, r0, c0, 16, lag=8, max_frames=2, amplitudes=amps, sign=1.0)
    assert few.shape == (16, 16) and abs(few.max() - sta.max()) < 15


def test_find_movie_uses_metadata_or_override(tmp_path):
    out, _, _ = _results_dir(tmp_path)
    meta = json.load(open(os.path.join(out, "fov_metadata.json")))
    mm = qc.find_movie(out, meta)
    assert mm is not None and mm.shape == (T, H, W) and mm.dtype == np.uint8
    assert qc.find_movie(out, meta, fov_dir=str(tmp_path / "nowhere")) is None
    assert qc.find_movie(out, {}) is None


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_run_writes_png_and_csv(tmp_path, capsys):
    out, _, _ = _results_dir(tmp_path)
    picked = qc.run(out, n=10, crop=16, dpi=60)
    png = os.path.join(out, qc.QC_DIR, qc.PNG)
    csv_path = os.path.join(out, qc.QC_DIR, qc.CSV)
    assert os.path.getsize(png) > 1000
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == qc.CSV_COLUMNS
    assert len(rows) == len(picked) and 1 <= len(rows) <= 10
    assert rows[0]["percentile"] == "5.0" and rows[-1]["percentile"] == "95.0"
    assert "movie yes" in capsys.readouterr().out


def test_main_without_movie(tmp_path, capsys):
    out, _, _ = _results_dir(tmp_path, with_movie=False)
    assert qc.main([out, "--n", "4", "--crop", "16", "--dpi", "50"]) == 0   # movie missing -> no STA
    assert "movie no" in capsys.readouterr().out
    assert qc.main([out, "--no-sta", "--n", "4", "--crop", "16", "--dpi", "50"]) == 0
    with open(os.path.join(out, qc.QC_DIR, qc.CSV)) as f:
        assert len(list(csv.DictReader(f))) == 4


# --------------------------------------------------------------------------- #
# spike map, named cells, before/after comparison (Cycle D2)
# --------------------------------------------------------------------------- #
def test_named_cells_and_spike_map(tmp_path):
    out, _, _ = _results_dir(tmp_path)
    R = load_results(out)
    df = qc.cell_table(R)
    picked = qc.pick_named(df, ["C0003", "exp_P04_A2_F041_C0005", "1"])
    assert picked["cell_id"].str[-5:].tolist() == ["C0003", "C0005", "C0002"]
    assert np.isnan(picked["percentile"]).all()
    with pytest.raises(KeyError):
        qc.pick_named(df, ["C9999"])
    # spikes_in_crop fills kept / cluster for older results and maps COMs into the crop
    sp = qc.spikes_in_crop(R, 0, 0, 50)                                       # covers the 40 x 50 frame
    assert {"x", "y", "kept", "cluster"} <= set(sp.columns) and len(sp) == len(R.spikes)
    assert np.allclose(sp["y"], R.spikes["row"] - 1.0) and np.allclose(sp["x"], R.spikes["col"] - 1.0)
    rows = qc.cells_in_crop(R, 0, 0, 20, exclude=[0])
    assert 0 not in rows and all(R.centers[k][0] - 1 < 20 and R.centers[k][1] - 1 < 20 for k in rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cmap = qc.draw_spike_map(ax, R, [3], 0, 0, 40)
    assert cmap[3] == qc.OWN_COLOUR and len(cmap) >= 2
    plt.close(fig)
    assert qc.main([out, "--cells", "C0003,C0005", "--crop", "16", "--dpi", "50", "--no-sta"]) == 0
    assert os.path.isfile(os.path.join(out, qc.QC_DIR, "footprints_traces_cells.png"))


def test_qc_compare_writes_png_and_csv(tmp_path):
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import qc_compare
    before, _, _ = _results_dir(tmp_path / "b")
    after, _, _ = _results_dir(tmp_path / "a")
    tab = qc_compare.run(before, after, cells=["C0003"], n=2, radius=6.0, crop=16, label="t", dpi=50)
    assert os.path.getsize(os.path.join(after, qc.QC_DIR, "compare_t.png")) > 1000
    assert list(tab.columns) == qc_compare.CSV_COLUMNS
    first = tab[tab["before_cell"].str.endswith("C0003")]
    assert (first["after_cell"].str.endswith("C0003")).any() and first["distance_px"].min() == 0.0
