"""Unit tests for the Cycle A detection/localization fixes in ``pyali.extract``:

* ``dedup_close_peaks``: keep the largest-amplitude AP within a ``min_interval`` window.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_cycle_a.py -q``
"""
import numpy as np

import pyali.extract as extract
from pyali.extract import (dedup_close_peaks, detect_region_aps, region_grow_brightest,
                           _weighted_com, com_via_svd, cluster_footprints)
from pyali.pipeline import process_fov
from tests.synthetic import make_movie, write_fov


def _blob(H, W, r, c, sigma=2.0):
    rr, cc = np.mgrid[:H, :W]
    return np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2.0 * sigma ** 2))


# --------------------------------------------------------------------------- #
# A1. dedup_close_peaks
# --------------------------------------------------------------------------- #
def _aps(frames, amps):
    frames = np.asarray(frames, float)
    amps = np.asarray(amps, float)
    return np.column_stack([np.full_like(frames, 5.0), np.full_like(frames, 7.0), frames, amps])


def test_dedup_burst_keeps_the_largest_member():
    out = dedup_close_peaks(_aps([100, 110, 120, 130], [1, 3, 2, 1]), 25)
    assert out[:, 2].tolist() == [110]
    assert out[0, 3] == 3 and out[0, 0] == 5 and out[0, 1] == 7       # whole row preserved


def test_dedup_keeps_well_separated_spikes():
    out = dedup_close_peaks(_aps([100, 200, 300], [1, 1, 1]), 25)
    assert out[:, 2].tolist() == [100, 200, 300]


def test_dedup_adjacent_runs_resolved_by_amplitude():
    # runs anchored at the first member: [100, 124] then [148]; 124 and 148 are only 24 apart
    out = dedup_close_peaks(_aps([100, 124, 148], [1, 2, 3]), 25)
    assert out[:, 2].tolist() == [148]
    out = dedup_close_peaks(_aps([100, 124, 148], [3, 2, 1]), 25)
    assert out[:, 2].tolist() == [100, 148]


def test_dedup_empty_passthrough():
    empty = np.zeros((0, 4))
    assert dedup_close_peaks(empty, 25) is empty


def test_dedup_unsorted_input_and_pairwise_spacing():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        frames = rng.integers(10, 800, size=n)
        amps = rng.random(n)
        ap = _aps(frames, amps)
        out = dedup_close_peaks(ap, 25)
        assert len(out) >= 1
        assert np.all(np.diff(out[:, 2]) >= 25)                       # sorted and spaced
        rows = {tuple(r) for r in ap.tolist()}
        assert all(tuple(r) in rows for r in out.tolist())            # subset of the input
        # every dropped spike lost to a spike of >= amplitude within 25 frames (its run max, or
        # the spike that beat its run max in the second pass); the dropped-to chain is what the
        # two-pass rule guarantees, not that a kept spike lies within 25 frames of every input
        kept = {tuple(r) for r in out.tolist()}
        for r in ap.tolist():
            if tuple(r) in kept:
                continue
            others = ap[(np.abs(ap[:, 2] - r[2]) < 25) & (ap[:, 3] >= r[3]) & (ap[:, 2] != r[2])]
            assert len(others) or np.sum(ap[:, 2] == r[2]) > 1, r


# --------------------------------------------------------------------------- #
# A2. detect_region_aps: coarse location restricted to the mask
# --------------------------------------------------------------------------- #
def test_coarse_location_ignores_bright_pixel_outside_mask():
    rng = np.random.default_rng(0)
    T, Hp, Wp = 100, 12, 12
    pm = np.zeros((T, Hp, Wp))
    pm[10:40] = 1e-3 * rng.normal(size=(30, Hp, Wp))                  # baseline noise frames
    sel = np.zeros((Hp, Wp))
    sel[4:8, 4:8] = 1.0
    pm[50, 4:8, 4:8] = 5.0                                            # the AP, inside the region
    pm[50, 5, 6] = 6.0                                                # brightest in-mask pixel
    pm[50, 0, 0] = 50.0                                               # much brighter, outside
    pm[50, 11, 2] = 40.0
    # the raw patch is the negative of the filtered one here: a spike is a dip in raw intensity
    ap, _ = detect_region_aps(-pm, sel, pm, (100, 200), 32, 16, 4.0)
    assert ap.shape == (1, 4)
    assert ap[0, 2] == 50
    assert ap[0, :2].tolist() == [105, 206]                           # [5, 6] + origin, not [0, 0]


# --------------------------------------------------------------------------- #
# A3. region_grow_brightest
# --------------------------------------------------------------------------- #
def test_region_grow_blob_count_radius_and_seed():
    rng = np.random.default_rng(1)
    img = 10.0 * _blob(28, 28, 10, 12) + 0.01 * rng.normal(size=(28, 28))
    picked = region_grow_brightest(img, radius=6.0, n_pixels=25)
    assert picked.shape == (25, 2)
    seed = np.unravel_index(int(np.argmax(img)), img.shape)
    assert tuple(picked[0]) == seed == (10, 12)
    d2 = (picked[:, 0] - seed[0]) ** 2 + (picked[:, 1] - seed[1]) ** 2
    assert d2.max() <= 36.0
    assert len({tuple(p) for p in picked}) == 25                      # no repeats


def test_region_grow_stops_on_empty_queue_without_raising():
    assert region_grow_brightest(np.array([[3.0]]), 6.0, 25).tolist() == [[0, 0]]
    island = np.zeros((9, 9))
    island[4, 4] = 1.0
    assert region_grow_brightest(island, radius=0.5, n_pixels=25).tolist() == [[4, 4]]
    picked = region_grow_brightest(island, radius=6.0, n_pixels=25)   # zeros are queued and picked
    assert tuple(picked[0]) == (4, 4) and 1 <= len(picked) <= 25
    corner = np.zeros((28, 28))
    corner[0, 0] = 5.0                                                # quarter disc of r=2: 6 px
    assert len(region_grow_brightest(corner, radius=2.0, n_pixels=25)) == 6


def test_region_grow_picks_the_max_inside_the_queue_not_anywhere():
    img = np.zeros((20, 20))
    img[10, 10] = 9.0                                                 # seed
    img[10, 11] = 3.0                                                 # queued neighbour
    img[2, 2] = 3.0                                                   # same value, far away, earlier in F-order
    picked = region_grow_brightest(img, radius=6.0, n_pixels=2)
    assert picked.tolist() == [[10, 10], [10, 11]]


# --------------------------------------------------------------------------- #
# A4. _weighted_com
# --------------------------------------------------------------------------- #
def test_weighted_com_nan_on_non_positive_mass():
    rows, cols = np.arange(1, 6), np.arange(1, 6)
    picked = np.array([[2, 2], [2, 3], [3, 2]])
    r, c = _weighted_com(-np.ones((5, 5)), picked, rows, cols)
    assert np.isnan(r) and np.isnan(c)
    r, c = _weighted_com(np.zeros((5, 5)), picked, rows, cols)
    assert np.isnan(r) and np.isnan(c)


def test_weighted_com_symmetric_blob_is_centred():
    img = _blob(28, 28, 10, 12)
    rows, cols = np.arange(101, 129), np.arange(201, 229)             # absolute 1-indexed
    picked = region_grow_brightest(img, radius=6.0, n_pixels=25)
    r, c = _weighted_com(img, picked, rows, cols)
    assert abs(r - 111.0) <= 0.05 and abs(c - 213.0) <= 0.05, (r, c)
    # weights come from the picked pixels only: a bright pixel in their row x column hull but
    # not itself picked must not move the centroid
    picked21 = region_grow_brightest(img, radius=6.0, n_pixels=21)  # diamond d^2 <= 5, 5x5 hull
    hull_only = {(pr, pc) for pr in picked21[:, 0] for pc in picked21[:, 1]} - {tuple(p) for p in picked21}
    assert len(hull_only) == 4
    r21, c21 = _weighted_com(img, picked21, rows, cols)
    img2 = img.copy()
    for pr, pc in hull_only:
        img2[pr, pc] = 50.0
    assert _weighted_com(img2, picked21, rows, cols) == (r21, c21)
    # negatives are clipped, not NaN-propagated
    img3 = img.copy()
    img3[picked[-1, 0], picked[-1, 1]] = -5.0
    r3, c3 = _weighted_com(img3, picked, rows, cols)
    assert np.isfinite(r3) and np.isfinite(c3)


# --------------------------------------------------------------------------- #
# A5. com_via_svd: mask + outside-mask rejection + empty-COM fallback
# --------------------------------------------------------------------------- #
def _ap_patch_movie(T=80, Hp=20, Wp=20, times=(20, 40, 60), seed=0):
    """Patch movie with low noise (so the SVD probe pixels are never exactly zero) and a
    7-frame window around each AP frame to paint on."""
    rng = np.random.default_rng(seed)
    pm = 1e-3 * rng.normal(size=(T, Hp, Wp))
    return pm, np.asarray(times, int)


def test_com_via_svd_rejects_com_outside_the_mask():
    pm, times = _ap_patch_movie()
    sel = np.zeros((20, 20))
    sel[8:13, 8:13] = 1.0
    sel[10, 10] = 0.0                                                 # ring: centroid lands in the hole
    for t in times:
        pm[t - 3:t + 4] += sel[None] * 1.0
    rows, cols = np.arange(1, 21), np.arange(1, 21)
    coms = com_via_svd(pm, times, rows, cols, sel, rank=15, radius=6.0, n_pixels=25)
    assert coms.shape == (0, 3)


def test_com_via_svd_ignores_bright_spot_outside_the_mask():
    pm, times = _ap_patch_movie()
    sel = np.zeros((20, 20))
    sel[2:6, 2:6] = 1.0
    for t in times:
        pm[t - 3:t + 4, 14:18, 14:18] += 10.0                         # the only bright thing: outside
        pm[t - 3:t + 4, 2:6, 2:6] -= 0.5                              # region itself has no positive signal
    coms = com_via_svd(pm, times, np.arange(1, 21), np.arange(1, 21), sel, 15, 6.0, 25)
    assert coms.shape == (0, 3)


def test_com_via_svd_keeps_in_mask_coms_in_absolute_coordinates():
    pm, times = _ap_patch_movie()
    sel = np.zeros((20, 20))
    sel[6:15, 6:15] = 1.0
    blob = _blob(20, 20, 10, 10, sigma=1.5)
    for t in times:
        pm[t - 3:t + 4] += blob[None] * 5.0
    rows, cols = np.arange(101, 121), np.arange(201, 221)             # absolute 1-indexed
    coms = com_via_svd(pm, times, rows, cols, sel, 15, 6.0, 25)
    assert coms.shape == (3, 3)
    assert coms[:, 2].tolist() == times.tolist()
    assert np.abs(coms[:, 0] - 111.0).max() < 0.1 and np.abs(coms[:, 1] - 211.0).max() < 0.1


def test_extract_footprints_falls_back_when_no_com_survives(tmp_path, monkeypatch):
    data = make_movie(seed=0)
    write_fov(tmp_path, data["movie"])
    monkeypatch.setattr(extract, "com_via_svd", lambda *a, **k: np.zeros((0, 3)))
    out = process_fov(str(tmp_path), out_dir=None, p=data["params"], save=False, verbose=False)
    assert len(out["regions"]) > 0
    assert np.asarray(out["COMs"]).shape == (0, 3)
    assert out["footprints"].dense().shape[2] == len(out["regions"])   # one fallback plane each
    assert np.isfinite(np.asarray(out["cell_traces"], float)).all()
    assert out["region_of_footprint"].tolist() == list(range(len(out["regions"])))
    assert out["region_of_ap"].shape == (len(out["APs"]),)
    had_ap = set(out["region_of_ap"].tolist())
    assert out["source_of_footprint"] == ["no_com" if c in had_ap else "no_ap"
                                          for c in range(len(out["regions"]))]
    assert out["region_of_spike"].shape == (0,) and out["spike_cell_row"].shape == (0,)


# --------------------------------------------------------------------------- #
# A6. cluster_footprints: negatives clipped
# --------------------------------------------------------------------------- #
def test_cluster_footprints_have_no_negative_weights():
    rng = np.random.default_rng(0)
    pm = rng.normal(size=(50, 20, 20))                                # plenty of negative pixels
    pm[10:13, 9, 9] += 20.0                                           # the cell: positive at AP frames
    sel = np.zeros((20, 20))
    sel[5:15, 5:15] = 1.0
    rows, cols = np.arange(31, 51), np.arange(41, 61)
    planes, centers = cluster_footprints(pm, [np.array([10, 11, 12])], sel, rows, cols, 80, 80)
    assert len(planes) == 1 and planes[0].shape == (80, 80)
    assert planes[0].min() >= 0.0
    assert centers[0] == [9 + 31, 9 + 41]                             # center pixel unaffected
    assert planes[0][9 + 30, 9 + 40] > 0 and np.count_nonzero(planes[0]) > 1


def test_pipeline_footprints_are_non_negative(tmp_path):
    data = make_movie(seed=0)
    write_fov(tmp_path, data["movie"])
    out = process_fov(str(tmp_path), out_dir=None, p=data["params"], save=False, verbose=False)
    fp = out["footprints"].dense()
    assert fp.shape[2] > 0
    assert fp.min() >= 0.0
