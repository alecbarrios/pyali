"""Cycle E2: per-region work restricted to the frames it reads, as pure tasks in a process pool.

Every new path is checked bit-for-bit against the verbatim pre-E2 code in
``tests/reference_extract.py``.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_cycle_e2.py -q``
"""
import numpy as np
import pytest

from pyali import extract as ex
from pyali import utils
from tests import reference_extract as ref


# --------------------------------------------------------------------------- #
# 1. findpeaks, movmedian_at, region_grow_brightest
# --------------------------------------------------------------------------- #
def _signals(rng, n=60):
    """Random signals with plateaus (quantized values), runs at both ends, NaNs."""
    out = []
    for _ in range(n):
        T = int(rng.integers(1, 200))
        x = np.round(rng.normal(size=T) * rng.choice([1, 2, 4]))           # many ties / plateaus
        if T > 5 and rng.random() < 0.3:
            x[: rng.integers(1, 4)] = x[0]                                  # flat start
        if T > 5 and rng.random() < 0.3:
            x[-rng.integers(1, 4):] = -10.0                                 # low flat end
        if T > 5 and rng.random() < 0.1:
            x[rng.integers(0, T)] = np.nan
        out.append(x)
    out += [np.array([]), np.array([1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 2.0, 2.0, 2.0, 1.0]), np.zeros(10), np.array([0, 1, 1, 0, 2, 0], float)]
    return out


def test_findpeaks_matches_the_loop():
    rng = np.random.default_rng(0)
    n_compared = 0
    for x in _signals(rng):
        try:
            v0, i0 = ref.findpeaks(x)
        except IndexError:                     # the loop indexes past a plateau that rises into the end
            v1, i1 = utils.findpeaks(x)
            assert np.all(i1 < x.size - 1)
            continue
        v1, i1 = utils.findpeaks(x)
        assert np.array_equal(i0, i1), (x, i0, i1)
        assert np.array_equal(v0, v1, equal_nan=True)
        assert i1.dtype.kind == "i"
        n_compared += 1
    assert n_compared > 40


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_movmedian_at_equals_full_filter_rows(dtype):
    rng = np.random.default_rng(1)
    n = 97
    a = rng.normal(size=(n, 5, 6)).astype(dtype)
    a[:, 0, 0] = np.round(a[:, 0, 0])                                       # ties
    full = utils.movmedian_time(a, 8, axis=0)
    frames = np.array([0, 1, 3, 4, 5, 50, 50, 92, 93, 94, 95, 96, 17])     # ends, interior, repeat
    rows = utils.movmedian_support(frames, n, 8)
    got = utils.movmedian_at(a[rows], rows, frames, n, 8)
    assert got.dtype == dtype and got.shape == (frames.size, 5, 6)
    assert np.array_equal(got, full[frames])
    # another window takes the reference path everywhere
    full5 = utils.movmedian_time(a, 5, axis=0)
    rows5 = utils.movmedian_support(frames, n, 5)
    assert np.array_equal(utils.movmedian_at(a[rows5], rows5, frames, n, 5), full5[frames])
    with pytest.raises(KeyError):
        utils.movmedian_at(a[rows[1:]], rows[1:], frames, n, 8)
    assert utils.movmedian_support([], n).size == 0


def _images(rng, n=200):
    out = []
    for _ in range(n):
        hp, wp = int(rng.integers(3, 30)), int(rng.integers(3, 30))
        img = rng.normal(size=(hp, wp))
        if rng.random() < 0.5:
            img = np.round(img * 2) / 2                                     # ties
        if rng.random() < 0.3:
            img[img < 0] = 0.0                                              # flat zero background
            img[0, 0] = -0.0
        out.append(img)
    return out


def test_region_grow_brightest_heap_matches_the_mask_loop():
    rng = np.random.default_rng(2)
    for img in _images(rng):
        for radius, npx in ((6.0, 25), (2.5, 9), (10.0, 60), (6.0, 1)):
            a = ref.region_grow_brightest(img, radius, npx)
            b = ex.region_grow_brightest(img, radius, npx)
            assert np.array_equal(a, b), (img.shape, radius, npx)
    img = rng.normal(size=(9, 9))
    img[3, 3] = np.nan
    assert np.array_equal(ex.region_grow_brightest(img), ref.region_grow_brightest(img))


# --------------------------------------------------------------------------- #
# 2. CorrectedPatch == CorrectedMovie.patch, piece by piece
# --------------------------------------------------------------------------- #
def _corrected_movie(seed=0, T=120, H=40, W=50, dtype=np.uint8, clip=None):
    from pyali import preprocess
    from pyali.movie import CorrectedMovie, estimate_shift, interp_weights
    rng = np.random.default_rng(seed)
    hi = 255 if dtype == np.uint8 else 4000
    raw = rng.integers(0, hi, size=(T + 5, H, W)).astype(dtype)
    ranges = [(1, 20), (50, 70), (100, 115)]
    samples, anchors = preprocess.background_samples(raw[:T], ranges, 3)
    Wt = interp_weights(anchors, T)
    shift, A, _ = estimate_shift(raw[:T], Wt, samples, 5)
    return CorrectedMovie(raw, samples, anchors, shift, A, n_frames=T, saturation_clip=clip)


@pytest.mark.parametrize("dtype, clip", [(np.uint8, None), (np.uint16, 3000.0)])
def test_corrected_patch_pieces_equal_patch(dtype, clip):
    cm = _corrected_movie(dtype=dtype, clip=clip)
    for (r0, r1, c0, c1) in ((3, 30, 4, 31), (0, 40, 0, 50), (33, 40, 41, 50)):
        old = ref.old_patch(cm, r0, r1, c0, c1)                      # (T, hp, wp) float32
        assert np.array_equal(cm.patch(r0, r1, c0, c1), old)
        cp = cm.region(r0, r1, c0, c1)
        assert cp.shape == old.shape
        assert np.array_equal(cp.full(), old)
        f = np.array([0, 5, 5, 119, 64, 7])                           # unsorted, repeated, ends
        assert np.array_equal(cp.frames(f), old[f])
        mask = np.random.default_rng(r0).random(old.shape[1:]) < 0.4
        assert np.array_equal(cp.columns(mask), old[:, mask])
        assert np.array_equal(cp.full(np.float64).astype(np.float32), old)


# --------------------------------------------------------------------------- #
# 3-4. process_region / extract_footprints == the pre-E2 loop; pool == inline
# --------------------------------------------------------------------------- #
import dataclasses                                                          # noqa: E402


def _assert_same_extraction(a, b):
    APs, COMs, fp, ctr, rof, roa, ros, scr, src = a
    APs2, COMs2, fp2, ctr2, rof2, roa2, ros2, scr2, src2 = b
    assert np.array_equal(APs, APs2) and np.array_equal(COMs, COMs2)
    assert np.array_equal(fp.pixel, fp2.pixel) and np.array_equal(fp.offsets, fp2.offsets)
    assert np.array_equal(fp.values, fp2.values) and fp.values.dtype == fp2.values.dtype
    assert np.array_equal(ctr, ctr2)
    for x, y in ((rof, rof2), (roa, roa2), (ros, ros2), (scr, scr2)):
        assert np.array_equal(x, y)
    assert list(src) == list(src2)


def _fake_region(r0, c0, h, w):
    """A rectangular region in segmentation's dict layout (1-indexed [col, row])."""
    rr, cc = np.mgrid[r0:r0 + h, c0:c0 + w]
    pixel_list = np.column_stack([cc.ravel(order="F") + 1, rr.ravel(order="F") + 1]).astype(float)
    return {"Centroid": np.array([c0 + (w - 1) / 2 + 1.0, r0 + (h - 1) / 2 + 1.0]),
            "BoundingBox": np.array([c0 + 0.5, r0 + 0.5, float(w), float(h)]),
            "PixelList": pixel_list}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from pyali.movie import corrected_movie
    from pyali.pipeline import process_fov
    from tests.synthetic import make_movie, write_fov
    d = tmp_path_factory.mktemp("e2")
    data = make_movie(seed=3)
    write_fov(d, data["movie"])
    p = data["params"]
    out = process_fov(str(d), out_dir=None, p=p, save=False, verbose=False)
    raw = np.asarray(data["movie"]).astype("<u2")
    T = raw.shape[0] - p.truncate_last
    cm, _ = corrected_movie(raw, p, n_frames=T)
    regions = list(out["regions"])
    return dict(p=p, cm=cm, regions=regions, spf=[r["PixelList"] for r in regions], out=out)


def test_extract_footprints_matches_reference_inline_and_pooled(synth):
    p, cm, regions, spf = synth["p"], synth["cm"], synth["regions"], synth["spf"]
    H, W = p.nrow, p.ncol
    expect = ref.extract_footprints(cm, regions, spf, p, H, W)
    assert len(expect[0]) > 0 and expect[2].n() > 0                       # spikes and cells exist
    _assert_same_extraction(expect, ex.extract_footprints(cm, regions, spf, p, H, W, workers=1))
    _assert_same_extraction(expect, ex.extract_footprints(cm, regions, spf, p, H, W, workers=3))
    # and the pipeline itself (which now runs this code) produced the same footprints
    assert np.array_equal(synth["out"]["footprints"].values, expect[2].values)


def test_dense_movie_path_matches_reference(synth):
    p, regions, spf = synth["p"], synth["regions"], synth["spf"]
    dense = synth["cm"].materialize(np.float32)
    expect = ref.extract_footprints(dense, regions, spf, p, p.nrow, p.ncol)
    _assert_same_extraction(expect, ex.extract_footprints(dense, regions, spf, p, p.nrow, p.ncol))


def test_fallback_paths_match_reference(synth):
    p, cm = synth["p"], synth["cm"]
    H, W = p.nrow, p.ncol
    # all_noise everywhere: DBSCAN can never form a cluster
    q = dataclasses.replace(p, dbscan_min_pts=10 ** 6)
    regions, spf = synth["regions"], synth["spf"]
    expect = ref.extract_footprints(cm, regions, spf, q, H, W)
    assert set(expect[8]) == {"all_noise"}
    _assert_same_extraction(expect, ex.extract_footprints(cm, regions, spf, q, H, W, workers=2))
    # no_ap: a region over a constant patch of a dense movie has a flat trace
    dense = cm.materialize(np.float32)
    dense[:, 80:96, 80:96] = 7.0
    extra = regions + [_fake_region(84, 84, 6, 5)]
    expect = ref.extract_footprints(dense, extra, [r["PixelList"] for r in extra], p, H, W)
    assert expect[8][-1] == "no_ap"
    _assert_same_extraction(expect, ex.extract_footprints(dense, extra, [r["PixelList"] for r in extra],
                                                          p, H, W, workers=2))


def test_worker_errors_propagate_and_empty_inputs(synth):
    p, cm = synth["p"], synth["cm"]
    bad = synth["regions"][:3] + [{"Centroid": np.array([5.0, 5.0]),
                                   "BoundingBox": np.array([1.5, 1.5, 3.0, 3.0]), "PixelList": None}]
    with pytest.raises(Exception):
        ex.extract_footprints(cm, bad, [r["PixelList"] for r in bad], p, p.nrow, p.ncol, workers=2)
    assert ex._POOL_CONTEXT is None                                         # cleaned up after an error
    out = ex.extract_footprints(cm, [], [], p, p.nrow, p.ncol, workers=4)
    assert out[0].shape == (0, 4) and out[2].n() == 0 and out[8] == []
    assert ex.resolve_workers(None) >= 1 and ex.resolve_workers(0) == ex.resolve_workers(None)
    assert ex.resolve_workers(3) == 3


def test_filtered_frames_equal_full_filter_rows(synth):
    cm = synth["cm"]
    src = cm.region(10, 37, 12, 39)
    full = ex.temporal_filter_patch(src.full(), 8)
    T = full.shape[0]
    full[:4] = 0.0
    full[T - 4:] = 0.0
    frames = np.array([0, 3, 4, 5, 200, 200, T - 5, T - 4, T - 1, 77])
    fs = ex.filtered_frames(src, frames, 8, 4)
    assert fs.shape == full.shape
    assert np.array_equal(fs[frames], full[frames]) and np.array_equal(fs[77], full[77])
    with pytest.raises(KeyError):
        fs[6]
