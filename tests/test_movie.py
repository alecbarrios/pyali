"""``pyali.movie``: lazily corrected raw movie against the dense in-place reference.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_movie.py -q``
"""
import numpy as np
import pytest

from pyali import movie as mv
from pyali import preprocess
from pyali.params import Params
from tests import reference_dense as ref

T_RAW, H, W = 60, 24, 30
TRUNC = 5
T = T_RAW - TRUNC
RANGES = [(1, 10), (30, 40), (50, 55)]            # anchors 6, 35, 53 (1-indexed); frames 1-5 and 54-55 clamp
DISK = 3


def _raw(seed=0, dtype=np.uint8):
    """8-bit movie: smooth illumination + two blobs that blink + slow drift + a sub-pixel wobble."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    base = 60 + 30 * np.exp(-((yy - 12) ** 2 + (xx - 15) ** 2) / 200.0)
    blob = 40 * np.exp(-((yy - 8) ** 2 + (xx - 9) ** 2) / 4.0) + 35 * np.exp(-((yy - 16) ** 2 + (xx - 21) ** 2) / 5.0)
    out = np.empty((T_RAW, H, W), np.float64)
    for t in range(T_RAW):
        dx, dy = 0.3 * np.sin(t / 7.0), 0.2 * np.cos(t / 5.0)
        img = base * (1 - 0.002 * t) + blob * (1 + 0.5 * (t % 17 == 3))
        img = img + dx * np.gradient(img, axis=1) + dy * np.gradient(img, axis=0)   # first-order shift
        out[t] = img + rng.normal(0, 2.0, (H, W))
    return np.clip(np.round(out), 0, 255).astype(dtype)


@pytest.fixture(scope="module")
def dense():
    """The pre-E1 dense path: float32 movie, in-place background then motion correction."""
    raw = _raw()
    m = raw[:T].astype(np.float32)
    m, samples, anchors = ref.adaptive_background(m, RANGES, DISK)
    m, shift, A, f0 = ref.motion_correct(m, TRUNC)
    return dict(raw=raw, corrected=m, samples=samples, anchors=anchors, shift=shift, A=A, f0=f0)


@pytest.fixture(scope="module")
def lazy(dense):
    p = Params(nrow=H, ncol=W, bkg_ranges=RANGES, disk_radius=DISK, truncate_last=TRUNC,
               saturation_clip=None)
    cm, info = mv.corrected_movie(dense["raw"], p, n_frames=T)
    return dict(cm=cm, **info)


def test_background_samples_on_integer_frames_match_the_float32_path(dense):
    s_int, a_int = preprocess.background_samples(dense["raw"][:T], RANGES, DISK)
    assert np.array_equal(a_int, dense["anchors"])
    assert np.array_equal(s_int, dense["samples"])                    # bit-identical: exact integer means


def test_interp_weights_match_reference_brackets(dense):
    Wt = mv.interp_weights(dense["anchors"], T)
    assert Wt.shape == (T, 3) and np.allclose(Wt.sum(axis=1), 1.0)
    for t in range(T):
        ib, ia, w = preprocess._interp_weight(t + 1, dense["anchors"])
        row = np.zeros(3); row[ib] += 1 - w; row[ia] += w
        assert np.allclose(Wt[t], row)
    assert Wt[0, 0] == 1.0 and Wt[T - 1, -1] == 1.0                   # clamped ends


def test_shift_and_reference_match_dense_motion_correct(dense, lazy):
    assert lazy["A"].shape == (H * W, 2)
    assert np.abs(lazy["f0"] - dense["f0"]).max() < 1e-3             # float32 vs float64 means
    assert np.abs(lazy["shift"] - dense["shift"]).max() < 1e-5        # px


def test_materialized_lazy_movie_matches_dense_to_float32_rounding(dense, lazy):
    cm = lazy["cm"]
    assert cm.shape == (T, H, W) and len(cm) == T
    full = cm.materialize()
    d = np.abs(full - dense["corrected"].astype(np.float64))
    assert d.max() < 1e-3 and np.median(d) < 1e-4
    assert np.array_equal(cm.materialize(np.float32), full.astype(np.float32))


def test_patch_gather_and_frames_are_consistent(dense, lazy):
    cm = lazy["cm"]
    full = cm.materialize()
    blk = cm.patch(5, 14, 7, 20, dtype=np.float64)
    assert blk.shape == (T, 9, 13) and np.allclose(blk, full[:, 5:14, 7:20], rtol=0, atol=1e-10)
    blk2 = cm.patch(5, 14, 7, 20, 10, 23, dtype=np.float64)
    assert np.allclose(blk2, full[10:23, 5:14, 7:20], rtol=0, atol=1e-10)
    blk32 = cm.patch(5, 14, 7, 20)                                    # default: float32 working precision
    assert blk32.dtype == np.float32 and np.array_equal(blk32, blk.astype(np.float32))
    pix = np.array([0, 17, 3 * W + 5, H * W - 1])
    g = cm.gather(pix, 4, 30)
    assert g.shape == (26, 4) and np.allclose(g, full[4:30].reshape(26, -1)[:, pix], rtol=0, atol=1e-10)
    assert np.allclose(cm.gather(pix), full.reshape(T, -1)[:, pix], rtol=0, atol=1e-10)
    f = cm.frames(2, 7)
    assert f.dtype == np.float32 and np.array_equal(f, dense["raw"][2:7].astype(np.float32))
    assert cm.frames(T - 2, T + 50).shape == (2, H, W)                 # clipped to the analysis frames
    assert cm.patch(0, H, 0, W, T - 3).shape == (3, H, W)


def test_saturation_clip_zeroes_corrected_values(dense):
    raw = dense["raw"].astype(np.uint16) * 300                         # 16-bit-like values
    p = Params(nrow=H, ncol=W, bkg_ranges=RANGES, disk_radius=DISK, truncate_last=TRUNC,
               saturation_clip=2000.0)
    cm, _ = mv.corrected_movie(raw, p, n_frames=T)
    full = cm.materialize()
    assert (full <= 2000.0).all() and (full == 0).any()
    pix = np.arange(0, H * W, 7)
    assert np.array_equal(cm.gather(pix, 0, 9), full[:9].reshape(9, -1)[:, pix])


def test_constructor_checks(dense, lazy):
    with pytest.raises(ValueError):
        mv.CorrectedMovie(dense["raw"], lazy["samples"], lazy["anchors"], lazy["shift"], lazy["A"],
                          n_frames=T_RAW + 1)
    with pytest.raises(ValueError):
        mv.CorrectedMovie(dense["raw"], lazy["samples"], lazy["anchors"], lazy["shift"][:, :10],
                          lazy["A"], n_frames=T)
