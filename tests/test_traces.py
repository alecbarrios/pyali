"""``pyali.traces``: sparse normal-equation trace solve against the dense pre-E1 reference.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_traces.py -q``
"""
import dataclasses

import numpy as np
import pytest

from pyali import traces as tr
from pyali.params import Params
from pyali.results import SparseBuilder, SparseFootprints
from tests import reference_dense as ref

H, W, T = 20, 25, 97


def _footprints(rng, N=7, dup=True, empty=True):
    """Compact positive blobs; cell 3 duplicates cell 1 and cell 5 is empty when requested."""
    fp = np.zeros((H, W, N))
    for k in range(N):
        r, c = rng.integers(3, H - 3), rng.integers(3, W - 3)
        rr, cc = np.mgrid[r - 2:r + 3, c - 2:c + 3]
        fp[rr, cc, k] = rng.uniform(0.2, 1.0, rr.shape)
    if dup:
        fp[:, :, 3] = fp[:, :, 1]
    if empty:
        fp[:, :, 5] = 0.0
    b = SparseBuilder(H, W)
    for k in range(N):
        b.add_plane(fp[:, :, k])
    return fp, b.build()


@pytest.fixture
def data():
    rng = np.random.default_rng(11)
    fp, sp = _footprints(rng)
    movie = rng.normal(100.0, 5.0, (T, H, W)).astype(np.float32)
    return dict(fp=fp, sp=sp, movie=movie, rng=rng)


# --------------------------------------------------------------------------- #
# operator primitives
# --------------------------------------------------------------------------- #
def test_operator_gram_and_project_match_dense(data):
    fp, sp, movie = data["fp"], data["sp"], data["movie"]
    op = tr.FootprintOperator(sp)
    F = fp.reshape(H * W, -1)
    M = movie.reshape(T, H * W).astype(np.float64)
    assert op.N == fp.shape[2] and op.P == H * W
    assert op.n_union == int((F != 0).any(axis=1).sum())
    assert np.allclose(op.gram(), F.T @ F, rtol=0, atol=1e-12)
    w = data["rng"].uniform(0.5, 2.0, H * W)
    assert np.allclose(op.gram(w), F.T @ (w[:, None] * F), rtol=0, atol=1e-12)
    assert np.allclose(op.project(movie), F.T @ M.T, rtol=1e-13, atol=1e-9)
    assert np.allclose(op.project(movie, w), (w[:, None] * F).T @ M.T, rtol=1e-13, atol=1e-9)
    # chunks that do not divide T give the same answer
    small = op.project(movie, chunk_bytes=op.n_union * 8 * 7)
    assert np.array_equal(small, op.project(movie))
    assert np.allclose(op.dense_union(), F[op.union], rtol=0, atol=0)
    with pytest.raises(TypeError):
        tr.FootprintOperator(fp)


def test_truncated_gram_solve_equals_pinv_including_rank_deficiency(data):
    fp, movie = data["fp"], data["movie"]
    F = fp.reshape(H * W, -1)
    M = movie.reshape(T, H * W).astype(np.float64).T                  # (P, T)
    rcond = tr.default_rcond(*F.shape)
    expect = np.linalg.pinv(F, rcond=rcond) @ M
    d = {}
    got = tr.truncated_gram_solve(F.T @ F, F.T @ M, rcond, diag=d)
    assert np.allclose(got, expect, rtol=1e-9, atol=1e-9 * np.abs(expect).max())
    assert d["n_dropped"] == 2 and np.isfinite(d["cond"])              # duplicate + empty column
    assert d["n_unresolved"] in (0, 1)                                 # the duplicate's rounding eigenvalue
    assert np.array_equal(got[5], 0.0 * got[5])                        # empty footprint -> zero row
    assert np.allclose(got[1], got[3])                                 # duplicates share the solution
    assert tr.truncated_gram_solve(np.zeros((0, 0)), np.zeros((0, T)), rcond).shape == (0, T)
    assert np.all(tr.truncated_gram_solve(np.zeros((2, 2)), np.ones((2, T)), rcond) == 0)
    # a direction pinv keeps but G cannot resolve is flagged: sigma ratio ~1e-7 (lambda ratio
    # ~1e-14, above float64 resolution of G but below the resolved floor)
    F2 = np.array([[1.0, 1.0], [0.0, 1e-7]])
    _, _, keep2, info2 = tr.gram_spectrum(F2.T @ F2, 1e-12)
    assert info2["n_unresolved"] == 1 and keep2.sum() == 1


# --------------------------------------------------------------------------- #
# solvers vs the dense reference
# --------------------------------------------------------------------------- #
def test_pinv_traces_matches_dense_reference(data):
    fp, sp, movie = data["fp"], data["sp"], data["movie"]
    expect = ref.pinv_traces(movie, fp)
    d = {}
    got = tr.pinv_traces(movie, sp, diag=d)
    scale = np.abs(expect).max()
    assert got.shape == expect.shape == (fp.shape[2], T)
    assert np.allclose(got, expect, rtol=0, atol=1e-9 * scale)
    # an exact duplicate column leaves a rounding-level eigenvalue: either it is dropped (normal
    # equations, duplicate direction contributes zero) or flagged unresolved (dense fallback);
    # both reproduce the pseudoinverse
    assert d["method"] in ("normal_equations", "dense_pinv_union") and d["n_cells"] == 7
    assert d["n_dropped"] == 2 and d["n_union"] == tr.FootprintOperator(sp).n_union
    # forcing the fallback reproduces the dense pseudoinverse through the union
    d2 = {}
    got2 = tr.pinv_traces(movie, sp, cond_limit=1.0, diag=d2)
    assert d2["method"] == "dense_pinv_union"
    assert np.allclose(got2, expect, rtol=0, atol=1e-9 * scale)
    # chunking does not change the answer
    assert np.allclose(tr.pinv_traces(movie, sp, chunk_bytes=1000), got, rtol=0, atol=1e-12 * scale)


def test_pinv_traces_empty_and_well_conditioned(data):
    rng = data["rng"]
    fp, sp = _footprints(rng, dup=False, empty=False)
    movie = data["movie"]
    expect = ref.pinv_traces(movie, fp)
    d = {}
    got = tr.pinv_traces(movie, sp, diag=d)
    assert d["method"] == "normal_equations" and d["n_dropped"] == 0 and d["cond"] < 1e6
    assert np.allclose(got, expect, rtol=0, atol=1e-10 * np.abs(expect).max())
    none = SparseBuilder(H, W).build()
    assert tr.pinv_traces(movie, none).shape == (0, T)
    assert tr.FootprintOperator(none).project(movie).shape == (0, T)


def test_gls_and_isolated_mask_match_dense_reference(data):
    fp, sp, movie, rng = data["fp"], data["sp"], data["movie"], data["rng"]
    filtered = ref.temporal_filter(movie.astype(np.float64), 8)
    std_frames = np.arange(5, 60)
    nm = ref.per_pixel_noise_map(filtered, std_frames, 5.0)
    assert np.array_equal(tr.per_pixel_noise_map(filtered, std_frames, 5.0), nm)
    expect = ref.whitened_gls_traces(movie, fp, nm, 1e-3)
    got = tr.gls_traces(movie, sp, nm, 1e-3)
    assert np.allclose(got, expect, rtol=0, atol=1e-9 * np.abs(expect).max())
    # the same answer from sigmas given only on the pixel union
    op = tr.FootprintOperator(sp)
    got_u = tr.gls_traces(movie, sp, nm.reshape(-1)[op.union], 1e-3)
    assert np.allclose(got_u, expect, rtol=0, atol=1e-9 * np.abs(expect).max())
    with pytest.raises(ValueError):
        tr.gls_traces(movie, sp, nm.reshape(-1)[:5], 1e-3)
    # union noise map == full map restricted to the union (same floor rule: percentile of the
    # pixels it is computed on, so compare without flooring)
    nm0 = ref.per_pixel_noise_map(filtered, std_frames, 0.0).reshape(-1)[op.union]
    nm_u = tr.noise_map_on_union(movie.astype(np.float64), op.union, std_frames, 8, 0.0)
    assert np.allclose(nm_u, nm0, rtol=1e-12, atol=0)
    nm_u32 = tr.noise_map_on_union(movie, op.union, std_frames, 8, 0.0)         # float32 movie
    assert np.allclose(nm_u32, nm0, rtol=1e-12, atol=0)
    assert np.array_equal(tr.isolated_mask(sp), ref.footprint_isolated_mask(fp))
    iso = tr.isolated_mask(sp)
    assert iso[5] and not iso[1] and not iso[3]                        # empty: nothing to overlap; duplicates overlap


def test_extract_cell_traces_dispatch(data):
    fp, sp, movie = data["fp"], data["sp"], data["movie"]
    p = Params(nrow=H, ncol=W)
    assert np.allclose(tr.extract_cell_traces(movie, sp, p), ref.extract_cell_traces(movie, fp, p),
                       rtol=0, atol=1e-9 * np.abs(movie).max())
    filtered = ref.temporal_filter(movie.astype(np.float64), 8)
    nm = ref.per_pixel_noise_map(filtered, np.arange(5, 60), 5.0)
    q = dataclasses.replace(p, whiten_traces=True)
    expect = ref.extract_cell_traces(movie, fp, q, noise_map=nm)
    got = tr.extract_cell_traces(movie, sp, q, noise_map=nm)
    assert np.allclose(got, expect, rtol=0, atol=1e-9 * np.abs(expect).max())
    q2 = dataclasses.replace(q, whiten_isolated_only=False)
    assert np.allclose(tr.extract_cell_traces(movie, sp, q2, noise_map=nm),
                       ref.extract_cell_traces(movie, fp, q2, noise_map=nm),
                       rtol=0, atol=1e-9 * np.abs(expect).max())
    with pytest.raises(ValueError):
        tr.extract_cell_traces(movie, sp, q)
