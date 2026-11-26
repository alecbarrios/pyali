"""Trace extraction: the least-squares projection of the corrected movie onto the footprints.

For footprints ``F`` (``P x N``, one column per cell, ``P = H * W`` pixels) and the corrected
movie ``M`` (``P x T``), the unweighted solution is ``pinv(F) @ (-M)`` (spikes are negative-going
in fluorescence and positive in the traces). Footprints are sparse -- tens to a hundred nonzero
pixels each -- so the solve is done on the normal equations restricted to the union ``U`` of
footprint pixels::

    G = F^T F   (N x N)         B = F^T (-M)   (N x T, a gather of U pixels per frame)
    traces = G^+ B              with the same singular-value truncation as ``np.linalg.pinv(F)``

``G^+`` keeps the eigen-directions with ``lambda_i > (rcond * sigma_max)^2``, i.e. exactly the
singular values ``np.linalg.pinv`` keeps (``sigma_i > rcond * sigma_max``). When ``G`` is too
ill-conditioned for the normal equations to be trusted (``cond_limit``), the solve falls back to
the dense pseudoinverse of ``F`` restricted to ``U`` (``U x N`` instead of ``P x N``), which is
the previous computation exactly. The whitened GLS estimator (opt-in) uses the same two
primitives with per-pixel weights.

Extension points: a movie only has to offer ``gather(pixels, t0, t1)`` (a dense ``(T, H, W)``
array works too); nuisance regressors (background / motion columns) would attach as a dense
border block of ``G``; a per-cell local solver is another entry in :data:`SOLVERS`.
"""
import numpy as np
import scipy.sparse as sps

from .results import SparseFootprints
from .utils import movmedian_time

DEFAULT_CHUNK_BYTES = 512 << 20          # per-chunk gather (frames x union pixels, float64)
DEFAULT_COND_LIMIT = 1e10                # lambda_max / lambda_min(kept) above which G is not trusted
_FLOOR_FACTOR = 10.0                     # G's spectrum is resolved only above ~N * eps * lambda_max


def _gather(movie, pixels, t0, t1):
    """``(t1 - t0, len(pixels))`` float64 block of the corrected movie at flat pixel indices;
    a dense ``(T, H, W)`` array or anything with ``gather(pixels, t0, t1)``."""
    if hasattr(movie, "gather"):
        return np.asarray(movie.gather(pixels, t0, t1), np.float64)
    T = movie.shape[0]
    return np.asarray(movie.reshape(T, -1)[t0:t1][:, pixels], np.float64)


def _n_frames(movie):
    return int(movie.shape[0]) if hasattr(movie, "shape") else int(movie.n_frames)


# --------------------------------------------------------------------------- #
# Sparse footprint operator
# --------------------------------------------------------------------------- #
class FootprintOperator:
    """``F`` as a sparse ``(U x N)`` matrix over ``U`` = the sorted union of all footprint pixels."""

    def __init__(self, sp):
        if not isinstance(sp, SparseFootprints):
            raise TypeError("FootprintOperator needs a SparseFootprints")
        self.N = sp.n()
        self.height, self.width = int(sp.height), int(sp.width)
        self.P = self.height * self.width
        pixel = np.asarray(sp.pixel, np.int64)
        if pixel.size:
            self.union, local = np.unique(pixel, return_inverse=True)
        else:
            self.union, local = np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.csc = sps.csc_matrix((np.asarray(sp.values, np.float64), local.astype(np.int64),
                                   np.asarray(sp.offsets, np.int64)),
                                  shape=(self.union.size, self.N))
        self.csc.sort_indices()

    @property
    def n_union(self):
        return int(self.union.size)

    def _weights_on_union(self, w):
        return None if w is None else np.asarray(w, np.float64).reshape(-1)[self.union]

    def gram(self, w=None):
        """``F^T W F`` as a dense ``(N, N)`` float64 array (``W = diag(w)``, ``w`` over all P pixels)."""
        wu = self._weights_on_union(w)
        A = self.csc if wu is None else sps.diags(wu) @ self.csc
        return np.asarray((self.csc.T @ A).toarray(), np.float64)

    def dense_union(self):
        """``F`` restricted to the union pixels as a dense ``(U, N)`` float64 array."""
        return np.asarray(self.csc.toarray(), np.float64)

    def project(self, movie, w=None, chunk_bytes=DEFAULT_CHUNK_BYTES, op=None):
        """``op @ M_U`` over time in chunks -> ``(N, T)`` float64, where ``M_U`` is the corrected
        movie gathered on the union pixels and ``op`` defaults to ``F^T W`` (``(N, U)``)."""
        T = _n_frames(movie)
        if op is None:
            wu = self._weights_on_union(w)
            op = self.csc.T if wu is None else self.csc.T @ sps.diags(wu)
            op = sps.csr_matrix(op)
        B = np.zeros((op.shape[0], T), np.float64)
        U = self.n_union
        if U == 0 or T == 0 or op.shape[0] == 0:
            return B
        step = max(1, int(chunk_bytes // (U * 8)))
        for t0 in range(0, T, step):
            t1 = min(T, t0 + step)
            sub = _gather(movie, self.union, t0, t1)                   # (n, U)
            B[:, t0:t1] = op @ np.ascontiguousarray(sub.T)             # (N, U) @ (U, n)
        return B


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def default_rcond(P, N):
    """``np.linalg.pinv``'s default cutoff for a ``P x N`` matrix: ``max(P, N) * eps``."""
    return max(int(P), int(N)) * np.finfo(np.float64).eps


def gram_spectrum(G, rcond):
    """Eigendecomposition of ``G = F^T F`` with the truncation bookkeeping.

    ``np.linalg.pinv(F, rcond)`` keeps ``sigma > rcond * sigma_max``, i.e. ``lambda > cut_pinv =
    (rcond * sigma_max)^2``. ``G`` only resolves eigenvalues above ``floor ~ N * eps * lambda_max``
    (squaring the condition number), so directions are kept above ``max(cut_pinv, floor)``;
    the count of directions pinv would keep but ``G`` cannot resolve is ``n_unresolved``, and
    a caller wanting exact pinv semantics there uses the dense fallback. Returns ``(lam, V,
    keep, info)`` with ``info = {cond, n_dropped, n_unresolved, lambda_max}``."""
    G = np.asarray(G, np.float64)
    N = G.shape[0]
    if N == 0:
        return np.zeros(0), np.zeros((0, 0)), np.zeros(0, bool), dict(
            cond=1.0, n_dropped=0, n_unresolved=0, lambda_max=0.0)
    lam, V = np.linalg.eigh(G)
    lmax = float(lam.max())
    if lmax <= 0:
        return lam, V, np.zeros(N, bool), dict(cond=np.inf, n_dropped=N, n_unresolved=0,
                                               lambda_max=lmax)
    cut_pinv = (float(rcond) * np.sqrt(lmax)) ** 2
    floor = _FLOOR_FACTOR * N * np.finfo(np.float64).eps * lmax
    keep = lam > max(cut_pinv, floor)
    unresolved = int(((lam > cut_pinv) & ~keep).sum())
    cond = float(lmax / lam[keep].min()) if keep.any() else np.inf
    return lam, V, keep, dict(cond=cond, n_dropped=int((~keep).sum()), n_unresolved=unresolved,
                              lambda_max=lmax)


def _apply_spectrum(lam, V, keep, B):
    Vk = V[:, keep]
    return Vk @ ((Vk.T @ B) / lam[keep][:, None])


def truncated_gram_solve(G, B, rcond, diag=None):
    """``G^+ B`` keeping the eigen-directions ``np.linalg.pinv(F, rcond)`` would keep (see
    :func:`gram_spectrum`). Dropped directions (duplicate or empty footprints) contribute zero,
    as in the pseudoinverse. ``diag`` (dict) receives the spectrum bookkeeping."""
    B = np.asarray(B, np.float64)
    lam, V, keep, info = gram_spectrum(G, rcond)
    if diag is not None:
        diag.update(info)
    if lam.size == 0 or not keep.any():
        return np.zeros((lam.size, B.shape[1]))
    return _apply_spectrum(lam, V, keep, B)


def pinv_traces(movie, sp, rcond=None, chunk_bytes=DEFAULT_CHUNK_BYTES,
                cond_limit=DEFAULT_COND_LIMIT, diag=None):
    """``pinv(F) @ (-movie)`` -> ``(N, T)`` float64 through the sparse normal equations, with the
    dense pseudoinverse on the pixel union as the fallback for an ill-conditioned ``G``."""
    op = FootprintOperator(sp)
    rcond = default_rcond(op.P, op.N) if rcond is None else float(rcond)
    d = {} if diag is None else diag
    d.update(method="normal_equations", n_cells=op.N, n_union=op.n_union, rcond=rcond)
    if op.N == 0:
        return np.zeros((0, _n_frames(movie)))
    lam, V, keep, info = gram_spectrum(op.gram(), rcond)
    d.update(info)
    if info["n_unresolved"] or info["cond"] > cond_limit:
        # G cannot resolve a direction pinv would keep (near-duplicate footprints) or is too
        # ill-conditioned for the normal equations: the exact pseudoinverse on the pixel union
        # (U x N instead of P x N) is the previous computation itself, at a bounded cost.
        d["method"] = "dense_pinv_union"
        pinv_FU = np.linalg.pinv(op.dense_union(), rcond=rcond)       # (N, U)
        return -op.project(movie, chunk_bytes=chunk_bytes, op=pinv_FU)
    B = op.project(movie, chunk_bytes=chunk_bytes)
    if not keep.any():
        return np.zeros((op.N, B.shape[1]))
    return -_apply_spectrum(lam, V, keep, B)


def per_pixel_noise_map(filtered_movie, std_frames, floor_pct=5.0):
    """Per-pixel high-frequency noise std from the baseline (no-stimulus) frames of the
    moving-median high-passed movie: ``1.4826 * MAD`` over time, floored at the ``floor_pct``
    percentile of the positive values. ``(T, H, W)`` in, ``(H, W)`` out."""
    base = filtered_movie[np.asarray(std_frames, int)]                 # [n_base, H, W]
    med = np.median(base, axis=0)
    sigma = 1.4826 * np.median(np.abs(base - med[None]), axis=0)       # [H, W]
    pos = sigma[sigma > 0]
    if pos.size:
        sigma = np.maximum(sigma, np.percentile(pos, floor_pct))
    else:                                                              # degenerate: uniform weights
        sigma = np.ones_like(sigma)
    return sigma


def noise_map_on_union(movie, pixels, std_frames, window, floor_pct=5.0):
    """:func:`per_pixel_noise_map` evaluated only on ``pixels`` (flat indices): gathers the
    baseline span (plus a ``window`` margin, so the moving median matches the full-movie one)
    there, high-passes it along time and returns ``(len(pixels),)`` sigmas floored at the
    ``floor_pct`` percentile of those pixels."""
    frames = np.asarray(std_frames, int)
    T = _n_frames(movie)
    if frames.size == 0:
        t0, t1 = 0, T
        frames = np.arange(T)
    else:
        t0, t1 = max(0, int(frames.min()) - int(window)), min(T, int(frames.max()) + 1 + int(window))
    full = _gather(movie, pixels, t0, t1)
    X = (movmedian_time(full, window, axis=0) - full)[frames - t0]        # high-pass, baseline frames
    med = np.median(X, axis=0)
    sigma = 1.4826 * np.median(np.abs(X - med[None]), axis=0)
    pos = sigma[sigma > 0]
    sigma = np.maximum(sigma, np.percentile(pos, floor_pct)) if pos.size else np.ones_like(sigma)
    return sigma


def isolated_mask(sp):
    """Boolean ``(N,)``: True where a footprint's bounding box overlaps no other footprint's
    (touching boxes count as overlapping); an empty footprint overlaps nothing and is True."""
    N = sp.n()
    boxes = np.full((N, 4), -1.0)
    for k in range(N):
        bb = sp.bbox(k)
        if bb is not None:
            boxes[k] = bb
    valid = boxes[:, 0] >= 0
    out = np.ones(N, bool)
    r0, r1, c0, c1 = boxes.T
    for i in np.flatnonzero(valid):
        hit = valid & (r0[i] <= r1) & (r0 <= r1[i]) & (c0[i] <= c1) & (c0 <= c1[i])
        hit[i] = False
        out[i] = not hit.any()
    return out


def gls_traces(movie, sp, noise_map, ridge_frac=1e-3, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Noise-weighted (GLS / BLUE) traces ``(F^T W F + lambda I)^-1 F^T W (-movie)`` with
    ``W = diag(1 / sigma^2)``; ``noise_map`` is ``(H, W)`` or a ``(U,)`` vector over the operator's
    pixel union. ``ridge_frac`` scales ``lambda`` by ``mean(diag(F^T W F))``."""
    op = FootprintOperator(sp)
    T = _n_frames(movie)
    if op.N == 0:
        return np.zeros((0, T))
    nm = np.asarray(noise_map, np.float64)
    if nm.size == op.P:
        w = 1.0 / (nm.reshape(-1) ** 2)
    elif nm.size == op.n_union:
        w = np.zeros(op.P)
        w[op.union] = 1.0 / (nm.reshape(-1) ** 2)
    else:
        raise ValueError(f"noise_map has {nm.size} values; expected {op.P} (H*W) or {op.n_union} (union)")
    A = op.gram(w)
    A = A + ridge_frac * np.mean(np.diag(A)) * np.eye(op.N)
    B = -op.project(movie, w, chunk_bytes=chunk_bytes)
    return np.linalg.solve(A, B)


SOLVERS = {"pinv": pinv_traces, "gls": gls_traces}


def extract_cell_traces(movie, sp, p, noise_map=None, verbose=False, diag=None):
    """Pipeline entry: unweighted :func:`pinv_traces` by default; with ``p.whiten_traces`` the
    GLS rows (``p.whiten_isolated_only``: only for cells whose footprint overlaps no other, the
    rest keeping the pinv row). Returns ``(N, T)`` float64."""
    rcond = getattr(p, "trace_rcond", None)
    chunk = int(getattr(p, "trace_chunk_bytes", DEFAULT_CHUNK_BYTES))
    cond_limit = float(getattr(p, "trace_cond_limit", DEFAULT_COND_LIMIT))
    if not getattr(p, "whiten_traces", False):
        return pinv_traces(movie, sp, rcond, chunk, cond_limit, diag)
    if noise_map is None:
        raise ValueError("whiten_traces=True requires a noise_map")
    gls = gls_traces(movie, sp, noise_map, p.whiten_ridge, chunk)
    if not getattr(p, "whiten_isolated_only", True):
        return gls
    ols = pinv_traces(movie, sp, rcond, chunk, cond_limit, diag)      # faithful rows for overlappers
    isolated = isolated_mask(sp)
    if verbose:
        print(f"[pyali]   whitened {int(isolated.sum())}/{len(isolated)} isolated cells; "
              f"{int((~isolated).sum())} overlapping cells kept faithful pinv", flush=True)
    out = ols.copy()
    out[isolated] = gls[isolated]
    return out
