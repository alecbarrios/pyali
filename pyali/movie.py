"""The movie as acquired, with its linear corrections applied where pixels are read.

The pipeline's two movie corrections are linear in the raw frames::

    corrected[t, px] = raw[t, px] - bkg(t)[px] - A[px, :] @ shift[:, t]

* ``bkg(t)`` is the adaptive background: a linear interpolation in time between a few anchor
  images (:func:`pyali.preprocess.background_samples`, morphological openings of mean frames),
  i.e. ``bkg(t) = W[t, :] @ samples`` with two nonzero weights per frame;
* the rigid motion term is the gradient operator ``A`` (``(P, 2)``, from
  :func:`pyali.preprocess.motion_operator`) times the per-frame least-squares shift.

Materializing the corrected movie costs a float copy of the acquisition (21.5 GB for an
800x800x8389 8-bit field as float32). Only two consumers read it -- region patches in
:func:`pyali.extract.extract_footprints` and the footprint-pixel gather in :mod:`pyali.traces` --
so :class:`CorrectedMovie` keeps the raw integer frames (5.4 GB) and evaluates the correction in
float64 for the pixels and frames requested. This is also more exact than the float32 in-place
path it replaces. ``frames()`` returns the *uncorrected* frames (exact integers) for the
reference / background stages, which must see the acquisition as recorded.

Extension points: ``raw`` can be a ``np.memmap`` (``pyali.io.open_bin_memmap``) for small-RAM
boxes, nothing else changes; :meth:`patch` is the unit of work a parallel region loop farms out.
"""
import numpy as np

from . import preprocess


def interp_weights(anchors, n_frames):
    """``(n_frames, len(anchors))`` float64 interpolation weights: row ``t`` (0-based frame)
    holds the two weights of :func:`pyali.preprocess._interp_weight` for 1-indexed frame
    ``t + 1`` (clamped to the first / last anchor outside their span)."""
    anchors = np.asarray(anchors, float)
    W = np.zeros((int(n_frames), anchors.size))
    for t in range(int(n_frames)):
        ib, ia, w = preprocess._interp_weight(t + 1, anchors)
        if ib == ia:
            W[t, ib] = 1.0
        else:
            W[t, ib] = 1.0 - w
            W[t, ia] = w
    return W


def estimate_shift(raw, W, samples, truncate_last=10, chunk_frames=64):
    """Per-frame rigid shift of the background-subtracted movie, as
    ``preprocess.motion_correct`` computed it on the materialized movie, streamed over frames.

    ``raw`` is ``(T, H, W)`` (any dtype; only the first ``T = W.shape[0]`` frames are used),
    ``W`` the interpolation weights, ``samples`` ``(n, H, W)``. The reference ``f0`` is the mean
    of the background-subtracted frames excluding the last ``truncate_last``; ``A`` its gradient
    operator; ``shift[:, t] = pinv(A) @ (raw[t] - bkg(t))``. Returns ``(shift (2, T), A (P, 2), f0)``.
    """
    T, n = W.shape
    H, Wd = raw.shape[1], raw.shape[2]
    P = H * Wd
    S = np.asarray(samples, np.float64).reshape(n, P)
    Tm = max(T - int(truncate_last), 1)
    # f0 = mean_t(raw[t] - bkg(t)) = mean(raw) - mean_t(W[t]) @ S: the reduction over frames
    # runs in chunks straight from the integer frames (no float copy of the movie)
    f0 = np.zeros(P)
    for t0 in range(0, Tm, chunk_frames):
        t1 = min(Tm, t0 + chunk_frames)
        f0 += raw[t0:t1].reshape(t1 - t0, P).sum(axis=0, dtype=np.float64)
    f0 /= Tm
    f0 -= W[:Tm].mean(axis=0) @ S
    f0 = f0.reshape(H, Wd)
    A = preprocess.motion_operator(f0)                                  # (P, 2)
    pinvA = np.linalg.pinv(A)                                           # (2, P)
    # shift = pinvA @ (raw - W S)^T = pinvA @ raw^T - (pinvA @ S^T) @ W^T: the background term
    # collapses to a (2, n) matrix, so only the raw frames are ever touched, chunk by chunk
    pinvA_S = pinvA @ S.T                                               # (2, n)
    shift = np.zeros((2, T))
    for t0 in range(0, T, chunk_frames):
        t1 = min(T, t0 + chunk_frames)
        chunk = np.asarray(raw[t0:t1], np.float64).reshape(t1 - t0, P)
        shift[:, t0:t1] = pinvA @ chunk.T
    shift -= pinvA_S @ W.T
    return shift, A, f0


class CorrectedMovie:
    """Raw integer frames plus the background / motion corrections, applied on read.

    Parameters
    ----------
    raw : (T_raw, H, W) array (uint8 / uint16 as acquired; a memmap works)
    samples, anchors : from :func:`pyali.preprocess.background_samples`
    shift, A : from :func:`estimate_shift`
    n_frames : int, optional
        Analysis frames (``T``); trailing raw frames beyond it are ignored.
    saturation_clip : float, optional
        Corrected values above it are zeroed (16-bit acquisitions), as ``clip_saturation`` did.
    """

    def __init__(self, raw, samples, anchors, shift, A, n_frames=None, saturation_clip=None):
        self.raw = raw
        self.height, self.width = int(raw.shape[1]), int(raw.shape[2])
        self.P = self.height * self.width
        self.n_frames = int(raw.shape[0] if n_frames is None else n_frames)
        if self.n_frames > raw.shape[0]:
            raise ValueError(f"n_frames {self.n_frames} > raw frames {raw.shape[0]}")
        self.samples_flat = np.asarray(samples, np.float64).reshape(len(samples), self.P)
        self.anchors = np.asarray(anchors, float)
        self.W = interp_weights(self.anchors, self.n_frames)
        self.shift = np.asarray(shift, np.float64).reshape(2, -1)[:, :self.n_frames]
        if self.shift.shape[1] != self.n_frames:
            raise ValueError(f"shift has {self.shift.shape[1]} frames, movie {self.n_frames}")
        self.A = np.asarray(A, np.float64).reshape(self.P, 2)
        self.saturation_clip = saturation_clip

    @property
    def shape(self):
        return (self.n_frames, self.height, self.width)

    def __len__(self):
        return self.n_frames

    # -- reads ---------------------------------------------------------------- #
    def frames(self, t0, t1, dtype=np.float32):
        """Uncorrected frames ``[t0, t1)`` cast to ``dtype`` (exact for integer acquisitions)."""
        t1 = min(int(t1), self.n_frames)
        return np.asarray(self.raw[int(t0):t1], dtype)

    def _correction(self, pixels, t0, t1):
        """``(t1 - t0, len(pixels))`` float64: background + motion term at those pixels."""
        out = self.W[t0:t1] @ self.samples_flat[:, pixels]
        out += self.shift[:, t0:t1].T @ self.A[pixels].T
        return out

    def _finish(self, X):
        if self.saturation_clip is not None:
            X[X > self.saturation_clip] = 0.0
        return X

    def gather(self, pixels, t0=0, t1=None):
        """Corrected values at flat pixel indices (``row * W + col``) for frames ``[t0, t1)``
        -> ``(t1 - t0, len(pixels))`` float64 (the trace solve's precision)."""
        t0, t1 = int(t0), self.n_frames if t1 is None else min(int(t1), self.n_frames)
        pixels = np.asarray(pixels, np.int64).reshape(-1)
        n = t1 - t0
        # gather the integer values first (a fraction of the frame), then promote
        X = self.raw[t0:t1].reshape(n, self.P)[:, pixels].astype(np.float64)
        X -= self._correction(pixels, t0, t1)
        return self._finish(X)

    def patch(self, r0, r1, c0, c1, t0=0, t1=None, dtype=np.float32):
        """Corrected ``(t1 - t0, r1 - r0, c1 - c0)`` block (0-indexed, half-open). The correction
        is evaluated in float64 and the block returned as ``dtype``: float32 by default, the
        working precision of the per-region detection / SVD path (the old float32 movie rounded
        twice, once per in-place correction; this rounds once)."""
        t0, t1 = int(t0), self.n_frames if t1 is None else min(int(t1), self.n_frames)
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
        n, hp, wp = t1 - t0, r1 - r0, c1 - c0
        X = np.ascontiguousarray(self.raw[t0:t1, r0:r1, c0:c1]).astype(np.float64)
        rows, cols = np.mgrid[r0:r1, c0:c1]
        pixels = (rows * self.width + cols).ravel()
        X -= self._correction(pixels, t0, t1).reshape(n, hp, wp)
        return np.asarray(self._finish(X), dtype)

    def materialize(self, dtype=np.float64, chunk_frames=256):
        """The whole corrected movie ``(T, H, W)`` -- for tests and small movies only."""
        out = np.empty((self.n_frames, self.height, self.width), dtype)
        for t0 in range(0, self.n_frames, chunk_frames):
            t1 = min(self.n_frames, t0 + chunk_frames)
            out[t0:t1] = self.patch(0, self.height, 0, self.width, t0, t1, dtype=np.float64)
        return out


def corrected_movie(raw, p, n_frames=None):
    """Build a :class:`CorrectedMovie` from raw frames and :class:`pyali.params.Params`: anchor
    backgrounds from ``p.bkg_ranges`` / ``p.disk_radius``, the shift with ``p.truncate_last``.
    Returns ``(movie, info)`` with ``info = {samples, anchors, shift, A, f0}``."""
    T = int(raw.shape[0] if n_frames is None else n_frames)
    samples, anchors = preprocess.background_samples(raw[:T], p.bkg_ranges, p.disk_radius)
    W = interp_weights(anchors, T)
    shift, A, f0 = estimate_shift(raw[:T], W, samples, p.truncate_last)
    movie = CorrectedMovie(raw, samples, anchors, shift, A, n_frames=T,
                           saturation_clip=getattr(p, "saturation_clip", None))
    return movie, dict(samples=samples, anchors=anchors, shift=shift, A=A, f0=f0)
