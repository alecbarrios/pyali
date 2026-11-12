"""Synthetic voltage-imaging movie with planted ground truth: the oracle for pyali reviews.

``make_movie`` builds a small ``[T + truncate_last, H, W]`` uint16 movie that the production
pipeline can run end to end in seconds under its **default** tunables (the defaults are what is
under review, so only the acquisition-specific fields are set):

* a smooth spatial background plus a slow bleaching drift, so ``adaptive_background`` has
  something to remove;
* Gaussian cells of ~6 px diameter, each with a shared slow common-mode fluctuation across its
  pixels (what makes the correlation image light a cell up);
* **negative-going** spikes, i.e. dips in intensity, which is the sign convention of the
  indicator and the reason ``extract.temporal_filter`` flips the sign;
* shot-noise-like Gaussian noise with sigma = sqrt(intensity).

The returned ``truth`` holds the planted centres, spike times, footprints and per-cell traces in
the same conventions the pipeline outputs use (0-indexed frames in the truncated movie; centres
as ``[row, col]`` 0-indexed), so tests and verifiers can compare directly.
"""
import os

import numpy as np

from pyali.params import Params

DEFAULT_CELLS = [(24, 28), (30, 66), (62, 22), (68, 70)]        # (row, col), 0-indexed, interior
SPIKE_KERNEL = np.array([0.6, 1.0, 0.35])                        # ~2.5 ms AP at 800 Hz


def synthetic_params(H=96, W=96, **overrides):
    """Production defaults plus only the acquisition-specific fields the synthetic movie needs.

    ``max_region_bbox_frac`` is raised from 0.01 to 0.05 because a 6-px cell's bounding box is
    already ~1% of a 96x96 frame and would be rejected by the production guard, which is tuned
    for 800x800 frames.
    """
    base = dict(nrow=H, ncol=W, n_ref=600, fps=800.0, truncate_last=10,
                read_dtype="uint16", compute_dtype="float64",
                bkg_ranges=[(1, 100), (380, 440), (740, 800)],
                std_ranges=[(1, 100), (380, 440), (740, 800)],
                saturation_clip=None, max_region_bbox_frac=0.05)
    base.update(overrides)
    return Params(**base)


def gaussian_blob(H, W, r, c, sigma):
    rr, cc = np.mgrid[:H, :W]
    return np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2.0 * sigma ** 2))


def lowpass_noise(rng, n, tau=40.0):
    """Unit-variance AR(1) noise with correlation time ``tau`` frames."""
    a = np.exp(-1.0 / tau)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = a * x[t - 1] + np.sqrt(1.0 - a * a) * rng.normal()
    return x


def plant_spike_times(rng, T, n, min_gap=25, forbidden=(), margin=8):
    """``n`` spike frames in ``[margin, T - margin)``, >= ``min_gap`` apart, outside ``forbidden``
    0-indexed inclusive frame ranges (the baseline windows, so the noise estimate stays clean)."""
    ok = np.ones(T, bool)
    ok[:margin] = False
    ok[T - margin:] = False
    for a, b in forbidden:
        ok[max(a, 0):min(b + 1, T)] = False
    times = []
    for _ in range(10000):
        if len(times) == n:
            break
        t = int(rng.integers(0, T))
        if ok[t] and all(abs(t - u) >= min_gap for u in times):
            times.append(t)
    return np.array(sorted(times), int)


def make_movie(seed=0, H=96, W=96, T=800, n_cells=4, cells=None, spike_times=None,
               spikes_per_cell=None, motion=False, blank=False, baseline=1500.0, cell_amp=700.0,
               sigma_px=2.2, spike_frac=0.25, common_frac=0.05, noise=True, drift=True):
    """Build the movie and its ground truth.

    ``T=800`` with ``n_ref=600`` matches the production reference length: the correlation image
    is averaged over as many frames as in a real run, which is what keeps background speckle
    from dominating it. ``common_frac=0.05`` gives neighbouring cell pixels a correlation
    clearly above the background's. Even so, the production segmentation admits some speckle
    regions (it does on real data too); the oracle therefore asserts that every planted cell is
    found, and reports the extra regions rather than forbidding them.

    Parameters
    ----------
    cells : list of (row, col), optional
        Planted centres (0-indexed); default the first ``n_cells`` of ``DEFAULT_CELLS``.
    spike_times : list of sequences, optional
        Explicit 0-indexed spike frames per cell; otherwise drawn at random.
    spikes_per_cell : int, optional
        Number of random spikes per cell (default 12-20).
    motion : bool
        Apply a slow sub-pixel rigid shift (default off: the extraction is what is tested).
    blank : bool
        No cells at all (the zero-region / zero-cell case).
    drift : bool
        Spatial gradient plus bleaching (default). ``False`` gives a flat constant background;
        with ``noise=False`` and ``blank=True`` that is a perfectly constant movie.

    Returns
    -------
    dict(movie, truth, params)
        ``movie`` is ``(T + truncate_last, H, W)`` uint16. ``truth`` has ``centers`` (N, 2)
        ``[row, col]``; ``spike_times`` list of int arrays; ``footprints`` (N, H, W);
        ``traces`` (N, T): the negated intensity of each cell's peak pixel, so a spike is a
        positive peak exactly as in ``cell_traces``; ``bleach`` (T,).
    """
    rng = np.random.default_rng(seed)
    p = synthetic_params(H, W)
    T_all = T + p.truncate_last
    rr, cc = np.mgrid[:H, :W]
    spatial = baseline + (300.0 * (0.5 * rr / H + 0.5 * cc / W) if drift else np.zeros((H, W)))
    bleach = 1.0 - (0.05 if drift else 0.0) * np.arange(T_all) / T_all
    movie = spatial[None, :, :] * bleach[:, None, None]            # (T_all, H, W) float64

    cells_rc = [] if blank else list(cells if cells is not None else DEFAULT_CELLS[:n_cells])
    N = len(cells_rc)
    footprints = np.stack([gaussian_blob(H, W, r, c, sigma_px) for r, c in cells_rc]) \
        if N else np.zeros((0, H, W))
    forbidden = [(a - 1, b - 1) for a, b in p.std_ranges]
    modulation = np.zeros((N, T_all))
    times_out = []
    for i in range(N):
        if spike_times is not None:
            times = np.asarray(spike_times[i], int)
        else:
            n = spikes_per_cell if spikes_per_cell is not None else int(rng.integers(12, 21))
            times = plant_spike_times(rng, T, n, forbidden=forbidden)
        times_out.append(times)
        s = np.zeros(T_all)
        for t in times:
            k = min(len(SPIKE_KERNEL), T_all - t)
            if t >= 0 and k > 0:
                s[t:t + k] += SPIKE_KERNEL[:k]
        modulation[i] = common_frac * lowpass_noise(rng, T_all) - spike_frac * s
        movie += cell_amp * footprints[i][None] * (1.0 + modulation[i])[:, None, None]

    if motion:
        from scipy import ndimage
        dr = 0.3 * lowpass_noise(rng, T_all, tau=80.0)
        dc = 0.3 * lowpass_noise(rng, T_all, tau=80.0)
        for t in range(T_all):
            movie[t] = ndimage.shift(movie[t], (dr[t], dc[t]), order=1, mode="nearest")
    if noise:
        movie += rng.normal(size=movie.shape) * np.sqrt(np.maximum(movie, 1.0))
    movie = np.clip(np.round(movie), 0, 65535).astype(np.uint16)

    peak = np.array([footprints[i][cells_rc[i]] for i in range(N)]) if N else np.zeros(0)
    traces = -(cell_amp * peak[:, None] * bleach[None, :T] * (1.0 + modulation[:, :T]))
    truth = dict(centers=np.array(cells_rc, float).reshape(-1, 2), spike_times=times_out,
                 footprints=footprints, traces=traces, bleach=bleach[:T], cell_amp=cell_amp,
                 spike_frac=spike_frac)
    return dict(movie=movie, truth=truth, params=p)


def write_fov(fov_dir, movie):
    """Write ``frames1.bin`` (little-endian uint16, frame-major, row-major) so
    :func:`pyali.pipeline.process_fov` can read the directory. Returns the path."""
    os.makedirs(fov_dir, exist_ok=True)
    path = os.path.join(fov_dir, "frames1.bin")
    np.ascontiguousarray(movie, dtype="<u2").tofile(path)
    return path


def make_fov(fov_dir, **kwargs):
    """``make_movie(**kwargs)`` written to ``fov_dir``; returns ``(truth, params)``."""
    data = make_movie(**kwargs)
    write_fov(fov_dir, data["movie"])
    return data["truth"], data["params"]


# --------------------------------------------------------------------------- #
# Scoring helpers shared by the tests and the verifier
# --------------------------------------------------------------------------- #
def match_centers(found_rc0, planted_rc0, tol=2.0):
    """Nearest-planted-centre match. ``found_rc0``/``planted_rc0`` are 0-indexed ``[row, col]``.

    Returns ``(assignment, dist)`` where ``assignment[k]`` is the planted index of found centre
    ``k`` (or -1 beyond ``tol``) and ``dist[k]`` its distance."""
    found = np.asarray(found_rc0, float).reshape(-1, 2)
    planted = np.asarray(planted_rc0, float).reshape(-1, 2)
    if not len(found) or not len(planted):
        return np.full(len(found), -1), np.full(len(found), np.inf)
    d = np.sqrt(((found[:, None, :] - planted[None, :, :]) ** 2).sum(-1))
    j = d.argmin(1)
    dist = d[np.arange(len(found)), j]
    return np.where(dist <= tol, j, -1), dist


def spike_recall_precision(found_frames, planted_frames, tol=1):
    """Recall and precision of detected spike frames against planted ones within ``tol`` frames."""
    f = np.asarray(sorted(set(int(v) for v in found_frames)))
    pl = np.asarray(sorted(set(int(v) for v in planted_frames)))
    if not len(pl):
        return (1.0 if not len(f) else 0.0), (1.0 if not len(f) else 0.0)
    if not len(f):
        return 0.0, 1.0
    hit_p = np.array([np.any(np.abs(f - t) <= tol) for t in pl])
    hit_f = np.array([np.any(np.abs(pl - t) <= tol) for t in f])
    return float(hit_p.mean()), float(hit_f.mean())


def spike_zscore(trace, spike_frames):
    """Mean height of ``trace`` at the planted spike peaks (frame + 1, the kernel's maximum)
    above its median, in robust-sigma units. The trace is a ``cell_traces`` row (spike-positive)."""
    x = np.asarray(trace, float)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) or 1.0
    idx = np.clip(np.asarray(spike_frames, int) + 1, 0, len(x) - 1)
    return float(np.mean(x[idx] - med) / mad)
