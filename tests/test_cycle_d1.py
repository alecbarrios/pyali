"""Unit tests for Cycle D1: MATLAB-style 1-D median filter, mean absolute deviation, the faithful
spike detector, and one-frame-per-spike SVD localization."""
import numpy as np
import pytest

from pyali.extract import median_filter_1d, mean_abs_dev


# --------------------------------------------------------------------------- #
# median_filter_1d (MATLAB medfilt1, zero padding)
# --------------------------------------------------------------------------- #
def _medfilt1_reference(x, n):
    """Brute-force MATLAB semantics: window k = x[k-nb : k+na], zeros beyond the ends."""
    x = np.asarray(x, float)
    T = len(x)
    nb, na = ((n - 1) // 2, (n - 1) // 2) if n % 2 else (n // 2, n // 2 - 1)
    out = np.empty(T)
    for k in range(T):
        w = [x[i] if 0 <= i < T else 0.0 for i in range(k - nb, k + na + 1)]
        out[k] = np.median(w)
    return out


@pytest.mark.parametrize("n", [1, 3, 8, 32, 33])
def test_median_filter_1d_matches_matlab_semantics(n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=100).cumsum() + rng.normal(size=100)
    y = median_filter_1d(x, n)
    assert y.shape == (100,) and y.dtype == np.float64
    assert np.allclose(y, _medfilt1_reference(x, n))


def test_median_filter_1d_even_window_alignment_and_zero_padding():
    x = np.arange(1.0, 11.0)                      # 1..10
    y = median_filter_1d(x, 4)                    # window k = x[k-2 : k+1]
    # k=0: [0, 0, 1, 2] -> 0.5 ; k=5: [4,5,6,7] -> 5.5 ; k=9: [8, 9, 10, 0] -> 8.5
    assert np.allclose(y[[0, 5, 9]], [0.5, 5.5, 8.5])
    assert np.allclose(median_filter_1d(x, 1), x)
    with pytest.raises(ValueError):
        median_filter_1d(x, 0)


# --------------------------------------------------------------------------- #
# mean_abs_dev (MATLAB mad, flag 0)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# detect_region_aps
# --------------------------------------------------------------------------- #
def _patch_with_spikes(T=800, Hp=10, Wp=10, spikes=(), amp=8.0, width=3, noise=1.0, seed=0):
    """A raw patch whose in-mask pixels share one trace: slow drift + bounded uniform noise +
    ``width``-frame negative dips of depth ``amp`` at ``spikes`` (a spike darkens the raw
    intensity). Uniform noise in [-noise, noise] keeps the test deterministic: 4 x the mean
    absolute deviation of the detection signal (~2 * noise) exceeds its largest excursion
    (~1.5 * noise), so no noise sample can cross the threshold. (With Gaussian noise the faithful
    ~3.2-sigma threshold admits about one false peak per 800 frames, as in the original.) The
    filtered patch ``pm`` is the negated raw patch (spikes positive), which is all the coarse
    location needs. Returns (patch_raw, mask, pm, truth_frames)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = 100.0 + 5.0 * np.sin(2 * np.pi * t / 400.0) + noise * rng.uniform(-1.0, 1.0, size=T)
    for s in spikes:
        base[s:s + width] -= amp
    raw = np.zeros((T, Hp, Wp))
    mask = np.zeros((Hp, Wp))
    mask[3:7, 3:7] = 1.0
    raw[:, mask > 0] = base[:, None]
    return raw, mask, -raw, np.asarray(spikes, int)


def test_detector_finds_planted_spikes_with_recall_and_precision_one():
    from pyali.extract import detect_region_aps, dedup_close_peaks
    spikes = [60, 130, 245, 300, 410, 522, 600, 690, 760]
    raw, mask, pm, truth = _patch_with_spikes(spikes=spikes)
    ap, diag = detect_region_aps(raw, mask, pm, (0, 0), 32, 16, 4.0)
    ap = dedup_close_peaks(ap, 25)                 # a 3-frame plateau can carry two adjacent maxima
    found = ap[:, 2].astype(int)
    assert len(found) == len(truth)
    tol = 2                                                              # a 3-frame dip: peak within width-1 of onset
    assert all(np.abs(found - s).min() <= tol for s in truth)           # recall
    assert all(np.abs(truth - f).min() <= tol for f in found)           # precision
    assert diag["threshold"] == pytest.approx(4.0 * mean_abs_dev(diag["dF"]))
    assert np.allclose(ap[:, 3], diag["dF"][found])                     # amplitude = dF at the peak


def test_detector_zeroes_the_edge_frames_and_returns_empty_without_spikes():
    from pyali.extract import detect_region_aps, dedup_close_peaks
    raw, mask, pm, _ = _patch_with_spikes(spikes=[5, 100, 790], amp=20.0)
    ap, diag = detect_region_aps(raw, mask, pm, (0, 0), 32, 16, 4.0)
    ap = dedup_close_peaks(ap, 25)
    assert ap[:, 2].astype(int).tolist() in ([100], [101], [102])      # frames 5 and 790 are in the edges
    assert np.all(diag["dF"][:16] == 0) and np.all(diag["dF"][-16:] == 0)
    raw, mask, pm, _ = _patch_with_spikes(spikes=[], noise=1.0)
    ap, _ = detect_region_aps(raw, mask, pm, (0, 0), 32, 16, 4.0)
    assert ap.shape == (0, 4)


def test_detector_32_frame_median_does_not_attenuate_a_short_spike():
    """A 3-frame spike inside a 32-frame median window leaves the median untouched, so the
    detection signal recovers the full spike depth (the old 8-frame per-pixel median biased
    the baseline up by the spike itself)."""
    from pyali.extract import detect_region_aps
    raw, mask, pm, _ = _patch_with_spikes(spikes=[400], amp=8.0, noise=0.0)
    ap, diag = detect_region_aps(raw, mask, pm, (0, 0), 32, 16, 4.0)
    peak = diag["dF"][398:404].max()
    scale = mask.sum() / mask.size                                        # trace is sum / patch area
    assert abs(peak - 8.0 * scale) < 0.05 * 8.0 * scale
    assert ap.shape[0] == 1 and ap[0, 2] in (400, 401, 402)


# --------------------------------------------------------------------------- #
# svd_denoise_frames
# --------------------------------------------------------------------------- #
def test_svd_denoise_frames_shapes_rank_cap_and_exact_below_rank():
    from pyali.extract import svd_denoise_frames
    rng = np.random.default_rng(3)
    T, Hp, Wp = 50, 9, 11
    pm = rng.normal(size=(T, Hp, Wp))
    times = np.array([5, 17, 23, 40])
    out = svd_denoise_frames(pm, times, rank=10)
    assert out.shape == (Hp, Wp, 4)
    assert np.allclose(out, np.transpose(pm[times], (1, 2, 0)))          # 4 spikes <= rank 10: exact
    assert svd_denoise_frames(pm, [], rank=10).shape == (Hp, Wp, 0)


def test_svd_denoise_frames_truncates_to_rank():
    from pyali.extract import svd_denoise_frames
    rng = np.random.default_rng(4)
    T, Hp, Wp, n = 400, 8, 8, 60
    # frames = 2 spatial patterns x random loadings + small noise -> rank-2 structure
    a, b = rng.normal(size=(Hp, Wp)), rng.normal(size=(Hp, Wp))
    pm = (rng.normal(size=(T, 1, 1)) * a + rng.normal(size=(T, 1, 1)) * b
          + 0.01 * rng.normal(size=(T, Hp, Wp)))
    times = np.sort(rng.choice(T, n, replace=False))
    out = svd_denoise_frames(pm, times, rank=2)
    X = out.reshape(Hp * Wp, n)
    assert np.linalg.matrix_rank(X, tol=1e-8) == 2
    raw = np.transpose(pm[times], (1, 2, 0))
    assert np.abs(out - raw).max() < 0.1 and np.abs(out - raw).mean() < 0.02   # only the noise is removed
    # no probe-pixel trick: a patch narrower than 10 px works
    assert svd_denoise_frames(pm[:, :5, :6], times[:7], rank=10).shape == (5, 6, 7)


def test_mean_abs_dev_is_the_mean_deviation_about_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=200_000)
    assert abs(mean_abs_dev(x) - 2.0 * np.sqrt(2 / np.pi)) < 0.02      # 0.798 sigma for Gaussian
    assert mean_abs_dev(np.full(10, 3.0)) == 0.0
    assert mean_abs_dev([0, 0, 0, 4]) == pytest.approx(1.5)             # mean 1: |dev| = 1,1,1,3
    assert mean_abs_dev([]) == 0.0
