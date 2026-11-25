"""Unit tests for Cycle C1: structuring elements, MATLAB-style Gaussian, mask-aware large-scale
background, hysteresis mask, soma watershed, and the combined segmentation."""
import numpy as np
import pytest

from pyali.utils import exact_disk, matlab_disk, strel_disk


# --------------------------------------------------------------------------- #
# structuring elements
# --------------------------------------------------------------------------- #
def _row_widths(se):
    return [int(r.sum()) for r in se]


def test_exact_disk_r2_is_the_13_pixel_disk():
    se = exact_disk(2)
    assert se.shape == (5, 5) and se.sum() == 13
    assert _row_widths(se) == [1, 3, 5, 3, 1]


def test_matlab_disk_matches_the_known_matlab_neighbourhoods():
    assert _row_widths(matlab_disk(3)) == [3, 5, 7, 7, 7, 5, 3]
    assert _row_widths(matlab_disk(5)) == [5, 7, 9, 9, 9, 9, 9, 7, 5]
    assert matlab_disk(2).shape == (5, 5) and matlab_disk(2).sum() == 13      # r < 3: exact disk
    assert matlab_disk(50).shape == (97, 97)
    se = matlab_disk(50)
    assert se[48, 48] and se[0, 48] and se[48, 0] and not se[0, 0]             # octagon: corners cut


def test_matlab_disk_r15_equals_the_existing_strel_disk():
    assert np.array_equal(matlab_disk(15), strel_disk(15))


# --------------------------------------------------------------------------- #
# gaussian_matlab
# --------------------------------------------------------------------------- #
def _gauss2d(sigma):
    half = int(np.ceil(2 * sigma))
    x = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g), half


def test_gaussian_matlab_kernel_size_and_equivalence_to_2d_correlate():
    from scipy import ndimage
    from pyali.preprocess import gaussian_matlab
    rng = np.random.default_rng(0)
    img = rng.random((40, 50))
    for sigma in (1.0, 2.0, 2.5):
        k2, half = _gauss2d(sigma)
        assert k2.shape == (2 * int(np.ceil(2 * sigma)) + 1,) * 2 and abs(k2.sum() - 1) < 1e-12
        ref = ndimage.correlate(img, k2, mode="nearest")
        out = gaussian_matlab(img, sigma)
        assert out.dtype == np.float64 and np.allclose(out, ref, atol=1e-12)


def test_gaussian_matlab_replicate_padding_keeps_a_constant_image():
    from pyali.preprocess import gaussian_matlab
    img = np.full((30, 30), 7.5)
    assert np.allclose(gaussian_matlab(img, 3.0), 7.5)
    ramp = np.tile(np.arange(30, dtype=float), (30, 1))           # linear ramp across columns
    out = gaussian_matlab(ramp, 2.0)
    assert np.allclose(out[:, 10:20], ramp[:, 10:20], atol=1e-9) # interior of a ramp is unchanged
    assert np.allclose(out[10:20, :], out[0, :][None], atol=1e-9) # constant down each column


# --------------------------------------------------------------------------- #
# masked_grey_opening / large_scale_background / raw_mean_reference
# --------------------------------------------------------------------------- #
def _blob(H, W, r, c, sigma):
    rr, cc = np.mgrid[:H, :W]
    return np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2.0 * sigma ** 2))


def test_masked_opening_without_mask_equals_plain_opening():
    from scipy import ndimage
    from pyali.preprocess import masked_grey_opening
    rng = np.random.default_rng(2)
    img = rng.random((40, 40)) * 10 + 100
    se = exact_disk(4)
    plain = ndimage.grey_opening(img, footprint=se, mode="nearest")
    out = masked_grey_opening(img, se, np.ones(img.shape, bool))
    assert np.allclose(out, plain)


def test_masked_opening_is_closer_to_truth_beside_a_masked_out_dark_band():
    """A ramp that is brightest at the well edge, with a dark band outside it. A plain opening
    erodes the edge pixels to the band's 0 and the dilation can only recover values from
    2*radius inside; the masked opening erodes over valid pixels only, so its edge error is
    bounded by radius * slope instead of 2 * radius * slope."""
    from scipy import ndimage
    from pyali.preprocess import masked_grey_opening
    H, W, r = 48, 64, 6
    cc = np.arange(W, dtype=float)[None, :].repeat(H, 0)
    img = 100.0 - 1.0 * (cc - 10)                       # slope 1 per column, 100 at the edge (col 10)
    img[:, :10] = 0.0                                   # dark band outside the well
    mask = np.ones((H, W), bool)
    mask[:, :10] = False
    se = exact_disk(r)
    plain = ndimage.grey_opening(img, footprint=se, mode="nearest")
    out = masked_grey_opening(img, se, mask)
    assert np.isnan(out[:, :10]).all() and np.isfinite(out[:, 10:]).all()
    edge_truth = img[24, 10]
    assert abs(out[24, 10] - edge_truth) <= r + 1e-9            # masked: within radius * slope
    assert abs(plain[24, 10] - edge_truth) >= 2 * r - 1e-9      # plain: pulled down by >= 2 * radius * slope
    assert out[24, 10] > plain[24, 10]
    assert np.allclose(out[:, 30:], plain[:, 30:])              # far from the band they agree
    assert np.isnan(masked_grey_opening(img, se, np.zeros((H, W), bool))).all()


def test_large_scale_background_removes_ramp_keeps_blob_and_respects_mask():
    from pyali.preprocess import large_scale_background
    H = W = 96
    rr, cc = np.mgrid[:H, :W]
    ramp = 100.0 + 0.5 * cc + 0.25 * rr
    blob = 40.0 * _blob(H, W, 48, 48, 2.0)
    img = ramp + blob
    bg = large_scale_background(img, radius=10, sigma=8.0)
    assert bg.dtype == np.float32 and bg.shape == (H, W)
    corrected = img - bg
    interior = (rr >= 12) & (rr < H - 12) & (cc >= 12) & (cc < W - 12)
    far = corrected[interior & (np.hypot(rr - 48, cc - 48) > 15)]
    assert np.abs(far).max() < 0.02 * 40.0              # ramp removed away from the border band
    assert abs(corrected[48, 48] - 40.0) < 0.06 * 40.0  # blob retained
    # mask-awareness: a dark band outside the mask must not drag the background down next to it.
    # (A normalized Gaussian over a one-sided window is still biased by slope * mean offset on a
    # ramp; that is inherent and small. The unmasked version, which averages in the zeros, is not.)
    mask = np.ones((H, W), bool)
    mask[:, :20] = False
    dark = img.copy()
    dark[:, :20] = 0.0
    bgm = large_scale_background(dark, radius=10, sigma=8.0, mask=mask)
    bgu = large_scale_background(dark, radius=10, sigma=8.0)
    assert np.isnan(bgm[:, :20]).all() and np.isfinite(bgm[:, 20:]).all()
    err_masked = np.abs(bgm[20:-20, 20:30] - ramp[20:-20, 20:30]).max()
    err_unmasked = np.abs(bgu[20:-20, 20:30] - ramp[20:-20, 20:30]).max()
    assert err_masked < 4.0 and err_unmasked > 20.0 and err_masked < 0.2 * err_unmasked
    assert np.allclose(large_scale_background(np.full((20, 20), 5.0), radius=3, sigma=0), 5.0)  # sigma 0 -> opening only


# --------------------------------------------------------------------------- #
# hysteresis_mask
# --------------------------------------------------------------------------- #
def _noise_field(H=200, W=200, bg=20.0, sigma=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return bg + sigma * rng.standard_normal((H, W))


def _components(BW):
    from scipy import ndimage
    return ndimage.label(BW, structure=np.ones((3, 3)))[1]


def test_hysteresis_calibrates_background_and_noise_from_the_field():
    from pyali.segmentation import hysteresis_mask
    img = _noise_field(bg=20.0, sigma=3.0)
    BW, info = hysteresis_mask(img, smooth_sigma=2.0)
    # smoothing with sigma 2 (9x9 kernel) shrinks white noise by ~1/(2*sqrt(pi)*sigma) -> ~0.14
    expected_sigma = 3.0 / (2.0 * np.sqrt(np.pi) * 2.0)
    assert abs(info["bg_center"] - 20.0) < 0.5 * expected_sigma
    assert abs(info["sigma_bg"] - expected_sigma) < 0.25 * expected_sigma
    assert info["th_high"] > info["th_low"] > info["bg_center"]
    assert BW.sum() == 0 and _components(BW) == 0                      # pure noise -> nothing


def test_hysteresis_keeps_bright_blob_rejects_isolated_dim_blob_keeps_attached_halo():
    from pyali.segmentation import hysteresis_mask
    H = W = 200
    img = _noise_field(H, W, bg=20.0, sigma=3.0, seed=1)
    _, info0 = hysteresis_mask(img)
    s = info0["sigma_bg"]                                            # the smoothed-noise sigma
    bright = 12 * s * _blob(H, W, 60, 60, 4.0)                        # >= 6 sigma core
    dim_isolated = 4.5 * s * _blob(H, W, 140, 140, 4.0)               # between 3 and 6 sigma: no seed
    halo = 4.5 * s * _blob(H, W, 60, 72, 4.0)                         # dim, but touching the bright blob
    BW, info = hysteresis_mask(img + bright + dim_isolated + halo)
    assert BW[60, 60] and not BW[140, 140]
    assert BW[60, 72]                                                 # halo kept via its bright neighbour
    n = _components(BW)
    assert n == 1                                                     # one object: bright core + halo
    assert BW[60, 60 - 10] or BW[60, 72 + 10] or BW.sum() > 60        # grew out to the 3-sigma extent


def test_hysteresis_respects_mask_and_min_area_and_empty_input():
    from pyali.segmentation import hysteresis_mask
    H = W = 120
    img = _noise_field(H, W, bg=10.0, sigma=2.0, seed=3)
    img += 40.0 * _blob(H, W, 30, 30, 3.0) + 40.0 * _blob(H, W, 90, 90, 3.0)
    mask = np.ones((H, W), bool)
    mask[:, 60:] = False                                             # the right half is outside the well
    BW, info = hysteresis_mask(img, mask=mask)
    assert BW[30, 30] and not BW[:, 60:].any()
    BW_big, _ = hysteresis_mask(img, min_area=10_000)                # nothing is that large
    assert BW_big.sum() == 0
    BW0, info0 = hysteresis_mask(np.full((20, 20), np.nan))
    assert BW0.sum() == 0 and np.isnan(info0["bg_center"])
    assert hysteresis_mask(np.full((20, 20), 5.0))[0].sum() == 0     # constant image: no seeds


# --------------------------------------------------------------------------- #
# soma_watershed_mask
# --------------------------------------------------------------------------- #
def _populated_field(H, W, seed, extra=()):
    """A field dense enough that the mask's fixed top-5% budget is spent on real somata: a grid
    of bright blobs (plus ``extra`` (row, col, amp) blobs) over faint noise. On a nearly empty
    field the 5% cut would be filled by noise pixels instead; that is the documented behaviour
    of this mask (the hysteresis AND removes them), not something to test here."""
    img = 100.0 + 0.05 * _noise_field(H, W, bg=0.0, sigma=1.0, seed=seed)
    centres = [(r, c) for r in range(15, H - 10, 24) for c in range(15, W - 10, 24)]
    for r, c in centres:
        img += 50.0 * _blob(H, W, r, c, 2.5)
    for r, c, amp in extra:
        img += amp * _blob(H, W, r, c, 2.5)
    return img, centres


def _label_at(BW, r, c):
    from scipy import ndimage
    return ndimage.label(BW, structure=np.ones((3, 3)))[0][r, c]


def test_soma_watershed_splits_two_touching_blobs_on_the_midline():
    from pyali.segmentation import soma_watershed_mask
    H = W = 160
    img, centres = _populated_field(H, W, seed=4, extra=[(100, 72, 50.0), (100, 86, 50.0)])  # 14 px apart
    BW = soma_watershed_mask(img, tophat_radius=3, background_radius=20, bg_sigma=8.0, percentile=95.0)
    assert BW[100, 72] and BW[100, 86]
    assert not BW[100, 79]                                           # the ridge line between them is cut
    assert _label_at(BW, 100, 72) != _label_at(BW, 100, 86)          # two basins
    assert all(BW[r, c] for r, c in centres)                         # every planted soma is foreground


def test_soma_watershed_one_component_per_soma_and_degenerate_inputs():
    from pyali.segmentation import soma_watershed_mask
    H = W = 160
    img, centres = _populated_field(H, W, seed=5)
    BW = soma_watershed_mask(img, background_radius=20, bg_sigma=8.0)
    assert all(BW[r, c] for r, c in centres)
    assert _components(BW) == len(centres)                           # no noise components, no splits
    assert BW.sum() <= 0.06 * H * W                                  # the 5% area rule bounds the mask
    assert soma_watershed_mask(np.full((40, 40), 3.0), background_radius=10).sum() == 0   # constant


def test_soma_watershed_faint_peak_below_the_cut_is_not_foreground():
    """A faint peak is an h-maxima marker only if it stands >= h above its surround, and even a
    marker claims no pixels unless they pass the top-5% cut: a faint blob yields no component."""
    from pyali.segmentation import soma_watershed_mask
    H = W = 160
    img, centres = _populated_field(H, W, seed=6, extra=[(123, 123, 0.3)])      # faint vs 50
    BW = soma_watershed_mask(img, background_radius=20, bg_sigma=8.0)
    assert not BW[123, 123]
    assert _components(BW) == len(centres)


# --------------------------------------------------------------------------- #
# cell_segmentation (combined) and the pipeline
# --------------------------------------------------------------------------- #
def test_cell_segmentation_one_region_per_soma_with_info():
    from pyali.params import Params
    from pyali.preprocess import large_scale_background
    from pyali.segmentation import cell_segmentation
    H = W = 160
    img, centres = _populated_field(H, W, seed=7)
    p = Params(background_radius=20, background_sigma=8.0, ws_bg_sigma=8.0)
    ref_corrected = img - large_scale_background(img, p.background_radius, p.background_sigma)
    regions, BW, fps, info = cell_segmentation(img, ref_corrected, p)
    assert len(regions) == len(centres) and len(fps) == len(centres)
    cen = np.array([[r["Centroid"][1] - 1, r["Centroid"][0] - 1] for r in regions])
    for r, c in centres:
        assert np.hypot(cen[:, 0] - r, cen[:, 1] - c).min() < 1.5
    for k in ("bg_center", "sigma_bg", "th_high", "th_low", "n_seed", "n_low",
              "n_px_intensity", "n_px_watershed", "n_px_final"):
        assert k in info
    assert info["n_px_final"] == BW.sum() <= min(info["n_px_intensity"], info["n_px_watershed"])


def test_cell_segmentation_touching_somata_are_two_regions():
    from pyali.params import Params
    from pyali.preprocess import large_scale_background
    from pyali.segmentation import cell_segmentation
    H = W = 160
    img, centres = _populated_field(H, W, seed=8, extra=[(100, 72, 50.0), (100, 86, 50.0)])
    p = Params(background_radius=20, background_sigma=8.0, ws_bg_sigma=8.0)
    ref_corrected = img - large_scale_background(img, p.background_radius, p.background_sigma)
    regions, BW, _, _ = cell_segmentation(img, ref_corrected, p)
    assert _label_at(BW, 100, 72) != _label_at(BW, 100, 86) and BW[100, 72] and BW[100, 86]
    assert len(regions) == len(centres) + 2


def test_pipeline_segments_planted_cells_and_records_segmentation_info(tmp_path):
    import json
    from pyali.pipeline import process_fov
    from tests.synthetic import make_movie, write_fov, match_centers
    data = make_movie(seed=11)
    write_fov(tmp_path, data["movie"])
    out_dir = str(tmp_path / "analysis")
    out = process_fov(str(tmp_path), out_dir=out_dir, p=data["params"], save=True, verbose=False)
    cen = np.array([[r["Centroid"][1] - 1, r["Centroid"][0] - 1] for r in out["regions"]]).reshape(-1, 2)
    j, _ = match_centers(cen, data["truth"]["centers"], tol=3.0)
    assert len(set(j[j >= 0].tolist())) == len(data["truth"]["centers"])
    assert out["reference_corrected"].dtype == np.float32
    assert out["background_map"].shape == out["reference_raw"].shape
    meta = json.load(open(tmp_path / "analysis" / "fov_metadata.json"))
    seg = meta["segmentation"]
    assert seg["th_high"] > seg["th_low"] > seg["bg_center"] and seg["n_px_final"] > 0
    assert meta["params"]["seg_high_sigma"] == 6.0 and "sharpen_k" not in meta["params"]
    assert "background" in meta["timing_s"] and "segmentation" in meta["timing_s"]


def test_raw_mean_reference_uses_only_the_first_frames_and_leaves_movie_untouched():
    from pyali.preprocess import raw_mean_reference
    movie = np.ones((20, 8, 8), np.float32)
    movie[10:] = 1000.0
    before = movie.copy()
    ref = raw_mean_reference(movie, n_frames=10)
    assert ref.dtype == np.float64 and np.allclose(ref, 1.0)
    assert np.allclose(raw_mean_reference(movie, n_frames=300), (10 + 10 * 1000.0) / 20)   # clipped to T
    assert np.array_equal(movie, before)
