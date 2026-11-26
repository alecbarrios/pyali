"""Focus QC score: sharp vs defocused mean images, and its presence in every run's metadata."""
import json
import os

import numpy as np

from pyali.preprocess import focus_score


def _field(H=160, W=160, seed=0, blur=0.0):
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 60.0)
    for _ in range(40):                                         # soma-like bright spots
        r, c = rng.integers(10, H - 10), rng.integers(10, W - 10)
        rr, cc = np.mgrid[:H, :W]
        img += 120.0 * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * 2.0 ** 2))
    if blur:
        img = ndimage.gaussian_filter(img, blur)                # defocus
    return img + rng.normal(0, 1.0, size=(H, W))


def test_focus_score_separates_sharp_from_defocused_and_is_exposure_invariant():
    sharp, soft = _field(), _field(blur=4.0)
    s, d = focus_score(sharp), focus_score(soft)
    assert s > 5 * d                                            # defocus drops the score by far more than noise
    assert abs(focus_score(2.0 * sharp) - s) < 0.2 * s          # same field, double exposure: ~same score
    assert focus_score(np.full((50, 50), 7.0)) == 0.0
    assert np.isnan(focus_score(np.full((5, 5), np.nan)))


def test_every_run_records_a_focus_score(tmp_path):
    from pyali.pipeline import process_fov
    from pyali.results import load_results
    from tests.synthetic import make_movie, write_fov
    data = make_movie(seed=4)
    write_fov(tmp_path, data["movie"])
    out_dir = str(tmp_path / "analysis")
    out = process_fov(str(tmp_path), out_dir=out_dir, p=data["params"], save=True, verbose=False)
    assert np.isfinite(out["focus_score"]) and out["focus_score"] > 0
    meta = json.load(open(os.path.join(out_dir, "fov_metadata.json")))
    assert meta["focus_score"] == out["focus_score"]
    assert load_results(out_dir).attrs["focus_score"] == out["focus_score"]
