"""Real-data oracles: the production pipeline on a spatial crop of a real 443screen2 FOV.

A crop keeps every frame, so the acquisition profile (baseline windows up to frame 8241) applies
unchanged; only ``nrow``/``ncol`` are overridden. Make the crops with ``tests/real_fov.py`` and
point ``PYALI_REAL_FOV`` at the cell-field crop and, optionally, ``PYALI_REAL_DARK`` at a crop
taken outside the illuminated well (a real zero-region input). Both are skipped when unset.
``PYALI_REAL_METRICS`` names a JSON file to receive the run metrics for the review ledger.

Run: ``PYALI_REAL_FOV=... /home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_real_fov.py -q -s``
"""
import dataclasses
import json
import os
import time

import numpy as np
import pytest

from pyali.params import Params
from pyali.pipeline import process_fov


def _params_for(crop_dir):
    with open(os.path.join(crop_dir, "crop.json")) as f:
        meta = json.load(f)
    return Params.profile_443screen2(nrow=meta["size"], ncol=meta["size"]), meta


def _run(crop_dir, p, **overrides):
    q = dataclasses.replace(p, **overrides)
    t0 = time.time()
    out = process_fov(crop_dir, out_dir=None, p=q, save=False, verbose=False)
    return out, round(time.time() - t0, 1)


def _metrics(out, seconds, meta, label):
    ct = np.asarray(out["cell_traces"])
    return dict(label=label, seconds=seconds, n_regions=len(out["regions"]),
                n_cells=int(ct.shape[0]), n_aps=int(np.asarray(out["APs"]).reshape(-1, 4).shape[0]),
                T=int(ct.shape[1]) if ct.ndim == 2 else None, crop=meta)


def _record(metrics):
    path = os.environ.get("PYALI_REAL_METRICS")
    print("\n[real]", json.dumps(metrics))
    if path:
        existing = []
        if os.path.isfile(path):
            with open(path) as f:
                existing = json.load(f)
        existing = [m for m in existing if m.get("label") != metrics["label"]] + [metrics]
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)


@pytest.fixture(scope="module")
def real(real_fov_dir):
    p, meta = _params_for(real_fov_dir)
    out, seconds = _run(real_fov_dir, p)
    m = _metrics(out, seconds, meta, "cells")
    _record(m)
    return dict(out=out, p=p, meta=meta, metrics=m, dir=real_fov_dir)


@pytest.fixture(scope="module")
def real_dark_dir():
    path = os.environ.get("PYALI_REAL_DARK")
    if not path or not os.path.isfile(os.path.join(path, "frames1.bin")):
        pytest.skip("set PYALI_REAL_DARK to a crop taken outside the illuminated well")
    return path


def test_real_crop_produces_cells(real):
    out, p = real["out"], real["p"]
    H, W = p.nrow, p.ncol
    fp = np.asarray(out["footprint"])
    fc = np.asarray(out["footprint_center"], float).reshape(-1, 2)
    ct = np.asarray(out["cell_traces"])
    assert len(out["regions"]) > 0
    assert fp.shape[:2] == (H, W) and fp.shape[2] == ct.shape[0] == fc.shape[0]
    # every region yields at least one footprint (fallback or one per cluster)
    assert ct.shape[0] >= len(out["regions"])
    assert np.isfinite(ct).all()
    assert (fc[:, 0] >= 1).all() and (fc[:, 0] <= H).all() and (fc[:, 1] >= 1).all() and (fc[:, 1] <= W).all()
    empty = [k for k in range(fp.shape[2]) if not np.any(fp[:, :, k] != 0)]
    assert not empty, f"footprints with no nonzero pixel: {empty}"


def test_real_patch_local_filter_is_bit_identical(real):
    ref = real["out"]
    alt, seconds = _run(real["dir"], real["p"], patch_local_filter=False)
    _record(_metrics(alt, seconds, real["meta"], "cells_global_filter"))
    for key in ("footprint", "footprint_center", "cell_traces"):
        assert np.array_equal(np.asarray(ref[key]), np.asarray(alt[key])), key


def test_real_dark_crop_runs_with_few_or_no_regions(real_dark_dir):
    p, meta = _params_for(real_dark_dir)
    out, seconds = _run(real_dark_dir, p)
    m = _metrics(out, seconds, meta, "dark")
    _record(m)
    ct = np.asarray(out["cell_traces"])
    assert np.asarray(out["footprint"]).shape[2] == ct.shape[0]
    assert np.isfinite(ct).all()
