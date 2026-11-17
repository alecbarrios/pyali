"""``scripts/run_corpus.py``: the done-marker is ``cells.parquet`` (Cycle B1).

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_run_corpus.py -q``
"""
import os
import subprocess
import sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _mod():
    sys.path.insert(0, _SCRIPTS)
    import run_corpus
    return run_corpus


def test_done_marker_is_cells_parquet():
    assert _mod().RESULT == "cells.parquet"


def test_existing_keys_parses_the_new_marker(monkeypatch):
    rc = _mod()
    listing = "\n".join([
        "2026-11-16 10:00:00     123456 AB/pyali_analysis/20260715/fovA/cells.parquet",
        "2026-11-16 10:00:00    9876543 AB/pyali_analysis/20260715/fovA/signals.h5",
        "2026-11-16 10:00:00      54321 AB/pyali_analysis/20260716/fovB/ALI_Result.mat",   # legacy: not done
        "2026-11-16 10:00:00     123456 AB/pyali_analysis/20260716/fovC/cells.parquet",
        "2026-11-16 10:00:00        100 AB/pyali_analysis/20260716/fovD/fov_metadata.json",
    ]) + "\n"

    def fake_run(cmd, capture_output, text):
        assert cmd[:3] == ["aws", "s3", "ls"]
        return subprocess.CompletedProcess(cmd, 0, stdout=listing, stderr="")
    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    assert rc.existing_keys("s3://spatial-technology-platform/AB/pyali_analysis") == {
        ("20260715", "fovA"), ("20260716", "fovC")}
