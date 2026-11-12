"""pytest configuration: make ``import pyali`` work from a checkout (the package is not installed).

Run with the project venv: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/ -q``.

The real-data tests need a FOV directory holding ``frames1.bin``. Point ``PYALI_REAL_FOV`` at
one (a spatial crop made by ``tests/real_fov.py`` is enough); they are skipped when it is unset.
"""
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@pytest.fixture(scope="session")
def real_fov_dir():
    path = os.environ.get("PYALI_REAL_FOV")
    if not path or not os.path.isfile(os.path.join(path, "frames1.bin")):
        pytest.skip("set PYALI_REAL_FOV to a directory holding frames1.bin")
    return path
