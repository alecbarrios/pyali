"""Provenance sidecar: record how a FOV was analysed, beside its outputs.

``fov_metadata.json`` lands in the output directory next to ``ALI_Result.mat`` so the settings a
result was produced with travel with it, and so a later aggregation over many FOVs is a single
glob. It records the input directory, every :class:`pyali.params.Params` field, the package
version, the git commit of the package when it is available, a UTC timestamp, and whatever
extras the pipeline passes (region and cell counts, frames analysed).
"""
import dataclasses
import datetime
import json
import os
import subprocess


def _git_sha():
    """Short commit hash of the checkout this package is imported from, or ``None``."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _jsonable(v):
    """Tuples -> lists (recursively) and numpy scalars -> Python scalars, for ``json.dump``."""
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return v.item()
        except (TypeError, ValueError):
            return v
    return v


def fov_metadata(fov_dir, params=None, extras=None):
    """Metadata record for one FOV, ready to write beside its analysis outputs.

    Parameters
    ----------
    fov_dir : str
        The FOV directory (the one holding ``frames1.bin``).
    params : Params, optional
        Records the settings the FOV was actually analysed with, so results stay interpretable.
    extras : dict, optional
        Merged in last (counts, timings, run ids, ...).
    """
    from . import __version__
    meta = {"fov_dir": os.path.abspath(fov_dir)}
    if params is not None:
        meta["analysis"] = _jsonable(dataclasses.asdict(params))
    meta["pyali_version"] = __version__
    meta["git_sha"] = _git_sha()
    meta["written_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if extras:
        meta.update(_jsonable(dict(extras)))
    return meta


def write_metadata(out_dir, meta, name="fov_metadata.json"):
    """Write ``meta`` into ``out_dir`` -- the same directory as ``ALI_Result.mat``, so metadata
    and results sit at identical depth and aggregation downstream is a single glob."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False, default=str)
    return path
