"""Per-FOV results: the only writer and the only loader of pyali's on-disk outputs.

Layout of one FOV output directory (schema version 2)::

    cells.parquet        one row per cell: cell_id, experiment_id, plate, well, fov, fov_id,
                         fov_index, local_cell_id, trace_row, x_px, y_px, region_id, n_spikes,
                         source (spiking | no_ap | no_com | all_noise), weighted_brightness
    cells.csv            the same table, for eyeballing
    spikes.parquet       one row per localized AP: region_id, frame (0-based), row, col (COM,
                         1-indexed), amplitude, cell_row (-1 = not assigned to a cell)
    signals.h5           /traces/raw          float32 [T, N] on disk (MATLAB reads it as [N, T]);
                                              chunks (min(2048, T), min(256, N)), gzip 4
                         /footprints/values   float32 [nnz]   sparse COO over all cells
                         /footprints/pixel    uint32  [nnz]   C-order pixel index = row * W + col
                         /footprints/offsets  uint32  [N+1]   cell k owns values[offsets[k]:offsets[k+1]]
                         /images/reference    float32 [H, W]  correlation-weighted reference
                         /images/binary_map   uint8   [H, W]  segmentation mask
                         /images/brightness_reference  float32 [H, W]  raw-unit mean minus its
                                              large-scale background: the segmentation input and
                                              the x in weighted_brightness
                         attrs: height, width, n_frames, fps, pyali_git, schema_version, ...
    fov_metadata.json    written separately by :mod:`pyali.metadata`

``cell_id = <fov_id>_C<local_cell_id:04d>`` with ``fov_id`` from :class:`pyali.metadata.Identity`.
``weighted_brightness = sum(w * x) / sum(w)`` over the cell's footprint pixels (``w`` = footprint
values, ``x`` = ``/images/brightness_reference``); NaN when the cell has no positive weight.

``/traces/raw`` is byte-compatible with the Octave corpus's ``signals.h5``. Trace polarity is
pyali's (spikes positive). Footprints are stored sparse because each cell's plane is an 11x11
window in a field of zeros; :meth:`SparseFootprints.dense` rebuilds the old ``(H, W, N)`` array
for the one consumer that needs it (registration).
"""
from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = 2

CELLS_FILE = "cells.parquet"
CELLS_CSV = "cells.csv"
SPIKES_FILE = "spikes.parquet"
SIGNALS_FILE = "signals.h5"

# string columns are plain Python str in an object column (what parquet round-trips to)
CELLS_DTYPES = {"cell_id": "object", "experiment_id": "object", "plate": "uint16",
                "well": "object", "fov": "uint16", "fov_id": "object", "fov_index": "uint32",
                "local_cell_id": "uint16", "trace_row": "uint32", "x_px": "float32",
                "y_px": "float32", "region_id": "uint32", "n_spikes": "uint32",
                "source": "object", "weighted_brightness": "float32"}
SOURCES = ("spiking", "no_ap", "no_com", "all_noise")
IMAGE_KEYS = ("reference", "binary_map", "brightness_reference")
SPIKES_DTYPES = {"region_id": "uint32", "frame": "uint32", "row": "float32", "col": "float32",
                 "amplitude": "float32", "cell_row": "int32"}


# --------------------------------------------------------------------------- #
# Sparse footprints
# --------------------------------------------------------------------------- #
@dataclass
class SparseFootprints:
    """All cells' footprints as one COO list: ``values[offsets[k]:offsets[k+1]]`` at C-order
    pixel indices ``pixel[offsets[k]:offsets[k+1]]`` is cell ``k``'s plane."""
    values: np.ndarray          # float32 [nnz]
    pixel: np.ndarray           # uint32 [nnz], row * width + col
    offsets: np.ndarray         # uint32 [N + 1]
    height: int
    width: int

    def n(self):
        return int(len(self.offsets)) - 1

    def plane(self, k):
        """Dense ``(H, W)`` float32 plane of cell ``k``."""
        if not 0 <= k < self.n():
            raise IndexError(f"cell {k} out of range for {self.n()} cells")
        a, b = int(self.offsets[k]), int(self.offsets[k + 1])
        out = np.zeros(self.height * self.width, np.float32)
        out[self.pixel[a:b]] = self.values[a:b]
        return out.reshape(self.height, self.width)

    def dense(self):
        """``(H, W, N)`` float64, the old ``footprint`` array (registration is the only user)."""
        out = np.zeros((self.height, self.width, self.n()), np.float64)
        for k in range(self.n()):
            out[:, :, k] = self.plane(k)
        return out

    @staticmethod
    def from_dense(fp, height=None, width=None):
        """Pack an ``(H, W, N)`` array; ``height``/``width`` are only needed when ``fp`` is empty
        and 2-D-less (``(H, W, 0)`` arrays carry them already)."""
        fp = np.asarray(fp)
        if fp.ndim != 3:
            raise ValueError(f"footprint must be (H, W, N), got shape {fp.shape}")
        H, W, N = fp.shape
        height = H if height is None else int(height)
        width = W if width is None else int(width)
        values, pixel, offsets = [], [], [0]
        for k in range(N):
            flat = np.ascontiguousarray(fp[:, :, k]).ravel()           # C order: row * W + col
            nz = np.flatnonzero(flat)
            values.append(flat[nz].astype(np.float32))
            pixel.append(nz.astype(np.uint32))
            offsets.append(offsets[-1] + len(nz))
        return SparseFootprints(
            values=np.concatenate(values) if values else np.zeros(0, np.float32),
            pixel=np.concatenate(pixel) if pixel else np.zeros(0, np.uint32),
            offsets=np.asarray(offsets, np.uint32), height=height, width=width)


def weighted_brightness(sp, ref):
    """Per-cell footprint-weighted brightness: ``sum(w * x) / sum(w)`` over each cell's
    footprint pixels, with ``w`` the footprint values and ``x`` the ``(H, W)`` brightness
    reference (raw mean minus :func:`pyali.preprocess.large_scale_background`). Pixels with ``w <= 0`` or a
    non-finite ``w``/``x`` are ignored; a cell with no usable pixel gets ``NaN``.

    Returns ``(N,)`` float32.
    """
    ref = np.asarray(ref, np.float64)
    if ref.shape != (sp.height, sp.width):
        raise ValueError(f"reference shape {ref.shape} != ({sp.height}, {sp.width})")
    flat = ref.ravel()
    out = np.full(sp.n(), np.nan, np.float32)
    for k in range(sp.n()):
        a, b = int(sp.offsets[k]), int(sp.offsets[k + 1])
        w = sp.values[a:b].astype(np.float64)
        x = flat[sp.pixel[a:b]]
        m = (w > 0) & np.isfinite(w) & np.isfinite(x)
        if m.any():
            out[k] = (w[m] * x[m]).sum() / w[m].sum()
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _typed_frame(columns, dtypes):
    """DataFrame with exactly ``dtypes``'s columns, cast; zero rows when ``columns`` is empty."""
    import pandas as pd
    return pd.DataFrame({name: pd.Series(np.asarray(columns.get(name, []), dtype=dt), dtype=dt)
                         for name, dt in dtypes.items()})


def make_cells(centers, region_id, n_spikes, source, brightness, identity):
    """The ``cells`` table for ``N`` cells.

    ``centers`` ``(N, 2) = [row, col]`` (1-indexed); ``region_id``, ``n_spikes``, ``brightness``
    ``(N,)``; ``source`` a length-``N`` sequence over :data:`SOURCES`; ``identity`` a
    :class:`pyali.metadata.Identity` (shared by every row; it supplies ``fov_id`` and the
    ``cell_id`` prefix).
    """
    centers = np.asarray(centers, float).reshape(-1, 2)
    N = len(centers)
    region_id, n_spikes = np.asarray(region_id).reshape(-1), np.asarray(n_spikes).reshape(-1)
    brightness = np.asarray(brightness, np.float32).reshape(-1)
    source = [str(s) for s in source]
    if not (len(region_id) == len(n_spikes) == len(source) == len(brightness) == N):
        raise ValueError(f"cells: {N} centers, {len(region_id)} region ids, {len(n_spikes)} spike "
                         f"counts, {len(source)} sources, {len(brightness)} brightness values")
    bad = sorted(set(source) - set(SOURCES))
    if bad:
        raise ValueError(f"cells: unknown source values {bad}; expected one of {SOURCES}")
    local = np.arange(1, N + 1)
    return _typed_frame(dict(cell_id=[identity.cell_id(i) for i in local],
                             experiment_id=[identity.experiment_id] * N,
                             plate=[int(identity.plate)] * N, well=[identity.well] * N,
                             fov=[int(identity.fov)] * N, fov_id=[identity.fov_id] * N,
                             fov_index=[int(identity.fov_index)] * N,
                             local_cell_id=local, trace_row=np.arange(N),
                             x_px=centers[:, 1], y_px=centers[:, 0],
                             region_id=region_id, n_spikes=n_spikes,
                             source=source, weighted_brightness=brightness), CELLS_DTYPES)


def pack_spikes(APs, COMs, spike_cell_row, region_of_spike, region_of_ap):
    """The ``spikes`` table: one row per localized spike (a ``COMs`` row), with the amplitude of
    the AP it came from joined by ``(region, frame)``.

    ``APs`` ``(nAP, 4) = [row, col, frame, amplitude]`` with ``region_of_ap`` ``(nAP,)``;
    ``COMs`` ``(nCOM, 3) = [row, col, frame]`` with ``region_of_spike`` and ``spike_cell_row``
    ``(nCOM,)``. Frames are unique within a region (``dedup_close_peaks``), so the key is unique;
    a spike with no AP at its key (should not happen) gets a NaN amplitude.
    """
    APs = np.asarray(APs, float).reshape(-1, 4)
    COMs = np.asarray(COMs, float).reshape(-1, 3)
    region_of_ap = np.asarray(region_of_ap).reshape(-1)
    region_of_spike = np.asarray(region_of_spike).reshape(-1)
    spike_cell_row = np.asarray(spike_cell_row).reshape(-1)
    if len(region_of_ap) != len(APs):
        raise ValueError(f"spikes: {len(APs)} APs but {len(region_of_ap)} AP region ids")
    if not (len(region_of_spike) == len(spike_cell_row) == len(COMs)):
        raise ValueError(f"spikes: {len(COMs)} COMs, {len(region_of_spike)} region ids, "
                         f"{len(spike_cell_row)} cell rows")
    amp_of = {(int(r), int(f)): a for r, f, a in zip(region_of_ap, APs[:, 2], APs[:, 3])}
    amplitude = np.array([amp_of.get((int(r), int(f)), np.nan)
                          for r, f in zip(region_of_spike, COMs[:, 2])], np.float32)
    return _typed_frame(dict(region_id=region_of_spike, frame=COMs[:, 2], row=COMs[:, 0],
                             col=COMs[:, 1], amplitude=amplitude, cell_row=spike_cell_row),
                        SPIKES_DTYPES)


# --------------------------------------------------------------------------- #
# Write / load
# --------------------------------------------------------------------------- #
@dataclass
class Results:
    """Everything in one FOV output directory (see the module docstring)."""
    cells: object                # DataFrame, CELLS_DTYPES columns
    traces: np.ndarray           # (N, T) float32, spikes positive
    centers: np.ndarray          # (N, 2) float64 = [row, col], 1-indexed (from cells y_px / x_px)
    footprints: SparseFootprints
    spikes: object               # DataFrame, SPIKES_DTYPES columns
    images: dict                 # reference / brightness_reference (H, W) float32, binary_map uint8
    attrs: dict                  # signals.h5 attributes


def git_sha():
    """Commit of the checked-out pyali, or ``''`` when git is unavailable."""
    import os
    import subprocess
    try:
        out = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                              "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                                   # noqa: BLE001
        return ""


def _check_frame(df, dtypes, name):
    if list(df.columns) != list(dtypes):
        raise ValueError(f"{name} columns {list(df.columns)} != {list(dtypes)}")
    return df.astype(dtypes)


def write_results(out_dir, *, cells_df, traces, footprints, spikes_df, images, attrs):
    """Write one FOV's results into ``out_dir`` (created if needed). Returns ``out_dir``.

    ``cells_df`` / ``spikes_df`` must carry exactly the schema columns (see :func:`make_cells`,
    :func:`pack_spikes`); ``traces`` is ``(N, T)``; ``footprints`` is the dense ``(H, W, N)``
    array (packed here) or an already packed :class:`SparseFootprints`; ``images``
    ``{reference, binary_map, brightness_reference}``, each ``(H, W)``; ``attrs`` must include ``fps`` and may carry
    any other scalars (stored as HDF5 attributes; ``None`` becomes ``''``). ``height``, ``width``,
    ``n_frames``, ``schema_version`` and ``pyali_git`` are filled in here. An ``N == 0`` FOV
    writes every file with zero-length datasets and zero-row tables.
    """
    import os
    import h5py

    os.makedirs(out_dir, exist_ok=True)
    cells_df = _check_frame(cells_df, CELLS_DTYPES, "cells")
    spikes_df = _check_frame(spikes_df, SPIKES_DTYPES, "spikes")
    traces = np.asarray(traces)
    if traces.ndim != 2:
        raise ValueError(f"traces must be (N, T), got shape {traces.shape}")
    N, T = traces.shape
    sp = footprints if isinstance(footprints, SparseFootprints) else \
        SparseFootprints.from_dense(footprints)
    H, W = sp.height, sp.width
    if sp.n() != N or len(cells_df) != N:
        raise ValueError(f"{N} traces, {sp.n()} footprints, {len(cells_df)} cell rows")
    for key in IMAGE_KEYS:
        if key not in images:
            raise ValueError(f"images is missing {key!r}; need {IMAGE_KEYS}")
        if np.asarray(images[key]).shape != (H, W):
            raise ValueError(f"images[{key!r}] shape {np.asarray(images[key]).shape} != ({H}, {W})")

    cells_df.to_parquet(os.path.join(out_dir, CELLS_FILE), index=False)
    cells_df.to_csv(os.path.join(out_dir, CELLS_CSV), index=False)
    spikes_df.to_parquet(os.path.join(out_dir, SPIKES_FILE), index=False)

    with h5py.File(os.path.join(out_dir, SIGNALS_FILE), "w") as f:
        kw = {}
        if N and T:                                   # HDF5 rejects zero-size chunks
            kw = dict(chunks=(min(2048, T), min(256, N)), compression="gzip",
                      compression_opts=4)
        f.create_dataset("/traces/raw", data=np.ascontiguousarray(traces.T, np.float32), **kw)
        f.create_dataset("/footprints/values", data=sp.values)
        f.create_dataset("/footprints/pixel", data=sp.pixel)
        f.create_dataset("/footprints/offsets", data=sp.offsets)
        f.create_dataset("/images/reference", data=np.asarray(images["reference"], np.float32),
                         compression="gzip", compression_opts=4)
        f.create_dataset("/images/binary_map", data=np.asarray(images["binary_map"], np.uint8),
                         compression="gzip", compression_opts=4)
        f.create_dataset("/images/brightness_reference",
                         data=np.asarray(images["brightness_reference"], np.float32),
                         compression="gzip", compression_opts=4)
        full = dict(attrs)
        full.update(height=int(H), width=int(W), n_frames=int(T), fps=float(attrs["fps"]),
                    schema_version=SCHEMA_VERSION, pyali_git=git_sha())
        for k, v in full.items():
            f.attrs[k] = "" if v is None else v
    return out_dir


def _py(v):
    """HDF5 attribute -> plain Python scalar / str."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_results(fov_dir):
    """Read a FOV output directory (or the path of its ``signals.h5``) back as :class:`Results`."""
    import os
    import h5py
    import pandas as pd

    if os.path.isfile(fov_dir):
        fov_dir = os.path.dirname(os.path.abspath(fov_dir))
    cells = pd.read_parquet(os.path.join(fov_dir, CELLS_FILE))
    spikes = pd.read_parquet(os.path.join(fov_dir, SPIKES_FILE))
    with h5py.File(os.path.join(fov_dir, SIGNALS_FILE), "r") as f:
        attrs = {k: _py(v) for k, v in f.attrs.items()}
        traces = np.ascontiguousarray(np.asarray(f["/traces/raw"]).T, np.float32)
        footprints = SparseFootprints(
            values=np.asarray(f["/footprints/values"], np.float32),
            pixel=np.asarray(f["/footprints/pixel"], np.uint32),
            offsets=np.asarray(f["/footprints/offsets"], np.uint32),
            height=int(attrs["height"]), width=int(attrs["width"]))
        images = dict(reference=np.asarray(f["/images/reference"], np.float32),
                      binary_map=np.asarray(f["/images/binary_map"], np.uint8),
                      brightness_reference=np.asarray(f["/images/brightness_reference"],
                                                      np.float32))
    centers = np.column_stack([cells["y_px"].to_numpy(), cells["x_px"].to_numpy()]).astype(np.float64)
    return Results(cells=cells, traces=traces, centers=centers.reshape(-1, 2),
                   footprints=footprints, spikes=spikes, images=images, attrs=attrs)
