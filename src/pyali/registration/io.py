"""Load exactly what's on disk for cross-modal registration.

Voltage side (per FOV): a raw-mean anatomical reference, the absolute microscope stage position,
and — only when needed — the pyali per-cell footprints/traces. SBS side: the JF608 anchor tiles,
the reconciled cell masks, per-tile stage metadata, and the ``sbs_info ⟕ cells`` genotype table.

Note this is ``pyali.registration.io`` — distinct from ``pyali.io`` (the movie/.mat reader), which
is pulled in lazily via ``from ..io import open_bin_memmap``.

Conventions: image arrays are ``[H, W]``; points are ``(row, col)`` here on the SBS/voltage image
side and converted to geometric ``(x, y)`` only inside ``coordinates``. Heavy libraries
(``tifffile``, ``h5py``) and heavy pyali siblings (``pipeline.process_fov``) are imported lazily so
importing this module stays cheap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..params import Params


# --------------------------------------------------------------------------- #
# voltage side (per FOV)
# --------------------------------------------------------------------------- #
def _h5str(ds) -> str:
    """Decode an HDF5 (MATLAB v7.3) string dataset.

    MATLAB char arrays are stored as ``uint16`` UTF-16 code units; some datasets are byte strings
    (``S``) or object refs. Returns a plain ``str`` (BMP code units are one ``uint16`` each).
    """
    a = np.asarray(ds)
    if a.dtype.kind == "S":
        return a.tobytes().decode("utf-8", "ignore").strip("\x00").strip()
    if a.dtype.kind in "iu":
        return "".join(chr(int(c)) for c in a.ravel() if int(c)).strip()
    return str(a).strip()


def read_stage_xy(mat_fp) -> np.ndarray:
    """Per-FOV absolute stage position ``(x, y)`` in microns, from ``output_data.mat`` (v7.3).

    The device structs live under ``#refs#`` with unstable keys, so we scan for the one whose
    ``deviceType == 'MAC5000_Stage_Controller'`` and read its ``pos/x``, ``pos/y``. Returns a
    ``(2,)`` float64 array. Raises ``KeyError`` if no such device is present.
    """
    import h5py

    with h5py.File(mat_fp, "r") as f:
        refs = f.get("#refs#")
        if refs is None:
            raise KeyError(f"no #refs# group in {mat_fp}")
        for g in refs.values():
            if not isinstance(g, h5py.Group) or "deviceType" not in g:
                continue                                    # most #refs# entries are plain datasets
            try:
                if _h5str(g["deviceType"]) == "MAC5000_Stage_Controller":
                    x = float(np.asarray(g["pos"]["x"]).ravel()[0])
                    y = float(np.asarray(g["pos"]["y"]).ravel()[0])
                    return np.array([x, y], float)
            except Exception:
                continue
    raise KeyError(f"no MAC5000_Stage_Controller device in {mat_fp}")


def raw_mean_reference(bin_fp, nrow, ncol, n_ref, chunk=256) -> np.ndarray:
    """Raw-brightness anatomical reference = mean of the first ``n_ref`` frames.

    Streamed through a ``uint16`` memmap in ``chunk``-frame blocks, so the multi-GB movie is never
    fully materialized in RAM. Returns ``(nrow, ncol)`` float32.

    (pyali's ``process_fov`` ``reference_image`` is correlation-weighted; cross-modal registration
    wants this plain intensity mean, which shares the JF608 blob structure of the SBS anchor.)
    """
    from ..io import open_bin_memmap

    mm = open_bin_memmap(bin_fp, nrow, ncol)                    # [T, H, W] uint16 memmap
    n = int(min(n_ref, mm.shape[0]))
    acc = np.zeros((nrow, ncol), np.float64)
    for t0 in range(0, n, chunk):
        acc += np.asarray(mm[t0:t0 + chunk]).sum(axis=0, dtype=np.float64)
    return (acc / n).astype(np.float32)


def read_voltage_reference(fov_dir, p: Params = None) -> dict:
    """LIGHT voltage load — just the image + stage position (no cell extraction).

    Used by the registration steps that need only an anatomical image to align (``mosaic`` global
    bootstrap, ``register.recover_dihedral``/``global_register``/``fine_register``). Avoids
    ``process_fov`` entirely, so no full-movie extraction. Returns::

        {"reference": (nrow, ncol) float32, "stage_xy": (2,) float64}
    """
    p = p or Params()
    return dict(
        reference=raw_mean_reference(f"{fov_dir}/frames1.bin", p.nrow, p.ncol, p.n_ref),
        stage_xy=read_stage_xy(f"{fov_dir}/output_data.mat"),
    )


def read_voltage_fov(fov_dir, plate, well, fov, p: Params = None) -> dict:
    """FULL voltage load — the light fields plus pyali's per-cell footprints/traces.

    HEAVY: ``process_fov`` runs the extraction pipeline and holds the full movie in RAM. Only the
    assignment step needs the per-cell fields, so most of the pipeline uses
    :func:`read_voltage_reference` instead. Returns::

        plate, well, fov,
        reference  (nrow, ncol) float32,   stage_xy (2,) float64,
        footprint  (nrow, ncol, N) float64,
        center_rc  (N, 2) float64  # (row, col), 0-indexed
        traces     (N, T) float64,
        snr        {noise_sigma, snr_median, spectral_hf_snr: (N,) float64; n_spikes: (N,) int}
    """
    p = p or Params()
    from ..pipeline import process_fov
    from ..metrics import per_cell_snr

    res = process_fov(fov_dir, save=False, p=p)
    snr = per_cell_snr(res["cell_traces"], p.fps)
    light = read_voltage_reference(fov_dir, p)
    return dict(
        plate=plate, well=well, fov=fov,
        reference=light["reference"], stage_xy=light["stage_xy"],
        footprint=res["footprint"],
        center_rc=np.asarray(res["footprint_center"], float) - 1.0,   # 1-indexed (row,col) -> 0-indexed
        traces=res["cell_traces"], snr=snr,
    )


# --------------------------------------------------------------------------- #
# SBS side
# --------------------------------------------------------------------------- #
def read_sbs_ref_tile(tiff_fp) -> np.ndarray:
    """JF608 anchor tile = channel ``[3]`` (the Cy5/A-base channel, where JF608 also lives) of the
    cycle-1 (``C-1``) image. Returns (1480,1480) uint16. Channel order is ``[DAPI, T, G, A, C]``."""
    import tifffile

    return tifffile.imread(tiff_fp)[3]


def read_sbs_stack(tiff_fp) -> np.ndarray:
    """Full SBS cycle image: ``(5, 1480, 1480)`` uint16.

    Channel order (per user, D11): ``[0]=DAPI, [1]=T (A594), [2]=G (Cy3), [3]=A (Cy5) + JF608,
    [4]=C (Cy7)``. The registration anchor JF608 lives ONLY in channel ``[3]``, co-located there with
    the A-base Cy5 amplicons and with Cy7 (channel ``[4]``) crosstalk — see :mod:`demix`.
    """
    import tifffile

    return tifffile.imread(tiff_fp)


def read_cells_mask(tiff_fp) -> np.ndarray:
    """Reconciled cell mask; labels equal ``sbs_info.cell`` for that tile. On-disk dtype varies
    (empty tiles uint16, populated int64), so cast. Returns (1480,1480) int64."""
    import tifffile

    return tifffile.imread(tiff_fp).astype(np.int64)


def read_sbs_metadata(parquet_fp) -> pd.DataFrame:
    """Per-tile stage metadata (one row per tile; the 17 SBS cycles collapse via drop_duplicates).

    Columns: ``tile`` (int), ``x_pos``, ``y_pos`` (stage microns), ``pixel_size_x`` (microns/px).
    """
    m = pd.read_parquet(parquet_fp, columns=["tile", "x_pos", "y_pos", "pixel_size_x"])
    return m.drop_duplicates("tile").reset_index(drop=True)


def read_sbs_cells(sbs_info_fp, cells_fp) -> pd.DataFrame:
    """``sbs_info ⟕ cells`` on ``(plate, well, tile, cell)`` — nucleus geometry + genotype.

    Left join keeps every segmented nucleus; genotype columns (``cell_barcode_*``,
    ``gene_symbol_*``, ``gene_id_*``) are nullable for nuclei that were not genotyped.
    """
    info = pd.read_parquet(sbs_info_fp)
    cells = pd.read_parquet(cells_fp)
    geno = [c for c in cells.columns
            if c.startswith(("cell_barcode_", "gene_symbol_", "gene_id_"))]
    keys = ["plate", "well", "tile", "cell"]
    return info.merge(cells[keys + geno], on=keys, how="left")
