"""Place SBS cell centroids into the mosaic frame.

Uses the SAME per-tile offsets (``tile_off`` from :func:`mosaic.build_sbs_mosaic`) that placed the
image tiles — this shared placement is the correctness invariant that keeps the SBS cells in lockstep
with the JF608 mosaic they will be matched against. Genotype columns are carried through unchanged.
"""
from __future__ import annotations

import numpy as np


def place_sbs_cells(sbs_cells, tile_off):
    """Add mosaic-frame centroids ``i_mos``, ``j_mos`` (float64) to the SBS cells table.

    ``sbs_cells`` has per-tile-local ``i`` (row), ``j`` (col); ``tile_off`` maps ``tile ->
    (row0, col0)``. Cells whose tile was not mosaicked (no tiff) are dropped [E12].
    """
    keep = sbs_cells[sbs_cells.tile.isin(list(tile_off))].copy()
    row0 = keep.tile.map(lambda t: tile_off[int(t)][0]).to_numpy(float)
    col0 = keep.tile.map(lambda t: tile_off[int(t)][1]).to_numpy(float)
    keep["i_mos"] = keep["i"].to_numpy(float) + row0
    keep["j_mos"] = keep["j"].to_numpy(float) + col0
    return keep
