"""JF608 <- (Cy5 + Cy7-crosstalk) demixing for the SBS registration anchor (D11).

Each SBS cycle image is a 5-channel stack ``[DAPI, T(A594), G(Cy3), A(Cy5), C(Cy7)]``. The
cross-modal registration target JF608 lives ONLY in channel ``[3]`` (the Cy5/A channel), where it is
buried under two bright in-nucleus nuisances:

* the A-base **Cy5** amplicon puncta (present when a cell's barcode base is A that cycle), and
* **Cy7 (C-base) crosstalk** from channel ``[4]`` bleeding into channel ``[3]``.

Both nuisances are bright, punctate, and sit inside nuclei; JF608 is the faint, diffuse, cell-shaped
signal we actually want. Registration NCC is dominated by the bright nuisances (which have no
counterpart in the voltage image), so this module builds a JF608-emphasized single-channel anchor.

Togglable methods (``method=``):
  ``raw``         channel [3] unchanged (baseline).
  ``winsorize``   clip channel [3] above a percentile (knock down the bright nuisance tail).
  ``log``         ``log1p`` dynamic-range compression.
  ``nuclei``      down-weight channel [3] inside nucleus-centroid disks (needs nucleus coords).
  ``nuclei_dapi`` down-weight by a soft DAPI (channel [0]) mask instead of centroid disks.
  ``cy5_regress`` MULTI-CYCLE barcode-informed clean-cycle JF608 (needs >1 cycle on disk — see
                  :func:`cy5_regress_anchor`; not runnable on a single downloaded cycle).

All single-cycle methods operate on one stack; :func:`make_anchor_reader` wires a chosen method into
``mosaic.build_sbs_mosaic`` via its ``read_tile`` hook so the whole pipeline can toggle demixing.
"""
from __future__ import annotations

import numpy as np

DAPI_CH = 0            # channel indices in the [DAPI, T, G, A, C] stack
CY5_CH = 3             # A-base Cy5 channel — JF608 also lives here (the anchor)
CY7_CH = 4             # C-base Cy7 channel — crosstalks into CY5_CH
DEFAULT_WINSOR_PCT = 99.0     # tuned empirically in benchmark_demix (see report)
DEFAULT_NUC_RADIUS = 22.0     # full-res px; ~ SBS nucleus radius (sqrt(area/pi) ~ 20-25)
DEFAULT_ALPHA = 1.0           # nucleus down-weight strength in [0, 1] (1 = fully suppress)
METHODS = ("raw", "winsorize", "log", "nuclei", "nuclei_dapi", "cy5_regress")


# --------------------------------------------------------------------------- #
# nucleus down-weight mask
# --------------------------------------------------------------------------- #
def _nucleus_mask(shape, nuclei_rc, radii, blur):
    """Soft [0,1] mask that is ~1 inside each nucleus disk, 0 elsewhere (Gaussian-softened edges)."""
    from skimage.draw import disk

    m = np.zeros(shape, np.float32)
    if nuclei_rc is None or len(nuclei_rc) == 0:
        return m
    radii = np.broadcast_to(np.asarray(radii, float), (len(nuclei_rc),))
    for (r, c), rad in zip(np.asarray(nuclei_rc, float), radii):
        rr, cc = disk((r, c), max(float(rad), 1.0), shape=shape)
        m[rr, cc] = 1.0
    if blur:
        from scipy.ndimage import gaussian_filter

        m = gaussian_filter(m, float(blur))
        peak = float(m.max())
        if peak > 0:
            m = np.clip(m / peak, 0.0, 1.0)
    return m


# --------------------------------------------------------------------------- #
# single-cycle anchors
# --------------------------------------------------------------------------- #
def jf608_anchor(ch4, method="raw", dapi=None, nuclei_rc=None, radii=None,
                 pct=DEFAULT_WINSOR_PCT, alpha=DEFAULT_ALPHA, blur=None):
    """Build a JF608-emphasized anchor from the Cy5 channel ``ch4`` (2-D, any resolution).

    ``method`` selects the suppression. ``nuclei_rc`` (``(N,2)`` (row,col) in ``ch4``'s pixel frame)
    and ``radii`` are required for ``nuclei``; ``dapi`` (same shape as ``ch4``) for ``nuclei_dapi``.
    ``blur`` softens the nucleus mask (default scales with radius). Returns float32.
    """
    a = np.asarray(ch4, float)
    if method == "raw":
        return a.astype(np.float32)
    if method == "winsorize":
        cap = float(np.percentile(a, pct))
        return np.minimum(a, cap).astype(np.float32)
    if method == "log":
        return np.log1p(np.maximum(a, 0.0)).astype(np.float32)
    if method == "nuclei":
        rad = DEFAULT_NUC_RADIUS if radii is None else radii
        b = blur if blur is not None else (np.median(np.atleast_1d(rad)) * 0.4)
        m = _nucleus_mask(a.shape, nuclei_rc, rad, b)
        return (a * (1.0 - alpha * m)).astype(np.float32)
    if method == "nuclei_dapi":
        if dapi is None:
            raise ValueError("method 'nuclei_dapi' requires the DAPI channel")
        d = np.asarray(dapi, float)
        lo, hi = np.percentile(d, [50, 99])
        m = np.clip((d - lo) / (hi - lo + 1e-9), 0.0, 1.0)      # soft nucleus membership
        return (a * (1.0 - alpha * m)).astype(np.float32)
    if method == "cy5_regress":
        raise ValueError("method 'cy5_regress' is multi-cycle; call cy5_regress_anchor(...) with "
                         "the per-cycle stacks + barcodes (needs >1 cycle on disk).")
    raise ValueError(f"unknown demix method {method!r}; choose from {METHODS}")


# --------------------------------------------------------------------------- #
# multi-cycle, barcode-informed Cy5 regression  (needs >1 cycle — see module docstring)
# --------------------------------------------------------------------------- #
def base_of_cycle(barcode, cycle_index):
    """Base called at 0-indexed ``cycle_index`` for a barcode string (e.g. 'CATTACTCTTCT'[c])."""
    if barcode is None or cycle_index >= len(barcode):
        return None
    return barcode[cycle_index]


def clean_cycles_for_barcode(barcode, n_cycles, cy5_null_bases=("T", "G")):
    """0-indexed cycles that are Cy5-NULL for this cell: base in {T, G} (NOT A=Cy5, NOT C=Cy7 xtalk).

    Earliest first (JF608 bleaches across cycles, so prefer early cycles).
    """
    return [c for c in range(min(n_cycles, len(barcode or "")))
            if barcode[c] in cy5_null_bases]


def cy5_regress_anchor(cycle_stacks, cells_df, sbs_info_df, tile, cell_mask=None,
                       cy5_null_bases=("T", "G"), prefer_early=True):
    """[MULTI-CYCLE] Per-cell clean-cycle JF608 anchor for one tile.

    For each cell, pick the earliest cycle whose barcode base is Cy5-null (T or G -> channel [3] is
    pure JF608, no Cy5 amplicon and no Cy7 crosstalk), and take that cycle's channel [3] over the
    cell's footprint. Cells with no clean cycle (all A/C) fall back to the per-pixel min across the
    non-C cycles. Assembles a JF608 anchor image for the tile.

    Args:
      cycle_stacks: list of ``(5,H,W)`` stacks, index c = cycle c (0-indexed, ordered by acquisition).
      cells_df:     rows for this tile with ``cell`` + ``cell_barcode_0`` (barcode per cell).
      sbs_info_df:  rows for this tile with ``cell`` + tile-local ``i,j`` (nucleus centroid).
      cell_mask:    optional ``(H,W)`` labeled cells.tiff mask (label == cell id) to scope each cell.

    Returns ``(H, W)`` float32 JF608 anchor. Requires ``len(cycle_stacks) > 1``.
    """
    if len(cycle_stacks) < 2:
        raise ValueError("cy5_regress_anchor needs >= 2 cycles on disk; only "
                         f"{len(cycle_stacks)} provided (download more C-* tiffs — see report)")
    n_cyc = len(cycle_stacks)
    H, W = cycle_stacks[0].shape[1:]
    ch4 = np.stack([s[CY5_CH].astype(np.float32) for s in cycle_stacks], 0)     # (n_cyc, H, W)
    # per-pixel fallback: min over non-C(Cy7-crosstalk) cycles ~ the cleanest JF608 estimate
    anchor = ch4.min(axis=0).astype(np.float32)
    if cell_mask is None:
        return anchor
    bc = dict(zip(cells_df["cell"].astype(int), cells_df["cell_barcode_0"].astype("string")))
    for cell_id in np.unique(cell_mask):
        if cell_id == 0:
            continue
        barcode = bc.get(int(cell_id))
        clean = clean_cycles_for_barcode(barcode, n_cyc, cy5_null_bases) if barcode else []
        if not clean:
            continue
        c = clean[0] if prefer_early else clean[-1]            # earliest clean cycle (least bleached)
        region = cell_mask == cell_id
        anchor[region] = ch4[c][region]
    return anchor


# --------------------------------------------------------------------------- #
# pipeline hook — togglable tile reader for mosaic.build_sbs_mosaic(read_tile=...)
# --------------------------------------------------------------------------- #
def make_anchor_reader(method, tiff_of, sbs_info=None, radius=DEFAULT_NUC_RADIUS,
                       pct=DEFAULT_WINSOR_PCT, alpha=DEFAULT_ALPHA):
    """Return ``reader(tile) -> (1480,1480) float32 JF608 anchor`` for the chosen single-cycle method.

    Pass to ``mosaic.build_sbs_mosaic(..., read_tile=reader, dtype=np.float32)`` to build a demixed
    mosaic. ``nuclei`` needs ``sbs_info`` (for tile-local nucleus centroids + per-cell radii from
    ``area``). ``cy5_regress`` is multi-cycle and is NOT provided here (see :func:`cy5_regress_anchor`).
    """
    from . import io

    if method == "cy5_regress":
        raise ValueError("cy5_regress is multi-cycle; not available through the single-cycle reader")

    def reader(t):
        stack = io.read_sbs_stack(tiff_of(int(t)))
        nuclei_rc = radii = None
        if method == "nuclei":
            if sbs_info is None:
                raise ValueError("method 'nuclei' needs sbs_info for nucleus centroids")
            sub = sbs_info[sbs_info.tile == int(t)]
            nuclei_rc = sub[["i", "j"]].to_numpy(float)
            radii = (np.sqrt(sub["area"].to_numpy(float) / np.pi)
                     if "area" in sub.columns else np.full(len(sub), radius))
        return jf608_anchor(stack[CY5_CH], method, dapi=stack[DAPI_CH],
                            nuclei_rc=nuclei_rc, radii=radii, pct=pct, alpha=alpha)

    return reader
