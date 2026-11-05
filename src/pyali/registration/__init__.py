"""Cross-modal registration core, living inside pyali (shares the one dedicated env).

Registers voltage-imaging FOVs to the brieflow SBS well via their shared JF608 fluorophore, then
assigns each voltage cell the genotype brieflow called for the nearest SBS cell.

Layers
------
* ``coordinates`` — the affine / axis-convention firewall (verified).
* ``phasecorr``   — Fourier–Mellin + ORB/RANSAC phase-correlation engine (verified).
* ``io``          — load SBS tiles/masks/parquets and voltage references/positions.
* ``mosaic``      — build the per-well SBS JF608 mosaic + shared stage->px transform.
* ``place_cells`` — place SBS genotyped cells into the mosaic frame.
* ``register``    — dihedral recovery + global bootstrap + per-FOV fine registration.
* (landing next)  ``assign``, ``outputs``.

``qc`` is a SEPARATE module and is intentionally NOT imported here: the core must be
importable/runnable without it, and QC runs independently over persisted artifacts.
"""
from . import coordinates, phasecorr, io, mosaic, place_cells, register  # noqa: F401

__all__ = ["coordinates", "phasecorr", "io", "mosaic", "place_cells", "register"]
