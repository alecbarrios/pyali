"""pyali — voltage-imaging waveform-extraction pipeline.

Loads a raw movie, builds a reference image, background-subtracts and motion-corrects,
segments cells, detects action potentials, extracts per-cell spatial footprints, and
recovers per-cell temporal traces.

Convention: arrays are row-major C-order; movies are ``[T, H, W]`` float64.
"""

__version__ = "0.1.0"

from . import utils, io, preprocess, segmentation, extract, params, pipeline  # noqa: F401
