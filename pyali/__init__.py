"""pyali — voltage-imaging waveform-extraction pipeline.

Loads a raw movie, builds a reference image, estimates the adaptive background and the rigid
motion shift (applied lazily where pixels are read -- the movie stays in its acquisition dtype),
segments cells, detects action potentials, extracts per-cell spatial footprints (sparse), and
recovers per-cell temporal traces by a sparse least-squares solve.

Convention: arrays are row-major C-order; movies are ``[T, H, W]``.
"""

__version__ = "0.1.0"

from . import (utils, io, preprocess, movie, segmentation, extract, assign, traces, params,  # noqa: F401
               results, telemetry, pipeline)
