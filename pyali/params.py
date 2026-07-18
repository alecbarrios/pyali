"""Pipeline parameters.

The dataclass field defaults are the legacy **6GP002** profile (312x1200, 16-bit). Use the
:meth:`Params.profile_443screen2` factory for the 800x800 8-bit batch — it bundles the frame
size, dtype, blue-stimulus baseline frame ranges, saturation, and detection threshold for that
acquisition so callers don't have to set them piecemeal.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Params:
    # acquisition (frame size; override per movie)
    nrow: int = 312
    ncol: int = 1200
    n_ref: int = 600                         # frames used to build the reference image
    fps: float = 800.0                       # frame rate (Hz)
    truncate_last: int = 10                  # trailing frames to drop

    # movie dtype (memory): on-disk sample dtype and in-RAM working dtype
    read_dtype: str = "uint16"               # "uint16" (6GP002 16-bit) | "uint8" (443screen2 8-bit)
    compute_dtype: str = "float64"           # "float64" (faithful) | "float32" (~halves peak RAM)

    # sharpening
    disk_radius: int = 15
    gauss_sigma: float = 1.75
    lap_alpha: float = 0.2
    sharpen_k: float = 2.25

    # background + baseline frame ranges (1-indexed inclusive; set to your acquisition protocol)
    bkg_ranges: list = field(default_factory=lambda: [(1, 750), (4100, 4200), (6100, 6350)])
    std_ranges: list = field(default_factory=lambda: [(1, 750), (4100, 4200), (6100, 6350)])
    saturation_clip: Optional[float] = 25000.0   # None => skip saturation clipping (8-bit has none)

    # filters
    filter_window: int = 8
    dog_sigma1: float = 1.0
    dog_sigma2: float = 3.0
    dog_width: int = 19

    # AP detection
    threshold_factor: float = 3.25
    min_peak_interval: int = 15

    # segmentation
    seg_threshold: float = 0.90
    seg_gauss: float = 0.1
    seg_region_size: int = 10

    # COM + clustering + footprints
    patch_size: tuple = (27, 27)
    svd_rank: int = 15
    com_radius: float = 8.5
    com_n_pixels: int = 50
    dbscan_eps: float = 1.5
    dbscan_min_pts: int = 3

    # trace extraction: whitened GLS (opt-in; default OFF = the baseline unweighted pinv)
    # See snr_analysis/ for the A/B benchmarking recipe. When False, cell_traces are byte-for-byte
    # the baseline unweighted pseudoinverse, so results are unchanged from the previous iteration.
    whiten_traces: bool = False              # True => noise-weighted GLS instead of unweighted pinv
    whiten_ridge: float = 1e-3               # ridge lambda as a fraction of mean(diag(F^T W F))
    whiten_sigma_floor_pct: float = 5.0      # floor per-pixel noise at this percentile (guards 1/sigma^2)
    whiten_isolated_only: bool = True        # only whiten cells with no overlapping-footprint neighbor
                                             #   (overlapping cells keep the faithful pinv row) -- guards
                                             #   against the neighbor-crosstalk backfire mode

    def std_frames_index(self):
        """Baseline frames as a 0-indexed concatenated array (from ``std_ranges``)."""
        import numpy as np
        return np.concatenate([np.arange(a - 1, b) for a, b in self.std_ranges])

    # --------------------------------------------------------------------- #
    # Acquisition profiles
    # --------------------------------------------------------------------- #
    @classmethod
    def profile_6GP002(cls, **overrides):
        """Legacy **6GP002** batch: 312x1200, 16-bit, ~6389 frames. These are the dataclass
        field defaults; provided as a named factory for symmetry/documentation."""
        return cls(**overrides)

    @classmethod
    def profile_443screen2(cls, **overrides):
        """**443screen2** batch: 800x800, 8-bit, fps=800, 8399 frames (~10.5 s), blue-stimulus.

        Baseline (stimulus-OFF) frame ranges are 1-indexed inclusive, derived from the OFF
        windows of the protocol at fps=800 with frame 1 <-> t=0:
            0.25-0.75 s -> 201-601      1.65-1.85 s -> 1321-1481    3.6-3.8 s -> 2881-3041
            7.3-7.49 s  -> 5841-5993    9.8-10.3 s  -> 7841-8241
        (If your acquisition indexes frame 1 at t=1/fps, shift each endpoint by -1; the >=150-frame
        margins to every stimulus edge make the ±1 immaterial.)

        Notes:
          * ``compute_dtype='float32'`` halves peak RAM (~43 GB vs ~86 GB) so an 800x800x8399 movie
            fits; the 8-bit source is represented losslessly by float32.
          * ``saturation_clip=None`` -- the 8-bit MATLAB miniALI (V1.3) does not clip saturation.
          * ``threshold_factor=3.0`` is user-chosen for consistency with the QC spike threshold
            (k=3 sigma). NB: the MATLAB 8-bit script uses 4.5 and the 6GP002 pyali default is 3.25,
            so this is more permissive than either -- confirm during validation.
          * sharpening / segmentation / clustering params are kept at the 6GP002-tuned defaults and
            should be re-checked on real 443screen2 FOVs (see the pipeline validation step).
        """
        ranges = [(201, 601), (1321, 1481), (2881, 3041), (5841, 5993), (7841, 8241)]
        base = dict(
            nrow=800, ncol=800, n_ref=600, fps=800.0, truncate_last=10,
            read_dtype="uint8", compute_dtype="float32",
            bkg_ranges=list(ranges), std_ranges=list(ranges),
            saturation_clip=None,
            threshold_factor=3.0,
        )
        base.update(overrides)
        return cls(**base)
