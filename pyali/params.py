"""Pipeline parameters (defaults tuned for the reference acquisition)."""
from dataclasses import dataclass, field


@dataclass
class Params:
    # acquisition (frame size; override per movie)
    nrow: int = 312
    ncol: int = 1200
    n_ref: int = 600                         # frames used to build the reference image
    fps: float = 800.0                       # frame rate (Hz)
    truncate_last: int = 10                  # trailing frames to drop

    # sharpening
    disk_radius: int = 15
    gauss_sigma: float = 1.75
    lap_alpha: float = 0.2
    sharpen_k: float = 2.25

    # background + baseline frame ranges (1-indexed inclusive; set to your acquisition protocol)
    bkg_ranges: list = field(default_factory=lambda: [(1, 750), (4100, 4200), (6100, 6350)])
    std_ranges: list = field(default_factory=lambda: [(1, 750), (4100, 4200), (6100, 6350)])
    saturation_clip: float = 25000.0

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
