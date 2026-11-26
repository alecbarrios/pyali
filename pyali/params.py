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

    # raw-unit reference and its large-scale background (segmentation input and the x in
    # weighted_brightness): mean of the first `reference_frames` RAW frames minus a mask-aware
    # exact-disk opening (radius > cell size) smoothed by a normalized Gaussian
    reference_frames: int = 300
    background_radius: int = 50              # px; also the soma watershed's large opening
    background_sigma: float = 40.0           # px; 0 = opening only

    # movie dtype (memory): on-disk sample dtype and in-RAM working dtype
    read_dtype: str = "uint16"               # "uint16" (6GP002 16-bit) | "uint8" (443screen2 8-bit)
    compute_dtype: str = "float64"           # "float64" (faithful) | "float32" (~halves peak RAM)

    # time-varying movie background (preprocess.adaptive_background): opening radius
    disk_radius: int = 15

    # background + baseline frame ranges (1-indexed inclusive; set to your acquisition protocol)
    bkg_ranges: list = field(default_factory=lambda: [(1, 750), (4100, 4200), (6100, 6350)])
    std_ranges: list = field(default_factory=lambda: [(1, 750), (4100, 4200), (6100, 6350)])
    saturation_clip: Optional[float] = 25000.0   # None => skip saturation clipping (8-bit has none)

    # filters
    filter_window: int = 8
    patch_local_filter: bool = True          # filter each region's patch on demand instead of
                                             #   materializing a whole second movie. Bit-identical
                                             #   (the filter is per-pixel along time) and avoids
                                             #   filtering the ~98% of pixels no region reads.
                                             #   Forced off when whiten_traces needs the global
                                             #   array. False = the old global-filter path.

    # AP detection (Mini ALI V2.2 mini_ali_patch_detect_spike_times): the region trace is
    # medfilt1-filtered with `detect_median_window` frames (spikes positive), `detect_edge_frames`
    # are zeroed at each end, and the threshold is `threshold_factor` x the mean absolute
    # deviation of the whole detection signal (MATLAB mad, ~0.8 sigma; 4 x mad ~ 3.2 sigma)
    detect_median_window: int = 32           # 4 x the 8-frame per-pixel window
    detect_edge_frames: int = 16             # round(8/2) x 4
    threshold_factor: float = 4.0            # x mean absolute deviation (was 3.25 x a stimulus-off std)
    min_peak_interval: int = 25              # frames; keep the largest AP within this window
                                             #   (Mini ALI V2.2 minimal interval; was 15 with
                                             #   a rule that dropped every member of a burst)

    # segmentation = hysteresis_mask(reference_corrected) & soma_watershed_mask(reference_raw)
    seg_smooth_sigma: float = 2.0            # Gaussian on the corrected reference before thresholding
    seg_high_sigma: float = 6.0              # seeds: background + 6 sigma_bg
    seg_low_sigma: float = 3.0               # growth: background + 3 sigma_bg (8-connected to a seed)
    seg_min_area: int = 15                   # px; the hysteresis mask's floor only (nothing after the AND)
    ws_tophat_radius: int = 3                # soma-scale top-hat (MATLAB N=4 octagon)
    ws_bg_sigma: float = 20.0                # Gaussian on the top-hat's large opening
    ws_percentile: float = 95.0              # the top 5% of the soma image is the watershed mask
    ws_close_radius: int = 2                 # exact-disk closing of that mask
    ws_h: float = 0.03                       # h-maxima depth for the watershed markers
    max_region_bbox_frac: Optional[float] = 0.01   # reject regions whose BOUNDING BOX exceeds this
                                                   #   fraction of the frame -- the un-illuminated
                                                   #   inter-well band on a well's first/last
                                                   #   bursts merges into one object spanning up
                                                   #   to 84% of the frame. None disables.
                                                   #   See segmentation.drop_oversized_regions.

    # COM + clustering + footprints
    patch_size: tuple = (27, 27)
    svd_rank: int = 10                       # one filtered frame per spike, svds rank (Mini ALI nsvd)
    com_radius: float = 6.0                  # px from the seed for the COM region growing
    com_n_pixels: int = 25                   #   (Mini ALI V2.2 fine_radius_px / fine_npix;
                                             #   were 8.5 / 50)
    dbscan_eps: float = 1.5
    dbscan_min_pts: int = 3

    # trace extraction: whitened GLS (opt-in; default OFF = the baseline unweighted pinv)
    # See snr_analysis/ for the A/B benchmarking recipe. When False, cell_traces are byte-for-byte
    # the baseline unweighted pseudoinverse, so results are unchanged from the previous iteration.
    whiten_traces: bool = False              # True => noise-weighted GLS instead of unweighted pinv
    whiten_ridge: float = 1e-3               # ridge lambda as a fraction of mean(diag(F^T W F))
    whiten_sigma_floor_pct: float = 5.0      # floor per-pixel noise at this percentile (guards 1/sigma^2)
    whiten_isolated_only: bool = True        # only whiten cells with no overlapping-footprint neighbor
    # sparse solve (pyali.traces): pinv cutoff (None = numpy's max(P, N) * eps), the per-chunk
    # gather size, and the G condition number above which the dense pseudoinverse on the pixel
    # union is used instead of the normal equations
    trace_rcond: Optional[float] = None
    trace_chunk_bytes: int = 512 << 20
    trace_cond_limit: float = 1e12
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
    def profile_6GP002_8bit(cls, **overrides):
        """**6GP002** 8-bit variant (the 20260401 ``*_8bit_*`` runs): 1080x1080, 6399 frames.

        Same stimulus protocol — and therefore the same background/baseline ranges — as
        :meth:`profile_6GP002`, but an 8-bit acquisition: ``saturation_clip=None`` (no ceiling to
        clip) and ``compute_dtype='float32'`` (the 8-bit source is represented losslessly).
        """
        base = dict(nrow=1080, ncol=1080, read_dtype="uint8", compute_dtype="float32",
                    saturation_clip=None)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def profile_443screen1(cls, **overrides):
        """**443screen1** batch (20260611/12; the directories name it ``443GP`` — same batch):
        1000x1000, 8-bit, fps=800, 6399 frames (~8.0 s).

        Background/baseline (stimulus-OFF) ranges are 1-indexed inclusive and fit inside the
        6399-frame movies. ``saturation_clip=None`` -- 8-bit acquisitions have no saturation
        ceiling to clip, as for :meth:`profile_443screen2`.
        """
        ranges = [(1, 750), (3750, 4000), (6000, 6250)]
        base = dict(
            nrow=1000, ncol=1000, n_ref=600, fps=800.0, truncate_last=10,
            read_dtype="uint8", compute_dtype="float32",
            bkg_ranges=list(ranges), std_ranges=list(ranges),
            saturation_clip=None,
        )
        base.update(overrides)
        return cls(**base)

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
          * AP detection follows Mini ALI V2.2 (dataclass defaults): 32-frame median on the
            region trace, 16 edge frames zeroed, threshold 4 x the mean absolute deviation of
            the detection signal (~3.2 sigma). The earlier 3.0 x stimulus-off std override and
            the 8-frame per-pixel detection trace (which attenuated 2-4 frame spikes) are gone.
          * segmentation follows Mini ALI V2.2 (hysteresis 6/3 sigma on the background-corrected
            reference, AND a soma watershed; dataclass defaults); clustering params are kept at
            the 6GP002-tuned defaults and should be re-checked on real 443screen2 FOVs.
          * ``min_peak_interval=25``, ``com_radius=6.0`` and ``com_n_pixels=25`` (dataclass
            defaults) follow Mini ALI V2.2: the largest AP within a 25-frame window is kept, and
            the COM is grown over up to 25 in-mask pixels within 6 px of the seed (Cycle A).
        """
        ranges = [(201, 601), (1321, 1481), (2881, 3041), (5841, 5993), (7841, 8241)]
        base = dict(
            nrow=800, ncol=800, n_ref=600, fps=800.0, truncate_last=10,
            read_dtype="uint8", compute_dtype="float32",
            bkg_ranges=list(ranges), std_ranges=list(ranges),
            saturation_clip=None,
        )
        base.update(overrides)
        return cls(**base)
