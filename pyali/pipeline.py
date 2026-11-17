"""End-to-end orchestrator.

``process_fov`` runs the full pipeline on one field of view and writes the per-FOV results
(``cells.parquet``/``cells.csv``, ``spikes.parquet``, ``signals.h5`` -- see :mod:`pyali.results`)
plus ``fov_metadata.json``. ``run_folder`` loops over sub-folders with per-FOV error logging.

Memory note: the working movie is held as float64 (~18 GB for a 6389-frame 312x1200 movie),
plus the filtered movie during extraction — run on a machine with adequate RAM.
"""
import os
import numpy as np

from .params import Params
from . import io, preprocess, segmentation, extract, results
from .utils import dog_kernel


def process_fov(fov_dir, out_dir=None, p=Params(), save=True, verbose=True, make_figures=False,
                keep_row=None, profile=None, day=None, identity=None):
    """Run the full extraction on one FOV; returns a dict of the key outputs.

    ``verbose`` prints per-stage progress (the full-movie steps take minutes each, so the run
    is visibly progressing rather than silent). ``make_figures`` (needs ``out_dir``) writes the
    presentation result figures. ``identity`` (:class:`pyali.metadata.Identity`) names the FOV
    in ``cells.parquet``; by default it is parsed from the directory name.
    """
    from time import perf_counter
    from .metadata import identity_from_dir_name

    def log(msg):
        if verbose:
            print(f"[pyali] {msg}", flush=True)

    timing = {}
    t_start = perf_counter()

    def tic():
        return perf_counter()

    def toc(stage, t0):
        timing[stage] = perf_counter() - t0

    H, W = p.nrow, p.ncol
    if save and out_dir:
        os.makedirs(out_dir, exist_ok=True)          # this function writes here; don't assume it exists
    identity = identity or identity_from_dir_name(fov_dir)

    # ---- load + truncate (io) ----
    log("loading movie ...")
    t0 = tic()
    movie = io.read_bin_mov(os.path.join(fov_dir, "frames1.bin"), H, W,
                            read_dtype=getattr(p, "read_dtype", "uint16"),
                            out_dtype=getattr(p, "compute_dtype", "float64"))
    movie = movie[:movie.shape[0] - p.truncate_last]                 # drop last 10 frames
    T = movie.shape[0]
    toc("load", t0)

    # ---- reference + correlation image, sharpening chain (preprocess) ----
    log("reference + correlation image ...")
    t0 = tic()
    reference_image, corr_image = preprocess.reference_and_correlation_image(movie[:p.n_ref])
    toc("reference", t0)
    # brightness reference in raw units: must precede the in-place background/motion steps
    log("brightness reference ...")
    t0 = tic()
    brightness_ref = preprocess.brightness_reference(movie, p.brightness_frames, p.brightness_radius)
    toc("brightness_reference", t0)
    log("sharpening ...")
    t0 = tic()
    *_chain, sharpened = preprocess.sharpen(                 # the intermediate images were only
        reference_image, p.disk_radius, p.gauss_sigma, p.lap_alpha, p.sharpen_k)   # ever debug output
    toc("sharpen", t0)

    # ---- adaptive background, motion correction, clip (preprocess) ----
    log("adaptive background subtraction ...")
    t0 = tic()
    movie, _samples, _anchors = preprocess.adaptive_background(movie, p.bkg_ranges, p.disk_radius)
    toc("adaptive_background", t0)
    log("rigid motion correction ...")
    t0 = tic()
    movie, _shift, _A, _f0 = preprocess.motion_correct(movie, p.truncate_last)
    if getattr(p, "saturation_clip", None) is not None:                  # 8-bit acquisitions: None => skip
        preprocess.clip_saturation(movie, p.saturation_clip)
    toc("motion_correct", t0)

    # ---- segmentation ----
    log("segmentation ...")
    t0 = tic()
    regions, binary_map, spatial_footprints = segmentation.cell_segmentation(
        sharpened, p.seg_threshold, p.seg_gauss, p.seg_region_size,
        getattr(p, "seg_threshold_mult", None), getattr(p, "seg_neighborhood", None))
    regions, binary_map, dropped = segmentation.drop_oversized_regions(
        regions, binary_map, getattr(p, "max_region_bbox_frac", None))
    if dropped:                                    # rebuild: extract_footprints indexes these by
        spatial_footprints = [r["PixelList"] for r in regions]   # region, so they must stay aligned
        frame_px = H * W
        worst = max((float(r["BoundingBox"][2]) * float(r["BoundingBox"][3]) for r in dropped))
        log(f"dropped {len(dropped)} oversized region(s) (bbox > "
            f"{100 * p.max_region_bbox_frac:g}% of frame; worst {100 * worst / frame_px:.1f}%)")
    log(f"segmented {len(regions)} regions")
    toc("segmentation", t0)

    # ---- filters + per-region extraction ----
    # extract_footprints only ever reads ~27x27 patches of the filtered movie, so the global
    # array is a full extra copy of the movie for mostly wasted work. Build it only when
    # something actually needs all of it: the whitened-GLS noise map.
    need_global_filter = getattr(p, "whiten_traces", False) or not getattr(
        p, "patch_local_filter", True)
    if need_global_filter:
        log("temporal median filter (global) ...")
        filtered_movie = extract.temporal_filter(movie, p.filter_window)
    else:
        filtered_movie = None
    dog = dog_kernel(p.dog_sigma1, p.dog_sigma2, p.dog_width)
    log(f"extracting footprints from {len(regions)} regions ...")
    t0 = tic()
    (APs, COMs, footprint, footprint_center, region_of_footprint, region_of_ap,
     region_of_spike, spike_cell_row, source_of_footprint) = extract.extract_footprints(
        movie, filtered_movie, regions, spatial_footprints, dog, p.std_frames_index(), p, H, W,
        verbose=verbose)
    toc("extract_footprints", t0)
    # Build the per-pixel noise map (whitened GLS only) BEFORE freeing filtered_movie.
    noise_map = None
    if getattr(p, "whiten_traces", False):
        log("building per-pixel noise map for whitened GLS ...")
        noise_map = extract.per_pixel_noise_map(filtered_movie, p.std_frames_index(),
                                                p.whiten_sigma_floor_pct)
    del filtered_movie

    # ---- trace extraction ----
    method = "whitened GLS" if getattr(p, "whiten_traces", False) else "pinv"
    log(f"trace extraction ({method}) for {footprint.shape[2]} footprints ...")
    t0 = tic()
    cell_traces = extract.extract_cell_traces(movie, footprint, p, noise_map=noise_map,
                                              verbose=verbose)
    toc("trace_extraction", t0)

    # ---- cells table: identity, source, brightness ----
    N = footprint.shape[2]
    n_spikes = np.bincount(spike_cell_row[spike_cell_row >= 0], minlength=N)
    sparse_fp = results.SparseFootprints.from_dense(footprint)
    brightness = results.weighted_brightness(sparse_fp, brightness_ref)
    n_cells_by_source = {s: int(source_of_footprint.count(s)) for s in results.SOURCES}
    log("done.")

    if save and out_dir:
        t0 = tic()
        results.write_results(
            out_dir,
            cells_df=results.make_cells(footprint_center, region_of_footprint, n_spikes,
                                        source_of_footprint, brightness, identity),
            traces=cell_traces, footprints=sparse_fp,
            spikes_df=results.pack_spikes(APs, COMs, spike_cell_row, region_of_spike,
                                          region_of_ap),
            images=dict(reference=reference_image, binary_map=binary_map,
                        brightness_reference=brightness_ref),
            attrs=dict(fps=p.fps, profile=profile, fov_id=identity.fov_id,
                       n_ap_detected=len(APs), n_ap_localized=len(COMs)))
        toc("write", t0)
        timing["total"] = perf_counter() - t_start
        # Metadata lands in out_dir alongside the results, at identical depth, so a
        # downstream aggregation over the corpus is a single glob.
        from .metadata import fov_metadata, write_metadata
        write_metadata(out_dir, fov_metadata(
            fov_dir, day=day, keep_row=keep_row, params=p, profile=profile,
            identity=identity, timing=timing,
            extras=dict(n_regions_kept=len(regions), n_regions_dropped=len(dropped),
                        n_cells=int(cell_traces.shape[0]), n_cells_by_source=n_cells_by_source,
                        n_ap_detected=int(len(APs)), n_ap_localized=int(len(COMs)),
                        n_frames_analyzed=int(T))))
    timing.setdefault("total", perf_counter() - t_start)

    if make_figures and out_dir:
        log("saving result figures ...")
        from .figures import save_result_figures
        save_result_figures(out_dir, reference_image, regions, binary_map, COMs,
                            footprint_center, cell_traces, fps=p.fps, footprint=footprint)

    return dict(reference_image=reference_image, corr_image=corr_image, sharpened=sharpened,
                regions=regions, binary_map=binary_map, footprint=footprint,
                footprint_center=footprint_center, cell_traces=cell_traces, APs=APs, COMs=COMs,
                region_of_footprint=region_of_footprint, region_of_ap=region_of_ap,
                region_of_spike=region_of_spike, spike_cell_row=spike_cell_row,
                source_of_footprint=source_of_footprint, brightness=brightness,
                brightness_reference=brightness_ref, identity=identity, timing=timing)


def run_folder(root_path, p=Params()):
    """Process every sub-folder of ``root_path``, writing results into ``root_path/Analysis``."""
    analysis = os.path.join(root_path, "Analysis")
    os.makedirs(analysis, exist_ok=True)
    subdirs = sorted(d for d in os.listdir(root_path)
                     if os.path.isdir(os.path.join(root_path, d)) and d != "Analysis")
    ok, errors = [], []
    for name in subdirs:
        out = os.path.join(analysis, name)
        os.makedirs(out, exist_ok=True)
        try:
            process_fov(os.path.join(root_path, name), out, p)
            ok.append(name)
        except Exception as e:                                       # keep going, log per-FOV errors
            errors.append((name, repr(e)))
    return ok, errors
