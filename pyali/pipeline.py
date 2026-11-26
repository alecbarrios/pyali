"""End-to-end orchestrator.

``process_fov`` runs the full pipeline on one field of view and writes the per-FOV results
(``cells.parquet``/``cells.csv``, ``spikes.parquet``, ``signals.h5`` -- see :mod:`pyali.results`)
plus ``fov_metadata.json``. ``run_folder`` loops over sub-folders with per-FOV error logging.

Every stage runs under a :class:`pyali.telemetry.StageMeter`, so the metadata records each
stage's wall seconds, CPU seconds and resident-memory start / end / peak.

Memory: the movie stays as acquired (uint8 / uint16, e.g. 5.4 GB for 800x800x8399 8-bit) and
the background / motion corrections are applied where pixels are read
(:class:`pyali.movie.CorrectedMovie`); footprints are packed sparse as they are produced and
the trace solve works on the footprint pixel union (:mod:`pyali.traces`).
"""
import os
import numpy as np

from .params import Params
from . import io, preprocess, segmentation, extract, traces, results
from .movie import CorrectedMovie, estimate_shift, interp_weights
from .telemetry import StageMeter


def process_fov(fov_dir, out_dir=None, p=Params(), save=True, verbose=True, make_figures=False,
                profile=None, day=None, identity=None):
    """Run the full extraction on one FOV; returns a dict of the key outputs.

    ``verbose`` prints per-stage progress and telemetry (the full-movie steps take minutes each,
    so the run is visibly progressing rather than silent). ``make_figures`` (needs ``out_dir``)
    writes the presentation result figures. ``identity`` (:class:`pyali.metadata.Identity`) names
    the FOV in ``cells.parquet``; by default it is parsed from the directory name.
    """
    from .metadata import identity_from_dir_name

    def log(msg):
        if verbose:
            print(f"[pyali] {msg}", flush=True)

    H, W = p.nrow, p.ncol
    if save and out_dir:
        os.makedirs(out_dir, exist_ok=True)          # this function writes here; don't assume it exists
    identity = identity or identity_from_dir_name(fov_dir)

    with StageMeter(log=lambda m: log("  " + m)) as meter:
        # ---- load (io): the acquisition as recorded; the last truncate_last frames are ignored ----
        log("loading movie ...")
        with meter.stage("load"):
            raw = io.read_bin_mov(os.path.join(fov_dir, "frames1.bin"), H, W,
                                  read_dtype=getattr(p, "read_dtype", "uint16"))
            T = raw.shape[0] - p.truncate_last

        # ---- reference images (preprocess), from the uncorrected frames as float32 ----
        log("reference + correlation image ...")
        with meter.stage("reference"):
            reference_image, corr_image = preprocess.reference_and_correlation_image(
                np.asarray(raw[:min(p.n_ref, T)], np.float32))
        # Raw-unit mean reference and its illumination background: reference_corrected drives
        # segmentation and per-cell brightness.
        log("large-scale background ...")
        with meter.stage("background"):
            reference_raw = preprocess.raw_mean_reference(
                np.asarray(raw[:min(p.reference_frames, T)], np.float32), p.reference_frames)
            focus = preprocess.focus_score(reference_raw)                # QC: defocused fields score ~1e-4 or less
            log(f"focus score {focus:.2e} (sharp fields of this preparation: ~1e-3 to 3.5e-3)")
            background_map = preprocess.large_scale_background(reference_raw, p.background_radius,
                                                               p.background_sigma)
            reference_corrected = (reference_raw - background_map).astype(np.float32)

        # ---- adaptive background anchors + rigid motion shift; the corrections are applied
        #      lazily by CorrectedMovie wherever pixels are read ----
        log("adaptive background anchors ...")
        with meter.stage("adaptive_background"):
            samples, anchors = preprocess.background_samples(raw[:T], p.bkg_ranges, p.disk_radius)
        log("rigid motion shift ...")
        with meter.stage("motion_correct"):
            W_interp = interp_weights(anchors, T)
            shift, A, _f0 = estimate_shift(raw[:T], W_interp, samples, p.truncate_last)
            movie = CorrectedMovie(raw, samples, anchors, shift, A, n_frames=T,
                                   saturation_clip=getattr(p, "saturation_clip", None))
            log(f"shift: |x| <= {np.abs(shift[0]).max():.2f} px, |y| <= {np.abs(shift[1]).max():.2f} px")

        # ---- segmentation ----
        log("segmentation ...")
        with meter.stage("segmentation"):
            regions, binary_map, spatial_footprints, seg_info = segmentation.cell_segmentation(
                reference_raw, reference_corrected, p)
            log(f"hysteresis: background {seg_info['bg_center']:.2f}, sigma {seg_info['sigma_bg']:.2f}, "
                f"thresholds {seg_info['th_low']:.2f} / {seg_info['th_high']:.2f}; mask px intensity "
                f"{seg_info['n_px_intensity']}, watershed {seg_info['n_px_watershed']}, final {seg_info['n_px_final']}")
            regions, binary_map, dropped = segmentation.drop_oversized_regions(
                regions, binary_map, getattr(p, "max_region_bbox_frac", None))
            if dropped:                                # rebuild: extract_footprints indexes these by
                spatial_footprints = [r["PixelList"] for r in regions]   # region, so they must stay aligned
                frame_px = H * W
                worst = max((float(r["BoundingBox"][2]) * float(r["BoundingBox"][3]) for r in dropped))
                log(f"dropped {len(dropped)} oversized region(s) (bbox > "
                    f"{100 * p.max_region_bbox_frac:g}% of frame; worst {100 * worst / frame_px:.1f}%)")
            log(f"segmented {len(regions)} regions")

        # ---- per-region extraction (reads corrected patches on demand) ----
        log(f"extracting footprints from {len(regions)} regions "
            f"({extract.resolve_workers(getattr(p, 'region_workers', 1))} worker(s)) ...")
        with meter.stage("extract_footprints"):
            (APs, COMs, footprints, footprint_center, region_of_footprint, region_of_ap,
             region_of_spike, spike_cell_row, source_of_footprint) = extract.extract_footprints(
                movie, regions, spatial_footprints, p, H, W, verbose=verbose,
                workers=getattr(p, "region_workers", 1))

        # ---- trace extraction (sparse normal equations on the footprint pixel union) ----
        noise_map = None
        if getattr(p, "whiten_traces", False):
            log("per-pixel noise on the footprint pixel union (whitened GLS) ...")
            noise_map = traces.noise_map_on_union(
                movie, traces.FootprintOperator(footprints).union, p.std_frames_index(),
                p.filter_window, p.whiten_sigma_floor_pct)
        method = "whitened GLS" if getattr(p, "whiten_traces", False) else "pinv"
        log(f"trace extraction ({method}) for {footprints.n()} footprints ...")
        solve = {}
        with meter.stage("trace_extraction"):
            cell_traces = traces.extract_cell_traces(movie, footprints, p, noise_map=noise_map,
                                                     verbose=verbose, diag=solve)
        log(f"solve: {solve.get('method')} on {solve.get('n_union')} union pixels, "
            f"cond {solve.get('cond', float('nan')):.2e}, {solve.get('n_dropped')} dropped, "
            f"{solve.get('n_unresolved')} unresolved direction(s)")
        solve = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                 for k, v in solve.items()}

        # ---- cells table: identity, source, brightness ----
        N = footprints.n()
        n_spikes = np.bincount(spike_cell_row[spike_cell_row >= 0], minlength=N)
        sparse_fp = footprints.astype(np.float32)                  # as stored (brightness, signals.h5)
        brightness = results.weighted_brightness(sparse_fp, reference_corrected)
        n_cells_by_source = {s: int(source_of_footprint.count(s)) for s in results.SOURCES}
        log("done.")

        if save and out_dir:
            with meter.stage("write"):
                results.write_results(
                    out_dir,
                    cells_df=results.make_cells(footprint_center, region_of_footprint, n_spikes,
                                                source_of_footprint, brightness, identity),
                    traces=cell_traces, footprints=sparse_fp,
                    spikes_df=results.pack_spikes(APs, COMs, spike_cell_row, region_of_spike,
                                                  region_of_ap),
                    images=dict(reference=reference_image, binary_map=binary_map,
                                brightness_reference=reference_corrected),
                    attrs=dict(fps=p.fps, profile=profile, fov_id=identity.fov_id, focus_score=focus,
                               n_ap_detected=len(APs), n_ap_localized=len(COMs)))

    telemetry = meter.report()
    timing = telemetry["timing_s"]
    log(f"total {timing['total']:.1f} s, peak rss {telemetry['peak_rss_gb']:.2f} GB, "
        f"peak memory incl. workers {telemetry['peak_mem_gb']:.2f} GB")
    if save and out_dir:
        # Metadata lands in out_dir alongside the results, at identical depth, so a
        # downstream aggregation over the corpus is a single glob.
        from .metadata import fov_metadata, write_metadata
        write_metadata(out_dir, fov_metadata(
            fov_dir, day=day, params=p, profile=profile,
            identity=identity, telemetry=telemetry,
            extras=dict(focus_score=focus,
                        n_regions_kept=len(regions), n_regions_dropped=len(dropped),
                        segmentation=seg_info, trace_solve=solve,
                        motion_shift_max_px=[float(np.abs(shift[0]).max()), float(np.abs(shift[1]).max())],
                        n_cells=int(cell_traces.shape[0]), n_cells_by_source=n_cells_by_source,
                        n_ap_detected=int(len(APs)), n_ap_localized=int(len(COMs)),
                        n_frames_analyzed=int(T))))

    if make_figures and out_dir:
        log("saving result figures ...")
        from .figures import save_result_figures
        save_result_figures(out_dir, reference_image, regions, binary_map, COMs,
                            footprint_center, cell_traces, fps=p.fps, footprint=footprints.dense())

    return dict(reference_image=reference_image, corr_image=corr_image,
                reference_raw=reference_raw, reference_corrected=reference_corrected,
                background_map=background_map, seg_info=seg_info, focus_score=focus,
                regions=regions, binary_map=binary_map, footprints=footprints,
                footprint_center=footprint_center, cell_traces=cell_traces, APs=APs, COMs=COMs,
                region_of_footprint=region_of_footprint, region_of_ap=region_of_ap,
                region_of_spike=region_of_spike, spike_cell_row=spike_cell_row,
                source_of_footprint=source_of_footprint, brightness=brightness,
                shift=shift, identity=identity, timing=timing, telemetry=telemetry)


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
