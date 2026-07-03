"""End-to-end orchestrator.

``process_fov`` runs the full pipeline on one field of view and writes ``ALI_Int_Result.mat``
and ``ALI_Result.mat``. ``run_folder`` loops over sub-folders with per-FOV error logging.

Memory note: the working movie is held as float64 (~18 GB for a 6389-frame 312x1200 movie),
plus the filtered movie during extraction — run on a machine with adequate RAM.
"""
import os
import numpy as np

from .params import Params
from . import io, preprocess, segmentation, extract
from .utils import dog_kernel


def process_fov(fov_dir, out_dir=None, p=Params(), save=True, verbose=True, make_figures=False):
    """Run the full extraction on one FOV; returns a dict of the key outputs.

    ``verbose`` prints per-stage progress (the full-movie steps take minutes each, so the run
    is visibly progressing rather than silent). ``make_figures`` (needs ``out_dir``) writes the
    presentation result figures.
    """
    def log(msg):
        if verbose:
            print(f"[pyali] {msg}", flush=True)

    H, W = p.nrow, p.ncol

    # ---- load + truncate (io) ----
    log("loading movie ...")
    movie = io.read_bin_mov(os.path.join(fov_dir, "frames1.bin"), H, W)
    movie = movie[:movie.shape[0] - p.truncate_last]                 # drop last 10 frames
    T = movie.shape[0]

    # ---- reference + correlation image, sharpening chain (preprocess) ----
    log("reference + correlation image ...")
    reference_image, corr_image = preprocess.reference_and_correlation_image(movie[:p.n_ref])
    log("sharpening ...")
    bkg, I2, blurred, h_lap, LoG, LoG_n, sharpened = preprocess.sharpen(
        reference_image, p.disk_radius, p.gauss_sigma, p.lap_alpha, p.sharpen_k)

    # ---- adaptive background, motion correction, clip (preprocess) ----
    log("adaptive background subtraction ...")
    movie, _samples, _anchors = preprocess.adaptive_background(movie, p.bkg_ranges, p.disk_radius)
    log("rigid motion correction ...")
    movie, _shift, _A, _f0 = preprocess.motion_correct(movie, p.truncate_last)
    preprocess.clip_saturation(movie, p.saturation_clip)

    # ---- segmentation ----
    log("segmentation ...")
    regions, binary_map, spatial_footprints = segmentation.cell_segmentation(
        sharpened, p.seg_threshold, p.seg_gauss, p.seg_region_size)
    log(f"segmented {len(regions)} regions")

    if save and out_dir:
        io.save_mat_v73(os.path.join(out_dir, "ALI_Int_Result.mat"),
                        reference_image=reference_image, bkg=bkg, I2=I2, blurred=blurred,
                        h_lap=h_lap, LoG_image=LoG, LoG_normalized=LoG_n, sharpened=sharpened)

    # ---- filters + per-region extraction ----
    log("temporal median filter ...")
    filtered_movie = extract.temporal_filter(movie, p.filter_window)
    dog = dog_kernel(p.dog_sigma1, p.dog_sigma2, p.dog_width)
    log(f"extracting footprints from {len(regions)} regions ...")
    APs, COMs, footprint, footprint_center = extract.extract_footprints(
        movie, filtered_movie, regions, spatial_footprints, dog, p.std_frames_index(), p, H, W,
        verbose=verbose)
    del filtered_movie

    # ---- trace extraction (pinv) ----
    log(f"trace extraction (pinv) for {footprint.shape[2]} footprints ...")
    cell_traces = extract.pinv_traces(movie, footprint)
    log("done.")

    if save and out_dir:
        io.save_mat_v73(os.path.join(out_dir, "ALI_Result.mat"),
                        footprint=footprint, footprint_center=footprint_center,
                        cell_traces=cell_traces)

    if make_figures and out_dir:
        log("saving result figures ...")
        from .figures import save_result_figures
        save_result_figures(out_dir, reference_image, regions, binary_map, COMs,
                            footprint_center, cell_traces, fps=p.fps)

    return dict(reference_image=reference_image, corr_image=corr_image, sharpened=sharpened,
                regions=regions, binary_map=binary_map, footprint=footprint,
                footprint_center=footprint_center, cell_traces=cell_traces, APs=APs, COMs=COMs)


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
