"""Visual / manual-inspection suite for the registration pipeline (A0-A4).

Renders EVERY pipeline intermediate to PNGs so the registration logic can be checked by eye,
SEPARATELY from running the pipeline (this module only reads; it never writes pipeline outputs).

USAGE — run with the pyali venv python; `-m` puts pyali on the path automatically::

    PY="/Users/alec/Claude/Projects/miniali python port/pyali/pyali/bin/python"

    "$PY" -m pyali.registration.viz --list            # list every stage + what it renders
    "$PY" -m pyali.registration.viz channels          # SBS 5-channel breakdown  (Cy5 vs JF608, Q4)
    "$PY" -m pyali.registration.viz orient            # mosaic ORIENTATION diagnosis (Q1)  <-- start here
    "$PY" -m pyali.registration.viz tiles stage       # one or more stages, space-separated
    "$PY" -m pyali.registration.viz all               # everything A1..A4 (slow, ~5-8 min)
    "$PY" -m pyali.registration.viz orient --out /tmp/x   # override the output directory

Every render prints the absolute path of the PNG it wrote. Default output dir:
    /Users/alec/Claude/Projects/pyreg_data/registration_viz/

Stages (fast to slow):
    channels tiles stage voltage axes orient   -> no full-mosaic build (fast, ~1 min each)
    mosaic cells coarse dihedral localize register -> build the full ~28k^2 mosaic first (heavy)

This is the INSPECTION tool. The pass/fail CODE tests live in scripts/dev_smoke_registration.py
(run that separately: `"$PY" .../scripts/dev_smoke_registration.py`).
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Rectangle                          # noqa: E402

from . import io, mosaic as mos, place_cells as pc, register as reg, coordinates as C  # noqa: E402
from ..params import Params                                       # noqa: E402

# ------------------------------------------------------------------ data paths (edit or --data)
DATA = os.environ.get("PYREG_DATA", "/Users/alec/Claude/Projects/pyreg_data")
SBS = f"{DATA}/brieflow_output"
VOLT = f"{DATA}/voltage/Data_A1"
META_FP = f"{SBS}/preprocess/metadata/sbs/P-1_W-A1__combined_metadata.parquet"
INFO_FP = f"{SBS}/sbs/parquets/P-1_W-A1__sbs_info.parquet"
CELLS_FP = f"{SBS}/sbs/parquets/P-1_W-A1__cells.parquet"
OUT = f"{DATA}/registration_viz"


def _tiff_of(t):
    return f"{SBS}/preprocess/images/sbs/P-1_W-A1_T-{t}_C-1__image.tiff"


def _norm(a, lo=1, hi=99):
    a = np.asarray(a, float)
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, name)
    fig.savefig(fp, dpi=95, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {fp}")
    return fp


# ------------------------------------------------------------------ lazy shared context
class Ctx:
    def __init__(self):
        self._meta = self._cells = self._tiles = self._small = None
        self._mosaic = self._off = self._axes = None
        self._fovs = None

    @property
    def meta(self):
        if self._meta is None:
            self._meta = io.read_sbs_metadata(META_FP)
        return self._meta

    @property
    def px(self):
        return float(self.meta.pixel_size_x.iloc[0])

    @property
    def sbs_cells(self):
        if self._cells is None:
            self._cells = io.read_sbs_cells(INFO_FP, CELLS_FP)
        return self._cells

    @property
    def tiles(self):
        if self._tiles is None:
            self._tiles = [int(t) for t in self.meta.tile.tolist() if os.path.exists(_tiff_of(int(t)))]
        return self._tiles

    def small_tiles(self, ds=16):
        """Cache of each tile's channel-[3] image downsampled by ``ds`` (for the fast low-res well)."""
        if self._small is None or self._small[0] != ds:
            cache = {}
            for i, t in enumerate(self.tiles):
                cache[t] = reg.downscale(io.read_sbs_ref_tile(_tiff_of(t)), ds)
                if (i + 1) % 80 == 0:
                    print(f"      cached {i+1}/{len(self.tiles)} low-res tiles")
            self._small = (ds, cache)
        return self._small[1]

    def mosaic(self):
        if self._mosaic is None:
            print("    building full mosaic (heavy)...")
            self._axes = mos.recover_stage_axes(self.meta, _tiff_of)
            self._mosaic, self._off, _ = mos.build_sbs_mosaic(
                self.meta.tile.tolist(), _tiff_of, self.meta, axes=self._axes)
        return self._mosaic, self._off, self._axes

    def fovs(self):
        if self._fovs is None:
            dirs = sorted(glob.glob(f"{VOLT}/*_burst*"))
            self._fovs = [dict(fov=i, **io.read_voltage_reference(d, Params())) for i, d in enumerate(dirs)]
        return self._fovs


# ==================================================================================== A1
def stage_channels(ctx):
    """[A1, Q4] All 5 channels of an interior + an edge tile, intensity histograms, and candidate
    JF608 extractions (Cy5-puncta suppression). The registration ANCHOR is channel [3]."""
    import tifffile
    from scipy.ndimage import white_tophat, median_filter

    xy = ctx.meta[["x_pos", "y_pos"]].to_numpy(float)
    cen = xy.mean(0)
    d = np.linalg.norm(xy - cen, axis=1)
    interior_t = int(ctx.meta.tile.iloc[int(np.argmin(d))])
    edge_t = int(ctx.meta.tile.iloc[int(np.argmax(d))])

    for label, t in [("interior", interior_t), ("edge", edge_t)]:
        if not os.path.exists(_tiff_of(t)):
            continue
        stk = tifffile.imread(_tiff_of(t))                        # (5,1480,1480)
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        for k in range(5):
            a = ax.ravel()[k]
            a.imshow(_norm(stk[k]), cmap="magma")
            a.set_title(f"channel [{k}]  max={int(stk[k].max())} p99={np.percentile(stk[k],99):.0f}"
                        + ("  <- registration anchor (JF608?)" if k == 3 else ""), fontsize=9)
            a.axis("off")
        # histogram (log) of channel 3
        a = ax.ravel()[5]
        a.hist(stk[3].ravel(), bins=200, log=True, color="crimson")
        a.set_title(f"tile T-{t} ({label}) channel[3] intensity (log count)\n"
                    f"bright puncta=Cy5 (T-base), faint outline=JF608", fontsize=9)
        _save(fig, f"A1_channels_T{t}_{label}.png")

        # Cy5-suppression / JF608-extraction candidates on channel [3]
        c3 = stk[3].astype(float)
        cap = np.clip(c3, 0, np.percentile(c3, 95))               # (a) winsorize bright puncta
        logc = np.log1p(c3)                                       # (b) log dynamic-range compression
        th = c3 - white_tophat(c3, size=7)                        # (c) remove small bright puncta
        med = median_filter(c3, size=5)                           # (d) median (kills 1-3px puncta)
        fig2, ax2 = plt.subplots(2, 3, figsize=(15, 10))
        for a, im, ttl in zip(ax2.ravel(),
                              [c3, cap, logc, th, med, stk[0]],
                              ["raw channel[3] (Cy5+JF608)", "(a) winsorize p95 (suppress Cy5)",
                               "(b) log1p compression", "(c) minus white_tophat (remove puncta)",
                               "(d) median filter 5px", "channel[0] DAPI (nuclei; Cy5 sits here)"]):
            a.imshow(_norm(im), cmap="magma"); a.set_title(ttl, fontsize=9); a.axis("off")
        _save(fig2, f"A1_jf608_extraction_T{t}_{label}.png")
    print(f"    (interior tile T-{interior_t}, edge tile T-{edge_t})")


def stage_tiles(ctx):
    """[A1] A representative interior + edge tile (channel [3]) at full res with the cells mask,
    so the well-boundary arc within an edge tile is visible (key for the orientation question)."""
    xy = ctx.meta[["x_pos", "y_pos"]].to_numpy(float)
    d = np.linalg.norm(xy - xy.mean(0), axis=1)
    for label, idx in [("interior", int(np.argmin(d))), ("edge", int(np.argmax(d)))]:
        t = int(ctx.meta.tile.iloc[idx])
        if not os.path.exists(_tiff_of(t)):
            continue
        tile = io.read_sbs_ref_tile(_tiff_of(t))
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(_norm(tile), cmap="gray")
        ax[0].set_title(f"T-{t} ({label}) channel[3] raw — note if a well-edge ARC is present & which way it curves")
        ax[0].axis("off")
        mask_fp = f"{SBS}/sbs/images/P-1_W-A1_T-{t}__cells.tiff"
        if os.path.exists(mask_fp):
            m = io.read_cells_mask(mask_fp)
            ax[1].imshow(_norm(tile), cmap="gray")
            ax[1].imshow(np.where(m > 0, m % 20 + 1, np.nan), cmap="tab20", alpha=0.45)
            ax[1].set_title(f"T-{t} cells.tiff mask overlay ({int((m>0).sum())} labeled px)")
        else:
            ax[1].text(0.5, 0.5, "no cells.tiff", ha="center")
        ax[1].axis("off")
        _save(fig, f"A1_tile_T{t}_{label}.png")


def stage_stage(ctx):
    """[A1] Tile stage-position layout (should trace the physical well) + voltage FOV stage positions."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    xy = ctx.meta[["x_pos", "y_pos"]].to_numpy(float)
    ax[0].scatter(xy[:, 0], xy[:, 1], s=8, c="steelblue")
    ax[0].set_aspect("equal"); ax[0].invert_yaxis()
    ax[0].set_title(f"SBS tile stage positions (µm) — {len(xy)} tiles (physical well layout)")
    ax[0].set_xlabel("x_pos µm"); ax[0].set_ylabel("y_pos µm")
    v = np.array([f["stage_xy"] for f in ctx.fovs()])
    ax[1].scatter(xy[:, 0], xy[:, 1], s=6, c="lightgray", label="SBS tiles")
    ax[1].scatter(v[:, 0], v[:, 1], s=80, c="crimson", marker="x", label="voltage FOVs")
    for i, (x, y) in enumerate(v):
        ax[1].annotate(str(i), (x, y), fontsize=8)
    ax[1].set_aspect("equal"); ax[1].legend()
    ax[1].set_title("voltage FOV stage positions (DIFFERENT scope frame; note overlap extent vs SBS)")
    _save(fig, "A1_stage_positions.png")


def stage_voltage(ctx):
    """[A1] The 8 voltage raw-mean references (the cross-modal moving images)."""
    fovs = ctx.fovs()
    fig, ax = plt.subplots(4, 2, figsize=(12, 9))
    for i, f in enumerate(fovs):
        a = ax.ravel()[i]
        r = f["reference"]
        a.imshow(_norm(r), cmap="gray", aspect="auto")
        a.set_title(f"fov{i} raw-mean ref  std/mean={r.std()/r.mean():.2f}"
                    + ("  (arc/fiber-dominated)" if r.std() / r.mean() > 1.0 else "  (cell-rich)"), fontsize=9)
        a.axis("off")
    _save(fig, "A1_voltage_refs.png")


# ==================================================================================== A2
def _lowres_well(ctx, tile_op="identity", axes=(1, 1, False), ds=16):
    """Assemble a LOW-RES well from cached downsampled tiles, applying ``tile_op`` to each tile
    IMAGE before placing it at the ``axes`` stage placement. Fast stand-in for the full mosaic used
    to diagnose orientation. Returns (well float32, boundary_roughness) — lower roughness == a
    cleaner continuous circular well (see mosaic.well_roughness)."""
    small = ctx.small_tiles(ds)
    px = ctx.px
    raw = {t: mos._stage_rc(*mos._xy(ctx.meta, t), px, axes) for t in ctx.tiles}
    rmin = min(r for r, _ in raw.values()); cmin = min(c for _, c in raw.values())
    place = {t: (int(round((r - rmin) / ds)), int(round((c - cmin) / ds))) for t, (r, c) in raw.items()}
    tds = max(s.shape[0] for s in small.values())
    H = max(r for r, _ in place.values()) + tds
    W = max(c for _, c in place.values()) + tds
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.uint16)
    for t in ctx.tiles:
        im = C.apply_dihedral_image(tile_op, small[t])
        r, c = place[t]; h, w = im.shape
        acc[r:r + h, c:c + w] += im; cnt[r:r + h, c:c + w] += 1
    well = acc / np.maximum(cnt, 1)
    return well, mos.well_roughness(well)


def stage_orient(ctx):
    """[A2, Q1] MOSAIC ORIENTATION DIAGNOSIS. The stitched well should be a filled circular disk
    with a CONTINUOUS boundary. Selection is by boundary ROUGHNESS (mosaic.well_roughness): the
    correct axes gives a continuous circular meniscus rim (low roughness); the wrong ones spike it
    in/out (high). We show the 8 stage-axis placement hypotheses (raw tiles) — the space
    recover_stage_axes chooses among — and, for reference, the 8 per-tile image dihedral ops."""
    # PRIMARY: the 8 stage-axis hypotheses (raw tiles) — this is what the pipeline actually varies.
    axhyp = []
    for hyp in mos.STAGE_AXIS_HYPS:
        well, rough = _lowres_well(ctx, tile_op="identity", axes=hyp)
        axhyp.append((hyp, well, rough))
        print(f"      axes={str(hyp):16s} roughness={rough:.4f}")
    axhyp.sort(key=lambda r: r[2])
    best_axes = axhyp[0][0]
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    for a, (hyp, well, rough) in zip(ax.ravel(), axhyp):
        a.imshow(_norm(well), cmap="gray")
        a.set_title(f"axes={hyp}\nroughness={rough:.4f}" + ("  <- CLEANEST CIRCLE" if hyp == best_axes else ""),
                    fontsize=9)
        a.axis("off")
    fig.suptitle(f"A2 orientation: 8 stage-axis placement hypotheses (raw tiles). Lower boundary "
                 f"roughness = cleaner circular well. recover_stage_axes should pick {best_axes!r}",
                 fontsize=12)
    _save(fig, "A2_orientation_by_stage_axes.png")

    # reference: per-tile IMAGE dihedral (at the best axes) — shows arc-curvature dependence
    fig2, ax2 = plt.subplots(2, 4, figsize=(18, 9))
    for a, op in zip(ax2.ravel(), C.DIHEDRAL_NAMES):
        well, rough = _lowres_well(ctx, tile_op=op, axes=best_axes)
        a.imshow(_norm(well), cmap="gray")
        a.set_title(f"tile_op={op}\nroughness={rough:.4f}", fontsize=9)
        a.axis("off")
    fig2.suptitle(f"A2 orientation (reference): per-tile IMAGE dihedral at axes={best_axes}", fontsize=12)
    _save(fig2, "A2_orientation_by_tile_op.png")
    actual = mos.recover_stage_axes(ctx.meta, _tiff_of)
    print(f"    recover_stage_axes returns {actual!r}; cleanest-circle axes = {best_axes!r}"
          + ("  (MATCH)" if actual == best_axes else "  (MISMATCH — investigate)"))
    return best_axes


def stage_axes(ctx):
    """[A2] recover_stage_axes internals: the 8 hypotheses' adjacent-tile overlap NCC scores."""
    from pyali.registration.mosaic import _neighbor_pairs, _overlap_ncc, _stage_rc, _xy
    px = ctx.px
    present = ctx.meta[ctx.meta.tile.map(lambda t: os.path.exists(_tiff_of(int(t))))]
    pairs = _neighbor_pairs(present, 6)
    cache = {t: io.read_sbs_ref_tile(_tiff_of(t)).astype(np.float32) for pr in pairs for t in pr}
    scores = {}
    for hyp in mos.STAGE_AXIS_HYPS:
        vals = []
        for a, b in pairs:
            ra, ca = _stage_rc(*_xy(ctx.meta, a), px, hyp)
            rb, cb = _stage_rc(*_xy(ctx.meta, b), px, hyp)
            v = _overlap_ncc(cache[a], cache[b], rb - ra, cb - ca)
            if v > -1:
                vals.append(v)
        scores[str(hyp)] = float(np.mean(vals)) if vals else -1.0
    fig, ax = plt.subplots(figsize=(11, 5))
    items = sorted(scores.items(), key=lambda kv: -kv[1])
    ax.bar([k for k, _ in items], [v for _, v in items], color="teal")
    ax.set_ylabel("mean adjacent-tile overlap NCC"); ax.tick_params(axis="x", rotation=45)
    ax.set_title("A2 recover_stage_axes: overlap NCC per stage-axis hypothesis (max wins)")
    _save(fig, "A2_recover_stage_axes_scores.png")


def stage_mosaic(ctx):
    """[A2] Full mosaic thumbnail + per-tile placement map + coverage (overlap) count."""
    m, off, axes = ctx.mosaic()
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(_norm(reg.downscale(m, 24)), cmap="gray")
    ax[0].set_title(f"mosaic {m.shape} (down x24), axes={axes}"); ax[0].axis("off")
    for t, (r0, c0) in off.items():
        ax[1].add_patch(Rectangle((c0, r0), mos.TILE, mos.TILE, fill=False, ec="steelblue", lw=0.4))
    ax[1].set_xlim(0, m.shape[1]); ax[1].set_ylim(m.shape[0], 0); ax[1].set_aspect("equal")
    ax[1].set_title(f"per-tile placement `off` ({len(off)} tiles)")
    # coverage
    cov = np.zeros((m.shape[0] // 24, m.shape[1] // 24), np.uint8)
    for t, (r0, c0) in off.items():
        cov[r0 // 24:(r0 + mos.TILE) // 24, c0 // 24:(c0 + mos.TILE) // 24] += 1
    im = ax[2].imshow(cov, cmap="viridis"); ax[2].set_title("tile coverage count (overlaps)")
    ax[2].axis("off"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    _save(fig, "A2_mosaic.png")


# ==================================================================================== A3
def stage_cells(ctx):
    """[A3] Placed SBS cell centroids (i_mos, j_mos) over the mosaic thumbnail."""
    m, off, _ = ctx.mosaic()
    placed = pc.place_sbs_cells(ctx.sbs_cells, off)
    ds = 24
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(_norm(reg.downscale(m, ds)), cmap="gray")
    ax[0].scatter(placed.j_mos / ds, placed.i_mos / ds, s=0.3, c="cyan", alpha=0.4)
    ax[0].set_title(f"{len(placed)} placed cell centroids on mosaic (down x{ds})"); ax[0].axis("off")
    geno = placed.gene_symbol_0.notna()
    ax[1].imshow(_norm(reg.downscale(m, ds)), cmap="gray")
    ax[1].scatter(placed.j_mos[geno] / ds, placed.i_mos[geno] / ds, s=0.5, c="lime", alpha=0.5)
    ax[1].set_title(f"{int(geno.sum())} genotyped cells (gene_symbol_0 not null)"); ax[1].axis("off")
    _save(fig, "A3_placed_cells.png")


# ==================================================================================== A4
def stage_coarse(ctx):
    """[A4] The downsampled + band-passed mosaic (what the coarse localizer matches against) and a
    band-passed voltage reference — shows the SIGNAL each side contributes to cross-modal matching."""
    m, _, _ = ctx.mosaic()
    md = reg.downscale(m, 12)
    mbp = reg._band_pass(md)
    ref = ctx.fovs()[1]["reference"]                              # a cell-rich FOV
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    ax[0, 0].imshow(_norm(md), cmap="gray"); ax[0, 0].set_title("mosaic down x12 (raw)"); ax[0, 0].axis("off")
    ax[0, 1].imshow(_norm(mbp), cmap="gray")
    ax[0, 1].set_title("mosaic band-pass DoG(1,8) — coarse-match target\n(dominated by bright Cy5 puncta?)")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(_norm(ref), cmap="gray"); ax[1, 0].set_title("voltage ref (fov1)"); ax[1, 0].axis("off")
    ax[1, 1].imshow(_norm(reg._band_pass(ref)), cmap="gray")
    ax[1, 1].set_title("voltage ref band-pass DoG(1,8)"); ax[1, 1].axis("off")
    _save(fig, "A4_coarse_bandpass.png")


def stage_dihedral(ctx):
    """[A4] The 8 dihedral candidates of a voltage FOV + recover_dihedral NCC per op."""
    m, _, _ = ctx.mosaic()
    mbp = reg._band_pass(reg.downscale(m, 12))
    ref = ctx.fovs()[0]["reference"]
    _op, scores = reg.recover_dihedral(ref, m, f=12, mosaic_bp=mbp, return_scores=True)
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    for a, op in zip(ax.ravel(), C.DIHEDRAL_NAMES):
        a.imshow(_norm(C.apply_dihedral_image(op, ref)), cmap="gray", aspect="auto")
        a.set_title(f"{op}  NCC={scores[op]:.3f}", fontsize=9); a.axis("off")
    fig.suptitle(f"A4 recover_dihedral(fov0): best={max(scores, key=scores.get)!r} "
                 f"(margin {sorted(scores.values())[-1]-sorted(scores.values())[-2]:.3f})")
    _save(fig, "A4_dihedral_candidates.png")


def stage_localize(ctx):
    """[A4] _localize response maps: where each voltage FOV best matches in the downsampled mosaic."""
    from scipy.signal import fftconvolve
    from skimage.transform import rescale
    m, _, _ = ctx.mosaic()
    mbp = reg._band_pass(reg.downscale(m, 12))
    dih = reg.recover_dihedral(ctx.fovs()[0]["reference"], m, f=12, mosaic_bp=mbp)
    fig, ax = plt.subplots(2, 4, figsize=(18, 9))
    for a, f in zip(ax.ravel(), ctx.fovs()):
        s, c, ctr = reg._localize(mbp, C.apply_dihedral_image(dih, f["reference"]), 12)
        a.imshow(_norm(mbp), cmap="gray")
        a.scatter([ctr[0]], [ctr[1]], c="red", marker="+", s=200)
        a.set_title(f"fov{f['fov']} best NCC={c:.3f} scale={s:.2f}", fontsize=9); a.axis("off")
    fig.suptitle(f"A4 _localize (dihedral={dih!r}): red + = best match center. "
                 f"Weak/scattered peaks => no lock.")
    _save(fig, "A4_localize.png")


def stage_register(ctx):
    """[A4] global_register anchors + per-FOV fine overlay (voltage ref warped into its mosaic crop)."""
    from skimage.transform import warp, SimilarityTransform
    m, _, _ = ctx.mosaic()
    mbp = reg._band_pass(reg.downscale(m, 12))
    dih = reg.recover_dihedral(ctx.fovs()[0]["reference"], m, f=12, mosaic_bp=mbp)
    S, exp, rep = reg.global_register(ctx.fovs(), m, dih, f=12, mosaic_bp=mbp, return_report=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(_norm(reg.downscale(m, 24)), cmap="gray")
    if S is not None:
        pred = C.apply_affine(S, np.array([f["stage_xy"] for f in ctx.fovs()]))
        ax[0].scatter(pred[:, 0] / 24, pred[:, 1] / 24, c="red", marker="x", s=80)
    ax[0].set_title(f"global_register: predicted FOV centers (S {'fit' if S is not None else 'None'}, "
                    f"scale={exp}); {sum(r['passed'] for r in rep)}/8 passed gates"); ax[0].axis("off")
    # fine overlay for one confident FOV (or fov0)
    idx = next((r["fov"] for r in rep if r["passed"]), 0)
    if S is not None:
        T, sc = reg.fine_register(ctx.fovs()[idx], m, S, dih, exp)
        crop, (c0, r0) = reg._crop_around(m, *C.apply_affine(S, ctx.fovs()[idx]["stage_xy"][None, :])[0], (1100, 3400))
        di = C.apply_dihedral_image(dih, ctx.fovs()[idx]["reference"])
        Tc = C.compose(C.translation(-c0, -r0), T)
        warped = warp(np.asarray(di, float), SimilarityTransform(matrix=C.invert(Tc)),
                      output_shape=crop.shape, order=1, cval=np.nan, preserve_range=True)
        ov = np.dstack([_norm(crop), _norm(np.nan_to_num(warped)), np.zeros_like(crop, float)])
        ax[1].imshow(ov); ax[1].set_title(f"fov{idx} fine overlay (R=mosaic, G=warped voltage) NCC={sc:.3f}")
    ax[1].axis("off")
    _save(fig, "A4_register.png")


# ==================================================================================== registry
STAGES = {
    "channels": (stage_channels, "A1/Q4: SBS 5-channel breakdown + JF608 (Cy5-suppressed) extraction"),
    "tiles": (stage_tiles, "A1: sample interior/edge tiles (channel[3]) + cells mask overlay"),
    "stage": (stage_stage, "A1: tile & voltage-FOV stage-position layouts"),
    "voltage": (stage_voltage, "A1: the 8 voltage raw-mean references"),
    "axes": (stage_axes, "A2: recover_stage_axes overlap-NCC per hypothesis"),
    "orient": (stage_orient, "A2/Q1: MOSAIC ORIENTATION diagnosis (per-tile image dihedral vs solidity)"),
    "mosaic": (stage_mosaic, "A2: full mosaic thumbnail + placement map + coverage [HEAVY]"),
    "cells": (stage_cells, "A3: placed SBS cell centroids on the mosaic [HEAVY]"),
    "coarse": (stage_coarse, "A4: downsampled + band-passed mosaic vs voltage ref [HEAVY]"),
    "dihedral": (stage_dihedral, "A4: 8 dihedral candidates of a FOV + NCC [HEAVY]"),
    "localize": (stage_localize, "A4: _localize response/peak per FOV [HEAVY]"),
    "register": (stage_register, "A4: global anchors + fine overlay [HEAVY]"),
}
ORDER = list(STAGES)


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description="Registration A0-A4 intermediate visualizations.")
    ap.add_argument("stages", nargs="*", help="stage names (default: all). See --list.")
    ap.add_argument("--out", default=OUT, help=f"output dir (default {OUT})")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args(argv)
    OUT = args.out
    if args.list:
        print("stages (fast first):")
        for k in ORDER:
            print(f"  {k:10s} {STAGES[k][1]}")
        return
    want = args.stages or (["all"])
    if want == ["all"]:
        want = ORDER
    ctx = Ctx()
    for s in want:
        if s not in STAGES:
            print(f"!! unknown stage {s!r}; --list to see options"); continue
        print(f"\n== {s}: {STAGES[s][1]} ==")
        STAGES[s][0](ctx)
    print(f"\nDONE. PNGs in {OUT}")


if __name__ == "__main__":
    main()
