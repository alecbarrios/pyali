"""Dev-smoke harness for the registration module (REGISTRATION_PSEUDOCODE.md §8).

Exercises the built slice against the REAL downloaded data at pyreg_data/. Grows as modules land:
  [A0] import + coordinates.fit_similarity
  [A1] io.py  — SBS loaders + LIGHT voltage load (reference + stage_xy)
  [A2..] mosaic / place_cells / register / assign  (stubs below; enable as implemented)

Deliberately does NOT run pyali.process_fov here (it holds the full ~19 GB movie in RAM); the
full read_voltage_fov / assignment step is verified separately once register+assign land.

Run:  "/Users/alec/Claude/Projects/miniali python port/pyali/pyali/bin/python" \
        "/Users/alec/Claude/Projects/miniali python port/pyali/scripts/dev_smoke_registration.py"
"""
import sys
import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # outer pyali/ -> import pyali

from pyali.registration import coordinates, phasecorr, io       # noqa: E402
from pyali.params import Params                                 # noqa: E402

DATA = "/Users/alec/Claude/Projects/pyreg_data"
SBS = f"{DATA}/brieflow_output"
VOLT = f"{DATA}/voltage/Data_A1"
PASS, FAIL = "  ok ", "FAIL "
n_fail = 0


def check(name, cond, detail=""):
    global n_fail
    ok = bool(cond)
    n_fail += (not ok)
    print(f"{PASS if ok else FAIL}{name}" + (f"  [{detail}]" if detail else ""))


# ----------------------------------------------------------------------------- [A0]
print("\n== A0: import + coordinates.fit_similarity ==")
check("import pyali.registration", all(m is not None for m in (coordinates, phasecorr, io)))

# known similarity: scale 2.0, rotation 30 deg, translation (10, -5); recover it from correspondences
M_true = coordinates.similarity_matrix(scale=2.0, rotation=np.deg2rad(30), tx=10.0, ty=-5.0)
src = np.array([[0, 0], [100, 0], [0, 50], [40, 80], [90, 20]], float)
dst = coordinates.apply_affine(M_true, src)
M_fit = coordinates.fit_similarity(src, dst)
resid = np.abs(coordinates.apply_affine(M_fit, src) - dst).max()
check("fit_similarity recovers a known similarity", resid < 1e-6, f"max resid {resid:.2e}px")

# ----------------------------------------------------------------------------- [A1] SBS
print("\n== A1: SBS loaders ==")
meta = io.read_sbs_metadata(f"{SBS}/preprocess/metadata/sbs/P-1_W-A1__combined_metadata.parquet")
check("read_sbs_metadata -> per-tile rows", meta.tile.nunique() == len(meta),
      f"{len(meta)} tiles")
check("metadata has 349 tiles", meta.tile.nunique() == 349, f"n={meta.tile.nunique()}")
px = float(meta.pixel_size_x.iloc[0])
check("pixel_size_x ~ 0.842914", abs(px - 0.842914) < 1e-4, f"{px:.6f} um/px")

sbs_cells = io.read_sbs_cells(f"{SBS}/sbs/parquets/P-1_W-A1__sbs_info.parquet",
                              f"{SBS}/sbs/parquets/P-1_W-A1__cells.parquet")
check("read_sbs_cells -> nuclei rows", len(sbs_cells) > 100_000, f"{len(sbs_cells)} rows")
check("has genotype cols", {"gene_symbol_0", "cell_barcode_0"} <= set(sbs_cells.columns),
      f"cols={[c for c in sbs_cells.columns if c.startswith(('gene_symbol','cell_barcode'))][:4]}")
check("sbs_info tiles start at 1 (no tile 0)", sbs_cells.tile.min() == 1,
      f"min tile {sbs_cells.tile.min()}")

tile_img = io.read_sbs_ref_tile(f"{SBS}/preprocess/images/sbs/P-1_W-A1_T-1_C-1__image.tiff")
check("read_sbs_ref_tile -> (1480,1480) uint16", tile_img.shape == (1480, 1480)
      and tile_img.dtype == np.uint16, f"{tile_img.shape} {tile_img.dtype}")

mask = io.read_cells_mask(f"{SBS}/sbs/images/P-1_W-A1_T-1__cells.tiff")
mask_labels = set(int(v) for v in np.unique(mask) if v != 0)
info_labels = set(int(c) for c in sbs_cells.loc[sbs_cells.tile == 1, "cell"])
check("read_cells_mask -> (1480,1480) int64", mask.shape == (1480, 1480) and mask.dtype == np.int64)
check("mask labels == sbs_info tile-1 cell ids", mask_labels == info_labels,
      f"mask{sorted(mask_labels)[:6]}.. == info{sorted(info_labels)[:6]}..")

# ----------------------------------------------------------------------------- [A1] voltage (light)
print("\n== A1: voltage light load (no process_fov) ==")
b1 = sorted(glob.glob(f"{VOLT}/*_burst1"))[0]
xy = io.read_stage_xy(f"{b1}/output_data.mat")
check("read_stage_xy -> (2,) float64 um", xy.shape == (2,) and xy.dtype == np.float64,
      f"stage=({xy[0]:.1f}, {xy[1]:.1f})")
# expected from the earlier data-schema exploration: burst1 ~ (-77642.40, 27365.80)
check("stage_xy matches known burst1 value", np.allclose(xy, [-77642.40, 27365.80], atol=1.0),
      f"{xy}")

p = Params()
ref = io.raw_mean_reference(f"{b1}/frames1.bin", p.nrow, p.ncol, p.n_ref)
check("raw_mean_reference -> (312,1200) float32", ref.shape == (p.nrow, p.ncol)
      and ref.dtype == np.float32, f"{ref.shape} {ref.dtype}")
check("reference finite + non-trivial", np.isfinite(ref).all() and ref.std() > 0,
      f"mean={ref.mean():.1f} std={ref.std():.1f}")

light = io.read_voltage_reference(b1, p)
check("read_voltage_reference bundles reference+stage_xy",
      set(light) == {"reference", "stage_xy"} and light["reference"].shape == (312, 1200))

# ----------------------------------------------------------------------------- [A2] mosaic
print("\n== A2: SBS mosaic on the real well (349 tiles) ==")
import time                                                        # noqa: E402
from scipy.spatial import cKDTree                                  # noqa: E402
from pyali.registration import mosaic as mos, place_cells as pc, register as reg   # noqa: E402
from pyali.registration.mosaic import _HALF                        # noqa: E402

tiff_of = lambda t: f"{SBS}/preprocess/images/sbs/P-1_W-A1_T-{t}_C-1__image.tiff"
axes = mos.recover_stage_axes(meta, tiff_of, verbose=True)
check("recover_stage_axes -> (sx,sy,swap)", isinstance(axes, tuple) and len(axes) == 3, f"{axes}")
# [D10] the well-geometry recovery must pick the orientation that stitches a CLEAN CIRCLE. For this
# well that is (1,1,False); the old overlap-NCC-only recovery wrongly returned (-1,1,False) (a jagged
# mirror). This guards the orientation bug the earlier A2 checks missed.
check("recover_stage_axes picks the circular-well orientation (1,1,False) [D10]",
      axes == (1, 1, False), f"got {axes}")

# regression [review]: a missing mid-well tiff must not crash axis recovery
_bad = int(meta.tile.iloc[len(meta) // 2])
tiff_missing1 = lambda t: (tiff_of(t) + ".NOPE") if int(t) == _bad else tiff_of(t)
try:
    mos.recover_stage_axes(meta, tiff_missing1)
    check("recover_stage_axes tolerates a missing tiff", True, f"dropped tile {_bad}")
except FileNotFoundError:
    check("recover_stage_axes tolerates a missing tiff", False, "raised FileNotFoundError")

t0 = time.time()
mosaic_img, off, A_sbs = mos.build_sbs_mosaic(meta.tile.tolist(), tiff_of, meta, axes=axes, verbose=True)
dt = time.time() - t0
check("build_sbs_mosaic -> uint16 mosaic", mosaic_img.dtype == np.uint16, f"{mosaic_img.shape} in {dt:.0f}s")
check("mosaic non-trivial", min(mosaic_img.shape) > 5000 and int(mosaic_img.max()) > 0)
check("placed most tiles", len(off) >= 340, f"{len(off)}/349 placed")

origins = np.array(list(off.values()), float)
dnn, _ = cKDTree(origins).query(origins, k=2)
pitch = float(np.median(dnn[:, 1]))
check("tiles overlap (pitch < 1480)", pitch < 1480, f"pitch {pitch:.0f}px (~{(1-pitch/1480)*100:.0f}% overlap)")

# [D10] the stitched well must be a clean CIRCLE — the direct guard against the orientation bug.
well_rough = mos.well_roughness(reg.downscale(mosaic_img, 24))
check("stitched well is a clean circle (boundary roughness < 0.02) [D10 orientation guard]",
      well_rough < 0.02, f"roughness {well_rough:.4f}")

scale = float(np.sqrt(abs(np.linalg.det(A_sbs[:2, :2]))))          # stage µm -> px, expect 1/pixel_size
check("A_sbs scale ~ 1/pixel_size", abs(scale - 1 / px) < 0.02, f"{scale:.4f} vs {1/px:.4f} px/um")
# [D7 REVISED — see D10] At the CORRECT orientation (1,1,False) the stage->pixel map is a pure
# positive scale (NO reflection), so fit_similarity is NOT catastrophic — the earlier "fit_similarity
# fails by 13326px" was an ARTIFACT of the wrong (-1,1,False) axes (sx=-1 injected a spurious
# reflection). Analytic construction is still exact & preferred; all three now agree (~rounding).
m_idx = meta.set_index("tile")
tp = list(off.keys())
src = np.array([[float(m_idx.at[t, "x_pos"]), float(m_idx.at[t, "y_pos"])] for t in tp])
ctr = np.array([[off[t][1] + _HALF, off[t][0] + _HALF] for t in tp])
r_analytic = mos.A_sbs_residual(meta, off, A_sbs)
r_affine = mos.A_sbs_residual(meta, off, coordinates.fit_affine(src, ctr))
r_sim = mos.A_sbs_residual(meta, off, coordinates.fit_similarity(src, ctr))
print(f"    D7/D10 A_sbs residual vs off (px): analytic={r_analytic:.2f} fit_affine={r_affine:.2f} fit_similarity={r_sim:.2f}")
check("A_sbs analytic residual < 1px (rounding only)", r_analytic < 1.0, f"{r_analytic:.3f}px")
check("analytic ~ fit_affine", abs(r_analytic - r_affine) < 1.0)
check("fit_similarity matches analytic at the reflection-free orientation (D7 revised)",
      r_sim < 2.0, f"{r_sim:.2f}px (was 13326 at the wrong axes)")

# ----------------------------------------------------------------------------- [A3] place_cells
print("\n== A3: place SBS cells into the mosaic frame ==")
placed = pc.place_sbs_cells(sbs_cells, off)
check("place_sbs_cells adds i_mos/j_mos", {"i_mos", "j_mos"} <= set(placed.columns), f"{len(placed)} cells")
check("placed centroids within mosaic bounds",
      bool(placed.i_mos.between(0, mosaic_img.shape[0] - 1).all()
           and placed.j_mos.between(0, mosaic_img.shape[1] - 1).all()))
check("genotype carried through", "gene_symbol_0" in placed.columns)
row0_t1 = off[1][0]
t1 = placed[placed.tile == 1]
check("i_mos round-trips (i_mos - tile_off row == i)", np.allclose(t1.i_mos - row0_t1, t1.i.to_numpy()))
# tie the i,j convention to the mask: tile-1 centroids should land in their own labeled region
inside = sum(int(mask[int(round(r)), int(round(c))]) == int(cell)
             for r, c, cell in zip(t1.i, t1.j, t1.cell))
check("tile-1 centroids fall in own mask label (majority)", inside >= 0.8 * len(t1),
      f"{inside}/{len(t1)} inside")

# ----------------------------------------------------------------------------- [A4a] register CODE
# register.py geometry is verified on SYNTHETIC ground truth (a locally-unique mosaic with FOVs
# planted at known dihedral/scale/stage), independent of whether the REAL modalities lock (A4b).
print("\n== A4a: register.py CODE correctness (synthetic ground truth) ==")
from pyali.registration import register as reg                        # noqa: E402
from pyali.registration.coordinates import (apply_dihedral_image, apply_affine as _aff,   # noqa: E402
                                             invert as _inv, similarity_matrix as _sim,
                                             DIHEDRAL_NAMES)
from scipy.ndimage import gaussian_filter                             # noqa: E402
from skimage.transform import SimilarityTransform as _ST, warp as _skwarp   # noqa: E402

_INV = {"identity": "identity", "rot90": "rot270", "rot180": "rot180", "rot270": "rot90",
        "fliplr": "fliplr", "flipud": "flipud", "transpose": "transpose",
        "anti_transpose": "anti_transpose"}
_rng = np.random.default_rng(0)
_SH, _SW, _F, _SCALE = 1500, 2000, 3, 3.0
_FH, _FW = 100, 140
_PH, _PW = _FH * _F, _FW * _F
_imp = np.zeros((_SH, _SW), np.float32)
_imp[_rng.integers(0, _SH, 15000), _rng.integers(0, _SW, 15000)] = _rng.uniform(0.4, 1.0, 15000)
_syn = (gaussian_filter(_imp, 1.5) / gaussian_filter(_imp, 1.5).max() * 4000).astype(np.uint16)
_syn_bp = reg._band_pass(reg.downscale(_syn, _F))


def _make_fov(center_rc, op):
    r, c = center_rc
    patch = _syn[r - _PH // 2:r + _PH // 2, c - _PW // 2:c + _PW // 2]
    return apply_dihedral_image(_INV[op], reg.downscale(patch, _F))   # fov_ref s.t. op(ref)=patch_ds

# recover_dihedral recovers every planted op
_ok_dih = True
for _op in DIHEDRAL_NAMES:
    _got = reg.recover_dihedral(_make_fov((750, 960), _op), _syn, f=_F, mosaic_bp=_syn_bp)
    _ok_dih &= (_got == _op)
check("synthetic recover_dihedral recovers all 8 planted ops", _ok_dih)

# global_register recovers a known stage->mosaic similarity S_true
_Strue = _sim(scale=0.4, rotation=np.deg2rad(15), tx=200.0, ty=-100.0)
_ctrs = [(450, 510), (450, 1410), (1050, 510), (1050, 1410), (750, 960)]
_syn_fovs = []
for _i, _ctr in enumerate(_ctrs):
    _cxy = np.array([_ctr[1], _ctr[0]], float)
    _syn_fovs.append(dict(fov=_i, reference=_make_fov(_ctr, "identity"),
                          stage_xy=_aff(_inv(_Strue), _cxy[None, :])[0], _true=_cxy))
_S, _exp = reg.global_register(_syn_fovs, _syn, "identity", f=_F, mosaic_bp=_syn_bp)
_pred = _aff(_S, np.array([f["stage_xy"] for f in _syn_fovs]))
_resid = np.abs(_pred - np.array([f["_true"] for f in _syn_fovs])).max()
check("synthetic global_register recovers S (< 5px, scale, rotation)",
      _S.shape == (3, 3) and _resid < 5.0 and abs(reg._recovered_scale(_S) - 0.4) < 0.02
      and abs(reg._recovered_rotation_deg(_S) - 15.0) < 1.0 and abs(_exp - _SCALE) < 0.3,
      f"resid={_resid:.2f}px Sscale={reg._recovered_scale(_S):.3f} "
      f"Srot={reg._recovered_rotation_deg(_S):+.2f} exp_scale={_exp:.2f}")

# fine_register maps the FOV to its true mosaic location (crop-offset E8 + prescale composed)
_Tf, _sc = reg.fine_register(_syn_fovs[4], _syn, _Strue, "identity", _SCALE, half=(250, 320))
_mapped = _aff(_Tf, reg._center_xy(apply_dihedral_image("identity", _syn_fovs[4]["reference"]))[None, :])[0]
_cerr = float(np.hypot(*(_mapped - _syn_fovs[4]["_true"])))
check("synthetic fine_register maps FOV center to truth (< 5px, NCC>0.5, scale~3)",
      _cerr < 5.0 and _sc > 0.5 and abs(reg._recovered_scale(_Tf) - _SCALE) < 0.3,
      f"cerr={_cerr:.2f}px NCC={_sc:.3f} scale={reg._recovered_scale(_Tf):.3f}")

# edge cases
try:
    reg.global_register(_syn_fovs[:1], _syn, "identity", f=_F, mosaic_bp=_syn_bp)
    check("global_register raises on <2 confident FOVs", False, "did not raise")
except RuntimeError:
    check("global_register raises on <2 confident FOVs", True)
_Sn, _en, _rn = reg.global_register([], _syn, "identity", f=_F, mosaic_bp=_syn_bp, return_report=True)
check("global_register(return_report) returns None on 0 anchors", _Sn is None)
_crp, (_c0, _r0) = reg._crop_around(_syn, -9999, -9999, (250, 320))
_crp2, _ = reg._crop_around(_syn, _SW + 9999, _SH + 9999, (250, 320))
check("crop_around stays non-empty off both mosaic corners", _crp.size > 0 and _crp2.size > 0,
      f"{_crp.shape}/{_crp2.shape}")

# phasecorr depad regression: different-size registration maps in ORIGINAL coords
_tex = gaussian_filter(np.where(_rng.random((256, 256)) > 0.985, 1.0, 0.0), 2.0)
_sub = _tex[60:180, 90:240]                                          # 120x150 window inside 256x256
_Msub, _csub = phasecorr.register(_sub, _tex, "translation", upsample=20)
_mp = _aff(_Msub, np.array([[0.0, 0.0], [149.0, 119.0]]))
_errsub = np.abs(_mp - np.array([[90, 60], [239, 179]], float)).max()
check("phasecorr different-size maps in original coords (depad fix)", _errsub < 2.0, f"{_errsub:.2f}px")

# regression [review CONFIRMED, high]: scale calibration is ORDER-INDEPENDENT — a spurious first FOV
# must NOT mis-calibrate the gate and reject the valid FOVs. Inject controlled localizer outputs.
_ctr_g = [(100.0, 50.0), (300.0, 80.0), (120.0, 350.0), (400.0, 300.0)]     # non-collinear centers
_stg_g = [(10.0, 7.0), (40.0, 9.0), (12.0, 55.0), (60.0, 44.0)]
_orig_localize = reg._localize


def _run_calib(bad_first):
    seq = ([("bad", None)] if bad_first else []) + \
          [("good", i) for i in range(4)] + ([] if bad_first else [("bad", None)])
    outs, fovs = [], []
    for tag, i in seq:
        if tag == "bad":
            outs.append((4.75, 0.35, np.array([500.0, 500.0])))            # spurious: passes NCC, scale outlier
            fovs.append(dict(reference=np.zeros((10, 10), np.float32), stage_xy=np.array([999.0, 999.0])))
        else:
            outs.append((3.0, 0.95, np.array(_ctr_g[i])))
            fovs.append(dict(reference=np.zeros((10, 10), np.float32), stage_xy=np.array(_stg_g[i])))
    _it = iter(outs)
    reg._localize = lambda *a, **k: next(_it)
    try:
        return reg.global_register(fovs, np.zeros((50, 50), np.float32), "identity", f=1,
                                   min_ncc=0.30, scale_tol=0.5, mosaic_bp=np.zeros((50, 50)),
                                   return_report=True)
    finally:
        reg._localize = _orig_localize


_Sa, _ea, _ra = _run_calib(bad_first=True)
_Sb, _eb, _rb = _run_calib(bad_first=False)
check("robust scale calibration is order-independent (spurious first FOV doesn't reject valid ones)",
      _Sa is not None and _Sb is not None and sum(r["passed"] for r in _ra) == 4
      and sum(r["passed"] for r in _rb) == 4 and np.allclose(_Sa, _Sb, atol=1e-6),
      f"bad-first passed={sum(r['passed'] for r in _ra)} bad-last passed={sum(r['passed'] for r in _rb)} "
      f"expected={_ea:.2f}")

# regression [review PLAUSIBLE]: a degenerate 0x0 reference must not crash the never-crash paths
_bad = np.zeros((0, 0), np.float32)
_no_crash = True
try:
    reg.global_register([dict(fov=0, reference=_bad, stage_xy=np.array([0.0, 0.0])),
                         dict(fov=1, reference=_bad, stage_xy=np.array([1.0, 1.0]))],
                        _syn, "identity", f=_F, mosaic_bp=_syn_bp, return_report=True)
    _Tb, _scb = reg.fine_register(dict(reference=_bad, stage_xy=np.array([0.0, 0.0])),
                                  _syn, _Strue, "identity", 3.0, half=(50, 50))
except Exception as _exc:
    _no_crash = False
check("degenerate 0x0 reference does not crash global/fine (never-crash contract)", _no_crash)

# ----------------------------------------------------------------------------- [A4b] register REAL
# SCIENTIFIC CRUX (real data): does the raw-mean voltage JF608 reference lock onto the SBS JF608
# mosaic? Reported honestly; NOT forced to pass (per the task). Only structural invariants are hard.
print("\n== A4b: register.py on the REAL well + 8 voltage FOVs (SCIENTIFIC CRUX report) ==")
n_crux_unmet = 0


def crux(name, cond, detail=""):
    global n_crux_unmet
    ok = bool(cond); n_crux_unmet += (not ok)
    print(f"{'  LOCK ' if ok else ' NOLOCK '}{name}" + (f"  [{detail}]" if detail else ""))


F_REAL = 12
mosaic_bp_real = reg._band_pass(reg.downscale(mosaic_img, F_REAL))
fov_dirs = sorted(glob.glob(f"{VOLT}/*_burst*"))
real_fovs = [dict(fov=i, **io.read_voltage_reference(d, Params())) for i, d in enumerate(fov_dirs)]
check("loaded 8 real voltage references", len(real_fovs) == 8, f"{len(real_fovs)} FOVs")

dih, dih_scores = reg.recover_dihedral(real_fovs[0]["reference"], mosaic_img, f=F_REAL,
                                       mosaic_bp=mosaic_bp_real, return_scores=True)
check("recover_dihedral returns one of the 8 names", dih in DIHEDRAL_NAMES, f"{dih!r}")
_best2 = sorted(dih_scores.values())[-2:]
print(f"    dihedral NCCs: " + " ".join(f"{k}={v:.2f}" for k, v in
      sorted(dih_scores.items(), key=lambda kv: -kv[1])))

S_real, exp_real, rep = reg.global_register(real_fovs, mosaic_img, dih, f=F_REAL,
                                            mosaic_bp=mosaic_bp_real, return_report=True)
for r in rep:
    print(f"    fov{r['fov']}: ncc={r['ncc']:+.3f} scale={r['scale']:.2f} "
          f"{'PASS' if r['passed'] else 'drop: ' + r['reason']}")
check("global_register returns 3x3 S or None (never crashes)", S_real is None or S_real.shape == (3, 3),
      f"expected_scale={exp_real}")
n_pass = sum(r["passed"] for r in rep)
confident_idx = [r["fov"] for r in rep if r["passed"]]
# Literal task-target metrics — REPORTED (they pass here, but on SPURIOUS coarse peaks; the
# decisive checks below expose that), not the basis of the verdict:
print(f"    [reported] {n_pass}/8 FOVs cleared the NCC>=0.30 gate; "
      f"confident-FOV coarse scales = {[round(r['scale'], 2) for r in rep if r['passed']]}")

# Decisive lock evidence (these DRIVE the verdict; naive NCC-count/scale-CV can pass on spurious peaks):
_edge = (reg.DEFAULT_SCALES[0], reg.DEFAULT_SCALES[-1])
_conf_scales = [r["scale"] for r in rep if r["passed"]]
_railed = bool(_conf_scales) and all(min(abs(s - _edge[0]), abs(s - _edge[1])) < 1e-6 for s in _conf_scales)
_margin = float(_best2[-1] - _best2[-2]) if len(_best2) == 2 else 0.0
crux("dihedral op is DECISIVE (best-vs-2nd margin > 0.10)", _margin > 0.10, f"margin {_margin:.3f}")
crux("coarse scale is an interior optimum (not railed to a grid edge)", not _railed,
     f"confident scales {[round(s, 2) for s in _conf_scales]} (grid edges {_edge})")

fine_nccs = []
if S_real is not None and confident_idx:
    for i in confident_idx[:3]:                                     # fine on confident FOVs (slow step)
        T, sc = reg.fine_register(real_fovs[i], mosaic_img, S_real, dih, exp_real)
        check(f"fine_register fov{i} -> (T 3x3, float score)", T.shape == (3, 3))
        fine_nccs.append(sc)
        print(f"    fov{i}: fine NCC={sc:+.3f} scale={reg._recovered_scale(T):.2f} "
              f"rot={reg._recovered_rotation_deg(T):+.1f}")
_fine_lock = sum(s >= 0.30 for s in fine_nccs)
crux("fine_register NCC >= 0.30 on >= half of confident FOVs (the decisive alignment test)",
     bool(fine_nccs) and _fine_lock >= max(1, len(fine_nccs) // 2),
     f"{_fine_lock}/{len(fine_nccs)} fine locks; NCCs={[round(s, 2) for s in fine_nccs]}")

print("\n  CRUX VERDICT: " + ("real modalities LOCK (unexpected here — investigate/celebrate)."
      if n_crux_unmet == 0 else
      "raw-mean voltage JF608 does NOT robustly lock onto the SBS JF608 mosaic. The coarse "
      "peaks are weak (~0.3-0.5), the winning dihedral is ambiguous, the recovered scale rails to "
      "the grid edge, and fine NCC collapses (<=0.1) -> spurious, not correspondence. This is a "
      "REPORTED finding, NOT a code fault (register.py is verified on synthetic ground truth in "
      "A4a). See register.py header + REGISTRATION_PSEUDOCODE.md §4. Candidate fixes: §10 band-pass "
      "tuning, cross-modal reference choice, partial voltage/SBS overlap + arc-artifact masking."))

# ----------------------------------------------------------------------------- summary
print(f"\n== summary: CODE {'ALL PASS' if n_fail == 0 else str(n_fail) + ' FAILED'}"
      f"  |  real cross-modal lock: {'ACHIEVED' if n_crux_unmet == 0 else 'NOT achieved (' + str(n_crux_unmet) + ' unmet — reported)'} ==")
print("A4 register.py: geometry verified on synthetic ground truth; real-data lock reported honestly.")
print("deferred (need later modules): assign[A5] outputs[A6]; full read_voltage_fov(process_fov) at A5.")
sys.exit(1 if n_fail else 0)
