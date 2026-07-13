"""Benchmark JF608 demixing methods by cross-modal NCC (D11), standalone in the registration module.

For each demixing method (raw / winsorize@pct / log / nuclei / nuclei_dapi) it stitches a DOWNSAMPLED
JF608 anchor mosaic and localizes the 8 real voltage FOVs, reporting the coarse peak NCC per FOV
(the cross-modal match score the whole pipeline hinges on) plus recovered scale/dihedral. A better
demix should raise the peak NCC and make the recovered scale/dihedral consistent across FOVs.

Tiles are read once (full 5-channel stacks); downsampled DAPI + Cy5 channels and nucleus centroids
are cached, so all methods share that single I/O pass. `cy5_regress` is multi-cycle (needs >1 cycle
on disk) and is reported as pending, not run.

Run:
  PY="/Users/alec/Claude/Projects/miniali python port/pyali/pyali/bin/python"
  "$PY" -m pyali.registration.benchmark_demix                 # coarse NCC table + anchor previews
  "$PY" -m pyali.registration.benchmark_demix --fine          # also full-res fine NCC on best method (slow)

Outputs a table to stdout + anchor-preview PNGs under pyreg_data/registration_viz/.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

from . import io, mosaic as mos, register as reg, demix, coordinates as C
from ..params import Params

DATA = os.environ.get("PYREG_DATA", "/Users/alec/Claude/Projects/pyreg_data")
SBS = f"{DATA}/brieflow_output"
VOLT = f"{DATA}/voltage/Data_A1"
OUT = f"{DATA}/registration_viz"
META_FP = f"{SBS}/preprocess/metadata/sbs/P-1_W-A1__combined_metadata.parquet"
INFO_FP = f"{SBS}/sbs/parquets/P-1_W-A1__sbs_info.parquet"
CELLS_FP = f"{SBS}/sbs/parquets/P-1_W-A1__cells.parquet"
AXES = (1, 1, False)                 # frozen orientation (D10)
F = 12                               # coarse downscale
SCALES = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)   # scale grid for the benchmark (coarser than pipeline default)


def _tiff_of(t):
    return f"{SBS}/preprocess/images/sbs/P-1_W-A1_T-{t}_C-1__image.tiff"


def _cache_tiles(meta, sbs_info):
    """Read each tile stack ONCE; cache downscaled Cy5 + DAPI channels and tile-local nucleus coords."""
    tiles = [int(t) for t in meta.tile.tolist() if os.path.exists(_tiff_of(int(t)))]
    ds_ch4, ds_dapi, nuc = {}, {}, {}
    t0 = time.time()
    for i, t in enumerate(tiles):
        stk = io.read_sbs_stack(_tiff_of(t))
        ds_ch4[t] = reg.downscale(stk[demix.CY5_CH], F)
        ds_dapi[t] = reg.downscale(stk[demix.DAPI_CH], F)
        sub = sbs_info[sbs_info.tile == t]
        rc = sub[["i", "j"]].to_numpy(float) / F
        rad = np.sqrt(sub["area"].to_numpy(float) / np.pi) / F if "area" in sub.columns else None
        nuc[t] = (rc, rad)
        if (i + 1) % 80 == 0:
            print(f"    cached {i+1}/{len(tiles)} tile stacks ({time.time()-t0:.0f}s)")
    return tiles, ds_ch4, ds_dapi, nuc


def _placement(meta, tiles, px):
    raw = {t: mos._stage_rc(*mos._xy(meta, t), px, AXES) for t in tiles}
    rmin = min(r for r, _ in raw.values()); cmin = min(c for _, c in raw.values())
    return {t: (int(round((r - rmin) / F)), int(round((c - cmin) / F))) for t, (r, c) in raw.items()}


def _assemble(method, tiles, ds_ch4, ds_dapi, nuc, place, pct):
    tds = max(a.shape[0] for a in ds_ch4.values())
    H = max(r for r, _ in place.values()) + tds
    W = max(c for _, c in place.values()) + tds
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.uint16)
    for t in tiles:
        rc, rad = nuc[t]
        anc = demix.jf608_anchor(ds_ch4[t], method, dapi=ds_dapi[t], nuclei_rc=rc, radii=rad, pct=pct)
        r, c = place[t]; h, w = anc.shape
        acc[r:r + h, c:c + w] += anc; cnt[r:r + h, c:c + w] += 1
    return acc / np.maximum(cnt, 1)


def _coarse_eval(method, pct, fovs, tiles, ds_ch4, ds_dapi, nuc, place):
    """Return dict: dihedral, per-FOV (ncc, scale). Coarse localization at the frozen downscale F."""
    mbp = reg._band_pass(_assemble(method, tiles, ds_ch4, ds_dapi, nuc, place, pct))
    dih, dscores = reg.recover_dihedral(fovs[0]["reference"], None, f=F, scales=SCALES,
                                        mosaic_bp=mbp, return_scores=True)
    per = []
    for fv in fovs:
        s, c, _ctr = reg._localize(mbp, C.apply_dihedral_image(dih, fv["reference"]), F, scales=SCALES)
        per.append((float(c), float(s)))
    margin = float(sorted(dscores.values())[-1] - sorted(dscores.values())[-2])
    return dict(dihedral=dih, margin=margin, per=per,
                mean_ncc=float(np.mean([p[0] for p in per])),
                scale_cv=float(np.std([p[1] for p in per]) / (np.mean([p[1] for p in per]) + 1e-9)))


def _preview(meta, sbs_info):
    """Render raw Cy5 channel vs each single-cycle demix anchor for one Cy5/Cy7-heavy tile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = int(meta.tile.iloc[len(meta) // 2])
    stk = io.read_sbs_stack(_tiff_of(t))
    sub = sbs_info[sbs_info.tile == t]
    rc = sub[["i", "j"]].to_numpy(float)
    rad = np.sqrt(sub["area"].to_numpy(float) / np.pi) if "area" in sub.columns else None

    def nrm(a):
        p1, p2 = np.percentile(a, [1, 99]); return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)

    panels = [("raw Ch4 (JF608 + Cy5 + Cy7 xtalk)", stk[demix.CY5_CH].astype(float)),
              ("winsorize p99", demix.jf608_anchor(stk[demix.CY5_CH], "winsorize", pct=99)),
              ("log1p", demix.jf608_anchor(stk[demix.CY5_CH], "log")),
              ("nuclei down-weight (coords)", demix.jf608_anchor(stk[demix.CY5_CH], "nuclei", nuclei_rc=rc, radii=rad)),
              ("nuclei_dapi down-weight", demix.jf608_anchor(stk[demix.CY5_CH], "nuclei_dapi", dapi=stk[demix.DAPI_CH])),
              ("DAPI (nuclei ref)", stk[demix.DAPI_CH].astype(float))]
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for a, (ttl, im) in zip(ax.ravel(), panels):
        a.imshow(nrm(im), cmap="magma"); a.set_title(ttl, fontsize=10); a.axis("off")
    os.makedirs(OUT, exist_ok=True)
    fp = f"{OUT}/D11_demix_anchors_T{t}.png"
    fig.savefig(fp, dpi=90, bbox_inches="tight"); plt.close(fig)
    print(f"    wrote {fp}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark JF608 demixing by cross-modal NCC (D11).")
    ap.add_argument("--fine", action="store_true", help="also run full-res fine NCC on the best method (slow)")
    args = ap.parse_args(argv)

    meta = io.read_sbs_metadata(META_FP)
    sbs_info = __import__("pandas").read_parquet(INFO_FP)
    px = float(meta.pixel_size_x.iloc[0])
    fov_dirs = sorted(glob.glob(f"{VOLT}/*_burst*"))
    fovs = [dict(fov=i, **io.read_voltage_reference(d, Params())) for i, d in enumerate(fov_dirs)]
    print(f"benchmark_demix: {len(fovs)} voltage FOVs, axes={AXES}, F={F}, scales={SCALES}")

    tiles, ds_ch4, ds_dapi, nuc = _cache_tiles(meta, sbs_info)
    place = _placement(meta, tiles, px)

    trials = [("raw", None), ("log", None), ("nuclei", None), ("nuclei_dapi", None),
              ("winsorize", 95.0), ("winsorize", 98.0), ("winsorize", 99.0)]
    print(f"\n{'method':16s} {'dih':10s} {'dmargin':>7s} {'meanNCC':>8s} {'scaleCV':>7s}   per-FOV NCC")
    results = []
    for method, pct in trials:
        r = _coarse_eval(method, pct, fovs, tiles, ds_ch4, ds_dapi, nuc, place)
        label = f"{method}@{pct:.0f}" if pct else method
        results.append((label, method, pct, r))
        nccs = " ".join(f"{c:+.2f}" for c, _ in r["per"])
        print(f"{label:16s} {r['dihedral']:10s} {r['margin']:>7.3f} {r['mean_ncc']:>8.3f} "
              f"{r['scale_cv']:>7.3f}   {nccs}")

    best = max(results, key=lambda x: x[3]["mean_ncc"])
    print(f"\nbest coarse mean NCC: {best[0]}  (meanNCC={best[3]['mean_ncc']:.3f}); "
          f"raw baseline = {results[0][3]['mean_ncc']:.3f}")
    print("NOTE: cy5_regress (multi-cycle, barcode-informed) NOT run — only cycle C-1 is on disk. "
          "See report for the exact data to download.")
    _preview(meta, sbs_info)

    if args.fine:
        method, pct = best[1], best[2]
        print(f"\n--fine: full-res mosaic for {best[0]} + fine_register 3 FOVs ...")
        reader = demix.make_anchor_reader(method, _tiff_of, sbs_info=sbs_info,
                                          pct=(pct or demix.DEFAULT_WINSOR_PCT))
        m, off, _ = mos.build_sbs_mosaic(meta.tile.tolist(), _tiff_of, meta, axes=AXES,
                                         read_tile=reader, dtype=np.float32)
        mbp = reg._band_pass(reg.downscale(m, F))
        dih = reg.recover_dihedral(fovs[0]["reference"], None, f=F, scales=SCALES, mosaic_bp=mbp)
        S, exp, rep = reg.global_register(fovs, m, dih, f=F, scales=SCALES, mosaic_bp=mbp, return_report=True)
        if S is not None:
            for i in [r["fov"] for r in rep if r["passed"]][:3]:
                T, sc = reg.fine_register(fovs[i], m, S, dih, exp)
                print(f"    fov{i}: fine NCC={sc:+.3f} scale={reg._recovered_scale(T):.2f}")
        else:
            print("    global_register: <2 FOVs passed gates even after demix (reported).")


if __name__ == "__main__":
    main()
