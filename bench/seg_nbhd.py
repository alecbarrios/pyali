#!/usr/bin/env python3
"""Sweep the adaptive-threshold neighbourhood at sharpen_k=3.0, multiplier=1.5.

The neighbourhood sets the spatial scale of "local" in ``T = local_mean * mult``. The default
``2*floor(size/16)+1`` is 101x101 at 800x800 and an anisotropic 39x151 at 312x1200 — far larger
than a cell (median region area ~30 px, so ~6 px across). At that scale two adjacent cells sit on
one common plateau of the local mean and fuse. Shrinking the window toward cell scale makes the
threshold track local structure and cut the neck between them — unlike raising the multiplier,
which separates blobs only by eroding them and loses dim cells.

Too small is degenerate: as the window approaches the cell itself the local mean rises to meet
the object and it vanishes. The useful range is roughly 3-10x the cell diameter.
"""
import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/jovyan/workbench/pyali")
sys.path.insert(0, "/home/jovyan/bench")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyali import preprocess, segmentation
from pyali.params import Params
from seg_grid import (FACTORY, PROFILE, ROOT_AB, ROOT_WB, ZOOM, _RAW,
                      colorize, densest_window, label_image, norm, pick)

OUT = "/home/jovyan/seg_nbhd"
SHARPEN_K = 3.0
MULT = 1.5
# None = the shape-dependent default (101x101 at 800x800, 39x151 at 312x1200)
NBHD = [None, 101, 75, 51, 35, 21]


def run_one(job):
    day, bit, rel, dir_name, burst = job
    rec = dict(day=day, bit=bit, dir_name=dir_name, burst=burst)
    try:
        profile = PROFILE[(day, bit)]
        p = FACTORY[profile]()
        H, W = p.nrow, p.ncol
        root = ROOT_WB if day == "20260331_dir1" else ROOT_AB
        raw = np.fromfile(os.path.join(root, rel, "frames1.bin"),
                          dtype=_RAW[p.read_dtype], count=p.n_ref * H * W)
        t = raw.size // (H * W)
        ref_frames = raw[:t * H * W].reshape(t, H, W).astype(p.compute_dtype)
        reference_image, _ = preprocess.reference_and_correlation_image(ref_frames)
        del ref_frames, raw
        gray = norm(reference_image)
        *_, sharp = preprocess.sharpen(reference_image, p.disk_radius, p.gauss_sigma,
                                       p.lap_alpha, SHARPEN_K)
        default_nb = tuple(int(2 * (s // 16) + 1) for s in (H, W))

        cells = {}
        for nb in NBHD:
            nbt = None if nb is None else (nb, nb)
            regions, bmap, _ = segmentation.cell_segmentation(
                sharp, p.seg_threshold, p.seg_gauss, p.seg_region_size,
                threshold_mult=MULT, neighborhood=nbt)
            kept, _bm, dropped = segmentation.drop_oversized_regions(
                regions, bmap, p.max_region_bbox_frac)
            cells[nb] = dict(lab=label_image(kept, H, W), n=len(kept), ndrop=len(dropped),
                             med_area=float(np.median([r["Area"] for r in kept])) if kept else 0.0)

        zr, zc, zs = densest_window(cells[None]["lab"], ZOOM)
        gdir = os.path.join(OUT, f"{day}__{bit}")
        os.makedirs(gdir, exist_ok=True)
        stem = f"burst{burst:04d}_{dir_name[:52]}"
        ncol = 3
        nrow = int(np.ceil(len(NBHD) / ncol))

        for tag, sl in (("full", (slice(None), slice(None))),
                        ("zoom", (slice(zr, zr + zs), slice(zc, zc + zs)))):
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.35 * nrow),
                                     constrained_layout=True, squeeze=False)
            for ax in axes.ravel():
                ax.axis("off")
            for i, nb in enumerate(NBHD):
                c = cells[nb]
                ax = axes[i // ncol][i % ncol]
                ax.imshow(colorize(c["lab"][sl], gray[sl]), interpolation="nearest",
                          aspect="equal")
                lbl = f"default {default_nb[0]}x{default_nb[1]}" if nb is None else f"{nb}x{nb}"
                ax.set_title(f"neighbourhood = {lbl}\n{c['n']} regions, median "
                             f"{c['med_area']:.0f} px", fontsize=9)
                ax.axis("off")
            fig.suptitle(f"{day} {bit} {H}x{W} burst {burst} — {tag}"
                         + (f"  [zoom {zs}x{zs} at row {zr}, col {zc}]" if tag == "zoom" else "")
                         + f"\n{dir_name}\nsharpen_k={SHARPEN_K}, multiplier={MULT} fixed  |  "
                           f"distinct colour = distinct object", fontsize=10)
            fig.savefig(os.path.join(gdir, f"{stem}_{tag}.png"), dpi=88)
            plt.close(fig)

        rec.update(ok=True, profile=profile, default_nb=list(default_nb),
                   counts={str(nb): cells[nb]["n"] for nb in NBHD},
                   areas={str(nb): round(cells[nb]["med_area"], 1) for nb in NBHD},
                   full=os.path.relpath(os.path.join(gdir, f"{stem}_full.png"), OUT),
                   zoom=os.path.relpath(os.path.join(gdir, f"{stem}_zoom.png"), OUT))
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-800:])
    return rec


def write_index(results):
    ok = [r for r in results if r.get("ok")]
    keys = [str(nb) for nb in NBHD]
    lbl = lambda k: "default" if k == "None" else f"{k}x{k}"
    h = ["<html><head><meta charset='utf-8'><title>pyali neighbourhood sweep</title><style>",
         "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;max-width:1600px}",
         "table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:12px}",
         "th{background:#f2f2f2}img{max-width:100%;border:1px solid #ddd;margin:6px 0}",
         "h2{margin-top:34px;border-bottom:2px solid #333;padding-bottom:4px}",
         ".d{background:#fffbe6}</style></head><body>",
         f"<h1>Adaptive-threshold neighbourhood sweep &mdash; sharpen_k={SHARPEN_K}, "
         f"multiplier={MULT}</h1>",
         "<p>The neighbourhood is the window over which the local mean is computed. The "
         "<b>default</b> (101&times;101 at 800&times;800, 39&times;151 at 312&times;1200) is far "
         "larger than a cell (~6 px across), so neighbouring cells share one plateau and fuse. "
         "Shrinking it should cut the neck <i>without</i> eroding blobs &mdash; so unlike the "
         "multiplier sweep, region count going <b>up</b> while median area holds roughly steady "
         "is the signature of genuine splitting.</p>",
         "<p>Watch the small windows for the degenerate failure: as the window approaches the "
         "cell size, the local mean rises to meet the object and regions break into rings or "
         "vanish.</p>",
         "<h2>Region counts</h2><table><tr><th>day</th><th>burst</th>"
         + "".join(f"<th{' class=d' if k=='None' else ''}>{lbl(k)}</th>" for k in keys) + "</tr>"]
    for r in sorted(ok, key=lambda r: (r["day"], r["burst"])):
        h.append(f"<tr><td>{r['day']} {r['bit']}</td><td>{r['burst']}</td>"
                 + "".join(f"<td{' class=d' if k=='None' else ''}>{r['counts'][k]}</td>"
                           for k in keys) + "</tr>")
    h.append("</table><h2>Median region area (px)</h2><table><tr><th>day</th><th>burst</th>"
             + "".join(f"<th{' class=d' if k=='None' else ''}>{lbl(k)}</th>" for k in keys) + "</tr>")
    for r in sorted(ok, key=lambda r: (r["day"], r["burst"])):
        h.append(f"<tr><td>{r['day']} {r['bit']}</td><td>{r['burst']}</td>"
                 + "".join(f"<td{' class=d' if k=='None' else ''}>{r['areas'][k]}</td>"
                           for k in keys) + "</tr>")
    h.append("</table>")
    for r in sorted(ok, key=lambda r: (r["day"], r["burst"])):
        h.append(f"<h2>{r['day']} {r['bit']} — burst {r['burst']}</h2>")
        h.append(f"<p style='font-size:12px;color:#555'>{r['dir_name']}</p>")
        h.append(f"<b>zoom</b><br><img src='{r['zoom']}'><br><b>full frame</b><br>"
                 f"<img src='{r['full']}'>")
    h.append("</body></html>")
    idx = os.path.join(OUT, "index.html")
    open(idx, "w").write("\n".join(h))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = pick()
    print(f"[nbhd] {len(jobs)} FOVs x {len(NBHD)} neighbourhoods "
          f"(sharpen_k={SHARPEN_K}, mult={MULT})", flush=True)
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r); done += 1
            print(f"[nbhd] {done}/{len(jobs)} {'ok ' if r.get('ok') else 'FAIL'} "
                  f"{r['day']} burst {r['burst']}"
                  + ("" if r.get("ok") else "\n" + r.get("error", "")), flush=True)
    idx = write_index(results)
    json.dump(results, open(os.path.join(OUT, "seg_nbhd.json"), "w"), indent=2)
    bad = [r for r in results if not r.get("ok")]
    print(f"\n[nbhd] done. {len(results)-len(bad)} ok, {len(bad)} errors", flush=True)
    print(f"[nbhd] OPEN THIS -> {idx}", flush=True)


if __name__ == "__main__":
    main()
