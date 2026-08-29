#!/usr/bin/env python3
"""Segmentation parameter grid: threshold multiplier x sharpen_k, same FOV in every panel.

Targets the merging question: does a higher local-mean multiplier (and/or a stronger unsharp
mask) split touching cells that the current 1.5 / 2.25 defaults fuse into one object?

Regions are drawn in *distinct random colours*, so a clump merged into one object shows as one
colour block and a correctly split pair shows as two. Two figures per FOV: the whole frame, and
a zoom on the densest patch where merging is actually judgeable.

Segmentation depends only on the first 600 frames, so each FOV is read once and the reference
image computed once; only sharpen (3x) and segmentation (12x) are repeated.
"""
import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/jovyan/workbench/pyali")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyali import preprocess, segmentation
from pyali.params import Params

KEEP = "/home/jovyan/spatial-technology-platform/AB/pyali_6b59e79_outputs/keep.csv"
SEGQC = "/home/jovyan/seg_qc/seg_qc.json"
ROOT_AB, ROOT_WB = "/mnt/s3ab/AB", "/mnt/s3wb/data"
OUT = "/home/jovyan/seg_grid"
_RAW = {"uint8": "<u1", "uint16": "<u2"}

MULTS = [1.5, 1.75, 2.0, 2.25]           # 1.5 = current default
SHARPKS = [2.25, 3.0, 4.0]               # 2.25 = current default
ZOOM = 220

PROFILE = {("20260331_dir1", "16-bit"): "6GP002", ("20260331_dir2", "16-bit"): "6GP002",
           ("20260401", "16-bit"): "6GP002", ("20260401", "8-bit"): "6GP002-8bit",
           ("20260611", "8-bit"): "443screen1", ("20260612", "8-bit"): "443screen1",
           ("20260715", "8-bit"): "443screen2", ("20260716", "8-bit"): "443screen2",
           ("20260717", "8-bit"): "443screen2", ("20260718", "8-bit"): "443screen2"}
FACTORY = {"6GP002": Params.profile_6GP002, "6GP002-8bit": Params.profile_6GP002_8bit,
           "443screen1": Params.profile_443screen1, "443screen2": Params.profile_443screen2}

# the FOVs you named, by (day, bit, burst)
REQUIRED = [("20260331_dir1", "16-bit", 342), ("20260331_dir2", "16-bit", 86),
            ("20260401", "16-bit", 180), ("20260612", "8-bit", 42),
            ("20260715", "8-bit", 70), ("20260716", "8-bit", 70),
            ("20260717", "8-bit", 11), ("20260718", "8-bit", 48)]
# fill out to ~25, weighted to June / July / 16-bit April, plus the over-segmenting April 8-bit
EXTRA_QUOTA = [(("20260401", "16-bit"), 2), (("20260401", "8-bit"), 2),
               (("20260611", "8-bit"), 3), (("20260612", "8-bit"), 2),
               (("20260715", "8-bit"), 2), (("20260716", "8-bit"), 2),
               (("20260717", "8-bit"), 2), (("20260718", "8-bit"), 2)]


def label_image(regions, H, W):
    lab = np.zeros((H, W), np.int32)
    for i, r in enumerate(regions, start=1):
        px = r["PixelList"].astype(int)
        lab[px[:, 1] - 1, px[:, 0] - 1] = i
    return lab


def norm(img):
    lo, hi = np.percentile(img, (1.0, 99.5))
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)


def colorize(lab, gray, alpha=0.55, seed=0):
    """Grayscale image with each region a distinct random colour."""
    rgb = np.dstack([gray] * 3)
    n = int(lab.max())
    if n:
        rng = np.random.default_rng(seed)
        colors = rng.uniform(0.35, 1.0, size=(n + 1, 3))
        colors[0] = 0
        m = lab > 0
        rgb[m] = (1 - alpha) * rgb[m] + alpha * colors[lab[m]]
    return rgb


def densest_window(lab, size):
    """Top-left corner of the size x size window containing the most labelled pixels."""
    H, W = lab.shape
    size = min(size, H, W)
    m = (lab > 0).astype(np.float32)
    ii = m.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    best, br, bc = -1, 0, 0
    step = max(8, size // 8)
    for r in range(0, H - size + 1, step):
        for c in range(0, W - size + 1, step):
            s = ii[r + size, c + size] - ii[r, c + size] - ii[r + size, c] + ii[r, c]
            if s > best:
                best, br, bc = s, r, c
    return br, bc, size


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

        cells = {}
        for k in SHARPKS:
            *_, sharp = preprocess.sharpen(reference_image, p.disk_radius, p.gauss_sigma,
                                           p.lap_alpha, k)
            for m in MULTS:
                regions, bmap, _ = segmentation.cell_segmentation(
                    sharp, p.seg_threshold, p.seg_gauss, p.seg_region_size, threshold_mult=m)
                kept, _bm, dropped = segmentation.drop_oversized_regions(
                    regions, bmap, p.max_region_bbox_frac)
                cells[(k, m)] = dict(lab=label_image(kept, H, W), n=len(kept),
                                     ndrop=len(dropped),
                                     med_area=float(np.median([r["Area"] for r in kept]))
                                     if kept else 0.0)

        base = cells[(2.25, 1.5)]
        zr, zc, zs = densest_window(base["lab"], ZOOM)
        gdir = os.path.join(OUT, f"{day}__{bit}")
        os.makedirs(gdir, exist_ok=True)
        stem = f"burst{burst:04d}_{dir_name[:52]}"

        for tag, sl in (("full", (slice(None), slice(None))),
                        ("zoom", (slice(zr, zr + zs), slice(zc, zc + zs)))):
            fig, axes = plt.subplots(len(SHARPKS), len(MULTS),
                                     figsize=(3.6 * len(MULTS), 3.85 * len(SHARPKS)),
                                     constrained_layout=True, squeeze=False)
            for i, k in enumerate(SHARPKS):
                for j, m in enumerate(MULTS):
                    c = cells[(k, m)]
                    ax = axes[i][j]
                    ax.imshow(colorize(c["lab"][sl], gray[sl]), interpolation="nearest",
                              aspect="equal")
                    dflt = " (current)" if (k == 2.25 and m == 1.5) else ""
                    ax.set_title(f"mult={m}  sharpen_k={k}{dflt}\n"
                                 f"{c['n']} regions, median {c['med_area']:.0f} px", fontsize=8.5)
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{day} {bit} {H}x{W} burst {burst} — {tag}"
                         + (f"  [zoom {zs}x{zs} at row {zr}, col {zc}]" if tag == "zoom" else "")
                         + f"\n{dir_name}\ncolumns: threshold multiplier (higher = tighter, splits "
                           f"clumps)   rows: sharpen_k   distinct colour = distinct object",
                         fontsize=10)
            png = os.path.join(gdir, f"{stem}_{tag}.png")
            fig.savefig(png, dpi=88)
            plt.close(fig)

        rec.update(ok=True, profile=profile,
                   counts={f"k{k}_m{m}": cells[(k, m)]["n"] for k in SHARPKS for m in MULTS},
                   areas={f"k{k}_m{m}": round(cells[(k, m)]["med_area"], 1)
                          for k in SHARPKS for m in MULTS},
                   full=os.path.relpath(os.path.join(gdir, f"{stem}_full.png"), OUT),
                   zoom=os.path.relpath(os.path.join(gdir, f"{stem}_zoom.png"), OUT))
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-800:])
    return rec


def pick():
    keep = list(csv.DictReader(open(KEEP)))
    qc = json.load(open(SEGQC)) if os.path.exists(SEGQC) else {}
    # index the FOVs already rendered in the QC sweep, so a named burst resolves to the exact
    # field that was inspected
    seen = {}
    for g, recs in qc.items():
        for r in recs:
            if r.get("ok"):
                seen[(r["day"], r["bit"], r["burst"])] = r["dir_name"]
    bypath = {(r["day"], r["dir_name"]): r["fov_path"] for r in keep}
    import re
    burst_of = {}
    for r in keep:
        m = re.search(r"burst(\d+)", r["dir_name"])
        burst_of.setdefault((r["day"], r["bit"]), []).append(
            (int(m.group(1)) if m else 0, r["dir_name"], r["fov_path"]))

    jobs, used = [], set()

    def add(day, bit, burst, dir_name, path):
        if (day, path) in used:
            return False
        used.add((day, path))
        jobs.append((day, bit, path, dir_name, burst))
        return True

    for day, bit, burst in REQUIRED:
        dn = seen.get((day, bit, burst))
        if dn and (day, dn) in bypath:
            add(day, bit, burst, dn, bypath[(day, dn)])
            continue
        cands = [c for c in burst_of.get((day, bit), []) if c[0] == burst]
        if cands:
            cands.sort(key=lambda c: c[2])
            add(day, bit, burst, cands[0][1], cands[0][2])
        else:
            print(f"[grid] WARNING: no FOV for {day} {bit} burst {burst}", flush=True)

    for (day, bit), n in EXTRA_QUOTA:
        lst = sorted(burst_of.get((day, bit), []))
        if not lst:
            continue
        for i in range(n):                       # interior positions, away from the well edges
            idx = int(round((i + 1) * (len(lst) - 1) / (n + 1)))
            b, dn, pth = lst[idx]
            for off in range(0, 40):             # step along if that one is already taken
                j = min(len(lst) - 1, idx + off)
                b, dn, pth = lst[j]
                if add(day, bit, b, dn, pth):
                    break
    return jobs


def write_index(results):
    ok = [r for r in results if r.get("ok")]
    keys = [f"k{k}_m{m}" for k in SHARPKS for m in MULTS]
    h = ["<html><head><meta charset='utf-8'><title>pyali segmentation grid</title><style>",
         "body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;max-width:1600px}",
         "table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:12px}",
         "th{background:#f2f2f2}img{max-width:100%;border:1px solid #ddd;margin:6px 0}",
         "h2{margin-top:34px;border-bottom:2px solid #333;padding-bottom:4px}",
         ".d{background:#fffbe6}</style></head><body>",
         "<h1>Segmentation grid — threshold multiplier &times; sharpen_k</h1>",
         "<p><b>The merging question.</b> Each region gets its own random colour: a clump merged "
         "into one object is a single colour block, a correctly split pair is two colours. "
         "<b>mult=1.5, sharpen_k=2.25 is the current default</b> (highlighted). Higher multiplier "
         "= higher threshold = tighter blobs that pull apart at the necks.</p>",
         "<p>Look at the <b>zoom</b> figures to judge merging; the full-frame ones are for "
         "overall sanity. Watch for the cost of raising the multiplier: blobs shrink, so "
         "footprints get smaller and faint cells drop out entirely.</p>",
         "<h2>Region counts per config</h2><table><tr><th>day</th><th>burst</th>"
         + "".join(f"<th{' class=d' if k=='k2.25_m1.5' else ''}>{k.replace('_',' ')}</th>" for k in keys)
         + "</tr>"]
    for r in sorted(ok, key=lambda r: (r["day"], r["burst"])):
        h.append(f"<tr><td>{r['day']} {r['bit']}</td><td>{r['burst']}</td>"
                 + "".join(f"<td{' class=d' if k=='k2.25_m1.5' else ''}>{r['counts'][k]}</td>"
                           for k in keys) + "</tr>")
    h.append("</table><h2>Median region area (px)</h2><table><tr><th>day</th><th>burst</th>"
             + "".join(f"<th{' class=d' if k=='k2.25_m1.5' else ''}>{k.replace('_',' ')}</th>" for k in keys)
             + "</tr>")
    for r in sorted(ok, key=lambda r: (r["day"], r["burst"])):
        h.append(f"<tr><td>{r['day']} {r['bit']}</td><td>{r['burst']}</td>"
                 + "".join(f"<td{' class=d' if k=='k2.25_m1.5' else ''}>{r['areas'][k]}</td>"
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
    print(f"[grid] {len(jobs)} FOVs x {len(MULTS)*len(SHARPKS)} configs", flush=True)
    for j in jobs:
        print(f"   {j[0]:<16} {j[1]:<7} burst {j[4]:<5} {j[3][:56]}", flush=True)

    results, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"[grid] {done}/{len(jobs)} {'ok ' if r.get('ok') else 'FAIL'} "
                  f"{r['day']} burst {r['burst']}"
                  + ("" if r.get("ok") else "\n" + r.get("error", "")), flush=True)
    idx = write_index(results)
    json.dump(results, open(os.path.join(OUT, "seg_grid.json"), "w"), indent=2)
    bad = [r for r in results if not r.get("ok")]
    print(f"\n[grid] done. {len(results)-len(bad)} ok, {len(bad)} errors", flush=True)
    print(f"[grid] OPEN THIS -> {idx}", flush=True)


if __name__ == "__main__":
    main()
