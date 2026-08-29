#!/usr/bin/env python3
"""Segmentation QC: render what the pipeline actually segments, per acquisition day.

Segmentation depends only on the first ``n_ref`` frames (``pipeline.py`` builds ``sharpened``
from ``movie[:n_ref]`` before ``adaptive_background`` runs), so this reproduces the exact regions
the full pipeline would find while reading ~1% of each movie.

Per FOV it writes a 3-panel PNG — reference image, sharpened image, and the reference with
region outlines: **green = kept, red = dropped by the 1% bounding-box guard**. Per day it writes
a contact sheet of the overlays so a whole day can be judged at a glance, plus an index.html
linking everything.
"""
import argparse
import csv
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/jovyan/workbench/pyali")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyali import extract, preprocess, segmentation
from pyali.params import Params

KEEP = "/home/jovyan/spatial-technology-platform/AB/pyali_6b59e79_outputs/keep.csv"
ROOT_AB, ROOT_WB = "/mnt/s3ab/AB", "/mnt/s3wb/data"
OUT = "/home/jovyan/seg_qc"
_RAW = {"uint8": "<u1", "uint16": "<u2"}

GROUPS = {
    ("20260331_dir1", "16-bit"): "6GP002",
    ("20260331_dir2", "16-bit"): "6GP002",
    ("20260401", "16-bit"): "6GP002",
    ("20260401", "8-bit"): "6GP002-8bit",
    ("20260611", "8-bit"): "443screen1",
    ("20260612", "8-bit"): "443screen1",
    ("20260715", "8-bit"): "443screen2",
    ("20260716", "8-bit"): "443screen2",
    ("20260717", "8-bit"): "443screen2",
    ("20260718", "8-bit"): "443screen2",
}
FACTORY = {"6GP002": Params.profile_6GP002, "6GP002-8bit": Params.profile_6GP002_8bit,
           "443screen1": Params.profile_443screen1, "443screen2": Params.profile_443screen2}


def group_key(day, bit):
    return f"{day}__{bit}"


def show(ax, img, title):
    lo, hi = np.percentile(img, (1.0, 99.5))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def overlay_rgb(ref, kept_mask, dropped_mask):
    """Reference image as RGB with kept regions tinted green and dropped ones red."""
    lo, hi = np.percentile(ref, (1.0, 99.5))
    g = np.clip((ref - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = np.dstack([g, g, g])
    rgb[kept_mask] = np.array([0.15, 1.0, 0.15])
    rgb[dropped_mask] = np.array([1.0, 0.15, 0.15])
    return rgb


def run_one(job):
    day, bit, rel, dir_name, burst, rank, profile = job
    rec = dict(day=day, bit=bit, dir_name=dir_name, burst=burst, rank=rank, profile=profile)
    try:
        p = FACTORY[profile]()
        H, W = p.nrow, p.ncol
        root = ROOT_WB if day == "20260331_dir1" else ROOT_AB
        path = os.path.join(root, rel, "frames1.bin")

        raw = np.fromfile(path, dtype=_RAW[p.read_dtype], count=p.n_ref * H * W)
        t = raw.size // (H * W)
        ref_frames = raw[:t * H * W].reshape(t, H, W).astype(p.compute_dtype)
        reference_image, _ = preprocess.reference_and_correlation_image(ref_frames)
        del ref_frames
        *_, sharpened = preprocess.sharpen(reference_image, p.disk_radius, p.gauss_sigma,
                                           p.lap_alpha, p.sharpen_k)
        regions, bmap, _sf = segmentation.cell_segmentation(
            sharpened, p.seg_threshold, p.seg_gauss, p.seg_region_size)
        kept, bmap_kept, dropped = segmentation.drop_oversized_regions(
            regions, bmap, p.max_region_bbox_frac)

        dropped_mask = bmap & ~bmap_kept          # exactly the pixels the guard removed
        rgb = overlay_rgb(reference_image, bmap_kept, dropped_mask)

        gdir = os.path.join(OUT, group_key(day, bit))
        os.makedirs(gdir, exist_ok=True)
        stem = f"burst{burst:04d}_{rank}_{dir_name[:60]}"
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
        show(axes[0], reference_image, "reference image")
        show(axes[1], sharpened, "sharpened (segmentation input)")
        axes[2].imshow(rgb, interpolation="nearest", aspect="equal")
        axes[2].set_title(f"green = kept ({len(kept)})   red = dropped by 1% guard ({len(dropped)})",
                          fontsize=8)
        axes[2].set_xticks([]); axes[2].set_yticks([])
        fig.suptitle(f"{day}  {bit}  {H}x{W}  |  profile {profile}  |  burst {burst} ({rank})\n"
                     f"{dir_name}", fontsize=9)
        png = os.path.join(gdir, stem + ".png")
        fig.savefig(png, dpi=95)
        plt.close(fig)

        frame_px = H * W
        bb = lambda r: float(r["BoundingBox"][2]) * float(r["BoundingBox"][3])
        rec.update(ok=True, png=os.path.relpath(png, OUT),
                   n_before=len(regions), n_kept=len(kept), n_dropped=len(dropped),
                   dropped_bbox_pct=sorted(round(100 * bb(r) / frame_px, 2) for r in dropped),
                   max_kept_bbox_pct=round(100 * max([bb(r) for r in kept], default=0) / frame_px, 3),
                   median_area=float(np.median([r["Area"] for r in kept])) if kept else None,
                   rgb_small=None)
        # keep a downsampled overlay for the contact sheet
        s = max(1, max(H, W) // 320)
        np.save(os.path.join(gdir, "." + stem + ".npy"), (rgb[::s, ::s] * 255).astype(np.uint8))
        rec["thumb"] = os.path.join(gdir, "." + stem + ".npy")
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-800:])
    return rec


def pick(per_group):
    rows = list(csv.DictReader(open(KEEP)))
    jobs = []
    for (day, bit), profile in GROUPS.items():
        sel = [r for r in rows if r["day"] == day and r["bit"] == bit]
        byb = []
        for r in sel:
            m = re.search(r"burst(\d+)", r["dir_name"])
            byb.append((int(m.group(1)) if m else 0, r))
        byb.sort(key=lambda t: t[0])
        n = len(byb)
        if not n:
            continue
        # spread across the burst sequence, and always include the first/last (worst artifacts)
        idx = sorted({0, 1, n - 2, n - 1} |
                     {round(i * (n - 1) / (per_group - 1)) for i in range(per_group)})
        for i in idx:
            if 0 <= i < n:
                b, r = byb[i]
                rank = ("first" if i == 0 else "last" if i == n - 1 else
                        "second" if i == 1 else "penultimate" if i == n - 2 else "interior")
                jobs.append((day, bit, r["fov_path"], r["dir_name"], b, rank, profile))
    seen, uniq = set(), []
    for j in jobs:
        if (j[0], j[2]) not in seen:
            seen.add((j[0], j[2])); uniq.append(j)
    return uniq


def contact_sheet(gkey, recs):
    recs = [r for r in recs if r.get("ok") and r.get("thumb") and os.path.exists(r["thumb"])]
    if not recs:
        return None
    recs.sort(key=lambda r: r["burst"])
    ncol = 6
    nrow = int(np.ceil(len(recs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.15 * nrow),
                             constrained_layout=True, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, r in enumerate(recs):
        ax = axes[k // ncol][k % ncol]
        ax.imshow(np.load(r["thumb"]), interpolation="nearest", aspect="equal")
        ax.set_title(f"burst {r['burst']} ({r['rank']})\nkept {r['n_kept']}  dropped {r['n_dropped']}",
                     fontsize=7.5)
        ax.axis("off")
    fig.suptitle(f"{gkey}   —   green = kept regions, red = dropped by the 1% bbox guard",
                 fontsize=12)
    out = os.path.join(OUT, gkey, "CONTACT_SHEET.png")
    fig.savefig(out, dpi=85)
    plt.close(fig)
    for r in recs:
        try:
            os.remove(r["thumb"])
        except OSError:
            pass
    return out


def write_index(by_group):
    rows = []
    for gkey in sorted(by_group):
        recs = [r for r in by_group[gkey] if r.get("ok")]
        if not recs:
            continue
        kept = [r["n_kept"] for r in recs]
        drp = [r["n_dropped"] for r in recs]
        med = [r["median_area"] for r in recs if r["median_area"]]
        rows.append(dict(
            group=gkey, n=len(recs),
            kept_med=int(np.median(kept)), kept_min=min(kept), kept_max=max(kept),
            zero=sum(1 for k in kept if k == 0),
            fovs_with_drop=sum(1 for d in drp if d), drops=int(sum(drp)),
            med_area=round(float(np.median(med)), 1) if med else None,
            sheet=f"{gkey}/CONTACT_SHEET.png",
            pngs=sorted(r["png"] for r in recs)))
    html = ["<html><head><meta charset='utf-8'><title>pyali segmentation QC</title>",
            "<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;max-width:1500px}",
            "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:5px 9px;font-size:13px}",
            "th{background:#f2f2f2}img{max-width:100%;border:1px solid #ddd;margin:6px 0}",
            "h2{margin-top:34px;border-bottom:2px solid #333;padding-bottom:4px}",
            "details{margin:6px 0}summary{cursor:pointer;color:#0645ad}</style></head><body>",
            "<h1>pyali segmentation QC</h1>",
            "<p>Green = regions kept. Red = regions dropped by the 1% bounding-box guard. "
            "Segmentation uses only the first 600 frames, so these are exactly the regions the "
            "full pipeline would find.</p>",
            "<p><b>What to look for:</b> are the green blobs cells? Is anything obviously a cell "
            "left unsegmented? Is red only ever the dark inter-well band?</p>",
            "<table><tr><th>day / bit</th><th>FOVs</th><th>kept regions (med, min–max)</th>"
            "<th>FOVs with 0 kept</th><th>FOVs with a drop</th><th>total dropped</th>"
            "<th>median region area px</th></tr>"]
    for r in rows:
        html.append(f"<tr><td><a href='#{r['group']}'>{r['group']}</a></td><td>{r['n']}</td>"
                    f"<td>{r['kept_med']} ({r['kept_min']}–{r['kept_max']})</td>"
                    f"<td>{r['zero']}</td><td>{r['fovs_with_drop']}</td><td>{r['drops']}</td>"
                    f"<td>{r['med_area']}</td></tr>")
    html.append("</table>")
    for r in rows:
        html.append(f"<h2 id='{r['group']}'>{r['group']}</h2>")
        html.append(f"<img src='{r['sheet']}'>")
        html.append("<details><summary>full-resolution per-FOV panels "
                    f"({len(r['pngs'])})</summary><ul>")
        for p in r["pngs"]:
            html.append(f"<li><a href='{p}'>{os.path.basename(p)}</a></li>")
        html.append("</ul></details>")
    html.append("</body></html>")
    idx = os.path.join(OUT, "index.html")
    open(idx, "w").write("\n".join(html))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=24)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = pick(a.per_group)
    print(f"[segqc] {len(jobs)} FOVs across {len(GROUPS)} day/shape groups -> {OUT}", flush=True)

    by_group, done = {}, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = dict(ok=False, error=repr(e), day="?", bit="?")
            by_group.setdefault(group_key(r["day"], r["bit"]), []).append(r)
            done += 1
            if done % 25 == 0 or done == len(jobs):
                bad = sum(1 for g in by_group.values() for x in g if not x.get("ok"))
                print(f"[segqc] {done}/{len(jobs)}  errors={bad}", flush=True)

    for gkey in sorted(by_group):
        contact_sheet(gkey, by_group[gkey])
    idx = write_index(by_group)
    json.dump({k: [{kk: vv for kk, vv in r.items() if kk != "thumb"} for r in v]
               for k, v in by_group.items()},
              open(os.path.join(OUT, "seg_qc.json"), "w"), indent=2)
    bad = [x for g in by_group.values() for x in g if not x.get("ok")]
    print(f"\n[segqc] done. {len(jobs)-len(bad)} ok, {len(bad)} errors", flush=True)
    for b in bad[:3]:
        print("  ERROR", b.get("error", "")[-300:], flush=True)
    print(f"[segqc] OPEN THIS -> {idx}", flush=True)


if __name__ == "__main__":
    main()
