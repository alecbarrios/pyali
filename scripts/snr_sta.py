#!/usr/bin/env python3
"""Per-cell and per-FOV spike-triggered averages -- README section 10f, stage 1b of the SNR pass.

Detects spikes exactly as :func:`pyali.metrics.per_cell_snr` does, then averages the waveform
around each spike. Two files land in each FOV directory beside ``snr_metrics.csv``:

* ``sta.csv``        -- the FOV summary curve: median / q25 / q75 / mean / sd **across cells**
* ``sta_cells.mat``  -- every cell's own STA, gzipped (for later per-cell genotype assignment)

    python scripts/snr_sta.py --workers 8
    python scripts/snr_sta.py --days 20260612 --limit 3 --dry-run

**Snippets come from the raw trace, detection from the filtered one.** The 20 Hz high-pass used for
detection is zero-phase, so its impulse response is symmetric to machine precision with negative
lobes at +/-10 ms. Averaging the *filtered* trace therefore stamps a spurious symmetric pre-spike
dip (-0.45 sigma) onto the waveform and inflates the undershoot (-0.71 sigma). Averaging the raw
trace instead shows a physiological rising ramp into the peak and only a -0.17 sigma undershoot.

Amplitudes are in units of that cell's own ``noise_sigma``. Absolute trace units are not comparable
across days -- bit depth, frame size and footprint weighting all differ -- whereas sigma-normalised
amplitude is (measured peak 4.75 sigma June vs 4.55 sigma July on the filtered trace).

Resumable and re-runnable, same as ``snr_corpus.py``; safe to run against a live extraction.
"""
import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.signal import find_peaks

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from pyali.io import load_v73, save_mat_v73                              # noqa: E402
from pyali.metrics import highpass, robust_sigma                         # noqa: E402
from snr_corpus import META, REQUIRED, RESULT, _write, list_fovs         # noqa: E402
from snr_corpus import READ_ROOT_DEFAULT, S3_DEFAULT                     # noqa: E402

STA = "sta.csv"
STA_CELLS = "sta_cells.mat"
HALF = 20                 # +/-20 samples = +/-25 ms at 800 Hz
MIN_SPIKES = 3            # a cell needs this many clean snippets to contribute an STA
SHOULDER = 6              # samples of the leading shoulder used as the baseline
COLUMNS = ("t_ms", "median", "q25", "q75", "mean", "sd",
           "n_cells", "n_spikes_used", "n_spikes_total", "frac_excluded")


def _cubic_shift(seg4, d, n):
    """Resample ``n`` points at fractional offset ``d`` using a Catmull-Rom cubic.

    ``seg4`` carries 2 guard samples on each side. **Linear** interpolation is not usable here:
    the AP is only ~2 samples wide at 800 Hz, and linear resampling broadened the averaged FWHM
    from 2.50 to 3.75 ms while *lowering* the peak (5.77 -> 5.56 sigma). A cubic keeps FWHM at
    2.50 ms and raises the peak to 5.88 sigma. A `scipy` spline object per snippet would be exact
    but costs ~40 us x ~34M spikes; this closed form is the same cubic family, vectorised.
    """
    i0 = int(np.floor(d))
    t = d - i0
    idx = np.arange(n) + 2 + i0
    pm1, p0, p1, p2 = seg4[idx - 1], seg4[idx], seg4[idx + 1], seg4[idx + 2]
    return (p0 + 0.5 * t * (p1 - pm1)
            + 0.5 * t * t * (2.0 * pm1 - 5.0 * p0 + 4.0 * p1 - p2)
            + 0.5 * t * t * t * (-pm1 + 3.0 * p0 - 3.0 * p1 + p2))


def cell_sta(x, xhp, sig, pk, half=HALF, align=True):
    """Mean waveform around this cell's spikes, in sigma units, or ``None``.

    Snippets are cut from ``x`` (raw) while ``pk`` came from ``xhp`` (filtered).
    """
    T = x.size
    pad = 2 if align else 0                # guard samples the cubic needs
    usable = []
    for j, p in enumerate(pk):
        if p < half + pad + 1 or p >= T - half - pad - 1:
            continue
        # Neighbour exclusion: another detected spike inside the window would add a second bump.
        # Costs ~9% of spikes at +/-25 ms; a wider window would lose most of them, since the
        # median ISI is only 72-85 ms.
        if (j and p - pk[j - 1] <= half) or (j + 1 < pk.size and pk[j + 1] - p <= half):
            continue
        usable.append(p)
    if len(usable) < MIN_SPIKES:
        return None, 0
    n = 2 * half + 1
    acc = np.zeros(n)
    for p in usable:
        if align:
            # Parabolic vertex of the filtered peak gives the sub-sample offset. The AP is FWHM
            # ~2.5 ms = 2 samples here, so nearest-sample alignment smears the average by +/-0.6 ms.
            ym1, y0, yp1 = xhp[p - 1], xhp[p], xhp[p + 1]
            den = ym1 - 2.0 * y0 + yp1
            d = 0.5 * (ym1 - yp1) / den if den != 0 else 0.0
            d = float(np.clip(d, -0.5, 0.5))
            w = _cubic_shift(x[p - half - 2:p + half + 3].astype(float), d, n)
        else:
            w = x[p - half:p + half + 1].astype(float)
        # Baseline from the LEADING shoulder only. Using both shoulders would force the post-spike
        # tail to zero and hide any genuine afterpotential.
        acc += w - np.median(w[:SHOULDER])
    return acc / len(usable) / sig, len(usable)


def _fmt(v):
    v = float(v)
    return "nan" if not math.isfinite(v) else repr(v)


def build_csv(t_ms, cells, n_used, n_total):
    """Across-cell summary as CSV text; a header-only file when no cell qualified."""
    head = ",".join(COLUMNS)
    if not len(cells):
        return head + "\n"
    A = np.asarray(cells, float)                       # (n_cells, 2*HALF+1)
    med = np.median(A, axis=0)
    q25, q75 = np.percentile(A, 25, axis=0), np.percentile(A, 75, axis=0)
    mean, sd = A.mean(axis=0), A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros(A.shape[1])
    frac = 1.0 - (n_used / n_total) if n_total else float("nan")
    rows = [head]
    for i, t in enumerate(t_ms):
        rows.append(",".join((_fmt(t), _fmt(med[i]), _fmt(q25[i]), _fmt(q75[i]),
                              _fmt(mean[i]), _fmt(sd[i]), str(A.shape[0]),
                              str(int(n_used)), str(int(n_total)), _fmt(frac))))
    return "\n".join(rows) + "\n"


def _run_one(job):
    day, dir_name, read_root, s3_prefix, mode, align, per_cell = job
    rec = dict(day=day, dir_name=dir_name)
    t0 = time.perf_counter()
    try:
        fov = os.path.join(read_root, day, dir_name)
        with open(os.path.join(fov, META)) as f:
            fps = float(json.load(f)["analysis"]["fps"])
        ct = np.asarray(load_v73(os.path.join(fov, RESULT), "cell_traces"))
        if ct.ndim != 2:
            ct = ct.reshape(0, 0)
        half = HALF
        t_ms = np.arange(-half, half + 1) * (1000.0 / fps)
        curves, idx, used, n_used, n_total = [], [], [], 0, 0
        for i in range(ct.shape[0]):
            x = np.asarray(ct[i], float)
            xhp = highpass(x, 20.0, fps)
            sig = robust_sigma(xhp)
            if sig <= 0:
                continue
            pk, _ = find_peaks(xhp, height=3.0 * sig, distance=max(1, int(round(0.01 * fps))))
            n_total += pk.size
            c, nu = cell_sta(x, xhp, sig, pk, half, align)
            if c is not None:
                curves.append(c.astype(np.float32)); idx.append(i); used.append(nu); n_used += nu
        rec.update(n_cells_total=int(ct.shape[0]), n_cells_sta=len(curves),
                   n_spikes_total=int(n_total), n_spikes_used=int(n_used), fps=fps)
        _write(build_csv(t_ms, curves, n_used, n_total), fov,
               f"{s3_prefix.rstrip('/')}/{day}/{dir_name}/{STA}", mode, name=STA)
        if per_cell and curves:
            save_mat_v73(os.path.join(fov, STA_CELLS),
                         sta_cells=np.asarray(curves, np.float32),
                         cell_index=np.asarray(idx, np.int32),
                         n_spikes_used=np.asarray(used, np.int32),
                         t_ms=t_ms.astype(np.float64))
            rec["per_cell_bytes"] = os.path.getsize(os.path.join(fov, STA_CELLS))
        rec.update(ok=True, total_s=round(time.perf_counter() - t0, 2))
    except Exception:
        rec.update(ok=False, error=traceback.format_exc()[-1200:],
                   total_s=round(time.perf_counter() - t0, 2))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s3-prefix", default=S3_DEFAULT)
    ap.add_argument("--read-root", default=READ_ROOT_DEFAULT)
    ap.add_argument("--manifest", default="/home/jovyan/snr_sta_manifest.jsonl")
    ap.add_argument("--upload", choices=["s3fs", "awscli"], default="s3fs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--days", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-align", action="store_true", help="disable sub-sample peak alignment")
    ap.add_argument("--no-per-cell", action="store_true", help=f"skip writing {STA_CELLS}")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.isdir(a.read_root):
        ap.error(f"{a.read_root} is not a directory -- pass --read-root")
    fovs = list_fovs(a.s3_prefix)
    keys = sorted(k for k, v in fovs.items() if set(REQUIRED).issubset(v))
    if a.days:
        keys = [k for k in keys if k[0] in a.days]
    done = [k for k in keys if STA in fovs[k]]
    if not a.force:
        keys = [k for k in keys if STA not in fovs[k]]
    if a.limit:
        keys = keys[:a.limit]
    print(f"[sta] {len(keys)} FOVs to process ({len(done)} already have {STA}"
          f"{', recomputing' if a.force else ', skipped'})", flush=True)
    print(f"[sta] window +/-{HALF} samples, min_spikes {MIN_SPIKES}, align="
          f"{not a.no_align}, per_cell={not a.no_per_cell}, workers {a.workers}", flush=True)
    if a.dry_run:
        print("[sta] dry run, stopping here", flush=True)
        return 0
    if not keys:
        return 0

    jobs = [(d, n, a.read_root, a.s3_prefix, a.upload, not a.no_align, not a.no_per_cell)
            for d, n in keys]
    t0 = time.perf_counter()
    ok = fail = cells = 0
    mf = open(a.manifest, "a")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(_run_one, j) for j in jobs]), start=1):
            rec = fut.result()
            rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            mf.write(json.dumps(rec) + "\n"); mf.flush()
            if rec.get("ok"):
                ok += 1; cells += rec.get("n_cells_sta", 0)
            else:
                fail += 1
                print(f"[sta] FAIL {rec['day']}/{rec['dir_name']}\n{rec.get('error','')}",
                      flush=True)
            if i % 50 == 0 or i == len(jobs):
                el = time.perf_counter() - t0
                print(f"[sta] {i}/{len(jobs)}  ok={ok} fail={fail}  {cells} cell STAs  "
                      f"{i/el*3600:.0f} FOV/h  eta {(len(jobs)-i)/max(i/el,1e-9)/60:.1f} min",
                      flush=True)
    mf.close()
    print(f"\n[sta] done in {(time.perf_counter()-t0)/60:.1f} min: {ok} ok, {fail} failed, "
          f"{cells} cell STAs", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
