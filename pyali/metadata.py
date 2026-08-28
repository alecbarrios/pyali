"""Per-FOV acquisition metadata, parsed from the directory name.

The ten acquisition days use slightly different naming conventions, but the *fields* are the
same throughout, so one tokenizer with per-field recognizers covers all of them — a regex per
day is not needed. Representative names::

    101452_P01_12w_A1-A4-A2-C4-B4-C3-C2_JF608_6GP002_DIV36_burst1   20260331_dir1
    123951_P01_12w_B4_6GP002_DIV36_burst1                           20260331_dir2
    094426_P02_6w_A1_JF608_6GP002_DIV37_burst1                      20260401
    162232_P02_6w_B1_JF608_6GP002_DIV37_8bit_burst27                20260401 8-bit
    122253_P06_0xSM_C2_DIV55_443GP_burst1                           20260611/12
    114944P-1_W-A1_443_443screen2_DIV34__burst1                     20260715  (no `_` after time)
    122502_P-4_W-C1_443-2_443screen2_DIV35_manual_prewash_          20260716  (no burst token)

Quirks the parser has to absorb, all confirmed against the 6504-FOV ``keep.csv``:

* the separator after the timestamp is sometimes absent (``114944P-1_…``);
* **6 of 6504** directories carry no ``burstN`` token — they are the manual/wash acquisitions,
  one of which misspells it (``maual_3rdwash``);
* ``dir1`` names list **all seven** wells, so the well is not recoverable from the name alone —
  see :func:`well_from_multiwell`;
* ``443GP`` and ``443screen1`` are the same batch.

Prefer ``keep.csv`` for plate/well: it already resolved them for every FOV, including the
``dir1`` rule. Use this module for what ``keep.csv`` does not carry — timestamp, DIV, batch,
plate format, and the condition tokens.
"""
import json
import os
import re

BATCHES = {"6GP002", "443GP", "443screen1", "443screen2"}
BATCH_ALIAS = {"443GP": "443screen1"}          # June dirs say 443GP; same batch
DYES = {"JF608"}
GENOTYPES = {"WT"}
CELL_LINES = {"443", "443-1", "443-2"}
WASH = {"manual", "maual", "prewash", "1stwash", "2ndwash", "3rdwash", "postwash"}

_TIMESTAMP = re.compile(r"^(\d{2})(\d{2})(\d{2})")
_PLATE = re.compile(r"^P-?\d+$")
_WELL = re.compile(r"^(?:W-)?([A-H]\d+)$")
_MULTIWELL = re.compile(r"^(?:[A-H]\d+-){2,}[A-H]\d+$")     # A1-A4-A2-C4-B4-C3-C2
_DIV = re.compile(r"^DIV(\d+)$")
_BURST = re.compile(r"^burst(\d+)$")
_PLATEFMT = re.compile(r"^(\d+)w$")
_DOSE = re.compile(r"^(\d+)x([A-Za-z]+)$")                  # 0xSM, 3xSM


def well_from_multiwell(wells, burst, bursts_per_well):
    """Well for a directory whose name lists several wells.

    The order the wells appear in the name is the acquisition order, so with a known number of
    bursts per well the well follows directly::

        well = wells[(burst - 1) // bursts_per_well]

    For ``20260331_dir1``: 393 bursts over 3 wells = 131 each, so bursts 1-131 are ``A1``,
    132-262 ``A4``, 263-393 ``A2`` — the first three entries of ``A1-A4-A2-C4-B4-C3-C2``.
    Returns ``None`` if the index falls outside the listed wells.
    """
    if not wells or not bursts_per_well or burst is None:
        return None
    i = (int(burst) - 1) // int(bursts_per_well)
    return wells[i] if 0 <= i < len(wells) else None


def parse_dir_name(dir_name):
    """Parse one FOV directory name into its fields.

    Returns a dict with ``timestamp``/``time_of_day``, ``plate``, ``well``, ``wells_in_name``,
    ``div``, ``burst``, ``batch_id``, ``plate_format``, ``eight_bit``, plus the semantic token
    buckets (``dye``, ``genotype``, ``cell_line``, ``dose``, ``wash``) and any leftover tokens
    in ``extra``. Every field is ``None``/empty rather than absent when not present in the name.
    """
    out = dict(dir_name=dir_name, timestamp=None, time_of_day=None, plate=None, well=None,
               wells_in_name=[], div=None, burst=None, batch_id=None, batch_raw=None,
               plate_format=None, eight_bit=False, dye=None, genotype=None, cell_line=None,
               dose=None, wash=None, extra=[])

    rest = dir_name
    m = _TIMESTAMP.match(rest)
    if m:
        out["timestamp"] = m.group(0)
        out["time_of_day"] = f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
        rest = rest[m.end():]                    # the separator is sometimes missing

    for tok in (t for t in rest.split("_") if t):
        if _PLATE.match(tok):
            out["plate"] = tok
        elif _MULTIWELL.match(tok):
            out["wells_in_name"] = tok.split("-")
        elif _WELL.match(tok) and out["well"] is None:
            out["well"] = _WELL.match(tok).group(1)
            out["wells_in_name"] = out["wells_in_name"] or [out["well"]]
        elif _DIV.match(tok):
            out["div"] = int(_DIV.match(tok).group(1))
        elif _BURST.match(tok):
            out["burst"] = int(_BURST.match(tok).group(1))
        elif _PLATEFMT.match(tok):
            out["plate_format"] = tok
        elif tok in BATCHES:
            out["batch_raw"] = tok
            out["batch_id"] = BATCH_ALIAS.get(tok, tok)
        elif tok.lower() == "8bit":
            out["eight_bit"] = True
        elif tok in DYES:
            out["dye"] = tok
        elif tok in GENOTYPES:
            out["genotype"] = tok
        elif tok in CELL_LINES:
            out["cell_line"] = tok
        elif _DOSE.match(tok):
            out["dose"] = tok
        elif tok.lower() in WASH:
            out["wash"] = f"{out['wash']}_{tok}" if out["wash"] else tok
        else:
            out["extra"].append(tok)
    return out


def fov_metadata(fov_dir, day=None, keep_row=None, params=None, profile=None, extras=None):
    """Full metadata record for one FOV, ready to write beside its analysis outputs.

    Parameters
    ----------
    fov_dir : str
        The FOV directory (the one holding ``frames1.bin``).
    day : str, optional
        Acquisition day. Defaults to the parent directory name, which is correct for the AB
        tree; pass it explicitly for nested layouts such as ``20260401/Data_B1/<fov>``.
    keep_row : dict, optional
        The FOV's ``keep.csv`` row. **Authoritative for plate/well** — it already applies the
        multi-well rule — and supplies bit depth, dimensions and frame count.
    params : Params, optional
        Records the settings the FOV was actually analysed with, so results stay interpretable.
    profile : str, optional
        Profile name, e.g. ``443screen2``.
    extras : dict, optional
        Merged in last (run id, code commit, timings, ...).
    """
    dir_name = os.path.basename(os.path.normpath(fov_dir))
    parsed = parse_dir_name(dir_name)
    meta = dict(parsed)
    meta["fov_dir"] = fov_dir
    meta["day"] = day or os.path.basename(os.path.dirname(os.path.normpath(fov_dir)))

    if keep_row:
        # keep.csv wins for well (it applies the multi-well rule); its burst column is populated
        # for dir1 only, so fall back to it rather than over it. Its `plate` is a float string
        # ("1.0"), so keep the readable name token as `plate` and store the number separately.
        for src, dst in (("well", "well"), ("well_source", "well_source"),
                         ("bit", "bit"), ("dims", "dims"), ("fov_path", "fov_path")):
            v = keep_row.get(src)
            if v not in (None, ""):
                meta[dst] = v
        pv = keep_row.get("plate")
        if pv not in (None, ""):
            try:
                meta["plate_num"] = int(float(pv))
            except (TypeError, ValueError):
                meta["plate_num"] = pv
        if meta.get("burst") is None and keep_row.get("burst") not in (None, ""):
            meta["burst"] = int(float(keep_row["burst"]))
        for c, k in (("n_frames_from_bin", "n_frames"), ("bin_bytes", "bin_bytes")):
            v = keep_row.get(c)
            if v not in (None, ""):
                meta[k] = int(float(v))
        if meta.get("day") != keep_row.get("day") and keep_row.get("day"):
            meta["day"] = keep_row["day"]

    if params is not None:
        meta["analysis"] = dict(
            profile=profile, nrow=params.nrow, ncol=params.ncol, fps=params.fps,
            read_dtype=params.read_dtype, compute_dtype=params.compute_dtype,
            bkg_ranges=[list(r) for r in params.bkg_ranges],
            std_ranges=[list(r) for r in params.std_ranges],
            sharpen_k=params.sharpen_k, seg_threshold=params.seg_threshold,
            seg_threshold_mult=getattr(params, "seg_threshold_mult", None),
            seg_neighborhood=list(getattr(params, "seg_neighborhood", None) or []) or None,
            seg_region_size=params.seg_region_size,
            max_region_bbox_frac=getattr(params, "max_region_bbox_frac", None),
            threshold_factor=params.threshold_factor,
            patch_local_filter=getattr(params, "patch_local_filter", True),
            whiten_traces=getattr(params, "whiten_traces", False))
    if extras:
        meta.update(extras)
    return meta


def write_metadata(out_dir, meta, name="fov_metadata.json"):
    """Write ``meta`` into ``out_dir`` — the same directory as ``ALI_Result.mat``, so metadata
    and results sit at identical depth and aggregation downstream is a single glob."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False, default=str)
    return path


def load_keep_index(keep_csv):
    """``{(day, dir_name): row}`` from ``keep.csv``, for joining during a run."""
    import csv
    with open(keep_csv) as f:
        return {(r["day"], r["dir_name"]): r for r in csv.DictReader(f)}
