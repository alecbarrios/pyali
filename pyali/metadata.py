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

Identity for outputs (``fov_id`` / ``cell_id``) is :class:`Identity`, built from the name by
:func:`identity_from_dir_name` or supplied by the corpus inventory (``scripts/run_corpus.py``).
"""
import json
import os
import re
from dataclasses import dataclass, asdict

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


# --------------------------------------------------------------------------- #
# FOV / cell identity
# --------------------------------------------------------------------------- #
@dataclass
class Identity:
    """Where a FOV sits in the experiment; the source of every ``fov_id`` / ``cell_id``.

    ``fov_id = <experiment_id>_P<plate:02d>_<well>_F<fov:03d>`` and
    ``cell_id = <fov_id>_C<local_cell_id:04d>``. ``fov`` is the burst number when derived from a
    directory name; ``fov_index`` is the FOV's 1-based position in a corpus run (0 = unknown; an
    inventory-driven run fills it). ``batch`` is the cell-line batch token (``443-1`` / ``443-2``).
    """
    experiment_id: str
    plate: int
    well: str
    fov: int
    fov_index: int = 0
    batch: str = ""

    @property
    def fov_id(self):
        return f"{self.experiment_id}_P{int(self.plate):02d}_{self.well}_F{int(self.fov):03d}"

    def cell_id(self, local_cell_id):
        return f"{self.fov_id}_C{int(local_cell_id):04d}"

    def as_dict(self):
        d = asdict(self)
        d["fov_id"] = self.fov_id
        return d


def identity_from_dir_name(fov_dir, experiment_id=None, plate=None, well=None, fov=None,
                           fov_index=0):
    """:class:`Identity` from a FOV directory name, with explicit overrides.

    Defaults: ``experiment_id`` = the batch token (``443screen2``), ``plate`` = the digits of the
    ``P-N`` / ``P0N`` token, ``well`` = the ``W-A2`` / ``A2`` token, ``fov`` = the ``burstN`` token.
    Anything the name lacks becomes ``"unknown"`` / ``0`` / ``""`` rather than an error, so an
    arbitrary directory (a synthetic test FOV) still gets a well-formed id. Pass ``plate`` for the
    relabelled days (20260715 ``P-1`` folders are plate 5).
    """
    d = parse_dir_name(os.path.basename(os.path.normpath(fov_dir)))
    if plate is None:
        digits = re.sub(r"\D", "", d["plate"] or "")
        plate = int(digits) if digits else 0
    if well is None:
        well = d["well"] or ""
    if fov is None:
        fov = d["burst"] if d["burst"] is not None else 0
    if experiment_id is None:
        experiment_id = d["batch_id"] or "unknown"
    return Identity(experiment_id=str(experiment_id), plate=int(plate), well=str(well),
                    fov=int(fov), fov_index=int(fov_index), batch=d.get("cell_line") or "")


def _jsonable(v):
    """Tuples -> lists (recursively) so ``Params`` fields serialize as JSON arrays."""
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def fov_metadata(fov_dir, day=None, params=None, profile=None, extras=None,
                 identity=None, timing=None, telemetry=None):
    """Full metadata record for one FOV, ready to write beside its analysis outputs.

    Parameters
    ----------
    fov_dir : str
        The FOV directory (the one holding ``frames1.bin``).
    day : str, optional
        Acquisition day. Defaults to the parent directory name, which is correct for the AB
        tree; pass it explicitly for nested layouts such as ``20260401/Data_B1/<fov>``.
    params : Params, optional
        Every field is recorded under ``params`` so results stay interpretable.
    profile : str, optional
        Profile name, e.g. ``443screen2``.
    extras : dict, optional
        Merged in last (run id, counts, ...).
    identity : Identity, optional
        Recorded under ``identity`` (plus a top-level ``fov_id``).
    timing : dict, optional
        Per-stage wall seconds, recorded under ``timing_s``.
    telemetry : dict, optional
        A :meth:`pyali.telemetry.StageMeter.report`: ``timing_s`` (takes precedence over
        ``timing``), ``cpu_s``, ``memory_gb`` (per-stage RSS start / end / peak), ``peak_rss_gb``,
        ``rss_start_gb`` and ``threads``, all recorded under the same keys.
    """
    dir_name = os.path.basename(os.path.normpath(fov_dir))
    parsed = parse_dir_name(dir_name)
    meta = dict(parsed)
    meta["fov_dir"] = fov_dir
    meta["day"] = day or os.path.basename(os.path.dirname(os.path.normpath(fov_dir)))
    if identity is not None:
        meta["identity"] = identity.as_dict()
        meta["fov_id"] = identity.fov_id
    meta["profile"] = profile
    from .results import git_sha
    meta["pyali_git"] = git_sha()

    if params is not None:
        meta["params"] = _jsonable(asdict(params))          # every field, not a hand-picked subset
    if timing:
        meta["timing_s"] = {k: round(float(v), 3) for k, v in timing.items()}
    if telemetry:
        for key in ("timing_s", "cpu_s", "memory_gb", "peak_rss_gb", "peak_mem_gb", "rss_start_gb",
                    "threads"):
            if key in telemetry:
                meta[key] = telemetry[key]
    if extras:
        meta.update(extras)
    return meta


def write_metadata(out_dir, meta, name="fov_metadata.json"):
    """Write ``meta`` into ``out_dir`` — the same directory as ``signals.h5`` / ``cells.parquet``,
    so metadata and results sit at identical depth and aggregation downstream is a single glob."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=False, default=str)
    return path
