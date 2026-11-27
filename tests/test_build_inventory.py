"""``scripts/build_inventory.py``: inventory of FOV directories for ``run_corpus``.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_build_inventory.py -q``
"""
import csv
import os
import sys

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, _SCRIPTS)
import build_inventory as bi                                               # noqa: E402


def _fov(root, day, name, n_bytes=640000 * 3, sidecar=True, target=3, bpf=640000, seen=None):
    """Make ``root/day/name/frames1.bin`` (+ sidecar) and return the directory."""
    d = os.path.join(root, day, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "frames1.bin"), "wb") as f:
        f.write(b"\0" * n_bytes)
    if sidecar:
        with open(os.path.join(d, "frames1_dropped_frames.txt"), "w") as f:
            f.write(f"target_frames={target}\nlast_frame_seen={seen if seen is not None else target}\n"
                    f"exposure_bytes_per_frame={bpf}\ndropped_frames_total=0\n\nDropped frame ranges:\n")
    return d


# --------------------------------------------------------------------------- #
# read_sidecar / scan_fovs
# --------------------------------------------------------------------------- #
def test_read_sidecar_parses_ints_and_tolerates_missing(tmp_path):
    d = _fov(str(tmp_path), "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1",
             target=8399, bpf=640000)
    s = bi.read_sidecar(d)
    assert s["target_frames"] == 8399 and s["exposure_bytes_per_frame"] == 640000
    assert s["last_frame_seen"] == 8399 and s["dropped_frames_total"] == 0
    assert isinstance(s["target_frames"], int)
    assert bi.read_sidecar(str(tmp_path)) == {}                          # no sidecar here


def test_scan_fovs_finds_nested_dirs_and_ignores_others(tmp_path):
    root = str(tmp_path)
    a = _fov(root, "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1")
    b = _fov(root, "20260716", "163547_P-4_W-A2_443-1_443screen2_DIV35__burst2")
    c = _fov(root, "20260717/20260717", "120000_P-2_W-C3_443-2_443screen2_DIV36__burst5")   # nested twin
    os.makedirs(os.path.join(root, "20260716", "Session_Notes_dir"))                           # no movie
    with open(os.path.join(root, "20260716", "Session_Notes.txt"), "w") as f:
        f.write("P-1 is actually P-5\n")
    assert bi.scan_fovs(root) == sorted([a, b, c])
    assert bi.scan_fovs(os.path.join(root, "20260716")) == [a, b]


# --------------------------------------------------------------------------- #
# inventory_row
# --------------------------------------------------------------------------- #
def test_inventory_row_443screen2_name_forms(tmp_path):
    root = str(tmp_path)
    a = _fov(root, "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1", target=3)
    b = _fov(root, "20260718", "154200P-1_W-A2_443-1_443screen2_DIV37__burst12", target=3)
    ra = bi.inventory_row(a, root, "443screen2")
    assert (ra["run_id"], ra["day"], ra["plate"], ra["plate_token"], ra["well"], ra["batch"], ra["burst"]) == \
        ("P4_A2", "20260716", 4, "P-4", "A2", "443-1", 1)
    assert ra["include"] is True and ra["exclude_reason"] == ""
    assert ra["bin_bytes"] == 640000 * 3 and ra["bytes_per_frame"] == 640000
    assert ra["frames_in_file"] == 3 and ra["frames_requested"] == 3
    assert ra["src_dir"] == os.path.abspath(a)
    rb = bi.inventory_row(b, root, "443screen2")
    assert (rb["run_id"], rb["plate"], rb["well"], rb["burst"]) == ("P1_A2", 1, "A2", 12)


def test_inventory_row_day_is_root_name_when_fovs_sit_in_root(tmp_path):
    root = os.path.join(str(tmp_path), "20260716")
    a = _fov(str(tmp_path), "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1")
    assert bi.inventory_row(a, root, "x")["day"] == "20260716"


def test_inventory_row_plate_map_relabels_only_that_day(tmp_path):
    root = str(tmp_path)
    a = _fov(root, "20260715", "114944P-1_W-A1_443_443screen2_DIV34__burst1")
    b = _fov(root, "20260718", "154200P-1_W-A2_443-1_443screen2_DIV37__burst1")
    pm = bi.parse_plate_map(["20260715:P-1=5"])
    assert pm == {("20260715", "P-1"): 5}
    assert bi.inventory_row(a, root, "x", pm)["run_id"] == "P5_A1"
    assert bi.inventory_row(a, root, "x", pm)["plate_token"] == "P-1"
    assert bi.inventory_row(b, root, "x", pm)["run_id"] == "P1_A2"
    with pytest.raises(ValueError):
        bi.parse_plate_map(["nonsense"])


@pytest.mark.parametrize("kw, reason", [
    (dict(n_bytes=0), "zero-byte"),
    (dict(n_bytes=640000 * 3 + 17), "whole number"),
    (dict(n_bytes=640000 * 2, target=8399, seen=2), "aborted acquisition (2 of 8399"),
])
def test_inventory_row_exclusions(tmp_path, kw, reason):
    root = str(tmp_path)
    d = _fov(root, "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1", **kw)
    r = bi.inventory_row(d, root, "x")
    assert r["include"] is False and reason in r["exclude_reason"]
    assert r["run_id"] == "P4_A2"                                   # excluded rows keep their identity


def test_inventory_row_no_burst_token_and_unparseable_name(tmp_path):
    root = str(tmp_path)
    m = _fov(root, "20260716", "122502_P-4_W-C1_443-2_443screen2_DIV35_manual_prewash_")
    r = bi.inventory_row(m, root, "x")
    assert r["include"] is False and "burstN" in r["exclude_reason"] and r["burst"] == 0
    u = _fov(root, "20260716", "random_folder_burst3")
    r = bi.inventory_row(u, root, "x")
    assert r["include"] is False and "plate/well" in r["exclude_reason"]
    assert r["run_id"] == "P0_"


def test_inventory_row_without_sidecar_is_included_with_blank_frame_fields(tmp_path):
    root = str(tmp_path)
    d = _fov(root, "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1", sidecar=False)
    r = bi.inventory_row(d, root, "x")
    assert r["include"] is True and r["bytes_per_frame"] == 0
    assert r["frames_in_file"] == "" and r["frames_requested"] == ""


# --------------------------------------------------------------------------- #
# build_inventory / CLI
# --------------------------------------------------------------------------- #
def test_build_inventory_writes_sorted_csv_with_columns_and_well_filter(tmp_path, capsys):
    root = str(tmp_path / "AB")
    _fov(root, "20260716", "163547_P-4_W-A2_443-1_443screen2_DIV35__burst2")
    _fov(root, "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1")
    _fov(root, "20260716", "135834_P-4_W-C3_443-2_443screen2_DIV35__burst1")
    _fov(root, "20260716", "135000_P-4_W-C1_443-2_443screen2_DIV35__burst10", n_bytes=0)
    _fov(root, "20260716", "140000_P-4_W-A2_443-1_443screen2_DIV35__burst10")
    out = str(tmp_path / "inv" / "inv.csv")
    rc = bi.main(["--root", root, "--out", out, "--experiment-id", "443screen2", "--wells", "A2", "C3"])
    assert rc == 0
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == bi.COLUMNS
    assert [(r["run_id"], int(r["burst"])) for r in rows] == \
        [("P4_A2", 1), ("P4_A2", 2), ("P4_A2", 10), ("P4_C3", 1)]      # burst sorted numerically, C1 dropped
    assert all(r["include"] == "True" for r in rows)
    printed = capsys.readouterr().out
    assert "P4_A2" in printed and "include=3" in printed and "P4_C3" in printed


def test_build_inventory_focus_column(tmp_path, capsys):
    """--focus-dims fills the focus QC column from the first 300 frames of each included movie
    (blank for excluded rows and without the flag)."""
    root = str(tmp_path)
    nrow = ncol = 8
    bpf = nrow * ncol
    good = _fov(root, "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1",
                n_bytes=bpf * 300, target=300, bpf=bpf)
    # write a sharp pattern into the frames so the score is non-zero
    import numpy as np
    frames = np.zeros((300, nrow, ncol), np.uint8)
    frames[:, 2:6, 2:6] = 200
    frames.tofile(os.path.join(good, "frames1.bin"))
    _fov(root, "20260716", "163547_P-4_W-A2_443-1_443screen2_DIV35__burst2", n_bytes=0)
    out = str(tmp_path / "inv.csv")
    rows = bi.build_inventory([root], out, "x", focus_dims=(nrow, ncol))
    assert rows[0]["focus"] != "" and float(rows[0]["focus"]) > 0
    assert rows[1]["focus"] == ""                                      # excluded: not scored
    with open(out) as f:
        hdr = f.readline().strip().split(",")
    assert hdr[-1] == "focus"
    rows_noflag = bi.build_inventory([root], out, "x")
    assert all(r["focus"] == "" for r in rows_noflag)
    bi.main(["--root", root, "--out", out, "--experiment-id", "x", "--focus-dims", str(nrow), str(ncol)])
    assert "[inventory] focus: median" in capsys.readouterr().out


def test_build_inventory_reports_exclusions(tmp_path, capsys):
    root = str(tmp_path)
    _fov(root, "20260716", "163532_P-4_W-A2_443-1_443screen2_DIV35__burst1")
    _fov(root, "20260716", "163547_P-4_W-A2_443-1_443screen2_DIV35__burst2", n_bytes=0)
    out = str(tmp_path / "inv.csv")
    rows = bi.build_inventory([root], out, "x")
    assert bi.summarize(rows) == [("P4_A2", 1, 1)]
    bi.main(["--root", root, "--out", out, "--experiment-id", "x"])
    assert "EXCLUDED 20260716/163547" in capsys.readouterr().out



def test_batch_labels_wt_genotype_batch_map_and_duplicates(tmp_path):
    """WT wells get the genotype as their label; --batch-map corrects a well's label on one day;
    a directory copied into a nested day folder is excluded as a duplicate of the shallow copy."""
    root = str(tmp_path)
    _fov(root, "20260717", "102853_P-2_W-A1_WT_443screen2_DIV36__burst1")
    _fov(root, "20260717/20260717", "102853_P-2_W-A1_WT_443screen2_DIV36__burst1")      # nested twin
    _fov(root, "20260715", "114944P-1_W-A1_443_443screen2_DIV34__burst1")
    _fov(root, "20260715", "115000P-1_W-A2_443-2_443screen2_DIV34__burst1")
    bm = bi.parse_batch_map(["20260715:P-1:a1=443-2"])
    assert bm == {("20260715", "P-1", "A1"): "443-2"}
    with pytest.raises(ValueError):
        bi.parse_batch_map(["20260715:P-1=443-2"])
    rows = bi.build_inventory([root], str(tmp_path / "inv.csv"), "x",
                              plate_map=bi.parse_plate_map(["20260715:P-1=5"]), batch_map=bm)
    by = {(r["day"], r["fov_name"], r["src_dir"].count(os.sep)): r for r in rows}
    wt = [r for r in rows if r["well"] == "A1" and r["plate"] == 2]
    assert len(wt) == 2 and all(r["batch"] == "WT" for r in wt)
    shallow, deep = sorted(wt, key=lambda r: r["src_dir"].count(os.sep))
    assert shallow["include"] and not deep["include"]
    assert deep["exclude_reason"] == f"duplicate of {shallow['src_dir']}"
    p5 = {r["well"]: r["batch"] for r in rows if r["plate"] == 5}
    assert p5 == {"A1": "443-2", "A2": "443-2"}                    # A1 corrected, A2 as named
    assert len(by) == 4
