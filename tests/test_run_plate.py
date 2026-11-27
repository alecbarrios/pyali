"""``scripts/run_plate.py``: run all wells of a plate, verify completeness, write the plate record.

S3 is faked: ``aws s3 ls`` returns a listing built from a set of done FOVs, ``aws s3 cp <src> -``
returns a metadata JSON, ``aws s3 cp - <dest>`` captures uploads.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_run_plate.py -q``
"""
import csv
import json
import os
import sys

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, _SCRIPTS)
import run_plate as rp                                                    # noqa: E402


def _inventory(path):
    rows = []
    for well in ("A1", "A2"):
        for b in (1, 2, 3):
            rows.append(dict(run_id=f"P3_{well}", fov_name=f"1200{b}0_P-3_W-{well}_443-2_443screen2_DIV34__burst{b}",
                             day="20260715", plate=3, plate_token="P-3", well=well, batch="443-2", burst=b,
                             src_dir=f"/raw/20260715/x{well}{b}", bin_bytes=1, bytes_per_frame=1,
                             frames_in_file=8399, frames_requested=8399,
                             include=not (well == "A2" and b == 3),
                             exclude_reason="" if not (well == "A2" and b == 3) else "zero-byte frames1.bin",
                             focus=""))
    rows.append(dict(rows[0], run_id="P5_A1", plate=5, plate_token="P-1", fov_name="other_plate__burst1"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


class FakeS3:
    def __init__(self, prefix, done, focus):
        self.prefix, self.done, self.focus, self.uploads = prefix, set(done), focus, {}

    def __call__(self, cmd, stdin=None):
        import subprocess
        if cmd[:3] == ["aws", "s3", "ls"]:
            lines = [f"2026-11-27 10:00:00 100 AB/x/plate_3/raw/Well_{w}/{n}/cells.parquet" for w, n in self.done]
            lines.append("2026-11-27 10:00:00 100 AB/x/plate_3/raw/_runs/P3_A1/manifest.jsonl")
            return subprocess.CompletedProcess(cmd, 0, "\n".join(lines), "")
        if cmd[:3] == ["aws", "s3", "cp"] and cmd[4] == "-":
            name = cmd[3].split("/")[-2]
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"focus_score": self.focus.get(name)}), "")
        if cmd[:4] == ["aws", "s3", "cp", "-"]:
            self.uploads[cmd[4]] = stdin
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(cmd)


def test_plate_rows_and_done_listing(tmp_path, monkeypatch):
    inv = str(tmp_path / "inv.csv")
    rows = _inventory(inv)
    pr = rp.plate_rows(inv, 3)
    assert len(pr) == 6 and [r["well"] for r in pr] == ["A1"] * 3 + ["A2"] * 3
    fake = FakeS3("s3://b/AB/x/plate_3/raw", {("A1", rows[0]["fov_name"])}, {})
    monkeypatch.setattr(rp, "sh", fake)
    assert rp.done_fovs("s3://b/AB/x/plate_3/raw") == {("A1", rows[0]["fov_name"])}


def test_focus_report_flags_absolute_and_relative():
    rows = [dict(well="A1", burst=str(b), fov_name=f"f{b}") for b in range(1, 6)]
    focus = {"f1": 2e-3, "f2": 2.2e-3, "f3": 2.1e-3, "f4": 6e-4, "f5": 1e-4}
    rep = {x["fov_name"]: x for x in rp.focus_report(rows, focus, abs_thr=3e-4, rel_thr=1 / 3)}
    assert rep["f5"]["suspect_defocus"] is True                    # absolute
    assert rep["f4"]["suspect_defocus"] is True                    # < 1/3 of the 2e-3 median
    assert rep["f1"]["suspect_defocus"] is False
    assert rp.focus_report([dict(well="A1", burst="1", fov_name="g")], {})[0]["suspect_defocus"] == ""


def test_main_runs_each_well_then_verifies_and_writes_the_record(tmp_path, monkeypatch):
    inv = str(tmp_path / "inv.csv")
    rows = _inventory(inv)
    calls = []
    monkeypatch.setattr(rp.run_corpus, "main", lambda args: calls.append(args) or 0)
    inc = [r for r in rows if r["plate"] == 3 and r["include"]]
    fake = FakeS3("s3://b/AB/x/plate_3/raw", {(r["well"], r["fov_name"]) for r in inc},
                  {r["fov_name"]: 2e-3 for r in inc})
    monkeypatch.setattr(rp, "sh", fake)
    scratch = tmp_path / "scratch"
    (scratch / "raw" / "P3_A1" / "Analysis" / "logs").mkdir(parents=True)
    (scratch / "raw" / "keep_me").mkdir()
    (scratch / "denoised-cellmincer" / "P3_A1").mkdir(parents=True)      # the other variant's scratch
    rc = rp.main(["--inventory", inv, "--plate", "3", "--variant", "raw", "--out-root", "s3://b/AB/x",
                  "--src-root", "/mnt/s3ab/AB", "--profile", "443screen2",
                  "--workers", "4", "--threads", "8", "--scratch", str(scratch),
                  "--", "--max-region-bbox-frac", "0.01"])
    assert rc == 0
    assert not (scratch / "raw" / "P3_A1").exists() and (scratch / "raw" / "keep_me").exists()  # plate's wells only
    assert (scratch / "denoised-cellmincer" / "P3_A1").exists()           # never another variant's files
    assert [c[c.index("--run-id") + 1] for c in calls] == ["P3_A1", "P3_A2"]   # well order, plate 3 only
    first = calls[0]
    assert first[first.index("--s3-prefix") + 1] == "s3://b/AB/x/plate_3/raw"
    assert first[first.index("--scratch") + 1] == str(scratch / "raw")
    assert first[first.index("--src-root") + 1] == "/mnt/s3ab/AB"
    assert first[first.index("--layout") + 1] == "well" and first[first.index("--profile") + 1] == "443screen2"
    assert first[-3:] == ["--", "--max-region-bbox-frac", "0.01"]
    summ = json.loads(fake.uploads["s3://b/AB/x/plate_3/raw/_runs/plate_summary.json"])
    assert summ["complete"] and summ["n_included"] == 5 and summ["n_done"] == 5 and summ["missing"] == []
    assert summ["variant"] == "raw" and summ["src_root"] == "/mnt/s3ab/AB"
    assert summ["out"] == "s3://b/AB/x/plate_3/raw"
    assert summ["wells"]["P3_A2"] == dict(rows=3, included=2, done=2, run_corpus_rc=0)
    inv_up = fake.uploads["s3://b/AB/x/plate_3/raw/_runs/inventory_plate_3.csv"]
    assert inv_up.count("\n") == 7                                 # header + 6 plate-3 rows
    assert "s3://b/AB/x/plate_3/raw/_runs/focus_report.csv" in fake.uploads
    assert not [k for k in fake.uploads if "/raw/" not in k]       # nothing lands in the flat plate folder


def test_variant_is_required_and_src_root_is_optional(tmp_path, monkeypatch):
    import pytest
    inv = str(tmp_path / "inv.csv")
    _inventory(inv)
    calls = []
    monkeypatch.setattr(rp.run_corpus, "main", lambda args: calls.append(args) or 0)
    with pytest.raises(SystemExit):                                # no --variant: refuse, run nothing
        rp.main(["--inventory", inv, "--plate", "3", "--out-root", "s3://b/AB/x"])
    with pytest.raises(SystemExit):                                # unknown variant
        rp.main(["--inventory", inv, "--plate", "3", "--variant", "denoised", "--out-root", "s3://b/AB/x"])
    assert calls == []
    assert rp.main(["--inventory", inv, "--plate", "3", "--variant", "denoised-cellmincer",
                    "--out-root", "s3://b/AB/x", "--scratch", str(tmp_path / "s"), "--dry-run"]) == 0
    assert all("--src-root" not in c for c in calls)               # inventory paths used as written
    assert calls[0][calls[0].index("--s3-prefix") + 1] == "s3://b/AB/x/plate_3/denoised-cellmincer"


def test_missing_fov_fails_and_verify_only_does_not_run(tmp_path, monkeypatch):
    inv = str(tmp_path / "inv.csv")
    rows = _inventory(inv)
    calls = []
    monkeypatch.setattr(rp.run_corpus, "main", lambda args: calls.append(args) or 0)
    inc = [r for r in rows if r["plate"] == 3 and r["include"]]
    fake = FakeS3("s3://b/AB/x/plate_3/raw", {(r["well"], r["fov_name"]) for r in inc[1:]}, {})
    monkeypatch.setattr(rp, "sh", fake)
    rc = rp.main(["--inventory", inv, "--plate", "3", "--variant", "raw", "--out-root", "s3://b/AB/x",
                  "--verify-only"])
    assert rc == 1 and calls == []
    summ = json.loads(fake.uploads["s3://b/AB/x/plate_3/raw/_runs/plate_summary.json"])
    assert not summ["complete"] and summ["missing"] == [f"A1/{inc[0]['fov_name']}"]
    assert rp.main(["--inventory", inv, "--plate", "9", "--variant", "raw", "--out-root", "s3://b/AB/x"]) == 2
