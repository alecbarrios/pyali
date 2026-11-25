"""``scripts/run_corpus.py``: inventory-driven corpus runner (Cycle B3).

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_run_corpus.py -q``
"""
import csv
import os
import subprocess
import sys

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, _SCRIPTS)
import run_corpus as rc                                                    # noqa: E402

COLS = ["run_id", "fov_name", "day", "plate", "plate_token", "well", "batch", "burst", "src_dir",
        "bin_bytes", "bytes_per_frame", "frames_in_file", "frames_requested", "include",
        "exclude_reason", "focus"]


def _inventory(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    return str(path)


def _row(burst, run_id="P4_A2", plate=4, well="A2", day="20260716", include="True", reason="",
         batch="443-1", root="/mnt/stp"):
    name = f"16{burst:04d}_P-{plate}_W-{well}_{batch}_443screen2_DIV35__burst{burst}"
    return dict(run_id=run_id, fov_name=name, day=day, plate=plate, plate_token=f"P-{plate}", well=well,
                batch=batch, burst=burst, src_dir=f"{root}/{day}/{name}", bin_bytes=5375360000,
                bytes_per_frame=640000, frames_in_file=8399, frames_requested=8399,
                include=include, exclude_reason=reason)


# --------------------------------------------------------------------------- #
# load_run / reroot / dest_prefix
# --------------------------------------------------------------------------- #
def test_reroot_replaces_everything_before_the_day_component():
    assert rc.reroot("/mnt/stp/20260716/fovA", "20260716", "/home/x/AB") == "/home/x/AB/20260716/fovA"
    assert rc.reroot("/mnt/stp/20260717/20260717/fovB", "20260717", "/r") == "/r/20260717/fovB"
    assert rc.reroot("/mnt/stp/20260716/fovA", "20260716", None) == "/mnt/stp/20260716/fovA"
    with pytest.raises(ValueError):
        rc.reroot("/mnt/stp/other/fovA", "20260716", "/r")


def test_load_run_numbers_included_and_excluded_rows_and_builds_identity(tmp_path):
    inv = _inventory(tmp_path / "inv.csv", [
        _row(3), _row(1), _row(2, include="False", reason="zero-byte frames1.bin"),
        _row(1, run_id="P4_C3", well="C3", batch="443-2"),              # another run-id
    ])
    rows = rc.load_run(inv, "P4_A2", src_root="/data/AB")
    assert [r["fov"] for r in rows] == [1, 2, 3]
    assert [r["fov_index"] for r in rows] == [1, 2, 3]
    assert [r["include"] for r in rows] == [True, False, True]
    assert rows[0]["src"] == f"/data/AB/20260716/{rows[0]['fov_name']}"
    i = rows[2]["identity"]
    assert (i.experiment_id, i.plate, i.well, i.fov, i.fov_index, i.batch) == \
        ("443screen2", 4, "A2", 3, 3, "443-1")
    assert rows[2]["fov_id"] == "443screen2_P04_A2_F003"
    assert all(r["run_id"] == "P4_A2" for r in rows)
    with pytest.raises(ValueError):
        rc.load_run(inv, "P9_Z9")


def test_load_run_experiment_id_override(tmp_path):
    inv = _inventory(tmp_path / "inv.csv", [_row(1)])
    rows = rc.load_run(inv, "P4_A2", experiment_id="exp")
    assert rows[0]["fov_id"] == "exp_P04_A2_F001"


def test_dest_prefix_layout():
    assert rc.dest_prefix("s3://b/AB/pyali-x/", 4, "A2") == "s3://b/AB/pyali-x/Plate_4/Well_A2"


# --------------------------------------------------------------------------- #
# existing_done / fov_command
# --------------------------------------------------------------------------- #
def test_existing_done_reads_markers_in_the_plate_well_layout(monkeypatch):
    listing = "\n".join([
        "2026-11-17 10:00:00     123456 AB/pyali-x/Plate_4/Well_A2/fovA/cells.parquet",
        "2026-11-17 10:00:00    9876543 AB/pyali-x/Plate_4/Well_A2/fovA/signals.h5",
        "2026-11-17 10:00:00        100 AB/pyali-x/Plate_4/Well_A2/fovB/fov_metadata.json",   # partial
        "2026-11-17 10:00:00      54321 AB/pyali-x/Plate_4/Well_A2/fovC/ALI_Result.mat",      # legacy
        "2026-11-17 10:00:00     123456 AB/pyali-x/Plate_4/Well_A2/fovD/cells.parquet",
    ]) + "\n"
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=listing, stderr="")
    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    assert rc.existing_done("s3://b/AB/pyali-x", 4, "A2") == {"fovA", "fovD"}
    assert calls == [["aws", "s3", "ls", "s3://b/AB/pyali-x/Plate_4/Well_A2/", "--recursive"]]

    monkeypatch.setattr(rc.subprocess, "run",
                        lambda cmd, capture_output, text: subprocess.CompletedProcess(cmd, 1, "", "nope"))
    assert rc.existing_done("s3://b/AB/pyali-x", 4, "A2") == set()


def test_fov_command_assembly():
    from pyali.metadata import Identity
    ident = Identity("443screen2", 5, "A2", 12, fov_index=7, batch="443-1")
    cmd = rc.fov_command("/py", "/src/fov", "/scratch/fov", ident)
    assert cmd[:2] == ["/py", rc.RUN_PYALI] and cmd[2] == "/src/fov"
    assert cmd[3:] == ["--out", "/scratch/fov", "--experiment-id", "443screen2", "--plate", "5",
                       "--well", "A2", "--fov", "12", "--fov-index", "7"]
    cmd = rc.fov_command("/py", "/src/fov", "/o", ident, profile="443screen2", figures=True,
                         extra=["--compute-dtype", "float32"])
    assert cmd[-6:] == ["--profile", "443screen2", "--figures", "--compute-dtype", "float32"][-6:] or \
        cmd[-5:] == ["--profile", "443screen2", "--figures", "--compute-dtype", "float32"]
    assert cmd.count("--profile") == 1 and cmd[-2:] == ["--compute-dtype", "float32"]


# --------------------------------------------------------------------------- #
# run_one with a fake run_pyali and a fake aws
# --------------------------------------------------------------------------- #
FAKE_RUN_PYALI = r'''
import json, os, sys, time
mode = os.environ.get("FAKE_MODE", "ok")
out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out, exist_ok=True)
if mode == "fail":
    print("boom"); sys.exit(3)
if mode == "hog":
    buf = bytearray(300 * 1024 * 1024)           # ~300 MB resident
    time.sleep(8)
if mode != "nomarker":
    open(os.path.join(out, "cells.parquet"), "wb").write(b"x")
json.dump(dict(n_cells=7, n_cells_by_source=dict(spiking=5, no_ap=2), n_regions_kept=9, focus_score=0.0021),
          open(os.path.join(out, "fov_metadata.json"), "w"))
print("argv", sys.argv[1:])
'''


def _cfg(tmp_path, monkeypatch, **over):
    import types
    fake = tmp_path / "fake_run_pyali.py"
    fake.write_text(FAKE_RUN_PYALI)
    monkeypatch.setattr(rc, "RUN_PYALI", str(fake))
    analysis = tmp_path / "scratch" / "P4_A2" / "Analysis"
    logd = analysis / "logs"
    logd.mkdir(parents=True)
    lines = []
    cfg = types.SimpleNamespace(
        python=sys.executable, analysis_dir=str(analysis), log_dir=str(logd),
        s3_prefix="s3://b/AB/pyali-x", threads=1, max_fov_rss_gb=0.0, min_avail_gb=0.0,
        min_free_disk_gb=0.0, profile=None, figures=False, extra=[], git_sha="abc123",
        run_id="P4_A2", log=lines.append, halt=None, launch_lock=__import__("threading").Lock(),
        stagger_s=0.0, watch_s=0.2)
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.lines = lines
    return cfg


def _fake_aws(monkeypatch, verify_ok=True, cp_ok=True, calls=None):
    calls = [] if calls is None else calls

    def fake_sh(cmd):
        calls.append(cmd)
        if cmd[:3] == ["aws", "s3", "cp"]:
            return subprocess.CompletedProcess(cmd, 0 if cp_ok else 1, "", "" if cp_ok else "denied")
        if cmd[:3] == ["aws", "s3", "ls"]:
            return subprocess.CompletedProcess(cmd, 0, f"2026-11-17 10:00:00 1 {cmd[3]}\n" if verify_ok else "", "")
        raise AssertionError(cmd)
    monkeypatch.setattr(rc, "sh", fake_sh)
    return calls


def _one_row(tmp_path):
    inv = _inventory(tmp_path / "inv.csv", [_row(1)])
    return rc.load_run(inv, "P4_A2")[0]


def test_run_one_ok_path_uploads_verifies_and_deletes_local(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    calls = _fake_aws(monkeypatch)
    rec = rc.run_one(_one_row(tmp_path), cfg)
    assert rec["ok"] is True and rec["rc"] == 0 and rec["n_cells"] == 7
    assert rec["n_cells_by_source"] == dict(spiking=5, no_ap=2) and rec["n_regions"] == 9
    assert rec["focus_score"] == 0.0021 and any("focus 2.10e-03" in line for line in cfg.lines)
    assert rec["dest"].startswith("s3://b/AB/pyali-x/Plate_4/Well_A2/16")
    local = os.path.join(cfg.analysis_dir, "Plate_4", "Well_A2", rec["fov_name"])
    assert not os.path.exists(local)                                   # deleted after read-back
    assert calls[0][:3] == ["aws", "s3", "cp"] and calls[0][3] == local and calls[0][4] == rec["dest"] + "/"
    assert calls[1] == ["aws", "s3", "ls", rec["dest"] + "/cells.parquet"]
    assert rec["killed"] is False and rec["peak_rss_gb"] >= 0 and rec["git_sha"] == "abc123"
    log = open(os.path.join(cfg.log_dir, "fov001_P4_A2.log")).read()
    assert "--fov-index 1" in log.replace("', '", " ").replace("'", "") and "--plate 4" in log.replace("', '", " ").replace("'", "")


def test_run_one_records_a_nonzero_exit_and_keeps_nothing_uploaded(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    calls = _fake_aws(monkeypatch)
    monkeypatch.setenv("FAKE_MODE", "fail")
    rec = rc.run_one(_one_row(tmp_path), cfg)
    assert rec["ok"] is False and rec["rc"] == 3 and "rc=3" in rec["error"] and "boom" in rec["error"]
    assert calls == []


def test_run_one_missing_marker_is_a_failure(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    _fake_aws(monkeypatch)
    monkeypatch.setenv("FAKE_MODE", "nomarker")
    rec = rc.run_one(_one_row(tmp_path), cfg)
    assert rec["ok"] is False and "marker=False" in rec["error"]


def test_run_one_upload_verify_failure_keeps_the_local_copy(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    _fake_aws(monkeypatch, verify_ok=False)
    rec = rc.run_one(_one_row(tmp_path), cfg)
    assert rec["ok"] is False and "verification failed" in rec["error"]
    local = os.path.join(cfg.analysis_dir, "Plate_4", "Well_A2", rec["fov_name"])
    assert os.path.isfile(os.path.join(local, "cells.parquet"))     # not deleted


def test_run_one_watchdog_kills_a_memory_hog(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, max_fov_rss_gb=0.1, watch_s=0.2)
    _fake_aws(monkeypatch)
    monkeypatch.setenv("FAKE_MODE", "hog")
    rec = rc.run_one(_one_row(tmp_path), cfg)
    assert rec["ok"] is False and rec["killed"] is True and rec["rc"] != 0
    assert rec["peak_rss_gb"] > 0.1 and rec["process_s"] < 8


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _main_env(tmp_path, monkeypatch, rows, done=(), verify_ok=True):
    """A fake run_pyali, a fake aws (listing ``done`` as already in S3), and an inventory."""
    fake = tmp_path / "fake_run_pyali.py"
    fake.write_text(FAKE_RUN_PYALI)
    monkeypatch.setattr(rc, "RUN_PYALI", str(fake))
    inv = _inventory(tmp_path / "inv.csv", rows)
    calls = []

    def fake_sh(cmd):
        calls.append(cmd)
        if cmd[:3] == ["aws", "s3", "ls"] and cmd[-1] == "--recursive":
            listing = "".join(f"2026-11-17 10:00:00 1 AB/x/Plate_4/Well_A2/{n}/cells.parquet\n" for n in done)
            return subprocess.CompletedProcess(cmd, 0, listing, "")
        if cmd[:3] == ["aws", "s3", "ls"]:
            return subprocess.CompletedProcess(cmd, 0, f"x {cmd[3]}\n" if verify_ok else "", "")
        if cmd[:3] == ["aws", "s3", "cp"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, "deadbeefcafe\n" if "rev-parse" in cmd else "", "")
        raise AssertionError(cmd)
    monkeypatch.setattr(rc, "sh", fake_sh)
    return inv, calls


def _base_args(tmp_path, inv, *more):
    return ["--inventory", inv, "--run-id", "P4_A2", "--s3-prefix", "s3://b/AB/x",
            "--scratch", str(tmp_path / "scratch"), "--workers", "2", "--stagger-s", "0",
            "--min-avail-gb", "0", "--min-free-disk-gb", "0", *more]


def test_main_dry_run_lists_and_writes_fov_index(tmp_path, monkeypatch, capsys):
    rows = [_row(1), _row(2, include="False", reason="zero-byte frames1.bin"), _row(3)]
    inv, calls = _main_env(tmp_path, monkeypatch, rows)
    rc_ = rc.main(_base_args(tmp_path, inv, "--dry-run", "--src-root", "/alt"))
    assert rc_ == 0
    out = capsys.readouterr().out
    assert "3 in inventory (1 excluded), 0 done in S3, 2 to run" in out
    assert "would run fov 001" in out and "/alt/20260716/" in out
    idx = list(csv.DictReader(open(tmp_path / "scratch" / "P4_A2" / "Analysis" / "fov_index.csv")))
    assert [(r["fov_index"], r["include"]) for r in idx] == [("1", "True"), ("2", "False"), ("3", "True")]
    assert idx[1]["exclude_reason"] == "zero-byte frames1.bin"
    assert not any(c[:3] == ["aws", "s3", "cp"] for c in calls)         # nothing uploaded


def test_main_expected_n_gate(tmp_path, monkeypatch, capsys):
    inv, _ = _main_env(tmp_path, monkeypatch, [_row(1), _row(2)])
    assert rc.main(_base_args(tmp_path, inv, "--dry-run", "--expected-n", "3")) == 2
    assert "refusing to run" in capsys.readouterr().out
    assert rc.main(_base_args(tmp_path, inv, "--dry-run", "--expected-n", "3", "--force")) == 0
    assert rc.main(_base_args(tmp_path, inv, "--dry-run", "--expected-n", "2")) == 0


def test_main_runs_skips_done_writes_manifest_and_archives(tmp_path, monkeypatch, capsys):
    rows = [_row(1), _row(2), _row(3)]
    inv, calls = _main_env(tmp_path, monkeypatch, rows, done=[rows[1]["fov_name"]])
    rc_ = rc.main(_base_args(tmp_path, inv))
    assert rc_ == 0
    out = capsys.readouterr().out
    assert "1 done in S3, 2 to run" in out and "ALL DONE P4_A2" in out and "2 ok, 0 failed" in out
    manifest = [__import__("json").loads(l) for l in open(tmp_path / "scratch" / "P4_A2" / "Analysis" / "manifest.jsonl")]
    assert sorted(m["fov"] for m in manifest) == [1, 3] and all(m["ok"] for m in manifest)
    assert all(m["n_cells"] == 7 and m["git_sha"].startswith("deadbeef") for m in manifest)
    uploads = [c for c in calls if c[:3] == ["aws", "s3", "cp"]]
    assert sum(1 for c in uploads if c[4].startswith("s3://b/AB/x/Plate_4/Well_A2/")) == 2
    archived = [c[4] for c in uploads if "/_runs/P4_A2/" in c[4]]
    assert "s3://b/AB/x/_runs/P4_A2/manifest.jsonl" in archived
    assert "s3://b/AB/x/_runs/P4_A2/fov_index.csv" in archived
    assert "s3://b/AB/x/_runs/P4_A2/logs/" in archived


def test_main_only_and_failure_exit_code(tmp_path, monkeypatch, capsys):
    inv, _ = _main_env(tmp_path, monkeypatch, [_row(1), _row(2)])
    monkeypatch.setenv("FAKE_MODE", "fail")
    rc_ = rc.main(_base_args(tmp_path, inv, "--only", "2", "--no-resume"))
    assert rc_ == 1
    out = capsys.readouterr().out
    assert "1 to run" in out and "0 ok, 1 failed" in out and "FAILED" in out


def test_main_passes_extra_args_after_double_dash(tmp_path, monkeypatch):
    inv, _ = _main_env(tmp_path, monkeypatch, [_row(1)])
    assert rc.main(_base_args(tmp_path, inv, "--no-resume", "--", "--compute-dtype", "float32")) == 0
    log = open(tmp_path / "scratch" / "P4_A2" / "Analysis" / "logs" / "fov001_P4_A2.log").read()
    assert "--compute-dtype" in log and "float32" in log


def test_rss_gb_and_avail_ram_are_sane():
    assert rc.rss_gb(os.getpid()) > 0.01
    assert rc.rss_gb(2 ** 22 + 12345) == 0.0
    assert rc.avail_ram_gb() > 0
