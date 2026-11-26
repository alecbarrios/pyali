"""``pyali.telemetry``: per-stage wall / CPU / RSS metering.

Run: ``/home/jovyan/.venvs/pyali/bin/python -m pytest tests/test_telemetry.py -q``
"""
import os
import time

import numpy as np

from pyali import telemetry as tm


def test_rss_and_avail_ram_are_sane():
    assert tm.rss_gb() > 0.01
    assert tm.rss_gb(os.getpid()) == tm.rss_gb("self") or abs(tm.rss_gb(os.getpid()) - tm.rss_gb()) < 0.5
    assert tm.rss_gb(2 ** 22 + 12345) == 0.0
    assert tm.avail_ram_gb() > 0


def test_thread_info_keys():
    info = tm.thread_info()
    assert info["cpu_count"] >= 1 and "omp_num_threads" in info and "blas" in info


def test_stage_meter_records_wall_cpu_and_rss_peak():
    lines = []
    with tm.StageMeter(interval=0.02, log=lines.append) as meter:
        with meter.stage("a"):
            x = np.arange(200_000, dtype=np.float64).sum()        # trivial
        with meter.stage("b"):
            big = np.ones(25_000_000, dtype=np.float64)           # 200 MB, touched
            big += 1.0
            time.sleep(0.1)                                      # let the sampler see it
            peak_inside = tm.rss_gb()
        del big
    rep = meter.report()
    assert list(rep["timing_s"]) == ["a", "b", "total"]
    assert rep["timing_s"]["total"] >= rep["timing_s"]["a"] + rep["timing_s"]["b"] - 1e-3
    assert rep["timing_s"]["b"] >= 0.1 and rep["cpu_s"]["a"] >= 0 and rep["cpu_s"]["b"] >= 0
    mb = rep["memory_gb"]["b"]
    assert mb["peak"] >= mb["start"] and mb["peak"] >= mb["end"]
    assert mb["peak"] - mb["start"] >= 0.15                       # the 200 MB allocation was seen
    assert abs(mb["peak"] - peak_inside) < 0.05
    assert rep["peak_rss_gb"] >= mb["peak"] and rep["rss_start_gb"] > 0
    assert set(rep) == {"timing_s", "cpu_s", "memory_gb", "peak_rss_gb", "peak_mem_gb", "rss_start_gb",
                        "threads"}
    assert rep["peak_mem_gb"] >= rep["peak_rss_gb"]
    assert "tree_pss_peak" not in mb                              # no children in this stage
    assert len(lines) == 2 and lines[0].startswith("a: ") and "GB peak" in lines[1]
    assert not meter._thread.is_alive()                           # closed on exit
    assert x > 0


def test_stage_meter_reuse_overwrites_and_survives_exceptions():
    meter = tm.StageMeter(interval=0.05)
    with meter.stage("s"):
        pass
    try:
        with meter.stage("s"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    meter.close()
    assert list(meter.stages) == ["s"] and meter.stages["s"]["wall_s"] >= 0
    assert meter._current is None


def _child_work(i):
    import time as _t
    x = np.ones(5_000_000)                                        # 40 MB private per child
    _t.sleep(0.6)
    return float(x.sum()) + i


def test_stage_meter_tree_pss_and_child_cpu_with_a_fork_pool():
    import multiprocessing as mp
    assert tm.child_pids() == [] and tm.pss_gb() > 0
    with tm.StageMeter(interval=0.05, tree_interval=0.1) as meter:
        with meter.stage("pool"):
            with mp.get_context("fork").Pool(3) as pool:
                pool.map(_child_work, range(3))
    rep = meter.report()
    mb = rep["memory_gb"]["pool"]
    assert "tree_pss_peak" in mb and mb["tree_pss_peak"] > 0.1      # 3 x 40 MB of private pages seen
    assert rep["peak_mem_gb"] >= mb["tree_pss_peak"]
    assert rep["cpu_s"]["pool"] >= 0
