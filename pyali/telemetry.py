"""Per-stage wall / CPU / memory telemetry for a pipeline run (Linux ``/proc``, no psutil).

``StageMeter`` wraps each pipeline stage::

    with StageMeter(log=print) as meter:
        with meter.stage("load"):
            ...
    rep = meter.report()        # timing_s, cpu_s, memory_gb, peak_rss_gb, threads

A sampler thread polls this process's resident set size every ``interval`` seconds, so the
record of a stage carries its RSS *peak* (what a concurrent run must budget for) as well as the
RSS at its start and end (the difference is what the stage retained). ``cpu_s`` is process CPU
time over all threads, so ``cpu_s / wall_s`` is the stage's effective parallelism: a stage near
1.0 is single-threaded Python and gains nothing from more BLAS threads.

The numbers land in ``fov_metadata.json`` (see :func:`pyali.metadata.fov_metadata`) and in the
corpus manifest, which is how threads-per-field and fields-per-machine get chosen.
"""
import os
import threading
import time
from contextlib import contextmanager

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_gb(pid="self"):
    """Resident set size of ``pid`` (default: this process) in GB from ``/proc/<pid>/statm``;
    ``0.0`` once the process is gone or ``/proc`` is unavailable."""
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * _PAGE / 1e9
    except (OSError, ValueError, IndexError):
        return 0.0


def avail_ram_gb():
    """``MemAvailable`` from ``/proc/meminfo`` in GB (``inf`` when unreadable)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("inf")


def thread_info():
    """CPU count, ``OMP_NUM_THREADS`` and the BLAS thread pools in use (via ``threadpoolctl``
    when installed) -- the denominators for reading ``cpu_s / wall_s``."""
    info = dict(cpu_count=os.cpu_count(), omp_num_threads=os.environ.get("OMP_NUM_THREADS"))
    try:
        from threadpoolctl import threadpool_info
        info["blas"] = [dict(api=p.get("user_api"), library=p.get("internal_api"),
                             num_threads=p.get("num_threads"))
                        for p in threadpool_info()]
    except Exception:                                       # optional dependency
        info["blas"] = None
    return info


class StageMeter:
    """Record wall s, CPU s and RSS (start / end / peak) per named stage of this process."""

    def __init__(self, interval=0.25, log=None):
        self.interval = float(interval)
        self._log = log
        self.stages = {}                                     # name -> record, in call order
        self._current = None
        self._peak = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self._rss0 = rss_gb()
        self._global_peak = self._rss0
        self._thread = threading.Thread(target=self._sample, name="StageMeter", daemon=True)
        self._thread.start()

    # -- sampling ----------------------------------------------------------- #
    def _sample(self):
        while not self._stop.wait(self.interval):
            self._observe(rss_gb())

    def _observe(self, gb):
        with self._lock:
            self._global_peak = max(self._global_peak, gb)
            if self._current is not None:
                self._peak[self._current] = max(self._peak.get(self._current, 0.0), gb)

    # -- stages ------------------------------------------------------------- #
    @contextmanager
    def stage(self, name):
        """Time and meter the enclosed block as stage ``name`` (re-using a name overwrites)."""
        rss_start = rss_gb()
        with self._lock:
            self._current = name
            self._peak[name] = rss_start
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            wall, cpu = time.perf_counter() - t0, time.process_time() - c0
            rss_end = rss_gb()
            self._observe(rss_end)
            with self._lock:
                peak = max(self._peak.get(name, 0.0), rss_start, rss_end)
                self._current = None
            self.stages[name] = dict(wall_s=wall, cpu_s=cpu, rss_start_gb=rss_start,
                                     rss_end_gb=rss_end, rss_peak_gb=peak)
            if self._log:
                self._log(f"{name}: {wall:.1f} s, cpu {cpu:.0f} s ({cpu / max(wall, 1e-9):.1f}x), "
                          f"rss {peak:.2f} GB peak, {rss_end - rss_start:+.2f} GB retained")

    # -- lifecycle ---------------------------------------------------------- #
    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, 4 * self.interval))
        self._observe(rss_gb())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- output ------------------------------------------------------------- #
    def elapsed_s(self):
        return time.perf_counter() - self._t0

    def report(self):
        """``{timing_s, cpu_s, memory_gb, peak_rss_gb, rss_start_gb, threads}``; ``timing_s`` also
        carries ``total`` = seconds since the meter was created."""
        timing = {k: round(v["wall_s"], 3) for k, v in self.stages.items()}
        timing["total"] = round(self.elapsed_s(), 3)
        return dict(
            timing_s=timing,
            cpu_s={k: round(v["cpu_s"], 3) for k, v in self.stages.items()},
            memory_gb={k: dict(start=round(v["rss_start_gb"], 3), end=round(v["rss_end_gb"], 3),
                               peak=round(v["rss_peak_gb"], 3)) for k, v in self.stages.items()},
            peak_rss_gb=round(self._global_peak, 3),
            rss_start_gb=round(self._rss0, 3),
            threads=thread_info())
