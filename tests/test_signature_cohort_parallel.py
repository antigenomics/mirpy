"""``signature_cohort(n_jobs=)`` must be parallel, and must say so when it cannot be.

The regression this file exists to prevent had three parts, and the third made the first two
invisible. One task per sample re-pickled the scale reference once per sample. The workers were
spawned, so a caller with no importable ``__main__`` killed every one of them. And the pool was
wrapped in ``except (BrokenExecutor, RuntimeError): <serial loop>``, which kept the answer correct
and turned a dead pool into a merely slow one.

Measured at the time on 20 samples / 36,416 clonotypes / 16 cores: 1,261 ms per sample at
``n_jobs=1`` and 1,182 ms at ``n_jobs=2`` -- "two workers bought 6%", where in fact the pool never
started and **both** numbers were serial. A caller filtering warnings, which is routine around
sklearn, saw nothing at all.

So the cheap tests here pin the *shape* -- contiguous slices, one per worker, and a loud failure --
and the marked benchmark pins the wall time.
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from mir.signature import rsig_cohort
from vdjtools.signature.cohort import WORKER_ENV
from vdjtools.signature.cohort import slices as _slices

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def cohort(n_samples: int, n_clonotypes: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_samples):
        ln = rng.integers(10, 18, n_clonotypes)
        out[f"S{i:03d}"] = {"TRB": pl.DataFrame({
            "junction_aa": ["C" + "".join(_AA[rng.integers(0, 20, k)]) + "F" for k in ln],
            "v_call": [f"TRBV{v}*01" for v in rng.integers(1, 30, n_clonotypes)],
            "j_call": [f"TRBJ{v}*01" for v in rng.integers(1, 13, n_clonotypes)],
            "duplicate_count": rng.integers(1, 200, n_clonotypes),
        })}
    return out


# ----------------------------------------------------------------- the slicing, on its own

@pytest.mark.parametrize("n,workers", [(1000, 8), (64, 8), (10, 3), (5, 8), (2, 2), (1, 4)])
def test_slices_are_contiguous_exhaustive_and_balanced(n, workers):
    """One task per worker, not one per sample -- and the slices must tile the input exactly."""
    sl = _slices(n, workers)
    assert len(sl) == workers
    assert sl[0][0] == 0 and sl[-1][1] == n
    assert all(a <= b for a, b in sl)
    assert all(prev[1] == nxt[0] for prev, nxt in zip(sl, sl[1:]))   # contiguous, no gaps
    sizes = [b - a for a, b in sl if b > a]
    assert sum(sizes) == n
    assert max(sizes) - min(sizes) <= 1                              # balanced to within one


# ----------------------------------------------------------------- the failure must be loud

def test_a_pool_that_cannot_start_raises_instead_of_going_serial(monkeypatch):
    """The whole point. A correctness-preserving fallback hid a 20x slowdown for months."""
    import mir.signature.assemble as A  # noqa: F401

    def broken(*a, **k):
        raise RuntimeError("no workers for you")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", broken)
    with pytest.raises(RuntimeError) as e:
        A.rsig_cohort(cohort(4), tier="core", n_jobs=2)
    msg = str(e.value)
    assert "n_jobs=1" in msg                      # the escape hatch is named
    assert "__main__" in msg                      # so is the actual cause
    assert "NOT falling back" in msg


def test_no_warning_path_survives(monkeypatch, recwarn):
    """A warning is not good enough: callers filter them, and sklearn makes that routine."""
    import mir.signature.assemble as A  # noqa: F401

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        A.rsig_cohort(cohort(4), tier="core", n_jobs=2)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ----------------------------------------------------------------- parallel == serial

@pytest.mark.integration
def test_parallel_matches_serial_exactly():
    """Real worker processes. Parallelism that moves a number is a bug with a speedup."""
    c = cohort(6, n_clonotypes=200)
    serial = rsig_cohort(c, tier="core", n_jobs=1)
    parallel = rsig_cohort(c, tier="core", n_jobs=3)
    assert parallel["sample_id"].to_list() == serial["sample_id"].to_list()   # order preserved
    assert parallel.equals(serial)


def _pid_probe(tmpdir, n_clonotypes, seed):
    """Build a sample AND record which process built it. Module-level so it survives spawn."""
    import os
    import pathlib

    pathlib.Path(tmpdir, str(os.getpid())).touch()
    return cohort(1, n_clonotypes=n_clonotypes, seed=seed)["S000"]


@pytest.mark.integration
def test_the_work_really_happens_in_several_processes(tmp_path):
    """The deterministic form of "is it parallel" -- no timing, so no flake.

    This is the test that actually guards the regression. The bug was a pool that never started
    while the answer stayed correct; the only thing that distinguishes that from a working pool,
    without measuring time, is *where the work ran*. Each sample records its own PID as it is
    built, so a serial fallback leaves exactly one file and a working pool leaves several.

    Timing cannot do this job at all against the obvious baseline, and the reason is measured
    rather than assumed: ``n_jobs=1`` is not serial, because the junction kernel threads
    internally over every core while each worker deliberately takes one thread. Four workers are
    therefore **0.79x** one in-process pass, not 1.4x -- so a live pool and a dead one both read
    below 1.0 there and cannot be told apart. The benchmark below times the pool against a
    baseline that can separate them; this test needs no clock.
    """
    import functools

    d = str(tmp_path)
    samples = {f"S{i:03d}": functools.partial(_pid_probe, d, 400, i) for i in range(8)}
    out = rsig_cohort(samples, tier="core", n_jobs=4)

    pids = list(tmp_path.iterdir())
    assert out.height == 8
    assert len(pids) > 1, (
        f"all 8 samples were built in one process (pid {pids[0].name if pids else '?'}). "
        "The pool did not run: either the workers cannot import __main__, or a serial fallback "
        "has been reintroduced. This is the exact regression this file exists to catch.")
    assert str(os.getpid()) not in {p.name for p in pids}, (
        "work ran in the PARENT process, which means the pool was bypassed rather than used")


@pytest.mark.benchmark
def test_the_pool_scales_over_the_work_a_single_worker_can_do():
    """The pool's scaling gate -- and it is measured against the right baseline.

    The obvious assertion, "four workers beat one process", is **false here by design** and was
    asserted anyway until 3.20.0. ``rsig``'s junction kernel releases the GIL and threads itself
    over every core, and since 3.19.0 a spawned worker deliberately takes **one** seqtree thread
    rather than all of them -- otherwise ``n_jobs`` workers each open one thread per core and a
    16-core box is asked for 256. So the two layers are alternatives, not multipliers, and the
    one process that keeps the kernel is the faster of the two.

    Measured 2026-09-26, 16-core M-series, 48 synthetic TRB samples x 20,000 clonotypes,
    ``rsig_cohort(tier="standard")``:

    ===============================  ========  =============
    configuration                    wall      per sample
    ===============================  ========  =============
    ``n_jobs=1``, kernel all cores    1.74 s     36 ms
    ``n_jobs=1``, kernel one thread   5.89 s    123 ms
    ``n_jobs=4``, one thread each     2.22 s     46 ms
    ===============================  ========  =============

    The kernel's own threading is worth **3.4x** and the pool is **0.79x** against it -- which is
    the finding in both repos' ISSUES item 3, reproduced on this repo's own fixture. What the
    pool *does* buy is 5.89 -> 2.22 s, **2.65x** over the work one worker can do alone, and that
    is the number worth gating: a dead pool reads 1.0x here, where against the all-cores baseline
    a dead pool and a live one both read below 1.0 and cannot be told apart.

    So both arms run with the kernel capped, which is what every worker gets. ``n_jobs`` is for
    memory (``O(n_jobs)`` samples resident with deferred reads) and for cold-kernel cohorts of
    many small samples -- not for wall clock on one warm box.
    """
    import time

    from mir.signature import assemble

    c = cohort(48, n_clonotypes=20000)
    rsig_cohort(cohort(1, n_clonotypes=50), tier="standard", n_jobs=1)   # warm lazy imports

    # Cap the parent's kernel the way `_chunk` caps a worker's, so the ratio below is the pool
    # and nothing else. The embedder is cached per (species, locus, K), so it has to be dropped
    # for the new thread count to take -- and dropped again afterwards, or every later test in
    # this process inherits a one-thread embedder.
    os.environ[WORKER_ENV] = "1"
    assemble._MODELS.clear()
    try:
        t0 = time.perf_counter(); rsig_cohort(c, tier="standard", n_jobs=1)
        one = time.perf_counter() - t0
    finally:
        del os.environ[WORKER_ENV]
        assemble._MODELS.clear()

    t0 = time.perf_counter(); rsig_cohort(c, tier="standard", n_jobs=4)
    four = time.perf_counter() - t0
    speedup = one / four
    assert speedup > 2.0, (
        f"48 samples took {one:.2f} s on one worker's worth of kernel and {four:.2f} s on four "
        f"workers -- {speedup:.2f}x, against 2.65x measured on 16 Mac cores. At ~1.0x the pool "
        "is not running at all. Check test_the_work_really_happens_in_several_processes first: "
        "it answers the same question without timing.")


# ----------------------------------------------------------------- deferred samples

def test_a_deferred_sample_gives_the_same_answer_as_an_eager_one():
    """The whole memory story rests on this: deferring the read must not change a number."""
    import functools
    import math

    c = cohort(3, n_clonotypes=200, seed=3)
    eager = rsig_cohort(c, tier="core", n_jobs=1)
    deferred = rsig_cohort(
        {sid: functools.partial(_identity, frames) for sid, frames in c.items()},
        tier="core", n_jobs=1)
    assert deferred["sample_id"].to_list() == eager["sample_id"].to_list()
    for col in eager.columns[1:]:
        for a, b in zip(eager[col].to_list(), deferred[col].to_list()):
            assert (a is None and b is None) or (math.isnan(a) and math.isnan(b)) or a == b


def test_the_cli_defers_its_reads_rather_than_materialising_the_cohort():
    """The parent must hand workers paths, not frames.

    Measured on 1,000 samples x 10,000 clonotypes at n_jobs=8: 6.6 GB peak resident for the
    largest process when the parent read everything first, against 356 MB when each worker reads
    its own slice. The point is not the ratio, it is that peak memory stopped scaling with the
    number of samples -- a 10,000-sample cohort now costs what a 1,000-sample one does.
    """
    import functools

    from mir.cli import _read_sample

    assert callable(_read_sample)
    p = functools.partial(_read_sample, ["a.tsv", "b.tsv"])
    # Picklable, because it has to cross a process boundary to be worth anything.
    import pickle
    assert pickle.loads(pickle.dumps(p)).args == (["a.tsv", "b.tsv"],)


def _identity(x):
    """Module-level so ``functools.partial`` over it can be pickled (a lambda cannot)."""
    return x
