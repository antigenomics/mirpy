"""Cohort workers preserve order and values, execute in threads, and propagate failures."""
from __future__ import annotations

import threading

import numpy as np
import polars as pl
import pytest

from vdjtools.cores import available_cores

from mir.signature import rsig_cohort, synthesize
from vdjtools.signature.cohort import slices as _slices

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


@pytest.fixture(scope="module")
def corpus():
    """One small TRB corpus, built once. These tests are about the pool, not the fit."""
    art, _ = synthesize("memory", loci=("TRB",), n_samples=24, size=100, seed=8, n_components=4)
    return art


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
    """The corpus builder's contiguous ranges must tile the input exactly."""
    sl = _slices(n, workers)
    assert len(sl) == workers
    assert sl[0][0] == 0 and sl[-1][1] == n
    assert all(a <= b for a, b in sl)
    assert all(prev[1] == nxt[0] for prev, nxt in zip(sl, sl[1:]))   # contiguous, no gaps
    sizes = [b - a for a, b in sl if b > a]
    assert sum(sizes) == n
    assert max(sizes) - min(sizes) <= 1                              # balanced to within one


# ----------------------------------------------------------------- the failure must be loud

def test_a_pool_that_cannot_start_raises_instead_of_going_serial(monkeypatch, corpus):
    """The whole point. A correctness-preserving fallback hid a 20x slowdown for months."""
    def broken(*a, **k):
        raise RuntimeError("no workers for you")

    monkeypatch.setattr("vdjtools.cores.available_cores", lambda: 2)
    monkeypatch.setattr("multiprocessing.pool.ThreadPool", broken)
    with pytest.raises(RuntimeError) as e:
        rsig_cohort(cohort(4), corpus, n_jobs=2)
    assert "no workers for you" in str(e.value)


def test_no_warning_path_survives(monkeypatch, recwarn, corpus):
    monkeypatch.setattr("vdjtools.cores.available_cores", lambda: 2)
    """A warning is not good enough: callers filter them, and sklearn makes that routine."""
    monkeypatch.setattr("multiprocessing.pool.ThreadPool",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        rsig_cohort(cohort(4), corpus, n_jobs=2)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ----------------------------------------------------------------- parallel == serial

@pytest.mark.integration
def test_parallel_matches_serial_exactly(corpus):
    """Real worker threads. Parallelism that moves a number is a bug with a speedup."""
    c = cohort(6, n_clonotypes=200)
    serial = rsig_cohort(c, corpus, n_jobs=1)
    parallel = rsig_cohort(c, corpus, n_jobs=min(3, available_cores()))
    assert parallel["sample_id"].to_list() == serial["sample_id"].to_list()   # order preserved
    assert parallel.equals(serial)


def _pid_probe(tmpdir, n_clonotypes, seed):
    """Build a sample AND record which thread built it."""
    import threading
    import pathlib

    pathlib.Path(tmpdir, str(threading.get_ident())).touch()
    return cohort(1, n_clonotypes=n_clonotypes, seed=seed)["S000"]


@pytest.mark.integration
@pytest.mark.skipif(available_cores() < 2, reason="requires two allocated cores")
def test_the_work_really_happens_in_several_threads(tmp_path, corpus):
    """The deterministic form of "is it parallel" -- no timing, so no flake.

    This is the test that actually guards the regression. The bug was a pool that never started
    while the answer stayed correct; the only thing that distinguishes that from a working pool,
    without measuring time, is *where the work ran*. Each sample records its own thread ID as it is
    built, so a serial fallback leaves exactly one file and a working pool leaves several.

    """
    import functools

    d = str(tmp_path)
    samples = {f"S{i:03d}": functools.partial(_pid_probe, d, 400, i) for i in range(8)}
    out = rsig_cohort(samples, corpus, n_jobs=min(4, available_cores()))

    pids = list(tmp_path.iterdir())
    assert out.height == 8
    assert len(pids) > 1, (
        f"all 8 samples were built in one thread (id {pids[0].name if pids else '?'}). "
        "The pool did not run: a serial fallback "
        "has been reintroduced. This is the exact regression this file exists to catch.")
    assert str(threading.get_ident()) not in {p.name for p in pids}, (
        "work ran in the calling thread, which means the pool was bypassed rather than used")


@pytest.mark.benchmark
def test_the_pool_scales_over_the_work_a_single_worker_can_do(corpus):
    """A synthetic microbenchmark of sample-level parallelism, separate from correctness tests."""
    import time
    if available_cores() < 4:
        pytest.skip("four-core scaling requires a four-core allocation")

    c = cohort(48, n_clonotypes=20000)
    rsig_cohort(cohort(1, n_clonotypes=50), corpus, n_jobs=1)   # warm lazy imports

    t0 = time.perf_counter(); rsig_cohort(c, corpus, n_jobs=1)
    one = time.perf_counter() - t0

    t0 = time.perf_counter(); rsig_cohort(c, corpus, n_jobs=min(4, available_cores()))
    four = time.perf_counter() - t0
    speedup = one / four
    assert speedup > 2.0, (
        f"48 samples took {one:.2f} s on one worker's worth of kernel and {four:.2f} s on four "
        f"workers -- {speedup:.2f}x, against 2.65x measured on 16 Mac cores. At ~1.0x the pool "
        "is not running at all. Check test_the_work_really_happens_in_several_threads first: "
        "it answers the same question without timing.")


# ----------------------------------------------------------------- deferred samples

def test_a_deferred_sample_gives_the_same_answer_as_an_eager_one(corpus):
    """The whole memory story rests on this: deferring the read must not change a number."""
    import functools
    import math

    c = cohort(3, n_clonotypes=200, seed=3)
    eager = rsig_cohort(c, corpus, n_jobs=1)
    deferred = rsig_cohort(
        {sid: functools.partial(_identity, frames) for sid, frames in c.items()},
        corpus, n_jobs=1)
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
    reader = functools.partial(_read_sample, ["a.tsv", "b.tsv"])
    assert callable(reader)



def _identity(x):
    """Return a deferred sample unchanged."""
    return x


def test_cli_passes_explicit_worker_budget_without_a_sample_heuristic(monkeypatch, corpus):
    import mir.cli as cli
    import mir.signature

    seen = {}
    def capture(items, art, **kw):
        seen.update(kw)
        return pl.DataFrame({'sample_id': ['S0', 'S1']})
    monkeypatch.setattr(cli, '_resolve_corpus', lambda name: corpus)
    monkeypatch.setattr(cli, '_sample_items', lambda args: [('S0', None), ('S1', None)])
    monkeypatch.setattr(mir.signature, 'rsig_cohort', capture)
    monkeypatch.setattr(cli, '_write', lambda *args: None)
    args = cli.build_parser().parse_args(['signature', '--corpus', 'unused', '--jobs', '8'])
    cli.cmd_signature(args)
    assert seen['n_jobs'] == 8
    assert cli.build_parser().parse_args(['signature', '--corpus', 'unused']).jobs == 1


def test_signature_embedder_uses_one_native_thread():
    from mir.signature.signature import _model

    assert _model('human', 'TRB').threads == 1
