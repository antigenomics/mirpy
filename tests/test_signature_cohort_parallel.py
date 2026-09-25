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

import numpy as np
import polars as pl
import pytest

from mir.signature.assemble import _slices, signature_cohort

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
    import mir.signature.assemble as A

    def broken(*a, **k):
        raise RuntimeError("no workers for you")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", broken)
    with pytest.raises(RuntimeError) as e:
        A.signature_cohort(cohort(4), tier="core", n_jobs=2)
    msg = str(e.value)
    assert "n_jobs=1" in msg                      # the escape hatch is named
    assert "__main__" in msg                      # so is the actual cause
    assert "NOT falling back" in msg


def test_no_warning_path_survives(monkeypatch, recwarn):
    """A warning is not good enough: callers filter them, and sklearn makes that routine."""
    import mir.signature.assemble as A

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        A.signature_cohort(cohort(4), tier="core", n_jobs=2)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ----------------------------------------------------------------- parallel == serial

@pytest.mark.integration
def test_parallel_matches_serial_exactly():
    """Real worker processes. Parallelism that moves a number is a bug with a speedup."""
    c = cohort(6, n_clonotypes=200)
    serial = signature_cohort(c, tier="core", n_jobs=1)
    parallel = signature_cohort(c, tier="core", n_jobs=3)
    assert parallel["sample_id"].to_list() == serial["sample_id"].to_list()   # order preserved
    assert parallel.equals(serial)


@pytest.mark.benchmark
def test_more_workers_is_actually_faster():
    """The check the old implementation would have failed, which is the point of having it.

    The bar is low on purpose, and the reason is not slack -- it is that ``n_jobs=1`` is **not
    serial**. The native Pgen batch threads internally, so one in-process pass over 20 samples
    measured 3.95 s wall against 25.39 s of CPU, about 6.4 cores busy. The machine is already
    largely saturated before a single worker is started, which is why the honest ceiling is ~2x at
    1,000 samples on 16 cores rather than 8x, and ~1.4x at the size below.

    What this test has to separate is therefore **1.0x from 1.4x**, not 4x from 8x. A pool that
    never runs returns 1.00x exactly -- measured, before the fix: 2.25 s at ``n_jobs=4`` against
    2.26 s at ``n_jobs=1`` on one input. 1.25x clears that cleanly and still leaves room for a
    loaded CI box with fewer cores.

    Worker startup is the other reason not to tighten it: each spawned worker pays a fresh
    interpreter plus a load of the frozen rotation and scale reference, about 1.3 s. A cohort
    small enough to be quick is dominated by it -- 32 samples of 2,000 clonotypes measured 1.09x,
    which says nothing at all. The cohort below is sized past that.

    For the shape of the real thing, see the 1,000-sample table in CHANGELOG 3.17.0.
    """
    import time

    c = cohort(48, n_clonotypes=5000)
    signature_cohort(cohort(1, n_clonotypes=50), tier="standard", n_jobs=1)   # warm lazy imports

    t0 = time.perf_counter(); signature_cohort(c, tier="standard", n_jobs=1); serial = time.perf_counter() - t0
    t0 = time.perf_counter(); signature_cohort(c, tier="standard", n_jobs=4); parallel = time.perf_counter() - t0
    speedup = serial / parallel
    assert speedup > 1.25, (
        f"48 samples took {serial:.2f} s in process and {parallel:.2f} s on four workers -- "
        f"{speedup:.2f}x. At ~1.0x the pool is not running at all; check that the workers can "
        "import __main__ and that nothing has reintroduced a serial fallback. Do NOT fix this by "
        "lowering the bar: 1.0x is the exact signature of the regression it exists to catch.")
