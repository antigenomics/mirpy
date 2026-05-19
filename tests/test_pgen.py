"""
Tests for OlgaModel across all nine built-in models.
- 100 sequences are generated to verify the sampler
- Exact Pgen is computed for 20 sequences (IGH exact pgen ~100 ms/seq)
- 1mm Pgen is computed for 5 sequences (len(seq) × exact cost)
Mean log10 Pgen for each model is printed for reference.
"""
import math
import os
import time

import pytest

from mir.basic.pgen import (
    McPgenPool,
    OlgaModel,
    _P_PRODUCTIVE_GENERIC,
    _P_PRODUCTIVE_TABLE,
    clear_mc_pool_cache,
    get_or_build_mc_pool,
    get_p_productive,
)
from tests.conftest import skip_benchmarks

ALL_MODELS = [
    ("TRA", "human"),
    ("TRB", "human"),
    ("TRG", "human"),
    ("TRD", "human"),
    ("IGH", "human"),
    ("IGK", "human"),
    ("IGL", "human"),
    ("TRA", "mouse"),
    ("TRB", "mouse"),
]


@pytest.fixture(scope="module", params=ALL_MODELS, ids=[f"{s}-{l}" for l, s in ALL_MODELS])
def olga_model(request):
    locus, species = request.param
    return locus, species, OlgaModel(locus=locus, species=species)


def test_seed_reproducibility():
    """Same seed must produce identical sequences; different seeds must differ."""
    model = OlgaModel(locus="TRB", species="human", seed=None)

    seqs_a = model.generate_sequences(20, seed=42)
    seqs_b = model.generate_sequences(20, seed=42)
    seqs_c = model.generate_sequences(20, seed=99)

    assert seqs_a == seqs_b, "same seed must yield the same sequences"
    assert seqs_a != seqs_c, "different seeds must yield different sequences"

    # generate_sequences_with_meta reproducibility
    meta_a = model.generate_sequences_with_meta(5, pgens=False, seed=7)
    meta_b = model.generate_sequences_with_meta(5, pgens=False, seed=7)
    assert [r["junction_aa"] for r in meta_a] == [r["junction_aa"] for r in meta_b]

    # generate_sequences_parallel reproducibility
    par_a = model.generate_sequences_parallel(20, n_jobs=2, seed=42)
    par_b = model.generate_sequences_parallel(20, n_jobs=2, seed=42)
    assert par_a == par_b, "same seed must yield the same parallel sequences"


def test_compute_usage_cache_parallel_branch(monkeypatch):
    """compute_usage_cache uses generate_pool when n_jobs > 1."""
    model = OlgaModel.__new__(OlgaModel)
    model._init_kwargs = {"locus": "TRB"}

    def _fail_if_called(*_args, **_kwargs):  # pragma: no cover
        raise AssertionError("generate_sequences_with_meta should not be used for n_jobs > 1")

    def _fake_generate_pool(*, n, n_jobs, seed):
        assert n == 4
        assert n_jobs == 3
        assert seed == 11
        return [
            {"v_gene": "TRBV20-1*01", "j_gene": "TRBJ2-7*01"},
            {"v_gene": "TRBV20-1*02", "j_gene": "TRBJ2-7*01"},
            {"v_gene": "TRBV5-1*01", "j_gene": "TRBJ1-2*01"},
            {"v_gene": "TRBV5-1*01", "j_gene": "TRBJ1-2*01"},
        ]

    monkeypatch.setattr(model, "generate_sequences_with_meta", _fail_if_called)
    monkeypatch.setattr(model, "generate_pool", _fake_generate_pool)

    gu = model.compute_usage_cache(n=4, seed=11, n_jobs=3)
    vj = gu.vj_usage("TRB")
    assert vj[("TRBV20-1", "TRBJ2-7")] == 2
    assert vj[("TRBV5-1", "TRBJ1-2")] == 2


def test_pgen_model(olga_model):
    locus, species, model = olga_model

    seqs = model.generate_sequences(100)
    assert len(seqs) == 100
    assert all(isinstance(s, str) and s for s in seqs), "empty or non-string sequence generated"

    # exact and 1mm Pgen on the same 5 sequences so 1mm >= exact is guaranteed per-sequence
    log_exact, log_1mm = [], []
    for s in seqs[:5]:
        p_exact = model.compute_pgen_junction_aa(s)
        p_1mm   = model.compute_pgen_junction_aa_1mm(s)
        assert p_exact is not None and p_exact >= 0, f"invalid exact Pgen for {s!r}"
        assert p_1mm   is not None and p_1mm   >= 0, f"invalid 1mm Pgen for {s!r}"
        assert p_1mm >= p_exact, f"1mm Pgen < exact Pgen for {s!r}"
        if p_exact > 0:
            log_exact.append(math.log10(p_exact))
        if p_1mm > 0:
            log_1mm.append(math.log10(p_1mm))

    mean_exact = sum(log_exact) / len(log_exact) if log_exact else float("-inf")
    mean_1mm   = sum(log_1mm)   / len(log_1mm)   if log_1mm   else float("-inf")

    print(f"\n{species} {locus}: exact={mean_exact:.2f}, 1mm={mean_1mm:.2f}")

    assert mean_exact > -25, f"mean log10 Pgen too low for {species} {locus}: {mean_exact}"


# ---------------------------------------------------------------------------
# McPgenPool unit tests
# ---------------------------------------------------------------------------

def test_get_p_productive_known_entries() -> None:
    """Calibrated p_productive values are within biologically plausible ranges."""
    assert 0.05 < get_p_productive("TRB", "human") < 0.50
    assert 0.05 < get_p_productive("TRA", "human") < 0.50
    assert 0.05 < get_p_productive("IGH", "human") < 0.50
    assert 0.05 < get_p_productive("TRB", "mouse") < 0.50


def test_get_p_productive_case_insensitive() -> None:
    assert get_p_productive("trb", "HUMAN") == get_p_productive("TRB", "human")


def test_get_p_productive_generic_fallback() -> None:
    p = get_p_productive("IGM", "zebrafish")
    assert p == _P_PRODUCTIVE_GENERIC
    assert 0.05 < p < 0.50


def test_p_productive_table_coverage() -> None:
    """Table must have entries for the 9 calibrated locus/species combinations."""
    expected = {
        ("TRA", "human"), ("TRB", "human"), ("TRG", "human"), ("TRD", "human"),
        ("IGH", "human"), ("IGK", "human"), ("IGL", "human"),
        ("TRA", "mouse"), ("TRB", "mouse"),
    }
    assert expected == set(_P_PRODUCTIVE_TABLE.keys())


def test_mc_pool_build_real_attributes() -> None:
    """McPgenPool.build_real sets n_total = len(sequences) and p_productive = 1.0."""
    seqs = ["CASSLGQETQYF", "CASSLGQETQYF", "CASSQGQETQYF", "CASSPGQETQYF"]
    pool = McPgenPool.build_real(seqs, locus="TRB", species="human", skip_ends=2)
    assert pool.n_productive == 4
    assert pool.n_total == 4
    assert pool.p_productive == pytest.approx(1.0)
    assert pool.locus == "TRB"
    assert pool.species == "human"


def test_mc_pool_pgen_exact_zero_for_unknown() -> None:
    pool = McPgenPool.build_real(["CASSLGQETQYF"], locus="TRB", species="human")
    assert pool.pgen_exact("NOTINPOOL") == pytest.approx(0.0)


def test_mc_pool_pgen_exact_frequency() -> None:
    """pgen_exact = count / n_total."""
    seqs = ["CASSLGQETQYF"] * 3 + ["CASSQGQETQYF"]
    pool = McPgenPool.build_real(seqs, locus="TRB", species="human")
    assert pool.pgen_exact("CASSLGQETQYF") == pytest.approx(3 / 4)
    assert pool.pgen_exact("CASSQGQETQYF") == pytest.approx(1 / 4)


def test_mc_pool_pgen_exact_bulk_matches_single() -> None:
    seqs = ["CASSLGQETQYF", "CASSQGQETQYF", "CASSPGQETQYF"]
    pool = McPgenPool.build_real(seqs, locus="TRB", species="human")
    bulk = pool.pgen_exact_bulk(seqs)
    for s, p in zip(seqs, bulk):
        assert p == pytest.approx(pool.pgen_exact(s))


def test_mc_pool_pgen_1mm_includes_exact() -> None:
    """pgen_1mm for a sequence in the pool must be >= pgen_exact."""
    seqs = ["CASSLGQETQYF", "CASSQGQETQYF"]
    pool = McPgenPool.build_real(seqs, locus="TRB", species="human", skip_ends=1)
    p1mm = pool.pgen_1mm("CASSLGQETQYF")
    p_exact = pool.pgen_exact("CASSLGQETQYF")
    assert p1mm >= p_exact, "1mm Pgen must be at least as large as exact Pgen"


def test_mc_pool_pgen_1mm_zero_for_unknown_no_neighbors() -> None:
    """Completely isolated sequence (no inner-1mm neighbors in pool) → pgen_1mm=0."""
    pool = McPgenPool.build_real(["CASSLGQETQYF"], locus="TRB", species="human")
    # "ZZZZZZZZZZZZ" is not in pool and has no Hamming-1 neighbors in pool
    assert pool.pgen_1mm("ZZZZZZZZZZZZ") == pytest.approx(0.0)


def test_mc_pool_skip_ends_excludes_terminal_mismatches() -> None:
    """Hamming-1 at terminal positions must be excluded with skip_ends=2."""
    # CASSLGQETQYF → AASSLGQETQYF: mismatch at position 0 (terminal) → excluded
    # CASSLGQETQYF → CASSLGQETQYY: mismatch at position 11 (terminal) → excluded
    pool = McPgenPool.build_real(
        ["AASSLGQETQYF", "CASSLGQETQYY"],
        locus="TRB", species="human", skip_ends=2,
    )
    p = pool.pgen_1mm("CASSLGQETQYF")
    # Both neighbors are at terminal positions → no inner-1mm contribution
    assert p == pytest.approx(0.0)


def test_mc_pool_pgen_1mm_includes_inner_mismatch() -> None:
    """An inner-position Hamming-1 neighbor must be counted."""
    # CASSLGQETQYF → CASSL_Q_ETQYF, mismatch at position 5 → inner (skip_ends=2, L=12)
    pool = McPgenPool.build_real(
        ["CASSAGQETQYF"],   # position 5: G→A
        locus="TRB", species="human", skip_ends=2,
    )
    p = pool.pgen_1mm("CASSLGQETQYF")
    assert p > 0.0, "Inner-1mm neighbor should contribute a non-zero pgen"


def test_mc_pool_generate_sequences_counted_denominator() -> None:
    """n_total > n_productive (non-productive events exist)."""
    model = OlgaModel(locus="TRB", species="human", seed=42)
    seqs, n_total = model.generate_sequences_counted(200, n_jobs=1, seed=42)
    model.close()
    assert len(seqs) == 200
    assert n_total > 200, "n_total must exceed productive count (non-productive rejections)"
    # p_productive ≈ 0.10–0.30 for TRB
    p_prod = len(seqs) / n_total
    assert 0.05 < p_prod < 0.50, f"Unexpected p_productive={p_prod:.3f}"


def test_mc_pool_generate_counted_parallel_correct_counts() -> None:
    """Parallel counted generation returns the requested number of sequences."""
    model = OlgaModel(locus="TRB", species="human", seed=42)
    seqs_s, n_s = model.generate_sequences_counted(100, n_jobs=1, seed=42)
    seqs_p, n_p = model.generate_sequences_counted(100, n_jobs=4, seed=42)
    model.close()
    assert len(seqs_s) == len(seqs_p) == 100
    # Both n_total must exceed productive count
    assert n_s > 100
    assert n_p > 100
    # Serial with same seed must be reproducible
    seqs_s2, n_s2 = model.generate_sequences_counted(100, n_jobs=1, seed=42)
    assert seqs_s2 == seqs_s and n_s2 == n_s, "Serial generation must be reproducible"


def test_get_or_build_mc_pool_caches() -> None:
    """get_or_build_mc_pool returns the same object on repeated calls."""
    clear_mc_pool_cache()
    p1 = get_or_build_mc_pool(locus="TRB", species="human", n=500, seed=1, skip_ends=2, n_jobs=1)
    p2 = get_or_build_mc_pool(locus="TRB", species="human", n=500, seed=1, skip_ends=2, n_jobs=1)
    assert p1 is p2, "Cache must return identical object on repeat call"
    clear_mc_pool_cache()


def test_mc_pool_pgen_1mm_bulk_empty() -> None:
    pool = McPgenPool.build_real(["CASSLGQETQYF"], locus="TRB", species="human")
    assert pool.pgen_1mm_bulk([]) == []


def test_bulk_pgen_parallel_matches_serial() -> None:
    model = OlgaModel(locus="TRB", species="human", seed=42)
    seqs = list(dict.fromkeys(model.generate_sequences(64, seed=123)))[:32]

    exact_serial = model.compute_pgen_junction_aa_bulk(seqs, max_mismatches=0, n_jobs=1)
    exact_parallel = model.compute_pgen_junction_aa_bulk(seqs, max_mismatches=0, n_jobs=4)
    one_mm_serial = model.compute_pgen_junction_aa_bulk(seqs[:8], max_mismatches=1, n_jobs=1)
    one_mm_parallel = model.compute_pgen_junction_aa_bulk(seqs[:8], max_mismatches=1, n_jobs=4)

    assert exact_parallel == exact_serial
    assert one_mm_parallel == one_mm_serial


# ---------------------------------------------------------------------------
# Parallel-generation benchmark (opt-in via RUN_BENCHMARK=1)
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestParallelGenerationBenchmark:
    """Compare 1-core vs 4-core generation throughput for TRB CDR3 sequences.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_pgen.py::TestParallelGenerationBenchmark
    """

    N = 10_000

    @pytest.fixture(scope="class")
    def trb_model(self):
        return OlgaModel(locus="TRB", species="human", seed=42)

    def test_single_core_generation(self, trb_model):
        t0 = time.perf_counter()
        seqs = trb_model.generate_sequences(self.N)
        elapsed = time.perf_counter() - t0

        assert len(seqs) == self.N
        assert all(isinstance(s, str) and s for s in seqs)
        rate = self.N / elapsed
        print(f"\n1-core: {self.N:,} seqs in {elapsed:.3f}s  ({rate:,.0f} seqs/s)")

    def test_parallel_4core_generation(self, trb_model):
        t0 = time.perf_counter()
        seqs = trb_model.generate_sequences_parallel(self.N, n_jobs=4, seed=42)
        elapsed = time.perf_counter() - t0

        assert len(seqs) == self.N
        assert all(isinstance(s, str) and s for s in seqs)
        rate = self.N / elapsed
        print(f"\n4-core: {self.N:,} seqs in {elapsed:.3f}s  ({rate:,.0f} seqs/s)")

    def test_speedup_report(self, trb_model):
        """Print 1-core vs 4-core wall-clock speedup for human-TRB generation."""
        # single core
        t1 = time.perf_counter()
        trb_model.generate_sequences(self.N)
        t_single = time.perf_counter() - t1

        # 4 cores
        t4 = time.perf_counter()
        trb_model.generate_sequences_parallel(self.N, n_jobs=4, seed=42)
        t_parallel = time.perf_counter() - t4

        speedup = t_single / t_parallel
        print(
            f"\nSpeedup: {speedup:.2f}x  "
            f"(1-core {t_single:.3f}s → 4-core {t_parallel:.3f}s, N={self.N:,})"
        )
        # Soft lower bound: parallel must not be catastrophically slower.
        # On macOS spawn-start adds ~1-2 s overhead; true speedup appears at N ≥ 100k.
        assert speedup > 0.1, f"4-core generation is >10x slower than 1-core ({speedup:.2f}x)"

    def test_pgen_pool_throughput_and_reuse(self, trb_model):
        """Benchmark persistent-pool Pgen throughput across two sequential calls.

        Verifies that the persistent pool is reused (second call is not slower
        than the first by more than 20 %) and that parallel throughput exceeds
        the minimum sustained rate.  tracemalloc is intentionally omitted to
        avoid the ~20x overhead it adds to OLGA's allocation-heavy computation.
        """
        seqs = trb_model.generate_sequences(300, seed=123)

        # First pass — may include pool creation for n_jobs=4.
        t0 = time.perf_counter()
        p1 = trb_model.compute_pgen_junction_aa_bulk(seqs, n_jobs=4)
        t_first = time.perf_counter() - t0

        # Second pass — pool already warm, should be at least as fast.
        t0 = time.perf_counter()
        p2 = trb_model.compute_pgen_junction_aa_bulk(seqs, n_jobs=4)
        t_second = time.perf_counter() - t0

        throughput = len(seqs) / t_second if t_second > 0 else float("inf")
        reuse_ratio = t_first / t_second if t_second > 0 else float("inf")
        print(
            f"\nPgen pool throughput: first={t_first:.3f}s second={t_second:.3f}s "
            f"reuse_ratio={reuse_ratio:.2f}x throughput={throughput:.0f} seqs/s"
        )

        import numpy as np
        np.testing.assert_allclose(p1, p2, rtol=1e-4, err_msg="Two identical pool runs diverged")
        # Second pass must not be slower than first by more than 20 % (pool reuse).
        assert reuse_ratio >= 0.8, f"Pool reuse regression: second pass {1/reuse_ratio:.2f}x slower than first"
        # Minimum sustained throughput at n_jobs=4.
        min_throughput = int(os.getenv("MIRPY_BENCH_PGEN_MIN_THROUGHPUT", "200"))
        assert throughput >= min_throughput, f"Pool throughput too low: {throughput:.0f} seqs/s (expected >= {min_throughput})"
