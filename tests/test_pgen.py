"""
Tests for OlgaModel across all nine built-in models.
- 100 sequences are generated to verify the sampler
- Exact Pgen is computed for 20 sequences (IGH exact pgen ~100 ms/seq)
- 1mm Pgen is computed for 5 sequences (len(seq) × exact cost)
Mean log10 Pgen for each model is printed for reference.
"""
import math
import time

import pytest

from mir.basic.pgen import OlgaModel
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
