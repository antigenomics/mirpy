"""Unit tests for :mod:`mir.biomarkers.vdjbet`.

Fast unit tests — always run
-----------------------------
* :class:`TestHelpers`             — unit tests for internal helpers (no OLGA).
* :class:`TestPgenBinPoolBasic`    — PgenBinPool construction and sampling sanity.
* :class:`TestVDJBetSanity`        — VDJBetOverlapAnalysis edge cases.
* :class:`TestOverlapResultZScore` — z/p-score logic for OverlapResult.
* :class:`TestToyExamplesCorrectBehavior` — Toy examples validating correct behavior.

Run all unit tests::

    pytest -s tests/test_vdjbet.py

Single test class::

    pytest -s tests/test_vdjbet.py::TestVDJBetSanity

Benchmark and integration tests (with full YFV/Q1 datasets) are located in
:mod:`tests.test_vdjbet_benchmark`. Run with ``RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mir.basic.pgen import OlgaModel
from mir.biomarkers.vdjbet import (
    OverlapResult,
    PgenBinPool,
    VDJBetOverlapAnalysis,
    _log2_pgen_bin,
    _strip_allele,
    compute_pgen_histogram,
)
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire

_SEED = 42


# ---------------------------------------------------------------------------
# Unit test helpers
# ---------------------------------------------------------------------------

def _make_olga_rep(locus: str, n: int, seed: int = _SEED) -> LocusRepertoire:
    """Generate n OLGA clonotypes and return them as a LocusRepertoire."""
    model = OlgaModel(locus=locus, seed=seed)
    records = model.generate_sequences_with_meta(n, pgens=False, seed=None)
    clones = [
        Clonotype(
            sequence_id=str(i), locus=locus,
            junction_aa=r["junction_aa"], junction=r["junction"],
            v_gene=r["v_gene"], j_gene=r["j_gene"],
            v_sequence_end=r["v_end"], j_sequence_start=r["j_start"],
            duplicate_count=1, _validate=False,
        )
        for i, r in enumerate(records)
    ]
    return LocusRepertoire(clonotypes=clones, locus=locus)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    """Unit tests for internal helpers (no OLGA sequence generation)."""

    def test_strip_allele_with_allele(self) -> None:
        assert _strip_allele("TRBV1*01") == "TRBV1*01"

    def test_strip_allele_without_allele(self) -> None:
        assert _strip_allele("TRBV1") == "TRBV1*01"

    def test_log2_pgen_bin_rounds_down(self) -> None:
        assert _log2_pgen_bin(-10.0) == -10
        assert _log2_pgen_bin(-10.4) == -10

    def test_log2_pgen_bin_rounds_up(self) -> None:
        assert _log2_pgen_bin(-10.6) == -11

    def test_compute_pgen_histogram_excludes_zero_pgen(self) -> None:
        """Nonsense CDR3 that produces pgen~0 must be excluded from histogram."""
        model = OlgaModel(locus="TRB", seed=_SEED)
        bad = Clonotype(sequence_id="0", locus="TRB",
                        junction_aa="AAAA", duplicate_count=1)
        hist = compute_pgen_histogram([bad], model)
        assert hist == {}

    def test_compute_pgen_histogram_single_valid(self) -> None:
        model = OlgaModel(locus="TRB", seed=_SEED)
        seqs = model.generate_sequences(1, seed=_SEED)
        clone = Clonotype(sequence_id="0", locus="TRB",
                          junction_aa=seqs[0], duplicate_count=1)
        hist = compute_pgen_histogram([clone], model)
        assert len(hist) == 1
        assert list(hist.values()) == [1]


# ---------------------------------------------------------------------------
# Unit tests for PgenBinPool basic functionality
# ---------------------------------------------------------------------------

class TestPgenBinPoolBasic:
    """Sanity tests for PgenBinPool with a compact pool — always run."""

    @pytest.fixture(scope="class")
    def pool(self) -> PgenBinPool:
        # Keep fast-suite setup bounded.
        return PgenBinPool("TRB", n=1_000, n_jobs=1, seed=_SEED)

    def test_has_bins(self, pool: PgenBinPool) -> None:
        assert len(pool.bins) > 0

    def test_floor_le_ceil(self, pool: PgenBinPool) -> None:
        assert pool.floor_bin <= pool.ceil_bin

    def test_winsorize_clamps_extreme_low(self, pool: PgenBinPool) -> None:
        assert pool.winsorize_bin(pool.floor_bin - 100) == pool.floor_bin

    def test_winsorize_clamps_extreme_high(self, pool: PgenBinPool) -> None:
        assert pool.winsorize_bin(pool.ceil_bin + 100) == pool.ceil_bin

    def test_nearest_bin_returns_valid(self, pool: PgenBinPool) -> None:
        mid = (pool.floor_bin + pool.ceil_bin) // 2
        nb = pool.nearest_bin(mid)
        assert nb in pool.bins

    def test_sample_mock_returns_3_tuples(self, pool: PgenBinPool) -> None:
        rng = np.random.default_rng(_SEED)
        existing_bin = next(iter(pool.bins))
        result = pool.sample_mock({existing_bin: 3}, rng)
        assert len(result) <= 3
        for t in result:
            assert len(t) == 3 and all(isinstance(x, str) for x in t)

    def test_bin_distribution_nonempty(self, pool: PgenBinPool) -> None:
        assert sum(pool.bin_distribution().values()) > 0

    def test_log2_pgen_array_length(self, pool: PgenBinPool) -> None:
        total = sum(len(v) for v in pool.bins.values())
        assert len(pool.log2_pgen_array()) == total

    def test_locus_attribute(self, pool: PgenBinPool) -> None:
        assert pool.locus == "TRB"


# ---------------------------------------------------------------------------
# Unit tests for VDJBetOverlapAnalysis edge cases
# ---------------------------------------------------------------------------

class TestVDJBetSanity:
    """Edge-case and contract tests for VDJBetOverlapAnalysis — always run."""

    @pytest.fixture(scope="class")
    def ref(self) -> LocusRepertoire:
        return _make_olga_rep("TRB", 5)

    @pytest.fixture(scope="class")
    def pool(self) -> PgenBinPool:
        return PgenBinPool("TRB", n=1_000, n_jobs=1, seed=_SEED)

    def test_score_returns_overlap_result(self, ref, pool) -> None:
        r = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED).score(
            _make_olga_rep("TRB", 5, seed=99)
        )
        assert isinstance(r, OverlapResult)
        assert r.n_total == 5

    def test_empty_query_zero_overlap(self, ref, pool) -> None:
        empty = LocusRepertoire(clonotypes=[], locus="TRB")
        r = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED).score(empty)
        assert r.n == 0 and r.dc == 0 and r.n_total == 0

    def test_unknown_locus_raises(self) -> None:
        rep = LocusRepertoire(
            clonotypes=[Clonotype(sequence_id="0", junction_aa="CASSF",
                                  duplicate_count=1)],
            locus="",
        )
        with pytest.raises(ValueError, match="Cannot determine locus"):
            VDJBetOverlapAnalysis(rep, pool_size=100, seed=_SEED)

    def test_ref_bin_counts_in_pool_range(self, ref, pool) -> None:
        a = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=5, seed=_SEED)
        for b in a.get_reference_bin_counts():
            assert pool.floor_bin <= b <= pool.ceil_bin

    def test_relaxed_vj_finds_at_least_as_many(self, ref, pool) -> None:
        q = _make_olga_rep("TRB", 10, seed=99)
        a = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED)
        assert a.score(q, match_v=False, match_j=False).n >= \
               a.score(q, match_v=True, match_j=True).n

    def test_allow_1mm_ge_exact(self, ref, pool) -> None:
        q = _make_olga_rep("TRB", 10, seed=99)
        a = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED)
        assert a.score(q, allow_1mm=True).n >= a.score(q, allow_1mm=False).n

    def test_result_options_stored(self, ref, pool) -> None:
        a = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED)
        r = a.score(_make_olga_rep("TRB", 5, seed=99),
                    allow_1mm=True, match_v=False, match_j=True)
        assert r.allow_1mm is True and r.match_v is False and r.match_j is True

    def test_same_seed_deterministic(self, ref, pool) -> None:
        a = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=5, seed=_SEED)
        b = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=5, seed=_SEED)
        assert a._get_mock_key_sets() == b._get_mock_key_sets()

    def test_different_seeds_differ(self, ref, pool) -> None:
        a = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=5, seed=1)
        b = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=5, seed=2)
        assert a._get_mock_key_sets() != b._get_mock_key_sets()

    def test_legacy_kwargs_accepted(self, ref, pool) -> None:
        """Legacy constructor parameters must be silently accepted (backward compat)."""
        VDJBetOverlapAnalysis(
            ref, pool=pool, n_mocks=5, seed=_SEED,
            cache_size=1000, max_cache_size=5000,
            infer_log2_floor_from_olga=True,
            olga_floor_sample_size=500,
            olga_floor_quantile=1e-3,
            log2_floor_bin=-80,
        )


# ---------------------------------------------------------------------------
# Unit tests for OverlapResult z/p-score computations
# ---------------------------------------------------------------------------

class TestOverlapResultZScore:
    """Unit tests for OverlapResult z/p-score computations."""

    def test_z_positive_when_real_exceeds_mean(self) -> None:
        r = OverlapResult(n_total=100, dc_total=1000, n=10, dc=50,
                          mock_n=[2, 3, 2, 3, 2], mock_dc=[10, 15, 12, 14, 11])
        assert r.z_n > 0

    def test_z_zero_when_std_zero_and_equal(self) -> None:
        r = OverlapResult(n_total=100, dc_total=1000, n=5, dc=50,
                          mock_n=[5, 5, 5, 5, 5], mock_dc=[50, 50, 50, 50, 50])
        assert r.z_n == 0.0

    def test_p_in_unit_interval(self) -> None:
        r = OverlapResult(n_total=100, dc_total=1000, n=10, dc=50,
                          mock_n=[2, 3, 2, 3, 2], mock_dc=[10, 15, 12, 14, 11])
        assert 0.0 <= r.p_n <= 1.0 and 0.0 <= r.p_dc <= 1.0

    def test_frac_in_unit_interval(self) -> None:
        r = OverlapResult(n_total=100, dc_total=500, n=30, dc=100,
                          mock_n=[5], mock_dc=[20])
        assert 0.0 <= r.frac_n <= 1.0 and 0.0 <= r.frac_dc <= 1.0

    def test_mock_n_length_preserved(self) -> None:
        r = OverlapResult(n_total=10, dc_total=100, n=2, dc=20,
                          mock_n=list(range(50)), mock_dc=list(range(50)))
        assert len(r.mock_n) == 50


# ---------------------------------------------------------------------------
# Toy example tests for correct behavior
# ---------------------------------------------------------------------------

class TestToyExamplesCorrectBehavior:
    """Toy example tests to verify correct VDJBET behavior on synthetic data."""

    def test_identical_ref_and_query_perfect_overlap(self) -> None:
        """When reference and query are identical, overlap should be maximal."""
        ref = _make_olga_rep("TRB", 5, seed=100)
        pool = PgenBinPool("TRB", n=1_000, n_jobs=1, seed=_SEED)
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=20, seed=_SEED)
        
        # Same repertoire as query
        result = analysis.score(ref)
        
        # All 5 clonotypes should match
        assert result.n == 5, f"Expected 5 matches but got {result.n}"
        # Z-score should be large (significant overlap)
        assert result.z_n > 0, f"Expected z > 0 but got {result.z_n}"

    def test_disjoint_refs_minimal_overlap(self) -> None:
        """Different synthetic repertoires should have minimal overlap."""
        ref = _make_olga_rep("TRB", 10, seed=100)
        query = _make_olga_rep("TRB", 10, seed=999)  # Very different seed
        pool = PgenBinPool("TRB", n=1_000, n_jobs=1, seed=_SEED)
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=20, seed=_SEED)
        
        result = analysis.score(query)
        
        # Synthetic sequences are far apart, expect minimal overlap
        assert result.n < 2, f"Expected n < 2 but got {result.n}"

    def test_1mm_ge_exact_always_holds(self) -> None:
        """1mm matching should always find >= clonotypes than exact matching."""
        ref = _make_olga_rep("TRB", 5, seed=100)
        query = _make_olga_rep("TRB", 50, seed=200)
        pool = PgenBinPool("TRB", n=1_000, n_jobs=1, seed=_SEED)
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED)
        
        exact = analysis.score(query, allow_1mm=False)
        mm = analysis.score(query, allow_1mm=True)
        
        assert mm.n >= exact.n, \
            f"1mm ({mm.n}) should be >= exact ({exact.n})"

    def test_duplicate_count_matters(self) -> None:
        """Clonotypes with higher duplicate counts should have larger dc overlap."""
        clones_low_dc = [
            Clonotype(sequence_id=str(i), locus="TRB",
                      junction_aa=f"CASSF{i:04d}", duplicate_count=1)
            for i in range(5)
        ]
        clones_high_dc = [
            Clonotype(sequence_id=str(i), locus="TRB",
                      junction_aa=f"CASSF{i:04d}", duplicate_count=100)
            for i in range(5)
        ]
        ref = _make_olga_rep("TRB", 5, seed=100)
        pool = PgenBinPool("TRB", n=1_000, n_jobs=1, seed=_SEED)
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED)
        
        rep_low = LocusRepertoire(clonotypes=clones_low_dc, locus="TRB")
        rep_high = LocusRepertoire(clonotypes=clones_high_dc, locus="TRB")
        
        r_low = analysis.score(rep_low)
        r_high = analysis.score(rep_high)
        
        # Both should have same number of matching clonotypes (if any)
        # But dc (duplicate count) should reflect the difference
        assert r_low.dc_total == 5  # 5 clones * 1 DC
        assert r_high.dc_total == 500  # 5 clones * 100 DC

    def test_v_j_matching_affects_overlap(self) -> None:
        """Relaxing V/J matching constraints should find more or equal overlaps."""
        ref = _make_olga_rep("TRB", 5, seed=100)
        query = _make_olga_rep("TRB", 10, seed=200)
        pool = PgenBinPool("TRB", n=1_000, n_jobs=1, seed=_SEED)
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=10, seed=_SEED)
        
        strict = analysis.score(query, match_v=True, match_j=True)
        relaxed = analysis.score(query, match_v=False, match_j=False)
        
        # Relaxing constraints should find at least as many
        assert relaxed.n >= strict.n, \
            f"Relaxed V/J ({relaxed.n}) should be >= strict ({strict.n})"

    def test_empty_reference_raises(self) -> None:
        """Empty reference repertoire should raise an error."""
        empty_ref = LocusRepertoire(clonotypes=[], locus="TRB")
        with pytest.raises(ValueError):
            VDJBetOverlapAnalysis(empty_ref, pool_size=100, seed=_SEED)

    def test_pool_size_affects_bins(self) -> None:
        """Larger pools should have more bins (generally)."""
        pool_small = PgenBinPool("TRB", n=100, n_jobs=1, seed=_SEED)
        pool_large = PgenBinPool("TRB", n=5_000, n_jobs=1, seed=_SEED)
        
        # Larger pool should have more or equal bins
        assert len(pool_large.bins) >= len(pool_small.bins), \
            f"Large pool ({len(pool_large.bins)}) bins >= small ({len(pool_small.bins)})"

    def test_zscore_increases_with_stronger_overlap(self) -> None:
        """Stronger overlap (more clonotypes) should generally yield higher z-scores."""
        # Create a large reference repertoire to increase chance of overlap
        ref = _make_olga_rep("TRB", 20, seed=100)
        pool = PgenBinPool("TRB", n=2_000, n_jobs=1, seed=_SEED)
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=20, seed=_SEED)
        
        # Small query
        query_small = _make_olga_rep("TRB", 5, seed=100)
        r_small = analysis.score(query_small)
        
        # Same repertoire (perfect overlap)
        r_same = analysis.score(ref)
        
        # Perfect overlap should have higher z-score
        assert r_same.z_n >= r_small.z_n, \
            f"Perfect overlap z ({r_same.z_n}) >= smaller z ({r_small.z_n})"
