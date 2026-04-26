"""Tests for :mod:`mir.comparative.overlap`.

Fast tests (always run)
-----------------------
* :class:`TestExpand1mm`         — unit tests for :func:`expand_1mm`.
* :class:`TestMakeReferenceKeys` — exact and 1mm key building.
* :class:`TestCountOverlap`      — overlap counting: exact, 1mm pre-expanded
  reference, 1mm compact reference (allow_1mm=True).
* :class:`TestComputeOverlaps`   — batch / parallel :func:`compute_overlaps`.

Benchmark tests (``RUN_BENCHMARK=1``)
--------------------------------------
* :class:`TestOverlapSpeed` — exact vs allow_1mm timing on a synthetic
  500-clonotype reference and 50 000-clonotype query; also tests the
  parallel compute_overlaps path.  Prints wall-clock times.
"""

from __future__ import annotations

import time

import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire
from mir.comparative.overlap import (
    _AA20,
    compute_overlaps,
    count_overlap,
    expand_1mm,
    make_query_index,
    make_reference_keys,
)
from tests.conftest import skip_benchmarks


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

def _make_rep(
    junction_aas: list[str],
    *,
    v_gene: str = "TRBV12-3*01",
    j_gene: str = "TRBJ2-2*01",
    locus: str = "TRB",
) -> LocusRepertoire:
    clones = [
        Clonotype(
            sequence_id=str(i),
            locus=locus,
            junction_aa=jaa,
            v_gene=v_gene,
            j_gene=j_gene,
            duplicate_count=i + 1,
        )
        for i, jaa in enumerate(junction_aas)
    ]
    return LocusRepertoire(clonotypes=clones, locus=locus)


# ---------------------------------------------------------------------------
# expand_1mm
# ---------------------------------------------------------------------------

class TestExpand1mm:
    def test_count_exact(self) -> None:
        variants = expand_1mm("CASSF")
        assert len(variants) == 19 * 5 + 1  # 96

    def test_original_is_first(self) -> None:
        assert expand_1mm("CASS")[0] == "CASS"

    def test_single_char_covers_all_20aa(self) -> None:
        variants = expand_1mm("C")
        assert len(variants) == 20
        assert set(variants) == set(_AA20)

    def test_no_duplicates(self) -> None:
        variants = expand_1mm("CASSF")
        assert len(variants) == len(set(variants))

    def test_all_same_length(self) -> None:
        seq = "CASSEGFTGELFF"
        for v in expand_1mm(seq):
            assert len(v) == len(seq)

    def test_each_position_19_variants(self) -> None:
        seq = "CASS"
        variants = expand_1mm(seq)
        for i in range(len(seq)):
            at_pos = [v for v in variants[1:] if v[:i] == seq[:i] and v[i+1:] == seq[i+1:]]
            assert len(at_pos) == 19

    def test_empty_string(self) -> None:
        assert expand_1mm("") == [""]


# ---------------------------------------------------------------------------
# make_reference_keys
# ---------------------------------------------------------------------------

class TestMakeReferenceKeys:
    def test_basic_exact(self) -> None:
        keys = make_reference_keys(_make_rep(["CASSF", "CASSY"]))
        assert len(keys) == 2
        assert all(len(k) == 3 for k in keys)

    def test_deduplication(self) -> None:
        assert len(make_reference_keys(_make_rep(["CASSF", "CASSF"]))) == 1

    def test_allele_stripped(self) -> None:
        clone = Clonotype(
            sequence_id="0", locus="TRB", junction_aa="CASSF",
            v_gene="TRBV1*02", j_gene="TRBJ1-1*01", duplicate_count=1,
        )
        rep = LocusRepertoire(clonotypes=[clone], locus="TRB")
        assert ("CASSF", "TRBV1", "TRBJ1-1") in make_reference_keys(rep)

    def test_empty_junction_skipped(self) -> None:
        clone = Clonotype(
            sequence_id="0", locus="TRB", junction_aa="",
            v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1,
        )
        rep = LocusRepertoire(clonotypes=[clone], locus="TRB")
        assert len(make_reference_keys(rep)) == 0

    def test_1mm_strictly_larger_than_exact(self) -> None:
        rep = _make_rep(["CASSF"])
        exact = make_reference_keys(rep, allow_1mm=False)
        fuzzy = make_reference_keys(rep, allow_1mm=True)
        assert len(fuzzy) > len(exact)
        assert exact.issubset(fuzzy)

    def test_1mm_size_upper_bound(self) -> None:
        jaa = "CASSF"
        fuzzy = make_reference_keys(_make_rep([jaa]), allow_1mm=True)
        assert len(fuzzy) <= 19 * len(jaa) + 1

    def test_1mm_contains_substitution(self) -> None:
        jaa = "CASSF"
        rep = _make_rep([jaa])
        fuzzy = make_reference_keys(rep, allow_1mm=True)
        v = rep.clonotypes[0].v_gene.split("*")[0]
        j = rep.clonotypes[0].j_gene.split("*")[0]
        assert ("AASSF", v, j) in fuzzy   # C→A at position 0


# ---------------------------------------------------------------------------
# count_overlap — exact path
# ---------------------------------------------------------------------------

class TestCountOverlapExact:
    def test_exact_match(self) -> None:
        ref = _make_rep(["CASSF"])
        query = _make_rep(["CASSF"])
        r = count_overlap(make_reference_keys(ref), make_query_index(query))
        assert r.n == 1
        assert r.dc == query.clonotypes[0].duplicate_count

    def test_no_match(self) -> None:
        ref = _make_rep(["CASSF"])
        query = _make_rep(["CASSY"])
        r = count_overlap(make_reference_keys(ref), make_query_index(query))
        assert r.n == 0 and r.dc == 0

    def test_empty_reference(self) -> None:
        ref = LocusRepertoire(clonotypes=[], locus="TRB")
        r = count_overlap(make_reference_keys(ref), make_query_index(_make_rep(["CASSF"])))
        assert r.n == 0 and r.dc == 0

    def test_duplicate_count_summed(self) -> None:
        ref = _make_rep(["CASSF"])
        c1 = Clonotype(sequence_id="0", locus="TRB", junction_aa="CASSF",
                       v_gene="TRBV12-3*01", j_gene="TRBJ2-2*01", duplicate_count=3)
        c2 = Clonotype(sequence_id="1", locus="TRB", junction_aa="CASSF",
                       v_gene="TRBV12-3*01", j_gene="TRBJ2-2*01", duplicate_count=7)
        query = LocusRepertoire(clonotypes=[c1, c2], locus="TRB")
        r = count_overlap(make_reference_keys(ref), make_query_index(query))
        assert r.dc == 10


# ---------------------------------------------------------------------------
# count_overlap — pre-expanded reference (make_reference_keys allow_1mm=True)
# ---------------------------------------------------------------------------

class TestCountOverlapPreExpanded:
    """Use make_reference_keys(allow_1mm=True) → fast exact path in count_overlap."""

    def test_1mm_catches_near_match(self) -> None:
        ref = _make_rep(["CASSF"])
        query = _make_rep(["AASSF"])  # one substitution away
        fuzzy_keys = make_reference_keys(ref, allow_1mm=True)
        assert count_overlap(fuzzy_keys, make_query_index(query)).n == 1

    def test_2mm_not_matched(self) -> None:
        ref = _make_rep(["CASSF"])
        query = _make_rep(["AASAF"])  # two substitutions
        fuzzy_keys = make_reference_keys(ref, allow_1mm=True)
        assert count_overlap(fuzzy_keys, make_query_index(query)).n == 0

    def test_exact_still_matched(self) -> None:
        ref = _make_rep(["CASSF"])
        query = _make_rep(["CASSF"])
        fuzzy_keys = make_reference_keys(ref, allow_1mm=True)
        assert count_overlap(fuzzy_keys, make_query_index(query)).n == 1


# ---------------------------------------------------------------------------
# count_overlap — compact reference keys with allow_1mm=True (lazy expansion)
# ---------------------------------------------------------------------------

class TestCountOverlapLazy1mm:
    """Pass compact reference keys (no pre-expansion) with allow_1mm=True.
    Must produce identical results to the pre-expanded approach."""

    def test_catches_near_match(self) -> None:
        ref = _make_rep(["CASSF"])
        query = _make_rep(["AASSF"])
        compact = make_reference_keys(ref, allow_1mm=False)
        assert count_overlap(compact, make_query_index(query), allow_1mm=True).n == 1

    def test_2mm_not_matched(self) -> None:
        ref = _make_rep(["CASSF"])
        query = _make_rep(["AASAF"])
        compact = make_reference_keys(ref, allow_1mm=False)
        assert count_overlap(compact, make_query_index(query), allow_1mm=True).n == 0

    def test_no_double_count_multiple_ref(self) -> None:
        # Two ref entries both 1mm from the same query entry → count once.
        ref = _make_rep(["CASSF", "CASSY"])  # AASSF is 1mm from CASSF, not CASSY
        query = _make_rep(["AASSF"])
        compact = make_reference_keys(ref, allow_1mm=False)
        assert count_overlap(compact, make_query_index(query), allow_1mm=True).n == 1

    def test_same_result_as_pre_expanded(self) -> None:
        ref = _make_rep(["CASSF", "CASSY", "CASSYA"])
        query = _make_rep(["AASSF", "CASSY", "BASSY", "XXXXX"])
        qi = make_query_index(query)
        compact = make_reference_keys(ref, allow_1mm=False)
        pre_exp = make_reference_keys(ref, allow_1mm=True)
        r_lazy = count_overlap(compact, qi, allow_1mm=True)
        r_pre  = count_overlap(pre_exp, qi)
        assert r_lazy.n == r_pre.n
        assert r_lazy.dc == r_pre.dc


# ---------------------------------------------------------------------------
# compute_overlaps
# ---------------------------------------------------------------------------

class TestComputeOverlaps:
    def test_single_threaded_matches_loop(self) -> None:
        ref_sets = [
            make_reference_keys(_make_rep(["CASSF"])),
            make_reference_keys(_make_rep(["CASSY"])),
            make_reference_keys(_make_rep(["CASSF", "CASSY"])),
        ]
        qi = make_query_index(_make_rep(["CASSF", "CASSY", "XXXXX"]))
        expected = [count_overlap(k, qi) for k in ref_sets]
        assert compute_overlaps(ref_sets, qi) == expected

    def test_returns_correct_length(self) -> None:
        ref_sets = [make_reference_keys(_make_rep([f"CASS{i}"])) for i in range(10)]
        qi = make_query_index(_make_rep(["CASS0", "CASS5"]))
        results = compute_overlaps(ref_sets, qi)
        assert len(results) == 10

    def test_parallel_matches_sequential(self) -> None:
        ref_sets = [make_reference_keys(_make_rep([f"CASS{c}"])) for c in _AA20]
        qi = make_query_index(_make_rep([f"CASS{c}" for c in _AA20[:10]]))
        seq_results = compute_overlaps(ref_sets, qi, n_jobs=1)
        par_results = compute_overlaps(ref_sets, qi, n_jobs=2)
        assert seq_results == par_results

    def test_1mm_parallel_matches_sequential(self) -> None:
        ref_sets = [make_reference_keys(_make_rep([f"CASSF"])) for _ in range(4)]
        qi = make_query_index(_make_rep(["AASSF", "CASSF"]))
        seq = compute_overlaps(ref_sets, qi, allow_1mm=True, n_jobs=1)
        par = compute_overlaps(ref_sets, qi, allow_1mm=True, n_jobs=2)
        assert seq == par

    def test_empty_list(self) -> None:
        qi = make_query_index(_make_rep(["CASSF"]))
        assert compute_overlaps([], qi) == []


# ---------------------------------------------------------------------------
# Benchmark — opt-in via RUN_BENCHMARK=1
# ---------------------------------------------------------------------------

def _make_synthetic_rep(n: int, offset: int = 0) -> LocusRepertoire:
    """Generate *n* distinct synthetic TRB clonotypes cycling through AA20."""
    aas = list(_AA20)
    clones = []
    for i in range(n):
        a0 = aas[(i + offset) % 20]
        a1 = aas[((i + offset) // 20) % 20]
        a2 = aas[((i + offset) // 400) % 20]
        clones.append(Clonotype(
            sequence_id=str(i), locus="TRB",
            junction_aa=f"CASS{a0}{a1}{a2}GELFF",
            v_gene="TRBV12-3*01", j_gene="TRBJ2-2*01",
            duplicate_count=i + 1,
        ))
    return LocusRepertoire(clonotypes=clones, locus="TRB")


@skip_benchmarks
@pytest.mark.benchmark
class TestOverlapSpeed:
    """Speed comparison: exact vs allow_1mm key building and overlap counting.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_overlap.py::TestOverlapSpeed
    """

    N_REF   = 500
    N_QUERY = 50_000
    N_MOCKS = 1_000

    @pytest.fixture(scope="class")
    def ref_rep(self):
        return _make_synthetic_rep(self.N_REF)

    @pytest.fixture(scope="class")
    def query_rep(self):
        return _make_synthetic_rep(self.N_QUERY, offset=3)

    @pytest.fixture(scope="class")
    def query_index(self, query_rep):
        return make_query_index(query_rep)

    @pytest.fixture(scope="class")
    def mock_key_sets(self, ref_rep):
        """Build N_MOCKS compact key sets (each identical to ref for timing)."""
        base = make_reference_keys(ref_rep, allow_1mm=False)
        return [base for _ in range(self.N_MOCKS)]

    # --- key building ---

    def test_key_build_exact_under_1ms(self, ref_rep) -> None:
        t0 = time.perf_counter()
        keys = make_reference_keys(ref_rep, allow_1mm=False)
        elapsed = time.perf_counter() - t0
        print(f"\nexact  key build ({self.N_REF} clones): {len(keys):,} keys  {elapsed*1e3:.1f} ms")
        assert elapsed < 0.001, f"expected < 1 ms, got {elapsed*1e3:.1f} ms"

    def test_key_build_1mm_under_100ms(self, ref_rep) -> None:
        t0 = time.perf_counter()
        keys = make_reference_keys(ref_rep, allow_1mm=True)
        elapsed = time.perf_counter() - t0
        print(f"\n1mm    key build ({self.N_REF} clones): {len(keys):,} keys  {elapsed*1e3:.1f} ms")
        assert elapsed < 0.1, f"expected < 100 ms, got {elapsed*1e3:.1f} ms"

    # --- single overlap ---

    def test_overlap_exact_under_1ms(self, ref_rep, query_index) -> None:
        keys = make_reference_keys(ref_rep, allow_1mm=False)
        t0 = time.perf_counter()
        r = count_overlap(keys, query_index)
        elapsed = time.perf_counter() - t0
        print(f"\nexact  overlap ({self.N_REF} ref × {self.N_QUERY} query): "
              f"n={r.n}  dc={r.dc}  {elapsed*1e3:.2f} ms")
        assert elapsed < 0.001

    def test_overlap_1mm_pre_expanded_under_50ms(self, ref_rep, query_index) -> None:
        keys = make_reference_keys(ref_rep, allow_1mm=True)
        t0 = time.perf_counter()
        r = count_overlap(keys, query_index)
        elapsed = time.perf_counter() - t0
        print(f"\n1mm    overlap pre-exp ({self.N_REF} ref × {self.N_QUERY} query): "
              f"n={r.n}  dc={r.dc}  {elapsed*1e3:.2f} ms")
        assert elapsed < 0.05

    def test_overlap_1mm_compact_under_50ms(self, ref_rep, query_index) -> None:
        keys = make_reference_keys(ref_rep, allow_1mm=False)
        t0 = time.perf_counter()
        r = count_overlap(keys, query_index, allow_1mm=True)
        elapsed = time.perf_counter() - t0
        print(f"\n1mm    overlap compact ({self.N_REF} ref × {self.N_QUERY} query): "
              f"n={r.n}  dc={r.dc}  {elapsed*1e3:.2f} ms")
        assert elapsed < 0.05

    # --- 1mm finds more matches with strict lower bound ---

    def test_1mm_finds_at_least_50pct_more_matches(
        self, ref_rep, query_index
    ) -> None:
        exact_keys = make_reference_keys(ref_rep, allow_1mm=False)
        fuzzy_keys = make_reference_keys(ref_rep, allow_1mm=True)
        n_exact = count_overlap(exact_keys, query_index).n
        n_fuzzy = count_overlap(fuzzy_keys, query_index).n
        gain = n_fuzzy - n_exact
        print(f"\nexact={n_exact}  1mm={n_fuzzy}  gain={gain}  "
              f"ratio={n_fuzzy/n_exact:.2f}x")
        assert n_fuzzy >= 1.5 * n_exact, (
            f"1mm ({n_fuzzy}) should be ≥ 1.5× exact ({n_exact}); "
            f"got {n_fuzzy/n_exact:.2f}×"
        )

    # --- batch / parallel compute_overlaps ---

    def test_compute_overlaps_sequential_timing(
        self, mock_key_sets, query_index
    ) -> None:
        t0 = time.perf_counter()
        results = compute_overlaps(mock_key_sets, query_index, n_jobs=1)
        elapsed = time.perf_counter() - t0
        print(f"\ncompute_overlaps seq  {self.N_MOCKS} exact mocks: {elapsed:.3f}s")
        assert len(results) == self.N_MOCKS

    def test_compute_overlaps_parallel_correct(
        self, mock_key_sets, query_index
    ) -> None:
        seq = compute_overlaps(mock_key_sets[:20], query_index, n_jobs=1)
        par = compute_overlaps(mock_key_sets[:20], query_index, n_jobs=2)
        assert seq == par, "parallel results differ from sequential"
        print(f"\ncompute_overlaps parallel (n_jobs=2): correct")

    def test_compute_overlaps_1mm_timing(
        self, mock_key_sets, query_index
    ) -> None:
        t0 = time.perf_counter()
        results = compute_overlaps(mock_key_sets[:100], query_index,
                                   allow_1mm=True, n_jobs=1)
        elapsed = time.perf_counter() - t0
        print(f"\ncompute_overlaps 1mm  100 compact mocks: {elapsed:.3f}s  "
              f"({elapsed/100*1e3:.1f} ms/mock)")
        assert len(results) == 100
