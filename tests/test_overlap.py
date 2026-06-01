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

import multiprocessing as mp
import os
import time

import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire
import math

from mir.comparative.overlap import (
    _AA20,
    compute_overlaps,
    count_overlap,
    expand_1mm,
    make_query_index,
    make_reference_keys,
    pairwise_overlap,
    pairwise_overlap_matrix,
    PairwiseOverlapResult,
)
import mir.comparative.overlap as overlap_mod
from tests.benchmark_helpers import many_vs_many_sample_overlap, many_vs_pool_sample_overlap
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
    junctions: list[str] | None = None,
) -> LocusRepertoire:
    clones = [
        Clonotype(
            sequence_id=str(i),
            locus=locus,
            junction_aa=jaa,
            junction=(junctions[i] if junctions is not None else ""),
            v_gene=v_gene,
            j_gene=j_gene,
            duplicate_count=i + 1,
            _validate=False,
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
        expected_count = 19 * 5 + 1  # 96 total: original + 19 subs per position
        assert len(variants) == expected_count, f"Expected {expected_count} variants for 5-char sequence, got {len(variants)}"

    def test_original_is_first(self) -> None:
        assert expand_1mm("CASS")[0] == "CASS"

    def test_single_char_covers_all_20aa(self) -> None:
        variants = expand_1mm("C")
        assert len(variants) == 20, f"Single char should generate 20 variants (all amino acids), got {len(variants)}"
        variant_set = set(variants)
        assert variant_set == set(_AA20), f"Variants don't match all 20 amino acids. Missing: {set(_AA20) - variant_set}"

    def test_no_duplicates(self) -> None:
        variants = expand_1mm("CASSF")
        unique_count = len(set(variants))
        assert len(variants) == unique_count, f"Found duplicate variants: {len(variants)} total vs {unique_count} unique"

    def test_all_same_length(self) -> None:
        seq = "CASSEGFTGELFF"
        variants = expand_1mm(seq)
        for v in variants:
            assert len(v) == len(seq), f"Variant {v!r} has length {len(v)}, expected {len(seq)}"

    def test_each_position_19_variants(self) -> None:
        seq = "CASS"
        variants = expand_1mm(seq)
        for i in range(len(seq)):
            at_pos = [v for v in variants[1:] if v[:i] == seq[:i] and v[i+1:] == seq[i+1:]]
            assert len(at_pos) == 19, f"Position {i} should have exactly 19 variants with different AA, got {len(at_pos)}"

    def test_empty_string(self) -> None:
        assert expand_1mm("") == [""]


# ---------------------------------------------------------------------------
# make_reference_keys
# ---------------------------------------------------------------------------

class TestMakeReferenceKeys:
    def test_basic_exact(self) -> None:
        keys = make_reference_keys(_make_rep(["CASSF", "CASSY"]))
        assert len(keys) == 2, f"Expected 2 reference keys for 2 sequences, got {len(keys)}"
        for key in keys:
            assert len(key) == 3, f"Each key should be a 3-tuple (locus, v_gene, junction_aa), got {type(key)}: {len(key)} elements"

    def test_deduplication(self) -> None:
        assert len(make_reference_keys(_make_rep(["CASSF", "CASSF"]))) == 1

    def test_allele_stripped(self) -> None:
        # Gene keys use bare gene names (allele stripped entirely) so any allele
        # of the same base gene maps to the same overlap key.
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
        from mir.common.alleles import strip_allele
        v = strip_allele(rep.clonotypes[0].v_gene)
        j = strip_allele(rep.clonotypes[0].j_gene)
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

    def test_trie_and_fallback_match(self, monkeypatch) -> None:
        ref = _make_rep(["CASSF", "CASSY"])
        query = _make_rep(["AASSF", "CASSY", "AASAF"])
        qi = make_query_index(query)
        compact = make_reference_keys(ref, allow_1mm=False)

        r_trie = count_overlap(compact, qi, allow_1mm=True)
        monkeypatch.setattr(overlap_mod, "Trie", None)
        r_fallback = count_overlap(compact, qi, allow_1mm=True)

        assert r_trie == r_fallback


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

    def test_parallel_process_pool_terminates_cleanly(self) -> None:
        ref_sets = [make_reference_keys(_make_rep([f"CASS{c}"])) for c in _AA20]
        qi = make_query_index(_make_rep([f"CASS{c}" for c in _AA20[:10]]))

        before = {p.pid for p in mp.active_children()}
        compute_overlaps(ref_sets, qi, n_jobs=2)

        # Give any newly spawned workers a chance to transition to terminal state.
        for p in mp.active_children():
            if p.pid not in before:
                p.join(timeout=0.5)

        lingering = [
            p for p in mp.active_children()
            if p.pid not in before and p.is_alive()
        ]
        assert not lingering, f"Found lingering overlap workers: {[p.pid for p in lingering]}"


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


    class TestPairwiseOverlapSpaces:
        """Sanity checks for overlap spaces and non-coding handling."""

        def test_exact_spaces_identical(self) -> None:
            rep = _make_rep(
                ["CASSF", "CASSY", "CASSW"],
                junctions=["TGTGCC", "TGTGCA", "TGTGCG"],
            )

            for overlap_space in ("ntvj", "nt", "aavj", "aa"):
                r = pairwise_overlap(rep, rep, overlap_space=overlap_space)
                assert r.n1_matched == 3
                assert r.n2_matched == 3
                assert r.jaccard == pytest.approx(1.0)

        def test_nt_spaces_forbid_mismatch(self) -> None:
            rep1 = _make_rep(["CASSF"], junctions=["TGTGCC"])
            rep2 = _make_rep(["CASSF"], junctions=["TGTGCC"])

            with pytest.raises(ValueError):
                pairwise_overlap(rep1, rep2, overlap_space="nt", metric="hamming", threshold=1)

            with pytest.raises(ValueError):
                pairwise_overlap(rep1, rep2, overlap_space="ntvj", metric="levenshtein", threshold=1)

        def test_noncoding_excluded_from_aa_overlap(self) -> None:
            rep1 = _make_rep(
                ["CASSF", "CAS*F", "CAS~F", "CAS_F"],
                junctions=["A", "B", "C", "D"],
            )
            rep2 = _make_rep(
                ["CASSF", "CAS*F", "CAS~F", "CAS_F"],
                junctions=["A", "B", "C", "D"],
            )

            r_aa = pairwise_overlap(rep1, rep2, overlap_space="aa")
            r_nt = pairwise_overlap(rep1, rep2, overlap_space="nt")

            assert r_aa.n1_matched == 1
            assert r_nt.n1_matched == 4

        def test_sample_vs_downsample_sanity(self) -> None:
            rep = _make_rep(
                ["CASSF", "CASSY", "CASSW", "CASRG", "CASST"],
                junctions=["AAA", "AAT", "AAC", "AAG", "ATA"],
            )
            downsample = LocusRepertoire(clonotypes=rep.clonotypes[:3], locus="TRB")

            for overlap_space in ("ntvj", "nt", "aavj", "aa"):
                r = pairwise_overlap(rep, downsample, overlap_space=overlap_space)
                assert r.n2_matched == 3
                assert r.n1_matched == 3

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
        max_s = float(os.getenv("MIRPY_BENCH_OVERLAP_EXACT_MAX_S", "0.001"))
        assert elapsed < max_s, f"expected < {max_s*1e3:.0f} ms, got {elapsed*1e3:.1f} ms"

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
        max_s = float(os.getenv("MIRPY_BENCH_OVERLAP_EXACT_MAX_S", "0.001"))
        assert elapsed < max_s, f"expected < {max_s*1e3:.0f} ms, got {elapsed*1e3:.1f} ms"

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
        max_s = float(os.getenv("MIRPY_BENCH_OVERLAP_1MM_MAX_S", "0.05"))
        assert elapsed < max_s, f"compact 1mm overlap took {elapsed*1e3:.1f}ms > {max_s*1e3:.0f}ms cap"

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


# ---------------------------------------------------------------------------
# pairwise_overlap — sanity checks
# ---------------------------------------------------------------------------

class TestPairwiseOverlapExact:
    """Sanity tests for :func:`pairwise_overlap` with exact matching."""

    def test_identical_reps_all_match(self) -> None:
        rep = _make_rep(["CASSF", "CASSY", "CASSW"], v_gene="TRBV12-3*01", j_gene="TRBJ2-2*01")
        r = pairwise_overlap(rep, rep)
        assert r.n1 == 3
        assert r.n1_matched == 3
        assert r.n2_matched == 3
        assert r.jaccard == pytest.approx(1.0)
        assert r.d_similarity == pytest.approx(1.0)
        assert r.f_similarity == pytest.approx(1.0)
        assert r.f2_similarity == pytest.approx(1.0)
        assert r.morisita_horn == pytest.approx(1.0)
        assert r.mode == "exact"
        assert not r.is_approximate

    def test_disjoint_reps_zero_overlap(self) -> None:
        rep1 = _make_rep(["CASSF"])
        rep2 = _make_rep(["CASSY"])
        r = pairwise_overlap(rep1, rep2)
        assert r.n1_matched == 0
        assert r.jaccard == 0.0
        assert r.d_similarity == 0.0
        assert r.f_similarity == 0.0
        assert r.morisita_horn == 0.0
        assert math.isnan(r.f2_similarity)

    def test_partial_overlap(self) -> None:
        rep1 = _make_rep(["CASSF", "CASSY"])  # dc = [1, 2]
        rep2 = _make_rep(["CASSF", "CASSW"])  # dc = [1, 2]
        r = pairwise_overlap(rep1, rep2)
        assert r.n1 == 2 and r.n2 == 2
        assert r.n1_matched == 1
        # Jaccard = 1 / (2 + 2 - 1) = 1/3
        assert r.jaccard == pytest.approx(1 / 3)
        # D = 1 / sqrt(2 * 2) = 0.5
        assert r.d_similarity == pytest.approx(0.5)
        assert r.f_similarity > 0.0

    def test_jaccard_bounds(self) -> None:
        rep1 = _make_rep(["CASSF", "CASSY", "CASSW"])
        rep2 = _make_rep(["CASSF"])
        r = pairwise_overlap(rep1, rep2)
        assert 0.0 <= r.jaccard <= 1.0
        assert 0.0 <= r.d_similarity <= 1.0

    def test_morisita_horn_range(self) -> None:
        rep1 = _make_rep(["CASSF", "CASSY"])
        rep2 = _make_rep(["CASSF", "CASSY"])
        r = pairwise_overlap(rep1, rep2)
        assert 0.0 <= r.morisita_horn <= 1.0

    def test_morisita_horn_identical(self) -> None:
        """MH = 1 for identical repertoires regardless of frequency distribution."""
        rep = _make_rep(["CASSF", "CASSY", "CASSW"])
        r = pairwise_overlap(rep, rep)
        assert r.morisita_horn == pytest.approx(1.0, abs=1e-9)

    def test_f2_leq_f_metric(self) -> None:
        """By Cauchy-Schwarz, F2 ≤ F when frequencies are unequal."""
        rep1 = _make_rep(["CASSF", "CASSY"])
        rep2 = _make_rep(["CASSF", "CASSY"])
        r = pairwise_overlap(rep1, rep2)
        # f2 = Σ sqrt(p_i * q_i) ≤ 1 in general; here reps are identical so
        # f2 should equal f_similarity (both = 1 for unit freq vectors).
        assert not math.isnan(r.f2_similarity)
        assert r.f2_similarity > 0

    def test_correlation_nan_for_single_clone(self) -> None:
        """Correlation requires ≥ 2 overlapping clones."""
        rep1 = _make_rep(["CASSF"])
        rep2 = _make_rep(["CASSF"])
        r = pairwise_overlap(rep1, rep2)
        assert math.isnan(r.correlation)

    def test_correlation_bounded(self) -> None:
        rep1 = _make_rep(["CASSF", "CASSY", "CASSW"])
        rep2 = _make_rep(["CASSF", "CASSY", "CASSW"])
        r = pairwise_overlap(rep1, rep2)
        assert not math.isnan(r.correlation)
        assert -1.0 <= r.correlation <= 1.0

    def test_symmetry(self) -> None:
        """Pairwise metrics should be symmetric: swap(rep1, rep2) = same result."""
        rep1 = _make_rep(["CASSF", "CASSY", "CASSW"])
        rep2 = _make_rep(["CASSF", "CASSZ"])
        r12 = pairwise_overlap(rep1, rep2)
        r21 = pairwise_overlap(rep2, rep1)
        assert r12.jaccard == pytest.approx(r21.jaccard)
        assert r12.d_similarity == pytest.approx(r21.d_similarity)
        assert r12.f_similarity == pytest.approx(r21.f_similarity)
        assert r12.morisita_horn == pytest.approx(r21.morisita_horn)

    def test_as_dict_keys(self) -> None:
        rep = _make_rep(["CASSF"])
        r = pairwise_overlap(rep, rep)
        d = r.as_dict()
        for key in ("jaccard", "d_similarity", "f_similarity", "morisita_horn",
                    "correlation", "f2_similarity", "mode", "is_approximate"):
            assert key in d

    def test_empty_rep_returns_zeros(self) -> None:
        empty = LocusRepertoire(clonotypes=[], locus="TRB")
        rep = _make_rep(["CASSF"])
        r = pairwise_overlap(empty, rep)
        assert r.n1 == 0
        assert r.jaccard == 0.0
        assert r.f_similarity == 0.0


class TestPairwiseOverlapApproximate:
    """Sanity tests for :func:`pairwise_overlap` with trie-based matching."""

    def test_hamming1_catches_1sub(self) -> None:
        rep1 = _make_rep(["CASSF"])  # query
        rep2 = _make_rep(["AASSF"])  # 1 substitution away
        r = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)
        assert r.n1_matched == 1
        assert r.is_approximate

    def test_hamming1_misses_2sub(self) -> None:
        rep1 = _make_rep(["CASSF"])
        rep2 = _make_rep(["AASAF"])  # 2 substitutions
        r = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)
        assert r.n1_matched == 0

    def test_levenshtein1_catches_indel(self) -> None:
        rep1 = _make_rep(["CASSF"])
        rep2 = _make_rep(["CASF"])  # 1 deletion
        r = pairwise_overlap(rep1, rep2, metric="levenshtein", threshold=1)
        assert r.n1_matched >= 1

    def test_approx_correlation_is_nan(self) -> None:
        rep1 = _make_rep(["CASSF", "CASSY"])
        rep2 = _make_rep(["AASSF", "AASSY"])
        r = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)
        assert math.isnan(r.correlation)

    def test_approx_f2_is_nan(self) -> None:
        rep1 = _make_rep(["CASSF"])
        rep2 = _make_rep(["AASSF"])
        r = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)
        assert math.isnan(r.f2_similarity)

    def test_approx_mode_string(self) -> None:
        rep = _make_rep(["CASSF"])
        r = pairwise_overlap(rep, rep, metric="hamming", threshold=1)
        assert r.mode == "hamming:1"

    def test_approx_metrics_in_range(self) -> None:
        rep1 = _make_rep(["CASSF", "CASSY"])
        rep2 = _make_rep(["AASSF", "AASSY"])
        r = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)
        assert 0.0 <= r.jaccard <= 1.0
        assert 0.0 <= r.d_similarity <= 1.0
        assert 0.0 <= r.f_similarity <= 1.0
        assert 0.0 <= r.morisita_horn

    def test_exact_leq_hamming1(self) -> None:
        """Approximate matching should find at least as many matches as exact."""
        rep1 = _make_rep(["CASSF", "CASSY", "CASSW"])
        rep2 = _make_rep(["CASSF", "CASSY", "AASSW"])
        r_exact = pairwise_overlap(rep1, rep2)
        r_approx = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)
        assert r_approx.n1_matched >= r_exact.n1_matched

    def test_threshold0_matches_exact(self) -> None:
        """threshold=0 with any metric should behave identically to exact."""
        rep1 = _make_rep(["CASSF", "CASSY"])
        rep2 = _make_rep(["CASSF", "CASSZ"])
        r_exact = pairwise_overlap(rep1, rep2)
        r_h0 = pairwise_overlap(rep1, rep2, metric="hamming", threshold=0)
        assert r_exact.n1_matched == r_h0.n1_matched
        assert r_exact.jaccard == pytest.approx(r_h0.jaccard)


class TestPairwiseOverlapMatrix:
    """Sanity tests for :func:`pairwise_overlap_matrix`."""

    def _reps(self):
        return [
            _make_rep(["CASSF", "CASSY"]),
            _make_rep(["CASSF", "CASSW"]),
            _make_rep(["CASSZ", "CASSX"]),
        ]

    def test_returns_dataframe(self) -> None:
        import pandas as pd
        reps = self._reps()
        df = pairwise_overlap_matrix(reps)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self) -> None:
        reps = self._reps()
        df = pairwise_overlap_matrix(reps)
        # 3 samples → 3 pairs
        assert len(df) == 3

    def test_has_metric_columns(self) -> None:
        reps = self._reps()
        df = pairwise_overlap_matrix(reps)
        for col in ("jaccard", "d_similarity", "f_similarity", "morisita_horn"):
            assert col in df.columns

    def test_sample_ids_used(self) -> None:
        reps = self._reps()
        ids = ["alpha", "beta", "gamma"]
        df = pairwise_overlap_matrix(reps, sample_ids=ids)
        assert set(df["sample_id_1"]) <= set(ids)
        assert set(df["sample_id_2"]) <= set(ids)

    def test_parallel_matches_serial(self) -> None:
        reps = self._reps()
        df_serial = pairwise_overlap_matrix(reps, n_jobs=1)
        df_parallel = pairwise_overlap_matrix(reps, n_jobs=2)
        for col in ("jaccard", "d_similarity", "f_similarity"):
            for v1, v2 in zip(df_serial[col], df_parallel[col]):
                if math.isnan(v1) and math.isnan(v2):
                    continue
                assert abs(v1 - v2) < 1e-10, f"{col}: {v1} vs {v2}"

    def test_too_few_reps_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            pairwise_overlap_matrix([_make_rep(["CASSF"])])

    def test_mismatched_ids_raises(self) -> None:
        reps = self._reps()
        with pytest.raises(ValueError, match="sample_ids length"):
            pairwise_overlap_matrix(reps, sample_ids=["a", "b"])


class TestManyVsManyHelpers:
    def _reps(self):
        return [
            _make_rep(["CASSF", "CASSY"]),
            _make_rep(["CASSF", "CASSW"]),
            _make_rep(["CASSZ", "CASSX"]),
        ]

    def test_many_vs_many_returns_dataframe(self) -> None:
        import pandas as pd

        reps = self._reps()
        df = many_vs_many_sample_overlap(reps, sample_ids=["a", "b", "c"], n_jobs=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "n_jobs_effective" in df.columns
        assert "qi_estimated_gb" in df.columns

    def test_many_vs_many_process_backend_uses_requested_jobs(self) -> None:
        reps = self._reps()
        df = many_vs_many_sample_overlap(
            reps,
            sample_ids=["a", "b", "c"],
            n_jobs=4,
        )
        assert (df["n_jobs_effective"] == 4).all()

    def test_many_vs_many_thread_backend_uses_requested_jobs(self) -> None:
        reps = self._reps()
        df = many_vs_many_sample_overlap(
            reps,
            sample_ids=["a", "b", "c"],
            n_jobs=4,
        )
        assert (df["n_jobs_effective"] == 4).all()

    def test_many_vs_many_thread_backend_approximate_runs(self) -> None:
        reps = self._reps()
        df = many_vs_many_sample_overlap(
            reps,
            sample_ids=["a", "b", "c"],
            metric="hamming",
            threshold=1,
            overlap_space="aavj",
            n_jobs=2,
        )
        assert len(df) == 3
        assert set(df["mode"]) == {"hamming:1"}

    def test_many_vs_pool_returns_dataframe(self) -> None:
        import pandas as pd

        reps = self._reps()
        pool_rep = LocusRepertoire(
            clonotypes=[c for rep in reps for c in rep.clonotypes],
            locus="TRB",
        )
        df = many_vs_pool_sample_overlap(
            reps,
            pool_rep,
            sample_ids=["a", "b", "c"],
            ages=[10, 20, 30],
            n_jobs=1,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "sample_id" in df.columns
        assert "age" in df.columns
