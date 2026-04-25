"""Tests for :mod:`mir.comparative.overlap`.

Fast tests (always run)
-----------------------
* :class:`TestExpand1mm`      — unit tests for :func:`expand_1mm`.
* :class:`TestMakeReferenceKeys` — exact and 1mm key building.
* :class:`TestCountOverlap`   — overlap counting with exact and 1mm keys.

Benchmark tests (``RUN_BENCHMARK=1``)
--------------------------------------
* :class:`TestOverlapSpeed`   — exact vs allow_1mm key building and
  :func:`count_overlap` timing on a synthetic 500-clonotype reference and
  50 000-clonotype query.  Prints wall-clock times and speedup ratios.
"""

from __future__ import annotations

import time

import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire
from mir.comparative.overlap import (
    _AA20,
    count_overlap,
    expand_1mm,
    make_query_index,
    make_reference_keys,
)
from tests.conftest import skip_benchmarks


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

def _make_locus_rep(junction_aas: list[str], locus: str = "TRB") -> LocusRepertoire:
    """Build a minimal LocusRepertoire from a list of junction_aa strings."""
    clones = [
        Clonotype(
            sequence_id=str(i),
            locus=locus,
            junction_aa=jaa,
            v_gene=f"TRBV1*0{i % 5 + 1}",
            j_gene="TRBJ1-1*01",
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
        seq = "CASSF"
        variants = expand_1mm(seq)
        assert len(variants) == 19 * len(seq) + 1

    def test_original_is_first(self) -> None:
        seq = "CASS"
        assert expand_1mm(seq)[0] == seq

    def test_single_char(self) -> None:
        variants = expand_1mm("C")
        assert len(variants) == 20  # original + 19 substitutions
        assert variants[0] == "C"
        assert set(variants) == set(_AA20)

    def test_no_duplicates_in_output(self) -> None:
        variants = expand_1mm("CASSF")
        assert len(variants) == len(set(variants))

    def test_all_variants_same_length(self) -> None:
        seq = "CASSEGFTGELFF"
        for v in expand_1mm(seq):
            assert len(v) == len(seq)

    def test_each_position_has_19_variants(self) -> None:
        seq = "CASS"
        variants = expand_1mm(seq)
        for i in range(len(seq)):
            at_pos = [v for v in variants[1:] if v[:i] == seq[:i] and v[i+1:] == seq[i+1:]]
            assert len(at_pos) == 19, f"position {i} should have 19 variants, got {len(at_pos)}"


# ---------------------------------------------------------------------------
# make_reference_keys
# ---------------------------------------------------------------------------

class TestMakeReferenceKeys:
    def test_exact_basic(self) -> None:
        rep = _make_locus_rep(["CASSF", "CASSY"])
        keys = make_reference_keys(rep)
        assert len(keys) == 2
        assert all(isinstance(k, tuple) and len(k) == 3 for k in keys)

    def test_exact_deduplication(self) -> None:
        rep = _make_locus_rep(["CASSF", "CASSF"])
        keys = make_reference_keys(rep)
        assert len(keys) == 1

    def test_allele_stripped(self) -> None:
        clone = Clonotype(
            sequence_id="0", locus="TRB", junction_aa="CASSF",
            v_gene="TRBV1*02", j_gene="TRBJ1-1*01", duplicate_count=1,
        )
        rep = LocusRepertoire(clonotypes=[clone], locus="TRB")
        keys = make_reference_keys(rep)
        assert ("CASSF", "TRBV1", "TRBJ1-1") in keys

    def test_empty_junction_aa_skipped(self) -> None:
        clone = Clonotype(
            sequence_id="0", locus="TRB", junction_aa="",
            v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1,
        )
        rep = LocusRepertoire(clonotypes=[clone], locus="TRB")
        assert len(make_reference_keys(rep)) == 0

    def test_1mm_larger_than_exact(self) -> None:
        rep = _make_locus_rep(["CASSF"])
        exact = make_reference_keys(rep, allow_1mm=False)
        fuzzy = make_reference_keys(rep, allow_1mm=True)
        assert len(fuzzy) > len(exact)

    def test_1mm_contains_exact(self) -> None:
        rep = _make_locus_rep(["CASSF"])
        exact = make_reference_keys(rep, allow_1mm=False)
        fuzzy = make_reference_keys(rep, allow_1mm=True)
        assert exact.issubset(fuzzy)

    def test_1mm_size_upper_bound(self) -> None:
        jaa = "CASSF"
        rep = _make_locus_rep([jaa])
        fuzzy = make_reference_keys(rep, allow_1mm=True)
        # At most 19 * len + 1 per clonotype (may be less due to same v/j dedup)
        assert len(fuzzy) <= 19 * len(jaa) + 1

    def test_1mm_single_substitution_present(self) -> None:
        jaa = "CASSF"
        rep = _make_locus_rep([jaa])
        fuzzy = make_reference_keys(rep, allow_1mm=True)
        # "AASSF" is a valid single substitution (C→A at position 0)
        v = rep.clonotypes[0].v_gene.split("*")[0]
        j = rep.clonotypes[0].j_gene.split("*")[0]
        assert ("AASSF", v, j) in fuzzy


# ---------------------------------------------------------------------------
# count_overlap
# ---------------------------------------------------------------------------

class TestCountOverlap:
    def test_exact_match(self) -> None:
        ref = _make_locus_rep(["CASSF"])
        query = _make_locus_rep(["CASSF"])
        ref_keys = make_reference_keys(ref)
        qi = make_query_index(query)
        n, dc = count_overlap(ref_keys, qi)
        assert n == 1
        assert dc == query.clonotypes[0].duplicate_count

    def test_no_match(self) -> None:
        ref = _make_locus_rep(["CASSF"])
        query = _make_locus_rep(["CASSY"])
        n, dc = count_overlap(make_reference_keys(ref), make_query_index(query))
        assert n == 0
        assert dc == 0

    def test_1mm_catches_near_match(self) -> None:
        ref = _make_locus_rep(["CASSF"])
        query = _make_locus_rep(["AASSF"])  # 1 substitution away

        exact_keys = make_reference_keys(ref, allow_1mm=False)
        fuzzy_keys = make_reference_keys(ref, allow_1mm=True)
        qi = make_query_index(query)

        n_exact, _ = count_overlap(exact_keys, qi)
        n_fuzzy, _ = count_overlap(fuzzy_keys, qi)

        assert n_exact == 0
        assert n_fuzzy == 1

    def test_1mm_does_not_match_2mm(self) -> None:
        ref = _make_locus_rep(["CASSF"])
        # Two substitutions away
        query = _make_locus_rep(["AASAF"])
        fuzzy_keys = make_reference_keys(ref, allow_1mm=True)
        n, _ = count_overlap(fuzzy_keys, make_query_index(query))
        assert n == 0

    def test_empty_reference(self) -> None:
        ref = LocusRepertoire(clonotypes=[], locus="TRB")
        query = _make_locus_rep(["CASSF"])
        n, dc = count_overlap(make_reference_keys(ref), make_query_index(query))
        assert n == 0
        assert dc == 0

    def test_duplicate_count_summed(self) -> None:
        ref = _make_locus_rep(["CASSF"])
        # Two query entries with same junction_aa (unusual but valid)
        c1 = Clonotype(sequence_id="0", locus="TRB", junction_aa="CASSF",
                       v_gene="TRBV1*01", j_gene="TRBJ1-1*01", duplicate_count=3)
        c2 = Clonotype(sequence_id="1", locus="TRB", junction_aa="CASSF",
                       v_gene="TRBV1*01", j_gene="TRBJ1-1*01", duplicate_count=7)
        query = LocusRepertoire(clonotypes=[c1, c2], locus="TRB")
        _, dc = count_overlap(make_reference_keys(ref), make_query_index(query))
        assert dc == 10


# ---------------------------------------------------------------------------
# Benchmark — opt-in via RUN_BENCHMARK=1
# ---------------------------------------------------------------------------

def _make_synthetic_ref(n: int) -> LocusRepertoire:
    """Generate *n* distinct synthetic CDR3s by cycling through amino acids."""
    aas = list(_AA20)
    clones = []
    for i in range(n):
        # 15-mer: fixed flanks + 3 variable positions encoded from index
        a0 = aas[i % 20]
        a1 = aas[(i // 20) % 20]
        a2 = aas[(i // 400) % 20]
        jaa = f"CASS{a0}{a1}{a2}GELFF"
        clones.append(Clonotype(
            sequence_id=str(i), locus="TRB", junction_aa=jaa,
            v_gene="TRBV12-3*01", j_gene="TRBJ2-2*01", duplicate_count=1,
        ))
    return LocusRepertoire(clonotypes=clones, locus="TRB")


def _make_synthetic_query(n: int) -> LocusRepertoire:
    """Generate *n* distinct query clonotypes."""
    aas = list(_AA20)
    clones = []
    for i in range(n):
        a0 = aas[(i + 3) % 20]
        a1 = aas[((i + 3) // 20) % 20]
        a2 = aas[((i + 3) // 400) % 20]
        jaa = f"CASS{a0}{a1}{a2}GELFF"
        clones.append(Clonotype(
            sequence_id=str(i), locus="TRB", junction_aa=jaa,
            v_gene="TRBV12-3*01", j_gene="TRBJ2-2*01", duplicate_count=i + 1,
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

    @pytest.fixture(scope="class")
    def ref_rep(self):
        return _make_synthetic_ref(self.N_REF)

    @pytest.fixture(scope="class")
    def query_rep(self):
        return _make_synthetic_query(self.N_QUERY)

    @pytest.fixture(scope="class")
    def query_index(self, query_rep):
        return make_query_index(query_rep)

    def test_key_build_exact(self, ref_rep) -> None:
        t0 = time.perf_counter()
        keys = make_reference_keys(ref_rep, allow_1mm=False)
        elapsed = time.perf_counter() - t0
        print(f"\nexact  key build ({self.N_REF} clones): {len(keys):,} keys  {elapsed*1e3:.1f} ms")
        assert len(keys) == self.N_REF

    def test_key_build_1mm(self, ref_rep) -> None:
        t0 = time.perf_counter()
        keys = make_reference_keys(ref_rep, allow_1mm=True)
        elapsed = time.perf_counter() - t0
        print(f"\n1mm    key build ({self.N_REF} clones): {len(keys):,} keys  {elapsed*1e3:.1f} ms")
        assert len(keys) > self.N_REF

    def test_overlap_exact_speed(self, ref_rep, query_index) -> None:
        keys = make_reference_keys(ref_rep, allow_1mm=False)
        t0 = time.perf_counter()
        n, dc = count_overlap(keys, query_index)
        elapsed = time.perf_counter() - t0
        print(f"\nexact  overlap ({self.N_REF} ref × {self.N_QUERY} query): "
              f"n={n}  dc={dc}  {elapsed*1e3:.2f} ms")

    def test_overlap_1mm_speed(self, ref_rep, query_index) -> None:
        keys = make_reference_keys(ref_rep, allow_1mm=True)
        t0 = time.perf_counter()
        n, dc = count_overlap(keys, query_index)
        elapsed = time.perf_counter() - t0
        print(f"\n1mm    overlap ({self.N_REF} ref × {self.N_QUERY} query): "
              f"n={n}  dc={dc}  {elapsed*1e3:.2f} ms")

    def test_1mm_finds_more_matches(self, ref_rep, query_rep, query_index) -> None:
        exact_keys = make_reference_keys(ref_rep, allow_1mm=False)
        fuzzy_keys = make_reference_keys(ref_rep, allow_1mm=True)
        n_exact, _ = count_overlap(exact_keys, query_index)
        n_fuzzy, _ = count_overlap(fuzzy_keys, query_index)
        print(f"\nexact matches: {n_exact}  1mm matches: {n_fuzzy}  "
              f"gain: {n_fuzzy - n_exact}")
        assert n_fuzzy >= n_exact
