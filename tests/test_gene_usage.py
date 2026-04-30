"""Unit tests for :mod:`mir.basic.gene_usage` and :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from tests.conftest import skip_benchmarks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _make_trb_clonotypes(*vj_counts: tuple[str, str, int, int]) -> list[Clonotype]:
    """Build TRB Clonotypes from (v, j, n_clones, dc_per_clone) tuples.

    Junction sequences: ``CASS<X>F`` where X cycles through the 20 standard
    amino-acid letters, yielding valid 6-residue CDR3s.
    """
    clones = []
    idx = 0
    for v, j, n, dc in vj_counts:
        for _ in range(n):
            junction_aa = "CASS" + _AA20[idx % len(_AA20)] + "F"
            clones.append(
                Clonotype(
                    sequence_id=str(idx),
                    locus="TRB",
                    junction_aa=junction_aa,
                    v_gene=v,
                    j_gene=j,
                    duplicate_count=dc,
                )
            )
            idx += 1
    return clones


# ---------------------------------------------------------------------------
# TestGeneUsage
# ---------------------------------------------------------------------------

class TestGeneUsage:
    """Unit tests for :class:`GeneUsage`."""

    def test_from_repertoire_counts(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 2, 5),
            ("TRBV2", "TRBJ1-2", 3, 2),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        assert gu.total("TRB", count="clonotypes") == 5
        assert gu.total("TRB", count="duplicates") == 2 * 5 + 3 * 2

        vj = gu.vj_usage("TRB")
        assert vj[("TRBV1", "TRBJ1-1")] == 2
        assert vj[("TRBV2", "TRBJ1-2")] == 3

    def test_allele_stripping(self) -> None:
        clones = _make_trb_clonotypes(("TRBV12-3*01", "TRBJ1-2*01", 2, 1))
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        vj = gu.vj_usage("TRB")
        assert ("TRBV12-3", "TRBJ1-2") in vj
        assert ("TRBV12-3*01", "TRBJ1-2*01") not in vj

    def test_duplicate_count_mode(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 1, 10),
            ("TRBV2", "TRBJ1-1", 1, 3),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        vj_dc = gu.vj_usage("TRB", count="duplicates")
        assert vj_dc[("TRBV1", "TRBJ1-1")] == 10
        assert vj_dc[("TRBV2", "TRBJ1-1")] == 3

    def test_marginal_v_usage(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 2, 1),
            ("TRBV1", "TRBJ1-2", 3, 1),
            ("TRBV2", "TRBJ1-1", 1, 1),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        v = gu.v_usage("TRB")
        assert v["TRBV1"] == 5
        assert v["TRBV2"] == 1
        assert sum(v.values()) == gu.total("TRB")

    def test_marginal_j_usage(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 2, 1),
            ("TRBV2", "TRBJ1-1", 1, 1),
            ("TRBV1", "TRBJ1-2", 3, 1),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        j = gu.j_usage("TRB")
        assert j["TRBJ1-1"] == 3
        assert j["TRBJ1-2"] == 3

    def test_vj_fraction_sums_to_one(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 3, 5),
            ("TRBV2", "TRBJ1-2", 2, 3),
            ("TRBV1", "TRBJ1-2", 1, 1),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        frac = gu.vj_fraction("TRB")
        assert abs(sum(frac.values()) - 1.0) < 1e-12

    def test_v_fraction_sums_to_one(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 3, 1),
            ("TRBV2", "TRBJ1-1", 2, 1),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)
        frac = gu.v_fraction("TRB")
        assert abs(sum(frac.values()) - 1.0) < 1e-12

    def test_from_list_accumulates(self) -> None:
        rep1 = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(("TRBV1", "TRBJ1-1", 2, 5)),
            locus="TRB",
        )
        rep2 = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(("TRBV1", "TRBJ1-1", 3, 2)),
            locus="TRB",
        )
        gu = GeneUsage.from_list([rep1, rep2])

        assert gu.total("TRB") == 5
        vj = gu.vj_usage("TRB")
        assert vj[("TRBV1", "TRBJ1-1")] == 5

    def test_empty_locus_returns_zero(self) -> None:
        gu = GeneUsage()
        assert gu.total("TRB") == 0
        assert gu.vj_usage("TRB") == {}
        assert gu.vj_fraction("TRB") == {}

    def test_loci_property(self) -> None:
        rep = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(("TRBV1", "TRBJ1-1", 1, 1)),
            locus="TRB",
        )
        gu = GeneUsage.from_repertoire(rep)
        assert "TRB" in gu.loci

    def test_pseudocount_equal_counts_stay_equal(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 10, 1),
            ("TRBV2", "TRBJ1-1", 10, 1),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        for pc in (0, 1, 5):
            frac = gu.vj_fraction("TRB", pseudocount=pc)
            assert abs(frac[("TRBV1", "TRBJ1-1")] - frac[("TRBV2", "TRBJ1-1")]) < 1e-12

    def test_count_mode_aliases(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 2, 7),
            ("TRBV2", "TRBJ1-1", 1, 3),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        assert gu.total("TRB", count="count_rearrangement") == 3
        assert gu.total("TRB", count="count_duplicates") == 17
        assert gu.v_usage("TRB", count="count_rearrangements")["TRBV1"] == 2
        assert gu.v_usage("TRB", count="duplicates")["TRBV1"] == 14

    def test_usage_comparison_uses_frequency_not_counts(self) -> None:
        target = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(
                ("TRBV1", "TRBJ1-1", 3, 1),
                ("TRBV2", "TRBJ1-2", 1, 1),
            ),
            locus="TRB",
        )
        ref = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(
                ("TRBV1", "TRBJ1-1", 1, 1),
                ("TRBV2", "TRBJ1-2", 3, 1),
            ),
            locus="TRB",
        )

        tgt_gu = GeneUsage.from_repertoire(target)
        ref_gu = GeneUsage.from_repertoire(ref)
        cmp_v = tgt_gu.usage_comparison(ref_gu, "TRB", scope="v", count="count_rearrangement", pseudocount=1.0)

        # with pseudocount=1 and 2 V genes in each set:
        # p_target(TRBV1) = (3+1)/(4+2)=4/6, p_ref(TRBV1)=(1+1)/(4+2)=2/6
        assert abs(cmp_v["TRBV1"]["p_self"] - (4.0 / 6.0)) < 1e-12
        assert abs(cmp_v["TRBV1"]["p_reference"] - (2.0 / 6.0)) < 1e-12
        assert abs(cmp_v["TRBV1"]["factor"] - 2.0) < 1e-12

    def test_usage_comparison_weighted_mode(self) -> None:
        target = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(
                ("TRBV1", "TRBJ1-1", 1, 10),
                ("TRBV2", "TRBJ1-2", 1, 2),
            ),
            locus="TRB",
        )
        ref = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(
                ("TRBV1", "TRBJ1-1", 1, 3),
                ("TRBV2", "TRBJ1-2", 1, 9),
            ),
            locus="TRB",
        )

        tgt_gu = GeneUsage.from_repertoire(target)
        ref_gu = GeneUsage.from_repertoire(ref)
        cmp_vj = tgt_gu.usage_comparison(ref_gu, "TRB", scope="vj", count="count_duplicates", pseudocount=1.0)

        # Target: totals 12, Ref: totals 12, 2 keys each.
        # p_target(v1j1)=(10+1)/(12+2)=11/14, p_ref(v1j1)=(3+1)/(12+2)=4/14
        assert abs(cmp_vj[("TRBV1", "TRBJ1-1")]["p_self"] - (11.0 / 14.0)) < 1e-12
        assert abs(cmp_vj[("TRBV1", "TRBJ1-1")]["p_reference"] - (4.0 / 14.0)) < 1e-12

    def test_usage_comparison_uses_union_keyspace_for_smoothing(self) -> None:
        target = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(("TRBV1", "TRBJ1-1", 4, 1)),
            locus="TRB",
        )
        ref = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(("TRBV2", "TRBJ1-2", 4, 1)),
            locus="TRB",
        )

        tgt_gu = GeneUsage.from_repertoire(target)
        ref_gu = GeneUsage.from_repertoire(ref)
        cmp_v = tgt_gu.usage_comparison(ref_gu, "TRB", scope="v", count="count_rearrangement", pseudocount=1.0)

        # Union key space has two V genes, so each denominator is (4 + 2*1) = 6.
        assert abs(cmp_v["TRBV1"]["p_self"] - (5.0 / 6.0)) < 1e-12
        assert abs(cmp_v["TRBV1"]["p_reference"] - (1.0 / 6.0)) < 1e-12
        assert abs(cmp_v["TRBV2"]["p_self"] - (1.0 / 6.0)) < 1e-12
        assert abs(cmp_v["TRBV2"]["p_reference"] - (5.0 / 6.0)) < 1e-12

    def test_correction_factors_strip_alleles(self) -> None:
        target = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(("TRBV1*01", "TRBJ1-1*01", 2, 1)),
            locus="TRB",
        )
        ref = LocusRepertoire(
            clonotypes=_make_trb_clonotypes(("TRBV1*02", "TRBJ1-1*02", 1, 1)),
            locus="TRB",
        )
        tgt_gu = GeneUsage.from_repertoire(target)
        ref_gu = GeneUsage.from_repertoire(ref)
        factors = tgt_gu.correction_factors(ref_gu, "TRB", scope="vj", count="count_rearrangement", pseudocount=1.0)
        assert ("TRBV1", "TRBJ1-1") in factors


# ---------------------------------------------------------------------------
# TestClonotypeValidation
# ---------------------------------------------------------------------------

class TestClonotypeValidation:
    """Tests that Clonotype rejects invalid junction_aa and inconsistent lengths."""

    def test_invalid_character_raises(self) -> None:
        with pytest.raises(ValueError, match="non-standard amino-acid"):
            Clonotype(junction_aa="CASS*F")

    def test_digit_raises(self) -> None:
        with pytest.raises(ValueError, match="non-standard amino-acid"):
            Clonotype(junction_aa="CASS1F")

    def test_gene_name_in_junction_raises(self) -> None:
        with pytest.raises(ValueError, match="non-standard amino-acid"):
            Clonotype(junction_aa="CTRBV1F")

    def test_stop_codon_raises(self) -> None:
        with pytest.raises(ValueError, match="non-standard amino-acid"):
            Clonotype(junction_aa="CASS*LAGQTLYF")

    def test_valid_junction_aa_accepted(self) -> None:
        c = Clonotype(junction_aa="CASSLAGQTLYF")
        assert c.junction_aa == "CASSLAGQTLYF"

    def test_empty_junction_aa_accepted(self) -> None:
        c = Clonotype(junction_aa="")
        assert c.junction_aa == ""

    def test_junction_length_mismatch_accepted(self) -> None:
        # Non-coding sequences may have junction_aa truncated at stop codon;
        # length inconsistency is not an error.
        c = Clonotype(junction="ATCATC", junction_aa="CASSF")
        assert c.junction_aa == "CASSF"

    def test_auto_translate(self) -> None:
        c = Clonotype(junction="ATCATCATC")  # 9 nt → 3 AA via translate_bidi
        assert len(c.junction_aa) == 3


# ---------------------------------------------------------------------------
# TestPgenGeneUsageAdjustment
# ---------------------------------------------------------------------------

# Most common VJ pairs in the OLGA TRB model (verified empirically):
#   TRBV20-1/TRBJ2-7 ≈ 1.4%,  TRBV5-1/TRBJ2-7 ≈ 1.1%
# These are used so that the IS property can be verified with < 5% error
# using a finite sample of 300k sequences and a 200k OLGA cache.

class TestPgenGeneUsageAdjustment:
    """Unit tests for :class:`PgenGeneUsageAdjustment`."""

    _V1, _J1 = "TRBV20-1", "TRBJ2-7"   # most common VJ pair in OLGA TRB
    _V2, _J2 = "TRBV5-1",  "TRBJ2-7"   # second most common
    _ABSENT_V, _ABSENT_J = "TRBV30", "TRBJ2-6"  # rare pair absent from target

    @pytest.fixture(scope="class")
    def target_gu(self) -> GeneUsage:
        """Target with 3:1 ratio between (V1,J1) and (V2,J2)."""
        clones = _make_trb_clonotypes(
            (self._V1, self._J1, 3, 1),
            (self._V2, self._J2, 1, 1),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        return GeneUsage.from_repertoire(rep)

    @pytest.fixture(scope="class")
    def adj(self, target_gu: GeneUsage) -> PgenGeneUsageAdjustment:
        # Keep default unit-test runtime small.
        return PgenGeneUsageAdjustment(target_gu, cache_size=25_000, seed=42)

    @pytest.fixture(scope="class")
    def adj_large(self, target_gu: GeneUsage) -> PgenGeneUsageAdjustment:
        # Large cache is used only for benchmark/statistical checks.
        return PgenGeneUsageAdjustment(target_gu, cache_size=200_000, seed=42)

    def test_factor_positive_for_known_pair(self, adj: PgenGeneUsageAdjustment) -> None:
        assert adj.factor("TRB", self._V1, self._J1) > 0

    def test_factor_ordering_follows_target_counts(self, adj: PgenGeneUsageAdjustment) -> None:
        # V1/J1 has 3 target clones vs V2/J2 with 1 → factor(V1,J1) > factor(V2,J2)
        f1 = adj.factor("TRB", self._V1, self._J1)
        f2 = adj.factor("TRB", self._V2, self._J2)
        assert f1 > f2

    def test_factor_ratio_matches_target_ratio(
        self, target_gu: GeneUsage, adj: PgenGeneUsageAdjustment
    ) -> None:
        """factor(v1,j1)/factor(v2,j2) equals (t1/o1)/(t2/o2) to machine precision."""
        f1 = adj.factor("TRB", self._V1, self._J1)
        f2 = adj.factor("TRB", self._V2, self._J2)
        assert f1 > 0 and f2 > 0

        olga = adj._get_olga_cache("TRB")
        olga_usage = olga.vj_usage("TRB")
        olga_total = olga.total("TRB")
        n_olga = len(olga_usage)
        o1 = (olga_usage.get((self._V1, self._J1), 0) + 1.0) / (olga_total + n_olga)
        o2 = (olga_usage.get((self._V2, self._J2), 0) + 1.0) / (olga_total + n_olga)

        target_usage = target_gu.vj_usage("TRB")
        target_total = target_gu.total("TRB")
        n_target = len(target_usage)
        t1 = (target_usage.get((self._V1, self._J1), 0) + 1.0) / (target_total + n_target)
        t2 = (target_usage.get((self._V2, self._J2), 0) + 1.0) / (target_total + n_target)

        expected_ratio = (t1 * o2) / (t2 * o1)
        assert abs(f1 / f2 - expected_ratio) / expected_ratio < 1e-10

    @skip_benchmarks
    @pytest.mark.benchmark
    def test_weighted_count_ratio_matches_target(
        self, target_gu: GeneUsage, adj_large: PgenGeneUsageAdjustment
    ) -> None:
        """Importance-sampling property: ratio of factor-weighted counts matches
        the target fraction ratio to < 5%.

        For n sequences generated by OLGA:
          weighted_count(v, j) = count(v, j) × factor(v, j)
        In expectation: weighted_count(v, j) = n × target_frac(v, j).
        So ratio(v1,j1)/ratio(v2,j2) → target_frac(v1,j1)/target_frac(v2,j2).

        Pairs TRBV20-1/TRBJ2-7 and TRBV5-1/TRBJ2-7 appear at ~1.4% and ~1.1%
        in the OLGA model.  With 300k generated sequences and a 200k cache the
        combined relative standard error is < 3%, well within the 5% tolerance.
        """
        model = OlgaModel(locus="TRB", seed=42)
        # pgens=False for speed — we only need V/J gene assignments
        records = model.generate_sequences_with_meta(300_000, pgens=False, seed=42)

        weighted: dict[tuple, float] = defaultdict(float)
        for rec in records:
            v = rec["v_gene"].split("*")[0]
            j = rec["j_gene"].split("*")[0]
            weighted[(v, j)] += adj_large.factor("TRB", v, j)

        w1 = weighted.get((self._V1, self._J1), 0.0)
        w2 = weighted.get((self._V2, self._J2), 0.0)

        assert w1 > 0 and w2 > 0, "Target VJ pairs must appear in 300k OLGA samples"

        actual_ratio   = w1 / w2
        target_frac    = target_gu.vj_fraction("TRB")
        expected_ratio = target_frac[(self._V1, self._J1)] / target_frac[(self._V2, self._J2)]

        rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
        assert rel_err < 0.05, (
            f"IS ratio {actual_ratio:.4f} deviates from expected {expected_ratio:.4f} "
            f"by {100*rel_err:.1f}% (> 5%)"
        )

    def test_adjust_pgen_multiplies_by_factor(self, adj: PgenGeneUsageAdjustment) -> None:
        f = adj.factor("TRB", self._V1, self._J1)
        pgen_raw = 1e-10
        assert abs(adj.adjust_pgen("TRB", self._V1, self._J1, pgen_raw) - pgen_raw * f) < 1e-25

    def test_generate_with_meta_stores_adjusted_pgen(
        self, adj: PgenGeneUsageAdjustment
    ) -> None:
        """rec['pgen'] == log10(pgen_raw * factor) for every generated sequence."""
        model = OlgaModel(locus="TRB", seed=42)
        recs_raw = model.generate_sequences_with_meta(20, pgens=True, seed=42)
        recs_adj = model.generate_sequences_with_meta(20, pgens=True, seed=42,
                                                       pgen_adjustment=adj)
        for r_raw, r_adj in zip(recs_raw, recs_adj):
            assert r_adj["pgen_raw"] == r_raw["pgen_raw"]
            v = r_adj["v_gene"].split("*")[0]
            j = r_adj["j_gene"].split("*")[0]
            f = adj.factor("TRB", v, j)
            p = r_raw["pgen_raw"]
            if p is not None and p > 0:
                expected = math.log10(p * f) if p * f > 0 else float("-inf")
                assert abs(r_adj["pgen"] - expected) < 1e-12
