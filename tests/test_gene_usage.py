"""Unit tests for :mod:`mir.basic.gene_usage` and :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`."""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trb_clonotypes(*vj_counts: tuple[str, str, int, int]) -> list[Clonotype]:
    """Build a list of TRB Clonotypes from (v, j, n_clones, dc_per_clone) tuples."""
    clones = []
    for v, j, n, dc in vj_counts:
        for i in range(n):
            clones.append(
                Clonotype(
                    sequence_id=f"{v}_{j}_{i}",
                    locus="TRB",
                    junction_aa=f"C{v}{j}{i}F",
                    v_gene=v,
                    j_gene=j,
                    duplicate_count=dc,
                )
            )
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

    def test_pseudocount_affects_fraction(self) -> None:
        clones = _make_trb_clonotypes(
            ("TRBV1", "TRBJ1-1", 10, 1),
            ("TRBV2", "TRBJ1-1", 10, 1),
        )
        rep = LocusRepertoire(clonotypes=clones, locus="TRB")
        gu = GeneUsage.from_repertoire(rep)

        # With zero pseudocount both pairs should be equal fractions
        frac0 = gu.vj_fraction("TRB", pseudocount=0)
        assert abs(frac0[("TRBV1", "TRBJ1-1")] - frac0[("TRBV2", "TRBJ1-1")]) < 1e-12

        # With pseudocount > 0 fractions should still be equal when counts equal
        frac1 = gu.vj_fraction("TRB", pseudocount=1)
        assert abs(frac1[("TRBV1", "TRBJ1-1")] - frac1[("TRBV2", "TRBJ1-1")]) < 1e-12


# ---------------------------------------------------------------------------
# TestPgenGeneUsageAdjustment
# ---------------------------------------------------------------------------

class TestPgenGeneUsageAdjustment:
    """Unit tests for :class:`PgenGeneUsageAdjustment`."""

    # Two common TRB V/J gene pairs (both present in OLGA TRB model)
    _V1, _J1 = "TRBV12-3", "TRBJ1-2"
    _V2, _J2 = "TRBV20-1", "TRBJ2-1"

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
        return PgenGeneUsageAdjustment(target_gu, cache_size=20_000, seed=42)

    def test_factor_positive_for_known_pair(self, adj: PgenGeneUsageAdjustment) -> None:
        f = adj.factor("TRB", self._V1, self._J1)
        assert f > 0

    def test_factor_small_for_absent_pair(self, adj: PgenGeneUsageAdjustment) -> None:
        # TRBV5-1/TRBJ2-7 is not in the target → factor should be < 1
        f_absent = adj.factor("TRB", "TRBV5-1", "TRBJ2-7")
        f_present = adj.factor("TRB", self._V1, self._J1)
        assert f_absent < f_present

    def test_factor_ratio_matches_target_ratio(
        self, target_gu: GeneUsage, adj: PgenGeneUsageAdjustment
    ) -> None:
        """factor(v1,j1) / factor(v2,j2) == target_frac(v1,j1) / target_frac(v2,j2)
        × [olga_frac(v2,j2) / olga_frac(v1,j1)].

        The V-J adjustment is: f = target_frac / olga_frac.
        Ratio of factors: f1/f2 = (t1/o1) / (t2/o2) = (t1*o2) / (t2*o1).
        """
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
        actual_ratio = f1 / f2
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10

    def test_weighted_count_ratio_matches_target(
        self, target_gu: GeneUsage, adj: PgenGeneUsageAdjustment
    ) -> None:
        """Importance-sampling property: ratio of weighted counts matches target ratio.

        For sequences generated by OLGA:
          weighted_count(v, j) = count(v, j) * factor(v, j)
        In expectation this equals n * target_frac(v, j).

        So ratio(v1,j1) / ratio(v2,j2) → target_frac(v1,j1) / target_frac(v2,j2).
        """
        model = OlgaModel(locus="TRB", seed=42)
        records = model.generate_sequences_with_meta(50_000, pgens=False, seed=42)

        weighted: dict[tuple, float] = defaultdict(float)
        for rec in records:
            v = rec["v_gene"].split("*")[0]
            j = rec["j_gene"].split("*")[0]
            weighted[(v, j)] += adj.factor("TRB", v, j)

        w1 = weighted.get((self._V1, self._J1), 0.0)
        w2 = weighted.get((self._V2, self._J2), 0.0)

        if w1 == 0 or w2 == 0:
            pytest.skip("One of the target V/J pairs not sampled — increase n")

        actual_ratio = w1 / w2

        target_frac = target_gu.vj_fraction("TRB")
        expected_ratio = target_frac[(self._V1, self._J1)] / target_frac[(self._V2, self._J2)]

        # Allow 60% deviation — finite-sample stochasticity with 50k samples
        # and a 20k OLGA cache makes this inherently noisy.  The exact
        # analytical relationship is verified in test_factor_ratio_matches_target_ratio.
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.60, (
            f"IS ratio {actual_ratio:.3f} deviates from expected {expected_ratio:.3f} "
            f"by > 60%"
        )

    def test_adjust_pgen_multiplies_by_factor(self, adj: PgenGeneUsageAdjustment) -> None:
        f = adj.factor("TRB", self._V1, self._J1)
        pgen_raw = 1e-10
        assert abs(adj.adjust_pgen("TRB", self._V1, self._J1, pgen_raw) - pgen_raw * f) < 1e-25

    def test_generate_with_meta_stores_adjusted_pgen(self, adj: PgenGeneUsageAdjustment) -> None:
        """When pgen_adjustment is provided, rec['pgen'] != log10(pgen_raw)."""
        model = OlgaModel(locus="TRB", seed=42)
        recs_raw = model.generate_sequences_with_meta(10, pgens=True, seed=42)
        recs_adj = model.generate_sequences_with_meta(10, pgens=True, seed=42,
                                                       pgen_adjustment=adj)
        # pgen_raw should be identical (same seed, same model)
        for r_raw, r_adj in zip(recs_raw, recs_adj):
            assert r_adj["pgen_raw"] == r_raw["pgen_raw"]
            # pgen (log10) should differ when factor != 1
            v = r_adj["v_gene"].split("*")[0]
            j = r_adj["j_gene"].split("*")[0]
            f = adj.factor("TRB", v, j)
            if f != 1.0 and r_raw["pgen_raw"] is not None and r_raw["pgen_raw"] > 0:
                expected_log_adj = math.log10(r_raw["pgen_raw"] * f) if r_raw["pgen_raw"] * f > 0 else float("-inf")
                assert abs(r_adj["pgen"] - expected_log_adj) < 1e-12
