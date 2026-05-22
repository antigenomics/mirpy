"""Unit tests for mir.distances.tcrdist."""

import numpy as np
import pytest

from mir.common.clonotype import Clonotype
from mir.common.gene_library import GeneLibrary
from mir.common.metaclonotype import MetaClonotypeDefinition
from mir.common.repertoire import LocusRepertoire
from mir.distances.aligner import GermlineAligner
from mir.distances.tcrdist import TcrDist, _MidGapScorer


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def germline_aligner():
    lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
    return GermlineAligner.from_library(lib, loci=["TRB"])


@pytest.fixture(scope="module")
def td(germline_aligner):
    return TcrDist(
        locus="TRB",
        species="human",
        germline_aligner=germline_aligner,
        fixed_gaps=(3, 4, -4, -3),
    )


@pytest.fixture
def influenza_clonotype():
    return Clonotype(
        sequence_id="flu1",
        v_gene="TRBV19*01",
        j_gene="TRBJ2-7*01",
        junction_aa="CASSIRSSYEQYF",
        duplicate_count=10,
    )


@pytest.fixture
def influenza_similar():
    return Clonotype(
        sequence_id="flu2",
        v_gene="TRBV19*01",
        j_gene="TRBJ2-7*01",
        junction_aa="CASSIRASYEQYF",
        duplicate_count=5,
    )


@pytest.fixture
def unrelated_clonotype():
    return Clonotype(
        sequence_id="unrelated",
        v_gene="TRBV5-1*01",
        j_gene="TRBJ1-2*01",
        junction_aa="CASSLGQGANVLTF",
        duplicate_count=3,
    )


@pytest.fixture
def small_repertoire(influenza_clonotype, influenza_similar, unrelated_clonotype):
    return LocusRepertoire(
        clonotypes=[influenza_clonotype, influenza_similar, unrelated_clonotype],
        locus="TRB",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestTcrDistConstruction:
    def test_from_library(self, td):
        assert td.locus == "TRB"
        assert td.species == "human"
        assert td.w_v == 1.0
        assert td.w_cdr3 == 3.0

    def test_cdrs_only_not_implemented(self, germline_aligner):
        with pytest.raises(NotImplementedError, match="cdrs_only"):
            TcrDist(
                locus="TRB",
                species="human",
                germline_aligner=germline_aligner,
                cdrs_only=True,
            )

    def test_invalid_v_alignment_type(self, germline_aligner):
        with pytest.raises(ValueError, match="v_alignment_type"):
            TcrDist(
                locus="TRB",
                species="human",
                germline_aligner=germline_aligner,
                v_alignment_type="cdrs_only",
            )

    def test_from_defaults(self):
        # Should complete without error; builds germline aligner internally
        td = TcrDist.from_defaults("TRB", "human", w_j=0.5)
        assert td.w_j == 0.5

    def test_fixed_gaps_none_uses_biopython(self, germline_aligner):
        from mir.distances.aligner import BioAlignerWrapper
        td = TcrDist(
            locus="TRB",
            species="human",
            germline_aligner=germline_aligner,
            fixed_gaps=None,
        )
        assert isinstance(td._cdr3, BioAlignerWrapper)

    def test_fixed_gaps_mid_uses_midgapscorer(self, germline_aligner):
        td = TcrDist(
            locus="TRB",
            species="human",
            germline_aligner=germline_aligner,
            fixed_gaps="Mid",
        )
        assert isinstance(td._cdr3, _MidGapScorer)

    def test_fixed_gaps_list_uses_junction_aligner(self, germline_aligner):
        from mir.distances.aligner import JunctionAligner
        td = TcrDist(
            locus="TRB",
            species="human",
            germline_aligner=germline_aligner,
            fixed_gaps=[3, 4, -4, -3],
        )
        assert isinstance(td._cdr3, JunctionAligner)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair distance
# ─────────────────────────────────────────────────────────────────────────────

class TestTcrDistPairwise:
    def test_self_distance_is_zero(self, td, influenza_clonotype):
        d = td.dist(influenza_clonotype, influenza_clonotype)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self, td, influenza_clonotype, influenza_similar):
        d12 = td.dist(influenza_clonotype, influenza_similar)
        d21 = td.dist(influenza_similar, influenza_clonotype)
        assert d12 == pytest.approx(d21, rel=1e-6)

    def test_non_negative(self, td, influenza_clonotype, unrelated_clonotype):
        d = td.dist(influenza_clonotype, unrelated_clonotype)
        assert d >= 0.0

    def test_similar_closer_than_unrelated(
        self, td, influenza_clonotype, influenza_similar, unrelated_clonotype
    ):
        d_sim = td.dist(influenza_clonotype, influenza_similar)
        d_unr = td.dist(influenza_clonotype, unrelated_clonotype)
        assert d_sim < d_unr

    def test_same_v_gene_zero_v_component(self, td, influenza_clonotype, influenza_similar):
        # Both have TRBV19*01, so V-gene distance should be 0
        d_v = td.germline_aligner.gene_dist(
            "TRB", influenza_clonotype.v_gene, influenza_similar.v_gene
        )
        assert d_v == pytest.approx(0.0, abs=1e-6)

    def test_different_v_gene_positive_distance(self, td, influenza_clonotype, unrelated_clonotype):
        d_v = td.germline_aligner.gene_dist(
            "TRB", influenza_clonotype.v_gene, unrelated_clonotype.v_gene
        )
        assert d_v > 0.0

    def test_w_v_zero_ignores_v(self, germline_aligner, influenza_clonotype, unrelated_clonotype):
        td_no_v = TcrDist(
            locus="TRB",
            species="human",
            germline_aligner=germline_aligner,
            w_v=0.0,
            w_cdr3=1.0,
        )
        td_no_cdr3 = TcrDist(
            locus="TRB",
            species="human",
            germline_aligner=germline_aligner,
            w_v=1.0,
            w_cdr3=0.0,
        )
        d_no_v = td_no_v.dist(influenza_clonotype, unrelated_clonotype)
        d_no_cdr3 = td_no_cdr3.dist(influenza_clonotype, unrelated_clonotype)
        d_both = TcrDist(
            locus="TRB",
            species="human",
            germline_aligner=germline_aligner,
            w_v=1.0,
            w_cdr3=1.0,
        ).dist(influenza_clonotype, unrelated_clonotype)
        assert d_both == pytest.approx(d_no_v + d_no_cdr3, rel=1e-6)

    def test_empty_junction_handled(self, td):
        cln1 = Clonotype(sequence_id="a", v_gene="TRBV19*01", junction_aa="")
        cln2 = Clonotype(sequence_id="b", v_gene="TRBV19*01", junction_aa="CASSIRSSYEQYF")
        d = td.dist(cln1, cln2)
        assert d >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Batch distance
# ─────────────────────────────────────────────────────────────────────────────

class TestTcrDistMatrix:
    def test_dist_matrix_shape(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        mat = td.dist_matrix(clns[:2], clns)
        assert mat.shape == (2, 3)
        assert mat.dtype == np.float64

    def test_self_dist_matrix_diagonal_zero(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        mat = td.self_dist_matrix(clns)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-6)

    def test_self_dist_matrix_symmetric(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        mat = td.self_dist_matrix(clns)
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_dist_one_to_many(self, td, influenza_clonotype, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        row = td.dist_one_to_many(influenza_clonotype, clns)
        assert row.shape == (3,)
        assert row[0] == pytest.approx(0.0, abs=1e-6)  # self

    def test_matrix_matches_pairwise(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        mat = td.dist_matrix(clns, clns)
        for i, c1 in enumerate(clns):
            for j, c2 in enumerate(clns):
                expected = td.dist(c1, c2)
                assert mat[i, j] == pytest.approx(expected, rel=1e-5)

    def test_parallel_matches_serial(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes) * 4  # expand for parallel chunking
        mat_s = td.dist_matrix(clns, clns, n_jobs=1)
        mat_p = td.dist_matrix(clns, clns, n_jobs=2)
        np.testing.assert_allclose(mat_s, mat_p, atol=1e-6)

    def test_empty_queries(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        mat = td.dist_matrix([], clns)
        assert mat.shape == (0, 3)

    def test_empty_refs(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        mat = td.dist_matrix(clns, [])
        assert mat.shape == (3, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Gap modes
# ─────────────────────────────────────────────────────────────────────────────

class TestGapModes:
    def _make_td(self, germline_aligner, fixed_gaps):
        return TcrDist(
            locus="TRB",
            species="human",
            germline_aligner=germline_aligner,
            w_v=0.0,
            w_cdr3=1.0,
            fixed_gaps=fixed_gaps,
            gap_penalty=-4.0,
        )

    def test_biopython_self_zero(self, germline_aligner):
        td = self._make_td(germline_aligner, None)
        s = "CASSIRSSYEQYF"
        d = td._cdr3.score_dist(s, s)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_midgap_self_zero(self, germline_aligner):
        td = self._make_td(germline_aligner, "Mid")
        s = "CASSIRSSYEQYF"
        d = td._cdr3.score_dist(s, s)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_midgap_symmetric(self, germline_aligner):
        td = self._make_td(germline_aligner, "Mid")
        s1, s2 = "CASSIRSSYEQYF", "CASSIRSYEQYF"
        d12 = td._cdr3.score_dist(s1, s2)
        d21 = td._cdr3.score_dist(s2, s1)
        assert d12 == pytest.approx(d21, rel=1e-6)

    def test_fixedgap_self_zero(self, germline_aligner):
        td = self._make_td(germline_aligner, [3, 4, -4, -3])
        s = "CASSIRSSYEQYF"
        d = td._cdr3.score_dist(s, s)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_all_modes_nonnegative(self, germline_aligner):
        s1, s2 = "CASSIRSSYEQYF", "CASSIRSYEQYF"
        for mode in [None, "Mid", [3, 4, -4, -3]]:
            td = self._make_td(germline_aligner, mode)
            d = td._cdr3.score_dist(s1, s2)
            assert d >= 0.0, f"Negative distance for gap mode {mode!r}"

    def test_modes_produce_reasonable_matrix(self, germline_aligner, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        for mode in [None, "Mid", [3, 4, -4, -3]]:
            td = self._make_td(germline_aligner, mode)
            mat = td.dist_matrix(clns, clns)
            assert np.all(np.diag(mat) < 1e-6), f"Non-zero diagonal for mode {mode!r}"
            assert np.all(mat >= 0), f"Negative entry for mode {mode!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Radius
# ─────────────────────────────────────────────────────────────────────────────

class TestRadius:
    def test_radius_shape(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        radii = td.compute_radius(clns[:2], clns)
        assert radii.shape == (2,)

    def test_radius_nonnegative(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        radii = td.compute_radius(clns, clns)
        assert np.all(radii >= 0)

    def test_self_median_zero(self, td, influenza_clonotype):
        # Median against itself is 0
        radii = td.compute_radius([influenza_clonotype], [influenza_clonotype], percentile=50)
        assert radii[0] == pytest.approx(0.0, abs=1e-6)

    def test_higher_percentile_larger(self, td, small_repertoire):
        clns = list(small_repertoire.clonotypes)
        r50 = td.compute_radius(clns[:1], clns, percentile=50)[0]
        r95 = td.compute_radius(clns[:1], clns, percentile=95)[0]
        assert r95 >= r50


# ─────────────────────────────────────────────────────────────────────────────
# Metaclonotypes
# ─────────────────────────────────────────────────────────────────────────────

class TestFindMetaclonotypes:
    def test_returns_metaclonotype_definition(self, td, small_repertoire):
        meta = td.find_metaclonotypes(
            small_repertoire, max_distance=500.0
        )
        assert isinstance(meta, MetaClonotypeDefinition)

    def test_tight_radius_only_self(self, td, small_repertoire):
        # At distance 0, each cluster should contain only its representative
        meta = td.find_metaclonotypes(
            small_repertoire, max_distance=0.0
        )
        for cluster_id in meta.cluster_ids:
            members = meta.members_of(cluster_id)
            assert len(members) == 1
            assert members["is_representative"][0]

    def test_wide_radius_captures_similar(self, td, small_repertoire, influenza_clonotype, influenza_similar):
        clns = list(small_repertoire.clonotypes)
        meta = td.find_metaclonotypes(
            small_repertoire,
            representative_ids=[influenza_clonotype.sequence_id],
            max_distance=500.0,
        )
        assert meta.n_clusters == 1
        members = meta.members_of(meta.cluster_ids[0])
        member_ids = set(members["clonotype_id"].to_list())
        assert influenza_clonotype.sequence_id in member_ids
        assert influenza_similar.sequence_id in member_ids

    def test_representative_marked(self, td, small_repertoire, influenza_clonotype):
        meta = td.find_metaclonotypes(
            small_repertoire,
            representative_ids=[influenza_clonotype.sequence_id],
            max_distance=500.0,
        )
        reps = meta.representatives()
        assert influenza_clonotype.sequence_id in reps["clonotype_id"].to_list()

    def test_match_v_gene_filters(self, td, small_repertoire, influenza_clonotype, unrelated_clonotype):
        meta = td.find_metaclonotypes(
            small_repertoire,
            representative_ids=[influenza_clonotype.sequence_id],
            max_distance=1e6,
            match_v_gene=True,
        )
        members = meta.members_of(meta.cluster_ids[0])
        member_ids = set(members["clonotype_id"].to_list())
        assert unrelated_clonotype.sequence_id not in member_ids

    def test_unknown_representative_id_skipped(self, td, small_repertoire):
        meta = td.find_metaclonotypes(
            small_repertoire,
            representative_ids=["nonexistent_id"],
            max_distance=500.0,
        )
        assert meta.n_clusters == 0

    def test_empty_repertoire(self, td, germline_aligner):
        rep = LocusRepertoire(clonotypes=[], locus="TRB")
        meta = td.find_metaclonotypes(rep, max_distance=100.0)
        assert meta.n_clusters == 0

    def test_n_jobs_consistency(self, td, small_repertoire):
        meta1 = td.find_metaclonotypes(small_repertoire, max_distance=300.0, n_jobs=1)
        meta2 = td.find_metaclonotypes(small_repertoire, max_distance=300.0, n_jobs=2)
        assert meta1.n_clusters == meta2.n_clusters


# ─────────────────────────────────────────────────────────────────────────────
# MidGapScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestMidGapScorer:
    @pytest.fixture
    def scorer(self):
        return _MidGapScorer(gap_penalty=-4.0)

    def test_equal_length_self_score(self, scorer):
        s = "CASSIRSSYEQYF"
        d = scorer.score_dist(s, s)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_unequal_length_nonneg(self, scorer):
        d = scorer.score_dist("CASSIRSSYEQYF", "CASSIRSYEQYF")
        assert d >= 0.0

    def test_symmetry(self, scorer):
        s1, s2 = "CASSIRSSYEQYF", "CASSIRSYEQYF"
        assert scorer.score_dist(s1, s2) == pytest.approx(scorer.score_dist(s2, s1), rel=1e-6)

    def test_self_cache(self, scorer):
        s = "CASSIRSSYEQYF"
        _ = scorer.score_dist(s, s)
        assert s in scorer._self_cache
