"""Unit tests for mir.embedding.tcremp and redesigned GermlineAligner.from_library.

By-hand reference values are computed using BLOSUM62 via BioAlignerWrapper
with the distance formula d(a,b) = s(a,a) + s(b,b) - 2*s(a,b).
"""

from __future__ import annotations

import numpy as np
import pytest

from mir.common.clonotype import Clonotype
from mir.common.gene_library import GeneLibrary
from mir.common.single_cell import PairedClonotype
from mir.distances.aligner import (
    BioAlignerWrapper,
    CDRAligner,
    GermlineAligner,
    Scoring,
)
from mir.embedding.prototypes import load_prototypes
from mir.embedding.tcremp import PairedTCREmp, TCREmp


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trb_library():
    return GeneLibrary.load_default(loci={"TRB"}, species={"human"})


@pytest.fixture(scope="module")
def trb_aligner(trb_library):
    return GermlineAligner.from_library(trb_library, loci=["TRB"])


@pytest.fixture(scope="module")
def tcremp_small():
    """TCREmp with 10 prototypes — fast, suitable for shape/value checks."""
    return TCREmp.from_defaults("human", "TRB", n_prototypes=10)


@pytest.fixture(scope="module")
def tcremp_medium():
    """TCREmp with 100 prototypes — used for symmetry/matrix checks."""
    return TCREmp.from_defaults("human", "TRB", n_prototypes=100)


@pytest.fixture(scope="module")
def paired_tcremp_small():
    """PairedTCREmp with 10 prototypes per chain for fast checks."""
    return PairedTCREmp.from_defaults("human", "TRA_TRB", n_prototypes=10)


def _clonotype(v, j, cdr3):
    return Clonotype(v_gene=v, j_gene=j, junction_aa=cdr3)


def _paired_clonotype(pair_id, tra, trb):
    return PairedClonotype(pair_id=pair_id, clonotype1=tra, clonotype2=trb)


# ---------------------------------------------------------------------------
# Scoring.score_dist base-class default
# ---------------------------------------------------------------------------

class TestScoringBaseScoreDist:
    """score_dist is now defined on Scoring base; BioAlignerWrapper inherits it."""

    def test_biopython_score_dist_self_is_zero(self):
        w = BioAlignerWrapper()
        for s in ["CASSLETGE", "CASSRGTGE", "ACDEF"]:
            assert w.score_dist(s, s) == pytest.approx(0.0, abs=1e-6)

    def test_biopython_score_dist_symmetric(self):
        w = BioAlignerWrapper()
        s1, s2 = "CASSLETGE", "CASSLRTGE"
        assert w.score_dist(s1, s2) == pytest.approx(w.score_dist(s2, s1), abs=1e-6)

    def test_biopython_score_dist_nonneg(self):
        w = BioAlignerWrapper()
        s1, s2 = "CASSLETGE", "CASSLRTGE"
        assert w.score_dist(s1, s2) >= 0.0


# ---------------------------------------------------------------------------
# GermlineAligner.from_library — construction and basic properties
# ---------------------------------------------------------------------------

class TestGermlineAlignerFromLibrary:

    def test_builds_without_error(self, trb_library):
        ga = GermlineAligner.from_library(trb_library, loci=["TRB"])
        assert len(ga._locus_dist) > 0

    def test_entry_count(self, trb_library, trb_aligner):
        # 89 V + 15 J = 104 genes; distances = 89² + 15² = 8146
        assert len(trb_aligner._locus_dist) == 89 ** 2 + 15 ** 2

    def test_self_distance_is_zero(self, trb_aligner):
        for gene in ["TRBV10-1*01", "TRBV1*01", "TRBJ1-1*01", "TRBJ2-7*01"]:
            d = trb_aligner.gene_dist("TRB", gene, gene)
            assert d == pytest.approx(0.0, abs=1e-9), f"self-dist nonzero for {gene}: {d}"

    def test_symmetry(self, trb_aligner):
        pairs = [
            ("TRBV10-1*01", "TRBV10-2*01"),
            ("TRBV1*01", "TRBV10-3*01"),
            ("TRBJ1-1*01", "TRBJ2-7*01"),
        ]
        for g1, g2 in pairs:
            locus = g1[:3]
            d12 = trb_aligner.gene_dist(locus, g1, g2)
            d21 = trb_aligner.gene_dist(locus, g2, g1)
            assert d12 == pytest.approx(d21, abs=1e-9), f"asymmetry for {g1}/{g2}"

    def test_non_negative(self, trb_aligner):
        checked = 0
        for (locus, g1, g2), d in trb_aligner._locus_dist.items():
            assert d >= -1e-6, f"negative dist {d} for {locus}/{g1}/{g2}"
            checked += 1
            if checked > 200:
                break

    def test_gene_dist_missing_key_returns_fallback(self, trb_aligner):
        # Unknown genes (e.g., pseudogenes) return max observed distance, not a KeyError.
        d = trb_aligner.gene_dist("TRB", "TRBV99-1*01", "TRBV99-2*01")
        assert isinstance(d, float)
        assert d >= 0.0

    def test_backwards_compat_from_seqs_still_works(self, trb_library):
        seqs = trb_library.get_sequences_aa(locus="TRB", gene="V")[:5]
        ga = GermlineAligner.from_seqs(seqs)
        d = ga.score_dist(seqs[0][0], seqs[0][0])
        assert d == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# GermlineAligner.from_library — by-hand reference values
# ---------------------------------------------------------------------------

class TestGermlineByHand:
    """Spot-check a specific distance value computed independently."""

    def test_by_hand_v_distance(self, trb_library):
        """Verify TRBV10-1*01 vs TRBV10-2*01 distance is consistent.

        Reference computed as: s(a,a) + s(b,b) - 2*s(a,b) via BioAlignerWrapper.
        Value is stable across runs (deterministic scoring).
        """
        ga = GermlineAligner.from_library(trb_library, loci=["TRB"])
        d = ga.gene_dist("TRB", "TRBV10-1*01", "TRBV10-2*01")
        seqs = dict(trb_library.get_sequences_aa(locus="TRB", gene="V"))
        bio = BioAlignerWrapper()
        s_aa = bio.score(seqs["TRBV10-1*01"], seqs["TRBV10-1*01"])
        s_bb = bio.score(seqs["TRBV10-2*01"], seqs["TRBV10-2*01"])
        s_ab = bio.score(seqs["TRBV10-1*01"], seqs["TRBV10-2*01"])
        expected = s_aa + s_bb - 2 * s_ab
        assert d == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# CDRAligner.score_dist — CDR3 by-hand sanity checks
# ---------------------------------------------------------------------------

class TestCDR3ByHand:

    def test_score_dist_self_is_zero(self):
        ca = CDRAligner()
        for s in ["CASSLETGE", "CASSIRSSYEQYF", "CASS"]:
            assert ca.score_dist(s, s) == pytest.approx(0.0, abs=1e-9)

    def test_score_dist_symmetric(self):
        ca = CDRAligner()
        pairs = [
            ("CASSLETGE", "CASSLRTGE"),
            ("CASSIRSSYEQYF", "CASSIRASYEQYF"),
            ("CASSLETGEACNQPQHF", "CASSRGTGE"),
        ]
        for s1, s2 in pairs:
            d12 = ca.score_dist(s1, s2)
            d21 = ca.score_dist(s2, s1)
            assert d12 == pytest.approx(d21, abs=1e-6), f"CDR3 asymmetry for {s1}/{s2}"

    def test_score_dist_nonneg_equal_length(self):
        ca = CDRAligner()
        s1, s2 = "CASSLETGE", "CASSLRTGE"
        assert ca.score_dist(s1, s2) >= 0.0

    def test_by_hand_equal_length(self):
        """
        s1='CASSLETGE', s2='CASSLRTGE' (both len=9), v_offset=3, j_offset=3.
        Score window: positions [3..6) = indices 3,4,5.
        s1[3:6] = 'S','L','E'  s2[3:6] = 'S','L','R'
        BLOSUM62: S-S=4, L-L=4, E-E=5, R-R=5, E-R=0
        score(s1,s1) = 10*(4+4+5) = 130
        score(s2,s2) = 10*(4+4+5) = 130
        score(s1,s2) = 10*(4+4+0) = 80
        dist = 130 + 130 - 2*80 = 100
        """
        ca = CDRAligner(v_offset=3, j_offset=3)
        s1 = "CASSLETGE"
        s2 = "CASSLRTGE"
        d = ca.score_dist(s1, s2)
        assert d == pytest.approx(100.0, abs=1e-4)


# ---------------------------------------------------------------------------
# TCREmp construction
# ---------------------------------------------------------------------------

class TestTCREmpConstruction:

    def test_from_defaults_returns_instance(self, tcremp_small):
        assert isinstance(tcremp_small, TCREmp)

    def test_n_prototypes(self, tcremp_small):
        assert tcremp_small.n_prototypes == 10
        assert tcremp_small.embedding_dim == 30

    def test_from_defaults_medium(self, tcremp_medium):
        assert tcremp_medium.n_prototypes == 100
        assert tcremp_medium.embedding_dim == 300

    def test_invalid_junction_method_raises(self):
        with pytest.raises(ValueError, match="junction_method"):
            TCREmp.from_defaults("human", "TRB", n_prototypes=5, junction_method="invalid")

    def test_biopython_method(self):
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=5, junction_method="biopython")
        assert isinstance(model.junction_aligner, BioAlignerWrapper)

    def test_prototypes_dataframe_shape(self, tcremp_small):
        assert tcremp_small.prototypes.shape == (10, 3)
        assert tcremp_small.prototypes.columns == ["v_gene", "j_gene", "junction_aa"]


# ---------------------------------------------------------------------------
# TCREmp embedding — shape and dtype
# ---------------------------------------------------------------------------

class TestTCREmpEmbedShape:

    def test_empty_input(self, tcremp_small):
        X = tcremp_small.embed([])
        assert X.shape == (0, 30)
        assert X.dtype == np.float32

    def test_single_clonotype(self, tcremp_small):
        c = _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF")
        X = tcremp_small.embed([c])
        assert X.shape == (1, 30)
        assert X.dtype == np.float32

    def test_multiple_clonotypes(self, tcremp_small):
        clonos = [
            _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF"),
            _clonotype("TRBV20-1*01", "TRBJ1-1*01", "CSARDSSYEQYF"),
            _clonotype("TRBV5-1*01", "TRBJ2-3*01", "CASSLGAYEQYF"),
        ]
        X = tcremp_small.embed(clonos)
        assert X.shape == (3, 30)
        assert X.dtype == np.float32


class TestPairedTCREmp:

    def test_from_defaults_returns_instance(self, paired_tcremp_small):
        assert isinstance(paired_tcremp_small, PairedTCREmp)

    def test_embedding_dim_is_sum_of_chain_dims(self, paired_tcremp_small):
        assert paired_tcremp_small.n_prototypes == (10, 10)
        assert paired_tcremp_small.embedding_dim == 60

    def test_empty_input(self, paired_tcremp_small):
        X = paired_tcremp_small.embed([])
        assert X.shape == (0, 60)
        assert X.dtype == np.float32

    def test_paired_embedding_matches_chain_concatenation(self, paired_tcremp_small):
        tra = _clonotype("TRAV26-1*01", "TRAJ42*01", "CIVRLRTNYGGSQGNLIF")
        trb = _clonotype("TRBV5-4*01", "TRBJ1-3*01", "CASSFDRGTGNTIYF")
        pair = _paired_clonotype("p1", tra, trb)

        X_pair = paired_tcremp_small.embed([pair])
        X_expected = np.concatenate(
            [
                paired_tcremp_small.chain1_model.embed([tra]),
                paired_tcremp_small.chain2_model.embed([trb]),
            ],
            axis=1,
            dtype=np.float32,
        )
        np.testing.assert_allclose(X_pair, X_expected)

    def test_paired_embedding_reorders_swapped_chain_input(self, paired_tcremp_small):
        tra = _clonotype("TRAV38-2/DV8*01", "TRAJ53*01", "CAYRSAGSGGSNYKLTF")
        trb = _clonotype("TRBV27*01", "TRBJ1-5*01", "CASSLMTNQPQHF")
        canonical = _paired_clonotype("p1", tra, trb)
        swapped = PairedClonotype(pair_id="p1", clonotype1=trb, clonotype2=tra)
        np.testing.assert_allclose(
            paired_tcremp_small.embed([canonical]),
            paired_tcremp_small.embed([swapped]),
        )

    def test_paired_embedding_requires_expected_loci(self, paired_tcremp_small):
        tra1 = _clonotype("TRAV26-1*01", "TRAJ42*01", "CIVRLRTNYGGSQGNLIF")
        tra2 = _clonotype("TRAV38-2/DV8*01", "TRAJ53*01", "CAYRSAGSGGSNYKLTF")
        bad_pair = _paired_clonotype("bad", tra1, tra2)
        with pytest.raises(ValueError, match="required loci"):
            paired_tcremp_small.embed([bad_pair])


# ---------------------------------------------------------------------------
# TCREmp embedding — value sanity checks
# ---------------------------------------------------------------------------

class TestTCREmpEmbedValues:

    def test_self_distances_are_zero(self, tcremp_small):
        """Embedding a prototype against itself → all three distance components = 0."""
        proto_df = tcremp_small.prototypes
        for row in proto_df.iter_rows(named=True):
            c = _clonotype(row["v_gene"], row["j_gene"], row["junction_aa"])
            X = tcremp_small.embed([c])
            for k in range(tcremp_small.n_prototypes):
                if (
                    tcremp_small._proto_v[k] == row["v_gene"]
                    and tcremp_small._proto_j[k] == row["j_gene"]
                    and tcremp_small._proto_cdr3[k] == row["junction_aa"]
                ):
                    assert X[0, 3 * k] == pytest.approx(0.0, abs=1e-4), f"V dist nonzero at k={k}"
                    assert X[0, 3 * k + 1] == pytest.approx(0.0, abs=1e-4), f"J dist nonzero at k={k}"
                    assert X[0, 3 * k + 2] == pytest.approx(0.0, abs=1e-4), f"CDR3 dist nonzero at k={k}"
                    break

    def test_all_distances_nonneg(self, tcremp_small):
        clonos = [
            _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF"),
            _clonotype("TRBV20-1*01", "TRBJ1-1*01", "CSARDSSYEQYF"),
        ]
        X = tcremp_small.embed(clonos)
        assert (X >= 0).all(), f"Negative values found: {X[X < 0]}"

    def test_v_distance_is_zero_when_same_v(self, tcremp_small):
        """V-distance component is 0 when clonotype V == prototype V."""
        proto_v0 = tcremp_small._proto_v[0]
        proto_j0 = tcremp_small._proto_j[0]
        c = _clonotype(proto_v0, proto_j0, "CASSIRSSYEQYF")
        X = tcremp_small.embed([c])
        assert X[0, 0] == pytest.approx(0.0, abs=1e-4), f"V dist={X[0,0]} should be 0"

    def test_by_hand_v_component(self, tcremp_small, trb_aligner):
        """Check that V distance in embedding matches gene_dist output."""
        c = _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF")
        X = tcremp_small.embed([c])
        for k in range(tcremp_small.n_prototypes):
            expected_v = trb_aligner.gene_dist("TRB", c.v_gene, tcremp_small._proto_v[k])
            assert X[0, 3 * k] == pytest.approx(expected_v, abs=1e-3), f"V mismatch at k={k}"

    def test_by_hand_cdr3_component(self, tcremp_small):
        """Check that CDR3 distance in embedding matches CDRAligner.score_dist."""
        ca = CDRAligner()
        c = _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF")
        X = tcremp_small.embed([c])
        for k in range(tcremp_small.n_prototypes):
            expected_cdr3 = ca.score_dist(c.junction_aa, tcremp_small._proto_cdr3[k])
            assert X[0, 3 * k + 2] == pytest.approx(expected_cdr3, abs=1e-3), f"CDR3 mismatch at k={k}"


# ---------------------------------------------------------------------------
# TCREmp symmetry / square matrix check (10 prototypes vs themselves)
# ---------------------------------------------------------------------------

class TestTCREmpSymmetricMatrix:
    """Embed the 10 prototypes against themselves → square symmetric matrix."""

    @pytest.fixture(scope="class")
    def X_self(self, tcremp_small):
        proto_df = tcremp_small.prototypes
        proto_clono = [
            _clonotype(r["v_gene"], r["j_gene"], r["junction_aa"])
            for r in proto_df.iter_rows(named=True)
        ]
        return tcremp_small.embed(proto_clono, n_jobs=1)

    def test_shape(self, X_self, tcremp_small):
        assert X_self.shape == (10, 30)

    def test_diagonal_v_is_zero(self, X_self):
        for i in range(10):
            assert X_self[i, 3 * i] == pytest.approx(0.0, abs=1e-4), f"V diag nonzero at {i}"

    def test_diagonal_j_is_zero(self, X_self):
        for i in range(10):
            assert X_self[i, 3 * i + 1] == pytest.approx(0.0, abs=1e-4), f"J diag nonzero at {i}"

    def test_diagonal_cdr3_is_zero(self, X_self):
        for i in range(10):
            assert X_self[i, 3 * i + 2] == pytest.approx(0.0, abs=1e-4), f"CDR3 diag nonzero at {i}"

    def test_v_symmetry(self, X_self):
        for i in range(10):
            for j in range(10):
                v_ij = float(X_self[i, 3 * j])
                v_ji = float(X_self[j, 3 * i])
                assert v_ij == pytest.approx(v_ji, abs=1e-3), f"V asymmetry at ({i},{j})"

    def test_j_symmetry(self, X_self):
        for i in range(10):
            for j in range(10):
                j_ij = float(X_self[i, 3 * j + 1])
                j_ji = float(X_self[j, 3 * i + 1])
                assert j_ij == pytest.approx(j_ji, abs=1e-3), f"J asymmetry at ({i},{j})"

    def test_cdr3_symmetry(self, X_self):
        for i in range(10):
            for j in range(10):
                c_ij = float(X_self[i, 3 * j + 2])
                c_ji = float(X_self[j, 3 * i + 2])
                assert c_ij == pytest.approx(c_ji, abs=1e-3), f"CDR3 asymmetry at ({i},{j})"

    def test_all_nonneg(self, X_self):
        assert (X_self >= 0).all()


# ---------------------------------------------------------------------------
# TCREmp — determinism
# ---------------------------------------------------------------------------

class TestTCREmpDeterminism:

    def test_same_result_on_repeated_calls(self, tcremp_small):
        c = _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF")
        X1 = tcremp_small.embed([c], n_jobs=1)
        X2 = tcremp_small.embed([c], n_jobs=1)
        np.testing.assert_array_equal(X1, X2)

    def test_subset_matches_larger_run(self, tcremp_small):
        """Embedding 1 clonotype must match its row in a batch of 3."""
        clonos = [
            _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF"),
            _clonotype("TRBV20-1*01", "TRBJ1-1*01", "CSARDSSYEQYF"),
            _clonotype("TRBV5-1*01", "TRBJ2-3*01", "CASSLGAYEQYF"),
        ]
        X_batch = tcremp_small.embed(clonos, n_jobs=1)
        for i, c in enumerate(clonos):
            X_single = tcremp_small.embed([c], n_jobs=1)
            np.testing.assert_allclose(X_batch[i], X_single[0], atol=1e-5)


# ---------------------------------------------------------------------------
# TCREmp — public API surface
# ---------------------------------------------------------------------------

class TestTCREmpPublicAPI:

    def test_imported_from_embedding(self):
        from mir.embedding import TCREmp as T
        assert T is TCREmp

    def test_properties(self, tcremp_small):
        assert tcremp_small.n_prototypes == 10
        assert tcremp_small.embedding_dim == 30
        assert tcremp_small.locus == "TRB"
        assert tcremp_small.species == "human"

    def test_aliases_resolved(self):
        model = TCREmp.from_defaults("hsa", "beta", n_prototypes=5)
        assert model.species == "human"
        assert model.locus == "TRB"


class TestTCREmpAutoNJobsPolicy:

    def test_auto_uses_serial_for_small_workload(self, tcremp_small):
        # 100 * 10 = 1,000 < threshold => serial
        resolved = tcremp_small._resolve_n_jobs(n_queries=100, n_jobs=None)
        assert resolved == 1

    def test_auto_uses_cpu_count_for_large_workload(self, tcremp_small, monkeypatch):
        import mir.embedding.tcremp as tcremp_module

        monkeypatch.setattr(tcremp_module.os, "cpu_count", lambda: 8)
        # 2,000,000 * 10 = 20,000,000 >= threshold => cpu_count
        resolved = tcremp_small._resolve_n_jobs(n_queries=2_000_000, n_jobs=None)
        assert resolved == 8

    def test_auto_keeps_serial_for_biopython_backend(self, monkeypatch):
        import mir.embedding.tcremp as tcremp_module

        monkeypatch.setattr(tcremp_module.os, "cpu_count", lambda: 8)
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=10, junction_method="biopython")
        resolved = model._resolve_n_jobs(n_queries=2_000_000, n_jobs=None)
        assert resolved == 1

    def test_explicit_n_jobs_always_wins(self, tcremp_small):
        resolved = tcremp_small._resolve_n_jobs(n_queries=10, n_jobs=3)
        assert resolved == 3


# ---------------------------------------------------------------------------
# Consistency test: TCREmp distances vs ClonotypeAligner.score_dist
# ---------------------------------------------------------------------------

class TestTCREmpConsistencyWithClonotypeAligner:
    """Verify that TCREmp V/J/CDR3 components match ClonotypeAligner.score_dist.

    ClonotypeAligner.from_library uses GermlineAligner.from_seqs (single-locus,
    old path). TCREmp uses GermlineAligner.from_library (multi-locus, new path).
    Both use BioAlignerWrapper scoring, so all three components must match.
    """

    @pytest.fixture(scope="class")
    def clono_aligner(self):
        from mir.distances.aligner import ClonotypeAligner
        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
        return ClonotypeAligner.from_library(lib, locus="TRB")

    @pytest.fixture(scope="class")
    def model(self):
        return TCREmp.from_defaults("human", "TRB", n_prototypes=10, junction_method="fixed_gap")

    def test_v_component_matches(self, model, clono_aligner):
        c = _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF")
        X = model.embed([c], n_jobs=1)
        for k in range(model.n_prototypes):
            proto_c = _clonotype(model._proto_v[k], model._proto_j[k], model._proto_cdr3[k])
            expected = clono_aligner.score_dist(c, proto_c).v_score
            assert X[0, 3 * k] == pytest.approx(expected, abs=1e-2), (
                f"V mismatch at k={k}: TCREmp={X[0, 3*k]:.4f} vs ClonoAligner={expected:.4f}"
            )

    def test_j_component_matches(self, model, clono_aligner):
        c = _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF")
        X = model.embed([c], n_jobs=1)
        for k in range(model.n_prototypes):
            proto_c = _clonotype(model._proto_v[k], model._proto_j[k], model._proto_cdr3[k])
            expected = clono_aligner.score_dist(c, proto_c).j_score
            assert X[0, 3 * k + 1] == pytest.approx(expected, abs=1e-2), (
                f"J mismatch at k={k}: TCREmp={X[0, 3*k+1]:.4f} vs ClonoAligner={expected:.4f}"
            )

    def test_cdr3_component_close_to_clono_aligner(self, model, clono_aligner):
        """CDR3 component matches ClonotypeAligner.score_dist within tolerance.

        TCREmp uses CDRAligner (fixed-gap model) by default; ClonotypeAligner
        also uses CDRAligner by default, so values should match exactly.
        """
        c = _clonotype("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF")
        X = model.embed([c], n_jobs=1)
        for k in range(model.n_prototypes):
            proto_c = _clonotype(model._proto_v[k], model._proto_j[k], model._proto_cdr3[k])
            expected = clono_aligner.score_dist(c, proto_c).cdr3_score
            assert X[0, 3 * k + 2] == pytest.approx(expected, abs=1e-2), (
                f"CDR3 mismatch at k={k}: TCREmp={X[0, 3*k+2]:.4f} vs ClonoAligner={expected:.4f}"
            )


# ---------------------------------------------------------------------------
# Multi-locus unit tests
# ---------------------------------------------------------------------------

class TestTCREmpMultiLocus:
    """Verify TCREmp builds and embeds correctly for all supported loci."""

    @pytest.mark.parametrize("locus", ["TRG", "TRD", "IGH", "IGK", "IGL"])
    def test_from_defaults_builds(self, locus):
        model = TCREmp.from_defaults("human", locus, n_prototypes=5)
        assert model.n_prototypes == 5
        assert model.locus == locus
        assert model.embedding_dim == 15

    @pytest.mark.parametrize("locus", ["TRG", "TRD", "IGH", "IGK", "IGL"])
    def test_embed_shape_and_dtype(self, locus):
        model = TCREmp.from_defaults("human", locus, n_prototypes=5)
        row = model.prototypes.row(0, named=True)
        c = _clonotype(row["v_gene"], row["j_gene"], row["junction_aa"])
        X = model.embed([c], n_jobs=1)
        assert X.shape == (1, 15)
        assert X.dtype == np.float32

    @pytest.mark.parametrize("locus", ["TRG", "TRD", "IGH", "IGK", "IGL"])
    def test_self_distances_zero(self, locus):
        """Embedding a prototype against itself → all three components = 0."""
        model = TCREmp.from_defaults("human", locus, n_prototypes=5)
        for k in range(model.n_prototypes):
            row = model.prototypes.row(k, named=True)
            c = _clonotype(row["v_gene"], row["j_gene"], row["junction_aa"])
            X = model.embed([c], n_jobs=1)
            assert X[0, 3 * k] == pytest.approx(0.0, abs=1e-3), f"{locus} V self-dist nonzero at k={k}"
            assert X[0, 3 * k + 1] == pytest.approx(0.0, abs=1e-3), f"{locus} J self-dist nonzero at k={k}"
            assert X[0, 3 * k + 2] == pytest.approx(0.0, abs=1e-3), f"{locus} CDR3 self-dist nonzero at k={k}"

    @pytest.mark.parametrize("locus", ["TRG", "TRD", "IGH", "IGK", "IGL"])
    def test_all_distances_nonneg(self, locus):
        model = TCREmp.from_defaults("human", locus, n_prototypes=5)
        rows = [model.prototypes.row(k, named=True) for k in range(5)]
        clonos = [_clonotype(r["v_gene"], r["j_gene"], r["junction_aa"]) for r in rows]
        X = model.embed(clonos, n_jobs=1)
        assert (X >= -1e-4).all(), f"{locus}: negative distances found"

    @pytest.mark.parametrize("locus", ["TRA", "TRB"])
    def test_mouse_loci(self, locus):
        model = TCREmp.from_defaults("mouse", locus, n_prototypes=5)
        assert model.species == "mouse"
        assert model.locus == locus
        row = model.prototypes.row(0, named=True)
        c = _clonotype(row["v_gene"], row["j_gene"], row["junction_aa"])
        X = model.embed([c], n_jobs=1)
        assert X.shape == (1, 15)
