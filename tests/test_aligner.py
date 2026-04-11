"""Correctness and speed benchmark tests for CDR3 / clonotype alignment.

CDR3 sequences generated from human TRB using OLGA
(Sethna et al., 2019, *Bioinformatics*).
"""

import time
import math
import pytest
import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

from mir.distances.aligner import (
    CDRAligner,
    BioAlignerWrapper,
    Scoring,
    GermlineAligner,
    ClonotypeScore,
    PairedCloneScore,
)

# ---------------------------------------------------------------------------
# OLGA-generated human TRB CDR3 amino-acid sequences (50 sequences)
# ---------------------------------------------------------------------------
OLGA_CDR3S = [
    "CASSLETGEACNQPQHF",
    "CATSGHRDRQVQPQHF",
    "CASSLGRDRGMNTEAFF",
    "CASSRGNTIYF",
    "CASSRGPQGPYRYGYTF",
    "CATSDLEGQGDKNTEAFF",
    "CASSRGTSRYPYEQYF",
    "CASSCHEVGTQHF",
    "CASSEGDRDEQYF",
    "CASIPGTSGSSTDTQYF",
    "CASSGKGLAGGSLENEQYF",
    "CASSRGHGNTIYF",
    "CASSWSKGITGELFF",
    "CASSPPIQGSGDWRIMVGNYEQYF",
    "CARGSRGGFSGANVLTF",
    "CELQQETQYF",
    "CASNRRGRDEAFF",
    "CASSKRGQGVLYGYTF",
    "CASSQVVQGAIETQYF",
    "CASSGCNRGYSNQPQHF",
    "CASSQVTESPDYEQYF",
    "CSGGTGTEAFF",
    "CASSYKVGSYGYTF",
    "CASSGPGCHAGELFF",
    "CASSPPPIGTLTDTQYF",
    "CASSTGPTLGNQPQHF",
    "CAWSGRGGRANAEKLFF",
    "CASSTRGINEKLFF",
    "CASSFWGHRGVEKLFF",
    "CASSYRGGRSYNSPLHF",
    "CASSGKRSCTTEAFF",
    "CASKTGRTGELFF",
    "CASSGVLAKNIQYF",
    "CASSSGKRNYGYTF",
    "CASSYLYTAKNIQYF",
    "CSDGTAYNEQFF",
    "CASSQVIPGQAYEGRAGAFF",
    "CASSERRQFGPRYEQYF",
    "CASTESGTSGGATGNVSSYEQYF",
    "CASFVRRSESYEQYF",
    "CASSFRNEQYF",
    "CASSPRTGPDQHF",
    "CASCLQGEESQHNEQFF",
    "CASRLGTRTGGTGANVLTF",
    "CASGLSLAVVSDEQFF",
    "CNIYIPGLAGGQAFHRGYEQYF",
    "CASSKLGPEASTDTQYF",
    "CASTFPSILGGGTMLTDTQYF",
    "CAIRAGQGFRVAKNIQYF",
    "CSVFPRAFRMNTEAFF",
]


# ===================================================================
# Correctness tests
# ===================================================================


class TestCDRAlignerBasic:
    """Unit tests for CDRAligner scoring logic."""

    def setup_method(self):
        self.aligner = CDRAligner()

    # -- Self-score properties -------------------------------------------

    def test_self_score_positive(self):
        """Self-score with BLOSUM62 must be positive for valid AA seqs."""
        for s in OLGA_CDR3S[:20]:
            sc = self.aligner.score(s, s)
            assert sc > 0, f"self-score for {s} should be > 0, got {sc}"

    def test_score_norm_self_is_nonpositive(self):
        """Normalised self-score must be <= 0.

        Note: CDRAligner.score_norm uses ``_selfscore_cached`` (full-length
        diagonal) while ``score(s, s)`` only covers [v_offset..L-j_offset),
        so score_norm(s, s) is typically negative, not zero.
        """
        for s in OLGA_CDR3S[:10]:
            sn = self.aligner.score_norm(s, s)
            assert sn <= 1e-9, f"norm(self) for {s} = {sn}"

    def test_score_dist_self_is_zero(self):
        """Distance to self must be zero."""
        for s in OLGA_CDR3S[:10]:
            d = self.aligner.score_dist(s, s)
            assert d == pytest.approx(0.0, abs=1e-9)

    # -- Symmetry --------------------------------------------------------

    def test_score_symmetry(self):
        """score(a,b) == score(b,a)."""
        pairs = list(zip(OLGA_CDR3S[:15], OLGA_CDR3S[15:30]))
        for a, b in pairs:
            assert self.aligner.score(a, b) == pytest.approx(
                self.aligner.score(b, a), abs=1e-9
            ), f"asymmetric score for {a}, {b}"

    def test_score_norm_symmetry(self):
        for a, b in zip(OLGA_CDR3S[:10], OLGA_CDR3S[10:20]):
            assert self.aligner.score_norm(a, b) == pytest.approx(
                self.aligner.score_norm(b, a), abs=1e-9
            )

    # -- Normalised score is non-positive --------------------------------

    def test_score_norm_nonpositive(self):
        """Normalised score must be <= 0 (similarity relative to self)."""
        for a, b in zip(OLGA_CDR3S[:15], OLGA_CDR3S[15:30]):
            sn = self.aligner.score_norm(a, b)
            assert sn <= 1e-9, f"score_norm({a},{b}) = {sn} > 0"

    # -- Equal-length alignment ------------------------------------------

    def test_equal_length_no_gap(self):
        """Equal-length sequences should not need gaps."""
        s1 = "CASSLETGE"
        s2 = "CASSRGTGE"
        a = CDRAligner(gap_positions=(3,), v_offset=3, j_offset=3)
        pads = a.pad(s1, s2)
        assert pads == ((s1, s2),)

    def test_equal_length_deterministic(self):
        """Same input → same output."""
        s1, s2 = OLGA_CDR3S[0], OLGA_CDR3S[1]
        sc1 = self.aligner.score(s1, s2)
        sc2 = self.aligner.score(s1, s2)
        assert sc1 == sc2

    # -- Gap padding -----------------------------------------------------

    def test_pad_length_consistency(self):
        """Padded sequences must have equal length (= max of originals)."""
        s1 = OLGA_CDR3S[0]
        s2 = OLGA_CDR3S[3]  # different length
        for p1, p2 in self.aligner.pad(s1, s2):
            assert len(p1) == len(p2), f"pad mismatch: {len(p1)} vs {len(p2)}"

    def test_alns_returns_score_per_gap(self):
        """alns() returns one (s1pad, s2pad, score) per gap position."""
        s1, s2 = OLGA_CDR3S[0], OLGA_CDR3S[3]
        result = self.aligner.alns(s1, s2)
        assert len(result) == len(self.aligner.gap_positions)
        for padded1, padded2, sc in result:
            assert isinstance(sc, float)
            assert len(padded1) == len(padded2)

    # -- score_dist triangle inequality ----------------------------------

    def test_triangle_inequality(self):
        """score_dist should satisfy the triangle inequality for simple cases."""
        seqs = OLGA_CDR3S[:5]
        d = {}
        for i, si in enumerate(seqs):
            for j, sj in enumerate(seqs):
                d[(i, j)] = self.aligner.score_dist(si, sj)

        for i in range(len(seqs)):
            for j in range(len(seqs)):
                for k in range(len(seqs)):
                    # Note: score_dist isn't a proper metric for gapped seqs,
                    # but for equal-length it should hold.
                    if len(seqs[i]) == len(seqs[j]) == len(seqs[k]):
                        assert d[(i, k)] <= d[(i, j)] + d[(j, k)] + 1e-6

    # -- Identity matrix scoring -----------------------------------------

    def test_identity_matrix_equal_len(self):
        """With identity matrix (no BLOSUM), equal-length mismatch count."""
        a = CDRAligner(mat=None, gap_penalty=0, v_offset=0, j_offset=0)
        s1 = "CASSLETGE"
        s2 = "CASSRXTGE"
        # mat=None → mismatch=1.0, match=0.0
        expected = sum(1.0 for c1, c2 in zip(s1, s2) if c1 != c2) * CDRAligner._factor
        assert a.score(s1, s2) == pytest.approx(expected)


# ===================================================================
# C vs Python fallback consistency
# ===================================================================


class TestCvsPythonFallback:
    """Verify that C-accelerated and pure-Python paths give identical results."""

    def setup_method(self):
        self.aligner = CDRAligner()

    def _py_score(self, s1, s2):
        """Force the pure-Python path."""
        if len(s1) == len(s2):
            return self.aligner._score_equal_len_py(s1, s2)
        best = -math.inf
        for p in self.aligner.gap_positions:
            sc = self.aligner._score_with_gap_py(s1, s2, int(p))
            if sc > best:
                best = sc
        return best

    def _py_selfscore(self, s):
        """Force the pure-Python selfscore path."""
        if self.aligner.mat is None:
            return 0.0
        x = 0.0
        m = self.aligner.mat
        for c in s:
            x += m[c, c]
        return self.aligner._factor * x

    def test_equal_len_c_vs_py(self):
        same_len_pairs = [
            (a, b) for a, b in zip(OLGA_CDR3S, OLGA_CDR3S[1:])
            if len(a) == len(b)
        ]
        if not same_len_pairs:
            # Make some equal-length pairs by truncation
            same_len_pairs = [(s[:10], s[:10].replace('S', 'T', 1)) for s in OLGA_CDR3S[:5]]
        for a, b in same_len_pairs:
            c_score = self.aligner.score(a, b)
            py_score = self._py_score(a, b)
            assert c_score == pytest.approx(py_score, abs=1e-6), \
                f"C vs Py mismatch for same-len {a}, {b}: {c_score} vs {py_score}"

    def test_diff_len_c_vs_py(self):
        diff_len_pairs = [
            (a, b) for a, b in zip(OLGA_CDR3S, OLGA_CDR3S[1:])
            if len(a) != len(b)
        ]
        for a, b in diff_len_pairs[:15]:
            c_score = self.aligner.score(a, b)
            py_score = self._py_score(a, b)
            assert c_score == pytest.approx(py_score, abs=1e-6), \
                f"C vs Py mismatch for diff-len {a}, {b}: {c_score} vs {py_score}"

    def test_selfscore_c_vs_py(self):
        for s in OLGA_CDR3S[:20]:
            from mir.distances.aligner import _get_seqdist
            cdr = _get_seqdist()
            if cdr is None:
                pytest.skip("C extension not available")
            c_val = cdr.selfscore(s, self.aligner._mat256, self.aligner._factor, self.aligner._use_mat)
            py_val = self._py_selfscore(s)
            assert c_val == pytest.approx(py_val, abs=1e-6), \
                f"selfscore mismatch for {s}: {c_val} vs {py_val}"


# ===================================================================
# BioPython cross-check
# ===================================================================


class TestBioPythonCrossCheck:
    """Cross-check CDRAligner against BioPython PairwiseAligner.

    For *equal-length* ungapped alignment with the same substitution
    matrix and no gap penalty the scores should agree (up to the offset
    trimming and the factor scaling that CDRAligner applies).
    """

    def test_ungapped_equal_length_vs_biopython(self):
        """Compare BLOSUM62 scores for equal-length CDR3 pairs.

        CDRAligner computes score over positions [v_offset .. L-j_offset),
        scaled by _factor.  When v_offset = j_offset = 0 and sequences
        have the same length (so no gap is needed), the raw score should
        equal BioPython's *ungapped* global alignment score × _factor.
        """
        mat = substitution_matrices.load("BLOSUM62")
        # CDRAligner with zero offsets to compare full-length
        cdr = CDRAligner(mat=mat, gap_penalty=-1000.0, v_offset=0, j_offset=0)

        bio = Align.PairwiseAligner()
        bio.mode = "global"
        bio.substitution_matrix = mat
        bio.open_gap_score = -1000.0  # effectively disable gaps
        bio.extend_gap_score = -1000.0

        # Gather equal-length pairs from OLGA set
        pairs = []
        for i, s1 in enumerate(OLGA_CDR3S):
            for s2 in OLGA_CDR3S[i + 1 :]:
                if len(s1) == len(s2):
                    pairs.append((s1, s2))
                    if len(pairs) >= 20:
                        break
            if len(pairs) >= 20:
                break

        # Also create some by trimming to a common length
        for s1, s2 in zip(OLGA_CDR3S[:10], OLGA_CDR3S[10:20]):
            L = min(len(s1), len(s2))
            pairs.append((s1[:L], s2[:L]))

        assert len(pairs) > 0, "No pairs generated"

        for s1, s2 in pairs:
            cdr_score = cdr.score(s1, s2) / CDRAligner._factor
            bio_score = bio.align(s1, s2).score
            assert cdr_score == pytest.approx(bio_score, abs=1e-4), (
                f"CDRAligner vs BioPython mismatch for "
                f"{s1}/{s2}: {cdr_score} vs {bio_score}"
            )

    def test_ungapped_selfscore_vs_biopython(self):
        """Self-score from CDRAligner must match BioPython diagonal sum."""
        mat = substitution_matrices.load("BLOSUM62")
        cdr = CDRAligner(mat=mat, v_offset=0, j_offset=0)

        bio = Align.PairwiseAligner()
        bio.mode = "global"
        bio.substitution_matrix = mat
        bio.open_gap_score = -1000.0
        bio.extend_gap_score = -1000.0

        for s in OLGA_CDR3S[:15]:
            cdr_ss = cdr.score(s, s) / CDRAligner._factor
            bio_ss = bio.align(s, s).score
            assert cdr_ss == pytest.approx(bio_ss, abs=1e-4), \
                f"self-score mismatch for {s}: {cdr_ss} vs {bio_ss}"


# ===================================================================
# Backward compatibility
# ===================================================================


class TestBackwardCompat:
    """Ensure public API has not changed."""

    def test_scoring_abc(self):
        assert hasattr(Scoring, "score")
        assert hasattr(Scoring, "score_norm")

    def test_cdraligner_interface(self):
        a = CDRAligner()
        assert callable(a.score)
        assert callable(a.score_norm)
        assert callable(a.score_dist)
        assert callable(a.pad)
        assert callable(a.alns)

    def test_bioaligner_wrapper(self):
        w = BioAlignerWrapper()
        sc = w.score("CASS", "CASS")
        assert isinstance(sc, float)

    def test_clonotype_score_attrs(self):
        cs = ClonotypeScore(1.0, 2.0, 3.0)
        assert cs.v_score == 1.0
        assert cs.j_score == 2.0
        assert cs.cdr3_score == 3.0
        assert cs.get_flatten_score() == [1.0, 2.0, 3.0]

    def test_paired_clone_score(self):
        a = ClonotypeScore(1, 2, 3)
        b = ClonotypeScore(4, 5, 6)
        p = PairedCloneScore(a, b)
        assert p.get_flatten_score() == [1, 2, 3, 4, 5, 6]

    def test_distances_init_exports(self):
        from mir.distances import GermlineAligner, ClonotypeAligner, ClonotypeScore
        assert GermlineAligner is not None
        assert ClonotypeAligner is not None
        assert ClonotypeScore is not None

    def test_seqdist_c_has_all_functions(self):
        """The merged seqdist_c module must expose all original functions."""
        from mir.distances import seqdist_c
        assert hasattr(seqdist_c, "hamming")
        assert hasattr(seqdist_c, "levenshtein")
        assert hasattr(seqdist_c, "score_max")
        assert hasattr(seqdist_c, "selfscore")


# ===================================================================
# Speed benchmarks (pytest-benchmark style, manual timing)
# ===================================================================


class TestAlignmentBenchmarks:
    """Speed benchmarks for CDR3 alignment.

    Compares C-accelerated scoring against the Python fallback and
    BioPython's PairwiseAligner.  Results are printed to stdout — run
    with ``pytest -s`` to see them.
    """

    N_PAIRS = 200  # number of pairs to score in each benchmark

    def _make_pairs(self):
        pairs = []
        n = len(OLGA_CDR3S)
        for i in range(self.N_PAIRS):
            a = OLGA_CDR3S[i % n]
            b = OLGA_CDR3S[(i * 7 + 3) % n]
            pairs.append((a, b))
        return pairs

    def test_benchmark_c_scoring(self):
        aligner = CDRAligner()
        pairs = self._make_pairs()

        # Warm up
        for a, b in pairs[:5]:
            aligner.score(a, b)

        t0 = time.perf_counter()
        for a, b in pairs:
            aligner.score(a, b)
        elapsed = time.perf_counter() - t0

        rate = self.N_PAIRS / elapsed
        print(f"\n  CDRAligner C  : {self.N_PAIRS} pairs in {elapsed*1000:.1f} ms "
              f"({rate:.0f} pairs/s)")

    def test_benchmark_python_fallback(self):
        aligner = CDRAligner()
        pairs = self._make_pairs()

        def py_score(s1, s2):
            if len(s1) == len(s2):
                return aligner._score_equal_len_py(s1, s2)
            best = -math.inf
            for p in aligner.gap_positions:
                sc = aligner._score_with_gap_py(s1, s2, int(p))
                if sc > best:
                    best = sc
            return best

        # Warm up
        for a, b in pairs[:5]:
            py_score(a, b)

        t0 = time.perf_counter()
        for a, b in pairs:
            py_score(a, b)
        elapsed = time.perf_counter() - t0

        rate = self.N_PAIRS / elapsed
        print(f"\n  CDRAligner Py : {self.N_PAIRS} pairs in {elapsed*1000:.1f} ms "
              f"({rate:.0f} pairs/s)")

    def test_benchmark_biopython(self):
        bio = BioAlignerWrapper()
        pairs = self._make_pairs()

        # Warm up
        for a, b in pairs[:5]:
            bio.score(a, b)

        t0 = time.perf_counter()
        for a, b in pairs:
            bio.score(a, b)
        elapsed = time.perf_counter() - t0

        rate = self.N_PAIRS / elapsed
        print(f"\n  BioPython     : {self.N_PAIRS} pairs in {elapsed*1000:.1f} ms "
              f"({rate:.0f} pairs/s)")

    def test_c_faster_than_python(self):
        """C extension should be significantly faster than Python fallback."""
        aligner = CDRAligner()
        pairs = self._make_pairs()

        def py_score(s1, s2):
            if len(s1) == len(s2):
                return aligner._score_equal_len_py(s1, s2)
            best = -math.inf
            for p in aligner.gap_positions:
                sc = aligner._score_with_gap_py(s1, s2, int(p))
                if sc > best:
                    best = sc
            return best

        # Time C path
        t0 = time.perf_counter()
        for a, b in pairs:
            aligner.score(a, b)
        t_c = time.perf_counter() - t0

        # Time Python path
        t0 = time.perf_counter()
        for a, b in pairs:
            py_score(a, b)
        t_py = time.perf_counter() - t0

        speedup = t_py / t_c if t_c > 0 else float("inf")
        print(f"\n  C/Py speedup  : {speedup:.1f}x")
        assert speedup > 2.0, f"C extension only {speedup:.1f}x faster than Python"
