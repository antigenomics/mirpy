"""Junction, germline and clonotype alignment scoring.

This module provides scoring classes for comparing TCR/BCR sequences:

* :class:`JunctionAligner` — junction amino-acid alignment with gap model and
  BLOSUM62 substitution scoring.  Delegates to the C extension
  ``seqdist_c`` when available, with a pure-Python fallback.

  Key methods:

  - :meth:`~JunctionAligner.score` — best alignment score across gap positions.
  - :meth:`~JunctionAligner.score_norm` / :meth:`~JunctionAligner.score_dist` —
    normalised / distance variants.
  - :meth:`~JunctionAligner.score_batch` — one query vs K refs, one C call.
  - :meth:`~JunctionAligner.score_matrix` — N queries vs K refs, (N,K) result,
    GIL released; suitable for parallel threading.
  - :meth:`~JunctionAligner.align` — best alignment with visualization strings.

* :class:`BioAlignerWrapper` — thin wrapper around BioPython's
  ``PairwiseAligner``.
* :class:`GermlineAligner` — dict-based germline gene scoring built from
  pairwise sequence alignment.
* :class:`ClonotypeAligner` — composite scorer combining V/J germline
  aligners with a junction aligner.
* :class:`ClonotypeScore` / :class:`PairedCloneScore` — score containers.

``CDRAligner`` is kept as an alias for :class:`JunctionAligner` for
backward compatibility.

Performance
-----------
Benchmarked on OLGA-generated human TRB junction sequences (lengths 10–24 aa),
measured on Apple M3, single thread:

+-------------------------------+------------------+---------------------+
| Method                        | Rate (pairs/s)   | Notes               |
+===============================+==================+=====================+
| JunctionAligner.score_matrix  | ~25 M pairs/s    | full N×K C loop,    |
|                               | effective        | GIL released        |
+-------------------------------+------------------+---------------------+
| JunctionAligner.score_batch   | ~25 M pairs/s    | one C call per      |
|                               | effective        | query vs K refs     |
+-------------------------------+------------------+---------------------+
| JunctionAligner.score (C)     | ~1 M pairs/s     | per-pair            |
+-------------------------------+------------------+---------------------+
| BioAlignerWrapper             | ~270 k pairs/s   | full DP alignment   |
+-------------------------------+------------------+---------------------+
| JunctionAligner (Python fb.)  | ~50 k pairs/s    | no C extension      |
+-------------------------------+------------------+---------------------+

:meth:`JunctionAligner.score_matrix` calls ``seqdist_c.score_matrix`` once
for the entire N×K loop in C with the GIL released.  When split across
Python threads, each chunk runs in true parallel — enabling linear scaling
up to the number of CPU cores.
"""

from abc import ABC, abstractmethod
from itertools import starmap
from multiprocessing import Pool
from Bio import Align
from Bio.Align import substitution_matrices

import typing as t
import numpy as np

from mir.common.alleles import allele_to_major, allele_with_default, strip_allele
from mir.common.clonotype import Clonotype
from mir.common.gene_library import GeneEntry, GeneLibrary

# ---------------------------------------------------------------------------
# Lazy-load C acceleration from seqdist_c (score_max, selfscore)
# ---------------------------------------------------------------------------
_seqdist_mod = None

def _get_seqdist():
    """Return the ``seqdist_c`` C module, or *None* if unavailable."""
    global _seqdist_mod
    if _seqdist_mod is None:
        try:
            from mir.distances import seqdist_c as _mod
            # Verify the junction scoring functions are present
            if hasattr(_mod, 'score_max') and hasattr(_mod, 'selfscore'):
                _seqdist_mod = _mod
        except ImportError:
            pass
    return _seqdist_mod

class Scoring(ABC):
    """Abstract base for pairwise sequence scoring."""

    @abstractmethod
    def score(self, s1: str, s2: str) -> float:
        """Raw alignment score between *s1* and *s2*."""

    def score_batch(self, query: str, refs: list[str]) -> np.ndarray:
        """Score *query* against every string in *refs*.

        Default implementation falls back to a Python loop over
        :meth:`score`.  Subclasses should override for better performance.

        Returns:
            Float64 numpy array of length ``len(refs)``.
        """
        return np.array([self.score(query, r) for r in refs], dtype=np.float64)

    def score_matrix(self, queries: list[str], refs: list[str]) -> np.ndarray:
        """Score all *queries* against all *refs*.

        Default implementation calls :meth:`score_batch` row-by-row.
        Subclasses should override for better performance.

        Returns:
            Float64 numpy array of shape ``(len(queries), len(refs))``.
        """
        N, K = len(queries), len(refs)
        result = np.empty((N, K), dtype=np.float64)
        for i, q in enumerate(queries):
            result[i] = self.score_batch(q, refs)
        return result

    def score_norm(self, s1: str, s2: str) -> float:
        """Normalised score: ``score(s1,s2) - max(score(s1,s1), score(s2,s2))``."""
        return self.score(s1, s2) - max(self.score(s1, s1), self.score(s2, s2))

    def score_dist(self, s1: str, s2: str) -> float:
        """Distance: ``score(s1,s1) + score(s2,s2) - 2*score(s1,s2)``."""
        return self.score(s1, s1) + self.score(s2, s2) - 2 * self.score(s1, s2)


class BioAlignerWrapper(Scoring):
    """Wrapper around :class:`Bio.Align.PairwiseAligner`.

    Safe to use with :mod:`multiprocessing` — the underlying BioPython
    aligner object is reconstructed on unpickling via ``__getstate__`` /
    ``__setstate__``.
    """

    def __init__(self, scoring: str = "blastp"):
        self._scoring_name = scoring
        self.aligner = Align.PairwiseAligner(scoring)

    def __getstate__(self):
        return {"_scoring_name": self._scoring_name}

    def __setstate__(self, state):
        self._scoring_name = state["_scoring_name"]
        self.aligner = Align.PairwiseAligner(self._scoring_name)

    def score(self, s1, s2) -> float:
        return self.aligner.align(s1, s2).score


class JunctionAligner(Scoring):
    """Junction amino-acid aligner with a simplified gap model.

    Aligns TCR/BCR junction (CDR3) sequences using fixed gap positions.
    Scores are computed over the interior of the junction (skipping
    *v_offset* positions from the start and *j_offset* from the end)
    using a substitution matrix (BLOSUM62 by default).  When sequences
    differ in length the shorter sequence is padded with a gap block
    placed at each of the *gap_positions* and the best score is kept.

    The heavy lifting is done in C (``seqdist_c.score_max`` /
    ``seqdist_c.selfscore``) when available; a pure-Python fallback
    is used otherwise.

    :meth:`score_matrix` computes an N×K score matrix in a single C call
    with the GIL released, enabling true thread-level parallelism.

    Parameters
    ----------
    gap_positions : iterable of int
        Candidate gap-insertion positions (negative = from end).
    mat : substitution_matrices.Array or None
        Amino-acid substitution matrix (e.g. BLOSUM62).
    gap_penalty : float
        Per-position gap penalty (typically negative).
    v_offset, j_offset : int
        Number of positions to skip at the V/J ends.
    """

    _factor = 10.0

    def __init__(self,
                 gap_positions: t.Iterable[int] = (3, 4, -4, -3),
                 mat: substitution_matrices.Array = substitution_matrices.load('BLOSUM62'),
                 gap_penalty: float = -3.0,
                 v_offset: int = 3,
                 j_offset: int = 3):
        self.gap_positions = np.asarray(tuple(gap_positions), dtype=np.int32)
        self.mat = mat
        self.gap_penalty = float(gap_penalty)
        self.v_offset = int(v_offset)
        self.j_offset = int(j_offset)
        self._use_mat = mat is not None
        self._mat256 = self._build_dense_mat(mat) if self._use_mat else np.zeros((256, 256), dtype=np.float64)

        self._self_cache: dict[str, float] = {}
        self._self_cache_max = 1 << 16

    @staticmethod
    def _build_dense_mat(mat):
        tbl = np.zeros((256, 256), dtype=np.float64)
        if mat is None:
            return tbl
        aa = [ord(c) for c in 'ACDEFGHIKLMNPQRSTVWYXBZJUO']
        for i in aa:
            ci = chr(i)
            for j in aa:
                cj = chr(j)
                try:
                    tbl[i, j] = mat[ci, cj]
                except Exception:
                    pass
        return tbl

    @staticmethod
    def _norm_pos(p: int, m: int) -> int:
        if p >= 0:
            return m if p > m else p
        q = m + int(p)
        return 0 if q < 0 else q

    def _score_equal_len_py(self, s1: str, s2: str) -> float:
        start = self.v_offset
        end = len(s1) - self.j_offset
        if end <= start:
            return 0.0
        mat = self.mat
        x = 0.0
        if mat is None:
            for i in range(start, end):
                x += 0.0 if s1[i] == s2[i] else 1.0
        else:
            for i in range(start, end):
                x += mat[s1[i], s2[i]]
        return self._factor * x

    def _score_with_gap_py(self, s1: str, s2: str, p_raw: int) -> float:
        n1, n2 = len(s1), len(s2)
        if n1 == n2:
            return self._score_equal_len_py(s1, s2)

        mat = self.mat
        gap_pen = self.gap_penalty

        if n1 < n2:
            gap_len = n2 - n1
            p = self._norm_pos(p_raw, n1)
            L = n2
            start = self.v_offset
            end = L - self.j_offset
            if end <= start:
                return 0.0
            g0 = max(start, p)
            g1 = min(end, p + gap_len)
            x = 0.0
            if mat is None:
                for i in range(start, g0):
                    x += 0.0 if s1[i] == s2[i] else 1.0
                if g1 > g0:
                    x += (g1 - g0) * gap_pen
                for i in range(g1, end):
                    j = i - gap_len
                    x += 0.0 if s1[j] == s2[i] else 1.0
            else:
                for i in range(start, g0):
                    x += mat[s1[i], s2[i]]
                if g1 > g0:
                    x += (g1 - g0) * gap_pen
                for i in range(g1, end):
                    j = i - gap_len
                    x += mat[s1[j], s2[i]]
            return self._factor * x
        else:
            gap_len = n1 - n2
            p = self._norm_pos(p_raw, n2)
            L = n1
            start = self.v_offset
            end = L - self.j_offset
            if end <= start:
                return 0.0
            g0 = max(start, p)
            g1 = min(end, p + gap_len)
            x = 0.0
            if mat is None:
                for i in range(start, g0):
                    x += 0.0 if s1[i] == s2[i] else 1.0
                if g1 > g0:
                    x += (g1 - g0) * gap_pen
                for i in range(g1, end):
                    j = i - gap_len
                    x += 0.0 if s1[i] == s2[j] else 1.0
            else:
                for i in range(start, g0):
                    x += mat[s1[i], s2[i]]
                if g1 > g0:
                    x += (g1 - g0) * gap_pen
                for i in range(g1, end):
                    j = i - gap_len
                    x += mat[s1[i], s2[j]]
            return self._factor * x

    def _selfscore_cached(self, s: str) -> float:
        val = self._self_cache.get(s)
        if val is not None:
            return val
        junction_scorer = _get_seqdist()
        if junction_scorer is not None:
            val = junction_scorer.selfscore(s, self._mat256, self._factor, self._use_mat)
        else:
            if self.mat is None:
                val = 0.0
            else:
                x = 0.0
                m = self.mat
                for c in s:
                    x += m[c, c]
                val = self._factor * x
        if len(self._self_cache) >= self._self_cache_max:
            self._self_cache.clear()
        self._self_cache[s] = val
        return val

    def score(self, s1, s2) -> float:
        junction_scorer = _get_seqdist()
        if junction_scorer is not None:
            return junction_scorer.score_max(
                s1, s2,
                self._mat256,
                np.asarray(self.gap_positions, dtype=np.int32),
                self.gap_penalty, self.v_offset, self.j_offset,
                self._factor, self._use_mat
            )
        if len(s1) == len(s2):
            return self._score_equal_len_py(s1, s2)
        best = float("-inf")
        for p in self.gap_positions:
            sc = self._score_with_gap_py(s1, s2, int(p))
            if sc > best:
                best = sc
        return best

    def score_batch(self, query: str, refs: list[str]) -> np.ndarray:
        """Score *query* against every string in *refs* in a single C call.

        Args:
            query: Query junction amino-acid sequence.
            refs: List of reference junction sequences to score against.

        Returns:
            Float64 numpy array of alignment scores, one per reference.
        """
        junction_scorer = _get_seqdist()
        if junction_scorer is not None and hasattr(junction_scorer, "score_batch_max"):
            return junction_scorer.score_batch_max(
                query, refs,
                self._mat256,
                self.gap_positions,
                self.gap_penalty, self.v_offset, self.j_offset,
                self._factor, self._use_mat,
            )
        return np.array([self.score(query, r) for r in refs], dtype=np.float64)

    def score_matrix(self, queries: list[str], refs: list[str]) -> np.ndarray:
        """Score all *queries* against all *refs* in a single C call.

        Computes the full N×K alignment score matrix with the GIL released
        for the entire inner loop.  Calling this from concurrent Python
        threads on disjoint query chunks achieves true CPU parallelism.

        Args:
            queries: List of N query junction sequences.
            refs: List of K reference junction sequences.

        Returns:
            Float64 numpy array of shape ``(N, K)``.
        """
        junction_scorer = _get_seqdist()
        if junction_scorer is not None and hasattr(junction_scorer, "score_matrix"):
            return junction_scorer.score_matrix(
                queries, refs,
                self._mat256,
                self.gap_positions,
                self.gap_penalty, self.v_offset, self.j_offset,
                self._factor, self._use_mat,
            )
        N, K = len(queries), len(refs)
        result = np.empty((N, K), dtype=np.float64)
        for i, q in enumerate(queries):
            result[i] = self.score_batch(q, refs)
        return result

    def selfscore_batch(self, seqs: list[str]) -> np.ndarray:
        """Self-alignment scores for a list of sequences.

        Args:
            seqs: List of N junction sequences.

        Returns:
            Float64 numpy array of shape ``(N,)`` with ``score(s, s)`` for each.
        """
        junction_scorer = _get_seqdist()
        if junction_scorer is not None and hasattr(junction_scorer, "selfscore_batch"):
            return junction_scorer.selfscore_batch(seqs, self._mat256, self._factor, self._use_mat)
        return np.array([self._selfscore_cached(s) for s in seqs], dtype=np.float64)

    def score_norm(self, s1, s2) -> float:
        return self.score(s1, s2) - max(self._selfscore_cached(s1), self._selfscore_cached(s2))

    def score_dist(self, s1, s2) -> float:
        return self.score(s1, s1) + self.score(s2, s2) - 2 * self.score(s1, s2)

    def pad(self, s1, s2) -> tuple[tuple[str, str]]:
        d = len(s1) - len(s2)
        if d == 0:
            return ((s1, s2),)
        elif d < 0:
            gap = '-' * (-d)
            m = len(s1)
            res = []
            for p in self.gap_positions:
                k = self._norm_pos(int(p), m)
                res.append((s1[:k] + gap + s1[k:], s2))
            return tuple(res)
        else:
            gap = '-' * d
            m = len(s2)
            res = []
            for p in self.gap_positions:
                k = self._norm_pos(int(p), m)
                res.append((s1, s2[:k] + gap + s2[k:]))
            return tuple(res)

    def alns(self, s1, s2) -> tuple[tuple[str, str, float]]:
        junction_scorer = _get_seqdist()
        if len(s1) == len(s2):
            if junction_scorer is not None:
                sc = junction_scorer.score_max(
                    s1, s2, self._mat256, np.array([0], dtype=np.int32),
                    self.gap_penalty, self.v_offset, self.j_offset,
                    self._factor, self._use_mat
                )
            else:
                sc = self._score_equal_len_py(s1, s2)
            return ((s1, s2, sc),)
        if junction_scorer is not None:
            scores = tuple(
                junction_scorer.score_max(
                    s1, s2, self._mat256, np.array([int(p)], dtype=np.int32),
                    self.gap_penalty, self.v_offset, self.j_offset,
                    self._factor, self._use_mat
                )
                for p in self.gap_positions
            )
        else:
            scores = tuple(self._score_with_gap_py(s1, s2, int(p)) for p in self.gap_positions)
        return tuple((sp1, sp2, sc) for (sp1, sp2), sc in zip(self.pad(s1, s2), scores))

    def align(self, s1: str, s2: str) -> tuple[str, str, str, float]:
        """Best alignment with visualization strings.

        Finds the gap position that maximises the score and returns
        three equal-length strings plus the score:

        * ``s1_gapped`` — first sequence with ``'-'`` at gap positions
        * ``midline``   — ``'|'`` exact match, ``':'`` positive
          substitution score, ``'.'`` non-positive, ``' '`` gap
        * ``s2_gapped`` — second sequence with ``'-'`` at gap positions
        * ``score``     — the best alignment score (scaled by ``_factor``)

        Uses the C extension (``seqdist_c.best_alignment``) when
        available, otherwise falls back to pure Python.

        Returns
        -------
        tuple[str, str, str, float]
            ``(s1_gapped, midline, s2_gapped, score)``
        """
        junction_scorer = _get_seqdist()
        if junction_scorer is not None and hasattr(junction_scorer, 'best_alignment'):
            return junction_scorer.best_alignment(
                s1, s2,
                self._mat256,
                np.asarray(self.gap_positions, dtype=np.int32),
                self.gap_penalty, self.v_offset, self.j_offset,
                self._factor, self._use_mat
            )
        return self._align_py(s1, s2)

    def _align_py(self, s1: str, s2: str) -> tuple[str, str, str, float]:
        """Pure-Python fallback for :meth:`align`."""
        n1, n2 = len(s1), len(s2)
        mat = self.mat

        def _mid(c1: str, c2: str) -> str:
            if c1 == c2:
                return '|'
            if mat is not None and mat[c1, c2] > 0:
                return ':'
            return '.'

        if n1 == n2:
            sc = self._score_equal_len_py(s1, s2)
            mid = ''.join(_mid(a, b) for a, b in zip(s1, s2))
            return (s1, mid, s2, sc)

        # Find best gap position
        best_sc = float('-inf')
        best_p = int(self.gap_positions[0])
        for p in self.gap_positions:
            sc = self._score_with_gap_py(s1, s2, int(p))
            if sc > best_sc:
                best_sc = sc
                best_p = int(p)

        if n1 < n2:
            gap_len = n2 - n1
            k = self._norm_pos(best_p, n1)
            gs1 = s1[:k] + '-' * gap_len + s1[k:]
            gs2 = s2
        else:
            gap_len = n1 - n2
            k = self._norm_pos(best_p, n2)
            gs1 = s1
            gs2 = s2[:k] + '-' * gap_len + s2[k:]

        mid_chars = []
        for a, b in zip(gs1, gs2):
            if a == '-' or b == '-':
                mid_chars.append(' ')
            else:
                mid_chars.append(_mid(a, b))
        return (gs1, ''.join(mid_chars), gs2, best_sc)

CDRAligner = JunctionAligner  # backward-compatibility alias


class _Scoring_Wrapper:
    def __init__(self, scoring: Scoring):
        self.scoring = scoring

    def __call__(self, gs1: tuple[str, str], gs2: tuple[str, str]):
        return ((gs1[0], gs2[0]), self.scoring.score(gs1[1], gs2[1]))




class GermlineAligner:
    """Gene-level aligner built from pre-computed pairwise scores.

    Two construction paths are available:

    * :meth:`from_seqs` — single-locus, backward-compatible; builds from a
      dict or list of ``(allele, sequence_aa)`` pairs.  Scores are accessed
      via :meth:`score` / :meth:`score_norm` / :meth:`score_dist`.
    * :meth:`from_library` — multi-locus; builds from a :class:`GeneLibrary`
      and computes all pairwise V/J distances per locus.  Distances are
      accessed via :meth:`dist`.

    Both paths store raw pairwise *scores* (higher = more similar).  The
    distance formula ``d(a,b) = s(a,a) + s(b,b) - 2 x s(a,b)`` converts
    scores to non-negative distances where ``d(a,a) = 0``.
    """

    def __init__(self, dist: dict[tuple[str, str], float]):
        self.dist = dist
        self.dist.update(dict(((g2, g1), score) for ((g1, g2), score) in dist.items()))
        self._locus_dist: dict[tuple[str, str, str], float] = {}
        self._fallback_dist: dict[tuple[str, str], float] = {}
        self._locus_gene_sets: dict[tuple[str, str], frozenset[str]] = {}

    def score(self, g1: str | GeneEntry, g2: str | GeneEntry) -> float:
        if isinstance(g1, GeneEntry):
            g1 = g1.allele
        if isinstance(g2, GeneEntry):
            g2 = g2.allele
        if (g1, g2) not in self.dist:
            raise ValueError(f'No pair {(g1, g2)} in distance dict!')
        return self.dist[(g1, g2)]

    def score_norm(self, g1: str | GeneEntry, g2: str | GeneEntry) -> float:
        return self.score(g1, g2) - max(self.score(g1, g1), self.score(g2, g2))

    def score_dist(self, g1: str | GeneEntry, g2: str | GeneEntry) -> float:
        return self.score(g1, g1) + self.score(g2, g2) - 2 * self.score(g1, g2)

    def gene_dist(self, locus: str, g1: str, g2: str) -> float:
        """Pre-computed distance between *g1* and *g2* for *locus*.

        Only available when the aligner was built via :meth:`from_library`.
        Returns ``d(g1,g2) = s(g1,g1) + s(g2,g2) − 2·s(g1,g2)`` where
        scores were computed at construction time.

        Args:
            locus: Receptor locus, e.g. ``'TRB'``.
            g1: First gene allele, e.g. ``'TRBV10-1*01'``.
            g2: Second gene allele, e.g. ``'TRBV10-2*01'``.

        Notes:
            Gene names without an explicit allele suffix are normalized to
            ``*01`` before lookup (for example ``TRBV10-1`` ->
            ``TRBV10-1*01``).

        Returns:
            Non-negative distance value (0 when ``g1 == g2``).

        Raises:
            KeyError: If ``(locus, g1, g2)`` was not in the library used
                during construction.
        """
        # Resolution chain: try both genes at the same fallback level.
        # Level 1 — exact allele (bare genes get *01 appended).
        # Level 2 — normalize both to *01 (handles minor alleles absent from library).
        # Level 3 — strip allele entirely (handles bare-gene libraries).
        # → NaN when the gene pair is not in the library at any level.
        g1_exact = allele_with_default(g1)
        g2_exact = allele_with_default(g2)
        for c1, c2 in (
            (g1_exact, g2_exact),
            (allele_to_major(g1_exact), allele_to_major(g2_exact)),
            (strip_allele(g1_exact), strip_allele(g2_exact)),
        ):
            val = self._locus_dist.get((locus, c1, c2))
            if val is not None:
                return val
        return float("nan")

    @classmethod
    def from_library(
        cls,
        lib: GeneLibrary,
        loci: list[str] | None = None,
        scoring: Scoring | None = None,
    ) -> 'GermlineAligner':
        """Build a multi-locus GermlineAligner from a :class:`GeneLibrary`.

        Computes all pairwise V-gene and J-gene distances for each locus in
        *lib* (or the requested subset) at construction time.  Distances are
        stored internally and retrieved in O(1) via :meth:`dist`.

        Args:
            lib: Gene library to extract sequences from.
            loci: Loci to include (e.g. ``['TRB', 'TRA']``).
                Defaults to all loci present in *lib*.
            scoring: Scoring function for sequence comparison.
                Defaults to :class:`BioAlignerWrapper` (full DP alignment).

        Returns:
            GermlineAligner populated with pre-computed pairwise distances.

        Example:
            >>> lib = GeneLibrary.load_default(loci={'TRB'}, species={'human'})
            >>> ga = GermlineAligner.from_library(lib, loci=['TRB'])
            >>> ga.dist('TRB', 'TRBV10-1*01', 'TRBV10-1*01')
            0.0
        """
        if scoring is None:
            scoring = BioAlignerWrapper()
        if loci is None:
            loci = sorted(lib.get_loci())

        locus_dist: dict[tuple[str, str, str], float] = {}
        fallback_dist: dict[tuple[str, str], float] = {}
        locus_gene_sets: dict[tuple[str, str], frozenset[str]] = {}

        for locus in loci:
            for gene_type in ('V', 'J'):
                seqs = lib.get_sequences_aa(locus=locus, gene=gene_type)
                if not seqs:
                    continue

                locus_gene_sets[(locus, gene_type)] = frozenset(a for a, _ in seqs)

                # Compute pairwise raw scores (half-matrix, then mirror).
                pair_scores: dict[tuple[str, str], float] = {}
                for i, (a1, s1) in enumerate(seqs):
                    for j, (a2, s2) in enumerate(seqs):
                        if (a2, a1) in pair_scores:
                            pair_scores[(a1, a2)] = pair_scores[(a2, a1)]
                        else:
                            pair_scores[(a1, a2)] = scoring.score(s1, s2)

                # Convert scores to distances and store with locus key.
                max_d = 0.0
                for a1, _ in seqs:
                    for a2, _ in seqs:
                        d = pair_scores[(a1, a1)] + pair_scores[(a2, a2)] - 2 * pair_scores[(a1, a2)]
                        locus_dist[(locus, a1, a2)] = float(d)
                        if d > max_d:
                            max_d = d
                fallback_dist[(locus, gene_type)] = max_d

        inst = cls.__new__(cls)
        inst.dist = {}  # single-locus path not used
        inst._locus_dist = locus_dist
        inst._fallback_dist = fallback_dist
        inst._locus_gene_sets = locus_gene_sets
        return inst

    @classmethod
    def from_library_region(
        cls,
        lib: GeneLibrary,
        loci: list[str] | None = None,
        region: str = "cdr1",
        scoring: Scoring | None = None,
    ) -> 'GermlineAligner':
        """Build a GermlineAligner over a single germline V-gene *region*.

        Like :meth:`from_library`, but distances are computed over the
        amino-acid subsequence of one region (e.g. ``'cdr1'``, ``'cdr2'``)
        rather than the full gene.  Only V genes are processed (CDR1/CDR2 are
        V-gene-determined); results are stored under the ``(locus, 'V')`` key so
        the same downstream matrix-building code works unchanged.

        Args:
            lib: Gene library loaded with ``with_regions=True``.
            loci: Loci to include.  Defaults to all loci present in *lib*.
            region: Region name in :attr:`GeneEntry.region_aa`
                (``'cdr1'``/``'cdr2'``/``'fwr1'``/...).
            scoring: Scoring function (defaults to :class:`BioAlignerWrapper`).

        Returns:
            GermlineAligner with pre-computed pairwise region distances.

        Raises:
            ValueError: If no entries carry *region* for the requested loci
                (e.g. the library lacks region annotations).
        """
        if scoring is None:
            scoring = BioAlignerWrapper()
        if loci is None:
            loci = sorted(lib.get_loci())

        locus_dist: dict[tuple[str, str, str], float] = {}
        fallback_dist: dict[tuple[str, str], float] = {}
        locus_gene_sets: dict[tuple[str, str], frozenset[str]] = {}

        for locus in loci:
            seqs = lib.get_region_sequences_aa(locus=locus, gene="V", region=region)
            if not seqs:
                continue
            locus_gene_sets[(locus, "V")] = frozenset(a for a, _ in seqs)
            pair_scores: dict[tuple[str, str], float] = {}
            for a1, s1 in seqs:
                for a2, s2 in seqs:
                    if (a2, a1) in pair_scores:
                        pair_scores[(a1, a2)] = pair_scores[(a2, a1)]
                    else:
                        pair_scores[(a1, a2)] = scoring.score(s1, s2)
            max_d = 0.0
            for a1, _ in seqs:
                for a2, _ in seqs:
                    d = pair_scores[(a1, a1)] + pair_scores[(a2, a2)] - 2 * pair_scores[(a1, a2)]
                    locus_dist[(locus, a1, a2)] = float(d)
                    if d > max_d:
                        max_d = d
            fallback_dist[(locus, "V")] = max_d

        if not locus_dist:
            raise ValueError(
                f"No region {region!r} annotations found for loci {loci}. "
                "Load the library with with_regions=True (and ensure "
                "region_annotations.txt is present)."
            )

        inst = cls.__new__(cls)
        inst.dist = {}
        inst._locus_dist = locus_dist
        inst._fallback_dist = fallback_dist
        inst._locus_gene_sets = locus_gene_sets
        return inst

    @classmethod
    def from_seqs(cls,
                  seqs: dict[str, str] | t.Iterable[tuple[str, str]] | list[GeneEntry],
                  scoring: Scoring = BioAlignerWrapper(),
                  nproc=1, chunk_sz=4096):
        scoring_wrapper = _Scoring_Wrapper(scoring)
        if type(seqs) is dict:
            seqs = seqs.items()
        elif isinstance(seqs, list) and len(seqs) > 0 and isinstance(seqs[0], GeneEntry):
            seqs = [(s.allele, s.sequence_aa) for s in seqs]
        gen = ((gs1, gs2) for gs1 in seqs for gs2 in seqs if gs1[0] >= gs2[0])
        if nproc == 1:
            dist = starmap(scoring_wrapper, gen)  # this operation is long, Bio issue :(
        else:
            with Pool(nproc) as pool:
                dist = pool.starmap(scoring_wrapper, gen, chunk_sz)
        return cls(dict(dist))


class ClonotypeScore:
    """Container for V / J / junction component scores."""

    __scores__ = ['v_score', 'j_score', 'junction_score']

    def __init__(self, v_score: float, j_score: float, junction_score: float):
        self.v_score = v_score
        self.j_score = j_score
        self.junction_score = junction_score

    @property
    def cdr3_score(self) -> float:
        """Backward-compatibility alias for :attr:`junction_score`."""
        return self.junction_score

    def __repr__(self):
        return (f'ClonotypeScore: v={self.v_score}, j={self.j_score}, '
                f'junction={self.junction_score}')

    def __str__(self):
        return self.__repr__()

    def get_flatten_score(self):
        return [self.v_score, self.j_score, self.junction_score]


class PairedCloneScore:
    """Score container for paired alpha/beta chain clonotypes."""

    def __init__(self, alpha_chain_score: ClonotypeScore, beta_chain_score: ClonotypeScore):
        self.alpha_chain_score = alpha_chain_score
        self.beta_chain_score = beta_chain_score

    def get_flatten_score(self):
        return [self.alpha_chain_score.v_score, self.alpha_chain_score.j_score, self.alpha_chain_score.junction_score,
                self.beta_chain_score.v_score, self.beta_chain_score.j_score, self.beta_chain_score.junction_score]


class ClonotypeAligner:
    """Composite aligner combining V gene, J gene, and junction scoring."""

    def __init__(self,
                 v_aligner: GermlineAligner,
                 j_aligner: GermlineAligner,
                 junction_aligner: JunctionAligner = JunctionAligner()):
        self.v_aligner = v_aligner
        self.j_aligner = j_aligner
        self.junction_aligner = junction_aligner

    @property
    def cdr3_aligner(self) -> JunctionAligner:
        """Backward-compatibility alias for :attr:`junction_aligner`."""
        return self.junction_aligner

    @classmethod
    def from_library(cls,
                     lib: GeneLibrary = None,
                     locus: str = None,
                     junction_aligner: JunctionAligner = JunctionAligner()):
        if lib is None:
            lib = GeneLibrary.load_default()
        v_aligner = GermlineAligner.from_seqs(lib.get_sequences_aa(locus=locus, gene='V'))
        j_aligner = GermlineAligner.from_seqs(lib.get_sequences_aa(locus=locus, gene='J'))
        return cls(v_aligner, j_aligner, junction_aligner)

    def score(self, cln1: Clonotype, cln2: Clonotype) -> ClonotypeScore:
        return ClonotypeScore(
            v_score=self.v_aligner.score(cln1.v_call, cln2.v_call),
            j_score=self.j_aligner.score(cln1.j_call, cln2.j_call),
            junction_score=self.junction_aligner.score(cln1.junction_aa, cln2.junction_aa),
        )

    def score_norm(self, cln1: Clonotype, cln2: Clonotype) -> ClonotypeScore:
        return ClonotypeScore(
            v_score=self.v_aligner.score_norm(cln1.v_call, cln2.v_call),
            j_score=self.j_aligner.score_norm(cln1.j_call, cln2.j_call),
            junction_score=self.junction_aligner.score_norm(cln1.junction_aa, cln2.junction_aa),
        )

    def score_dist(self, cln1: Clonotype, cln2: Clonotype) -> ClonotypeScore:
        return ClonotypeScore(
            v_score=self.v_aligner.score_dist(cln1.v_call, cln2.v_call),
            j_score=self.j_aligner.score_dist(cln1.j_call, cln2.j_call),
            junction_score=self.junction_aligner.score_dist(cln1.junction_aa, cln2.junction_aa),
        )

