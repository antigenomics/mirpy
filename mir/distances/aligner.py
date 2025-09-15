import time
from abc import abstractmethod
from itertools import starmap
from multiprocessing import Pool
from Bio import Align
from Bio.Align import substitution_matrices

import typing as t
import importlib
import numpy as np
from functools import lru_cache

from mir.common.clonotype import ClonotypeAA, PairedChainClone
from mir.common.segments import Segment, SegmentLibrary

_cdrscore_mod = None
def _get_cdrscore():
    global _cdrscore_mod
    if _cdrscore_mod is None:
        _cdrscore_mod = importlib.import_module('mir.distances.cdrscore')
    return _cdrscore_mod

class Scoring:
    @abstractmethod
    def score(self, s1: str, s2: str) -> float:
        pass

    def score_norm(self, s1: str, s2: str) -> float:
        return self.score(s1, s2) - max(self.score(s1, s1), self.score(s2, s2))


class BioAlignerWrapper(Scoring):
    def __init__(self, scoring: str = "blastp"):
        self.aligner = Align.PairwiseAligner(scoring)

    def score(self, s1, s2) -> float:
        return self.aligner.align(s1, s2).score


# TODO substitution matrix wrapper to load from dict
class CDRAligner(Scoring):
    _factor = 10.0
    _cdr_mod = None

    @staticmethod
    def _get_cdrscore():
        if CDRAligner._cdr_mod is not None:
            return CDRAligner._cdr_mod
        try:
            import importlib
            CDRAligner._cdr_mod = importlib.import_module('mir.distances.cdrscore')
        except Exception:
            CDRAligner._cdr_mod = None
        return CDRAligner._cdr_mod

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
        cdr = self._get_cdrscore()
        if cdr is not None:
            val = cdr.selfscore(s, self._mat256, self._factor, self._use_mat)
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
        cdr = self._get_cdrscore()
        if cdr is not None:
            return cdr.score_max(
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
        cdr = self._get_cdrscore()
        if len(s1) == len(s2):
            if cdr is not None:
                sc = cdr.score_max(
                    s1, s2, self._mat256, np.array([0], dtype=np.int32),
                    self.gap_penalty, self.v_offset, self.j_offset,
                    self._factor, self._use_mat
                )
            else:
                sc = self._score_equal_len_py(s1, s2)
            return ((s1, s2, sc),)
        if cdr is not None:
            scores = tuple(
                cdr.score_max(
                    s1, s2, self._mat256, np.array([int(p)], dtype=np.int32),
                    self.gap_penalty, self.v_offset, self.j_offset,
                    self._factor, self._use_mat
                )
                for p in self.gap_positions
            )
        else:
            scores = tuple(self._score_with_gap_py(s1, s2, int(p)) for p in self.gap_positions)
        return tuple((sp1, sp2, sc) for (sp1, sp2), sc in zip(self.pad(s1, s2), scores))

class _Scoring_Wrapper:
    def __init__(self, scoring: Scoring):
        self.scoring = scoring

    def __call__(self, gs1: tuple[str, str], gs2: tuple[str, str]):
        return ((gs1[0], gs2[0]), self.scoring.score(gs1[1], gs2[1]))


class GermlineAligner:
    def __init__(self, dist: dict[tuple[str, str], float]):
        self.dist = dist
        self.dist.update(dict(((g2, g1), score) for ((g1, g2), score) in dist.items()))

    def score(self, g1: str | Segment, g2: str | Segment) -> float:
        if isinstance(g1, Segment):
            g1 = g1.id
        if isinstance(g2, Segment):
            g2 = g2.id
        if (g1, g2) not in self.dist:
            raise ValueError(f'No pair {(g1, g2)} in distance dict!')
        return self.dist[(g1, g2)]

    def score_norm(self, g1: str | Segment, g2: str | Segment) -> float:
        return self.score(g1, g2) - max(self.score(g1, g1), self.score(g2, g2))

    def score_dist(self, g1: str | Segment, g2: str | Segment) -> float:
        return self.score(g1, g1) + self.score(g2, g2) - 2 * self.score(g1, g2)

    @classmethod
    def from_seqs(cls,
                  seqs: dict[str, str] | t.Iterable[tuple[str, str]] | list[Segment],
                  scoring: Scoring = BioAlignerWrapper(),
                  nproc=1, chunk_sz=4096):
        scoring_wrapper = _Scoring_Wrapper(scoring)
        if type(seqs) is dict:
            seqs = seqs.items()
        elif isinstance(seqs, list) and len(seqs) > 0 and isinstance(seqs[0], Segment):
            # фикс: делаем список пар (id, seqaa)
            seqs = [(s.id, s.seqaa) for s in seqs]
        gen = ((gs1, gs2) for gs1 in seqs for gs2 in seqs if gs1[0] >= gs2[0])
        if nproc == 1:
            dist = starmap(scoring_wrapper, gen)  # this operation is long, Bio issue :(
        else:
            with Pool(nproc) as pool:
                dist = pool.starmap(scoring_wrapper, gen, chunk_sz)
        return cls(dict(dist))


class ClonotypeScore:
    __scores__ = ['v_score', 'j_score', 'cdr3_score']

    def __init__(self, v_score: float, j_score: float, cdr3_score: float):
        self.v_score = v_score
        self.j_score = j_score
        self.cdr3_score = cdr3_score

    def __repr__(self):
        return f'Clonotype score: v_score={self.v_score}, j_score={self.j_score}, cdr3_score={self.cdr3_score}'
    def __str__(self):
        return f'Clonotype score: v_score={self.v_score}, j_score={self.j_score}, cdr3_score={self.cdr3_score}'

    def get_flatten_score(self):
        return [self.v_score, self.j_score, self.cdr3_score]


class PairedCloneScore:
    def __init__(self, alpha_chain_score: ClonotypeScore, beta_chain_score: ClonotypeScore):
        self.alpha_chain_score = alpha_chain_score
        self.beta_chain_score = beta_chain_score

    def get_flatten_score(self):
        return [self.alpha_chain_score.v_score, self.alpha_chain_score.j_score, self.alpha_chain_score.cdr3_score,
                self.beta_chain_score.v_score, self.beta_chain_score.j_score, self.beta_chain_score.cdr3_score]


class ClonotypeAligner:
    def __init__(self,
                 v_aligner: GermlineAligner,
                 j_aligner: GermlineAligner,
                 cdr3_aligner: CDRAligner = CDRAligner()):
        self.v_aligner = v_aligner
        self.j_aligner = j_aligner
        self.cdr3_aligner = cdr3_aligner

    @classmethod
    def from_library(cls,
                     lib: SegmentLibrary = SegmentLibrary.load_default(),
                     gene: str = None,
                     cdr3_aligner: CDRAligner = CDRAligner()):
        v_aligner = GermlineAligner.from_seqs(lib.get_seqaas(gene=gene, stype='V'))
        j_aligner = GermlineAligner.from_seqs(lib.get_seqaas(gene=gene, stype='J'))
        return cls(v_aligner, j_aligner, cdr3_aligner)

    def score(self, cln1: ClonotypeAA, cln2: ClonotypeAA) -> ClonotypeScore:
        return ClonotypeScore(v_score=self.v_aligner.score(cln1.v, cln2.v),
                              j_score=self.j_aligner.score(cln1.j, cln2.j),
                              cdr3_score=self.cdr3_aligner.score(cln1.cdr3aa, cln2.cdr3aa))

    def score_norm(self, cln1: ClonotypeAA, cln2: ClonotypeAA) -> ClonotypeScore:
        return ClonotypeScore(v_score=self.v_aligner.score_norm(cln1.v, cln2.v),
                              j_score=self.j_aligner.score_norm(cln1.j, cln2.j),
                              cdr3_score=self.cdr3_aligner.score_norm(cln1.cdr3aa, cln2.cdr3aa))

    def score_dist(self, cln1: ClonotypeAA, cln2: ClonotypeAA) -> ClonotypeScore:
        return ClonotypeScore(v_score=self.v_aligner.score_dist(cln1.v, cln2.v),
                              j_score=self.j_aligner.score_dist(cln1.j, cln2.j),
                              cdr3_score=self.cdr3_aligner.score_dist(cln1.cdr3aa, cln2.cdr3aa))

    def score_paired(self, cln1: PairedChainClone, cln2: PairedChainClone) -> PairedCloneScore:
        return PairedCloneScore(self.score(cln1.chainA, cln2.chainA),
                                self.score(cln1.chainB, cln2.chainB))

    def score_norm_paired(self, cln1: PairedChainClone, cln2: PairedChainClone) -> PairedCloneScore:
        return PairedCloneScore(self.score_norm(cln1.chainA, cln2.chainA),
                                self.score_norm(cln1.chainB, cln2.chainB))

    def score_dist_paired(self, cln1: PairedChainClone, cln2: PairedChainClone) -> PairedCloneScore:
        return PairedCloneScore(self.score_dist(cln1.chainA, cln2.chainA),
                                self.score_dist(cln1.chainB, cln2.chainB))
