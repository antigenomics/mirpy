import time
from abc import abstractmethod
from itertools import starmap
from multiprocessing import Pool
from Bio import Align
from Bio.Align import substitution_matrices
import numpy as np
import typing as t

from mir.common.clonotype import ClonotypeAA, PairedChainClone
from mir.common.segments import Segment, SegmentLibrary


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
    def __init__(self,
                 gap_positions: t.Iterable[int] = (3, 4, -4, -3),
                 matrix_name: str = 'BLOSUM62',
                 gap_penalty: float = -3.0,
                 v_offset: int = 3,
                 j_offset: int = 3,
                 factor: float = 10.0):
        mat = substitution_matrices.load(matrix_name)
        self.mat = np.array(mat, dtype=np.float32)
        self.aa_to_index = {aa: i for i, aa in enumerate(mat.alphabet)}

        self.gap_positions = list(gap_positions)
        self.gap_penalty = gap_penalty
        self.v_offset = v_offset
        self.j_offset = j_offset
        self.factor = factor

    def get_matrix_distance(self, c1, c2):
        if self.mat is not None:
            return self.mat[c1, c2]
        else:
            return 0 if c1 == c2 else 1

    def _score_shifted(self, s1: str, s2: str, shift: int,
                       shift_pos: int, gap_in_s1: bool) -> float:
        L1, L2 = len(s1), len(s2)
        L = max(L1, L2) + abs(shift)
        score = 0.0
        for i in range(self.v_offset, L - self.j_offset):
            if gap_in_s1:
                pos1 = i if i < shift_pos else i - shift
                pos2 = i
            else:
                pos1 = i
                pos2 = i if i < shift_pos else i + shift

            if pos1 < 0 or pos1 >= L1 or pos2 < 0 or pos2 >= L2:
                score += self.gap_penalty
            else:
                a1, a2 = s1[pos1], s2[pos2]
                i1, i2 = self.aa_to_index[a1], self.aa_to_index[a2]
                score += self.mat[i1, i2]

        return score * self.factor

    def score(self, s1: str, s2: str) -> float:
        best = float('-inf')
        d = len(s1) - len(s2)
        for p in self.gap_positions:
            if d == 0:
                val = 0.0
                for i in range(self.v_offset, len(s1) - self.j_offset):
                    a1, a2 = s1[i], s2[i]
                    i1, i2 = self.aa_to_index[a1], self.aa_to_index[a2]
                    val += self.mat[i1, i2]
                val *= self.factor
            else:
                shift = d if d > 0 else d
                gap_in_s1 = d < 0
                shift_pos = p if p >= 0 else (len(s2) + p if gap_in_s1 else len(s1) + p)
                val = self._score_shifted(s1, s2, shift, shift_pos, gap_in_s1)

            if val > best:
                best = val
        return best

    def score_norm(self, s1, s2) -> float:
        score1 = self._factor * sum(self.get_matrix_distance(c, c) for c in s1)
        score2 = self._factor * sum(self.get_matrix_distance(c, c) for c in s2)
        return self.score(s1, s2) - max(score1, score2)

    def score_dist(self, s1, s2) -> float:
        return self.score(s1, s1) + self.score(s2, s2) - 2 * self.score(s1, s2)


class _Scoring_Wrapper:
    def __init__(self, scoring: Scoring):
        self.scoring = scoring

    def __call__(self, gs1: tuple[str, str], gs2: tuple[str, str]):
        return ((gs1[0], gs2[0]), self.scoring.score(gs1[1], gs2[1]))


class GermlineAligner:
    def __init__(self, dist: dict[tuple[str, str], float]):
        self.dist = dist
        self.dist.update(dict(((g2, g1), score)
                         for ((g1, g2), score) in dist.items()))

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
        elif isinstance(seqs, list) and isinstance(seqs[0], Segment):
            seqs = dict({s.id, s.seqaa} for s in seqs)
        gen = ((gs1, gs2) for gs1 in seqs for gs2 in seqs if
               gs1[0] >= gs2[0])
        if nproc == 1:
            dist = starmap(scoring_wrapper, gen) # this operation is long, Bio issue:(
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
