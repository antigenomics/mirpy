from Bio import Align


class AlignCDR:
    def __init__(self, 
                 gap_positions = (3, 4, -4, -3),
                 mat = Align.substitution_matrices.load("BLOSUM62"),
                 gap_penalty = -5):
        self.gap_positions = gap_positions
        self.mat = mat
        self.gap_penalty = gap_penalty
        
    def pad(self, s1, s2) -> tuple[tuple[str, str]]:
        d = len(s1) - len(s2)
        if d == 0:
            return tuple([(s1, s2)])
        elif d < 0:
            return tuple((s1[:p] + ('-' * d) + s1[p:], s2) for p in self.gap_positions)
        else:
            return tuple((s1, s2[:p] + ('-' * d) + s2[p:]) for p in self.gap_positions)
        
    def __score(self, s1, s2) -> float:
        x = 0
        for i in range(len(s1)):
            c1 = s1[i]
            c2 = s2[i]
            if c1 == '-' or c2 == '-':
                x = x + self.gap_penalty
            else:
                x = x + self.mat[c1, c2]
        return x
    
    def alns(self, s1, s2) -> tuple[tuple[str, str, float]]:
        return tuple((sp1, sp2, self.__score(sp1, sp2)) for (sp1, sp2) in self.pad(s1, s2))
    
    def score(self, s1, s2) -> float:
        return max(self.__score(sp1, sp2) for (sp1, sp2) in self.pad(s1, s2))
    
    def score_norm(self, s1, s2) -> float:
        score1 = sum(self.mat[c, c] for c in s1)
        score2 = sum(self.mat[c, c] for c in s2)
        return self.score(s1, s2) - max(score1, score2)


class AlignGermline:
    def __init__(self, dist : dict[tuple[str, str], float]):
        self.dist = dist
    
    def score(self, g1, g2) -> float:
        return self.dist[tuple(g1, g2)]
    
    def score_norm(self, g1, g2) -> float:
        return self.score(g1, g2) - max(self.score(g1, g1), self.score(g2, g2))
    
    @classmethod
    def from_seqs(cls,
                  seqs : dict[str, str],
                  aligner = Align.PairwiseAligner("blastp")):
        dists = {}
        seqs = list(seqs.items())
        for (g1, s1) in seqs:
            for (g2, s2) in seqs:
                score = aligner.align(s1, s2).score
                if g1 >= g2:
                    dists[(g1, g2)] = score
                    dists[(g2, g1)] = score
        return cls(dists)
    

class AlignDefault:
    def __init__(self, aligner = Align.PairwiseAligner("blastp")):
        self.aligner = aligner

    def score(self, s1, s2) -> float:
        return self.aligner.align(s1, s2).score
    
    def score_norm(self, s1, s2) -> float:
        return self.score(s1, s2) - max(self.score(s1, s1), self.score(s2, s2))