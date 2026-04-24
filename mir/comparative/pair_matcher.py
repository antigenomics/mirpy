from dataclasses import dataclass

from mir.distances.aligner import Scoring


@dataclass
class ClonotypeRepresentation:
    junction_aa: str = None
    v_gene: str = None
    j_gene: str = None
    junction: str = None
    embedding = None

    def __eq__(self, other):
        if not isinstance(other, ClonotypeRepresentation):
            return NotImplemented
        return (self.junction_aa == other.junction_aa and
                self.v_gene == other.v_gene and
                self.j_gene == other.j_gene and
                self.junction == other.junction and
                self.embedding == other.embedding)

    def __hash__(self):
        hashed_val = 0
        for x in [self.junction_aa, self.junction, self.j_gene, self.embedding]:
            if x is not None:
                hashed_val += hash(x)
        return hashed_val

    def __lt__(self, other):
        if self.v_gene is not None and other.v_gene is not None and self.v_gene != other.v_gene:
            return self.v_gene < other.v_gene
        if self.j_gene is not None and other.j_gene is not None and self.j_gene != other.j_gene:
            return self.j_gene < other.j_gene
        if self.junction_aa is not None and other.junction_aa is not None and self.junction_aa != other.junction_aa:
            return self.junction_aa < other.junction_aa
        if self.junction is not None and other.junction is not None:
            return self.junction < other.junction
        if self.embedding is not None and other.embedding is not None:
            return self.embedding < other.embedding
        return True


class PairMatcher:
    def __init__(self,
                 clonotype_representation_method='junction_aa',
                 clonotype_comparison_method='any',
                 aligner: Scoring = None,
                 scoring_threshold: dict = None):
        self.__clonotype_representation_method = clonotype_representation_method
        self.__clonotype_comparison_method = clonotype_comparison_method
        self._aligner = aligner
        self._scoring_threshold = scoring_threshold

    def get_clonotype_repr(self, x):
        m = self.__clonotype_representation_method
        if m == 'junction_aa':
            return PairMatcher.repr_cdr3(x, seq_type='aa')
        elif m == 'junction':
            return PairMatcher.repr_cdr3(x, seq_type='nt')
        elif m == 'junction_aa,v_gene':
            return PairMatcher.repr_cdr3_v(x, seq_type='aa')
        elif m == 'junction,v_gene':
            return PairMatcher.repr_cdr3_v(x, seq_type='nt')
        elif m == 'junction_aa,j_gene':
            return PairMatcher.repr_cdr3_j(x, seq_type='aa')
        elif m == 'junction,j_gene':
            return PairMatcher.repr_cdr3_j(x, seq_type='nt')
        elif m == 'junction_aa,v_gene,j_gene':
            return PairMatcher.repr_cdr3_vj(x, seq_type='aa')
        elif m == 'junction,v_gene,j_gene':
            return PairMatcher.repr_cdr3_vj(x, seq_type='nt')
        else:
            raise Exception(f'Clonotype representation method {m!r} is unknown!')

    def check_repr_similar(self, x: ClonotypeRepresentation, y: ClonotypeRepresentation):
        if self.__clonotype_comparison_method == 'any':
            return True

        if not isinstance(x, ClonotypeRepresentation):
            x = self.get_clonotype_repr(x)
        if not isinstance(y, ClonotypeRepresentation):
            y = self.get_clonotype_repr(y)

        m = self.__clonotype_comparison_method
        if m == 'v':
            return x.v_gene == y.v_gene
        elif m == 'j':
            return x.j_gene == y.j_gene
        elif m == 'vj':
            return x.v_gene == y.v_gene and x.j_gene == y.j_gene
        elif m == 'full':
            return x.v_gene == y.v_gene and x.j_gene == y.j_gene and x.junction_aa == y.junction_aa
        elif m == 'substitution':
            if len(x.junction_aa) not in self._scoring_threshold:
                return False
            return self._aligner.score_norm(x.junction_aa, y.junction_aa) > self._scoring_threshold[len(x.junction_aa)]
        else:
            raise Exception(f'Clonotype comparison method {m!r} is unknown!')

    @staticmethod
    def repr_cdr3(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(junction_aa=x.junction_aa)
        else:
            return ClonotypeRepresentation(junction=x.junction)

    @staticmethod
    def repr_cdr3_v(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(junction_aa=x.junction_aa, v_gene=str(x.v_gene))
        else:
            return ClonotypeRepresentation(junction=x.junction, v_gene=str(x.v_gene))

    @staticmethod
    def repr_cdr3_j(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(junction_aa=x.junction_aa, j_gene=str(x.j_gene))
        else:
            return ClonotypeRepresentation(junction=x.junction, j_gene=str(x.j_gene))

    @staticmethod
    def repr_cdr3_vj(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(junction_aa=x.junction_aa, v_gene=str(x.v_gene), j_gene=str(x.j_gene))
        else:
            return ClonotypeRepresentation(junction=x.junction, v_gene=str(x.v_gene), j_gene=str(x.j_gene))
