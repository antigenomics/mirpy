from dataclasses import dataclass


@dataclass
class ClonotypeRepresentation:
    cdr3aa: str=None
    v: str=None
    j: str=None
    cdr3nt: str=None
    embedding=None

    def __eq__(self, other):
        if not isinstance(other, ClonotypeRepresentation):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.cdr3aa == other.cdr3aa and self.v == other.v and self.j == other.j and self.cdr3nt == other.cdr3nt and self.embedding == other.embedding

    def __hash__(self):
        hashed_val = 0
        for x in [self.cdr3aa, self.cdr3nt, self.j, self.j, self.embedding]:
            if x is not None:
                hashed_val += hash(x)
        return hashed_val

    def __lt__(self, other):
        if self.v is not None and other.v is not None and self.v != other.v:
            return self.v < other.v
        if self.j is not None and other.j is not None and self.j != other.j:
            return self.j < other.j
        if self.cdr3aa is not None and other.cdr3aa is not None and self.cdr3aa != other.cdr3aa:
            return self.cdr3aa < other.cdr3aa
        if self.cdr3nt is not None and other.cdr3nt is not None:
            return self.cdr3nt < other.cdr3nt
        if self.embedding is not None and other.embedding is not None:
            return self.embedding < other.embedding
        return True

class PairMatcher:
    def __init__(self,
                 clonotype_representation_method='cdr3aa',
                 clonotype_comparison_method='any'):
        self.__clonotype_representation_method = clonotype_representation_method
        self.__clonotype_comparison_method = clonotype_comparison_method

    def get_clonotype_repr(self, x):
        if self.__clonotype_representation_method == 'cdr3aa':
            return PairMatcher.repr_cdr3(x, seq_type='aa')
        elif self.__clonotype_representation_method == 'cdr3nt':
            return PairMatcher.repr_cdr3(x, seq_type='nt')
        elif self.__clonotype_representation_method == 'cdr3aa,v':
            return PairMatcher.repr_cdr3_v(x, seq_type='aa')
        elif self.__clonotype_representation_method == 'cdr3nt,v':
            return PairMatcher.repr_cdr3_v(x, seq_type='nt')
        elif self.__clonotype_representation_method == 'cdr3aa,j':
            return PairMatcher.repr_cdr3_j(x, seq_type='aa')
        elif self.__clonotype_representation_method == 'cdr3nt,j':
            return PairMatcher.repr_cdr3_j(x, seq_type='nt')
        elif self.__clonotype_representation_method == 'cdr3aa,vj':
            return PairMatcher.repr_cdr3_vj(x, seq_type='aa')
        elif self.__clonotype_representation_method == 'cdr3nt,vj':
            return PairMatcher.repr_cdr3_vj(x, seq_type='nt')
        else:
            raise Exception(f'Clonotype representation mathod {self.__clonotype_representation_method} is unknown!')

    def check_repr_similar(self, x:ClonotypeRepresentation, y: ClonotypeRepresentation):
        if self.__clonotype_comparison_method == 'any':
            return True
        elif self.__clonotype_comparison_method == 'v':
            return x.v == y.v
        elif self.__clonotype_comparison_method == 'j':
            return x.j == y.j
        elif self.__clonotype_comparison_method == 'vj':
            return x.v == y.v and x.j == y.j
        else:
            raise Exception(f'Clonotype comparison mathod {self.__clonotype_comparison_method} is unknown!')

    @staticmethod
    def repr_cdr3(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(cdr3aa=x.cdr3aa)
        else:
            return ClonotypeRepresentation(cdr3nt=x.cdr3nt)
    @staticmethod
    def repr_cdr3_v(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(cdr3aa=x.cdr3aa, v=x.v.id)
        else:
            return ClonotypeRepresentation(cdr3nt=x.cdr3aa, v=x.v.id)
    @staticmethod

    def repr_cdr3_j(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(cdr3aa=x.cdr3aa, j=x.j.id)
        else:
            return ClonotypeRepresentation(cdr3nt=x.cdr3aa, j=x.j.id)

    @staticmethod
    def repr_cdr3_vj(x, seq_type='aa'):
        if seq_type == 'aa':
            return ClonotypeRepresentation(cdr3aa=x.cdr3aa, v=x.v.id, j=x.j.id)
        else:
            return ClonotypeRepresentation(cdr3nt=x.cdr3aa, v=x.v.id, j=x.j.id)
