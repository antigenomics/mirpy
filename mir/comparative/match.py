# todo: vdjmatch

from collections import namedtuple
from multiprocessing import Pool
from pyparsing import Iterable
from mir.common.repertoire import ClonotypeAA
from mir.distances.aligner import ClonotypeAligner, ClonotypeScore


DatabaseMatch = namedtuple('DatabaseMatch', 'db_clonotype scores')
DatabaseMatches = namedtuple('DatabaseMatches', 'clonotype matches')


class DenseMatch:
    def __init__(self, 
                 database : list[ClonotypeAA],
                 aligner : ClonotypeAligner,
                 norm_scoring : bool = False):
        self.database = database
        if norm_scoring:
            self._score = aligner.score_norm
        else:
            self._score = aligner.score

    def match_single(self, clonotype : ClonotypeAA) -> list[DatabaseMatch]:
        return [DatabaseMatch(c, self._score(c, clonotype)) for c in self.database]
    
    def _match_single_wrapper(self, clonotype : ClonotypeAA) -> DatabaseMatches:
        return DatabaseMatches(clonotype, self.match_single(clonotype))
    
    def match(self, clonotypes : list[ClonotypeAA],
                  nproc = 1, chunk_sz = 4096) -> Iterable[DatabaseMatches]:
        if nproc == 1:
            matches = map(self._match_single_wrapper, clonotypes)            
        else:
            with Pool(nproc) as pool:
                matches = pool.map(self._match_single_wrapper, clonotypes, chunk_sz)  
        return matches


class SparseMatch:
    pass