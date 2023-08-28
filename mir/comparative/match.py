# todo: vdjmatch

from multiprocessing import Pool
from pyparsing import Iterable
import pandas as pd
from mir.common.repertoire import Clonotype, ClonotypeAA
from mir.distances.aligner import ClonotypeAligner, ClonotypeScore


class DatabaseMatch:
    __slots__ = ['db_clonotype', 'scores']

    def __init__(self, db_clonotype : Clonotype, scores : ClonotypeScore):
        self.db_clonotype = db_clonotype
        self.scores = scores

    def __dict__(self):
        return {str(self.db_clonotype.id) + '_v_score' : self.scores.v_score,
                str(self.db_clonotype.id) + '_j_score' : self.scores.j_score,
                str(self.db_clonotype.id) + '_cdr3_score' : self.scores.cdr3_score}
    
    def __str__(self):
        return f'(v:{self.scores.v_score},j:{self.scores.j_score},cdr3:{self.scores.cdr3_score})'
    

class DatabaseMatches:
    __slots__ = ['clonotype', 'matches']

    def __init__(self, clonotype : Clonotype, matches : Iterable[DatabaseMatch]):
        self.clonotype = clonotype
        self.matches = matches

    def __dict__(self):
        d = {'id' : self.clonotype.id}
        for m in self.matches:
            d.update(m.__dict__())
        return d


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
                  nproc = 1, chunk_sz = 1) -> Iterable[DatabaseMatches]:
        if nproc == 1:
            matches = map(self._match_single_wrapper, clonotypes)            
        else:
            with Pool(nproc) as pool:
                matches = pool.map(self._match_single_wrapper, clonotypes, chunk_sz)  
        return matches
    
    def match_to_df(self, clonotypes : list[ClonotypeAA],
                    nproc = 1, chunk_sz = 16) -> pd.DataFrame:
        return pd.DataFrame.from_records([m.__dict__() for m in self.match(clonotypes, 
                                                                           nproc,
                                                                           chunk_sz)])


class SparseMatch:
    pass