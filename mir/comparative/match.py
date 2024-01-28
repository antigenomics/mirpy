# todo: vdjmatch
from collections import defaultdict, Counter
from multiprocessing import Pool, Manager
from pyparsing import Iterable
import pandas as pd
from mir.common.clonotype import Clonotype, ClonotypeAA
from mir.common.repertoire_dataset import RepertoireDataset
from ..distances import ClonotypeAligner, ClonotypeScore


class DatabaseMatch:
    __slots__ = ['db_clonotype', 'scores']

    def __init__(self, db_clonotype: Clonotype, scores: ClonotypeScore):
        self.db_clonotype = db_clonotype
        self.scores = scores

    def __dict__(self):
        return {str(self.db_clonotype.id) + '_v_score': self.scores.v_score,
                str(self.db_clonotype.id) + '_j_score': self.scores.j_score,
                str(self.db_clonotype.id) + '_cdr3_score': self.scores.cdr3_score}

    def __str__(self):
        return f'(v:{self.scores.v_score},j:{self.scores.j_score},cdr3:{self.scores.cdr3_score})'


class DatabaseMatches:
    __slots__ = ['clonotype', 'matches']

    def __init__(self, clonotype: Clonotype, matches: Iterable[DatabaseMatch]):
        self.clonotype = clonotype
        self.matches = matches

    def __dict__(self):
        d = {'id': self.clonotype.id}
        for m in self.matches:
            d.update(m.__dict__())
        return d


class DenseMatcher:
    def __init__(self,
                 database: list[ClonotypeAA],
                 aligner: ClonotypeAligner,
                 norm_scoring: bool = False):
        self.database = database
        if norm_scoring:
            self._score = aligner.score_norm
        else:
            self._score = aligner.score

    def match_single(self, clonotype: ClonotypeAA) -> list[DatabaseMatch]:
        return [DatabaseMatch(c, self._score(c, clonotype)) for c in self.database]

    def _match_single_wrapper(self, clonotype: ClonotypeAA) -> DatabaseMatches:
        return DatabaseMatches(clonotype, self.match_single(clonotype))

    def match(self, clonotypes: list[ClonotypeAA],
              nproc=1, chunk_sz=1) -> Iterable[DatabaseMatches]:
        if nproc == 1:
            matches = map(self._match_single_wrapper, clonotypes)
        else:
            with Pool(nproc) as pool:
                matches = pool.map(
                    self._match_single_wrapper, clonotypes, chunk_sz)
        return matches

    def match_to_df(self, clonotypes: list[ClonotypeAA],
                    nproc=1, chunk_sz=16) -> pd.DataFrame:
        return pd.DataFrame.from_records([m.__dict__() for m in self.match(clonotypes,
                                                                           nproc,
                                                                           chunk_sz)])


class SparseMatcher:
    pass


class MultipleRepertoireDenseMatcher:
    def __init__(self, mismatch_max=1):
        self.mismatch_max = mismatch_max
        self.length_to_mismatch_clones = {}
        self.mismatch_clone_to_cdr3aa = defaultdict(set)

    @staticmethod
    def check_mismatch_clone(cur_clone, mismatch_clones):
        occurences = set()
        for i in range(len(cur_clone)):
            clone_to_search = cur_clone[: i] + 'X' + cur_clone[i + 1:]
            cur_clones_set = mismatch_clones[i]
            if clone_to_search in cur_clones_set:
                occurences.add(clone_to_search)
        return occurences

    def check(self, clone1, clone2):
        ans = 0
        for c1, c2 in zip(clone1, clone2):
            if c1 != c2:
                ans += 1
        return ans <= self.mismatch_max

    def create_clonotype_matrix_for_clones(self, most_common_clonotypes, repertoire_dataset: RepertoireDataset, threads=32):
        global run_to_presence_of_clonotypes
        run_to_presence_of_clonotypes = Manager().dict()
        global process_one_file

        from mir.common.repertoire import Repertoire

        def process_one_file(x):
            run, i = x
            res = []
            cur_cdrs = Counter([x.cdr3aa for x in run.clonotypes])
            length_to_clones = defaultdict(set)
            for clonotype in cur_cdrs:
                length_to_clones[len(clonotype.cdr3aa)].add(clonotype.cdr3aa)

            length_to_mismatch_clones = {}
            mismatch_clone_to_cdr3aa = defaultdict(set)
            for length, cdr_set in length_to_clones.items():
                length_to_mismatch_clones[length] = defaultdict(set)
                for clone in cdr_set:
                    if not clone.isalpha():
                        continue
                    for i in range(len(clone)):
                        mismatch_clone = clone[:i] + 'X' + clone[i + 1:]
                        length_to_mismatch_clones[length][i].add(mismatch_clone)
                        mismatch_clone_to_cdr3aa[mismatch_clone].add(clone)

            for clone in most_common_clonotypes['cdr3aa']:
                if len(clone) in length_to_mismatch_clones:
                    mismatch_clones = MultipleRepertoireDenseMatcher.check_mismatch_clone(
                        clone,
                        length_to_mismatch_clones[len(clone)])
                    cdr3aa_found_clones = set()
                    for mismatch_clone in mismatch_clones:
                        cdr3aa_found_clones.update(mismatch_clone_to_cdr3aa[mismatch_clone])
                    sum_occurences = 0
                    for cdr3_mismatch_clone in cdr3aa_found_clones:
                        if self.check(clone, cdr3_mismatch_clone):
                            sum_occurences += cur_cdrs[cdr3_mismatch_clone]
                    res.append(sum_occurences)
                else:
                    res.append(0)
            run_to_presence_of_clonotypes[x] = pd.Series(res)

        run_to_presence_of_clonotypes['cdr3aa'] = pd.Series(most_common_clonotypes['cdr3aa'])
        runs = [(x, i) for i, x in enumerate(repertoire_dataset.repertoires)]
        with Pool(threads) as p:
            p.map(process_one_file, runs)
        data = {x: y for x, y in run_to_presence_of_clonotypes.items()}
        return pd.DataFrame.from_dict(data=data)

