# todo: vdjmatch
from typing import Set

import math
import sys
from collections import defaultdict, Counter
from multiprocessing import Pool, Manager
from pyparsing import Iterable
import pandas as pd
from mir.common.clonotype import Clonotype, ClonotypeAA
from mir.common.repertoire_dataset import RepertoireDataset
from mir.comparative.pair_matcher import PairMatcher
from mir.distances.aligner import ClonotypeAligner, ClonotypeScore, CDRAligner, Scoring
from tqdm.contrib.concurrent import process_map
from scipy.sparse import lil_array, vstack
import time
from datetime import datetime
from tqdm import tqdm


# from memory_profiler import profile


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


class SubstitutionMatrixSearchRepertoire:
    def __init__(self, repertoire,
                 pair_matcher: PairMatcher):
        self.clonotypes = repertoire.clonotypes
        self.matcher = pair_matcher

    def find_matches_count_in_database(self, clonotype):
        matches_count = 0
        for c in self.clonotypes:
            if len(c.cdr3aa) == len(clonotype.cdr3aa) and self.matcher.check_repr_similar(c, clonotype):
                matches_count += 1
        return matches_count


class XEncodedRepertoire:
    def __init__(self, repertoire,
                 pair_matcher: PairMatcher,
                 with_counts=True):
        if with_counts:
            self.exact_cdr3_seqs = defaultdict(int)
            for x in repertoire:
                self.exact_cdr3_seqs[pair_matcher.get_clonotype_repr(x)] += x.cells
        else:
            self.exact_cdr3_seqs = Counter([pair_matcher.get_clonotype_repr(x) for x in repertoire.clonotypes])
        self.length_to_clones = defaultdict(set)
        for clonotype in repertoire:
            self.length_to_clones[len(clonotype.cdr3aa)].add(clonotype)
        self.length_to_mismatch_clones = {}
        self.mismatch_clone_to_clono_representation = defaultdict(set)
        self.pair_matcher = pair_matcher
        for length, cdr_set in self.length_to_clones.items():
            self.length_to_mismatch_clones[length] = defaultdict(set)
            for clone in cdr_set:
                if not clone.cdr3aa.isalpha():
                    continue
                for i in range(len(clone.cdr3aa)):
                    mismatch_clone = clone.cdr3aa[:i] + 'X' + clone.cdr3aa[i + 1:]
                    self.length_to_mismatch_clones[length][i].add(mismatch_clone)
                    self.mismatch_clone_to_clono_representation[mismatch_clone].add(
                        pair_matcher.get_clonotype_repr(clone))

    # @staticmethod
    def check_distance(self, clone1, clone2):
        if not self.pair_matcher.check_repr_similar(clone1, clone2):
            return False
        ans = 0
        for c1, c2 in zip(clone1.cdr3aa, clone2.cdr3aa):
            if c1 != c2:
                ans += 1
        return ans <= 1

    def check_mismatch_clone(self, cur_clone):
        mismatch_occurences = set()
        mismatch_clones = self.length_to_mismatch_clones[len(cur_clone)]
        for i in range(len(cur_clone)):
            x_enc_clone_to_search = cur_clone[: i] + 'X' + cur_clone[i + 1:]
            cur_clones_set = mismatch_clones[i]
            if x_enc_clone_to_search in cur_clones_set:
                mismatch_occurences.add(x_enc_clone_to_search)
        return mismatch_occurences

    def find_matches_count_in_database(self, clonotype):
        if len(clonotype.cdr3aa) in self.length_to_mismatch_clones:
            mismatch_clones = self.check_mismatch_clone(clonotype.cdr3aa)
            found_mismatch_representations: set[set] = set()
            for mismatch_clone in mismatch_clones:
                found_mismatch_representations.update(self.mismatch_clone_to_clono_representation[mismatch_clone])
            sum_occurences = 0
            for cdr3_mismatch_clone in found_mismatch_representations:
                if self.check_distance(clonotype, cdr3_mismatch_clone):
                    sum_occurences += self.exact_cdr3_seqs[cdr3_mismatch_clone]
            return sum_occurences
        else:
            return 0


# @profile
def get_clonotypes_usage_for_repertoire_chunk(args):
    rep_indices, rd, clonotypes_for_analysis, pair_matcher, mismatch_max, with_counts, chunk_idx = args
    # print(
    #     f'chunk num {chunk_idx}, rep_indices in chunk {len(rep_indices)}, chunk size is {asizeof(rep_indices) / 1024 ** 2}, clonotypes object size is {asizeof(clonotypes_for_analysis) / 1024 ** 2}')
    current_matrix = lil_array((len(rep_indices), len(clonotypes_for_analysis)))
    for i, rep_index in enumerate(rep_indices):
        # print(f'[{datetime.now()}, {chunk_idx}]: started {i} rep in chunk')
        t0 = time.time()
        if mismatch_max == 1:  # TODO fix the mismatches and substitutions changes
            encoded_repertoire = XEncodedRepertoire(rd[rep_index],
                                                    pair_matcher=pair_matcher,
                                                    with_counts=with_counts)
        else:
            encoded_repertoire = SubstitutionMatrixSearchRepertoire(repertoire=rd[rep_index],
                                                                    pair_matcher=pair_matcher)
        t1 = time.time()
        # print(f'[{datetime.now()}, {chunk_idx}]: created XEncoded in {t1 - t0}, size {asizeof(encoded_repertoire) / 1024 ** 2}')
        for j, clone in enumerate(clonotypes_for_analysis):
            found_matches_count = encoded_repertoire.find_matches_count_in_database(clone)
            if found_matches_count > 0:
                current_matrix[i, j] += found_matches_count
        # print(
        #     f'[{datetime.now()}, {chunk_idx}]: browsed through all the clones in {time.time() - t1}, matrix size {asizeof(current_matrix) / 1024 ** 2}')
        del encoded_repertoire
        # del rep_indices[i]
    return current_matrix


class MultipleRepertoireDenseMatcher:
    def __init__(self, mismatch_max=1):
        print('created MultipleRepertoireDenseMatcher')
        self.mismatch_max = mismatch_max
        self.length_to_mismatch_clones = {}
        self.mismatch_clone_to_cdr3aa = defaultdict(set)
        self.clonotypes_to_choose_from = None

    # @profile
    def get_clonotype_database_usage_for_cohort(self,
                                                most_common_clonotypes,
                                                repertoire_dataset,
                                                threads=4,
                                                pair_matcher=PairMatcher(),
                                                with_counts=True):
        print(f'started with {threads} threads')
        self.clonotypes_to_choose_from = most_common_clonotypes
        # print(f'repertoire dataset size is {asizeof(repertoire_dataset) / 1024 ** 2}')

        repertoire_dataset.serialize_repertoires()

        data_size = len(repertoire_dataset.repertoires)
        chunk_size = min(8, math.ceil(data_size / threads))
        iters = max(1, data_size // chunk_size + math.ceil(data_size / chunk_size - data_size // chunk_size))

        print(f'all in all {data_size} reps, chunk size is {chunk_size}, number of batches {iters}')
        resulting_values = process_map(get_clonotypes_usage_for_repertoire_chunk,
                                       [([i for i in range(chunk_size * i, min(chunk_size * (i + 1), data_size))],
                                         repertoire_dataset,
                                         most_common_clonotypes,
                                         pair_matcher,
                                         self.mismatch_max,
                                         with_counts,
                                         i) for i in
                                        range(iters)],
                                       max_workers=threads, desc='clonotype usage matrix preparation')
        clonotype_usage_matrix = vstack(resulting_values).tocsc()
        return clonotype_usage_matrix
