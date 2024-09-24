from datetime import datetime

import pandas as pd

# class PublicClonotypesSelectionMethod(Enum):
from mir.common.clonotype import ClonotypeAA
from mir.comparative.pair_matcher import PairMatcher


class ClonotypeUsageTable:
    """
    A class which stores the clonotype usage matrix. Can be created from a dataset of repertoires (ReperoireDataset).
    The class stores the matrix, public clonotypes and the link to the repertoire dataset it was created from.

    """

    def __init__(self, public_clonotypes, repertoire_dataset,
                 mismatch_max=1,
                 threads=32,
                 with_counts=True,
                 pair_matcher=PairMatcher()):
        """
        Creating a new clonotypeUsageMatrix object

        :type pair_matcher: PairMatcher
        :param public_clonotypes: a list of ClonotypeRepresentation for the clonotypes which should be included into usage table
        :param repertoire_dataset: a repertoire dataset the matrix was derived from
        :param mismatch_max: a parameter which specifies the number of mismatches to search within (TODO add description for blosum62)
        :param threads: number of python processes to perform multiprocessing with
        :param with_counts: whether to consider counts for clonotypes (number of reads the clonotype is represented with) or not
        """
        self.repertoire_dataset = repertoire_dataset
        self.mismatch_max = mismatch_max
        self.threads = threads
        self.__clonotype_database_usage = None
        self.public_clonotypes = public_clonotypes
        self.clonotype_to_matrix_index = {x: i for i, x in enumerate(self.public_clonotypes)}
        self.pair_matcher = pair_matcher
        self.with_counts = with_counts

    @property
    def clonotype_database_usage(self):
        """
        a property method which created the usage database
        :return: the clonotype database object
        """
        if self.__clonotype_database_usage is None:
            from mir.comparative.match import MultipleRepertoireDenseMatcher
            dense_repertoire_matcher = MultipleRepertoireDenseMatcher(mismatch_max=self.mismatch_max)
            self.__clonotype_database_usage = dense_repertoire_matcher.get_clonotype_database_usage_for_cohort(
                self.public_clonotypes,
                self.repertoire_dataset,
                self.threads,
                self.pair_matcher,
                self.with_counts
            )
        return self.__clonotype_database_usage

    @classmethod
    def load_from_repertoire_dataset(cls, repertoire_dataset,
                                     clonotypes_count_for_public_extraction=2,
                                     method_for_public_extraction='unique-occurence',
                                     mismatch_max=1, threads=32, public_clonotypes=None,
                                     with_counts=True,
                                     pair_matcher=PairMatcher()):
        """
        TODO make this method different in order to pass the method which assimes two clonotypes are similar
        TODO currently it is lambda x, y: hamming(x, y) <= 1
        A function which creates a ClonotypeUsageMatrix object from repertoire dataset

        :param pair_matcher: the PairMatcher object, needed to perform difficult comparisons
        :param with_counts: whether to consider counts for clonotypes (number of reads the clonotype is represented with) or not
        :param public_clonotypes: the list of clonotypes which usage should be calculated in the clonotype matrix. \
        if None all public clonotypes would be found in the given dataset
        :param repertoire_dataset: a repertoire dataset the matrix was derived from
        :param clonotypes_count_for_public_extraction: the number of top clones to be chosen to be considered \
        as public for ['top', 'random-uniform', 'random-roulette'] methods; the number of samples where the clone should \
        be found to be called public for 'unique-occurence method'
        :param method_for_public_extraction: the method for extracting public clonotypes; can be one of \
        ['top',  'random-roulette', 'random-uniform', 'unique-occurence']
        :param mismatch_max: number of mismatches allowed for the clones to be considered similar (e.g. for mismatch=1 \
        clones ``CASS`` and ``CASR`` are similar whereas ``CASS`` and ``CAIR`` are different)
        :param threads: number of threads provided for the processing
        :return: a `ClonotypeUsageMatrix` object
        """
        if public_clonotypes is None:
            print(f'started public clonotypes extraction at {datetime.now()}')
            public_clonotypes = ClonotypeUsageTable.extract_public_clonotypes_for_dataset(
                repertoire_dataset=repertoire_dataset,
                method=method_for_public_extraction,
                count_of_clones=clonotypes_count_for_public_extraction,
                pair_matcher=pair_matcher
            )
            print(f'finished public clonotypes extraction at {datetime.now()}')
        print(f'there are {len(public_clonotypes)} public clonotypes')

        return cls(public_clonotypes, repertoire_dataset, mismatch_max,
                   threads, with_counts, pair_matcher)

    @staticmethod
    def extract_public_clonotypes_for_dataset(repertoire_dataset,
                                              method='unique-occurence',
                                              count_of_clones=2,
                                              pair_matcher=PairMatcher()
                                              ):
        """
        A function which searches for the public clonotypes within a given dataset

        :param pair_matcher: the PairMatcher object, needed to perform difficult comparisons
        :param repertoire_dataset:  repertoire dataset the clonotypes should be derived from
        :param method: the method for extracting public clonotypes; can be one of \
        ``['top',  'random-roulette', 'random-uniform', 'unique-occurence']``
        :param count_of_clones: the number of top clones to be chosen to be considered \
        as public for ``['top', 'rnadom-uniform', 'random-roulette']`` methods; the number of samples where the clone should \
        be found to be called public for 'unique-occurence method'
        :return: a Python list of public clonotypes within the given dataset
        """
        datasets_to_concat = []
        for run in repertoire_dataset:
            cur_data = pd.DataFrame(
                {'cdr3aa': [pair_matcher.get_clonotype_repr(x) for x in run.clonotypes if x.cdr3aa.isalpha()],
                 'count': 1})
            datasets_to_concat.append(cur_data)
        full_data = pd.concat(datasets_to_concat)
        # full_data = full_data[full_data.cdr3aa.str.isalpha()]
        top = full_data.groupby(['cdr3aa'], as_index=False).count()

        if method == 'top':
            top = top.sort_values(by=['count'], ascending=False).head(count_of_clones)
        elif method == 'random-roulette':
            top = top.sample(n=count_of_clones, random_state=42, weights='count')
        elif method == 'random-uniform':
            top = top.sample(n=count_of_clones, random_state=42)
        elif method == 'unique-occurence':
            top = top[top['count'] >= count_of_clones]
        print(f'there are {len(top)} public clonotypes')
        return list(top.cdr3aa)

    def get_clone_usage(self, clonotype):
        """
        A function which finds the number of clonotype occurences within the dataset. Different nucleotide sequences
        which are translated into the same CDR3aa within one patient are supposed as unique occurences

        :param clonotype: a string representing the CDR3 amino acid sequence or a `ClonotypeAA` object
        :return: A `float` number of occurences for the given clonotype
        """

        if isinstance(clonotype, str):
            clonotype = ClonotypeAA(cdr3aa=clonotype)
        if isinstance(clonotype, ClonotypeAA):
            clonotype = self.pair_matcher.get_clonotype_repr(clonotype)
        if clonotype not in self.clonotype_to_matrix_index:
            return 0
        return self.clonotype_database_usage[:, [self.clonotype_to_matrix_index[clonotype]]].sum()
