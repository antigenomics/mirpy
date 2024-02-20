import pandas as pd

# class PublicClonotypesSelectionMethod(Enum):
from mir.common.clonotype import ClonotypeAA


class ClonotypeUsageTable:
    """
    A class which stores the clonotype usage matrix. Can be created from a dataset of repertoires (ReperoireDataset).
    The class stores the matrix, public clonotypes and the link to the repertoire dataset it was created from.

    """
    def __init__(self, clonotype_matrix, repertoire_dataset, public_clonotypes):
        """
        Creating a new clonotypeUsageMatrix object

        :param clonotype_matrix: a clonotype matrix itself, should be a pd.DataFrame where each line represents a sample, each column is a clonotype
        :param repertoire_dataset: a repertoire dataset the matrix was derived from
        :param public_clonotypes: a Python list of public clonotypes (by default which are met in at least two samples)
        """
        self.clonotype_matrix = clonotype_matrix
        self.repertoire_dataset = repertoire_dataset
        self.public_clonotypes = public_clonotypes

    @classmethod
    def load_from_repertoire_dataset(cls, repertoire_dataset,
                                     clonotypes_count_for_public_extraction=2,
                                     method_for_public_extraction='unique-occurence',
                                     mismatch_max=1, threads=32):
        """
        A function which creates a ClonotypeUsageMatrix object from repertoire dataset

        :param repertoire_dataset: a repertoire dataset the matrix was derived from
        :param clonotypes_count_for_public_extraction: the number of top clones to be chosen to be considered \
        as public for ['top', 'rnadom-uniform', 'random-roulette'] methods; the number of samples where the clone should \
        be found to be called public for 'unique-occurence method'
        :param method_for_public_extraction: the method for extracting public clonotypes; can be one of \
        ['top',  'random-roulette', 'random-uniform', 'unique-occurence']
        :param mismatch_max: number of mismatches allowed for the clones to be considered similar (e.g. for mismatch=1 \
        clones ``CASS`` and ``CASR`` are similar whereas ``CASS`` and ``CAIR`` are different)
        :param threads: number of threads provided for the processing
        :return: a `ClonotypeUsageMatrix` object
        """
        public_clonotypes = ClonotypeUsageTable.extract_public_clonotypes_for_dataset(
            repertoire_dataset=repertoire_dataset,
            method=method_for_public_extraction,
            count_of_clones=clonotypes_count_for_public_extraction)

        from mir.comparative.match import MultipleRepertoireDenseMatcher
        dense_repertoire_matcher = MultipleRepertoireDenseMatcher(mismatch_max=mismatch_max)
        clonotype_matrix = dense_repertoire_matcher.create_clonotype_matrix_for_clones(
            public_clonotypes, repertoire_dataset, threads)

        return cls(clonotype_matrix, repertoire_dataset, public_clonotypes)

    @staticmethod
    def extract_public_clonotypes_for_dataset(repertoire_dataset,
                                              method='unique-occurence',
                                              count_of_clones=2,
                                              ):
        """
        A function which searches for the public clonotypes within a given dataset

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
            cur_data = pd.DataFrame({'cdr3aa': [x.cdr3aa for x in run.clonotypes], 'count': 1})
            datasets_to_concat.append(cur_data)
        full_data = pd.concat(datasets_to_concat)
        full_data = full_data[full_data.cdr3aa.str.isalpha()]
        print(len(full_data))
        top = full_data.groupby(['cdr3aa'], as_index=False).count()

        if method == 'top':
            top = top.sort_values(by=['count'], ascending=False).head(count_of_clones)
        elif method == 'random-roulette':
            top = top.sample(n=count_of_clones, random_state=42, weights='count')
        elif method == 'random-uniform':
            top = top.sample(n=count_of_clones, random_state=42)
        elif method == 'unique-occurence':
            top = top[top['count'] >= count_of_clones]
        return list(top.cdr3aa)

    def get_clone_usage(self, clonotype):
        """
        A function which finds the number of clonotype occurences within the dataset. Different nucleotide sequences
        which are translated into the same CDR3aa within one patient are supposed as unique occurences

        :param clonotype: a string representing the CDR3 amino acid sequence or a `ClonotypeAA` object
        :return: A `float` number of occurences for the given clonotype
        """
        if isinstance(clonotype, ClonotypeAA):
            clonotype = clonotype.cdr3aa
        if clonotype not in self.public_clonotypes:
            return 0
        return self.clonotype_matrix[clonotype].sum()
