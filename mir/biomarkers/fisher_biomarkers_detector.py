from multiprocessing import Pool, Manager

import numpy as np
from multipy.fdr import lsu
from scipy.stats import fisher_exact

from mir.common.repertoire_dataset import RepertoireDataset


class FisherBiomarkersDetector:
    """
    A class which is made to detect phenotype associated clonotypes (biomarkers). It uses Fisher exact test for
    comparison and adjusts the p-values afterwards.
    """

    def __init__(self, control_repertoire_dataset: RepertoireDataset,
                 ill_repertoire_dataset: RepertoireDataset, adjusted_p_value=0.05, threads=32):
        """
        The initializing function. It takes two repertoires of control and ill people.

        :param control_repertoire_dataset: a `RepertoireDataset` object with control samples in it
        :param ill_repertoire_dataset: a `RepertoireDataset` object with ill samples in it
        :param adjusted_p_value: the desired adjusted p-value to assume the clonotype is a biomarker
        :param threads: number of threads to perform analysis with
        """
        self.adjusted_p_value = adjusted_p_value
        self.control_repertoire_dataset = control_repertoire_dataset
        self.ill_repertoire_dataset = ill_repertoire_dataset
        self.clonotype_to_p_value = Manager().dict()
        self.threads = threads

    def get_p_value_for_one_clonotype(self, clonotype):
        """
        Performs fisher exact test for one clonotype

        :param clonotype: a clonotype to assess
        :return: the Fisher exact test p-value result
        """
        ill_data_clonotype_usage = self.ill_repertoire_dataset.clonotype_usage_matrix.get_clone_usage(clonotype)
        control_data_clonotype_usage = self.control_repertoire_dataset.clonotype_usage_matrix.get_clone_usage(clonotype)
        res = fisher_exact([[ill_data_clonotype_usage,
                             self.ill_repertoire_dataset.joint_number_of_clones - ill_data_clonotype_usage],
                            [control_data_clonotype_usage,
                             self.control_repertoire_dataset.joint_number_of_clones - control_data_clonotype_usage]],
                           alternative='greater')
        self.clonotype_to_p_value[clonotype] = res[1]

    def detect_biomarkers(self):
        """
        A function which runs the biomarker detection procedure

        :return: a list of significant clonotypes (cdr3aa sequences), without objects
        """
        # TODO change to return objects?
        all_clonotypes_to_consider = list(
            set(self.control_repertoire_dataset.clonotype_usage_matrix.public_clonotypes).union(
                set(self.ill_repertoire_dataset.clonotype_usage_matrix.public_clonotypes)))

        with Pool(self.threads, maxtasksperchild=2) as p:
            p.map(self.get_p_value_for_one_clonotype, all_clonotypes_to_consider)
            print('finished testing')

        pvals = []
        for clone in all_clonotypes_to_consider:
            pvals.append(self.clonotype_to_p_value[clone])
        significant_pvals = lsu(np.array(pvals), q=self.adjusted_p_value)
        self.significant_clones = []
        for pval, clone in zip(significant_pvals, all_clonotypes_to_consider):
            if pval:
                self.significant_clones.append(clone)

        return self.significant_clones


def create_significant_clonotype_matrix(clonotype_matrix, significant_clones, clone_column='clone'):
    """
    A function which creates a clonotype matrix with only a subset of clonotypes

    :param clonotype_matrix: a `pd.DataFrame` object which contains clonotype usage matrix
    :param significant_clones: a list of significant clones to perform selection on
    :param clone_column: the column in `clonotype_matrix` which contains cdr3aa
    :return: a subselected clonotype matrix
    """
    # TODO move to clonotype usage matrix mb?
    clonotype_matrix = clonotype_matrix.set_index('cdr3aa').T.reset_index().rename(columns={'index': 'run'})
    useful_columns = [x for x in significant_clones[clone_column] if x in clonotype_matrix.columns] + ['run']
    return clonotype_matrix[useful_columns]
