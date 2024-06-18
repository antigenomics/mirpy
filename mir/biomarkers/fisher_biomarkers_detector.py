from multiprocessing import Pool, Manager

import numpy as np
from multipy.fdr import lsu
from scipy.stats import fisher_exact

from mir.common.repertoire_dataset import RepertoireDataset
from pympler.asizeof import asizeof
from tqdm.contrib.concurrent import process_map, thread_map
from datetime import datetime

def get_p_value_for_one_clonotype(args):
    """
    Performs fisher exact test for one clonotype

    :param clonotype: a clonotype to assess
    :return: the Fisher exact test p-value result
    """
    clonotype, ill_data_clonotype_usage, control_data_clonotype_usage, ill_joint_number_of_clones, healthy_joint_number_of_clones = args
    res = fisher_exact([[ill_data_clonotype_usage,
                         control_data_clonotype_usage],
                        [ill_joint_number_of_clones - ill_data_clonotype_usage,
                         healthy_joint_number_of_clones - control_data_clonotype_usage]],
                       alternative='greater')
    return res[1]


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
        self.clonotype_to_p_value = None
        self.threads = threads
        print(f'created a fisher biomarker detector with {self.threads} threads')

    def detect_biomarkers(self, adjusted_p_value=None):
        """
        A function which runs the biomarker detection procedure

        :return: a list of significant clonotypes (cdr3aa sequences), without objects
        """
        # TODO change to return objects?
        all_clonotypes_to_consider = set(self.ill_repertoire_dataset.clonotype_usage_matrix.public_clonotypes).union(
            set(self.control_repertoire_dataset.clonotype_usage_matrix.public_clonotypes)
        )
        print(f'[{datetime.now()}]: there are {len(all_clonotypes_to_consider)} public clonotypes in ill repertoire')

        print(f'[{datetime.now()}]: started creating func arguments')
        all_clonotypes_to_consider = [(x, self.ill_repertoire_dataset.clonotype_usage_matrix.get_clone_usage(x),
                                       self.control_repertoire_dataset.clonotype_usage_matrix.get_clone_usage(x),
                                       self.ill_repertoire_dataset.joint_number_of_clones,
                                       self.control_repertoire_dataset.joint_number_of_clones) for x in
                                      all_clonotypes_to_consider]
        print(f'[{datetime.now()}]: finished creating func arguments')
        print(f'[{datetime.now()}]: chunksize is {len(all_clonotypes_to_consider) // 1000}')

        pvals = process_map(get_p_value_for_one_clonotype,
                    all_clonotypes_to_consider,
                    max_workers=self.threads,
                    desc='fisher testing in progress',
                    chunksize=max(1, len(all_clonotypes_to_consider) // 1000))
        self.clonotype_to_p_value = {clonotype[0]: pval for clonotype, pval in zip(all_clonotypes_to_consider, pvals)}
        significant_pvals = lsu(np.array(pvals), q=adjusted_p_value if adjusted_p_value else self.adjusted_p_value)
        self.significant_clones = []
        for pval, clone in zip(significant_pvals, all_clonotypes_to_consider):
            if pval:
                self.significant_clones.append(clone[0])

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
