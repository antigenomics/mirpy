from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
from multipy.fdr import lsu
from scipy.stats import fisher_exact
from tqdm import tqdm

from mir.common.repertoire_dataset import RepertoireDataset


class FisherBiomarkersDetector:
    def __init__(self, control_repertoire_dataset: RepertoireDataset,
                 ill_repertoire_dataset: RepertoireDataset, adjusted_p_value=0.05, threads=32):
        self.adjusted_p_value = adjusted_p_value
        self.control_repertoire_dataset = control_repertoire_dataset
        self.ill_repertoire_dataset = ill_repertoire_dataset
        self.clonotype_to_p_value = Manager().dict()
        self.threads = 32

    def get_p_value_for_one_clonotype(self, clonotype):
        ill_data_clonotype_usage = self.ill_repertoire_dataset.clonotype_usage_matrix.get_clone_usage(clonotype)
        control_data_clonotype_usage = self.control_repertoire_dataset.clonotype_usage_matrix.get_clone_usage(clonotype)
        res = fisher_exact([[ill_data_clonotype_usage,
                             self.ill_repertoire_dataset.joint_number_of_clones - ill_data_clonotype_usage],
                            [control_data_clonotype_usage,
                             self.control_repertoire_dataset.joint_number_of_clones - control_data_clonotype_usage]],
                           alternative='greater')
        self.clonotype_to_p_value[clonotype] = res[1]

    def detect_biomarkers(self):
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
    clonotype_matrix = clonotype_matrix.set_index('cdr3aa').T.reset_index().rename(columns={'index': 'run'})
    useful_columns = [x for x in significant_clones[clone_column] if x in clonotype_matrix.columns] + ['run']
    return clonotype_matrix[useful_columns]

