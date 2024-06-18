import os
import unittest
import pandas as pd

from mir.biomarkers.fisher_biomarkers_detector import FisherBiomarkersDetector
from mir.common.clonotype import ClonotypeAA
from mir.common.clonotype_dataset import ClonotypeDataset
from mir.common.parser import VDJtoolsParser
from mir.common.repertoire_dataset import RepertoireDataset

class TestRepertoireDataset(unittest.TestCase):
    def setUp(self):
        self.meta = pd.read_csv('test_repertoires/test_meta.csv')
        self.rd = RepertoireDataset.load(parser=VDJtoolsParser(sep=','),
                                         metadata=self.meta,
                                         threads=1,
                                         paths=[f'test_repertoires/{x}' for x in self.meta.file_name])

        self.ill_rd, self.healthy_rd = self.rd.split_by_metadata_function(splitting_method=lambda x: x.status == 'ill')

        self.fisher = FisherBiomarkersDetector(control_repertoire_dataset=self.healthy_rd,
                                               ill_repertoire_dataset=self.ill_rd,
                                               threads=1)
    def test_dataset_size(self):
        assert len(self.rd.repertoires) == 4

    def test_dataset_gene(self):
        assert self.rd.gene is None

    def test_public_clonotypes_size(self):
        assert len(self.rd.clonotype_usage_matrix.public_clonotypes) == 5

    def test_public_clonotypes(self):
        assert 'CGGGF' in self.rd.clonotype_usage_matrix.public_clonotypes
        assert 'CASTA' in self.rd.clonotype_usage_matrix.public_clonotypes
        assert 'CFRRA' in self.rd.clonotype_usage_matrix.public_clonotypes


    def test_usage_full_matrix_values_for_CGGGF(self):
        assert self.rd.clonotype_usage_matrix.get_clone_usage('CGGGF') == 8

    def test_usage_ill_matrix_values_for_CGGGF(self):
        assert self.ill_rd.clonotype_usage_matrix.get_clone_usage('CGGGF') == 7

    def test_usage_healthy_matrix_values_for_CGGGF(self):
        assert self.healthy_rd.clonotype_usage_matrix.get_clone_usage('CGGGF') == 1

    def test_fisher_correctness(self):
        markers = self.fisher.detect_biomarkers(adjusted_p_value=0.05*5)
        print(self.fisher.clonotype_to_p_value)
        assert 'CGGGF' in markers
        assert self.fisher.clonotype_to_p_value['CGGGF'] - 0.04830917874396135 < 0.0001

    def test_clustering(self):
        markers = self.fisher.detect_biomarkers(adjusted_p_value=0.05 * 5)
        cd = ClonotypeDataset([ClonotypeAA(cdr3aa=x) for x in markers])
        assert len(cd.clonotype_clustering.cluster.unique()) == 1
        cd.serialize()

if __name__ == "__main__":
    print(os.getcwd())
    unittest.main()