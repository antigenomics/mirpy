# todo: tcrnet/alice
from mir.common.clonotype_dataset import ClonotypeDataset


class TcrNet:
    def __init__(self, graph):
        pass

    def load_from_clonotype_dataset(self, cd: ClonotypeDataset):
        clusters = cd.clonotype_clustering
        cdr3aa_to_clonotype = cd.clonotypes_cdr3aa