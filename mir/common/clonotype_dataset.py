from collections import defaultdict

from mir import get_resource_path
from mir.basic.pgen import OlgaModel
from mir.common.clonotype import ClonotypeAA
import igraph as ig
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


class ClonotypeDataset:
    """
    The dataset of clonotypes is stored in this object. To be used to store a set of clonotypes with specific features
    """

    def __init__(self, clonotypes: list[ClonotypeAA]):
        for clonotype in clonotypes:
            if not isinstance(clonotype, ClonotypeAA):
                raise Exception('You must have cdr3aa sequence for ClonotypeDataset creation')
        self.clonotypes = {x.cdr3aa: x for x in clonotypes}
        self.masks = None
        self.masks_to_clonotypes = defaultdict(set)
        self.clusters = None
        self.pgen = None
        self.cluster_pgen = None

    def get_number_of_samples_for_clonotype(self, clonotype_seqaa):
        pass

    def get_masked_clonotypes_set(self):
        if self.masks is not None:
            return self.masks
        self.masks = set()
        for c in self.clonotypes.keys():
            for i in range(len(c)):
                self.masks.add(c[:i] + 'X' + c[i + 1:])
                self.masks_to_clonotypes[c[:i] + 'X' + c[i + 1:]].add(c)
        return self.masks

    def get_matching_clonotypes(self, clonotype_of_interest: str, return_clusters=False):
        found_matches = set()
        if self.masks is None:
            self.get_masked_clonotypes_set()
        for i in range(len(clonotype_of_interest)):
            current_mask = clonotype_of_interest[:i] + 'X' + clonotype_of_interest[i + 1:]
            if current_mask in self.masks:
                found_matches = found_matches.union(self.masks_to_clonotypes[current_mask])
        if return_clusters:
            return set(self.clonotype_clustering[self.clonotype_clustering.cdr3aa.isin(found_matches)].cluster)
        return found_matches

    @staticmethod
    def hdist(s1, s2):
        if len(s1) != len(s2):
            return float('inf')
        else:
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    @property
    def clonotype_clustering(self, threshold=1):
        if self.clusters is not None:
            return self.clusters
        seqs = np.array(list(self.clonotypes.keys())).astype("str")
        dm = squareform(pdist(seqs.reshape(-1, 1), metric=lambda x, y: ClonotypeDataset.hdist(x[0], y[0])))
        dmf = pd.DataFrame(dm, index=seqs, columns=seqs).stack().reset_index()
        dmf.columns = ['id1', 'id2', 'distance']
        dmf = dmf[dmf['distance'] <= threshold]

        # graph
        graph = ig.Graph.TupleList(dmf[['id1', 'id2']].itertuples(index=False))
        # clusters
        self.clusters = pd.DataFrame(data={'cdr3aa': graph.get_vertex_dataframe().name,
                                           'cluster': graph.components().membership})
        return self.clusters

    @property
    def pgens(self):
        if self.pgen is not None:
            return self.pgen
        self.pgen = {}
        olga = OlgaModel(model=get_resource_path('olga/default_models/human_T_beta'))
        for clonotype in self.clonotypes.keys():
            self.pgen[clonotype] = olga.compute_pgen_cdr3aa(clonotype)
        return self.pgen

    @property
    def cluster_pgens(self):
        if self.cluster_pgen is not None:
            return self.cluster_pgen
        self.cluster_pgen = {}
        for cluster_index in self.clonotype_clustering.cluster.unique():
            self.cluster_pgen[cluster_index] = sum(
                [self.pgens[x] for x in self.clusters[self.clusters.cluster == cluster_index].cdr3aa])
        return self.cluster_pgen
