from collections import defaultdict

from mir import get_resource_path
from mir.basic.pgen import OlgaModel
from mir.common.clonotype import ClonotypeAA
import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.__clusters_df = None
        self.__pgen_dict = None
        self.__cluster_pgen_dict = None
        self.__graph_object = None
        self.__graph_coords = None
        self.cluster_payload = None


    def serialize(self, file_name='biomarkers.csv'):
        marker_df = pd.DataFrame({'cdr3': pd.Series(list(self.clonotypes.keys()))})

        if self.__clusters_df is not None:
            marker_df = marker_df.merge(self.__clusters_df)

        if self.__pgen_dict is not None:
            marker_df['pgen'] = pd.Series(self.__pgen_dict[cdr3] for cdr3 in marker_df.cdr3)

        if self.__graph_coords is not None:
            marker_df = marker_df.merge(self.__graph_coords)
        marker_df.to_csv(file_name)


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

    def get_matching_clonotypes(self, clonotype_of_interest: str, return_clusters=False, threshold=1):
        found_matches = set()
        if self.masks is None:
            self.get_masked_clonotypes_set()
        if threshold == 1:
            for i in range(len(clonotype_of_interest)):
                current_mask = clonotype_of_interest[:i] + 'X' + clonotype_of_interest[i + 1:]
                if current_mask in self.masks:
                    found_matches = found_matches.union(self.masks_to_clonotypes[current_mask])
        elif threshold == 0:
            found_matches = [clonotype_of_interest] if clonotype_of_interest in self.clonotypes else []
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
    def graph(self, threshold=1):
        if self.__graph_object is not None:
            return self.__graph_object
        edges = []
        for c1 in self.clonotypes.keys():
            for c2  in self.clonotypes.keys():
                if len(c1) == len(c2):
                    dist = sum(c1 != c2 for c1, c2 in zip(c1, c2))
                    if dist <= threshold:
                        edges.append((c1, c2))

        self.__graph_object = ig.Graph.TupleList(edges)
        return self.__graph_object

    @property
    def clonotype_clustering(self):
        if self.__clusters_df is not None:
            return self.__clusters_df
        graph = self.graph
        self.__clusters_df = pd.DataFrame(data={'cdr3aa': graph.get_vertex_dataframe().name,
                                           'cluster': graph.components().membership})
        return self.__clusters_df

    @property
    def pgens(self):
        if self.__pgen_dict is not None:
            return self.__pgen_dict
        self.__pgen_dict = {}
        olga = OlgaModel(model=get_resource_path('olga/default_models/human_T_beta'))
        for clonotype in self.clonotypes.keys():
            self.__pgen_dict[clonotype] = olga.compute_pgen_cdr3aa(clonotype)
        return self.__pgen_dict

    @property
    def cluster_pgens(self):
        if self.__cluster_pgen_dict is not None:
            return self.__cluster_pgen_dict
        self.__cluster_pgen_dict = {}
        for cluster_index in self.clonotype_clustering.cluster.unique():
            self.__cluster_pgen_dict[cluster_index] = sum(
                [self.pgens[x] for x in self.__clusters_df[self.__clusters_df.cluster == cluster_index].cdr3aa])
        return self.__cluster_pgen_dict

    @property
    def clonotype_coords(self):
        if len(self.clonotypes) < 1000:
            viz_method = 'graphopt'
        else:
            viz_method = 'drl'
        if self.__graph_coords is not None:
            return self.__graph_coords
        layout = self.graph.layout(viz_method)
        coords = np.array(layout.coords)

        self.__graph_coords = pd.DataFrame(
            {'cdr3': self.graph.vs()['name'],
             'cluster': self.graph.components().membership,
             'x': coords[:, 0],
             'y': coords[:, 1]
             })
        self.__graph_coords['cluster_size'] = self.__graph_coords.cluster.apply(
            lambda x: self.__graph_coords.cluster.value_counts()[x])
        return self.__graph_coords

    def update_cluster_payload(self, cluster_to_feature: dict, feature_name: str):
        if self.cluster_payload is None:
            self.cluster_payload = pd.DataFrame(cluster_to_feature, index=[0]).T.reset_index().rename(
                columns={'index': 'cluster', 0: feature_name})
        else:
            self.cluster_payload[feature_name] = self.cluster_payload.cluster.apply(lambda x: cluster_to_feature[x])

    def plot_clonotype_clustering(self, color_by: str, ax=None, plot_unclustered=True):
        if self.cluster_payload is not None:
            plotting_df = self.clonotype_coords.merge(self.cluster_payload)
        else:
            plotting_df = self.clonotype_coords

        if ax is None:
            fig, (ax) = plt.subplots(1, 1)

        if len(plotting_df[color_by].unique()) > 10:
            palette = sns.color_palette("tab20b", 100)
        else:
            palette = sns.color_palette("tab10")
        sns.scatterplot(plotting_df[plotting_df.cluster_size > 1], x='x', y='y', hue=color_by,
                        palette=palette, ax=ax)
        if plot_unclustered:
            sns.scatterplot(plotting_df[plotting_df.cluster_size == 1], x='x', y='y', hue=color_by,
                            palette=['grey'],
                            legend=False, ax=ax)

    def __repr__(self):
        return f'A dataset of {len(self.clonotypes)} clonotypes ' + \
                    f'and {len(self.clonotype_clustering.cluster.unique())} clusters'
