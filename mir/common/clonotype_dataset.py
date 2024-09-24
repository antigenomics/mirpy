from collections import defaultdict

from mir import get_resource_path
from mir.basic.pgen import OlgaModel
from mir.common.clonotype import ClonotypeAA, ClonotypeNT
import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multipy.fdr import lsu
from tqdm import tqdm

from mir.comparative.pair_matcher import ClonotypeRepresentation
from scipy.stats import fisher_exact


class ClonotypeDataset:
    """
    The dataset of clonotypes is stored in this object. To be used to store a set of clonotypes with specific features
    """

    # TODO add loading clonotypeDataset from file
    def __init__(self, clonotypes: list[ClonotypeAA]):
        for clonotype in clonotypes:
            if not isinstance(clonotype, ClonotypeAA):
                raise Exception('You must have cdr3aa sequence for ClonotypeDataset creation')
        self.clonotypes_cdr3aa = {x.cdr3aa: x for x in clonotypes}
        self.additional_info = pd.DataFrame([x.serialize() for x in clonotypes])
        self.__masks = None
        self.masks_to_clonotypes = defaultdict(set)
        self.__clusters_df = None
        self.__pgen_df = None
        self.__cluster_pgen_dict = None
        self.__graph_object = None
        self.__graph_coords = None
        self.cluster_payload = None

    @classmethod
    def from_representations(cls, clonotype_reprs: list[ClonotypeRepresentation]):
        """
        a function that creates ClonotypeDataset from a list of `ClonotypeRepresentation`
        :param clonotype_reprs: list of representations
        :return: `ClonotypeDataset` obj
        """
        # TODO add payload?
        clonotypes = []
        for representation in clonotype_reprs:
            if representation.cdr3nt is not None:
                clonotypes.append(ClonotypeNT(cdr3nt=representation.cdr3nt,
                                              cdr3aa=representation.cdr3aa,
                                              v=representation.v,
                                              j=representation.j))
            else:
                clonotypes.append(ClonotypeAA(cdr3aa=representation.cdr3aa,
                                              v=representation.v,
                                              j=representation.j))
        return cls(clonotypes)

    def serialize(self, file_name='biomarkers.csv'):
        """
        serializes all the available information about the clonotype dataset into a file
        :param file_name: the name of file to perform the serialization into
        :return: --
        """
        marker_df = self.additional_info

        if self.__pgen_df is not None:
            marker_df = marker_df.merge(self.__pgen_df).drop_duplicates()

        if self.__graph_coords is not None:
            marker_df = marker_df.merge(self.__graph_coords).drop_duplicates()
        elif self.__clusters_df is not None:
            marker_df = marker_df.merge(self.__clusters_df).drop_duplicates()

        if file_name is None:
            return marker_df

        marker_df.to_csv(file_name)

    def get_number_of_samples_for_clonotype(self, clonotype_seqaa):
        # todo, sep upd Liza: should be moved to clonotype usage matrix, no use here
        pass

    @property
    def masks(self):
        # TODO masking is written in three places along the library: here, in fast_clust, in clonotype usage matrix; to fix
        """
        creates a set of masked CDR3 sequences; needed for fast matching
        :return:
        """
        if self.__masks is not None:
            return self.__masks
        self.__masks = set()
        for c in self.clonotypes_cdr3aa.keys():
            for i in range(len(c)):
                self.__masks.add(c[:i] + 'X' + c[i + 1:])
                self.masks_to_clonotypes[c[:i] + 'X' + c[i + 1:]].add(c)
        return self.__masks

    def get_matching_clonotypes(self, clonotype_of_interest: str, return_clusters=False, threshold=1):
        """
        get the clonotypes which match hte given pattern
        :param clonotype_of_interest: CDR3aa sequence
        :param return_clusters: whether to return the exact sequences which match or to return the clusters where those matches come from
        :param threshold: 0 or 1 for exact or 1mm search #TODO also to be fixed here
        :return: the matched seqs from clonotype_dataset or cluster ids
        """
        found_matches = set()
        if threshold == 1:
            for i in range(len(clonotype_of_interest)):
                current_mask = clonotype_of_interest[:i] + 'X' + clonotype_of_interest[i + 1:]
                if current_mask in self.masks:
                    found_matches = found_matches.union(self.masks_to_clonotypes[current_mask])
        elif threshold == 0:
            found_matches = [clonotype_of_interest] if clonotype_of_interest in self.clonotypes_cdr3aa else []
        if return_clusters:
            return set(self.clonotype_clustering[self.clonotype_clustering.cdr3aa.isin(found_matches)].cluster)
        return found_matches

    def get_matching_clonotypes_for_set(self, clonotypes_of_interest, return_clusters=False, threshold=1):
        """
        :param clonotypes_of_interest:
        :param return_clusters: whether to return the exact sequences which match or to return the clusters where those matches come from
        :param threshold: 0 or 1 for exact or 1mm search #TODO also to be fixed here
        :return: all the clonotype/cluster matches within 0/1 mm
        """
        set_of_matches = set()
        for clonotype in clonotypes_of_interest:
            set_of_matches = set_of_matches.union(
                self.get_matching_clonotypes(clonotype, return_clusters=return_clusters,
                                             threshold=threshold))
        return set_of_matches

    @property
    def graph(self, threshold=1):
        """
        creates the igraph object with the vertices stored
        :param threshold: threshold for connecting two clonotypes with an edge
        :return:
        """
        if self.__graph_object is not None:
            return self.__graph_object
        edges = []
        for c1 in self.clonotypes_cdr3aa.keys():
            for c2 in self.clonotypes_cdr3aa.keys():
                if len(c1) == len(c2):
                    dist = sum(c1 != c2 for c1, c2 in zip(c1, c2))
                    if dist <= threshold:
                        edges.append((c1, c2))

        self.__graph_object = ig.Graph.TupleList(edges)
        return self.__graph_object

    @property
    def clonotype_clustering(self):
        """
        creates the mapping of a clonotype to a cluster it belongs to
        :return:
        """
        if self.__clusters_df is not None:
            return self.__clusters_df
        graph = self.graph
        self.__clusters_df = pd.DataFrame(data={'cdr3aa': graph.get_vertex_dataframe().name,
                                                'cluster': graph.components().membership})
        return self.__clusters_df

    @property
    def pgens(self):
        """
        creates a dataframe of clonotype pgens
        :return:
        """
        if self.__pgen_df is not None:
            return self.__pgen_df
        self.__pgen_df = {}
        olga = OlgaModel(model=get_resource_path('olga/default_models/human_T_beta'))
        for clonotype in self.clonotypes_cdr3aa.keys():
            self.__pgen_df[clonotype] = olga.compute_pgen_cdr3aa(clonotype)
        self.__pgen_df = pd.DataFrame(data={'cdr3aa': self.__pgen_df.keys(),
                                            'pgen': self.__pgen_df.values()})
        return self.__pgen_df

    @property
    def cluster_pgens(self):
        """
        created a data frame with pgens for cluster (calculated as joint probability of a cluster)
        :return:
        """
        if self.__cluster_pgen_dict is not None:
            return self.__cluster_pgen_dict
        self.__cluster_pgen_dict = {}
        for cluster_index in self.clonotype_clustering.cluster.unique():
            self.__cluster_pgen_dict[cluster_index] = sum(
                [self.pgens[x] for x in self.__clusters_df[self.__clusters_df.cluster == cluster_index].cdr3aa])
        return self.__cluster_pgen_dict

    @property
    def clonotype_coords(self):
        """
        calculates clonotype cooredinates for plotting
        :return:
        """
        if len(self.clonotypes_cdr3aa) < 1000:
            viz_method = 'graphopt'
        else:
            viz_method = 'drl'
        if self.__graph_coords is not None:
            return self.__graph_coords
        layout = self.graph.layout(viz_method)
        coords = np.array(layout.coords)

        self.__graph_coords = pd.DataFrame(
            {'cdr3aa': self.graph.vs()['name'],
             'cluster': self.graph.components().membership,
             'x': coords[:, 0],
             'y': coords[:, 1]
             })
        self.__graph_coords['cluster_size'] = self.__graph_coords.cluster.apply(
            lambda x: self.__graph_coords.cluster.value_counts()[x])
        return self.__graph_coords

    def update_cluster_payload(self, cluster_to_feature: dict, feature_name: str):
        """
        adds new payload column to the data
        :param cluster_to_feature: mapping of the cluster to the feature value
        :param feature_name: name of the feature for the payload
        """
        if self.cluster_payload is None:
            self.cluster_payload = pd.DataFrame(cluster_to_feature, index=[0]).T.reset_index().rename(
                columns={'index': 'cluster', 0: feature_name})
        else:
            self.cluster_payload[feature_name] = self.cluster_payload.cluster.apply(lambda x: cluster_to_feature[x])

    def plot_clonotype_clustering(self, color_by: str, ax=None, plot_unclustered=True):
        """
        visualizes the data
        :param color_by: which column from the payload you should color by
        :param ax: axes can be passed
        :param plot_unclustered: whether to plot singletons (points with cluster size equal to 1)
        """
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

    def test_cluster_usage_significance_for_comparison(self, cluster_idx, database_records_for_comparison):
        """
        tests whether the usage of items from `database_records_for_comparison` is enriched in the given cluster `cluster_idx`
        :param cluster_idx: the cluster index which we check the enrichment for
        :param database_records_for_comparison: the database sequences which we analyze the enriched usage of
        :return: the Fisher exact test p value of enrichment
        """
        matched_clonotypes = pd.Series(list(self.get_matching_clonotypes_for_set(database_records_for_comparison)))
        matched_clonotypes_within_cluster = pd.DataFrame({'cdr3aa': matched_clonotypes}).merge(
            self.clonotype_clustering[self.clonotype_clustering.cluster == cluster_idx])

        num_positive_within_cluster = len(matched_clonotypes_within_cluster)
        num_positive_outside_cluster = len(matched_clonotypes) - num_positive_within_cluster
        num_negative_within_cluster = len(
            self.clonotype_clustering[self.clonotype_clustering.cluster == cluster_idx]) - num_positive_within_cluster
        num_negative_outside_cluster = len(
            self.clonotype_clustering[self.clonotype_clustering.cluster != cluster_idx]) - num_positive_outside_cluster

        return fisher_exact([[num_negative_within_cluster, num_negative_outside_cluster],
                             [num_positive_within_cluster, num_positive_outside_cluster]], alternative='less')[1]

    def annotate_with_database(self, database, comparison_by='antigen.epitope', cdr3aa_col='cdr3',
                               return_significant=True, alpha=0.05):
        """
        tests the significant enrichment of each type of data from the given database for each cluster;
        LSU is applied within each comparison type (e.g. for each cluster)
        :param database: the dataframe with the database of clonotypes and its features
        :param comparison_by: which column of the database to perform enrichment test by
        :param cdr3aa_col: cdr3aa column name in database
        :param return_significant: whether LSU correction should be performed or not
        :param alpha: threshold for LSU correction
        :return: dataframe/dict with p values
        """
        # todo add threads
        pvals = {}
        unique_comparisons = database[comparison_by].unique()
        for comp in tqdm(unique_comparisons):
            pvals[comp] = {}
            for cluster in range(self.clonotype_clustering.cluster.max() + 1):
                pvals[comp][cluster] = self.test_cluster_usage_significance_for_comparison(
                    cluster, database[database[comparison_by] == comp][cdr3aa_col])
        if not return_significant:
            return pvals

        sign_cluster = []
        sign_epi = []
        sign_pval = []
        for comp in unique_comparisons:
            epitope_results = list(pvals[comp].values())
            is_significant = lsu(np.array(epitope_results), q=alpha)
            for i, res in enumerate(is_significant):
                if res:
                    sign_cluster.append(i)
                    sign_epi.append(comp)
                    sign_pval.append(pvals[comp][i])
        return pd.DataFrame({'cluster': sign_cluster, comparison_by: sign_epi, 'pval': sign_pval})

    def __repr__(self):
        return f'A dataset of {len(self.clonotypes_cdr3aa)} clonotypes ' + \
            f'and {len(self.clonotype_clustering.cluster.unique())} clusters'
