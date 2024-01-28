import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import logistic
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mir.common.repertoire_dataset import RepertoireDataset

epsilon = 0.000001


class SegmentUsageTable:
    def __init__(self, segment_usage_matrix: pd.DataFrame, metadata: pd.DataFrame, gene: str = 'TRA', segment_type: str = 'V'):
        self.gene = gene
        self.segment_type = segment_type
        self.gene_mask = self.gene + self.segment_type
        self.segment_usage_matrix = segment_usage_matrix
        self.metadata = metadata

    @staticmethod
    def preprocess_usage_table(matrix):
        return matrix

    @classmethod
    def load_from_repertoire_dataset(cls, repertoire_dataset: RepertoireDataset, gene: str, segment_type: str,
                                     ):
        matrix = repertoire_dataset.evaluate_segment_usage()
        gene_mask = gene + segment_type
        matrix = matrix[[x for x in matrix.columns if gene_mask in x]]

        return cls(cls.preprocess_usage_table(matrix), repertoire_dataset.metadata,
                   gene=gene, segment_type=segment_type)

    def plot_pca_results_colored_by(self, target: pd.Series, method=PCA, n_components: int = 2,
                                    plot_gradient: bool = False, ax=None):
        def plot_results(pca_results, target, ax):
            pca_results['target'] = target
            if ax is None:
                fig, ax = plt.subplots()
            if plot_gradient:
                if len(target.unique()) == 1:
                    print('Bad gene:(')
                    return
                if n_components > 2:
                    sns.pairplot(pca_results, plot_kws=dict(
                        hue=target,
                        palette=mpl.cm.viridis
                    ))
                else:
                    sc = ax.scatter(pca_results['PC1'], pca_results['PC2'], c=target, vmin=min(target),
                                    vmax=max(target))
                    plt.colorbar(sc, ax=ax)
            else:
                if n_components > 21:
                    sns.pairplot(pca_results, hue='target')
                else:
                    sns.scatterplot(x='PC1', y='PC2', data=pca_results, hue='target', ax=ax)
                    sns.move_legend(ax, "upper right", bbox_to_anchor=(2, 1.1))
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')


        pca = method(n_components=n_components, random_state=42)
        usage_matrix_genes = self.segment_usage_matrix.copy()
        pca_results = pca.fit_transform(usage_matrix_genes)
        pca_results = pd.DataFrame(data=pca_results[:, :], columns=[f'PC{i + 1}' for i in range(n_components)])

        plot_results(pca_results, target, ax)

    def plot_clustermap_axes_based(self, genes=['TRBV28', 'TRBV4-3', 'TRBV6-2'], ax=None):
        Z = linkage(self.segment_usage_matrix[genes], 'complete')
        labels = fcluster(Z, 0.00001, criterion='distance')
        labels_order = np.argsort(labels)
        needed_data = self.segment_usage_matrix[genes]
        if ax is None:
            fig, ax = plt.subplots()
        for gene in genes:
            needed_data[gene] = (self.segment_usage_matrix[gene] - self.segment_usage_matrix[gene].min()) / (
                        self.segment_usage_matrix[gene].max() - self.segment_usage_matrix[gene].min())
        sns.heatmap(needed_data.loc[labels_order, :][genes].T, ax=ax, cbar_kws={'label': 'V gene usage in a sample'},
                    cmap='vlag')
        for k, gene in enumerate(genes):
            zscores = zscore(self.segment_usage_matrix.loc[labels_order, :][gene])
            sns.lineplot(x=[i for i in range(len(self.segment_usage_matrix[gene]))],
                         y=-(zscores - zscores.min()) / (zscores.max() - zscores.min()) + 1 + k, ax=ax, color='black',
                         linewidth=0.5)
        ax.get_xaxis().set_visible(False)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.segment_usage_matrix.loc[i, :].to_dict()
        if i not in list(self.metadata.run):
            raise Exception(f'{i} not found in metadata!')
        return self.segment_usage_matrix.loc[self.metadata[self.metadata.run == i].index[0], :].to_dict()

    def __repr__(self):
        return repr(self.segment_usage_matrix)


class NormalizedSegmentUsageTable(SegmentUsageTable):
    def __init__(self, segment_usage_matrix: pd.DataFrame, metadata: pd.DataFrame, gene: str = 'TRA', segment_type: str = 'V'):
        super().__init__(segment_usage_matrix, metadata, gene, segment_type)

    @staticmethod
    def preprocess_usage_table(matrix):
        return NormalizedSegmentUsageTable.__normalize_usage_matrix_by_rows(matrix)

    @staticmethod
    def __normalize_usage_matrix_by_rows(matrix: pd.DataFrame):
        """
        A function that normalizes gene usages within one run.
        This is made to translate all the runs' data into vector normalized by 1.
        The given matrix can have any annotation columns, they will be thrown away,
        but the genes should be given in the following format: IGHV.-(.*)?
        :param matrix: the gene usage matrix, which is not preprocessed yet;
        :return: the normalized usage matrix without annotation columns
        """
        usage_matrix = matrix.copy()
        usage_matrix["sum"] = usage_matrix.sum(axis=1)
        for column in usage_matrix.columns:
            usage_matrix[column] = usage_matrix[column] / usage_matrix['sum']
        usage_matrix = usage_matrix.drop(columns=['sum'])
        return usage_matrix


class StandardizedSegmentUsageTable(NormalizedSegmentUsageTable):
    def __init__(self, segment_usage_matrix: pd.DataFrame, metadata: pd.DataFrame, gene: str = 'TRA', segment_type: str = 'V',
                 standardization_method: str = 'z_score'):
        super().__init__(segment_usage_matrix, metadata, gene, segment_type)
        if standardization_method not in ['z_score', 'log_exp']:
            raise NotImplementedError('No such standardization method!')
        self.standardization_method = standardization_method

    @staticmethod
    def preprocess_usage_table(group_mapping, standardization_method):
        if standardization_method == 'z_score':
            return StandardizedSegmentUsageTable.__standardize_usage_matrix(group_mapping)
        elif standardization_method == 'log_exp':
            return StandardizedSegmentUsageTable.__standardize_usage_matrix_log_exp(group_mapping)

    @classmethod
    def load_from_repertoire_dataset(cls, repertoire_dataset: RepertoireDataset, gene: str, segment_type: str,
                                     group_mapping: dict, standardization_method: str):
        matrix = repertoire_dataset.evaluate_segment_usage()
        gene_mask = gene + segment_type
        matrix = matrix[[x for x in matrix.columns if gene_mask in x]]
        group_to_df_mapping = {}
        if group_mapping is not None:
            for k, v in group_mapping.items():
                group_to_df_mapping[k] = matrix.loc[repertoire_dataset.metadata[
                                                        repertoire_dataset.metadata.run.apply(lambda x: x in v)].index,
                                         :]
        return cls(cls.preprocess_usage_table(group_to_df_mapping, standardization_method), repertoire_dataset.metadata,
                   gene=gene, segment_type=segment_type)


    @staticmethod
    def __normalize_usage_matrix_by_rows(matrix: pd.DataFrame):
        usage_matrix = matrix.copy()
        usage_matrix["sum"] = usage_matrix.sum(axis=1)
        for column in usage_matrix.columns:
            usage_matrix[column] = usage_matrix[column] / usage_matrix['sum']
        usage_matrix = usage_matrix.drop(columns=['sum'])
        return usage_matrix

    @staticmethod
    def __standardize_usage_matrix(group_mapping: dict[str, pd.DataFrame]):
        matrices = []
        for group, usage_matrix_group in group_mapping.items():
            v_gene_names = usage_matrix_group.columns
            usage_matrix_group = pd.DataFrame(data=StandardScaler().fit_transform(usage_matrix_group),
                                              columns=v_gene_names)
            matrices.append(usage_matrix_group)
        return pd.concat(matrices).reset_index(drop=True)

    @staticmethod
    def __standardize_usage_matrix_log(group_mapping: dict[str, pd.DataFrame]):
        matrices = []
        for group, usage_matrix_group in group_mapping.items():
            v_genes = usage_matrix_group.apply(lambda x: np.log(x + epsilon))
            v_gene_names = v_genes.columns
            v_genes = pd.DataFrame(data=StandardScaler().fit_transform(v_genes), columns=v_gene_names)
            matrices.append(v_genes)
        cur_usage_matrix = pd.concat(matrices).reset_index(drop=True)
        return cur_usage_matrix

    @staticmethod
    def __standardize_usage_matrix_log_exp(group_mapping: dict[str, pd.DataFrame]):
        norm_usage_matrix = pd.concat(list(group_mapping.values()))
        for k, v in group_mapping.items():
            group_mapping[k] = StandardizedSegmentUsageTable.__normalize_usage_matrix_by_rows(v)
        log_stand_usage_matrix = StandardizedSegmentUsageTable.__standardize_usage_matrix_log(group_mapping)
        for name in log_stand_usage_matrix.columns:
            b = norm_usage_matrix[name].mean()
            log_stand_usage_matrix[name] = log_stand_usage_matrix[name].apply(lambda x: 2 * b * logistic.cdf(x))
        return StandardizedSegmentUsageTable.__normalize_usage_matrix_by_rows(log_stand_usage_matrix)
