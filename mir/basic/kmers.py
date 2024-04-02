from itertools import islice
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from mir.common.repertoire import Repertoire
import matplotlib.pyplot as plt


def comparison_plotter(res: pd.DataFrame, plot_type=None, ax=None):
    if plot_type == 'line':
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(np.log2(res.freq_1), np.log2(res.freq_2), s=1)
        ax.plot([np.log2(res[res.freq_1 != 0].freq_1.min()), np.log2(res.freq_1.max())],
                [np.log2(res[res.freq_1 != 0].freq_1.min()), np.log2(res.freq_1.max())], '--', c='red')
        ax.set_xlabel('log2(freq_1)')
        ax.set_ylabel('log2(freq_2)')

    if plot_type == 'volcano':
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(np.log2(res.freq_fc), -np.log2(res.p_val.apply(lambda x: 2 ** (-100) if x < 2 ** (-100) else x)),
                   s=1)
        ax.set_xlabel('logFC')
        ax.set_ylabel('-log(p)')

        top_results = res[res.freq_fc.apply(lambda x: not np.isinf(np.log2(x)))].sort_values(by='p_val').head(10)
        top_results.p_val = top_results.p_val.apply(lambda x: 2 ** (-100) if x < 2 ** (-100) else x)
        for index, row in top_results.iterrows():
            x = np.log2(row['freq_fc'])
            y = -np.log2(row['p_val'])
            ax.annotate(index, (x, y), textcoords="offset points", ha='center',
                        xytext=(0, 10), arrowprops=dict(arrowstyle='-', lw=0.5))


def generate_comparison_from_counts(count_res, p_adj_func):
    count_res['p_val'] = np.nan
    n = count_res.count_1.sum()
    m = count_res.count_2.sum()
    count_res['freq_1'] = count_res.count_1 / n
    count_res['freq_2'] = count_res.count_2 / m

    for kmer in count_res.index:
        count_res.loc[kmer, 'p_val'] = chi2_contingency([[count_res['count_1'][kmer], n - count_res['count_1'][kmer]],
                                                         [count_res['count_2'][kmer], m - count_res['count_2'][kmer]]])[
            1]

    count_res['freq_fc'] = (count_res.count_1 / n) / (count_res.count_2 / m)

    if not p_adj_func:
        count_res['p_val_adj'] = multipletests(count_res.p_val, method='holm')[1]
    else:
        count_res['p_val_adj'] = p_adj_func(count_res.p_val)

    return count_res


class KmersTable:
    """
    Class to generate series of subsequnces of length k from Repertoire
    """

    def __init__(self, k: int, repertoire: Repertoire):
        """
        Creating a new KmeresTable object

        :param k: length of subsequnces to be generated
        :param repertoire: Repertoire class object
        """
        self.k = k
        self.repertoire = repertoire
        self.vc_k_mers_list_generator = np.vectorize(self._k_mers_list_generator, otypes=[np.ndarray])
        self.kmer_table = list()
        self.count_table = dict()

    def _over_slice(self, cdr3):
        """
        Generator of kmers
        :param cdr3: cdr3 aa sequence
        """
        iterator = iter(cdr3)
        res = tuple(islice(iterator, self.k))
        if len(res) == self.k:
            yield res
        for elem in iterator:
            res = res[1:] + (elem,)
            yield res

    def _k_mers_list_generator(self, clonotype) -> list:
        """
        Form a list from generator
        :param cdr3: cdr3 aa sequence
        :return: list of kmers for cdr3
        """
        cdr3 = clonotype.cdr3aa
        res = ["".join(elem) for elem in self._over_slice(cdr3)]
        return res

    def generate_kmers_table(self) -> np.ndarray:
        """
        Generate the table with kmers for repertoire
        :return: Series of kmers arrays for each cdr3 in repertoire
        """
        if len(self.kmer_table):
            return self.kmer_table
        else:
            self.kmer_table = self.vc_k_mers_list_generator(self.repertoire.clonotypes)
            return self.kmer_table

    def generate_kmers_count_table(self) -> dict[str:int]:
        """
        Generate the dict with numbers of occurrence of each kmer in repertoire
        :return: dict with amount of each kmer
        """
        if len(self.count_table):
            return self.count_table
        else:
            kmers_array = self.generate_kmers_table()
            unique_kmers, kmers_counts = np.unique((np.concatenate(kmers_array)), return_counts=True)
            self.count_table = dict(zip(unique_kmers, kmers_counts))
            return self.count_table

    def compare_with_another_KmersTable(self,
                                        kmers_table: 'KmersTable',
                                        plot_comparison=None,
                                        ax=None,
                                        p_adj_func=None
                                        ):
        """
        Function to compare self with another KmersTable
        :param kmers_table: another KmersTable object to be compared with self
        :param plot_comparison:
        None - do not plot comparison
        'line' - plot log2(frequences) of kmers from two sets
        'volcano' - plot Volcanoplot of kmers comparison
        :param ax: ax to plot comparison
        :param p_adj_func: function to adjust p_values array, which returns array of adjusted ps
        :return: pd.DataFrame with comparison results
        """

        if self.k != kmers_table.k:
            raise ValueError('K should be equal for comparison')

        table_1 = pd.DataFrame.from_dict(self.generate_kmers_count_table(), orient='index')
        table_2 = pd.DataFrame.from_dict(kmers_table.generate_kmers_count_table(), orient='index')

        table_1.columns = ['count_1']
        table_2.columns = ['count_2']

        count_res = table_1.join(table_2, how='outer').fillna(0)
        res = generate_comparison_from_counts(count_res, p_adj_func)

        if plot_comparison:
            comparison_plotter(res=res, plot_type=plot_comparison, ax=ax)
        return res


def compare_two_repertoire_kmers(repertoire_1: Repertoire,
                                 repertoire_2: Repertoire,
                                 k: int,
                                 plot_comparison=None,
                                 ax=None,
                                 p_adj_func=None
                                 ) -> pd.DataFrame:
    """
    Function to compare 2 Repertoires on the kmers level
    :param repertoire_1: first repertoire
    :param repertoire_2: first repertoire
    :param k: length of subsequnces to be compared
    :param plot_comparison:
    None - do not plot comparison
    'line' - plot log2(frequences) of kmers from two sets
    'volcano' - plot Volcanoplot of kmers comparison
    :param ax: ax to plot comparison
    :param p_adj_func: function to adjust p_values array, which returns array of adjusted ps
    :return: pd.DataFrame with comparison results
    """

    table_1 = pd.DataFrame.from_dict(KmersTable(k, repertoire_1).generate_kmers_count_table(), orient='index')
    table_2 = pd.DataFrame.from_dict(KmersTable(k, repertoire_2).generate_kmers_count_table(), orient='index')

    table_1.columns = ['count_1']
    table_2.columns = ['count_2']

    res = table_1.join(table_2, how='outer').fillna(0)
    generate_comparison_from_counts(res, p_adj_func)

    if plot_comparison:
        comparison_plotter(res=res, plot_type=plot_comparison, ax=ax)
    return res
