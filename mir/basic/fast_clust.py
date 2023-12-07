from multiprocessing import Pool
import pandas as pd


class FastClust:

    def __init__(self, cdr_aas: pd.Series) -> None:
        """
        Class for fast clustering o
        :param cdr_aas: pd.Series with cdrs amino acids sequences to be classified
        """

        self.cdr_aas = cdr_aas
        self.clustered_sequences = None
        self.clustered_series = None
        self.unclustered_dropout = 20
        self.lengths_sample_size_dropout = 100

    @staticmethod
    def _generate_masks(seq: str) -> set[str]:
        """
        Generate masked sequences with 1 masked with X aa
        :param seq: cdr3 sequence to be masked
        :return: set of masked cdr3 sequences
        """
        masks = set()
        for i in range(len(seq)):
            clone_to_search = seq[: i] + 'X' + seq[i + 1:]
            masks.add(clone_to_search)
        return masks

    @staticmethod
    def _process_one_length(masks_series: pd.Series) -> list[set]:
        """
        Clusters cdrs of one length
        :param masks_series: pd.Series with cdr as index and the set of its masks as values
        :return: list of sets which represents one cluster
        """
        res_list = []
        print(f'processing lengths {len(masks_series.index[0])} started')
        while len(masks_series) > 20:
            cur_clust = set()
            cur_masks = set()

            cur_clust.add(masks_series.index[0])
            cur_masks = cur_masks.union(masks_series.iloc[0])

            for cdr3, masks in masks_series.items():

                if cur_masks.intersection(masks):
                    cur_masks = cur_masks.union(masks)
                    cur_clust.add(cdr3)

            masks_series.drop(list(cur_clust), inplace=True)
            res_list.append(cur_clust)
        print(f'processing lengths {len(cdr3)} completed')
        return res_list

    def cluster(self) -> list[set]:
        """
        Clusters the given set of cdrs
        :return: list of sets which represents one cluster for whole cdrs set
        """
        len_dist = self.cdr_aas.apply(len).value_counts()

        lengths_dfs = []
        for i in len_dist[len_dist > self.lengths_sample_size_dropout].index:
            top_cdr_i = self.cdr_aas[self.cdr_aas.apply(lambda x: len(x) == i)]

            top_cdr_i_masks = top_cdr_i.apply(lambda x: self._generate_masks(x))
            top_cdr_i_masks.index = top_cdr_i
            lengths_dfs.append(top_cdr_i_masks)

        print('Start processing')
        with Pool(len(lengths_dfs)) as p:
            res_list = list(p.map(self._process_one_length, lengths_dfs))

        self.clustered_sequences = sum(res_list, [])
        return sum(res_list, [])

    def make_clusters_series(self) -> pd.Series:
        """
        Makes a pd.Series with cdr as index and cluster ID as value from self.cluster results
        :return: pd.Series with cdr as index and cluster ID as value
        """
        if self.clustered_sequences:
            exploded_series = pd.Series(self.clustered_sequences).explode()

            res_exploded = pd.DataFrame(exploded_series).reset_index()
            res_exploded.rename(columns={0: 'sequence', 'index': 'cluster_id'}, inplace=True)
            res_exploded.set_index('sequence', inplace=True)
            self.clustered_series = res_exploded
            return res_exploded

        else:
            self.cluster()
            return self.make_clusters_series()
