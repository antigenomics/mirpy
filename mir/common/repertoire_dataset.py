from multiprocessing import Pool, Manager
import typing as t

from tqdm import tqdm

from mir.basic.clonotype_usage import ClonotypeUsageTable
from mir.common.repertoire import Repertoire
import pandas as pd
from mir.common.parser import ClonotypeTableParser
from tqdm.contrib.concurrent import process_map, thread_map

from mir.comparative.pair_matcher import PairMatcher

import pickle
import os
import uuid


class RepertoireDataset:
    def __init__(self,
                 repertoires: t.Iterable[Repertoire],
                 metadata: pd.DataFrame = None,
                 gene=None,
                 threads=4,
                 public_clonotypes=None,
                 mismatch_max=1,
                 with_counts=False,
                 pair_matcher=PairMatcher(),
                 storage_dir="repertoires_storage") -> None:
        # TODO: lazy read files for large cross-sample comparisons
        # not to alter metadata
        self.repertoires = [r for r in repertoires]
        # will overwrite metadata if specified
        if not (metadata is None or metadata.empty):
            if len(metadata.index) != len(repertoires):
                raise ValueError(
                    "Metadata length doesn't match number of repertoires")
            for idx, row in metadata.iterrows():
                self.repertoires[idx].metadata = row
        else:
            metadata = pd.DataFrame([r.metadata for r in repertoires])
        self.metadata = metadata
        self.__segment_usage_matrix = None
        self.joint_number_of_clones = sum([x.number_of_clones for x in self.repertoires])
        self.__clonotype_matrix = None
        self.gene = gene
        self.threads=threads
        self.repertoire_matrix_public_clonotypes = public_clonotypes
        self.clonotype_pair_matcher = pair_matcher
        self.mismatch_max = mismatch_max
        self.with_counts=with_counts

        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)  # Ensure storage directory exists
        self.repertoire_filenames = [None] * len(self.repertoires)  # No files at start

    # def __del__(self):
    #     for file in self.repertoire_filenames:
    #         if file is not None:
    #             file_path = os.path.join(self.storage_dir, file)
    #             if os.path.exists(file_path):
    #                 os.remove(file_path)
    #     if not os.listdir(self.storage_dir):  # Check if it's empty
    #         os.rmdir(self.storage_dir)

    def serialize_repertoires(self):
        # TODO make it parallel? or no use?
        """Pickles only the repertoires that are currently in memory and removes them."""
        if sum([x is None for x in self.repertoires]) == len(self.repertoires):
            return

        for i, repertoire in tqdm(enumerate(self.repertoires), 'repertoire dataset serialization'):
            if repertoire is not None:  # Consider in-memory objects as modified
                if self.repertoire_filenames[i] is None:
                    self.repertoire_filenames[i] = f"{uuid.uuid4().hex}.pkl"  # Assign unique filename
                file_path = os.path.join(self.storage_dir, self.repertoire_filenames[i])
                with open(file_path, 'wb') as f:
                    pickle.dump(repertoire, f)
                self.repertoires[i] = None  # Remove from memory after serialization

    def deserialize_repertoire(self, idx):
        """Loads a repertoire from file if it was serialized."""
        if self.repertoires[idx] is None and self.repertoire_filenames[idx] is not None:
            file_path = os.path.join(self.storage_dir, self.repertoire_filenames[idx])
            with open(file_path, 'rb') as f:
                self.repertoires[idx] = pickle.load(f)  # Load back into memory

    def __getitem__(self, idx):
        """Returns a repertoire, loading from disk if necessary."""
        self.deserialize_repertoire(idx)
        return self.repertoires[idx]

    @property
    def clonotype_usage_matrix(self):
        if self.__clonotype_matrix is not None:
            return self.__clonotype_matrix
        print(f'clonotype usage matrix should be calculated. it would take a while')
        self.__clonotype_matrix = ClonotypeUsageTable.load_from_repertoire_dataset(
            repertoire_dataset=self,
            threads=self.threads,
            public_clonotypes=self.repertoire_matrix_public_clonotypes,
            mismatch_max=self.mismatch_max,
            with_counts=self.with_counts,
            pair_matcher=self.clonotype_pair_matcher)
        return self.__clonotype_matrix

    @classmethod
    def load(cls,
             parser: ClonotypeTableParser,
             metadata: pd.DataFrame,
             paths: list[str] = None,
             n: int = None,
             threads: int = 1,
             gene=None,
             with_counts=False,
             mismatch_max=1,
             clonotype_pair_matcher=PairMatcher()):
        global inner_repertoire_load
        metadata = metadata.copy()
        if paths:
            metadata['path'] = paths
        elif 'path' not in metadata.columns:
            raise ValueError("'path' column missing in metadata")


        def inner_repertoire_load(row):
            row_dict = dict(row)
            return Repertoire.load(parser, metadata=row_dict, n=n, gene=gene)

        repertoire_jobs = [row for _, row in metadata.iterrows()]
        repertoires = process_map(inner_repertoire_load,
                    repertoire_jobs,
                    chunksize=1,
                    max_workers=threads,
                    desc='loading Repertoire objects')
        # with Pool(threads) as p:
        #     p.map(inner_repertoire_load, repertoire_jobs)
        # repertoires = [repertoires_dct[path] for path in metadata.path]
        return cls(repertoires, metadata,
                   gene=gene,
                   threads=threads,
                   mismatch_max=mismatch_max,
                   with_counts=with_counts,
                   pair_matcher=clonotype_pair_matcher)

    @classmethod
    def load_from_single_df(cls, dataset_df,
                            metadata_columns=['severity', 'disease_status', 'study'],
                            cdr3_column='cdr3_b_aa',
                            sample_column='sample_id',
                            v_gene_column='v_b_gene',
                            j_gene_column='j_b_column',
                            d_gene_column=None,
                            gene=None,
                            with_counts=False,
                            mismatch_max=1,
                            threads: int = 1,
                            clonotype_pair_matcher=PairMatcher()):
        sample_order = dataset_df[sample_column].drop_duplicates()
        metadata = dataset_df[[sample_column] + metadata_columns].drop_duplicates().set_index(sample_column).loc[
            sample_order, :].reset_index(drop=True)
        repertoires = [Repertoire.load_from_df(df=dataset_df[dataset_df[sample_column] == sample],
                                               cdr3_column=cdr3_column,
                                               v_gene_column=v_gene_column,
                                               j_gene_column=j_gene_column,
                                               d_gene_column=d_gene_column,
                                               gene=gene) for sample in sample_order]
        return cls(repertoires=repertoires,
                   metadata=metadata,
                   gene=gene,
                   mismatch_max=mismatch_max,
                   threads=threads,
                   with_counts=with_counts,
                   pair_matcher=clonotype_pair_matcher)

    def __len__(self):
        return len(self.repertoires)

    def __str__(self):
        return f'There are {len(self.metadata)} repertoires in the dataset\n' + str(self.metadata.head(5))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.repertoires)

    # def __getitem__(self, idx):
    #     return self.repertoires[idx]

    def serialize(self, threads: int = 1) -> list[pd.DataFrame]:
        global inner_serialize
        serialized_dict = {}

        def inner_serialize(x):
            serialized_dict[x] = self.repertoires[x].serialize()

        with Pool(threads) as p:
            p.map(inner_serialize, [x for x in range(len(self.repertoires))])
        return [serialized_dict[x] for x in range(len(self.repertoires))]

    def evaluate_segment_usage(self) -> pd.DataFrame:
        if self.__segment_usage_matrix is None:
            rep_to_usage = {}
            segment_names = set()
            for rep in self.repertoires:
                rep_segment_dict = rep.evaluate_segment_usage
                segment_names = segment_names.union(set(rep_segment_dict.keys()))
                rep_to_usage[rep] = rep_segment_dict
            self.__segment_usage_matrix = pd.DataFrame(
                {k: [rep_to_usage[r][k] for r in self.repertoires] for k in segment_names})
        return self.__segment_usage_matrix

    def get_gene_types_in_repertoire_dataset(self):
        segment_usage = self.evaluate_segment_usage()
        return list(set([x[:3] for x in segment_usage.columns]))

    def get_segment_types_in_repertoire_dataset(self):
        segment_usage = self.evaluate_segment_usage()
        return set([x[3] for x in segment_usage.columns])

    def split_by_metadata_function(self, splitting_method=lambda x: x.COVID_status == 'COVID'):
        metadata_passed = self.metadata[self.metadata.apply(splitting_method, axis=1)]
        metadata_not_passed = self.metadata[~self.metadata.apply(splitting_method, axis=1)]
        repertoires_passed = [x for i, x in enumerate(self.repertoires) if i in list(metadata_passed.index)]
        repertoires_not_passed = [x for i, x in enumerate(self.repertoires) if i in list(metadata_not_passed.index)]
        return RepertoireDataset(repertoires=repertoires_passed,
                                 metadata=metadata_passed.reset_index(drop=True),
                                 gene=self.gene,
                                 threads=self.threads,
                                 public_clonotypes=self.clonotype_usage_matrix.public_clonotypes,
                                 mismatch_max=self.mismatch_max,
                                 with_counts=self.with_counts,
                                 pair_matcher=self.clonotype_pair_matcher), \
            RepertoireDataset(repertoires=repertoires_not_passed,
                              metadata=metadata_not_passed.reset_index(drop=True),
                              gene=self.gene,
                              threads=self.threads,
                              public_clonotypes=self.clonotype_usage_matrix.public_clonotypes,
                              mismatch_max=self.mismatch_max,
                              with_counts=self.with_counts,
                              pair_matcher=self.clonotype_pair_matcher)

    def create_sub_repertoire_by_field_function(self, selection_method=lambda x: x.number_of_reads > 10000):
        selected_repertoire_indices = [i for i in range(len(self.repertoires)) if selection_method(self.repertoires[i])]
        return RepertoireDataset(repertoires=[x for i, x in enumerate(self.repertoires) if i in selected_repertoire_indices],
                                 metadata=self.metadata.loc[selected_repertoire_indices].reset_index(drop=True),
                                 threads=self.threads,
                                 with_counts=self.with_counts,
                                 mismatch_max=self.mismatch_max,
                                 pair_matcher=self.clonotype_pair_matcher)

    def merge_with_another_dataset(self, other):
        return RepertoireDataset(repertoires=self.repertoires + other.repertoires,
                                 metadata=pd.concat([self.metadata, other.metadata]),
                                 threads=self.threads,
                                 with_counts=self.with_counts,
                                 mismatch_max=self.mismatch_max,
                                 pair_matcher=self.clonotype_pair_matcher
                                 )

    def resample(self, updated_segment_usage_tables: list = None, n: int = None, threads: int = 1):
        global resampling_repertoire

        def resampling_repertoire(idx):
            from mir.basic.sampling import RepertoireSampling
            return RepertoireSampling().sample(repertoire=self.repertoires[idx],
                                                               old_usage_matrix=initial_segment_usage_tables,
                                                               new_usage_matrix=updated_segment_usage_tables,
                                                               n=n)

        from mir.basic.segment_usage import NormalizedSegmentUsageTable
        gene_type = self.get_gene_types_in_repertoire_dataset()
        if len(gene_type) > 1:
            raise Exception(f'Repertoire dataset can only contain one chain, but contains {gene_type}')
        initial_segment_usage_tables = []
        if updated_segment_usage_tables is None:
            initial_segment_usage_tables = [NormalizedSegmentUsageTable.load_from_repertoire_dataset(
                repertoire_dataset=self,
                gene=gene_type,
                segment_type=segment_type) for segment_type in self.get_segment_types_in_repertoire_dataset()]
            updated_segment_usage_tables = initial_segment_usage_tables
        else:
            for um in updated_segment_usage_tables:
                initial_segment_usage_tables.append(NormalizedSegmentUsageTable.load_from_repertoire_dataset(
                    repertoire_dataset=self,
                    gene=um.gene,
                    segment_type=um.segment_type))

        metadata = self.metadata.copy()
        repertoire_jobs = [i for i in range(len(self.repertoires))]
        repertoires = process_map(resampling_repertoire, repertoire_jobs,
                    max_workers=threads,
                    desc='repertoire resampling in progress')

        # repertoires = [repertoires_dct[idx] for idx in range(len(self.repertoires))]
        return RepertoireDataset(repertoires, metadata,
                                 gene=self.gene,
                                 threads=self.threads,
                                 with_counts=self.with_counts,
                                 mismatch_max=self.mismatch_max,
                                 pair_matcher=self.clonotype_pair_matcher)

    def subsample_functional(self):
        return RepertoireDataset(repertoires=[x.subsample_functional() for x in self.repertoires],
                                 metadata=self.metadata,
                                 gene=self.gene,
                                 with_counts=self.with_counts,
                                 threads=self.threads,
                                 mismatch_max=self.mismatch_max,
                                 pair_matcher=self.clonotype_pair_matcher)

    def subsample_nonfunctional(self):
        return RepertoireDataset(repertoires=[x.subsample_nonfunctional() for x in self.repertoires],
                                 metadata=self.metadata,
                                 gene=self.gene,
                                 with_counts=self.with_counts,
                                 threads=self.threads,
                                 mismatch_max=self.mismatch_max,
                                 pair_matcher=self.clonotype_pair_matcher)
    def subsample_by_lambda(self, function=lambda x: x.cdr3aa.isalpha()):
        return RepertoireDataset(repertoires=[x.subsample_by_lambda(function) for x in self.repertoires],
                                 metadata=self.metadata,
                                 gene=self.gene,
                                 with_counts=self.with_counts,
                                 threads=self.threads,
                                 mismatch_max=self.mismatch_max,
                                 pair_matcher=self.clonotype_pair_matcher)