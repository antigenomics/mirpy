from multiprocessing import Pool, Manager
import typing as t

from mir.basic.clonotype_usage import ClonotypeUsageTable
from mir.common.repertoire import Repertoire
import pandas as pd
from mir.common.parser import ClonotypeTableParser


class RepertoireDataset:
    def __init__(self,
                 repertoires: t.Iterable[Repertoire],
                 metadata: pd.DataFrame = None,
                 gene=None) -> None:
        # TODO: lazy read files for large cross-sample comparisons
        # not to alter metadata
        self.repertoires = [r.__copy__() for r in repertoires]
        # will overwrite metadata if specified
        if not metadata.empty:
            if len(metadata.index) != len(repertoires):
                raise ValueError(
                    "Metadata length doesn't match number of repertoires")
            for idx, row in metadata.iterrows():
                self.repertoires[idx].metadata = row
        else:
            metadata = pd.DataFrame([r.metadata for r in repertoires])
        self.metadata = metadata
        self.segment_usage_matrix = None
        self.joint_number_of_clones = sum([x.number_of_clones for x in self.repertoires])
        self.clonotype_usage_matrix = ClonotypeUsageTable.load_from_repertoire_dataset(repertoire_dataset=self)
        self.gene = gene

    @classmethod
    def load(cls,
             parser: ClonotypeTableParser,
             metadata: pd.DataFrame,
             paths: list[str] = None,
             n: int = None,
             threads: int = 1,
             gene=None):
        global inner_repertoire_load
        metadata = metadata.copy()
        if paths:
            metadata['path'] = paths
        elif 'path' not in metadata.columns:
            raise ValueError("'path' column missing in metadata")

        repertoires_dct = Manager().dict()

        def inner_repertoire_load(row):
            row_dict = dict(row)
            path = row_dict['path']
            repertoires_dct[path] = Repertoire.load(parser, metadata=row_dict, n=n, gene=gene)

        repertoire_jobs = [row for _, row in metadata.iterrows()]
        with Pool(threads) as p:
            p.map(inner_repertoire_load, repertoire_jobs)

        repertoires = [repertoires_dct[path] for path in metadata.path]
        return cls(repertoires, metadata, gene=gene)

    def __len__(self):
        return len(self.repertoires)

    def __str__(self):
        return f'There are {len(self.metadata)} repertoires in the dataset\n' + str(self.metadata.head(5))

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.repertoires)

    def __getitem__(self, idx):
        return self.repertoires[idx]

    def serialize(self, threads: int = 1) -> list[pd.DataFrame]:
        global inner_serialize
        serialized_dict = {}

        def inner_serialize(x):
            serialized_dict[x] = self.repertoires[x].serialize()

        with Pool(threads) as p:
            p.map(inner_serialize, [x for x in range(len(self.repertoires))])
        return [serialized_dict[x] for x in range(len(self.repertoires))]

    def evaluate_segment_usage(self) -> pd.DataFrame:
        if self.segment_usage_matrix is None:
            rep_to_usage = {}
            segment_names = set()
            for rep in self.repertoires:
                rep_segment_dict = rep.evaluate_segment_usage()
                segment_names = segment_names.union(set(rep_segment_dict.keys()))
                rep_to_usage[rep] = rep_segment_dict
            self.segment_usage_matrix = pd.DataFrame(
                {k: [rep_to_usage[r][k] for r in self.repertoires] for k in segment_names})
        return self.segment_usage_matrix

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
        return RepertoireDataset(repertoires_passed, metadata_passed), \
            RepertoireDataset(repertoires_not_passed, metadata_not_passed)

    def create_sub_repertoire_by_field_function(self, selection_method=lambda x: x.number_of_reads > 10000):
        selected_repertoire_indices = [i for i in range(len(self.repertoires)) if selection_method(self.repertoires[i])]
        return RepertoireDataset([x for i, x in enumerate(self.repertoires) if i in selected_repertoire_indices],
                                 self.metadata.loc[selected_repertoire_indices])

    def resample(self, updated_segment_usage_tables: list = None, n: int = None, threads: int = 1):
        global resampling_repertoire
        repertoires_dct = Manager().dict()

        def resampling_repertoire(idx):
            from mir.basic.sampling import RepertoireSampling
            repertoires_dct[idx] = RepertoireSampling().sample(repertoire=self.repertoires[idx],
                                                               old_usage_matrix=initial_segment_usage_tables,
                                                               new_usage_matrix=updated_segment_usage_tables,
                                                               n=n)

        from mir.basic.segment_usage import NormalizedSegmentUsageTable
        gene_type = self.get_gene_types_in_repertoire_dataset()
        if len(gene_type) > 1:
            raise Exception(f'Repertoire dataset can only contain one chain, but contains {gene_type}')
        gene_type = gene_type[0]
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
        with Pool(threads) as p:
            p.map(resampling_repertoire, repertoire_jobs)

        repertoires = [repertoires_dct[idx] for idx in range(len(self.repertoires))]
        return RepertoireDataset(repertoires, metadata)
