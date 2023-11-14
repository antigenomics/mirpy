import typing as t
from collections import defaultdict
from multiprocessing import Pool, Manager

import pandas as pd

from mir.basic.sampling import RepertoireSampling
from mir.basic.segment_usage import SegmentUsageTable, NormalizedSegmentUsageTable
from . import Clonotype, ClonotypeTableParser


class Repertoire:
    def __init__(self,
                 clonotypes: list[Clonotype],
                 sorted: bool = False,
                 metadata: dict[str, str] | pd.Series = dict()):
        self.clonotypes = clonotypes
        self.sorted = sorted
        self.metadata = metadata
        self.segment_usage = None
        self.number_of_clones = len(self.clonotypes)
        self.number_of_reads = sum([x.size() for x in self.clonotypes])

    @classmethod
    def load(cls,
             parser: ClonotypeTableParser,
             metadata: dict[str, str] | pd.Series = dict(),
             path: str = None,
             n: int = None):
        if not path:
            if 'path' not in metadata:
                raise ValueError("'path' is missing in metadata")
            path = metadata['path']
        else:
            metadata['path'] = path
        return cls(clonotypes=parser.parse(path, n=n), metadata=metadata)

    def __copy__(self):
        return Repertoire(self.clonotypes, self.sorted, self.metadata)

    def sort(self):
        self.sorted = True
        self.sorted_by = 'cells'
        self.clonotypes.sort(key=lambda x: x.size(), reverse=True)

    def sort_by_clone_metadata(self, sort_by: str, reverse=False):
        self.sorted = True
        self.sorted_by = sort_by
        for clone in self.clonotypes:
            if sort_by not in clone.clone_metadata:
                raise Exception(f'Cannot sort by {sort_by}, {clone} has no such metadata!')
        self.clonotypes.sort(key=lambda x: x.clone_metadata[sort_by], reverse=reverse)

    def top(self, n: int = 100):
        if not sorted:
            self.sort()
        return self.clonotypes[0:n]

    def total(self):
        return sum(c.size() for c in self.clonotypes)

    def evaluate_segment_usage(self):
        if self.segment_usage is None:
            self.segment_usage = defaultdict(int)
            for c in self.clonotypes:
                # TODO add type of clonotype assertion
                self.segment_usage[c.v.id] += 1
                self.segment_usage[c.j.id] += 1
        return self.segment_usage

    def serialize(self):
        serialization_dct = defaultdict(list)
        for i, clonotype in enumerate(self.clonotypes):
            serialization_res = clonotype.serialize()
            keys_to_process = set(serialization_res.keys()).union(serialization_dct.keys())
            for k in keys_to_process:
                if k in serialization_res:
                    serialization_dct[k].append(serialization_res[k])
                else:
                    serialization_dct[k].append(None)
        return pd.DataFrame(serialization_dct)

    def __getitem__(self, idx):
        return self.clonotypes[idx]

    def __len__(self):
        return len(self.clonotypes)

    def __str__(self):
        return f'Repertoire of {self.__len__()} clonotypes and {self.total()} cells:\n' + \
               '\n'.join([str(x) for x in self.clonotypes[0:5]]) + \
               '\n' + str(self.metadata) + '\n...'

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.clonotypes)

    # TODO subsample
    # TODO aggregate redundant
    # TODO group by and aggregate


class RepertoireDataset:
    def __init__(self,
                 repertoires: t.Iterable[Repertoire],
                 metadata: pd.DataFrame = None) -> None:
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

    @classmethod
    def load(cls,
             parser: ClonotypeTableParser,
             metadata: pd.DataFrame,
             paths: list[str] = None,
             n: int = None,
             threads: int = 1):
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
            repertoires_dct[path] = Repertoire.load(parser, metadata=row_dict, n=n)

        repertoire_jobs = [row for _, row in metadata.iterrows()]
        with Pool(threads) as p:
            p.map(inner_repertoire_load, repertoire_jobs)

        repertoires = [repertoires_dct[path] for path in metadata.path]
        return cls(repertoires, metadata)

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

    def resample(self, updated_segment_usage_tables: list[SegmentUsageTable] = None, n: int=None, threads: int = 1):
        pass
        #TODO
        # global resampling_repertoire
        # repertoires_dct = Manager().dict()
        # def resampling_repertoire(idx):
        #     repertoires_dct[idx] = RepertoireSampling().sample(repertoire=self.repertoires[idx],
        #                                                        old_usage_matrix=initial_segment_usage_tables,
        #                                                        new_usage_matrix=updated_segment_usage_tables,
        #                                                        n=n)
        # initial_segment_usage_tables = []
        # if updated_segment_usage_tables is None:
        #     initial_segment_usage_tables = [NormalizedSegmentUsageTable.load_from_repertoire_dataset(
        #         repertoire_dataset=self,
        #         gene=,
        #         segment_type=um.segment_type)]
        # for um in updated_segment_usage_tables:
        #     initial_segment_usage_tables.append(NormalizedSegmentUsageTable.load_from_repertoire_dataset(
        #         repertoire_dataset=self,
        #         gene=um.gene,
        #         segment_type=um.segment_type))
        #
        # metadata = self.metadata.copy()
        # repertoire_jobs = [i for i in range(len(self.repertoires))]
        # with Pool(threads) as p:
        #     p.map(inner_repertoire_load, repertoire_jobs)
        #
        # repertoires = [repertoires_dct[idx] for idx in range(len(self.repertoires))]
        # return RepertoireDataset(repertoires, metadata)
