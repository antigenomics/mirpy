from collections import defaultdict

import pandas as pd

from mir.common.clonotype import ClonotypeAA
from mir.common.parser import ClonotypeTableParser


class Repertoire:
    def __init__(self,
                 clonotypes: list[ClonotypeAA],
                 sorted: bool = False,
                 metadata: dict[str, str] | pd.Series = dict(),
                 gene: str = None):
        if gene is not None:
            self.clonotypes = [x for x in clonotypes if gene in x.v.gene and gene in x.j.gene  and (
                    x.d.gene is None or gene in x.d.gene)]
        else:
            self.clonotypes = clonotypes
        self.sorted = sorted
        self.metadata = metadata
        self.segment_usage = None
        self.number_of_clones = len(self.clonotypes)
        self.number_of_reads = sum([x.size() for x in self.clonotypes])
        self.gene = gene

    @classmethod
    def load(cls,
             parser: ClonotypeTableParser,
             metadata: dict[str, str] | pd.Series = dict(),
             path: str = None,
             n: int = None,
             sample: bool = False,
             gene: str = None):
        if not path:
            if 'path' not in metadata:
                raise ValueError("'path' is missing in metadata")
            path = metadata['path']
        else:
            metadata['path'] = path
        return cls(clonotypes=parser.parse(path, n=n, sample=sample), metadata=metadata, gene=gene)

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
                if c.d is not None:
                    self.segment_usage[c.d.id] += 1
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

    def __add__(self, other):
        if not isinstance(other, Repertoire):
            raise ValueError('Can only sum objects of class Repertoire')
        new_metadata = dict(self.metadata)  # or orig.copy()
        new_metadata.update(dict(other.metadata))
        new_clonotypes = self.clonotypes + other.clonotypes
        return Repertoire(
            clonotypes=new_clonotypes,
            sorted=False,
            metadata=new_metadata)
    # TODO subsample
    # TODO aggregate redundant
    # TODO group by and aggregate
