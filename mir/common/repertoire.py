from . import Clonotype, ClonotypeTableParser
import pandas as pd
import typing as t


class Repertoire:
    def __init__(self,
                 clonotypes: list[Clonotype],
                 sorted: bool = False,
                 metadata: dict[str, str] | pd.Series = dict()):
        self.clonotypes = clonotypes
        self.sorted = sorted
        self.metadata = metadata

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
        self.clonotypes.sort(key=lambda x: x.size(), reverse=True)

    def top(self,
            n: int = 100):
        if not sorted:
            self.sort()
        return self.clonotypes[0:n]

    def total(self):
        return sum(c.size() for c in self.clonotypes)

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

    @classmethod
    def load(cls,
             parser: ClonotypeTableParser,
             metadata: pd.DataFrame,
             paths: list[str] = None,
             n: int = None):
        metadata = metadata.copy()
        if paths:
            metadata['path'] = paths
        elif 'path' not in metadata.columns:
            raise ValueError("'path' column missing in metadata")
        repertoires = [Repertoire.load(parser, metadata=dict(row), n=n)
                       for _, row in metadata.iterrows()]
        return cls(repertoires, metadata)

    # TODO load parallel

    def __len__(self):
        return len(self.repertoires)

    def __str__(self):
        return str(self.metadata)

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.repertoires)

    def __getitem__(self, idx):
        return self.repertoires[idx]
