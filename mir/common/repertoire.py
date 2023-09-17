from ..basic import FrequencyTable
from . import Clonotype
import typing as t


class Repertoire:
    def __init__(self,
                 clonotypes: list[Clonotype],
                 sorted: bool = False,
                 metadata: t.Any = None):
        self.clonotypes = clonotypes
        self.sorted = sorted
        self.metadata = metadata

    def sort(self):
        self.sorted = True
        self.clonotypes.sort(key=lambda x: x.size(), reverse=True)

    def top(self,
            n: int = 100):
        if not sorted:
            self.sort()
        return self.clonotypes[0:n]

    def frequency_table(self) -> FrequencyTable:
        tbl = dict()
        for cc in self.clonotypes:
            tbl[cc.cells] = tbl.get(cc.cells, 0) + 1
        return FrequencyTable(tbl)

    def total(self):
        return sum(c.size() for c in self.clonotypes)

    def __len__(self):
        return len(self.clonotypes)

    def __str__(self):
        return f'Repertoire of {self.__len__()} clonotypes and {self.total} cells:\n' + \
            '\n'.join(map(str, self.clonotypes[0:5])
                      ) + '\n' + self.metadata + '\n...'

    def __repr__(self):
        return self.__str__()

    def __iter__(self):
        return iter(self.clonotypes)

    # TODO subsample
    # TODO group my and aggregate
