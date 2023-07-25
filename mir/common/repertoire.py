from collections import namedtuple
from functools import cached_property
from mir.common.segments import _SEGMENT_CACHE, Segment
from Bio.Seq import translate
import re
import typing as t


_CODING_AA = re.compile('^[ARNDCQEGHILKMFPSTWYV]+$')
_CANONICAL_AA = re.compile('^C[ARNDCQEGHILKMFPSTWYV]+[FW]$')


class ClonotypePayload:
    def __init__(self) -> None:
        pass
# todo: tcrnet payload etc

class Clonotype:
    __slots__ = 'id', 'cells', 'payload'

    def __init__(self, 
                 id: int | str, 
                 cells : int | list[str] = 1, 
                 payload : t.Any = None):
        self.id = id
        self.cells = cells
        self.payload = payload

    def size(self) -> int:
        if isinstance(self.cells, int):
            return self.cells
        else:
            return len(self.cells)
        
    def __str__(self):
        return f'Clonotype {self.id}'
    
    def __repr__(self):
        return self.__str__()


class ClonotypeAA(Clonotype):
    __slots__ = 'cdr3aa', 'v', 'j'

    def __init__(self, cdr3aa : str,
                 v : str | Segment = None, 
                 j : str | Segment = None,
                 id: int | str = -1,
                 cells : int | list[str] = 1, 
                 payload : t.Any  = None):
        super().__init__(id, cells, payload)
        self.cdr3aa = cdr3aa
        if isinstance(v, str):
            v = _SEGMENT_CACHE.get_or_create(v)
        self.v = v
        if isinstance(j, str):
            j = _SEGMENT_CACHE.get_or_create(j)
        self.j = j

    def is_coding(self):
        return _CODING_AA.match(self.cdr3aa)
    
    def is_canonical(self):
        return _CANONICAL_AA.match(self.cdr3aa)
    
    def __str__(self):
        return f'Clonotype {self.id} {self.cdr3aa}'


JunctionMarkup = namedtuple('JunctionMarkup', 'vend dstart dend jstart')
_DUMMY_JUNCTION = JunctionMarkup(-1, -1, -1, -1)


class ClonotypeNT(ClonotypeAA):
    __slots__ = 'cdr3nt', 'junction'

    def __init__(self, 
                 cdr3nt : str,
                 junction : JunctionMarkup = _DUMMY_JUNCTION,
                 cdr3aa : str = None,
                 v : str | Segment = None, 
                 j : str | Segment = None,
                 id: int | str = -1, 
                 cells : int | list[str] = 1, 
                 payload : t.Any = None):
        if not cdr3aa:
            cdr3aa = translate(cdr3nt)
        super().__init__(cdr3aa, v, j, id, cells, payload)
        self.cdr3nt = cdr3nt
        self.junction = junction

    def __str__(self):
        return f'Clonotype {self.id} {self.cdr3aa} {self.cdr3nt}'


class PairedChainClone:
    def __init__(self, chainA : Clonotype, chainB : Clonotype):
        self.chainA = chainA
        self.chainB = chainB


class ClonalLineage:
    def __init__(self, clonotypes : list[Clonotype]):
        self.clonotypes = clonotypes


class Repertoire:
    def __init__(self, 
                 clonotypes : list[Clonotype],
                 sorted : bool = False,
                 metadata : t.Any = None):
        self.clonotypes = clonotypes
        self.sorted = sorted
        self.metadata = metadata

    def sort(self):
        self.sorted = True
        self.clonotypes.sort(key = lambda x: x.size(), reverse = True)

    def top(self, n : int = 100):
        if not sorted:
            self.sort()
        return self.clonotypes[0:n]
    
    @cached_property
    def total(self):
        return sum(c.size() for c in self.clonotypes)
    
    def __len__(self):
        return len(self.clonotypes)
    
    def __str__(self):
        return f'Repertoire of {self.__len__()} clonotypes and {self.total} cells:\n' + \
            '\n'.join(map(str, self.clonotypes[0:5])) + '\n...'