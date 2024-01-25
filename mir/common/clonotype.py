import re
import typing as t

from Bio.Seq import translate

from mir.common.segments import Segment, _SEGMENT_CACHE

_CODING_AA = re.compile('^[ARNDCQEGHILKMFPSTWYV]+$')
_CANONICAL_AA = re.compile('^C[ARNDCQEGHILKMFPSTWYV]+[FW]$')


class ClonotypePayload:
    def __init__(self) -> None:
        self.number_of_reads = None
        self.number_of_samples = None


# TODO tcrnet payload etc / consider moving to separate module


class Clonotype:
    __slots__ = 'id', 'cells', 'payload', 'clone_metadata'

    def __init__(self,
                 id: int | str,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        self.id = id
        self.cells = cells
        self.payload = payload
        self.clone_metadata = {}

    def size(self) -> int:
        if isinstance(self.cells, int):
            return self.cells
        else:
            return len(self.cells)

    def serialize(self) -> dict:
        return {'id': self.id,
                'cells': self.cells}

    def __str__(self):
        return 'κ' + str(self.id)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return Clonotype(self.id, self.cells, self.payload)


class ClonotypeAA(Clonotype):
    __slots__ = 'cdr3aa', 'v', 'd', 'j'

    def __init__(self, cdr3aa: str,
                 v: str | Segment = None,
                 d: str | Segment = None,
                 j: str | Segment = None,
                 id: int | str = -1,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        super().__init__(id, cells, payload)
        self.cdr3aa = cdr3aa
        if isinstance(v, str):
            v = _SEGMENT_CACHE.get_or_create(v)
        self.v = v
        if isinstance(d, str):
            d = _SEGMENT_CACHE.get_or_create(d)
        self.d = d
        if isinstance(j, str):
            j = _SEGMENT_CACHE.get_or_create(j)
        self.j = j

    def is_coding(self):
        return _CODING_AA.match(self.cdr3aa)

    def is_canonical(self):
        return _CANONICAL_AA.match(self.cdr3aa)

    def serialize(self) -> dict:
        return {'id': self.id,
                'cells': self.cells,
                'cdr3aa': self.cdr3aa,
                'v': self.v,
                'd': self.d,
                'j': self.j}

    def __str__(self):
        return super().__str__() + ' ' + self.cdr3aa

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return ClonotypeAA(self.cdr3aa, self.v, self.d, self.j, self.id, self.cells, self.payload)

    def __getitem__(self, segment):
        if segment == 'v':
            return self.v
        elif segment == 'd':
            return self.d
        elif segment == 'j':
            return self.j
        else:
            raise Exception('You can get only v/d/j gene by itemization')


class JunctionMarkup:
    __slots__ = 'vend', 'dstart', 'dend', 'jstart'

    def __init__(self,
                 vend: int,
                 dstart: int,
                 dend: int,
                 jstart: int):
        self.vend = vend
        self.dstart = dstart
        self.dend = dend
        self.jstart = jstart


class ClonotypeNT(ClonotypeAA):
    __slots__ = 'cdr3nt', 'junction'

    def __init__(self,
                 cdr3nt: str,
                 junction: JunctionMarkup = None,
                 cdr3aa: str = None,
                 v: str | Segment = None,
                 d: str | Segment = None,
                 j: str | Segment = None,
                 id: int | str = -1,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        if not cdr3aa:
            cdr3aa = translate(cdr3nt)
        super().__init__(cdr3aa, v, d, j, id, cells, payload)
        self.cdr3nt = cdr3nt
        self.junction = junction

    def serialize(self) -> dict:
        return {'id': self.id,
                'cells': self.cells,
                'cdr3aa': self.cdr3aa,
                'cdr3nt': self.cdr3nt,
                'v': self.v,
                'd': self.d,
                'j': self.j}

    def __str__(self):
        return super().__str__() + ' ' + self.cdr3nt

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return ClonotypeNT(self.cdr3nt, self.junction, self.cdr3aa, self.v, self.d, self.j, self.id, self.cells,
                           self.payload)


# TODO
class PairedChainClone(Clonotype):
    def __init__(self,
                 chainA: Clonotype,
                 chainB: Clonotype,
                 id: int | str = -1,
                 cells: int | list[str] = 1,
                 payload: t.Any = None):
        super().__init__(id, cells, payload)
        self.chainA = chainA
        self.chainB = chainB

    def __str__(self):
        return 'alpha ' + self.chainA.__str__() + ' beta ' + self.chainB.__str__()

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return PairedChainClone(self.chainA, self.chainB)


# TODO
class ClonalLineage:
    def __init__(self, clonotypes: list[Clonotype]):
        self.clonotypes = clonotypes

class ClonotypeDatset:
    def __init__(self, clonotypes):
        self.clonotypes = clonotypes