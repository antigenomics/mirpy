from mir.common import translate
from mir.common.segments import Segment
from Bio import Seq
import re


_coding = re.compile('^[ARNDCQEGHILKMFPSTWYV]+$')
_canonical = re.compile('^C[ARNDCQEGHILKMFPSTWYV]+[FW]$')


class Clonotype:
    def __init__(self, id: int | str, cells : int | list[str] = 1):
        self.id = id
        self.cells = cells

    def size(self):
        if type(self.cells) == int:
            return self.cells
        else:
            return len(self.cells)


class ClonotypeAA(Clonotype):
    def __init__(self, cdr3aa : str,
                 v : Segment = None, j : Segment = None,
                 id: int | str = -1, cells : int | list[str] = 1):
        super().__init__(id, cells)
        self.cdr3aa = cdr3aa
        self.v = v
        self.j = j

    def is_coding(self):
        return _coding.match(self.cdr3aa)
    
    def is_canonical(self):
        return _canonical.match(self.cdr3aa)


class ClonotypeNT(ClonotypeAA):
    def __init__(self, 
                 cdr3nt : str,
                 junction : tuple[int, int, int, int] = (-1, -1, -1, -1), # vend, dstart, dend, jstart
                 cdr3aa : str = None,
                 v : Segment = None, j : Segment = None,
                 id: int | str = -1, cells : int | list[str] = 1):
        if not cdr3aa:
            cdr3aa = str(Seq.translate(cdr3nt))
        super().__init__(cdr3aa, v, j, id, cells)
        self.cdr3nt = cdr3nt
        self.junction = junction
        self.v = v
        self.j = j


class PairedChainClone:
    def __init__(self, chainA : Clonotype, chainB : Clonotype):
        self.chainA = chainA
        self.chainB = chainB


class ClonalLineage:
    def __init__(self, clonotypes : list[Clonotype]):
        self.clonotypes = clonotypes