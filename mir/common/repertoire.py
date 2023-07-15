from mir.common import translate
from mir.common.segments import Segment


class Clonotype:
    def __init__(self, id: int | str, cells : int | list[str] = 1):
        self.id = id
        self.cells = cells

    def size(self):
        if type(self.cells) == int:
            return self.cells
        else:
            return len(self.cells)


class ClonotypeJ(Clonotype):
    def __init__(self, cdr3aa : str,
                 v : Segment = None, j : Segment = None,
                 id: int | str = -1,
                 cells : int | list[str] = 1):
        super().__init__(id, cells)
        self.cdr3aa = cdr3aa
        self.v = v
        self.j = j


class ClonotypeR(ClonotypeJ):
    def __init__(self, 
                 cdr3nt : str,
                 junction = (-1, -1, -1, -1), # vend, dstart, dend, jstart
                 cdr3aa : str = None,
                 v : Segment = None, j : Segment = None,
                 id: int | str = -1,
                 cells : int | list[str] = 1):
        super().__init__(cdr3aa, v, j, id, cells)
        self.cdr3nt = cdr3nt
        self.junction = junction
        if not self.cdr3aa:
            self.cdr3aa = translate(self.cdr3nt) 
        self.v = v
        self.j = j