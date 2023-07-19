# todo: vdjmatch

from pyparsing import Iterable
from mir.common.repertoire import ClonotypeAA


class DenseMatch:
    def __init__(self, database : list[ClonotypeAA]):
        self.database = database

    def match(self, sample : list[ClonotypeAA]) -> Iterable[tuple[ClonotypeAA, ClonotypeAA,
                                                              tuple[float, float, float]]]:
        pass


class SparseMatch:
    pass