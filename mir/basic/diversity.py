from collections import namedtuple
from functools import cached_property
import typing as t
import math


class FrequencyTable:
    def __init__(self, table: dict[int, int]) -> None:
        self._table = table

    def items(self) -> t.ItemsView[int, int]:
        return self._table.items()

    @cached_property
    def singletons(self) -> int:
        return self._table.get(1, 0)

    @cached_property
    def doubletons(self) -> int:
        return self._table.get(2, 0)

    @cached_property
    def tripletons(self) -> int:
        return self._table.get(3, 0)

    @cached_property
    def large(self) -> int:
        return sum(species for (count, species) in self._table.items() if count > 3)

    # TODO: quantiles

    @cached_property
    def unique(self) -> int:
        return sum(self._table.values())

    @cached_property
    def total(self) -> int:
        return sum(count * species for (count, species) in self._table.items())

    def __str__(self) -> str:
        return f'Frequency table of {self.unique} species and {self.total} reads:\n' + \
            f's1={self.singletons}\ns2={self.doubletons}\n' + \
            f's3={self.tripletons}\ns4+={self.large}\n'

    def __repr__(self) -> str:
        return self.__str__()


HillCurvePoint = namedtuple(
    'HillCurvePoint', 'q H_q')


class DiversityIndices:
    def __init__(self, table: FrequencyTable) -> None:
        self._table = table

    @cached_property
    def obs(self) -> int:
        return self._table.unique
    
    @cached_property
    def obs_per_mil(self) -> int:
        return self._table.unique * 1.0e6 / self._table.total

    @cached_property
    def shannon(self) -> float:
        return -sum(species * count / self._table.total * math.log(count / self._table.total) for
                    (count, species) in self._table.items())

    @cached_property
    def pielou(self) -> float:
        return self.shannon / math.log(self._table.unique)

    @cached_property
    def simpson(self) -> float:
        return -sum(species * (count / self._table.total) ** 2 for
                    (count, species) in self._table.items())

    @cached_property
    def unseen(self) -> float:
        return (self._table.singletons + 1.) * (self._table.singletons - 1.) / \
            2. / (self._table.doubletons + 1.)

    @cached_property
    def chao(self) -> float:
        return self._table.unique + self.unseen

    def hill(self, q) -> float:
        if q == 1:
            return math.exp(self.shannon)
        else:
            return sum(species * (count / self._table.total) ** q for
                       (count, species) in self._table.items()) ** (1. / (1 - q))

    @cached_property
    def hill_curve(self) -> list[HillCurvePoint]:
        return [HillCurvePoint(x, self.hill(x)) for x in [0.005, 0.01, 0.05, 0.1, 0.5,
                                                          1., 
                                                          2., 10., 20., 100., 200.]]

    def __str__(self) -> str:
        return f'Diversity indices for {self._table.unique} species and {self._table.total} reads:\n' + \
            f'Obs={self.obs}\nOPM={self.obs_per_mil}\n' + \
            f'H={self.shannon}\nHpielou={self.pielou}\n' + \
            f'Chao={self.chao}\nHill={self.hill_curve}'

    def __repr__(self) -> str:
        return self.__str__()


RarefactionPoint = namedtuple(
    'RarefactionPoint', 'count_star species var_species interp')


class RarefactionCurve:
    def __init__(self, table: FrequencyTable) -> None:
        self._table = table
        self._div_indices = DiversityIndices(table)

    def rarefy(self, count_star: int) -> RarefactionPoint:
        phi = count_star / self._table.total
        if phi <= 1:
            species_star = 0
            species_star_sq = 0
            for (count, species) in self._table.items():
                species_star += species * (1. - (1. - phi) ** count)
                species_star_sq += species * (1. - (1. - phi) ** count) ** 2
            return RarefactionPoint(count_star,
                                    species_star,
                                    species_star_sq - 
                                    species_star * species_star / self._div_indices.chao,
                                    True)
        else:
            return RarefactionPoint(count_star,
                                    self._table.unique + self._div_indices.unseen *
                                    (1. - math.exp(-(phi - 1.) *
                                     self._table.singletons / self._div_indices.unseen)),
                                    -1.,  # TODO: revise error bars
                                    False)

    def __str__(self) -> str:
        return f'Rarefaction curve={[self.rarefy(x) for x in range(0, 1000, 100)]}'

    def __repr__(self) -> str:
        return self.__str__()
