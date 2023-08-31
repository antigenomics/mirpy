from collections import namedtuple
import math
from typing import Iterable
from mir.common.repertoire import Clonotype


class FrequencyTable:
    def __init__(self, table: dict[int, int] = dict()) -> None:
        self.table = table

    def update(self, count: int) -> None:
        self.table[count] = self.table.get(count, 0) + 1

    def update(self, clonotype: Clonotype) -> None:
        self.update(clonotype.cells)

    def update(self, arr: Iterable[int] | Iterable[Clonotype]):
        for x in arr:
            self.update(x)

    def singletons(self) -> int:
        return self.table.get(1, 0)

    def doubletons(self) -> int:
        return self.table.get(2, 0)

    def tripletons(self) -> int:
        return self.table.get(3, 0)

    def large(self) -> int:
        return sum(species for (count, species) in self.table.items() if count > 3)

    # TODO: quantiles

    def unique(self) -> int:
        return sum(self.table.values())

    def total(self) -> int:
        return sum(count * species for (count, species) in self.table.items())


class DiversityIndices:
    def __init__(self, table: FrequencyTable) -> None:
        self.table = table
        self.total = self.table.total()
        self.singletons = self.table.singletons()
        self.doubletons = self.table.doubletons()
        self.species = self.table.unique()

    def shannon(self) -> float:
        return -sum(species * count / self.total * math.log(count / self.total) for
                    (count, species) in self.table.items())

    def simpson(self) -> float:
        return -sum(species * (count / self.total) ** 2 for
                    (count, species) in self.table.items())

    def hill(self, q) -> float:
        if q == 1:
            return math.exp(self.shannon())
        else:
            return sum(species * (count / self.total) ** q for
                       (count, species) in self.table.items()) ** (1. / (1 - q))

    def unseen(self) -> float:
        return (self.table.singletons() + 1.) * (self.table.singletons() - 1.) / \
            2. / (self.table.doubletons() + 1)

    def chao(self) -> float:
        return self.table.unique() + self.unseen()


RarefactionPoint = namedtuple(
    'RarefactionPoint', 'count_star species var_species interpolation')


class Rarefaction:
    def __init__(self, table: FrequencyTable) -> None:
        self.table = table
        div_indices = DiversityIndices(self.table)
        self.total = div_indices.total
        self.species_obs = div_indices.species
        self.singletons = div_indices.singletons
        self.species_unseen = div_indices.unseen()
        self.species_est = self.species_obs + self.species_unseen

    def rarefy(self, count_star: int) -> RarefactionPoint:
        phi = count_star / self.total
        if phi <= 1:
            species_star = 0
            species_star_sq = 0
            for (count, species) in self.table.items():
                species_star += species * (1. - (1. - phi) ** count)
                species_star_sq += species * (1. - (1. - phi) ** count) ** 2
            return RarefactionPoint(count_star,
                                    species_star,
                                    species_star_sq - species_star * species_star / self.species_est,
                                    True)
        else:
            return RarefactionPoint(count_star,
                                    self.species_obs + self.species_unseen *
                                    (1. - math.exp(-(phi - 1.) *
                                     self.singletons / self.species_unseen)),
                                    -1., # TODO: revise error bars
                                    False)
