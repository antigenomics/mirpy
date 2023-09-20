from collections import namedtuple
from functools import cached_property
import typing as t
import pandas as pd
import numpy as np
import math

from ..common import Repertoire, RepertoireDataset


class FrequencyTable:
    def __init__(self, table: dict[int, int]) -> None:
        self._table = table

    @classmethod
    def from_repertoire(cls, repertoire: Repertoire):
        tbl = dict()
        for clonotype in repertoire:
            tbl[clonotype.cells] = tbl.get(clonotype.cells, 0) + 1
        return cls(tbl)

    def items(self) -> t.ItemsView[int, int]:
        return self._table.items()

    def singletons(self) -> int:
        return self._table.get(1, 0)

    def doubletons(self) -> int:
        return self._table.get(2, 0)

    def tripletons(self) -> int:
        return self._table.get(3, 0)

    def large(self) -> int:
        return sum(species for (count, species) in self._table.items() if count > 3)

    # TODO: quantiles

    @cached_property
    def species(self) -> int:
        return sum(self._table.values())

    @cached_property
    def individuals(self) -> int:
        return sum(count * species for (count, species) in self._table.items())

    def __str__(self) -> str:
        return f'Frequency table of {self.species} species and ' \
            f'{self.individuals} individuals:\n' + \
            f's1={self.singletons()}\ns2={self.doubletons()}\n' + \
            f's3={self.tripletons()}\ns4+={self.large()}\n'

    def __repr__(self) -> str:
        return self.__str__()


HillCurvePoint = namedtuple(
    'HillCurvePoint', 'q H_q')


class DiversityIndices:
    def __init__(self, source: FrequencyTable | Repertoire) -> None:
        if isinstance(source, Repertoire):
            source = FrequencyTable.from_repertoire(source)
        self._table = source

    def obs(self) -> int:
        return self._table.species

    def obs_per_mil(self) -> int:
        return self._table.species * 1.0e6 / self._table.individuals

    def shannon(self) -> float:
        return -sum(species * count / self._table.individuals *
                    math.log(count / self._table.individuals) for
                    (count, species) in self._table.items())

    def pielou(self) -> float:
        return self.shannon() / math.log(self._table.species)

    def simpson(self) -> float:
        return -sum(species * (count / self._table.individuals) ** 2 for
                    (count, species) in self._table.items())

    def unseen(self) -> float:
        return self._table.singletons() * (self._table.singletons() - 1.) / \
            2. / (self._table.doubletons() + 1.)

    def chao(self) -> float:
        return self._table.species + self.unseen()

    def hill(self, q) -> float:
        if q == 1:
            return math.exp(self.shannon())
        else:
            return sum(species * (count / self._table.individuals) ** q for
                       (count, species) in self._table.items()) ** (1. / (1. - q))

    def hill_curve(self, q_values: list[float] = [0., 1., 2.] +
                   [2 ** x for x in np.linspace(-5, 5, 100)]) -> list[HillCurvePoint]:
        return [HillCurvePoint(x, self.hill(x)) for x in q_values]

    @staticmethod
    def for_dataset(dataset: RepertoireDataset):
        return pd.DataFrame([dict(repertoire.metadata) |
                             DiversityIndices(
                                 FrequencyTable.from_repertoire(repertoire)).as_dict()
                             for repertoire in dataset])

    @staticmethod
    def hill_curve_for_dataset(dataset: RepertoireDataset,
                               q_values: list[float] = [0., 1., 2.] +
                               [2 ** x for x in np.linspace(-5, 5, 100)]):
        rows = []
        for repertoire in dataset:
            meta = dict(repertoire.metadata)
            div = DiversityIndices(repertoire)
            for point in div.hill_curve(q_values):
                rows.append(meta | point._asdict())
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        return f'Diversity indices for {self._table.species} species and ' + \
            f'{self._table.individuals} individuals:\n' + \
            f'Obs={self.obs()}\nOPM={self.obs_per_mil()}\n' + \
            f'H={self.shannon()}\nHpielou={self.pielou()}\n' + \
            f'Chao={self.chao()}\nHill={self.hill_curve()}'

    def __repr__(self) -> str:
        return self.__str__()

    def as_dict(self):
        return {'depth': self._table.individuals,
                'obs': self.obs(),
                'opm': self.obs_per_mil(),
                'shannon': self.shannon(),
                'pielou': self.pielou(),
                'chao': self.chao()}


RarefactionPoint = namedtuple(
    'RarefactionPoint', 'count_star species var_species interp')


class RarefactionCurve:
    def __init__(self, source: FrequencyTable | Repertoire) -> None:
        if isinstance(source, Repertoire):
            source = FrequencyTable.from_repertoire(source)
        self._table = source
        self._div_indices = DiversityIndices(source)

    def span(self) -> int:
        return self._table.individuals

    def rarefy(self,
               count_star: int) -> RarefactionPoint:
        phi = count_star / self.span()
        if phi <= 1:
            species_star = 0
            species_star_sq = 0
            for (count, species) in self._table.items():
                species_star += species * (1. - (1. - phi) ** count)
                species_star_sq += species * (1. - (1. - phi) ** count) ** 2
            return RarefactionPoint(count_star,
                                    species_star,
                                    species_star_sq -
                                    species_star * species_star / self._div_indices.chao(),
                                    True)
        else:
            # differentiate byurselves lads, don't steal the code it had a mismatch in vdjtools;)
            n = self.span()
            x = count_star - n
            s = self._div_indices.obs()
            s0 = self._div_indices.unseen()
            s1 = self._table.singletons()
            s2 = self._table.doubletons()
            dsds1 = (2. * s1 - 1.) / 2. / (s2 + 1.)
            dsds2 = -s1 * (s1 - 1.) / 2. / (s2 + 1.) / (s2 + 1.)
            b = 1. - (1. - s1 / n / s0) ** x
            db = -x * (1. - s1 / n / s0) ** (x - 1.)
            jac1 = dsds1 * b + s0 * db * \
                (-1. / n / s0 + s1 / n / s0 / s0 * dsds1)
            jac2 = dsds2 * b + s0 * db * (s1 / n / s0 / s0 * dsds2)
            cov12 = -s1 * s2 / (s + s0)
            cov11 = s1 * (1. - s1 / (s + s0))
            cov22 = s2 * (1. - s2 / (s + s0))
            return RarefactionPoint(count_star,
                                    s + s0 * b,
                                    s * (1. - s / (s + s0)) +
                                              jac1 * jac1 * cov11 +
                                              jac2 * jac2 * cov22 +
                                              2. * jac1 * jac2 * cov12,
                                    False)

    def rarefy_points(self,
                      counts_star: list[int] = [],
                      n_points: int = 30, extrapolate_factor: float = 1.) -> list[RarefactionPoint]:
        if not counts_star.any():
            counts_star = np.linspace(
                0., self.span() * extrapolate_factor, n_points)
        return [self.rarefy(cc) for cc in counts_star]

    @staticmethod
    def for_dataset(dataset: RepertoireDataset,
                    counts_star: list[int] = [],
                    n_points: int = 30, extrapolate_factor: float = 1.):
        curves = [(repertoire, RarefactionCurve(repertoire))
                  for repertoire in dataset]
        if not counts_star:
            max_span = max([curve.span() for _, curve in curves])
            counts_star = np.linspace(
                0., max_span * extrapolate_factor, n_points)
        rows = []
        for repertoire, curve in curves:
            meta = dict(repertoire.metadata)
            for point in curve.rarefy_points(counts_star=counts_star):
                rows.append(meta | point._asdict())
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        return f'Rarefaction curve={[self.rarefy(x) for x in range(0, 1000, 100)]}'

    def __repr__(self) -> str:
        return self.__str__()
