"""Resource-backed germline distance lookup for TCREMP embeddings.

Loads the per-locus distance matrices baked by
``mir/resources/germline_dist/build_germline_dist.py`` and gathers, for a batch
of query alleles against a fixed set of prototype alleles, the ``(n_query,
n_proto)`` distance matrix for a component (``V``/``J``/``CDR1``/``CDR2``).

Runtime is pure numpy — no BioPython, no gene library. Unknown alleles resolve
via the cascade *exact allele → ``*01`` → bare gene → fallback* (the fallback is
the maximum observed distance for that component, so an unknown gene sits maximally
far from every prototype).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np

from mir import get_resource_path
from mir.aliases import normalize_locus_alias, normalize_species_alias
from mir.alleles import allele_to_major, allele_with_default, strip_allele

_RESOURCE_SUBDIR = "germline_dist"
COMPONENTS = ("V", "J", "CDR1", "CDR2")


@dataclass(slots=True)
class _Component:
    """One germline component: allele index + a fallback-padded distance matrix."""

    idx: dict[str, int]
    ext: np.ndarray  # (n+1, n+1) float32, last row/col = fallback
    fallback_idx: int
    fallback: float

    def resolve(self, allele: str | None) -> int:
        for cand in (allele_with_default(allele), allele_to_major(allele), strip_allele(allele)):
            if cand and cand in self.idx:
                return self.idx[cand]
        return self.fallback_idx


class GermlineDistances:
    """Baked germline distances for one ``(species, locus)``.

    Example:
        >>> gd = GermlineDistances.load("human", "TRB")
        >>> D = gd.matrix("V", ["TRBV20-1*01"], ["TRBV20-1*01", "TRBV6-5*01"])
        >>> D.shape
        (1, 2)
        >>> float(D[0, 0])   # identical gene -> 0
        0.0
    """

    def __init__(self, species: str, locus: str, components: dict[str, _Component]):
        self.species = species
        self.locus = locus
        self._components = components

    @classmethod
    def load(cls, species: str, locus: str) -> "GermlineDistances":
        species_c = normalize_species_alias(species)
        locus_c = normalize_locus_alias(locus)
        base = Path(cast(str, get_resource_path(_RESOURCE_SUBDIR)))
        path = base / f"{species_c}_{locus_c}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No germline distances for species={species_c!r}, locus={locus_c!r} "
                f"(expected {path}). Run mir/resources/germline_dist/build_germline_dist.py."
            )
        components: dict[str, _Component] = {}
        with np.load(path) as npz:
            for comp in COMPONENTS:
                key = f"{comp}_dist"
                if key not in npz:
                    continue
                alleles = [str(a) for a in npz[f"{comp}_alleles"]]
                dist = npz[key].astype(np.float32)
                fallback = float(npz[f"{comp}_fallback"])
                n = len(alleles)
                ext = np.full((n + 1, n + 1), fallback, dtype=np.float32)
                ext[:n, :n] = dist
                components[comp] = _Component(
                    idx={a: i for i, a in enumerate(alleles)},
                    ext=ext,
                    fallback_idx=n,
                    fallback=fallback,
                )
        return cls(species_c, locus_c, components)

    def has(self, component: str) -> bool:
        return component in self._components

    def matrix(
        self, component: str, query_alleles, proto_alleles
    ) -> np.ndarray:
        """Return the ``(len(query), len(proto))`` germline distance matrix.

        Args:
            component: One of ``V``/``J``/``CDR1``/``CDR2``.
            query_alleles: Iterable of query allele strings.
            proto_alleles: Iterable of prototype allele strings.

        Raises:
            KeyError: If *component* is not available for this locus.
        """
        try:
            c = self._components[component]
        except KeyError:
            raise KeyError(
                f"Component {component!r} unavailable for {self.species}_{self.locus}; "
                f"have {sorted(self._components)}"
            ) from None
        q = np.fromiter((c.resolve(a) for a in query_alleles), dtype=np.intp)
        p = np.fromiter((c.resolve(a) for a in proto_alleles), dtype=np.intp)
        return c.ext[np.ix_(q, p)]


@lru_cache(maxsize=None)
def _load_cached(species_c: str, locus_c: str) -> GermlineDistances:
    return GermlineDistances.load(species_c, locus_c)


def load_germline_distances(species: str, locus: str) -> GermlineDistances:
    """Cached :meth:`GermlineDistances.load`, keyed by canonical species/locus."""
    return _load_cached(normalize_species_alias(species), normalize_locus_alias(locus))


if __name__ == "__main__":
    gd = GermlineDistances.load("human", "TRB")
    D = gd.matrix("V", ["TRBV20-1*01", "TRBV6-5", "TRBVNOPE*99"],
                  ["TRBV20-1*01", "TRBV6-5*01"])
    assert D.shape == (3, 2)
    assert D[0, 0] == 0.0, D[0, 0]          # identical gene -> 0
    assert D[1, 1] == 0.0                    # bare gene resolves to *01
    assert D[2, 0] == gd._components["V"].fallback  # unknown -> fallback
    assert gd.matrix("CDR1", ["TRBV20-1*01"], ["TRBV20-1*01"])[0, 0] == 0.0
    print("mir.distances.germline self-check OK; components:", sorted(gd._components))
