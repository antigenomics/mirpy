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
                # arda emits an *ambiguity group* — "IGHV1-69*01,IGHV1-69D*01" — as one row when
                # the alleles share an identical germline region, so their distances are identical
                # by construction. Index each member too, or a query naming one of them (e.g. the
                # very common IGHV1-69*01) resolves to nothing and silently takes the max-distance
                # fallback. Members use setdefault so a standalone row always wins.
                idx: dict[str, int] = {}
                for i, a in enumerate(alleles):
                    idx[a] = i
                for i, a in enumerate(alleles):
                    if "," in a:
                        for member in a.split(","):
                            idx.setdefault(member, i)
                components[comp] = _Component(
                    idx=idx,
                    ext=ext,
                    fallback_idx=n,
                    fallback=fallback,
                )
        return cls(species_c, locus_c, components)

    def has(self, component: str) -> bool:
        return component in self._components

    def resolve_all(self, component: str, alleles) -> np.ndarray:
        """Resolve allele strings to row indices in this component's distance table.

        Exposed so a caller holding a **fixed** panel can resolve it once. ``resolve`` runs a
        regex per allele through ``strip_allele``/``allele_with_default``, which is cheap once
        and expensive 3,584 times a sample -- see :meth:`matrix`.
        """
        return np.fromiter((self._component(component).resolve(a) for a in alleles),
                           dtype=np.intp)

    def _component(self, component: str):
        try:
            return self._components[component]
        except KeyError:
            raise KeyError(
                f"Component {component!r} unavailable for {self.species}_{self.locus}; "
                f"have {sorted(self._components)}"
            ) from None

    def matrix(
        self, component: str, query_alleles, proto_alleles, *, proto_idx=None
    ) -> np.ndarray:
        """Return the ``(len(query), len(proto))`` germline distance matrix.

        Args:
            component: One of ``V``/``J``/``CDR1``/``CDR2``.
            query_alleles: Iterable of query allele strings.
            proto_alleles: Iterable of prototype allele strings. Ignored when *proto_idx* is
                given, which is the only reason it stays in the signature.
            proto_idx: The prototype panel already resolved by :meth:`resolve_all`. The panel is
                **fixed for the life of an embedder**, so resolving it inside this method made
                the per-call cost proportional to K: measured 3,584 ``resolve`` calls per sample
                (14 calls x 256 prototypes), 3.41 ms, 10.4% of ``rsig``. Hoisting it is worth
                18.3% of a sample and changes no value -- not a cache, a loop invariant moved
                out of the loop.

        Raises:
            KeyError: If *component* is not available for this locus.
        """
        c = self._component(component)
        p = self.resolve_all(component, proto_alleles) if proto_idx is None else proto_idx
        # A repertoire has ~10^5-10^6 rows but only ~10^2 distinct alleles: resolve each distinct
        # string once, gather the small (n_distinct, K) table, then expand by row. Dict-keyed (not
        # np.unique) so nulls are handled — allele_with_default(None) is a valid fallback lookup.
        # Measured 2026-09-26: np.unique(return_inverse=True) here is 1.12x on 9,800 calls, i.e.
        # 0.05 ms of a 32.8 ms sample, and raises TypeError on a null v_call. Not a lever.
        rows: dict[str | None, int] = {}
        inv = np.fromiter(
            (rows.setdefault(a, len(rows)) for a in query_alleles), dtype=np.intp)
        small = c.ext[np.ix_(np.fromiter((c.resolve(a) for a in rows), dtype=np.intp), p)]
        return small[inv]


#: Bounded at the number of artifacts that can exist, with headroom: 14 ship (2 species x 7 loci,
#: 1.0 MB in total) and a combination that does not ship raises inside ``load`` rather than
#: landing here. ``maxsize=None`` was unbounded by construction, which is a leak waiting on a
#: caller that iterates species -- and an unbounded cache is the construct this module is not
#: allowed to have, whatever its realistic size.
@lru_cache(maxsize=16)
def _load_cached(species_c: str, locus_c: str) -> GermlineDistances:
    """Keyed on the **canonical** species and locus, which is the whole of the artifact's identity.

    Canonicalisation happens in :func:`load_germline_distances`, before the key is formed, so two
    spellings of one locus share an entry instead of loading the same file twice.
    """
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
