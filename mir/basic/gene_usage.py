"""Gene usage tracking for immune repertoires.

:class:`GeneUsage` accumulates V-J gene combination counts from
:class:`~mir.common.repertoire.LocusRepertoire` or
:class:`~mir.common.repertoire.SampleRepertoire` objects and exposes joint and
marginal usage statistics together with Laplace-smoothed fractions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire

_VJPair = tuple[str, str]


def _strip_allele(gene: str) -> str:
    """``"TRBV1*01"`` → ``"TRBV1"``."""
    return gene.split("*")[0]


class GeneUsage:
    """Joint and marginal V-J gene usage statistics.

    Stores per-locus clonotype counts and duplicate-count totals for every
    observed (V-gene, J-gene) pair.  Alleles are stripped automatically so
    ``TRBV1*01`` and ``TRBV1`` are treated as the same gene.

    Build with the class-method constructors rather than calling ``__init__``
    directly.

    Examples
    --------
    >>> gu = GeneUsage.from_repertoire(trb_repertoire)
    >>> gu.vj_fraction("TRB")
    {('TRBV12-3', 'TRBJ1-2'): 0.42, ...}
    """

    def __init__(self) -> None:
        # locus → {(v_base, j_base): [n_clones, n_dc]}
        self._data: dict[str, dict[_VJPair, list[int]]] = {}
        # locus → [total_clones, total_dc]
        self._totals: dict[str, list[int]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_repertoire(
        cls,
        repertoire: "LocusRepertoire",
        *,
        locus: str = "",
    ) -> "GeneUsage":
        """Build from a :class:`~mir.common.repertoire.LocusRepertoire`.

        Args:
            repertoire: Source locus repertoire.
            locus: Override locus.  When empty the repertoire's own locus is used.
        """
        obj = cls()
        obj._add_locus_repertoire(repertoire, locus=locus)
        return obj

    @classmethod
    def from_sample(cls, sample: "SampleRepertoire") -> "GeneUsage":
        """Build from a :class:`~mir.common.repertoire.SampleRepertoire`.

        Iterates over all loci in the sample.
        """
        obj = cls()
        for loc, locus_rep in sample.loci.items():
            obj._add_locus_repertoire(locus_rep, locus=loc)
        return obj

    @classmethod
    def from_list(cls, repertoires) -> "GeneUsage":
        """Build by accumulating data from a list of repertoire objects.

        Each element may be a :class:`~mir.common.repertoire.LocusRepertoire`
        or a :class:`~mir.common.repertoire.SampleRepertoire`.
        """
        from mir.common.repertoire import SampleRepertoire

        obj = cls()
        for rep in repertoires:
            if isinstance(rep, SampleRepertoire):
                for loc, locus_rep in rep.loci.items():
                    obj._add_locus_repertoire(locus_rep, locus=loc)
            else:
                obj._add_locus_repertoire(rep)
        return obj

    def _add_locus_repertoire(self, repertoire, *, locus: str = "") -> None:
        loc = locus or repertoire.locus or ""
        locus_data = self._data.setdefault(loc, {})
        locus_totals = self._totals.setdefault(loc, [0, 0])
        for clone in repertoire.clonotypes:
            v = _strip_allele(clone.v_gene or "")
            j = _strip_allele(clone.j_gene or "")
            dc = clone.duplicate_count or 0
            entry = locus_data.setdefault((v, j), [0, 0])
            entry[0] += 1
            entry[1] += dc
            locus_totals[0] += 1
            locus_totals[1] += dc

    # ------------------------------------------------------------------
    # Loci
    # ------------------------------------------------------------------

    @property
    def loci(self) -> list[str]:
        """Loci with observed data."""
        return list(self._data.keys())

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------

    def total(self, locus: str, *, count: str = "clonotypes") -> int:
        """Total count for *locus*.

        Args:
            locus: IMGT locus code.
            count: ``"clonotypes"`` (unique rearrangements) or ``"duplicates"``.
        """
        totals = self._totals.get(locus, [0, 0])
        return totals[0] if count == "clonotypes" else totals[1]

    # ------------------------------------------------------------------
    # Usage accessors
    # ------------------------------------------------------------------

    def vj_usage(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
    ) -> dict[_VJPair, int]:
        """Joint V-J usage for *locus*.

        Args:
            locus: IMGT locus code.
            count: ``"clonotypes"`` or ``"duplicates"``.

        Returns:
            Dict mapping ``(v_base, j_base)`` to the requested count.
        """
        idx = 0 if count == "clonotypes" else 1
        return {pair: vals[idx] for pair, vals in self._data.get(locus, {}).items()}

    def v_usage(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
    ) -> dict[str, int]:
        """Marginal V-gene usage (sum over all J) for *locus*."""
        idx = 0 if count == "clonotypes" else 1
        result: dict[str, int] = defaultdict(int)
        for (v, _j), vals in self._data.get(locus, {}).items():
            result[v] += vals[idx]
        return dict(result)

    def j_usage(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
    ) -> dict[str, int]:
        """Marginal J-gene usage (sum over all V) for *locus*."""
        idx = 0 if count == "clonotypes" else 1
        result: dict[str, int] = defaultdict(int)
        for (_v, j), vals in self._data.get(locus, {}).items():
            result[j] += vals[idx]
        return dict(result)

    # ------------------------------------------------------------------
    # Fractions with Laplace smoothing
    # ------------------------------------------------------------------

    def vj_fraction(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
        pseudocount: float = 1.0,
    ) -> dict[_VJPair, float]:
        """Laplace-smoothed V-J fraction for *locus*.

        Fractions sum to 1 over observed pairs using::

            (n_observed + pseudocount) / (total + n_observed_pairs * pseudocount)

        Args:
            locus: IMGT locus code.
            count: ``"clonotypes"`` or ``"duplicates"``.
            pseudocount: Added to each count and the denominator term.
        """
        usage = self.vj_usage(locus, count=count)
        total = self.total(locus, count=count)
        n_pairs = len(usage)
        denom = total + n_pairs * pseudocount
        if denom == 0:
            return {}
        return {pair: (n + pseudocount) / denom for pair, n in usage.items()}

    def v_fraction(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
        pseudocount: float = 1.0,
    ) -> dict[str, float]:
        """Laplace-smoothed marginal V-gene fraction for *locus*."""
        usage = self.v_usage(locus, count=count)
        total = self.total(locus, count=count)
        n_genes = len(usage)
        denom = total + n_genes * pseudocount
        if denom == 0:
            return {}
        return {v: (n + pseudocount) / denom for v, n in usage.items()}

    def j_fraction(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
        pseudocount: float = 1.0,
    ) -> dict[str, float]:
        """Laplace-smoothed marginal J-gene fraction for *locus*."""
        usage = self.j_usage(locus, count=count)
        total = self.total(locus, count=count)
        n_genes = len(usage)
        denom = total + n_genes * pseudocount
        if denom == 0:
            return {}
        return {j: (n + pseudocount) / denom for j, n in usage.items()}
