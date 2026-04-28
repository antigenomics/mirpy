"""Gene usage tracking for immune repertoires.

:class:`GeneUsage` accumulates V-J gene combination counts from
:class:`~mir.common.repertoire.LocusRepertoire` or
:class:`~mir.common.repertoire.SampleRepertoire` objects and exposes joint and
marginal usage statistics together with Laplace-smoothed fractions.

Allele Handling
~~~~~~~~~~~~~~~
By default, gene allele suffixes are stripped during initialization
(e.g., ``TRBV1*01`` → ``TRBV1``) so that different allele naming conventions
are treated as the same gene. This behavior can be disabled by setting
``strip_alleles=False`` when constructing a ``GeneUsage`` object.

When resampling using :func:`mir.common.sampling.resample_to_gene_usage`,
clonotypes retain their original alleles while only stripped gene bases are
used for frequency comparison.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire

_VJPair = tuple[str, str]

_COUNT_MODE_ALIASES = {
    "clonotypes": "clonotypes",
    "clonotype": "clonotypes",
    "rearrangement": "clonotypes",
    "rearrangements": "clonotypes",
    "count_rearrangement": "clonotypes",
    "count_rearrangements": "clonotypes",
    "duplicates": "duplicates",
    "duplicate": "duplicates",
    "count_duplicates": "duplicates",
}


def _normalize_count_mode(count: str) -> str:
    """Normalize public count-mode aliases.

    Supported modes:
    - ``clonotypes`` / ``count_rearrangement`` (unweighted, default)
    - ``duplicates`` / ``count_duplicates`` (weighted by duplicate_count)
    """
    normalized = _COUNT_MODE_ALIASES.get(str(count).strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unknown count mode: {count!r}. "
            "Use 'clonotypes'/'count_rearrangement' or "
            "'duplicates'/'count_duplicates'."
        )
    return normalized


def _strip_allele(gene: str) -> str:
    """Strip allele suffix: ``"TRBV1*01"`` → ``"TRBV1"``."""
    return gene.split("*")[0] if gene else ""


class GeneUsage:
    """Joint and marginal V-J gene usage statistics.

    Stores per-locus clonotype counts and duplicate-count totals for every
    observed (V-gene, J-gene) pair.

    Parameters
    ----------
    strip_alleles : bool, optional
        When ``True`` (default), remove allele suffixes during initialization
        so that ``TRBV1*01`` and ``TRBV1`` are treated as the same gene.
        When ``False``, alleles are preserved as-is.

    Attributes
    ----------
    strip_alleles : bool
        Whether allele suffixes were stripped during initialization.

    Examples
    --------
    Build from a repertoire, automatically stripping alleles::

        gu = GeneUsage.from_repertoire(trb_repertoire)
        gu.vj_fraction("TRB")
        {('TRBV12-3', 'TRBJ1-2'): 0.42, ...}

    Build with alleles preserved::

        gu = GeneUsage.from_repertoire(trb_repertoire, strip_alleles=False)
        gu.vj_fraction("TRB")
        {('TRBV12-3*01', 'TRBJ1-2*01'): 0.42, ...}
    """

    def __init__(self, *, strip_alleles: bool = True) -> None:
        # locus → {(v_base, j_base): [n_clones, n_dc]}
        self._data: dict[str, dict[_VJPair, list[int]]] = {}
        # locus → [total_clones, total_dc]
        self._totals: dict[str, list[int]] = {}
        self.strip_alleles = strip_alleles

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_repertoire(
        cls,
        repertoire: "LocusRepertoire",
        *,
        locus: str = "",
        strip_alleles: bool = True,
    ) -> "GeneUsage":
        """Build from a :class:`~mir.common.repertoire.LocusRepertoire`.

        Parameters
        ----------
        repertoire
            Source locus repertoire.
        locus
            Override locus.  When empty the repertoire's own locus is used.
        strip_alleles
            Whether to strip allele suffixes (default ``True``).
        """
        obj = cls(strip_alleles=strip_alleles)
        obj._add_locus_repertoire(repertoire, locus=locus)
        return obj

    @classmethod
    def from_sample(
        cls,
        sample: "SampleRepertoire",
        *,
        strip_alleles: bool = True,
    ) -> "GeneUsage":
        """Build from a :class:`~mir.common.repertoire.SampleRepertoire`.

        Iterates over all loci in the sample.

        Parameters
        ----------
        sample
            Source sample repertoire.
        strip_alleles
            Whether to strip allele suffixes (default ``True``).
        """
        obj = cls(strip_alleles=strip_alleles)
        for loc, locus_rep in sample.loci.items():
            obj._add_locus_repertoire(locus_rep, locus=loc)
        return obj

    @classmethod
    def from_list(
        cls,
        repertoires,
        *,
        strip_alleles: bool = True,
    ) -> "GeneUsage":
        """Build by accumulating data from a list of repertoire objects.

        Each element may be a :class:`~mir.common.repertoire.LocusRepertoire`
        or a :class:`~mir.common.repertoire.SampleRepertoire`.

        Parameters
        ----------
        repertoires
            List of LocusRepertoire or SampleRepertoire objects.
        strip_alleles
            Whether to strip allele suffixes (default ``True``).
        """
        from mir.common.repertoire import SampleRepertoire

        obj = cls(strip_alleles=strip_alleles)
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
            v = self._normalize_gene(clone.v_gene or "")
            j = self._normalize_gene(clone.j_gene or "")
            dc = clone.duplicate_count or 0
            entry = locus_data.setdefault((v, j), [0, 0])
            entry[0] += 1
            entry[1] += dc
            locus_totals[0] += 1
            locus_totals[1] += dc

    def _normalize_gene(self, gene: str) -> str:
        """Apply gene normalization based on strip_alleles setting."""
        return _strip_allele(gene) if self.strip_alleles else gene

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
        normalized = _normalize_count_mode(count)
        totals = self._totals.get(locus, [0, 0])
        return totals[0] if normalized == "clonotypes" else totals[1]

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
        normalized = _normalize_count_mode(count)
        idx = 0 if normalized == "clonotypes" else 1
        return {pair: vals[idx] for pair, vals in self._data.get(locus, {}).items()}

    def v_usage(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
    ) -> dict[str, int]:
        """Marginal V-gene usage (sum over all J) for *locus*."""
        normalized = _normalize_count_mode(count)
        idx = 0 if normalized == "clonotypes" else 1
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
        normalized = _normalize_count_mode(count)
        idx = 0 if normalized == "clonotypes" else 1
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

    # ------------------------------------------------------------------
    # Cross-dataset comparison helpers
    # ------------------------------------------------------------------

    def usage_comparison(
        self,
        reference: "GeneUsage",
        locus: str,
        *,
        scope: str = "vj",
        count: str = "count_rearrangement",
        pseudocount: float = 1.0,
    ) -> dict[object, dict[str, float]]:
        """Compare smoothed usage frequencies against another GeneUsage.

        Frequencies are computed independently for ``self`` and ``reference``
        using Laplace smoothing with the same pseudocount:

        ``(n_key + pseudocount) / (total + n_observed_keys * pseudocount)``.

        Args:
            reference: Baseline gene usage to compare against (e.g. OLGA).
            locus: IMGT locus code.
            scope: ``"v"``, ``"j"``, or ``"vj"``.
            count: Count mode alias (default ``count_rearrangement``).
            pseudocount: Additive smoothing constant (must be >= 0).

        Returns:
            Mapping from key (gene or VJ tuple) to:
            ``{"p_self": ..., "p_reference": ..., "factor": ...}``.
        """
        if pseudocount < 0:
            raise ValueError("pseudocount must be non-negative")

        scope_norm = str(scope).strip().lower()
        if scope_norm == "v":
            self_usage = self.v_usage(locus, count=count)
            ref_usage = reference.v_usage(locus, count=count)
        elif scope_norm == "j":
            self_usage = self.j_usage(locus, count=count)
            ref_usage = reference.j_usage(locus, count=count)
        elif scope_norm == "vj":
            self_usage = self.vj_usage(locus, count=count)
            ref_usage = reference.vj_usage(locus, count=count)
        else:
            raise ValueError("scope must be one of: 'v', 'j', 'vj'")

        self_total = self.total(locus, count=count)
        ref_total = reference.total(locus, count=count)
        n_self = len(self_usage)
        n_ref = len(ref_usage)

        self_denom = self_total + n_self * pseudocount
        ref_denom = ref_total + n_ref * pseudocount

        all_keys = sorted(set(self_usage) | set(ref_usage))
        result: dict[object, dict[str, float]] = {}
        for key in all_keys:
            p_self = (
                (self_usage.get(key, 0) + pseudocount) / self_denom
                if self_denom > 0
                else 0.0
            )
            p_ref = (
                (ref_usage.get(key, 0) + pseudocount) / ref_denom
                if ref_denom > 0
                else 0.0
            )
            factor = (p_self / p_ref) if p_ref > 0 else float("inf")
            result[key] = {
                "p_self": float(p_self),
                "p_reference": float(p_ref),
                "factor": float(factor),
            }
        return result

    def correction_factors(
        self,
        reference: "GeneUsage",
        locus: str,
        *,
        scope: str = "vj",
        count: str = "count_rearrangement",
        pseudocount: float = 1.0,
    ) -> dict[object, float]:
        """Return correction factors ``P_self / P_reference`` by key."""
        comparison = self.usage_comparison(
            reference,
            locus,
            scope=scope,
            count=count,
            pseudocount=pseudocount,
        )
        return {k: v["factor"] for k, v in comparison.items()}
