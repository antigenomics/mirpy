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
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire
    from mir.common.repertoire_dataset import RepertoireDataset

_VJPair = tuple[str, str]
GeneScope = Literal["v", "j", "vj"]


def _normalize_count_mode(count: str) -> str:
    """Normalize public count-mode aliases.

    Supported modes:
    - ``clonotypes`` / ``count_rearrangement`` (unweighted, default)
    - ``duplicates`` / ``count_duplicates`` (weighted by duplicate_count)
    """
    mode = str(count).strip().lower()
    if mode in {
        "clonotypes",
        "clonotype",
        "rearrangement",
        "rearrangements",
        "count_rearrangement",
        "count_rearrangements",
    }:
        return "clonotypes"
    if mode in {"duplicates", "duplicate", "count_duplicates"}:
        return "duplicates"
    raise ValueError(
        f"Unknown count mode: {count!r}. "
        "Use 'clonotypes'/'count_rearrangement' or "
        "'duplicates'/'count_duplicates'."
    )


def _count_index(count: str) -> int:
    """Return storage index for normalized count mode."""
    return 0 if _normalize_count_mode(count) == "clonotypes" else 1


def _laplace_fraction(usage: dict, total: int, pseudocount: float) -> dict:
    """Compute Laplace-smoothed fractions for an observed usage map."""
    n_keys = len(usage)
    denom = total + n_keys * pseudocount
    if denom == 0:
        return {}
    return {k: (n + pseudocount) / denom for k, n in usage.items()}


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

        table = getattr(repertoire, "_polars_table", None)
        if table is not None:
            try:
                import polars as pl

                if table.height == 0:
                    return
                grouped = (
                    table
                    .select([
                        pl.col("v_gene").cast(pl.Utf8).fill_null(""),
                        pl.col("j_gene").cast(pl.Utf8).fill_null(""),
                        pl.col("duplicate_count").cast(pl.Int64).fill_null(0),
                    ])
                    .group_by(["v_gene", "j_gene"])
                    .agg([
                        pl.len().alias("n_clones"),
                        pl.col("duplicate_count").sum().alias("n_dc"),
                    ])
                )

                for row in grouped.iter_rows(named=True):
                    v = self._normalize_gene(str(row.get("v_gene") or ""))
                    j = self._normalize_gene(str(row.get("j_gene") or ""))
                    n_clones = int(row.get("n_clones") or 0)
                    n_dc = int(row.get("n_dc") or 0)
                    entry = locus_data.setdefault((v, j), [0, 0])
                    entry[0] += n_clones
                    entry[1] += n_dc
                    locus_totals[0] += n_clones
                    locus_totals[1] += n_dc
                return
            except Exception:
                # Fall back to the generic Python path if polars operations fail.
                pass

        # Fast path for lazily loaded repertoires: consume raw columns directly
        # and avoid constructing per-clonotype Python objects.
        pending = getattr(repertoire, "_pending_cols", None)
        if pending is not None:
            v_genes = pending.get("v_genes", [])
            j_genes = pending.get("j_genes", [])
            dups = pending.get("dup_counts", [])
            for v_gene, j_gene, dc in zip(v_genes, j_genes, dups):
                v = self._normalize_gene(v_gene or "")
                j = self._normalize_gene(j_gene or "")
                dc_i = int(dc or 0)
                entry = locus_data.setdefault((v, j), [0, 0])
                entry[0] += 1
                entry[1] += dc_i
                locus_totals[0] += 1
                locus_totals[1] += dc_i
            return

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
        idx = _count_index(count)
        return {pair: vals[idx] for pair, vals in self._data.get(locus, {}).items()}

    def _marginal_usage(self, locus: str, *, count: str, axis: int) -> dict[str, int]:
        """Generic V/J marginal usage helper.

        Parameters
        ----------
        axis
            0 for V-gene aggregation, 1 for J-gene aggregation.
        """
        idx = _count_index(count)
        result: dict[str, int] = defaultdict(int)
        for pair, vals in self._data.get(locus, {}).items():
            result[pair[axis]] += vals[idx]
        return dict(result)

    def v_usage(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
    ) -> dict[str, int]:
        """Marginal V-gene usage (sum over all J) for *locus*."""
        return self._marginal_usage(locus, count=count, axis=0)

    def j_usage(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
    ) -> dict[str, int]:
        """Marginal J-gene usage (sum over all V) for *locus*."""
        return self._marginal_usage(locus, count=count, axis=1)

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
        return _laplace_fraction(usage, self.total(locus, count=count), pseudocount)

    def v_fraction(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
        pseudocount: float = 1.0,
    ) -> dict[str, float]:
        """Laplace-smoothed marginal V-gene fraction for *locus*."""
        usage = self.v_usage(locus, count=count)
        return _laplace_fraction(usage, self.total(locus, count=count), pseudocount)

    def j_fraction(
        self,
        locus: str,
        *,
        count: str = "clonotypes",
        pseudocount: float = 1.0,
    ) -> dict[str, float]:
        """Laplace-smoothed marginal J-gene fraction for *locus*."""
        usage = self.j_usage(locus, count=count)
        return _laplace_fraction(usage, self.total(locus, count=count), pseudocount)

    def _usage_by_scope(self, locus: str, *, scope: str, count: str) -> dict:
        """Dispatch helper for v/j/vj usage maps."""
        scope_norm = str(scope).strip().lower()
        if scope_norm == "v":
            return self.v_usage(locus, count=count)
        if scope_norm == "j":
            return self.j_usage(locus, count=count)
        if scope_norm == "vj":
            return self.vj_usage(locus, count=count)
        raise ValueError("scope must be one of: 'v', 'j', 'vj'")

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

        self_usage = self._usage_by_scope(locus, scope=scope, count=count)
        ref_usage = reference._usage_by_scope(locus, scope=scope, count=count)

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


# ------------------------------------------------------------------
# Batch correction utilities
# ------------------------------------------------------------------


def zscore_to_sigmoid(z: "np.ndarray | float") -> "np.ndarray | float":
    """Map a (batch-corrected) z-score to a bounded sigmoid value in ``(0, 1)``.

    ``sigmoid(z) = 1 / (1 + exp(-z))``

    This is the canonical transform to turn per-gene z-scores from
    :func:`compute_batch_corrected_gene_usage` into bounded, comparable
    corrected probabilities that can be directly used in PCA/UMAP embeddings.

    Parameters
    ----------
    z
        Scalar or array of z-scores.

    Returns
    -------
    np.ndarray or float with the same shape as *z*, values in ``(0, 1)``.
    """
    arr = np.asarray(z, dtype=float)
    result = 1.0 / (1.0 + np.exp(-arr))
    return float(result) if arr.ndim == 0 else result


def _winsorized_mean_std(values: pd.Series, *, lower_q: float = 0.025, upper_q: float = 0.975) -> tuple[float, float]:
    """Return mean and SD after clipping to the winsorized interval."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    lo = float(np.quantile(arr, lower_q))
    hi = float(np.quantile(arr, upper_q))
    clipped = np.clip(arr, lo, hi)
    mean = float(np.mean(clipped))
    std = float(np.std(clipped, ddof=1)) if clipped.size > 1 else 0.0
    if not np.isfinite(std):
        std = 0.0
    return mean, std


def compute_batch_corrected_gene_usage(
    dataset: "RepertoireDataset",
    *,
    batch_field: str = "batch_id",
    scope: GeneScope = "vj",
    weighted: bool = True,
    pseudocount: float = 1.0,
    z_cap: float = 6.0,
) -> pd.DataFrame:
    """Compute batch-corrected gene usage for all samples/loci/genes.

    Uses a pseudocount on raw counts prior to normalization:

    ``p = (count + pseudocount) / (total + pseudocount * n_genes)``

    Then computes ``log_p``, batch-wise winsorized (95%) ``mu`` and ``sigma``
    over ``(locus, gene, batch_id)``, capped z-scores, and final corrected
    probabilities:

    ``pfinal = 2 * pavg / (1 - exp(-z))``

    Empty sample loci and loci absent in a sample are skipped without error.
    """
    if pseudocount < 0:
        raise ValueError("pseudocount must be >= 0")
    if z_cap <= 0:
        raise ValueError("z_cap must be > 0")

    count_mode = "duplicates" if weighted else "clonotypes"

    sample_usage: dict[tuple[str, str], dict[object, int]] = {}
    genes_by_locus: dict[str, set[object]] = defaultdict(set)

    for sample_id, sample in dataset.samples.items():
        gu = GeneUsage.from_sample(sample)
        for locus, locus_rep in sample.loci.items():
            if locus_rep is None or getattr(locus_rep, "clonotype_count", 0) == 0:
                continue
            usage = gu._usage_by_scope(locus, scope=scope, count=count_mode)
            sample_usage[(sample_id, locus)] = usage
            genes_by_locus[locus].update(usage.keys())

    columns = [
        "sample_id", "batch_id", "locus", "gene", "count", "total", "n_genes",
        "p", "log_p", "mu", "sigma", "z", "pavg", "pfinal",
    ]
    if not sample_usage:
        return pd.DataFrame(columns=columns)

    pooled_counts: dict[tuple[str, object], float] = defaultdict(float)
    pooled_totals: dict[str, float] = defaultdict(float)
    for (_, locus), usage in sample_usage.items():
        for gene, val in usage.items():
            pooled_counts[(locus, gene)] += float(val)
            pooled_totals[locus] += float(val)

    pavg: dict[tuple[str, object], float] = {}
    for locus, genes in genes_by_locus.items():
        denom = float(pooled_totals.get(locus, 0.0))
        if denom <= 0:
            for gene in genes:
                pavg[(locus, gene)] = 0.0
            continue
        for gene in genes:
            pavg[(locus, gene)] = pooled_counts[(locus, gene)] / denom

    rows: list[dict[str, object]] = []
    for sample_id, sample in dataset.samples.items():
        metadata = dataset.metadata.get(sample_id, {})
        if batch_field not in metadata:
            raise ValueError(f"metadata for sample_id={sample_id!r} missing required field {batch_field!r}")
        batch_id = metadata[batch_field]

        for locus, locus_rep in sample.loci.items():
            if locus not in genes_by_locus:
                continue
            if locus_rep is None or getattr(locus_rep, "clonotype_count", 0) == 0:
                continue

            usage = sample_usage.get((sample_id, locus), {})
            n_genes = len(genes_by_locus[locus])
            if weighted:
                total = float(getattr(locus_rep, "duplicate_count", 0))
            else:
                total = float(getattr(locus_rep, "clonotype_count", 0))
            denom = total + pseudocount * n_genes

            for gene in sorted(genes_by_locus[locus]):
                count = float(usage.get(gene, 0.0))
                p = ((count + pseudocount) / denom) if denom > 0 else 0.0
                log_p = float(np.log(p)) if p > 0 else float("-inf")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "batch_id": batch_id,
                        "locus": locus,
                        "gene": gene,
                        "count": count,
                        "total": total,
                        "n_genes": n_genes,
                        "p": p,
                        "log_p": log_p,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns)

    stats = (
        df.groupby(["locus", "gene", "batch_id"], dropna=False)["log_p"]
        .apply(_winsorized_mean_std)
        .reset_index(name="mu_sigma")
    )
    stats[["mu", "sigma"]] = pd.DataFrame(stats["mu_sigma"].tolist(), index=stats.index)
    stats = stats.drop(columns=["mu_sigma"])

    df = df.merge(stats, on=["locus", "gene", "batch_id"], how="left")
    df["sigma"] = pd.to_numeric(df["sigma"], errors="coerce").fillna(0.0)
    df["mu"] = pd.to_numeric(df["mu"], errors="coerce").fillna(0.0)

    raw_z = np.where(df["sigma"].to_numpy(dtype=float) > 0,
                     (df["log_p"].to_numpy(dtype=float) - df["mu"].to_numpy(dtype=float))
                     / df["sigma"].to_numpy(dtype=float),
                     0.0)
    df["z"] = np.clip(raw_z, -z_cap, z_cap)

    df["pavg"] = df.apply(lambda r: pavg[(r["locus"], r["gene"])], axis=1)

    denom = 1.0 - np.exp(-df["z"].to_numpy(dtype=float))
    tiny = np.abs(denom) < 1e-12
    denom[tiny] = np.where(denom[tiny] < 0, -1e-12, 1e-12)
    df["pfinal"] = 2.0 * df["pavg"].to_numpy(dtype=float) / denom

    return df[columns].sort_values(["sample_id", "locus", "gene"]).reset_index(drop=True)
