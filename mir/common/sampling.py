"""Repertoire downsampling and resampling utilities.

This module provides functions to downsample immune repertoires by randomly
sampling clonotypes according to their abundance, and to resample them to
match target gene usage distributions.

Important Notes on Allele Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module preserves the original allele information from clonotypes while
using stripped gene bases for gene usage comparisons. For example:
- When comparing gene usage, ``TRBV1*01`` and ``TRBV1*02`` are treated as
  the same V-gene (``TRBV1``)
- Clonotypes in the output repertoire retain their original alleles
- Target gene usage dicts should use base gene names (without alleles)

This design ensures:
1. Downsampled/resampled repertoires preserve input allele information
2. Gene usage statistics are comparable across datasets with different
   allele naming conventions
3. No accidental data loss due to allele normalization
"""

from __future__ import annotations

import warnings
from copy import copy
from typing import Any, TypeAlias, cast

import numpy as np

from mir.basic.gene_usage import GeneUsage, _strip_allele
from mir.common.repertoire import LocusRepertoire, SampleRepertoire

GeneUsageMap: TypeAlias = dict[Any, int]


def downsample_locus(
    repertoire: LocusRepertoire,
    downsample_count: int,
    random_seed: int | None = None,
) -> LocusRepertoire:
    """Downsample a LocusRepertoire to a target duplicate count.

    Uses multinomial sampling based on clonotype frequencies. Each clonotype's
    new duplicate count is drawn from a multinomial distribution with
    probabilities proportional to its initial abundance.

    Parameters
    ----------
    repertoire:
        The repertoire to downsample.
    downsample_count:
        Target total duplicate count for downsampled repertoire. Must be > 0.
    random_seed:
        Numpy random seed for reproducibility. If None, uses current random state.

    Returns
    -------
    LocusRepertoire
        New repertoire with downsampled clonotypes. Clonotypes with zero
        new duplicate count are omitted.

    Warnings
    --------
    UserWarning
        If downsample_count >= repertoire.duplicate_count.

    Notes
    -----
    The total duplicate count of the returned repertoire is guaranteed to equal
    *downsample_count*.
    """
    if downsample_count <= 0:
        raise ValueError(f"downsample_count must be > 0, got {downsample_count}")

    total_duplicates = repertoire.duplicate_count

    if downsample_count >= total_duplicates:
        warnings.warn(
            f"downsample_count ({downsample_count}) >= total duplicate count ({total_duplicates}); "
            "no downsampling performed",
            UserWarning,
        )
        # Return a copy with the same clonotypes
        return LocusRepertoire(
            clonotypes=[copy(c) for c in repertoire.clonotypes],
            locus=repertoire.locus,
            repertoire_id=repertoire.repertoire_id,
            repertoire_metadata=dict(repertoire.repertoire_metadata),
        )

    # Compute frequencies
    frequencies = np.array(
        [c.duplicate_count / total_duplicates for c in repertoire.clonotypes],
        dtype=np.float64,
    )

    # Multinomial sampling
    if random_seed is not None:
        np.random.seed(random_seed)

    new_counts = np.random.multinomial(downsample_count, frequencies)

    # Build new repertoire, omitting zero-count clonotypes
    new_clonotypes = []
    for clonotype, new_count in zip(repertoire.clonotypes, new_counts):
        if new_count > 0:
            new_clonotype = copy(clonotype)
            new_clonotype.duplicate_count = int(new_count)
            new_clonotypes.append(new_clonotype)

    return LocusRepertoire(
        clonotypes=new_clonotypes,
        locus=repertoire.locus,
        repertoire_id=repertoire.repertoire_id,
        repertoire_metadata=dict(repertoire.repertoire_metadata),
    )


def downsample(
    repertoire: LocusRepertoire | SampleRepertoire,
    downsample_count: int,
    random_seed: int | None = None,
) -> LocusRepertoire | SampleRepertoire:
    """Downsample a repertoire to a target duplicate count.

    For :class:`LocusRepertoire`, applies downsampling directly.
    For :class:`SampleRepertoire`, applies downsampling to each locus independently.

    Parameters
    ----------
    repertoire:
        The repertoire to downsample.
    downsample_count:
        Target total duplicate count per locus.
    random_seed:
        Numpy random seed for reproducibility.

    Returns
    -------
    LocusRepertoire | SampleRepertoire
        Downsampled repertoire of the same type.
    """
    if isinstance(repertoire, LocusRepertoire):
        return downsample_locus(repertoire, downsample_count, random_seed=random_seed)

    # For SampleRepertoire, downsample each locus independently
    downsampled_loci = {
        locus: downsample_locus(lr, downsample_count, random_seed=random_seed)
        for locus, lr in repertoire.loci.items()
    }

    return SampleRepertoire(
        loci=downsampled_loci,
        sample_id=repertoire.sample_id,
        sample_metadata=dict(repertoire.sample_metadata),
    )


# ------------------------------------------------------------------
# Resample to gene usage
# ------------------------------------------------------------------


def _resample_to_gene_usage_locus(
    repertoire: LocusRepertoire,
    target_gene_usage: GeneUsageMap,
    original_gene_usage: GeneUsageMap | None = None,
    *,
    scope: str = "v",
    weighted: bool = True,
    random_seed: int | None = None,
) -> LocusRepertoire:
    """Resample a LocusRepertoire to match target gene usage.

    Uses correction factors based on the ratio of target to original gene usage,
    then applies multinomial sampling.

    Parameters
    ----------
    repertoire:
        The repertoire to resample.
    target_gene_usage:
        Dict mapping gene (or VJ pair) to desired count.
    original_gene_usage:
        Dict mapping gene (or VJ pair) to original count. If None, computed
        from the repertoire using weighted/scope settings.
    scope:
        ``"v"`` for V-gene usage, ``"j"`` for J-gene usage, or ``"vj"`` for V-J usage.
    weighted:
        If True, compute gene usage weighted by duplicate_count.
        If False, count by clonotypes.
    random_seed:
        Numpy random seed for reproducibility.

    Returns
    -------
    LocusRepertoire
        Resampled repertoire with gene usage closer to target.
    """
    if scope not in ("v", "j", "vj"):
        raise ValueError(f"scope must be 'v', 'j', or 'vj', got {scope}")

    # Compute original gene usage if not provided
    if original_gene_usage is None:
        gu = GeneUsage.from_repertoire(repertoire)
        count_mode = "duplicates" if weighted else "clonotypes"
        if scope == "v":
            original_gene_usage = cast(GeneUsageMap, gu.v_usage(repertoire.locus, count=count_mode))
        elif scope == "j":
            original_gene_usage = cast(GeneUsageMap, gu.j_usage(repertoire.locus, count=count_mode))
        else:  # vj
            original_gene_usage = cast(GeneUsageMap, gu.vj_usage(repertoire.locus, count=count_mode))

    # Narrow Optional for type checkers.
    assert original_gene_usage is not None

    # Build resampling weights: p_i * (target_usage[gene_i] / original_usage[gene_i])
    weights = np.ones(len(repertoire.clonotypes), dtype=np.float64)

    for idx, clonotype in enumerate(repertoire.clonotypes):
        # Get gene for this clonotype (strip alleles to base gene)
        if scope == "v":
            gene = _strip_allele(clonotype.v_gene or "")
        elif scope == "j":
            gene = _strip_allele(clonotype.j_gene or "")
        else:  # vj
            v_base = _strip_allele(clonotype.v_gene or "")
            j_base = _strip_allele(clonotype.j_gene or "")
            gene = (v_base, j_base)

        # Get original usage for this gene
        orig_usage = original_gene_usage.get(gene, 0)
        if orig_usage == 0:
            # Gene not in original, skip correction
            weights[idx] = 0.0
            continue

        # Get target usage for this gene
        target_usage = target_gene_usage.get(gene, 0)

        # Correction factor
        factor = target_usage / orig_usage if orig_usage > 0 else 0.0

        # Weight by clonotype abundance if weighted mode
        if weighted:
            weights[idx] = factor * clonotype.duplicate_count
        else:
            weights[idx] = factor

    # Normalize to probabilities
    total_weight = np.sum(weights)
    if total_weight <= 0:
        raise ValueError("Total resampling weight is <= 0; no valid genes in repertoire")

    probabilities = weights / total_weight

    # Total reads to resample
    total_duplicates = repertoire.duplicate_count

    # Multinomial sampling
    if random_seed is not None:
        np.random.seed(random_seed)

    new_counts = np.random.multinomial(total_duplicates, probabilities)

    # Build new repertoire, omitting zero-count clonotypes
    new_clonotypes = []
    for clonotype, new_count in zip(repertoire.clonotypes, new_counts):
        if new_count > 0:
            new_clonotype = copy(clonotype)
            new_clonotype.duplicate_count = int(new_count)
            new_clonotypes.append(new_clonotype)

    return LocusRepertoire(
        clonotypes=new_clonotypes,
        locus=repertoire.locus,
        repertoire_id=repertoire.repertoire_id,
        repertoire_metadata=dict(repertoire.repertoire_metadata),
    )


def resample_to_gene_usage(
    repertoire: LocusRepertoire | SampleRepertoire,
    target_gene_usage: GeneUsageMap,
    original_gene_usage: GeneUsageMap | None = None,
    *,
    scope: str = "v",
    weighted: bool = True,
    random_seed: int | None = None,
) -> LocusRepertoire | SampleRepertoire:
    """Resample a repertoire to match target gene usage.

    For :class:`LocusRepertoire`, applies resampling directly.
    For :class:`SampleRepertoire`, applies resampling to each locus independently.

    Uses correction factors: for each clonotype, its resampling weight is
    multiplied by (target_usage / original_usage) for its gene.

    Parameters
    ----------
    repertoire:
        The repertoire to resample.
    target_gene_usage:
        Dict mapping gene (or VJ pair) to desired count.
    original_gene_usage:
        Dict mapping gene (or VJ pair) to original count. If None, computed
        from the repertoire(s).
    scope:
        ``"v"`` for V-gene usage, ``"j"`` for J-gene usage, or ``"vj"`` for V-J usage.
    weighted:
        If True, compute gene usage weighted by duplicate_count.
        If False, count by clonotypes.
    random_seed:
        Numpy random seed for reproducibility.

    Returns
    -------
    LocusRepertoire | SampleRepertoire
        Resampled repertoire of the same type.
    """
    if isinstance(repertoire, LocusRepertoire):
        return _resample_to_gene_usage_locus(
            repertoire,
            target_gene_usage,
            original_gene_usage,
            scope=scope,
            weighted=weighted,
            random_seed=random_seed,
        )

    # For SampleRepertoire, resample each locus independently
    resampled_loci = {
        locus: _resample_to_gene_usage_locus(
            lr,
            target_gene_usage,
            original_gene_usage,
            scope=scope,
            weighted=weighted,
            random_seed=random_seed,
        )
        for locus, lr in repertoire.loci.items()
    }

    return SampleRepertoire(
        loci=resampled_loci,
        sample_id=repertoire.sample_id,
        sample_metadata=dict(repertoire.sample_metadata),
    )


# ------------------------------------------------------------------
# Select top clonotypes
# ------------------------------------------------------------------


def select_top(
    repertoire: LocusRepertoire | SampleRepertoire,
    top_n: int,
) -> LocusRepertoire | SampleRepertoire:
    """Select the top *top_n* most abundant clonotypes by duplicate count.

    Clonotypes are ranked by duplicate_count in descending order. If *top_n*
    exceeds the number of clonotypes, a warning is issued and all clonotypes
    are returned.

    Parameters
    ----------
    repertoire:
        The repertoire to subsample.
    top_n:
        Number of top clonotypes to keep. Must be > 0.

    Returns
    -------
    LocusRepertoire | SampleRepertoire
        Repertoire containing only the top *top_n* clonotypes, sorted by
        duplicate_count descending.

    Warnings
    --------
    UserWarning
        If *top_n* exceeds the number of clonotypes in the repertoire.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be > 0, got {top_n}")

    if isinstance(repertoire, LocusRepertoire):
        if top_n >= len(repertoire.clonotypes):
            warnings.warn(
                f"top_n ({top_n}) >= number of clonotypes ({len(repertoire.clonotypes)}); "
                "returning all clonotypes",
                UserWarning,
            )
            selected = list(repertoire.clonotypes)
        else:
            # Sort by duplicate count descending and take top N
            sorted_clonotypes = sorted(
                repertoire.clonotypes,
                key=lambda c: c.duplicate_count,
                reverse=True,
            )
            selected = sorted_clonotypes[:top_n]

        return LocusRepertoire(
            clonotypes=selected,
            locus=repertoire.locus,
            repertoire_id=repertoire.repertoire_id,
            repertoire_metadata=dict(repertoire.repertoire_metadata),
        )

    # For SampleRepertoire, select top from each locus independently
    top_loci: dict[str, LocusRepertoire] = {
        locus: cast(LocusRepertoire, select_top(lr, top_n))
        for locus, lr in repertoire.loci.items()
    }

    return SampleRepertoire(
        loci=top_loci,
        sample_id=repertoire.sample_id,
        sample_metadata=dict(repertoire.sample_metadata),
    )
