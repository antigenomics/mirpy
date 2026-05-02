"""Neighborhood enrichment statistics for TCRnet and ALICE algorithms.

Computes neighborhood statistics for clonotypes: for each clonotype, counts
the number of neighbors within a specified edit distance and optional V/J
gene matching constraints.
"""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from mir.graph.distance_utils import PairRecord, compute_distance, should_compare_pair

if TYPE_CHECKING:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire


def compute_neighborhood_stats(
    repertoire: LocusRepertoire | SampleRepertoire,
    metric: str = "hamming",
    threshold: int = 1,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
) -> dict[str, dict]:
    """Compute neighborhood statistics for clonotypes in a repertoire.

    For each clonotype, counts the number of neighbors (including itself) with
    edit distance ≤ *threshold* in the junction_aa sequence. Optionally filters
    neighbors by V and/or J gene matching.

    Parameters
    ----------
    repertoire
        Input repertoire (LocusRepertoire or SampleRepertoire).
    metric
        Distance metric: ``"hamming"`` or ``"levenshtein"``.
    threshold
        Maximum edit distance for a clonotype to be considered a neighbor.
    match_v_gene
        If True, only count neighbors with matching v_gene.
    match_j_gene
        If True, only count neighbors with matching j_gene.

    Returns
    -------
    dict[str, dict]
        Mapping from clonotype sequence_id to neighbor statistics dictionary with keys:
        - ``"neighbor_count"``: Number of neighbors (including self)
        - ``"potential_neighbors"``: Total number of clonotypes considered as potential neighbors
          (based on gene matching constraints)

    Notes
    -----
    For SampleRepertoire, computes statistics per locus independently.
    """
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")

    from mir.common.repertoire import LocusRepertoire, SampleRepertoire

    if isinstance(repertoire, SampleRepertoire):
        results: dict[str, dict] = {}
        for locus, locus_rep in repertoire.loci.items():
            locus_results = compute_neighborhood_stats(
                locus_rep,
                metric=metric,
                threshold=threshold,
                match_v_gene=match_v_gene,
                match_j_gene=match_j_gene,
            )
            results.update(locus_results)
        return results

    # Handle LocusRepertoire
    clonotypes = repertoire.clonotypes
    n = len(clonotypes)

    if n == 0:
        return {}

    # Pre-extract fields
    seqs = [c.junction_aa for c in clonotypes]
    v_genes = [c.v_gene for c in clonotypes]
    j_genes = [c.j_gene for c in clonotypes]
    seq_ids = [c.sequence_id for c in clonotypes]

    results = {}

    for i in range(n):
        # Count potential neighbors based on gene constraints
        if match_v_gene and match_j_gene:
            # Both must match
            potential_neighbors = sum(
                1 for j in range(n)
                if v_genes[i] == v_genes[j] and j_genes[i] == j_genes[j]
            )
        elif match_v_gene:
            # Only V must match
            potential_neighbors = sum(1 for j in range(n) if v_genes[i] == v_genes[j])
        elif match_j_gene:
            # Only J must match
            potential_neighbors = sum(1 for j in range(n) if j_genes[i] == j_genes[j])
        else:
            # No gene constraints
            potential_neighbors = n

        # Count actual neighbors within threshold
        neighbor_count = 0
        for j in range(n):
            # Check gene matching
            if not should_compare_pair(
                PairRecord(i, j, seqs[i], seqs[j], v_genes[i], v_genes[j], j_genes[i], j_genes[j]),
                match_v_gene=match_v_gene,
                match_j_gene=match_j_gene,
            ):
                continue

            # Compute distance
            dist = compute_distance(seqs[i], seqs[j], metric)
            if dist <= threshold:
                neighbor_count += 1

        results[seq_ids[i]] = {
            "neighbor_count": neighbor_count,
            "potential_neighbors": potential_neighbors,
        }

    return results


def add_neighborhood_metadata(
    repertoire: LocusRepertoire | SampleRepertoire,
    metric: str = "hamming",
    threshold: int = 1,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
    metadata_prefix: str = "neighborhood",
) -> None:
    """Add neighborhood statistics as metadata to clonotypes in-place.

    Parameters
    ----------
    repertoire
        Input repertoire to modify.
    metric
        Distance metric: ``"hamming"`` or ``"levenshtein"``.
    threshold
        Maximum edit distance for a clonotype to be considered a neighbor.
    match_v_gene
        If True, only count neighbors with matching v_gene.
    match_j_gene
        If True, only count neighbors with matching j_gene.
    metadata_prefix
        Prefix for metadata keys (e.g., ``"neighborhood_count"`` and
        ``"neighborhood_potential"``).
    """
    from mir.common.repertoire import SampleRepertoire

    if isinstance(repertoire, SampleRepertoire):
        for locus_rep in repertoire.loci.values():
            add_neighborhood_metadata(
                locus_rep,
                metric=metric,
                threshold=threshold,
                match_v_gene=match_v_gene,
                match_j_gene=match_j_gene,
                metadata_prefix=metadata_prefix,
            )
        return

    stats = compute_neighborhood_stats(
        repertoire,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
    )

    for clonotype in repertoire.clonotypes:
        if clonotype.sequence_id in stats:
            stat = stats[clonotype.sequence_id]
            clonotype.clone_metadata[f"{metadata_prefix}_count"] = stat["neighbor_count"]
            clonotype.clone_metadata[f"{metadata_prefix}_potential"] = stat["potential_neighbors"]
