"""Neighborhood enrichment statistics for TCRnet and ALICE algorithms.

Computes neighborhood statistics for clonotypes. For each clonotype in a query
repertoire, counts the number of neighbors within a specified edit distance and
optional V/J gene matching constraints, either against itself or against an
explicit background repertoire.
"""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from mir.graph.distance_utils import PairRecord, compute_distance, should_compare_pair

if TYPE_CHECKING:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire


def _validate_metric(metric: str) -> None:
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")


def _is_same_background(
    repertoire: "LocusRepertoire | SampleRepertoire",
    background: "LocusRepertoire | SampleRepertoire | None",
) -> bool:
    return background is None or background is repertoire


def _iter_loci(
    repertoire: "LocusRepertoire | SampleRepertoire",
) -> dict[str, "LocusRepertoire"]:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire

    if isinstance(repertoire, SampleRepertoire):
        return dict(repertoire.loci)
    if isinstance(repertoire, LocusRepertoire):
        return {repertoire.locus: repertoire}
    raise TypeError("repertoire must be LocusRepertoire or SampleRepertoire")


def _background_locus_map(
    query_loci: dict[str, "LocusRepertoire"],
    background: "LocusRepertoire | SampleRepertoire | None",
) -> dict[str, "LocusRepertoire"]:
    if background is None:
        return query_loci

    bg_loci = _iter_loci(background)
    out: dict[str, "LocusRepertoire"] = {}
    for locus, qrep in query_loci.items():
        out[locus] = bg_loci.get(locus, qrep.__class__(clonotypes=[], locus=locus))
    return out


def _compute_locus_stats(
    query_locus: "LocusRepertoire",
    background_locus: "LocusRepertoire",
    *,
    metric: str,
    threshold: int,
    match_v_gene: bool,
    match_j_gene: bool,
    add_self_pseudocount: bool,
) -> dict[str, dict[str, int]]:
    query_clonotypes = query_locus.clonotypes
    background_clonotypes = background_locus.clonotypes

    if not query_clonotypes:
        return {}

    q_seqs = [c.junction_aa for c in query_clonotypes]
    q_v_genes = [c.v_gene for c in query_clonotypes]
    q_j_genes = [c.j_gene for c in query_clonotypes]
    q_seq_ids = [c.sequence_id for c in query_clonotypes]

    b_seqs = [c.junction_aa for c in background_clonotypes]
    b_v_genes = [c.v_gene for c in background_clonotypes]
    b_j_genes = [c.j_gene for c in background_clonotypes]

    n_query = len(query_clonotypes)
    n_background = len(background_clonotypes)

    results: dict[str, dict[str, int]] = {}
    for i in range(n_query):
        potential_neighbors = 0
        neighbor_count = 0

        for j in range(n_background):
            rec = PairRecord(
                i,
                j,
                q_seqs[i],
                b_seqs[j],
                q_v_genes[i],
                b_v_genes[j],
                q_j_genes[i],
                b_j_genes[j],
            )
            if not should_compare_pair(
                rec,
                match_v_gene=match_v_gene,
                match_j_gene=match_j_gene,
            ):
                continue

            potential_neighbors += 1
            if compute_distance(q_seqs[i], b_seqs[j], metric) <= threshold:
                neighbor_count += 1

        if add_self_pseudocount:
            # Background-mode smoothing: count query clonotype itself as one extra
            # background member to avoid zero-neighbor/zero-potential artifacts.
            potential_neighbors += 1
            neighbor_count += 1

        results[q_seq_ids[i]] = {
            "neighbor_count": int(neighbor_count),
            "potential_neighbors": int(potential_neighbors),
        }

    return results


def _set_clonotype_stats_metadata(
    repertoire: "LocusRepertoire | SampleRepertoire",
    stats_by_locus: dict[str, dict[str, dict[str, int]]],
    *,
    count_key: str,
    potential_key: str,
) -> None:
    for locus, locus_rep in _iter_loci(repertoire).items():
        locus_stats = stats_by_locus.get(locus, {})
        for clonotype in locus_rep.clonotypes:
            stat = locus_stats.get(clonotype.sequence_id)
            if stat is None:
                continue
            clonotype.clone_metadata[count_key] = stat["neighbor_count"]
            clonotype.clone_metadata[potential_key] = stat["potential_neighbors"]


def _compute_stats_by_locus(
    repertoire: "LocusRepertoire | SampleRepertoire",
    *,
    background: "LocusRepertoire | SampleRepertoire | None",
    metric: str,
    threshold: int,
    match_v_gene: bool,
    match_j_gene: bool,
) -> dict[str, dict[str, dict[str, int]]]:
    query_loci = _iter_loci(repertoire)
    bg_loci = _background_locus_map(query_loci, background)
    add_self_pseudocount = not _is_same_background(repertoire, background)

    return {
        locus: _compute_locus_stats(
            qrep,
            bg_loci[locus],
            metric=metric,
            threshold=threshold,
            match_v_gene=match_v_gene,
            match_j_gene=match_j_gene,
            add_self_pseudocount=add_self_pseudocount,
        )
        for locus, qrep in query_loci.items()
    }


def compute_neighborhood_stats(
    repertoire: LocusRepertoire | SampleRepertoire,
    background: LocusRepertoire | SampleRepertoire | None = None,
    metric: str = "hamming",
    threshold: int = 1,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
) -> dict[str, dict]:
    """Compute neighborhood statistics for clonotypes in a repertoire.

    For each query clonotype, counts neighbors with edit distance ≤ *threshold*
    in ``junction_aa`` against either the query repertoire itself (default) or
    a provided background repertoire. Optionally filters neighbors by V and/or
    J gene matching.

    Parameters
    ----------
    repertoire
        Input repertoire (LocusRepertoire or SampleRepertoire).
    background
        Optional background repertoire. If omitted (or identical object to
        ``repertoire``), neighborhood statistics are computed in self-mode.
        If a different background is provided, counts are computed against the
        matching locus in background and a ``+1`` pseudocount is added to both
        ``neighbor_count`` and ``potential_neighbors``.
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
    _validate_metric(metric)

    stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
    )

    merged: dict[str, dict[str, int]] = {}
    for locus_stats in stats_by_locus.values():
        merged.update(locus_stats)
    return merged


def add_neighborhood_metadata(
    repertoire: LocusRepertoire | SampleRepertoire,
    background: LocusRepertoire | SampleRepertoire | None = None,
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
        Query repertoire to modify.
    background
        Optional background repertoire. If provided and distinct from
        ``repertoire``, neighborhood counts are computed against background
        with ``+1`` pseudocount smoothing.
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
    stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
    )

    _set_clonotype_stats_metadata(
        repertoire,
        stats_by_locus,
        count_key=f"{metadata_prefix}_count",
        potential_key=f"{metadata_prefix}_potential",
    )


def add_neighborhood_enrichment_metadata(
    repertoire: LocusRepertoire | SampleRepertoire,
    background: LocusRepertoire | SampleRepertoire,
    metric: str = "hamming",
    threshold: int = 1,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
    metadata_prefix: str = "neighborhood",
) -> None:
    """Add parent/background neighborhood stats and enrichment metadata.

    This computes two sets of neighborhood statistics for every clonotype in
    ``repertoire``:

    - parent/self statistics against ``repertoire`` itself
    - background statistics against explicit ``background`` repertoire

    Background-mode statistics include a ``+1`` pseudocount in both neighbor and
    potential counts when background is a distinct object.

    Parameters
    ----------
    repertoire
        Query repertoire to modify in-place.
    background
        Background repertoire used for enrichment baseline.
    metric
        Distance metric: ``"hamming"`` or ``"levenshtein"``.
    threshold
        Maximum edit distance for a clonotype to be considered a neighbor.
    match_v_gene
        If True, only count neighbors with matching ``v_gene``.
    match_j_gene
        If True, only count neighbors with matching ``j_gene``.
    metadata_prefix
        Prefix for metadata keys.
    """
    _validate_metric(metric)

    parent_stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=None,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
    )
    background_stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
    )

    for locus, locus_rep in _iter_loci(repertoire).items():
        parent_stats = parent_stats_by_locus.get(locus, {})
        background_stats = background_stats_by_locus.get(locus, {})
        for clonotype in locus_rep.clonotypes:
            pid = clonotype.sequence_id
            p_stat = parent_stats.get(pid)
            b_stat = background_stats.get(pid)
            if p_stat is None or b_stat is None:
                continue

            p_count = int(p_stat["neighbor_count"])
            p_potential = int(p_stat["potential_neighbors"])
            b_count = int(b_stat["neighbor_count"])
            b_potential = int(b_stat["potential_neighbors"])

            p_density = (p_count / p_potential) if p_potential else 0.0
            b_density = (b_count / b_potential) if b_potential else 0.0
            enrichment = (p_density / b_density) if b_density > 0 else 0.0

            clonotype.clone_metadata[f"{metadata_prefix}_parent_count"] = p_count
            clonotype.clone_metadata[f"{metadata_prefix}_parent_potential"] = p_potential
            clonotype.clone_metadata[f"{metadata_prefix}_background_count"] = b_count
            clonotype.clone_metadata[f"{metadata_prefix}_background_potential"] = b_potential
            clonotype.clone_metadata[f"{metadata_prefix}_parent_density"] = p_density
            clonotype.clone_metadata[f"{metadata_prefix}_background_density"] = b_density
            clonotype.clone_metadata[f"{metadata_prefix}_enrichment"] = enrichment
