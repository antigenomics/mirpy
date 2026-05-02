"""Neighborhood enrichment statistics for TCRnet and ALICE algorithms.

Computes neighborhood statistics for clonotypes. For each clonotype in a query
repertoire, counts the number of neighbors within a specified edit distance and
optional V/J gene matching constraints, either against itself or against an
explicit background repertoire.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import ceil
import typing as t
from typing import TYPE_CHECKING

from mir.graph._trie_utils import resolve_n_jobs, search_limits, validate_metric

if TYPE_CHECKING:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire

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


def _build_potential_counter(
    background_clonotypes: list,
    *,
    match_v_gene: bool,
    match_j_gene: bool,
) -> dict[t.Any, int] | None:
    if not match_v_gene and not match_j_gene:
        return None
    counter: dict[t.Any, int] = {}
    for clonotype in background_clonotypes:
        if match_v_gene and match_j_gene:
            key = (clonotype.v_gene, clonotype.j_gene)
        elif match_v_gene:
            key = clonotype.v_gene
        else:
            key = clonotype.j_gene
        counter[key] = counter.get(key, 0) + 1
    return counter


def _potential_neighbor_count(
    clonotype,
    *,
    background_size: int,
    match_v_gene: bool,
    match_j_gene: bool,
    counter: dict[t.Any, int] | None,
) -> int:
    if counter is None:
        return background_size
    if match_v_gene and match_j_gene:
        key = (clonotype.v_gene, clonotype.j_gene)
    elif match_v_gene:
        key = clonotype.v_gene
    else:
        key = clonotype.j_gene
    return int(counter.get(key, 0))


def _compute_query_batch(
    query_clonotypes: list,
    sequence_ids: list[str],
    trie,
    *,
    metric: str,
    threshold: int,
    match_v_gene: bool,
    match_j_gene: bool,
    background_size: int,
    potential_counter: dict[t.Any, int] | None,
    add_self_pseudocount: bool,
    start: int,
    stop: int,
) -> dict[str, dict[str, int]]:
    max_substitution, max_insertion, max_deletion, max_edits = search_limits(metric, threshold)
    out: dict[str, dict[str, int]] = {}
    for i in range(start, stop):
        clonotype = query_clonotypes[i]
        hits = trie.SearchIndices(
            query=clonotype.junction_aa,
            maxSubstitution=max_substitution,
            maxInsertion=max_insertion,
            maxDeletion=max_deletion,
            maxEdits=max_edits,
            vGeneFilter=clonotype.v_gene if match_v_gene else None,
            jGeneFilter=clonotype.j_gene if match_j_gene else None,
        )
        potential_neighbors = _potential_neighbor_count(
            clonotype,
            background_size=background_size,
            match_v_gene=match_v_gene,
            match_j_gene=match_j_gene,
            counter=potential_counter,
        )
        neighbor_count = len(hits)
        if add_self_pseudocount:
            potential_neighbors += 1
            neighbor_count += 1
        out[sequence_ids[i]] = {
            "neighbor_count": int(neighbor_count),
            "potential_neighbors": int(potential_neighbors),
        }
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
    n_jobs: int,
) -> dict[str, dict[str, int]]:
    query_clonotypes = query_locus.clonotypes
    background_clonotypes = background_locus.clonotypes

    if not query_clonotypes:
        return {}
    q_seq_ids = [c.sequence_id for c in query_clonotypes]
    n_background = len(background_clonotypes)
    if n_background == 0:
        return {
            seq_id: {
                "neighbor_count": int(1 if add_self_pseudocount else 0),
                "potential_neighbors": int(1 if add_self_pseudocount else 0),
            }
            for seq_id in q_seq_ids
        }

    trie = background_locus.trie
    potential_counter = _build_potential_counter(
        background_clonotypes,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
    )
    n_query = len(query_clonotypes)
    if n_jobs <= 1 or n_query < 32:
        return _compute_query_batch(
            query_clonotypes,
            q_seq_ids,
            trie,
            metric=metric,
            threshold=threshold,
            match_v_gene=match_v_gene,
            match_j_gene=match_j_gene,
            background_size=n_background,
            potential_counter=potential_counter,
            add_self_pseudocount=add_self_pseudocount,
            start=0,
            stop=n_query,
        )

    batch_size = max(1, ceil(n_query / n_jobs))
    ranges = [(start, min(start + batch_size, n_query)) for start in range(0, n_query, batch_size)]
    results: dict[str, dict[str, int]] = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(
                _compute_query_batch,
                query_clonotypes,
                q_seq_ids,
                trie,
                metric=metric,
                threshold=threshold,
                match_v_gene=match_v_gene,
                match_j_gene=match_j_gene,
                background_size=n_background,
                potential_counter=potential_counter,
                add_self_pseudocount=add_self_pseudocount,
                start=start,
                stop=stop,
            )
            for start, stop in ranges
        ]
        for future in futures:
            results.update(future.result())
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
    add_background_pseudocount: bool | None = None,
    n_jobs: int = 4,
) -> dict[str, dict[str, dict[str, int]]]:
    n_jobs = resolve_n_jobs(n_jobs=n_jobs, nproc=None, default=4)
    query_loci = _iter_loci(repertoire)
    bg_loci = _background_locus_map(query_loci, background)
    if add_background_pseudocount is None:
        add_self_pseudocount = not _is_same_background(repertoire, background)
    else:
        add_self_pseudocount = add_background_pseudocount and (not _is_same_background(repertoire, background))

    return {
        locus: _compute_locus_stats(
            qrep,
            bg_loci[locus],
            metric=metric,
            threshold=threshold,
            match_v_gene=match_v_gene,
            match_j_gene=match_j_gene,
            add_self_pseudocount=add_self_pseudocount,
            n_jobs=n_jobs,
        )
        for locus, qrep in query_loci.items()
    }


def compute_neighborhood_stats_by_locus(
    repertoire: LocusRepertoire | SampleRepertoire,
    background: LocusRepertoire | SampleRepertoire | None = None,
    metric: str = "hamming",
    threshold: int = 1,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
    add_background_pseudocount: bool | None = None,
    n_jobs: int = 4,
) -> dict[str, dict[str, dict[str, int]]]:
    """Compute neighborhood stats grouped by locus and sequence id.

    Parameters are identical to :func:`compute_neighborhood_stats`, with one
    additional control:

    add_background_pseudocount
        When ``None`` (default), preserve historical behavior (+1 pseudocount
        only when ``background`` is provided and is a different object).
        Set to ``False`` to disable pseudocounts in background mode.
    """
    validate_metric(metric)
    return _compute_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
        add_background_pseudocount=add_background_pseudocount,
        n_jobs=n_jobs,
    )


def compute_neighborhood_stats(
    repertoire: LocusRepertoire | SampleRepertoire,
    background: LocusRepertoire | SampleRepertoire | None = None,
    metric: str = "hamming",
    threshold: int = 1,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
    n_jobs: int = 4,
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
    validate_metric(metric)

    stats_by_locus = compute_neighborhood_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
        n_jobs=n_jobs,
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
    n_jobs: int = 4,
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
        n_jobs=n_jobs,
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
    n_jobs: int = 4,
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
    validate_metric(metric)

    parent_stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=None,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
        n_jobs=n_jobs,
    )
    background_stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_gene=match_v_gene,
        match_j_gene=match_j_gene,
        n_jobs=n_jobs,
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
