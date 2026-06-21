"""Neighborhood enrichment statistics for TCRNET and ALICE algorithms.

For each clonotype in a query repertoire, counts the number of neighbors within
a given search scope:

- **Sequence scope**: ``junction_aa`` mismatches (Hamming distance, default
  threshold=1) or insertions/deletions (Levenshtein distance).
- **Gene scope**: optionally restrict neighbors to the same V gene, J gene, or
  both (``match_v_call`` / ``match_j_call``).  This is the V+J restriction used
  by the original ALICE paper.

Performance
-----------
Searches are backed by **seqtree** (``seqtm`` engine) for sub-linear lookups.  A
single index is built per locus over the canonical background sequences; V/J
gene matching is applied as a cheap :func:`~mir.common.alleles.genes_match`
post-filter on the small hit set, and ``potential_neighbors`` is read from a
gene-key counter.

Parallelism is automatic: :func:`compute_neighborhood_stats_by_locus` spawns
``n_jobs`` worker processes via ``ProcessPoolExecutor`` when the repertoire
exceeds ``_NEIGHBOR_PARALLEL_MIN_CLONOTYPES`` (20 000) clonotypes.  Background
sequence arrays are passed through shared memory to avoid per-worker copies.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from math import ceil
import signal
import typing as t

import numpy as np

_MP_CTX = multiprocessing.get_context("spawn")
from typing import TYPE_CHECKING

from mir.common.alleles import genes_match
from mir.graph._trie_utils import (
    make_index,
    make_params,
    resolve_n_jobs,
    search_canon_indices,
    validate_metric,
)
from mir.utils.shared_memory import (
    SharedArraySpec,
    attach_shared_array,
    close_unlink_many,
    create_shared_array,
    fixed_bytes_array,
)

if TYPE_CHECKING:
    from mir.common.repertoire import LocusRepertoire, SampleRepertoire


_NEIGHBOR_WORKER_STATE: dict[str, t.Any] = {}
_NEIGHBOR_PARALLEL_MIN_CLONOTYPES = 20_000


def _gene_key(
    v_call: str | None,
    j_call: str | None,
    *,
    match_v_call: bool,
    match_j_call: bool,
) -> tuple[str, ...]:
    """Return the V/J grouping key for a clonotype, preserving allele suffix."""
    assert match_v_call or match_j_call, "_gene_key requires at least one gene flag"
    vv = str(v_call or "")
    jj = str(j_call or "")
    if match_v_call and match_j_call:
        return (vv, jj)
    if match_v_call:
        return (vv,)
    return (jj,)


def _matching_group_keys(
    query_v: str | None,
    query_j: str | None,
    available_keys: t.Iterable[tuple[str, ...]],
    *,
    match_v_call: bool,
    match_j_call: bool,
) -> list[tuple[str, ...]]:
    """Return all group keys from *available_keys* that match the query genes.

    Uses :func:`~mir.common.alleles.genes_match` semantics: a bare gene is a
    wildcard that matches any allele of the same base gene.
    """
    result: list[tuple[str, ...]] = []
    qv = str(query_v or "")
    qj = str(query_j or "")
    for key in available_keys:
        if match_v_call and match_j_call:
            gv, gj = key
            if genes_match(qv, gv) and genes_match(qj, gj):
                result.append(key)
        elif match_v_call:
            (gv,) = key
            if genes_match(qv, gv):
                result.append(key)
        else:
            (gj,) = key
            if genes_match(qj, gj):
                result.append(key)
    return result


def _build_potential_counter(
    background_clonotypes: list,
    *,
    match_v_call: bool,
    match_j_call: bool,
) -> dict[t.Any, int] | None:
    """Count background clonotypes per V/J gene key (``None`` when no gene flag)."""
    if not match_v_call and not match_j_call:
        return None
    counter: dict[t.Any, int] = {}
    for clonotype in background_clonotypes:
        key = _gene_key(
            clonotype.v_call,
            clonotype.j_call,
            match_v_call=match_v_call,
            match_j_call=match_j_call,
        )
        counter[key] = counter.get(key, 0) + 1
    return counter


def _compute_query_range(
    query_sequences: list[str],
    query_v_calls: list[str],
    query_j_calls: list[str],
    query_sequence_ids: list[str],
    index,
    idx_to_orig: list[int],
    background_v_calls: list[str],
    background_j_calls: list[str],
    params,
    background_size: int,
    match_v_call: bool,
    match_j_call: bool,
    potential_counter: dict[t.Any, int] | None,
    bg_keys: list[tuple[str, ...]],
    add_self_pseudocount: bool,
    start: int,
    stop: int,
) -> dict[str, dict[str, int]]:
    """Count neighbors and potential neighbors for query indices in [start, stop).

    ``neighbor_count`` = index hits whose background V/J pass ``genes_match``;
    ``potential_neighbors`` = sum of gene-key counts matching the query (wildcard),
    or ``background_size`` when no gene flag is set.
    """
    out: dict[str, dict[str, int]] = {}
    for i in range(start, stop):
        qv = query_v_calls[i]
        qj = query_j_calls[i]
        neighbor_count = 0
        for ci in search_canon_indices(index, query_sequences[i], params):
            orig = idx_to_orig[ci]
            if match_v_call and not genes_match(qv, background_v_calls[orig]):
                continue
            if match_j_call and not genes_match(qj, background_j_calls[orig]):
                continue
            neighbor_count += 1

        if potential_counter is None:
            potential_neighbors = background_size
        else:
            potential_neighbors = sum(
                potential_counter[key]
                for key in _matching_group_keys(
                    qv, qj, bg_keys, match_v_call=match_v_call, match_j_call=match_j_call
                )
            )

        if add_self_pseudocount:
            neighbor_count += 1
            potential_neighbors += 1
        out[query_sequence_ids[i]] = {
            "neighbor_count": int(neighbor_count),
            "potential_neighbors": int(potential_neighbors),
        }
    return out


def _init_neighbor_worker(
    query_sequences: list[str],
    query_sequence_ids: list[str],
    query_v_calls: list[str],
    query_j_calls: list[str],
    background_sequences_spec: SharedArraySpec,
    background_v_calls_spec: SharedArraySpec,
    background_j_calls_spec: SharedArraySpec,
    metric: str,
    threshold: int,
    match_v_call: bool,
    match_j_call: bool,
    background_size: int,
    potential_counter: dict[t.Any, int] | None,
    bg_keys: list[tuple[str, ...]],
    add_self_pseudocount: bool,
) -> None:
    """Initialize per-process state for neighborhood batch workers."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    bg_seq_arr, bg_seq_shm = attach_shared_array(background_sequences_spec)
    bg_v_arr, bg_v_shm = attach_shared_array(background_v_calls_spec)
    bg_j_arr, bg_j_shm = attach_shared_array(background_j_calls_spec)

    background_sequences = np.char.decode(bg_seq_arr, "ascii").tolist()
    index, idx_to_orig = make_index(background_sequences)

    _NEIGHBOR_WORKER_STATE.update(
        query_sequences=query_sequences,
        query_sequence_ids=query_sequence_ids,
        query_v_calls=query_v_calls,
        query_j_calls=query_j_calls,
        background_v_calls=np.char.decode(bg_v_arr, "ascii").tolist(),
        background_j_calls=np.char.decode(bg_j_arr, "ascii").tolist(),
        index=index,
        idx_to_orig=idx_to_orig,
        params=make_params(metric, threshold),
        background_size=background_size,
        match_v_call=match_v_call,
        match_j_call=match_j_call,
        potential_counter=potential_counter,
        bg_keys=bg_keys,
        add_self_pseudocount=add_self_pseudocount,
        shm_handles=(bg_seq_shm, bg_v_shm, bg_j_shm),
    )


def _compute_query_batch_worker(range_pair: tuple[int, int]) -> dict[str, dict[str, int]]:
    """Process-pool worker for neighborhood query ranges."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    s = _NEIGHBOR_WORKER_STATE
    return _compute_query_range(
        s["query_sequences"],
        s["query_v_calls"],
        s["query_j_calls"],
        s["query_sequence_ids"],
        s["index"],
        s["idx_to_orig"],
        s["background_v_calls"],
        s["background_j_calls"],
        s["params"],
        s["background_size"],
        s["match_v_call"],
        s["match_j_call"],
        s["potential_counter"],
        s["bg_keys"],
        s["add_self_pseudocount"],
        range_pair[0],
        range_pair[1],
    )


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
    match_v_call: bool,
    match_j_call: bool,
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
        v = 1 if add_self_pseudocount else 0
        return {sid: {"neighbor_count": v, "potential_neighbors": v} for sid in q_seq_ids}

    background_sequences = [c.junction_aa for c in background_clonotypes]
    background_v_calls = [c.v_call or "" for c in background_clonotypes]
    background_j_calls = [c.j_call or "" for c in background_clonotypes]
    query_sequences = [c.junction_aa for c in query_clonotypes]
    query_v_calls = [c.v_call or "" for c in query_clonotypes]
    query_j_calls = [c.j_call or "" for c in query_clonotypes]
    n_query = len(query_clonotypes)

    potential_counter = _build_potential_counter(
        background_clonotypes, match_v_call=match_v_call, match_j_call=match_j_call
    )
    bg_keys = list(potential_counter) if potential_counter is not None else []

    if n_jobs <= 1 or n_query < _NEIGHBOR_PARALLEL_MIN_CLONOTYPES:
        index, idx_to_orig = make_index(background_sequences)
        return _compute_query_range(
            query_sequences,
            query_v_calls,
            query_j_calls,
            q_seq_ids,
            index,
            idx_to_orig,
            background_v_calls,
            background_j_calls,
            make_params(metric, threshold),
            n_background,
            match_v_call,
            match_j_call,
            potential_counter,
            bg_keys,
            add_self_pseudocount,
            0,
            n_query,
        )

    batch_size = max(1, ceil(n_query / n_jobs))
    ranges = [(start, min(start + batch_size, n_query)) for start in range(0, n_query, batch_size)]
    results: dict[str, dict[str, int]] = {}
    shared_handles = []
    try:
        bg_seq_spec, bg_seq_shm = create_shared_array(fixed_bytes_array(background_sequences))
        shared_handles.append(bg_seq_shm)
        bg_v_spec, bg_v_shm = create_shared_array(fixed_bytes_array(background_v_calls))
        shared_handles.append(bg_v_shm)
        bg_j_spec, bg_j_shm = create_shared_array(fixed_bytes_array(background_j_calls))
        shared_handles.append(bg_j_shm)

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=_MP_CTX,
            initializer=_init_neighbor_worker,
            initargs=(
                query_sequences,
                q_seq_ids,
                query_v_calls,
                query_j_calls,
                bg_seq_spec,
                bg_v_spec,
                bg_j_spec,
                metric,
                threshold,
                match_v_call,
                match_j_call,
                n_background,
                potential_counter,
                bg_keys,
                add_self_pseudocount,
            ),
        ) as executor:
            for batch_result in executor.map(_compute_query_batch_worker, ranges):
                results.update(batch_result)
    finally:
        close_unlink_many(shared_handles)
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
    match_v_call: bool,
    match_j_call: bool,
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
            match_v_call=match_v_call,
            match_j_call=match_j_call,
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
    match_v_call: bool = False,
    match_j_call: bool = False,
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
        match_v_call=match_v_call,
        match_j_call=match_j_call,
        add_background_pseudocount=add_background_pseudocount,
        n_jobs=n_jobs,
    )


def compute_neighborhood_stats(
    repertoire: LocusRepertoire | SampleRepertoire,
    background: LocusRepertoire | SampleRepertoire | None = None,
    metric: str = "hamming",
    threshold: int = 1,
    match_v_call: bool = False,
    match_j_call: bool = False,
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
    match_v_call
        If True, only count neighbors with matching v_call.
    match_j_call
        If True, only count neighbors with matching j_call.

    Returns
    -------
    dict[str, dict]
        Mapping from ``sequence_id`` to a stats dict. Keys are
        ``"neighbor_count"`` (neighbors within threshold, including self)
        and ``"potential_neighbors"`` (total clonotypes satisfying gene
        matching constraints).

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
        match_v_call=match_v_call,
        match_j_call=match_j_call,
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
    match_v_call: bool = False,
    match_j_call: bool = False,
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
    match_v_call
        If True, only count neighbors with matching v_call.
    match_j_call
        If True, only count neighbors with matching j_call.
    metadata_prefix
        Prefix for metadata keys (e.g., ``"neighborhood_count"`` and
        ``"neighborhood_potential"``).
    """
    stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_call=match_v_call,
        match_j_call=match_j_call,
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
    match_v_call: bool = False,
    match_j_call: bool = False,
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
    match_v_call
        If True, only count neighbors with matching ``v_call``.
    match_j_call
        If True, only count neighbors with matching ``j_call``.
    metadata_prefix
        Prefix for metadata keys.
    """
    validate_metric(metric)

    parent_stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=None,
        metric=metric,
        threshold=threshold,
        match_v_call=match_v_call,
        match_j_call=match_j_call,
        n_jobs=n_jobs,
    )
    background_stats_by_locus = _compute_stats_by_locus(
        repertoire,
        background=background,
        metric=metric,
        threshold=threshold,
        match_v_call=match_v_call,
        match_j_call=match_j_call,
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
