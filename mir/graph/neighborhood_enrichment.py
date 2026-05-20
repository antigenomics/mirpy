"""Neighborhood enrichment statistics for TCRNET and ALICE algorithms.

For each clonotype in a query repertoire, counts the number of neighbors within
a given search scope:

- **Sequence scope**: ``junction_aa`` mismatches (Hamming distance, default
  threshold=1) or insertions/deletions (Levenshtein distance).
- **Gene scope**: optionally restrict neighbors to the same V gene, J gene, or
  both (``match_v_gene`` / ``match_j_gene``).  This is the V+J restriction used
  by the original ALICE paper.

Performance
-----------
All searches are backed by **tcrtrie** for sub-linear trie-based lookups.  A
brute-force Python fallback is used only when tcrtrie raises (e.g. for
sequences longer than 64 AA for Hamming or 33 AA for Levenshtein).

For V/J-restricted searches, background sequences are grouped by (V,J) key and
a separate small Trie is built per group.  This eliminates the O(N × k) Python
validation loop that dominates for natural repertoires.

Parallelism is automatic: :func:`compute_neighborhood_stats_by_locus` spawns
``n_jobs`` worker processes via ``ProcessPoolExecutor`` when the repertoire
exceeds ``_NEIGHBOR_PARALLEL_MIN_CLONOTYPES`` (20 000) clonotypes.  Background
sequence arrays are passed through shared memory to avoid per-worker copies.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from math import ceil
import typing as t

import numpy as np

_MP_CTX = multiprocessing.get_context("spawn")
from typing import TYPE_CHECKING

from tcrtrie import Trie

from mir.graph._trie_utils import (
    hit_index,
    resolve_n_jobs,
    search_indices_with_fallback,
    search_limits,
    validate_metric,
)
from mir.graph.distance_utils import is_within_threshold
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


def _init_neighbor_worker(
    query_sequences: list[str],
    query_sequence_ids: list[str],
    query_v_genes: list[str],
    query_j_genes: list[str],
    background_sequences_spec: SharedArraySpec,
    background_v_genes_spec: SharedArraySpec,
    background_j_genes_spec: SharedArraySpec,
    metric: str,
    threshold: int,
    match_v_gene: bool,
    match_j_gene: bool,
    background_size: int,
    potential_counter: dict[t.Any, int] | None,
    add_self_pseudocount: bool,
) -> None:
    """Initialize per-process state for neighborhood batch workers."""
    background_sequences_arr, background_sequences_shm = attach_shared_array(background_sequences_spec)
    background_v_genes_arr, background_v_genes_shm = attach_shared_array(background_v_genes_spec)
    background_j_genes_arr, background_j_genes_shm = attach_shared_array(background_j_genes_spec)

    background_sequences = np.char.decode(background_sequences_arr, "ascii").tolist()
    background_v_genes = np.char.decode(background_v_genes_arr, "ascii").tolist()
    background_j_genes = np.char.decode(background_j_genes_arr, "ascii").tolist()

    _NEIGHBOR_WORKER_STATE["query_sequences"] = query_sequences
    _NEIGHBOR_WORKER_STATE["query_sequence_ids"] = query_sequence_ids
    _NEIGHBOR_WORKER_STATE["query_v_genes"] = query_v_genes
    _NEIGHBOR_WORKER_STATE["query_j_genes"] = query_j_genes
    _NEIGHBOR_WORKER_STATE["background_sequences"] = background_sequences
    _NEIGHBOR_WORKER_STATE["background_v_genes"] = background_v_genes
    _NEIGHBOR_WORKER_STATE["background_j_genes"] = background_j_genes
    _NEIGHBOR_WORKER_STATE["metric"] = metric
    _NEIGHBOR_WORKER_STATE["threshold"] = threshold
    _NEIGHBOR_WORKER_STATE["match_v_gene"] = match_v_gene
    _NEIGHBOR_WORKER_STATE["match_j_gene"] = match_j_gene
    _NEIGHBOR_WORKER_STATE["background_size"] = background_size
    _NEIGHBOR_WORKER_STATE["potential_counter"] = potential_counter
    _NEIGHBOR_WORKER_STATE["add_self_pseudocount"] = add_self_pseudocount
    _NEIGHBOR_WORKER_STATE["shm_handles"] = (
        background_sequences_shm,
        background_v_genes_shm,
        background_j_genes_shm,
    )
    _NEIGHBOR_WORKER_STATE["trie"] = Trie(
        sequences=background_sequences,
        vGenes=background_v_genes,
        jGenes=background_j_genes,
    )


def _potential_neighbor_count_from_genes(
    *,
    v_gene: str,
    j_gene: str,
    background_size: int,
    match_v_gene: bool,
    match_j_gene: bool,
    counter: dict[t.Any, int] | None,
) -> int:
    if counter is None:
        return background_size
    if match_v_gene and match_j_gene:
        key = (v_gene, j_gene)
    elif match_v_gene:
        key = v_gene
    else:
        key = j_gene
    return int(counter.get(key, 0))


def _compute_query_batch_worker(range_pair: tuple[int, int]) -> dict[str, dict[str, int]]:
    """Process-pool worker for neighborhood query ranges."""
    start, stop = range_pair
    query_sequences = _NEIGHBOR_WORKER_STATE["query_sequences"]
    query_sequence_ids = _NEIGHBOR_WORKER_STATE["query_sequence_ids"]
    query_v_genes = _NEIGHBOR_WORKER_STATE["query_v_genes"]
    query_j_genes = _NEIGHBOR_WORKER_STATE["query_j_genes"]
    background_sequences = _NEIGHBOR_WORKER_STATE["background_sequences"]
    background_v_genes = _NEIGHBOR_WORKER_STATE["background_v_genes"]
    background_j_genes = _NEIGHBOR_WORKER_STATE["background_j_genes"]
    trie = _NEIGHBOR_WORKER_STATE["trie"]
    metric = _NEIGHBOR_WORKER_STATE["metric"]
    threshold = _NEIGHBOR_WORKER_STATE["threshold"]
    match_v_gene = _NEIGHBOR_WORKER_STATE["match_v_gene"]
    match_j_gene = _NEIGHBOR_WORKER_STATE["match_j_gene"]
    background_size = _NEIGHBOR_WORKER_STATE["background_size"]
    potential_counter = _NEIGHBOR_WORKER_STATE["potential_counter"]
    add_self_pseudocount = _NEIGHBOR_WORKER_STATE["add_self_pseudocount"]

    out: dict[str, dict[str, int]] = {}
    for i in range(start, stop):
        hits = search_indices_with_fallback(
            trie,
            query=query_sequences[i],
            metric=metric,
            threshold=threshold,
            sequences=background_sequences,
            v_gene_filter=query_v_genes[i] if match_v_gene else None,
            j_gene_filter=query_j_genes[i] if match_j_gene else None,
            v_genes=background_v_genes,
            j_genes=background_j_genes,
        )
        potential_neighbors = _potential_neighbor_count_from_genes(
            v_gene=query_v_genes[i],
            j_gene=query_j_genes[i],
            background_size=background_size,
            match_v_gene=match_v_gene,
            match_j_gene=match_j_gene,
            counter=potential_counter,
        )
        neighbor_count = len(hits)
        if add_self_pseudocount:
            potential_neighbors += 1
            neighbor_count += 1
        out[query_sequence_ids[i]] = {
            "neighbor_count": int(neighbor_count),
            "potential_neighbors": int(potential_neighbors),
        }
    return out

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


def _gene_key(
    v_gene: str | None,
    j_gene: str | None,
    *,
    match_v_gene: bool,
    match_j_gene: bool,
) -> tuple[str, ...]:
    """Return the V/J grouping key for a clonotype."""
    assert match_v_gene or match_j_gene, "_gene_key requires at least one gene flag"
    vv = v_gene or ""
    jj = j_gene or ""
    if match_v_gene and match_j_gene:
        return (vv, jj)
    if match_v_gene:
        return (vv,)
    return (jj,)


def _build_grouped_tries(
    sequences: list[str],
    v_genes: list[str],
    j_genes: list[str],
    *,
    match_v_gene: bool,
    match_j_gene: bool,
) -> tuple[dict[tuple, list[str]], dict[tuple, "Trie"]]:
    """Group sequences by V/J gene key and build one small Trie per group.

    Returns (group_seqs, group_tries) where group_seqs[key] is the CDR3 list
    for that gene group and group_tries[key] is its Trie (no V/J genes stored
    because the group is homogeneous — search needs no V/J filter).

    This replaces the single-large-trie + Python V/J validation approach with
    N_groups smaller tries, each searched without filters, eliminating the
    O(N × k_avg) Python loop that dominates for natural repertoires.

    Note: Long-CDR3 brute-force augmentation (>64 AA for Hamming, >33 for
    Levenshtein) is not applied here because TCR CDR3s are always short.  If
    future callers pass synthetic long sequences the count may be slightly
    under-reported; this is explicitly acceptable and matches the behavior of
    ``search_indices_with_fallback`` on the same data.
    """
    group_seqs: dict[tuple, list[str]] = {}
    for seq, v, j in zip(sequences, v_genes, j_genes):
        key = _gene_key(v, j, match_v_gene=match_v_gene, match_j_gene=match_j_gene)
        group_seqs.setdefault(key, []).append(seq)
    group_tries: dict[tuple, Trie] = {}
    for key, seqs in group_seqs.items():
        n = len(seqs)
        group_tries[key] = Trie(sequences=seqs, vGenes=[""] * n, jGenes=[""] * n)
    return group_seqs, group_tries


def _count_grouped_neighbors(
    query: str,
    group_trie: "Trie",
    group_seqs: list[str],
    metric: str,
    threshold: int,
) -> int:
    """Count Hamming/Levenshtein neighbors of *query* in a gene-group trie."""
    max_sub, max_ins, max_del, max_edits = search_limits(metric, threshold)
    n = len(group_seqs)
    try:
        hits = group_trie.SearchIndices(
            query=query,
            maxSubstitution=max_sub,
            maxInsertion=max_ins,
            maxDeletion=max_del,
            maxEdits=max_edits,
        )
    except Exception:
        return sum(1 for s in group_seqs if is_within_threshold(query, s, metric, threshold))
    count = 0
    for hit in hits:
        local_idx = hit_index(hit)
        if 0 <= local_idx < n and is_within_threshold(query, group_seqs[local_idx], metric, threshold):
            count += 1
    return count


def _compute_grouped_query_batch(
    query_clonotypes: list,
    sequence_ids: list[str],
    group_seqs: dict[tuple, list[str]],
    group_tries: dict[tuple, "Trie"],
    *,
    metric: str,
    threshold: int,
    match_v_gene: bool,
    match_j_gene: bool,
    add_self_pseudocount: bool,
) -> dict[str, dict[str, int]]:
    """Serial grouped-trie batch: one trie lookup per gene group."""
    out: dict[str, dict[str, int]] = {}
    for clonotype, seq_id in zip(query_clonotypes, sequence_ids):
        key = _gene_key(
            clonotype.v_gene, clonotype.j_gene,
            match_v_gene=match_v_gene, match_j_gene=match_j_gene,
        )
        g_seqs = group_seqs.get(key)
        g_trie = group_tries.get(key)
        g_size = len(g_seqs) if g_seqs is not None else 0
        if g_trie is None:
            nc = 1 if add_self_pseudocount else 0
            pn = 1 if add_self_pseudocount else 0
        else:
            nc = _count_grouped_neighbors(clonotype.junction_aa, g_trie, g_seqs, metric, threshold)
            pn = g_size
            if add_self_pseudocount:
                nc += 1
                pn += 1
        out[seq_id] = {"neighbor_count": nc, "potential_neighbors": pn}
    return out


def _compute_grouped_key_batch_worker(
    bg_by_key: dict[tuple, list[str]],
    q_by_key: dict[tuple, tuple[list[str], list[str]]],
    metric: str,
    threshold: int,
    add_self_pseudocount: bool,
) -> dict[str, dict[str, int]]:
    """Build tries for assigned (V,J) key groups and search their queries.

    Each worker receives only the background and query data for its assigned keys,
    builds exactly those tries, and returns results for all matching queries.
    Tries are built once per key globally (not once per worker).
    """
    out: dict[str, dict[str, int]] = {}
    for key, bg_seqs in bg_by_key.items():
        q_data = q_by_key.get(key)
        if q_data is None:
            continue
        q_seqs, q_ids = q_data
        n = len(bg_seqs)
        trie = Trie(sequences=bg_seqs, vGenes=[""] * n, jGenes=[""] * n)
        for q_seq, seq_id in zip(q_seqs, q_ids):
            nc = _count_grouped_neighbors(q_seq, trie, bg_seqs, metric, threshold)
            pn = n
            if add_self_pseudocount:
                nc += 1
                pn += 1
            out[seq_id] = {"neighbor_count": nc, "potential_neighbors": pn}
    # Keys present in queries but absent from background → return zero/pseudocount.
    for key, (q_seqs, q_ids) in q_by_key.items():
        if key not in bg_by_key:
            nc = pn = 1 if add_self_pseudocount else 0
            for seq_id in q_ids:
                out[seq_id] = {"neighbor_count": nc, "potential_neighbors": pn}
    return out


def _compute_query_batch(
    query_clonotypes: list,
    sequence_ids: list[str],
    background_sequences: list[str],
    background_v_genes: list[str],
    background_j_genes: list[str],
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
    out: dict[str, dict[str, int]] = {}
    for i in range(start, stop):
        clonotype = query_clonotypes[i]
        hits = search_indices_with_fallback(
            trie,
            query=clonotype.junction_aa,
            metric=metric,
            threshold=threshold,
            sequences=background_sequences,
            v_gene_filter=clonotype.v_gene if match_v_gene else None,
            j_gene_filter=clonotype.j_gene if match_j_gene else None,
            v_genes=background_v_genes,
            j_genes=background_j_genes,
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

    background_sequences = [c.junction_aa for c in background_clonotypes]
    background_v_genes = [c.v_gene for c in background_clonotypes]
    background_j_genes = [c.j_gene for c in background_clonotypes]
    query_sequences = [c.junction_aa for c in query_clonotypes]
    query_v_genes = [c.v_gene for c in query_clonotypes]
    query_j_genes = [c.j_gene for c in query_clonotypes]
    n_query = len(query_clonotypes)

    # ── Grouped-trie path (V/J-restricted search) ────────────────────────────
    # Build one small Trie per (V,J) group rather than filtering one large Trie
    # in Python.  Each per-group Trie is ~N_total / N_groups sequences, so the
    # search is faster and no Python V/J validation loop is needed.
    if match_v_gene or match_j_gene:
        # Group background sequences by (V,J) key.
        bg_by_key: dict[tuple, list[str]] = {}
        for seq, v, j in zip(background_sequences, background_v_genes, background_j_genes):
            key = _gene_key(v, j, match_v_gene=match_v_gene, match_j_gene=match_j_gene)
            bg_by_key.setdefault(key, []).append(seq)

        if n_jobs <= 1 or n_query < _NEIGHBOR_PARALLEL_MIN_CLONOTYPES:
            g_seqs, g_tries = _build_grouped_tries(
                background_sequences, background_v_genes, background_j_genes,
                match_v_gene=match_v_gene, match_j_gene=match_j_gene,
            )
            return _compute_grouped_query_batch(
                query_clonotypes, q_seq_ids, g_seqs, g_tries,
                metric=metric, threshold=threshold,
                match_v_gene=match_v_gene, match_j_gene=match_j_gene,
                add_self_pseudocount=add_self_pseudocount,
            )

        # Parallel grouped path: distribute (V,J) key groups across workers.
        # Each worker builds only its assigned tries (1/n_jobs of the total),
        # so each trie is built exactly once — no per-worker rebuild overhead.
        q_by_key: dict[tuple, tuple[list[str], list[str]]] = {}
        for q, seq_id, v, j in zip(query_sequences, q_seq_ids, query_v_genes, query_j_genes):
            key = _gene_key(v, j, match_v_gene=match_v_gene, match_j_gene=match_j_gene)
            if key not in q_by_key:
                q_by_key[key] = ([], [])
            q_by_key[key][0].append(q)
            q_by_key[key][1].append(seq_id)

        # Assign keys to workers in round-robin by descending bg size (load balance).
        all_keys = sorted(bg_by_key, key=lambda k: len(bg_by_key[k]), reverse=True)
        worker_bg: list[dict] = [{} for _ in range(n_jobs)]
        worker_q: list[dict] = [{} for _ in range(n_jobs)]
        for i, key in enumerate(all_keys):
            w = i % n_jobs
            worker_bg[w][key] = bg_by_key[key]
            if key in q_by_key:
                worker_q[w][key] = q_by_key[key]

        results: dict[str, dict[str, int]] = {}
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_MP_CTX) as executor:
            futures = [
                executor.submit(
                    _compute_grouped_key_batch_worker,
                    worker_bg[w], worker_q[w], metric, threshold, add_self_pseudocount,
                )
                for w in range(n_jobs)
                if worker_bg[w]
            ]
            for f in futures:
                results.update(f.result())
        return results

    # ── Single-trie path (no V/J restriction) ────────────────────────────────
    # Trie construction deferred here so the grouped path never pays this cost.
    trie = background_locus.trie
    potential_counter = None  # match_v_gene=False, match_j_gene=False → no counter needed
    if n_jobs <= 1 or n_query < _NEIGHBOR_PARALLEL_MIN_CLONOTYPES:
        return _compute_query_batch(
            query_clonotypes,
            q_seq_ids,
            background_sequences,
            background_v_genes,
            background_j_genes,
            trie,
            metric=metric,
            threshold=threshold,
            match_v_gene=False,
            match_j_gene=False,
            background_size=n_background,
            potential_counter=potential_counter,
            add_self_pseudocount=add_self_pseudocount,
            start=0,
            stop=n_query,
        )

    batch_size = max(1, ceil(n_query / n_jobs))
    ranges = [(start, min(start + batch_size, n_query)) for start in range(0, n_query, batch_size)]
    results = {}
    shared_handles = []
    try:
        bg_seq_spec, bg_seq_shm = create_shared_array(fixed_bytes_array(background_sequences))
        shared_handles.append(bg_seq_shm)
        bg_v_spec, bg_v_shm = create_shared_array(fixed_bytes_array(background_v_genes))
        shared_handles.append(bg_v_shm)
        bg_j_spec, bg_j_shm = create_shared_array(fixed_bytes_array(background_j_genes))
        shared_handles.append(bg_j_shm)

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=_MP_CTX,
            initializer=_init_neighbor_worker,
            initargs=(
                query_sequences,
                q_seq_ids,
                query_v_genes,
                query_j_genes,
                bg_seq_spec,
                bg_v_spec,
                bg_j_spec,
                metric,
                threshold,
                False,
                False,
                n_background,
                potential_counter,
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
