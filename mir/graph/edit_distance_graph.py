"""Edit-distance graph construction for Clonotypes.

Builds an ``igraph.Graph`` from a list of :class:`~mir.common.clonotype.Clonotype`
objects where edges connect sequences whose pairwise Hamming or Levenshtein
distance is at most *threshold*. Search is backed by ``tcrtrie`` and falls
back to constrained brute-force only when trie search raises an error.
"""

from __future__ import annotations

import logging
import multiprocessing
import typing as t
from concurrent.futures import ProcessPoolExecutor
from math import ceil

_MP_CTX = multiprocessing.get_context("spawn")

import igraph as ig

from mir.common.alleles import genes_match, strip_allele
from mir.common.metaclonotype import MetaClonotypeClustering
from mir.common.clonotype import Clonotype
from mir.graph._trie_utils import (
    _is_trie_safe,
    make_trie,
    resolve_n_jobs,
    search_indices_with_fallback,
    validate_metric,
)
from mir.graph.distance_utils import is_within_threshold
from mir.utils.metaclonotype_clustering import metaclonotypes_from_graph_communities

_logger = logging.getLogger(__name__)


_EDGE_WORKER_STATE: dict[str, t.Any] = {}


def _init_edge_worker(
    seqs: list[str],
    v_calls: list[str],
    j_calls: list[str],
    c_calls: list[str],
    metric: str,
    threshold: int,
    v_call_match: bool,
    c_call_match: bool,
) -> None:
    _EDGE_WORKER_STATE["seqs"] = seqs
    _EDGE_WORKER_STATE["v_calls"] = v_calls
    _EDGE_WORKER_STATE["j_calls"] = j_calls
    _EDGE_WORKER_STATE["c_calls"] = c_calls
    _EDGE_WORKER_STATE["metric"] = metric
    _EDGE_WORKER_STATE["threshold"] = threshold
    _EDGE_WORKER_STATE["v_call_match"] = v_call_match
    _EDGE_WORKER_STATE["c_call_match"] = c_call_match
    # Trie uses stripped alleles for coarse grouping; original alleles are kept
    # separately so genes_match can apply fine-grained allele semantics afterwards.
    v_stripped = [strip_allele(v) for v in v_calls]
    j_stripped = [strip_allele(j) for j in j_calls]
    trie, trie_to_orig = make_trie(seqs, v_stripped, j_stripped)
    trie_orig_set = set(trie_to_orig)
    _EDGE_WORKER_STATE["trie"] = trie
    _EDGE_WORKER_STATE["trie_to_orig"] = trie_to_orig
    _EDGE_WORKER_STATE["canon_seqs"] = [seqs[i] for i in trie_to_orig]
    _EDGE_WORKER_STATE["canon_v"]    = [v_stripped[i] for i in trie_to_orig]
    _EDGE_WORKER_STATE["canon_j"]    = [j_stripped[i] for i in trie_to_orig]
    _EDGE_WORKER_STATE["non_canon_indices"] = [j for j in range(len(seqs)) if j not in trie_orig_set]


def _build_batch_edges_worker(range_pair: tuple[int, int]) -> set[tuple[int, int]]:
    start, stop = range_pair
    return _build_batch_edges(
        _EDGE_WORKER_STATE["seqs"],
        _EDGE_WORKER_STATE["v_calls"],
        _EDGE_WORKER_STATE["j_calls"],
        _EDGE_WORKER_STATE["trie"],
        _EDGE_WORKER_STATE["c_calls"],
        _EDGE_WORKER_STATE["trie_to_orig"],
        _EDGE_WORKER_STATE["canon_seqs"],
        _EDGE_WORKER_STATE["canon_v"],
        _EDGE_WORKER_STATE["canon_j"],
        _EDGE_WORKER_STATE["non_canon_indices"],
        metric=_EDGE_WORKER_STATE["metric"],
        threshold=_EDGE_WORKER_STATE["threshold"],
        v_call_match=_EDGE_WORKER_STATE["v_call_match"],
        c_call_match=_EDGE_WORKER_STATE["c_call_match"],
        start=start,
        stop=stop,
    )


def _build_batch_edges(
    seqs: list[str],
    v_calls: list[str],
    j_calls: list[str],
    trie,
    c_calls: list[str],
    trie_to_orig: list[int],
    canon_seqs: list[str],
    canon_v: list[str],
    canon_j: list[str],
    non_canon_indices: list[int],
    *,
    metric: str,
    threshold: int,
    v_call_match: bool,
    c_call_match: bool,
    start: int,
    stop: int,
) -> set[tuple[int, int]]:
    """Build unique edges for query indices in [start, stop).

    Canonical queries use the trie; non-canonical sequences are also checked via
    brute-force so canonical↔non-canonical edges are not silently dropped.
    """
    edges: set[tuple[int, int]] = set()
    for i in range(start, stop):
        if not _is_trie_safe(seqs[i]):  # non-canonical query — skip
            continue
        # Trie uses stripped alleles for coarse grouping.
        v_filter = strip_allele(v_calls[i]) if v_call_match else None
        # Returns indices into canon_seqs (canonical space, 0..len(trie_to_orig)-1).
        canon_hits = search_indices_with_fallback(
            trie,
            query=seqs[i],
            metric=metric,
            threshold=threshold,
            sequences=canon_seqs,
            v_call_filter=v_filter,
            j_call_filter=None,
            v_calls=canon_v,
            j_calls=canon_j,
        )
        for ci in canon_hits:
            j = trie_to_orig[ci]  # canonical index → original index
            if j <= i:
                continue
            # Fine-grained allele semantics: bare = wildcard, specific = exact.
            if v_call_match and not genes_match(v_calls[i], v_calls[j]):
                continue
            if c_call_match and c_calls[i] != c_calls[j]:
                continue
            edges.add((i, j))
        # Brute-force for non-canonical sequences not indexed by the trie.
        for j in non_canon_indices:
            if j <= i:
                continue
            if v_call_match and not genes_match(v_calls[i], v_calls[j]):
                continue
            if c_call_match and c_calls[i] != c_calls[j]:
                continue
            if is_within_threshold(seqs[i], seqs[j], metric, threshold):
                edges.add((i, j))
    return edges


def _build_edges_parallel(
    *,
    n: int,
    jobs: int,
    chunk_sz: int,
    seqs: list[str],
    v_calls: list[str],
    j_calls: list[str],
    c_calls: list[str],
    metric: str,
    threshold: int,
    v_call_match: bool,
    c_call_match: bool,
) -> set[tuple[int, int]]:
    if n <= 1:
        return set()
    if jobs <= 1 or n <= chunk_sz:
        v_stripped = [strip_allele(v) for v in v_calls]
        j_stripped = [strip_allele(j) for j in j_calls]
        trie, trie_to_orig = make_trie(seqs, v_stripped, j_stripped)
        trie_orig_set = set(trie_to_orig)
        canon_seqs = [seqs[i] for i in trie_to_orig]
        canon_v    = [v_stripped[i] for i in trie_to_orig]
        canon_j    = [j_stripped[i] for i in trie_to_orig]
        non_canon_indices = [j for j in range(n) if j not in trie_orig_set]
        return _build_batch_edges(
            seqs,
            v_calls,
            j_calls,
            trie,
            c_calls,
            trie_to_orig,
            canon_seqs,
            canon_v,
            canon_j,
            non_canon_indices,
            metric=metric,
            threshold=threshold,
            v_call_match=v_call_match,
            c_call_match=c_call_match,
            start=0,
            stop=n,
        )

    batch_size = max(1, chunk_sz, ceil(n / jobs))
    ranges = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]
    edges: set[tuple[int, int]] = set()
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=_MP_CTX,
        initializer=_init_edge_worker,
        initargs=(seqs, v_calls, j_calls, c_calls, metric, threshold, v_call_match, c_call_match),
    ) as executor:
        for chunk_edges in executor.map(_build_batch_edges_worker, ranges):
            edges.update(chunk_edges)
    return edges


def build_edit_distance_graph(
    rearrangements: list[Clonotype],
    metric: str = "hamming",
    threshold: int = 1,
    v_call_match: bool = False,
    c_call_match: bool = False,
    n_jobs: int | None = None,
    nproc: int | None = None,
    chunk_sz: int = 2048,
) -> ig.Graph:
    """Build an edit-distance graph from a list of Clonotypes.

    One vertex is created per rearrangement (duplicates are preserved).
    An edge is added between every pair whose ``junction_aa`` distance is
    ≤ *threshold*.  For Hamming distance, sequences of unequal length are
    never connected. For Levenshtein fallback, only candidates with
    ``abs(len(seq1) - len(seq2)) <= threshold`` are compared.

    Args:
        rearrangements: Input rearrangements.
        metric: ``"hamming"`` or ``"levenshtein"``.
        threshold: Maximum distance for an edge.
        v_call_match: When ``True``, only compare pairs with matching ``v_call``.
        c_call_match: When ``True``, only compare pairs with matching ``c_call``.
        n_jobs: Worker count for trie-query batches.
        nproc: Backward-compat alias for ``n_jobs``.
        chunk_sz: Query sequences per worker batch.

    Returns:
        Undirected ``igraph.Graph`` with vertex attributes ``name``
        (``junction_aa``), ``r_id`` (:attr:`Clonotype.id`),
        ``v_call``, ``j_call``, and ``c_call``.
    """
    validate_metric(metric)
    jobs = resolve_n_jobs(n_jobs=n_jobs, nproc=nproc, default=4)

    n = len(rearrangements)
    seqs = [str(getattr(r, "junction_aa", "") or "") for r in rearrangements]
    v_calls = [str(getattr(r, "v_call", "") or "") for r in rearrangements]
    j_calls = [str(getattr(r, "j_call", "") or "") for r in rearrangements]
    c_calls = [str(getattr(r, "c_call", "") or "") for r in rearrangements]

    edges = _build_edges_parallel(
        n=n,
        jobs=jobs,
        chunk_sz=chunk_sz,
        seqs=seqs,
        v_calls=v_calls,
        j_calls=j_calls,
        c_calls=c_calls,
        metric=metric,
        threshold=threshold,
        v_call_match=v_call_match,
        c_call_match=c_call_match,
    )

    g = ig.Graph(n=n, directed=False)
    g.vs["name"]   = [r.junction_aa for r in rearrangements]
    g.vs["r_id"]   = [r.id          for r in rearrangements]
    g.vs["v_call"] = [r.v_call      for r in rearrangements]
    g.vs["j_call"] = [r.j_call      for r in rearrangements]
    g.vs["c_call"] = [r.c_call      for r in rearrangements]
    if edges:
        g.add_edges(sorted(edges))
    return g


def metaclonotypes_from_edit_distance_graph(
    graph: ig.Graph,
    *,
    method: str = "components",
    min_cluster_size: int = 1,
) -> MetaClonotypeClustering:
    """Convert edit-distance graph communities/components into metaclonotypes.

    Args:
        graph: Graph produced by ``build_edit_distance_graph``.
        method: ``components``, ``leiden``, or ``louvain``.
        min_cluster_size: Drop clusters smaller than this size.
    """
    return metaclonotypes_from_graph_communities(
        graph,
        vertex_id_attr="r_id",
        method=method,
        min_cluster_size=min_cluster_size,
        weights=None,
    )
