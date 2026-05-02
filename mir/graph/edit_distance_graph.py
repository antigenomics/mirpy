"""Edit-distance graph construction for Clonotypes.

Builds an ``igraph.Graph`` from a list of :class:`~mir.common.clonotype.Clonotype`
objects where edges connect sequences whose pairwise Hamming or Levenshtein
distance is at most *threshold*. Search is backed by ``tcrtrie``.
"""

from __future__ import annotations

import typing as t
from concurrent.futures import ThreadPoolExecutor
from math import ceil

import igraph as ig
from tcrtrie import Trie

from mir.common.clonotype import Clonotype
from mir.distances.seqdist import hamming as _hamming
from mir.distances.seqdist import levenshtein as _levenshtein
from mir.graph._trie_utils import hit_index, resolve_n_jobs, search_limits, validate_metric


def _build_batch_edges(
    seqs: list[str],
    v_genes: list[str],
    trie,
    c_genes: list[str],
    *,
    metric: str,
    threshold: int,
    v_gene_match: bool,
    c_gene_match: bool,
    start: int,
    stop: int,
) -> set[tuple[int, int]]:
    """Build unique edges for query indices in [start, stop)."""
    max_substitution, max_insertion, max_deletion, max_edits = search_limits(metric, threshold)
    edges: set[tuple[int, int]] = set()
    for i in range(start, stop):
        hits = trie.SearchIndices(
            query=seqs[i],
            maxSubstitution=max_substitution,
            maxInsertion=max_insertion,
            maxDeletion=max_deletion,
            maxEdits=max_edits,
            vGeneFilter=v_genes[i] if v_gene_match else None,
            jGeneFilter=None,
        )
        for hit in hits:
            j = hit_index(hit)
            if j <= i:
                continue
            if c_gene_match and c_genes[i] != c_genes[j]:
                continue
            edges.add((i, j))
    return edges


def _build_batch_edges_bruteforce(
    seqs: list[str],
    v_genes: list[str],
    c_genes: list[str],
    *,
    metric: str,
    threshold: int,
    v_gene_match: bool,
    c_gene_match: bool,
    start: int,
    stop: int,
) -> set[tuple[int, int]]:
    """Build unique edges for query indices in [start, stop) via C-distance kernels."""
    n = len(seqs)
    edges: set[tuple[int, int]] = set()
    for i in range(start, stop):
        seq_i = seqs[i]
        v_i = v_genes[i]
        c_i = c_genes[i]
        for j in range(i + 1, n):
            if v_gene_match and v_i != v_genes[j]:
                continue
            if c_gene_match and c_i != c_genes[j]:
                continue
            if metric == "hamming":
                if len(seq_i) != len(seqs[j]):
                    continue
                dist = _hamming(seq_i, seqs[j])
            else:
                dist = _levenshtein(seq_i, seqs[j])
            if dist <= threshold:
                edges.add((i, j))
    return edges


def _build_edges_parallel(
    *,
    n: int,
    jobs: int,
    chunk_sz: int,
    builder: t.Callable[[int, int], set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    if n <= 1:
        return set()
    if jobs <= 1 or n <= chunk_sz:
        return builder(0, n)

    batch_size = max(1, chunk_sz, ceil(n / jobs))
    ranges = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]
    edges: set[tuple[int, int]] = set()
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(builder, start, stop) for start, stop in ranges]
        for future in futures:
            edges.update(future.result())
    return edges


def build_edit_distance_graph(
    rearrangements: list[Clonotype],
    metric: str = "hamming",
    threshold: int = 1,
    v_gene_match: bool = False,
    c_gene_match: bool = False,
    n_jobs: int | None = None,
    nproc: int | None = None,
    chunk_sz: int = 2048,
) -> ig.Graph:
    """Build an edit-distance graph from a list of Clonotypes.

    One vertex is created per rearrangement (duplicates are preserved).
    An edge is added between every pair whose ``junction_aa`` distance is
    ≤ *threshold*.  For Hamming distance, sequences of unequal length are
    never connected.

    Args:
        rearrangements: Input rearrangements.
        metric: ``"hamming"`` or ``"levenshtein"``.
        threshold: Maximum distance for an edge.
        v_gene_match: When ``True``, only compare pairs with matching ``v_gene``.
        c_gene_match: When ``True``, only compare pairs with matching ``c_gene``.
        n_jobs: Worker count for trie-query batches.
        nproc: Backward-compat alias for ``n_jobs``.
        chunk_sz: Query sequences per worker batch.

    Returns:
        Undirected ``igraph.Graph`` with vertex attributes ``name``
        (``junction_aa``), ``r_id`` (:attr:`Clonotype.id`),
        ``v_gene``, and ``c_gene``.
    """
    validate_metric(metric)
    jobs = resolve_n_jobs(n_jobs=n_jobs, nproc=nproc, default=4)

    n = len(rearrangements)
    seqs = [str(getattr(r, "junction_aa", "") or "") for r in rearrangements]
    v_genes = [str(getattr(r, "v_gene", "") or "") for r in rearrangements]
    j_genes = [str(getattr(r, "j_gene", "") or "") for r in rearrangements]
    trie = Trie(sequences=seqs, vGenes=v_genes, jGenes=j_genes)
    c_genes = [str(getattr(r, "c_gene", "") or "") for r in rearrangements]

    if metric == "hamming":
        edges = _build_edges_parallel(
            n=n,
            jobs=jobs,
            chunk_sz=chunk_sz,
            builder=lambda start, stop: _build_batch_edges(
                seqs,
                v_genes,
                trie,
                c_genes,
                metric=metric,
                threshold=threshold,
                v_gene_match=v_gene_match,
                c_gene_match=c_gene_match,
                start=start,
                stop=stop,
            ),
        )
    else:
        edges = _build_edges_parallel(
            n=n,
            jobs=jobs,
            chunk_sz=chunk_sz,
            builder=lambda start, stop: _build_batch_edges_bruteforce(
                seqs,
                v_genes,
                c_genes,
                metric=metric,
                threshold=threshold,
                v_gene_match=v_gene_match,
                c_gene_match=c_gene_match,
                start=start,
                stop=stop,
            ),
        )

    g = ig.Graph(n=n, directed=False)
    g.vs["name"]   = [r.junction_aa for r in rearrangements]
    g.vs["r_id"]   = [r.id          for r in rearrangements]
    g.vs["v_gene"] = [r.v_gene      for r in rearrangements]
    g.vs["c_gene"] = [r.c_gene      for r in rearrangements]
    if edges:
        g.add_edges(sorted(edges))
    return g
