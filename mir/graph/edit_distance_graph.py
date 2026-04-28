"""Edit-distance graph construction for Clonotypes.

Builds an ``igraph.Graph`` from a list of :class:`~mir.basic.token_tables.Clonotype`
objects where edges connect sequences whose pairwise Hamming or Levenshtein
distance is at most *threshold*.  Computation is parallelised over pair chunks
via :class:`multiprocessing.Pool`.
"""

from __future__ import annotations

import typing as t
from functools import partial
from itertools import islice
from multiprocessing import Pool
from typing import NamedTuple

import igraph as ig

from mir.common.clonotype import Clonotype
from mir.distances.seqdist import hamming as _hamming
from mir.distances.seqdist import levenshtein as _levenshtein


class _PairRecord(NamedTuple):
    """Pair of rearrangements to compare, with pre-extracted fields."""

    i: int
    j: int
    seq1: str
    seq2: str
    v1: str
    v2: str
    c1: str
    c2: str


def _process_chunk(
    chunk: list[_PairRecord],
    metric: str,
    threshold: int,
    v_gene_match: bool,
    c_gene_match: bool,
) -> list[tuple[int, int]]:
    """Return edges for all pairs in *chunk* that are within *threshold*."""
    edges: list[tuple[int, int]] = []
    for rec in chunk:
        if v_gene_match and rec.v1 != rec.v2:
            continue
        if c_gene_match and rec.c1 != rec.c2:
            continue
        if metric == "hamming":
            if len(rec.seq1) != len(rec.seq2):
                continue
            d = _hamming(rec.seq1, rec.seq2)
        else:
            d = _levenshtein(rec.seq1, rec.seq2)
        if d <= threshold:
            edges.append((rec.i, rec.j))
    return edges


def _iter_chunks(
    rearrangements: list[Clonotype],
    chunk_sz: int,
) -> t.Generator[list[_PairRecord], None, None]:
    n = len(rearrangements)
    seqs   = [r.junction_aa for r in rearrangements]
    v_genes = [r.v_gene     for r in rearrangements]
    c_genes = [r.c_gene     for r in rearrangements]
    pair_gen = (
        _PairRecord(i, j, seqs[i], seqs[j], v_genes[i], v_genes[j], c_genes[i], c_genes[j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    while True:
        chunk = list(islice(pair_gen, chunk_sz))
        if not chunk:
            break
        yield chunk


def build_edit_distance_graph(
    rearrangements: list[Clonotype],
    metric: str = "hamming",
    threshold: int = 1,
    v_gene_match: bool = False,
    c_gene_match: bool = False,
    nproc: int = 4,
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
        nproc: Worker processes.  ``1`` skips ``Pool`` entirely (recommended
            for small datasets where spawn overhead exceeds compute time).
        chunk_sz: Pairs per worker chunk.

    Returns:
        Undirected ``igraph.Graph`` with vertex attributes ``name``
        (``junction_aa``), ``r_id`` (:attr:`Clonotype.id`),
        ``v_gene``, and ``c_gene``.
    """
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")

    n = len(rearrangements)
    worker = partial(
        _process_chunk,
        metric=metric,
        threshold=threshold,
        v_gene_match=v_gene_match,
        c_gene_match=c_gene_match,
    )
    chunks = _iter_chunks(rearrangements, chunk_sz)

    edges: list[tuple[int, int]] = []
    if nproc == 1:
        for result in map(worker, chunks):
            edges.extend(result)
    else:
        with Pool(nproc) as pool:
            for result in pool.imap_unordered(worker, chunks, chunksize=1):
                edges.extend(result)

    g = ig.Graph(n=n, directed=False)
    g.vs["name"]   = [r.junction_aa for r in rearrangements]
    g.vs["r_id"]   = [r.id          for r in rearrangements]
    g.vs["v_gene"] = [r.v_gene      for r in rearrangements]
    g.vs["c_gene"] = [r.c_gene      for r in rearrangements]
    if edges:
        g.add_edges(edges)
    return g
