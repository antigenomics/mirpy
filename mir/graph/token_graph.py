"""Bipartite token graph construction for Rearrangements and Kmers.

Builds an ``igraph.Graph`` with two vertex types:

* **rearrangement** — one vertex per :class:`~mir.basic.token_tables.Rearrangement`.
* **kmer** — one vertex per unique :class:`~mir.basic.token_tables.Kmer` key
  present in the token table.

Edges connect each rearrangement vertex to every kmer vertex whose sequence
appears in that rearrangement's ``junction_aa``, as recorded by the token
table produced by :func:`~mir.basic.token_tables.tokenize_rearrangements`.

Use :func:`~mir.basic.token_tables.filter_token_table` to restrict the
token table before building the graph — by regex pattern, minimum
rearrangement count, or both.
"""

from __future__ import annotations

import igraph as ig

from mir.basic.token_tables import Kmer, KmerMatch, Rearrangement


def build_token_graph(
    rearrangements: list[Rearrangement],
    token_table: dict[Kmer, list[KmerMatch]],
) -> ig.Graph:
    """Build a bipartite Rearrangement–Kmer graph from a token table.

    Vertices 0 … n_r-1 represent the rearrangements (in list order).
    Vertices n_r … n_r+n_k-1 represent the unique kmers (in token-table
    insertion order).  An edge exists between rearrangement *i* and kmer *j*
    when kmer *j* appears in rearrangement *i*'s ``junction_aa``.  Parallel
    edges are deduplicated (a kmer may match a rearrangement at multiple
    positions).

    Vertex attributes
    -----------------
    ``node_type`` : ``"rearrangement"`` or ``"kmer"``
    ``name``      : ``junction_aa`` for rearrangements; decoded kmer sequence
                    for kmers.
    ``r_id``      : :attr:`Rearrangement.id` for rearrangement vertices;
                    ``-1`` for kmer vertices.
    ``v_gene``    : ``v_gene`` field (rearrangements) or kmer v-gene annotation.
    ``c_gene``    : ``c_gene`` field (rearrangements) or kmer c-gene annotation.
    ``locus``     : locus field for both vertex types.

    Args:
        rearrangements: Full list of rearrangements.  All are included as
            vertices even if they have no edges in the (filtered) token table.
        token_table: Output of :func:`~mir.basic.token_tables.tokenize_rearrangements`,
            optionally pre-filtered by
            :func:`~mir.basic.token_tables.filter_token_table`.

    Returns:
        Undirected bipartite ``igraph.Graph``.
    """
    n_r = len(rearrangements)
    r_id_to_idx = {r.id: i for i, r in enumerate(rearrangements)}

    kmers = list(token_table.keys())
    n_k = len(kmers)

    # Build edge set — deduplicate (r, k) pairs that appear at multiple positions
    edge_set: set[tuple[int, int]] = set()
    for ki, (_, matches) in enumerate(token_table.items()):
        kv = n_r + ki
        for match in matches:
            ri = r_id_to_idx.get(match.rearrangement.id)
            if ri is not None:
                edge_set.add((ri, kv))

    g = ig.Graph(n=n_r + n_k, directed=False)
    g.vs["node_type"] = ["rearrangement"] * n_r + ["kmer"] * n_k
    g.vs["name"] = (
        [r.junction_aa for r in rearrangements]
        + [k.seq.decode("ascii") for k in kmers]
    )
    g.vs["r_id"] = [r.id for r in rearrangements] + [-1] * n_k
    g.vs["v_gene"] = (
        [r.v_gene for r in rearrangements] + [k.v_gene for k in kmers]
    )
    g.vs["c_gene"] = (
        [r.c_gene for r in rearrangements] + [k.c_gene for k in kmers]
    )
    g.vs["locus"] = (
        [r.locus for r in rearrangements] + [k.locus for k in kmers]
    )
    if edge_set:
        g.add_edges(list(edge_set))
    return g
