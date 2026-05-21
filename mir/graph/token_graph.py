"""Bipartite token graph construction for clonotypes and tokens.

Builds an ``igraph.Graph`` with two vertex types:

* **clonotype** — one vertex per :class:`~mir.basic.token_tables.Clonotype`.
* **token** — one vertex per unique :class:`~mir.basic.token_tables.Kmer` key
  present in the token table.

Edges connect each clonotype vertex to every token vertex whose sequence
appears in that clonotype's ``junction_aa``, as recorded by the token
table produced by :func:`~mir.basic.token_tables.tokenize_clonotypes`.

Use :func:`~mir.basic.token_tables.filter_token_table` to restrict the
token table before building the graph — by regex pattern, minimum
clonotype count, or both.

GLIPH-style graph construction
-------------------------------
- :func:`combine_enriched_token_maps` — merge enriched token neighborhoods across families.
- :func:`build_full_gliph_clonotype_graph` — build combined k-mer/Hamming clonotype graph.
- :func:`build_kmer_projection_graph` — project token co-occurrence graph.
"""

from __future__ import annotations

from collections import defaultdict

import igraph as ig
import polars as pl

from mir.basic.token_tables import Kmer, KmerMatch, Clonotype
from mir.common.metaclonotype import MetaClonotypeDefinition
from mir.common.repertoire import LocusRepertoire
from mir.graph.edit_distance_graph import build_edit_distance_graph
from mir.utils.metaclonotype_clustering import metaclonotypes_from_graph_communities


def _locus_repertoire_from_dataframe(df: "pl.DataFrame | pd.DataFrame", *, locus: str = "TRB") -> LocusRepertoire:
    """Build a LocusRepertoire from a Polars or pandas DataFrame."""
    if not isinstance(df, pl.DataFrame):
        import pandas as pd
        tmp = df.copy()
        if "sequence_id" not in tmp.columns and "row_id" in tmp.columns:
            tmp = tmp.rename(columns={"row_id": "sequence_id"})
        data = {col: [None if pd.isna(v) else v for v in tmp[col].tolist()] for col in tmp.columns}
        df = pl.DataFrame(data, strict=False)
    elif "sequence_id" not in df.columns and "row_id" in df.columns:
        df = df.rename({"row_id": "sequence_id"})
    return LocusRepertoire.from_polars(df, locus=locus)


def build_token_graph(
    rearrangements: list[Clonotype],
    token_table: dict[Kmer, list[KmerMatch]],
) -> ig.Graph:
    """Build a bipartite clonotype-token graph from a token table.

    Vertices 0 … n_r-1 represent the clonotypes (in list order).
    Vertices n_r … n_r+n_k-1 represent the unique tokens (in token-table
    insertion order).  An edge exists between clonotype *i* and token *j*
    when token *j* appears in clonotype *i*'s ``junction_aa``.  Parallel
    edges are deduplicated (a token may match a clonotype at multiple
    positions).

    Each vertex carries the following attributes:

    ``node_type``
        ``"rearrangement"`` or ``"kmer"`` (legacy naming retained for compatibility)
    ``name``
        ``junction_aa`` for clonotypes; decoded token sequence for token vertices.
    ``r_id``
        :attr:`Clonotype.id` for clonotype vertices; ``-1`` for token vertices.
    ``v_gene``
        ``v_gene`` field (clonotypes) or token v-gene annotation.
    ``c_gene``
        ``c_gene`` field (clonotypes) or token c-gene annotation.
    ``locus``
        locus field for both vertex types.

    Args:
        rearrangements: Full list of clonotypes.  All are included as
            vertices even if they have no edges in the (filtered) token table.
        token_table: Output of :func:`~mir.basic.token_tables.tokenize_clonotypes`,
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


# ---------------------------------------------------------------------------
# GLIPH-style graph construction (clonotype enrichment + Hamming neighbors)
# ---------------------------------------------------------------------------


def combine_enriched_token_maps(
    artifacts_by_family: dict[str, object],  # GliphTokenArtifacts
    enriched_tokens_by_family: dict[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Merge enriched token neighborhoods across token families.

    Parameters
    ----------
    artifacts_by_family : dict[str, GliphTokenArtifacts]
        Token artifacts indexed by family name (see :mod:`mir.biomarkers.gliph`).
    enriched_tokens_by_family : dict[str, set[str]]
        Enriched tokens selected per family.

    Returns
    -------
    tuple
        ``(token_to_clones, clone_to_tokens, token_family)`` where:
        - ``token_to_clones[token]`` is the clonotype-id set carrying ``token``;
        - ``clone_to_tokens[clone_id]`` are enriched tokens linked to the clonotype;
        - ``token_family[token]`` stores the source family key.
    """
    token_to_clones: dict[str, set[str]] = defaultdict(set)
    clone_to_tokens: dict[str, set[str]] = defaultdict(set)
    token_family: dict[str, str] = {}

    for family, artifacts in artifacts_by_family.items():
        tokens = enriched_tokens_by_family.get(family, set())
        for token in tokens:
            clone_ids = set(artifacts.token_to_clone.get(token, set()))
            if not clone_ids:
                continue
            token_to_clones[token].update(clone_ids)
            token_family[token] = family
            for clone_id in clone_ids:
                clone_to_tokens[clone_id].add(token)

    return dict(token_to_clones), dict(clone_to_tokens), token_family


def build_full_gliph_clonotype_graph(
    study_df: pl.DataFrame,
    token_to_clones: dict[str, set[str]],
    *,
    hamming_threshold: int = 1,
    hamming_threads: int = 4,
    expand_hamming_neighbors: bool = True,
    min_kmer_edge_weight: float = 0.35,
    hamming_bonus: float = 1.0,
) -> tuple[ig.Graph, dict[str, set[str]], ig.Graph]:
    """Build the combined GLIPH clonotype graph with Hamming expansion.

    The graph is built in three stages:

    1. Start from clonotypes linked to at least one enriched token.
    2. Add edges between clonotypes sharing enriched tokens.
    3. Add Hamming ``<= threshold`` edges, and (optionally) one-hop Hamming
       neighbors of already-active clonotypes.

    Parameters
    ----------
    study_df : polars.DataFrame
        Clonotype table with columns ``row_id`` (or ``sequence_id``), ``junction_aa``, ``v_gene``, ``duplicate_count``.
    token_to_clones : dict[str, set[str]]
        Token → clonotype-ID mapping (from :func:`combine_enriched_token_maps`).
    hamming_threshold : int, optional
        Hamming distance cutoff (default 1).
    hamming_threads : int, optional
        Threads for Hamming distance computation (default 4).
    expand_hamming_neighbors : bool, optional
        Include one-hop Hamming neighbors of enriched clonotypes (default True).
    min_kmer_edge_weight : float, optional
        Minimum normalized weight to retain k-mer edges (default 0.35).
    hamming_bonus : float, optional
        Weight bonus for Hamming edges (default 1.0).

    Returns
    -------
    tuple
        ``(full_clone_graph, clone_to_tokens_expanded, hamming_graph)`` where full_clone_graph
        contains both k-mer and Hamming edges with weights, and hamming_graph is the raw
        Hamming distance graph before filtering.
    """
    all_clones = _locus_repertoire_from_dataframe(study_df, locus="TRB").clonotypes
    hamming_graph = build_edit_distance_graph(
        all_clones,
        metric="hamming",
        threshold=hamming_threshold,
        n_jobs=hamming_threads,
    )

    all_clone_ids = [str(clone.id) for clone in all_clones]
    initial_active = set(str(clone_id) for clone_ids in token_to_clones.values() for clone_id in clone_ids)
    active = set(initial_active)

    # Add one hop of Hamming neighbors around the currently active set.
    if expand_hamming_neighbors and active and hamming_graph.vcount() > 0:
        id_to_idx = {str(rid): idx for idx, rid in enumerate(hamming_graph.vs["r_id"])}
        for clone_id in list(active):
            idx = id_to_idx.get(clone_id)
            if idx is None:
                continue
            for nbr in hamming_graph.neighbors(idx):
                active.add(str(hamming_graph.vs[nbr]["r_id"]))

    # Build shared-kmer edge counts and specificity-weighted contributions over active nodes.
    active_clone_nodes = sorted(active)
    clone_idx = {clone_id: i for i, clone_id in enumerate(active_clone_nodes)}
    edge_shared_kmers: dict[tuple[int, int], int] = defaultdict(int)
    edge_kmer_weight: dict[tuple[int, int], float] = defaultdict(float)
    for clone_ids in token_to_clones.values():
        present = sorted(set(str(clone_id) for clone_id in clone_ids if str(clone_id) in clone_idx))
        degree = len(present)
        if degree < 2:
            continue
        contribution = 1.0 / max(1.0, float(degree - 1))
        for left_i in range(len(present) - 1):
            left = present[left_i]
            for right in present[left_i + 1 :]:
                edge = tuple(sorted((clone_idx[left], clone_idx[right])))
                edge_shared_kmers[edge] += 1
                edge_kmer_weight[edge] += contribution

    # Add hamming edges among active nodes.
    edge_hamming: set[tuple[int, int]] = set()
    if hamming_graph.vcount() > 0 and active_clone_nodes:
        id_to_local = {str(rid): clone_idx[str(rid)] for rid in active_clone_nodes if str(rid) in clone_idx}
        for edge in hamming_graph.es:
            source = str(hamming_graph.vs[edge.source]["r_id"])
            target = str(hamming_graph.vs[edge.target]["r_id"])
            if source not in id_to_local or target not in id_to_local:
                continue
            edge_hamming.add(tuple(sorted((id_to_local[source], id_to_local[target]))))

    keep_kmer_edges = {edge for edge, weight in edge_kmer_weight.items() if weight >= min_kmer_edge_weight}
    all_edges = sorted(keep_kmer_edges | edge_hamming)
    graph = ig.Graph(n=len(active_clone_nodes), directed=False)
    graph.vs["name"] = active_clone_nodes
    if all_edges:
        graph.add_edges(all_edges)
        graph.es["shared_kmers"] = [int(edge_shared_kmers.get(edge, 0)) for edge in all_edges]
        graph.es["kmer_weight"] = [float(edge_kmer_weight.get(edge, 0.0)) for edge in all_edges]
        graph.es["is_hamming"] = [edge in edge_hamming for edge in all_edges]
        graph.es["weight"] = [
            float(edge_kmer_weight.get(edge, 0.0)) + (hamming_bonus if edge in edge_hamming else 0.0)
            for edge in all_edges
        ]

    clone_to_tokens_expanded: dict[str, set[str]] = {
        clone_id: set() for clone_id in all_clone_ids if clone_id in active
    }
    for token, clone_ids in token_to_clones.items():
        for clone_id in clone_ids:
            clone_id = str(clone_id)
            if clone_id in clone_to_tokens_expanded:
                clone_to_tokens_expanded[clone_id].add(token)

    return graph, clone_to_tokens_expanded, hamming_graph


def build_kmer_projection_graph(
    token_to_clones: dict[str, set[str]],
) -> tuple[ig.Graph, dict[str, int]]:
    """Project token-clone bipartite links to a token co-occurrence graph.

    This is the one-mode projection (token side) of the underlying bipartite
    graph, where tokens are connected if at least one clonotype carries both.

    Parameters
    ----------
    token_to_clones : dict[str, set[str]]
        Token → clonotype-ID mapping.

    Returns
    -------
    tuple
        ``(graph, token_degree)`` where graph is an igraph.Graph with tokens as
        vertices and co-occurrence edges, and token_degree maps each token to
        the number of unique clonotypes carrying it.
    """
    tokens = sorted(token_to_clones)
    token_idx = {token: idx for idx, token in enumerate(tokens)}
    graph = ig.Graph(n=len(tokens), directed=False)
    graph.vs["name"] = tokens

    clone_to_tokens: dict[str, list[str]] = defaultdict(list)
    for token, clone_ids in token_to_clones.items():
        for clone_id in clone_ids:
            clone_to_tokens[str(clone_id)].append(token)

    edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    for token_list in clone_to_tokens.values():
        unique_tokens = sorted(set(token_list))
        if len(unique_tokens) < 2:
            continue
        for left_i in range(len(unique_tokens) - 1):
            left = unique_tokens[left_i]
            for right in unique_tokens[left_i + 1 :]:
                edge = tuple(sorted((token_idx[left], token_idx[right])))
                edge_weights[edge] += 1

    if edge_weights:
        edges = list(edge_weights.keys())
        graph.add_edges(edges)
        graph.es["weight"] = [float(edge_weights[edge]) for edge in edges]

    token_degree = {token: len(token_to_clones.get(token, set())) for token in tokens}
    return graph, token_degree


def metaclonotypes_from_token_clonotype_graph(
    clonotype_graph: ig.Graph,
    *,
    method: str = "components",
    min_cluster_size: int = 1,
    vertex_id_attr: str = "name",
) -> MetaClonotypeDefinition:
    """Convert token/GLIPH clonotype graph communities to metaclonotypes.

    Args:
        clonotype_graph: Clonotype-only graph (e.g. output from
            ``build_full_gliph_clonotype_graph``).
        method: ``components``, ``leiden``, or ``louvain``.
        min_cluster_size: Minimum retained cluster size.
        vertex_id_attr: Vertex attribute containing clonotype IDs.
    """
    return metaclonotypes_from_graph_communities(
        clonotype_graph,
        vertex_id_attr=vertex_id_attr,
        method=method,
        min_cluster_size=min_cluster_size,
        weights="weight",
    )


def build_gliph_metaclonotypes(
    study_df: pl.DataFrame,
    token_to_clones: dict[str, set[str]],
    *,
    method: str = "leiden",
    min_cluster_size: int = 2,
    hamming_threshold: int = 1,
    hamming_threads: int = 4,
    expand_hamming_neighbors: bool = True,
    min_kmer_edge_weight: float = 0.35,
    hamming_bonus: float = 1.0,
) -> MetaClonotypeDefinition:
    """Build GLIPH-style clonotype graph and return metaclonotypes.

    This wraps:
    1. ``build_full_gliph_clonotype_graph``
    2. community detection (components/Leiden/Louvain)
    3. conversion to metaclonotype membership table.
    """
    clone_graph, _clone_tokens, _ham_graph = build_full_gliph_clonotype_graph(
        study_df,
        token_to_clones,
        hamming_threshold=hamming_threshold,
        hamming_threads=hamming_threads,
        expand_hamming_neighbors=expand_hamming_neighbors,
        min_kmer_edge_weight=min_kmer_edge_weight,
        hamming_bonus=hamming_bonus,
    )
    return metaclonotypes_from_token_clonotype_graph(
        clone_graph,
        method=method,
        min_cluster_size=min_cluster_size,
        vertex_id_attr="name",
    )
