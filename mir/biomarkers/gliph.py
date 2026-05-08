"""GLIPH-style k-mer token artifact extraction for TCR repertoires.

Provides:
- :class:`GliphTokenArtifacts` — counts and bipartite adjacency for one token family.
- :func:`rows_to_clonotypes` — fast DataFrame-to-Clonotype conversion.
- :func:`deduplicate_clonotype_rows` — aggregate repeated clonotype rows.
- :func:`extract_v3mer_artifacts` — V-gene anchored 3-mer extraction.
- :func:`extract_pos3mer_artifacts` — V-gene + junction position + 3-mer extraction.
- :func:`extract_vpos3mer_artifacts` — deprecated alias for :func:`extract_pos3mer_artifacts`.
- :func:`extract_u4mer_artifacts` — ungapped 4-mer extraction.
- :func:`extract_g4mer_artifacts` — gapped 4-mer extraction.
- :func:`extract_g5mer_artifacts` — gapped 5-mer extraction.
- :func:`combine_enriched_token_maps` — merge enriched token neighborhoods across families.
- :func:`build_full_gliph_clonotype_graph` — build a combined k-mer/Hamming clonotype graph.
- :func:`normalize_control_v` — resample control to match sample unweighted V usage.
- :func:`normalize_control_vj` — resample control to match sample unweighted VJ usage.

Threaded tokenisation
---------------------
The extraction helpers accept a ``threads`` argument. When ``threads > 1`` the
input DataFrame is split into chunks and processed via
:class:`concurrent.futures.ThreadPoolExecutor`.  The underlying tokeniser spends
most of its time in the C-extension, so threads retain a lightweight API while
keeping naming consistent with the rest of the codebase.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal
import warnings

import igraph as ig
import numpy as np
import pandas as pd

from mir.basic.token_tables import tokenize_rearrangements
from mir.common.clonotype import Clonotype


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class GliphTokenArtifacts:
    """Token counts and bidirectional clonotype ↔ token mappings.

    Attributes
    ----------
    counts : dict[str, int]
        Active token count dictionary used for enrichment denominators.
    token_to_clone : dict[str, set[str]]
        Maps each token string to the set of clonotype row-IDs containing it.
    clone_to_tokens : dict[str, set[str]]
        Maps each clonotype row-ID to the set of tokens it contains.
    occurrence_counts : dict[str, int]
        Raw token occurrence counts across all windows.
    clonotype_counts : dict[str, int]
        Number of unique clonotypes containing each token.
    count_mode : {"occurrence", "clonotype"}
        Which count dictionary populates ``counts``.
    """

    counts: dict[str, int]
    token_to_clone: dict[str, set[str]]
    clone_to_tokens: dict[str, set[str]]
    occurrence_counts: dict[str, int]
    clonotype_counts: dict[str, int]
    count_mode: Literal["occurrence", "clonotype"] = "clonotype"


TOKEN_FAMILY = Literal["v3", "pos3", "u4", "g4", "g5", "vpos3"]
COUNT_MODE = Literal["occurrence", "clonotype"]

_DEFAULT_UNIQUE_CLONOTYPE_COLUMNS = (
    "reference_id",
    "junction_aa",
    "v_gene",
    "j_gene",
)


# ---------------------------------------------------------------------------
# Row → Clonotype conversion
# ---------------------------------------------------------------------------


def rows_to_clonotypes(df: pd.DataFrame) -> list[Clonotype]:
    """Convert a DataFrame of AIRR-schema rows to :class:`Clonotype` objects.

    Uses column-level list extraction (not :meth:`DataFrame.iterrows`) for
    substantially faster conversion on large tables.

    Required columns
    ----------------
    ``row_id``, ``junction_aa``, ``v_gene``, ``duplicate_count``.
    Optional: ``j_gene`` (defaults to ``""`` if absent).
    """
    row_ids = df["row_id"].tolist()
    jaa = df["junction_aa"].tolist()
    vg = df["v_gene"].tolist()
    jg = df["j_gene"].tolist() if "j_gene" in df.columns else [""] * len(df)
    dc = df["duplicate_count"].tolist()
    return [
        Clonotype(
            sequence_id=str(rid),
            locus="TRB",
            junction_aa=str(jaa_i),
            v_gene=str(vg_i),
            j_gene=str(jg_i),
            duplicate_count=int(dc_i),
            _validate=False,
        )
        for rid, jaa_i, vg_i, jg_i, dc_i in zip(row_ids, jaa, vg, jg, dc)
    ]


def _first_nonempty(series: pd.Series):
    """Return the first non-empty value from a grouped series."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    if non_null.dtype == object:
        stripped = non_null.astype(str).str.strip()
        valid = stripped[~stripped.str.lower().isin({"", "nan", "none", "na"})]
        if not valid.empty:
            return valid.iloc[0]
    return non_null.iloc[0]


def deduplicate_clonotype_rows(
    df: pd.DataFrame,
    *,
    subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    duplicate_count_col: str = "duplicate_count",
) -> pd.DataFrame:
    """Aggregate repeated clonotype rows to one row per unique clonotype.

    ``duplicate_count`` values are summed; all remaining metadata columns keep
    the first non-empty value observed within each group.
    """
    group_cols = [col for col in subset if col in df.columns]
    if not group_cols:
        return df.copy()

    agg: dict[str, str | callable] = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if col == duplicate_count_col:
            agg[col] = "sum"
        else:
            agg[col] = _first_nonempty

    dedup = df.groupby(group_cols, sort=False, dropna=False, as_index=False).agg(agg)
    if duplicate_count_col not in dedup.columns:
        dedup[duplicate_count_col] = 1
    dedup[duplicate_count_col] = (
        pd.to_numeric(dedup[duplicate_count_col], errors="coerce").fillna(1).astype(int)
    )
    dedup = dedup.reset_index(drop=True)
    dedup["row_id"] = dedup.index.astype(str)
    return dedup


# ---------------------------------------------------------------------------
# Internal builders (pure Python/C, no closures — needed for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _token_table_for_family(
    clones: list[Clonotype],
    family: TOKEN_FAMILY,
):
    if family == "vpos3":
        family = "pos3"
    if family == "v3":
        return tokenize_rearrangements(clones, k=3, mask_byte=None)
    if family == "pos3":
        return tokenize_rearrangements(clones, k=3, mask_byte=None)
    if family == "u4":
        return tokenize_rearrangements(clones, k=4, mask_byte=None)
    if family == "g4":
        return tokenize_rearrangements(clones, k=4, mask_byte=ord("X"))
    if family == "g5":
        return tokenize_rearrangements(clones, k=5, mask_byte=ord("X"))
    raise ValueError(f"Unknown GLIPH token family: {family}")


def _token_from_match(family: TOKEN_FAMILY, kmer, match) -> str:
    if family == "vpos3":
        family = "pos3"
    seq = kmer.seq.decode("ascii")
    v_base = (kmer.v_gene or "").split("*")[0]
    if family == "v3":
        return f"v3::{v_base}::{seq}"
    if family == "pos3":
        return f"pos3::{v_base}::{match.position}::{seq}"
    if family == "u4":
        return f"u4::{seq}"
    if family == "g4":
        return f"g4::{seq}"
    if family == "g5":
        return f"g5::{seq}"
    raise ValueError(f"Unknown GLIPH token family: {family}")


def _build_artifacts_from_clones(
    clones: list[Clonotype],
    family: TOKEN_FAMILY,
    count_mode: COUNT_MODE,
) -> GliphTokenArtifacts:
    """Build token artifacts for one family from a clonotype list."""
    token_table = _token_table_for_family(clones, family)

    occurrence_counts: Counter[str] = Counter()
    token_to_clone: dict[str, set[str]] = defaultdict(set)
    clone_to_tokens: dict[str, set[str]] = defaultdict(set)

    for kmer, matches in token_table.items():
        for match in matches:
            token = _token_from_match(family, kmer, match)
            rid = str(match.rearrangement.id)
            occurrence_counts[token] += 1
            token_to_clone[token].add(rid)
            clone_to_tokens[rid].add(token)

    clonotype_counts = {token: len(cloneset) for token, cloneset in token_to_clone.items()}
    counts = dict(clonotype_counts if count_mode == "clonotype" else occurrence_counts)

    return GliphTokenArtifacts(
        counts=counts,
        token_to_clone=dict(token_to_clone),
        clone_to_tokens=dict(clone_to_tokens),
        occurrence_counts=dict(occurrence_counts),
        clonotype_counts=clonotype_counts,
        count_mode=count_mode,
    )


# ---------------------------------------------------------------------------
# Process-pool worker functions — must be importable at module level
# ---------------------------------------------------------------------------


def _worker_extract(
    chunk_df: pd.DataFrame,
    family: TOKEN_FAMILY,
    count_mode: COUNT_MODE,
) -> GliphTokenArtifacts:
    """ThreadPoolExecutor worker for GLIPH token extraction."""
    return _build_artifacts_from_clones(rows_to_clonotypes(chunk_df), family=family, count_mode=count_mode)


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def _merge_artifact_parts(
    parts: list[GliphTokenArtifacts],
) -> GliphTokenArtifacts:
    """Merge partial artifact objects produced by threaded workers."""
    if not parts:
        return GliphTokenArtifacts(
            counts={},
            token_to_clone={},
            clone_to_tokens={},
            occurrence_counts={},
            clonotype_counts={},
            count_mode="clonotype",
        )

    count_mode = parts[0].count_mode
    merged_occurrences: Counter[str] = Counter()
    merged_t2c: dict[str, set[str]] = defaultdict(set)
    merged_c2t: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        merged_occurrences.update(part.occurrence_counts)
        for tok, rids in part.token_to_clone.items():
            merged_t2c[tok].update(rids)
        for rid, toks in part.clone_to_tokens.items():
            merged_c2t[rid].update(toks)

    merged_clonotype_counts = {
        token: len(rids) for token, rids in merged_t2c.items()
    }
    merged_counts = (
        dict(merged_clonotype_counts)
        if count_mode == "clonotype"
        else dict(merged_occurrences)
    )
    return GliphTokenArtifacts(
        counts=merged_counts,
        token_to_clone=dict(merged_t2c),
        clone_to_tokens=dict(merged_c2t),
        occurrence_counts=dict(merged_occurrences),
        clonotype_counts=merged_clonotype_counts,
        count_mode=count_mode,
    )


def _split_dataframe(df: pd.DataFrame, threads: int) -> list[pd.DataFrame]:
    """Split a DataFrame into row-wise chunks while preserving DataFrame type."""
    if threads <= 1 or len(df) == 0:
        return [df]

    boundaries = np.linspace(0, len(df), num=threads + 1, dtype=int)
    chunks: list[pd.DataFrame] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start == stop:
            continue
        chunks.append(df.iloc[start:stop].reset_index(drop=True))
    return chunks


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------


def extract_gliph_token_artifacts(
    df: pd.DataFrame,
    family: TOKEN_FAMILY,
    *,
    threads: int = 1,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract GLIPH-style token artifacts for one token family.

    Parameters
    ----------
    df : pd.DataFrame
        Clonotype table with columns ``row_id``, ``junction_aa``, ``v_gene``,
        ``j_gene`` (optional), ``duplicate_count``.
    family : {"v3", "pos3", "u4", "g4", "g5"}
        Token family to extract.
    threads : int, optional
        Number of worker threads (default ``1``).
    count_mode : {"occurrence", "clonotype"}, optional
        Whether enrichment counts should reflect raw token occurrences or token
        presence per unique clonotype (default ``"clonotype"``).
    unique_clonotypes : bool, optional
        When ``True``, aggregate repeated clonotype rows before tokenisation.
    unique_subset : tuple[str, ...], optional
        Columns defining clonotype uniqueness when ``unique_clonotypes=True``.
    n_workers : int, optional
        Backward-compatible alias for ``threads``.
    """
    if n_workers is not None:
        warnings.warn(
            "n_workers is deprecated for GLIPH extraction; use threads instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        threads = n_workers

    if unique_clonotypes:
        df = deduplicate_clonotype_rows(df, subset=unique_subset)
    else:
        df = df.copy()

    if threads <= 1:
        return _worker_extract(df, family=family, count_mode=count_mode)

    chunks = _split_dataframe(df.reset_index(drop=True), threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [
            pool.submit(_worker_extract, chunk, family, count_mode)
            for chunk in chunks
        ]
        parts = [future.result() for future in futures]
    return _merge_artifact_parts(parts)


def extract_v3mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract V-gene anchored 3-mer (V+3-mer) token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="v3",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        n_workers=n_workers,
    )


def extract_vpos3mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Deprecated alias for :func:`extract_pos3mer_artifacts`."""
    warnings.warn(
        "extract_vpos3mer_artifacts is deprecated; use extract_pos3mer_artifacts.",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_pos3mer_artifacts(
        df,
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        n_workers=n_workers,
    )


def extract_pos3mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract pos+3-mer token artifacts (V-gene + junction position + 3-mer)."""
    return extract_gliph_token_artifacts(
        df,
        family="pos3",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        n_workers=n_workers,
    )


def extract_u4mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract ungapped 4-mer token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="u4",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        n_workers=n_workers,
    )


def extract_g4mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract gapped 4-mer token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="g4",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        n_workers=n_workers,
    )


def extract_g5mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract gapped 5-mer token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="g5",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        n_workers=n_workers,
    )


# ---------------------------------------------------------------------------
# Full-GLIPH graph helpers (k-mer enrichment + hamming expansion)
# ---------------------------------------------------------------------------


def combine_enriched_token_maps(
    artifacts_by_family: dict[str, GliphTokenArtifacts],
    enriched_tokens_by_family: dict[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Merge enriched token neighborhoods across token families.

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
    study_df: pd.DataFrame,
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

    Returns
    -------
    tuple
        ``(full_clone_graph, clone_to_tokens_expanded, hamming_graph)``.
    """
    # Import lazily to keep tokenization utilities usable without optional trie deps.
    from mir.graph.edit_distance_graph import build_edit_distance_graph

    all_clones = rows_to_clonotypes(study_df)
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


# ---------------------------------------------------------------------------
# Gene-usage-normalised control resampling
# ---------------------------------------------------------------------------


def _normalize_gene_usage_series(df: pd.DataFrame, gene_col: str) -> pd.Series:
    return df[gene_col].fillna("").astype(str).str.strip().str.split("*").str[0]


def _normalize_control_by_gene_columns(
    sample_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    n: int,
    *,
    gene_columns: tuple[str, ...],
    seed: int = 42,
) -> pd.DataFrame:
    """Resample *control_pool_df* to match unweighted gene usage in *sample_df*.

    **Unweighted** means every clonotype (row) counts once regardless of its
    ``duplicate_count``.  The returned rows all receive ``duplicate_count=1``.

    Algorithm
    ---------
     1. Compute the frequency of the requested base-gene columns in *sample_df*
         (one count per row).
     2. For each gene-usage bucket, sample from the matching rows in
         *control_pool_df* proportionally; use replacement when the control has
         fewer rows than needed.
     3. If a bucket present in the sample is absent from the control, those slots
         are back-filled from the global control at random.
    4. Truncate / pad to exactly *n* rows.

    Parameters
    ----------
    sample_df, control_pool_df : pd.DataFrame
        Must contain the requested gene columns (alleles are stripped).
    n : int
        Target number of control clonotypes after resampling.
    gene_columns : tuple[str, ...]
        Columns whose stripped base-gene usage should be matched.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Resampled control rows with ``duplicate_count=1`` and freshly assigned
        ``row_id`` values (``ctrl_0``, ``ctrl_1``, …).
    """
    rng = np.random.default_rng(seed)

    if not gene_columns:
        raise ValueError("gene_columns must not be empty")

    sample_keys = pd.DataFrame(
        {
            f"g{i}": _normalize_gene_usage_series(sample_df, col)
            for i, col in enumerate(gene_columns)
        }
    )
    for col in sample_keys.columns:
        sample_keys = sample_keys[sample_keys[col] != ""]
    key_counts = sample_keys.groupby(list(sample_keys.columns)).size()
    key_freq = key_counts / float(key_counts.sum())

    # --- Group control by stripped gene-usage key ---
    ctrl = control_pool_df.copy()
    key_cols = []
    for i, col in enumerate(gene_columns):
        key_col = f"_g{i}"
        key_cols.append(key_col)
        ctrl[key_col] = _normalize_gene_usage_series(ctrl, col)
        ctrl = ctrl[ctrl[key_col] != ""]
    ctrl_groups = {
        key: grp for key, grp in ctrl.groupby(key_cols, sort=False)
    }

    # --- Sample per gene-usage bucket ---
    sampled: list[pd.DataFrame] = []
    for key, freq in key_freq.items():
        if not isinstance(key, tuple):
            key = (key,)
        n_target = max(1, round(n * float(freq)))
        grp = ctrl_groups.get(tuple(key))
        if grp is None or len(grp) == 0:
            continue
        replace = n_target > len(grp)
        n_draw = n_target if replace else min(n_target, len(grp))
        samp = grp.sample(n=n_draw, replace=replace, random_state=int(rng.integers(0, 2**31)))
        sampled.append(samp)

    if not sampled:
        # Fallback: pure random sample from whole control
        result = control_pool_df.sample(
            n=min(n, len(control_pool_df)), replace=n > len(control_pool_df),
            random_state=seed,
        )
    else:
        result = pd.concat(sampled, ignore_index=True)
        # Adjust to exactly n rows
        if len(result) > n:
            result = result.sample(n=n, random_state=seed, replace=False)
        elif len(result) < n:
            extra = control_pool_df.sample(
                n=n - len(result), replace=True, random_state=seed,
            )
            result = pd.concat([result, extra], ignore_index=True)

    result = result.drop(columns=key_cols, errors="ignore").reset_index(drop=True)
    result["duplicate_count"] = 1
    result["row_id"] = ["ctrl_" + str(i) for i in range(len(result))]
    return result


def normalize_control_v(
    sample_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    n: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample *control_pool_df* to match the unweighted V usage of *sample_df*."""
    return _normalize_control_by_gene_columns(
        sample_df,
        control_pool_df,
        n,
        gene_columns=("v_gene",),
        seed=seed,
    )


def normalize_control_vj(
    sample_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    n: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample *control_pool_df* to match the unweighted VJ usage of *sample_df*."""
    return _normalize_control_by_gene_columns(
        sample_df,
        control_pool_df,
        n,
        gene_columns=("v_gene", "j_gene"),
        seed=seed,
    )
