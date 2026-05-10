"""Shared helpers for tcrtrie-backed neighborhood and graph search."""

from __future__ import annotations

import typing as t

from mir.graph.distance_utils import is_within_threshold


_TRIE_LONG_QUERY_AUGMENT_THRESHOLD: dict[str, int] = {
    "levenshtein": 33,
    "hamming": 64,
}


def validate_metric(metric: str) -> None:
    """Validate supported edit-distance metric names."""
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")


def search_limits(metric: str, threshold: int) -> tuple[int, int, int, int]:
    """Map metric/threshold to tcrtrie edit constraints."""
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold!r}")
    if metric == "hamming":
        return threshold, 0, 0, threshold
    return threshold, threshold, threshold, threshold


def resolve_n_jobs(*, n_jobs: int | None, nproc: int | None, default: int = 4) -> int:
    """Resolve worker count while keeping backward compatibility with ``nproc``."""
    resolved = default if n_jobs is None else n_jobs
    if n_jobs is None and nproc is not None:
        resolved = nproc
    if resolved <= 0:
        raise ValueError(f"n_jobs must be >= 1, got {resolved!r}")
    return resolved


def hit_index(hit: t.Any) -> int:
    """Extract sequence index from a tcrtrie hit payload."""
    if isinstance(hit, int):
        return hit
    if isinstance(hit, tuple | list):
        if not hit:
            raise ValueError("empty trie hit")
        return int(hit[0])
    if hasattr(hit, "index"):
        return int(hit.index)
    raise TypeError(f"Unsupported trie hit payload: {type(hit).__name__}")


def _bruteforce_search_indices(
    *,
    query: str,
    metric: str,
    threshold: int,
    sequences: list[str],
    v_gene_filter: str | None,
    j_gene_filter: str | None,
    v_genes: list[str] | None,
    j_genes: list[str] | None,
) -> list[int]:
    """Fallback candidate search with strict metric and gene filtering."""
    out: list[int] = []
    for idx, seq in enumerate(sequences):
        if v_gene_filter is not None:
            if v_genes is None or idx >= len(v_genes) or v_genes[idx] != v_gene_filter:
                continue
        if j_gene_filter is not None:
            if j_genes is None or idx >= len(j_genes) or j_genes[idx] != j_gene_filter:
                continue
        if is_within_threshold(query, seq, metric, threshold):
            out.append(idx)
    return out


def search_indices_with_fallback(
    trie: t.Any,
    *,
    query: str,
    metric: str,
    threshold: int,
    sequences: list[str],
    v_gene_filter: str | None = None,
    j_gene_filter: str | None = None,
    v_genes: list[str] | None = None,
    j_genes: list[str] | None = None,
) -> list[int]:
    """Search with tcrtrie, falling back to constrained brute-force on errors.

    The fallback preserves metric semantics by enforcing:
    - hamming: equal sequence lengths only
    - levenshtein: candidate length difference <= threshold
    """
    validate_metric(metric)
    max_substitution, max_insertion, max_deletion, max_edits = search_limits(metric, threshold)
    try:
        hits = trie.SearchIndices(
            query=query,
            maxSubstitution=max_substitution,
            maxInsertion=max_insertion,
            maxDeletion=max_deletion,
            maxEdits=max_edits,
            vGeneFilter=v_gene_filter,
            jGeneFilter=j_gene_filter,
        )
        indices = [hit_index(hit) for hit in hits]
    except Exception:
        return _bruteforce_search_indices(
            query=query,
            metric=metric,
            threshold=threshold,
            sequences=sequences,
            v_gene_filter=v_gene_filter,
            j_gene_filter=j_gene_filter,
            v_genes=v_genes,
            j_genes=j_genes,
        )

    # Keep only valid indices and exact metric-threshold matches.
    validated: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= len(sequences):
            continue
        if v_gene_filter is not None:
            if v_genes is None or idx >= len(v_genes) or v_genes[idx] != v_gene_filter:
                continue
        if j_gene_filter is not None:
            if j_genes is None or idx >= len(j_genes) or j_genes[idx] != j_gene_filter:
                continue
        if is_within_threshold(query, sequences[idx], metric, threshold):
            validated.append(idx)
    # For long queries, augment trie hits with exact brute-force to avoid
    # false negatives in bit-parallel kernels while preserving trie-first flow.
    length_limit = _TRIE_LONG_QUERY_AUGMENT_THRESHOLD[metric]
    if len(query) > length_limit:
        brute = _bruteforce_search_indices(
            query=query,
            metric=metric,
            threshold=threshold,
            sequences=sequences,
            v_gene_filter=v_gene_filter,
            j_gene_filter=j_gene_filter,
            v_genes=v_genes,
            j_genes=j_genes,
        )
        return sorted(set(validated).union(brute))
    return validated
