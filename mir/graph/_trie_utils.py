"""Shared helpers for tcrtrie-backed neighborhood and graph search."""

from __future__ import annotations

import logging
import typing as t

from tcrtrie import Trie
from mir.graph.distance_utils import is_within_threshold

_logger = logging.getLogger(__name__)

# Standard 20 amino acids — sequences containing other characters (*, _, B, …)
# cause tcrtrie to emit per-sequence warnings to stdout (compiled C++ code).
# Pre-filtering to "" silences those warnings; the empty string is never matched.
_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_TRIE_WARNING_EMITTED = False


def _is_trie_safe(seq: str) -> bool:
    """Return True if every character is a standard amino acid."""
    return bool(seq) and all(c in _STANDARD_AA for c in seq.upper())


def make_trie(seqs: list[str], v_calls: list[str], j_calls: list[str]) -> tuple:
    """Build a tcrtrie Trie with only canonical sequences.

    Non-canonical sequences (containing *, _, or non-standard AA chars) are
    excluded from the trie index.  The compiled tcrtrie extension emits
    per-sequence prints to C-level stdout; excluding them entirely prevents
    those prints.  A single summary is emitted via ``logging.warning``.

    Returns:
        Tuple of ``(trie, trie_to_orig)`` where ``trie_to_orig[trie_idx]``
        gives the corresponding original-list index.  Non-canonical entries
        have no trie index and produce no edges.
    """
    from tcrtrie import Trie

    global _TRIE_WARNING_EMITTED

    trie_to_orig = [i for i, s in enumerate(seqs) if _is_trie_safe(s)]
    n_skip = len(seqs) - len(trie_to_orig)
    if n_skip and not _TRIE_WARNING_EMITTED:
        _logger.warning(
            "Skipping %d sequences with non-canonical amino acids (*, _, or non-standard chars)",
            n_skip,
        )
        _TRIE_WARNING_EMITTED = True

    if trie_to_orig:
        canon_seqs = [seqs[i] for i in trie_to_orig]
        canon_v    = [v_calls[i] for i in trie_to_orig]
        canon_j    = [j_calls[i] for i in trie_to_orig]
    else:
        canon_seqs = canon_v = canon_j = []

    return Trie(sequences=canon_seqs, vGenes=canon_v, jGenes=canon_j), trie_to_orig


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
    v_call_filter: str | None,
    j_call_filter: str | None,
    v_calls: list[str] | None,
    j_calls: list[str] | None,
) -> list[int]:
    """Fallback candidate search with strict metric and gene filtering."""
    out: list[int] = []
    for idx, seq in enumerate(sequences):
        if v_call_filter is not None:
            if v_calls is None or idx >= len(v_calls) or v_calls[idx] != v_call_filter:
                continue
        if j_call_filter is not None:
            if j_calls is None or idx >= len(j_calls) or j_calls[idx] != j_call_filter:
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
    v_call_filter: str | None = None,
    j_call_filter: str | None = None,
    v_calls: list[str] | None = None,
    j_calls: list[str] | None = None,
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
            vGeneFilter=v_call_filter,
            jGeneFilter=j_call_filter,
        )
        indices = [hit_index(hit) for hit in hits]
    except Exception:
        return _bruteforce_search_indices(
            query=query,
            metric=metric,
            threshold=threshold,
            sequences=sequences,
            v_call_filter=v_call_filter,
            j_call_filter=j_call_filter,
            v_calls=v_calls,
            j_calls=j_calls,
        )

    # Keep only valid indices and exact metric-threshold matches.
    validated: list[int] = []
    for idx in indices:
        if idx < 0 or idx >= len(sequences):
            continue
        if v_call_filter is not None:
            if v_calls is None or idx >= len(v_calls) or v_calls[idx] != v_call_filter:
                continue
        if j_call_filter is not None:
            if j_calls is None or idx >= len(j_calls) or j_calls[idx] != j_call_filter:
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
            v_call_filter=v_call_filter,
            j_call_filter=j_call_filter,
            v_calls=v_calls,
            j_calls=j_calls,
        )
        return sorted(set(validated).union(brute))
    return validated
