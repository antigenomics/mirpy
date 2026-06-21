"""Shared helpers for seqtree-backed neighborhood and graph search."""

from __future__ import annotations

import logging

import seqtree

_logger = logging.getLogger(__name__)

# seqtree's "aa" alphabet also accepts B/Z/X/*; restricting the index to the 20
# standard amino acids is a safe superset filter — anything seqtree would reject
# (``_``, ``.``, …) is excluded, and the rare ``*``/X/B/Z sequences simply fall
# through to the brute-force path with identical metric results.
_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
_NONCANONICAL_WARNING_EMITTED = False


def _is_trie_safe(seq: str) -> bool:
    """Return True if every character is a standard amino acid."""
    return bool(seq) and all(c in _STANDARD_AA for c in seq.upper())


def make_index(seqs: list[str]) -> tuple:
    """Build a seqtree :class:`~seqtree.Index` over canonical sequences only.

    Returns:
        Tuple of ``(index, idx_to_orig)`` where ``idx_to_orig[ref_id]`` maps a
        seqtree ref_id back to the original-list position.  Non-canonical
        sequences (``*``, ``_``, or non-standard chars) are excluded; ``index``
        is ``None`` when no canonical sequence remains.
    """
    global _NONCANONICAL_WARNING_EMITTED

    idx_to_orig = [i for i, s in enumerate(seqs) if _is_trie_safe(s)]
    n_skip = len(seqs) - len(idx_to_orig)
    if n_skip and not _NONCANONICAL_WARNING_EMITTED:
        _logger.warning(
            "Skipping %d sequences with non-canonical amino acids (*, _, or non-standard chars)",
            n_skip,
        )
        _NONCANONICAL_WARNING_EMITTED = True

    if not idx_to_orig:
        return None, idx_to_orig
    canon = [seqs[i] for i in idx_to_orig]
    return seqtree.Index.build(canon, alphabet="aa"), idx_to_orig


def validate_metric(metric: str) -> None:
    """Validate supported edit-distance metric names."""
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")


def make_params(metric: str, threshold: int) -> "seqtree.SearchParams":
    """Map a metric/threshold to a seqtree ``SearchParams`` (seqtm engine).

    Hamming forbids indels (equal-length, substitutions only); Levenshtein
    allows substitutions and indels, all bounded by ``threshold``.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold!r}")
    if metric == "hamming":
        return seqtree.SearchParams(
            max_subs=threshold, max_total_edits=threshold, engine="seqtm"
        )
    return seqtree.SearchParams(
        max_subs=threshold,
        max_ins=threshold,
        max_dels=threshold,
        max_total_edits=threshold,
        engine="seqtm",
    )


def resolve_n_jobs(*, n_jobs: int | None, nproc: int | None, default: int = 4) -> int:
    """Resolve worker count while keeping backward compatibility with ``nproc``."""
    resolved = default if n_jobs is None else n_jobs
    if n_jobs is None and nproc is not None:
        resolved = nproc
    if resolved <= 0:
        raise ValueError(f"n_jobs must be >= 1, got {resolved!r}")
    return resolved


def search_canon_indices(index, query: str, params) -> list[int]:
    """Return seqtree ref_ids (canonical space) within scope for *query*.

    Empty when the index is empty or the query is non-canonical (which seqtree's
    ``aa`` alphabet would reject).  seqtm guarantees results already satisfy the
    metric/threshold, so callers need no further distance re-check.
    """
    if index is None or not _is_trie_safe(query):
        return []
    return [hit.ref_id for hit in index.search(query, params)]
