"""Shared distance computation utilities for graph construction and neighborhood analysis.

Provides reusable pair-distance filtering logic to avoid duplication between
:mod:`edit_distance_graph` and :mod:`neighborhood_enrichment`.
"""

from __future__ import annotations

from typing import NamedTuple

from mir.distances.seqdist import hamming as _hamming
from mir.distances.seqdist import levenshtein as _levenshtein


class PairRecord(NamedTuple):
    """Pair of rearrangements to compare, with pre-extracted fields."""

    i: int
    j: int
    seq1: str
    seq2: str
    v1: str
    v2: str
    j1: str
    j2: str


def compute_distance(seq1: str, seq2: str, metric: str) -> int:
    """Compute distance between two sequences.

    Parameters
    ----------
    seq1
        First sequence.
    seq2
        Second sequence.
    metric
        Distance metric: ``"hamming"`` or ``"levenshtein"``.

    Returns
    -------
    int
        Distance between sequences.

    Raises
    ------
    ValueError
        If metric is not supported.
    """
    if metric == "hamming":
        if len(seq1) != len(seq2):
            return 10000  # Large number for unequal-length pairs (no edge for hamming)
        return _hamming(seq1, seq2)
    elif metric == "levenshtein":
        return _levenshtein(seq1, seq2)
    else:
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")


def is_length_compatible(seq1: str, seq2: str, metric: str, threshold: int) -> bool:
    """Check if pair lengths can satisfy the metric/threshold constraints.

    For Hamming distance, only equal-length sequences are comparable.
    For Levenshtein distance, a necessary condition for ``d <= threshold`` is
    ``abs(len(seq1) - len(seq2)) <= threshold``.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold!r}")
    if metric == "hamming":
        return len(seq1) == len(seq2)
    if metric == "levenshtein":
        return abs(len(seq1) - len(seq2)) <= threshold
    raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")


def is_within_threshold(seq1: str, seq2: str, metric: str, threshold: int) -> bool:
    """Return True when pair distance is within threshold.

    Applies cheap length prefilters before invoking C-backed kernels.
    """
    if not is_length_compatible(seq1, seq2, metric, threshold):
        return False
    return compute_distance(seq1, seq2, metric) <= threshold


def should_compare_pair(
    rec: PairRecord,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
) -> bool:
    """Check if a pair should be compared based on gene matching criteria.

    Parameters
    ----------
    rec
        Pair record with extracted fields.
    match_v_gene
        If True, only compare pairs with matching v_gene.
    match_j_gene
        If True, only compare pairs with matching j_gene.

    Returns
    -------
    bool
        True if pair meets all matching criteria.
    """
    if match_v_gene and rec.v1 != rec.v2:
        return False
    if match_j_gene and rec.j1 != rec.j2:
        return False
    return True
