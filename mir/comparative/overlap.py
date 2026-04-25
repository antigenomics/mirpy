"""Repertoire overlap utilities.

Functions for computing clonotype overlap between a query repertoire and
one or more reference sets.  Matching is by (junction_aa, V-gene base,
J-gene base) — allele suffixes are stripped before comparison.

API
---
* :func:`make_reference_keys` — build a ``frozenset`` of clonotype keys from a
  :class:`LocusRepertoire`.
* :func:`make_query_index` — build a ``{key: duplicate_count}`` lookup for fast
  overlap computation.
* :func:`count_overlap` — count matching clonotypes and their total
  ``duplicate_count``.
"""

from __future__ import annotations

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gene_base(gene: str) -> str:
    """Strip allele suffix: ``"TRBV6-2*01"`` → ``"TRBV6-2"``."""
    return gene.split("*")[0] if gene else ""


def _clonotype_key(clone: Clonotype) -> tuple[str, str, str]:
    """Return ``(junction_aa, v_base, j_base)`` for *clone*."""
    return (clone.junction_aa, _gene_base(clone.v_gene), _gene_base(clone.j_gene))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_reference_keys(repertoire: LocusRepertoire) -> frozenset[tuple[str, str, str]]:
    """Build a ``frozenset`` of ``(junction_aa, v_base, j_base)`` keys.

    Parameters
    ----------
    repertoire:
        Reference repertoire.

    Returns
    -------
    frozenset
        One key per unique clonotype (duplicates are collapsed).
    """
    return frozenset(
        _clonotype_key(c)
        for c in repertoire.clonotypes
        if c.junction_aa
    )


def make_query_index(repertoire: LocusRepertoire) -> dict[tuple[str, str, str], int]:
    """Build a ``{(junction_aa, v_base, j_base): duplicate_count}`` index.

    When the same key appears more than once (unlikely in a typical
    repertoire), ``duplicate_count`` values are summed.

    Parameters
    ----------
    repertoire:
        Query repertoire.

    Returns
    -------
    dict
        Maps each unique clonotype key to its total ``duplicate_count``.
    """
    index: dict[tuple[str, str, str], int] = {}
    for c in repertoire.clonotypes:
        if not c.junction_aa:
            continue
        key = _clonotype_key(c)
        index[key] = index.get(key, 0) + c.duplicate_count
    return index


def count_overlap(
    reference_keys: frozenset[tuple[str, str, str]],
    query_index: dict[tuple[str, str, str], int],
) -> tuple[int, int]:
    """Count matching clonotypes and their aggregate ``duplicate_count``.

    Parameters
    ----------
    reference_keys:
        Set of clonotype keys from the reference (from
        :func:`make_reference_keys`).
    query_index:
        Clonotype → duplicate_count mapping for the query (from
        :func:`make_query_index`).

    Returns
    -------
    (n_clonotypes, sum_duplicate_count)
        * ``n_clonotypes`` — number of unique clonotypes in the query that
          appear in *reference_keys*.
        * ``sum_duplicate_count`` — sum of ``duplicate_count`` for those
          matching clonotypes.
    """
    n = 0
    total_dc = 0
    # Iterate over the smaller reference set and probe the larger query dict —
    # for a 409-entry VDJdb reference vs 100k-entry repertoire this is ~250x
    # faster than iterating over the query index.
    for key in reference_keys:
        dc = query_index.get(key)
        if dc is not None:
            n += 1
            total_dc += dc
    return n, total_dc
