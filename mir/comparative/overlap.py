"""Repertoire overlap utilities.

Functions for computing clonotype overlap between a query repertoire and
one or more reference sets.  Matching is by (junction_aa, V-gene base,
J-gene base) — allele suffixes are stripped before comparison.

API
---
* :func:`expand_1mm` — expand a junction_aa to all single-substitution variants.
* :func:`make_reference_keys` — build a ``frozenset`` of clonotype keys from a
  :class:`LocusRepertoire`.  Pass ``allow_1mm=True`` to also include all
  single amino-acid substitution variants of each junction_aa.
* :func:`make_query_index` — build a ``{key: duplicate_count}`` lookup for fast
  overlap computation.
* :func:`count_overlap` — count matching clonotypes and their total
  ``duplicate_count``.
"""

from __future__ import annotations

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire

# Standard 20 amino acids used for 1-mismatch expansion.
_AA20 = "ACDEFGHIKLMNPQRSTVWY"


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

def expand_1mm(seq: str) -> list[str]:
    """Return *seq* plus all single amino-acid substitution variants.

    Produces ``19 * len(seq) + 1`` strings in total (the original plus one
    variant per position per alternative amino acid).

    Parameters
    ----------
    seq:
        Amino-acid sequence (typically a CDR3 / junction_aa).

    Returns
    -------
    list[str]
        The original sequence followed by all 1-substitution variants.
    """
    result = [seq]
    for i, orig in enumerate(seq):
        prefix = seq[:i]
        suffix = seq[i + 1:]
        for aa in _AA20:
            if aa != orig:
                result.append(prefix + aa + suffix)
    return result


def make_reference_keys(
    repertoire: LocusRepertoire,
    *,
    allow_1mm: bool = False,
) -> frozenset[tuple[str, str, str]]:
    """Build a ``frozenset`` of ``(junction_aa, v_base, j_base)`` keys.

    Parameters
    ----------
    repertoire:
        Reference repertoire.
    allow_1mm:
        When ``True``, each junction_aa is expanded to all single amino-acid
        substitution variants before inserting into the key set.  This allows
        :func:`count_overlap` to match query clonotypes that differ from the
        reference by a single amino acid.  The key set grows by roughly
        ``19 * mean_cdr3_length`` per reference clonotype.

    Returns
    -------
    frozenset
        One key per unique (possibly expanded) clonotype key.
    """
    keys: set[tuple[str, str, str]] = set()
    for c in repertoire.clonotypes:
        if not c.junction_aa:
            continue
        v = _gene_base(c.v_gene)
        j = _gene_base(c.j_gene)
        if allow_1mm:
            for variant in expand_1mm(c.junction_aa):
                keys.add((variant, v, j))
        else:
            keys.add((c.junction_aa, v, j))
    return frozenset(keys)


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
