"""Repertoire overlap utilities.

Functions for computing clonotype overlap between a query repertoire and
one or more reference sets.  Matching is by CDR3 junction_aa and, optionally,
by V-gene and/or J-gene (allele suffixes are stripped before comparison).

API
---
* :class:`OverlapCounts` — named result object: ``n`` unique clonotypes and
  ``dc`` total duplicate count.
* :func:`expand_1mm` — expand a junction_aa to all single-substitution variants.
* :func:`make_reference_keys` — build a ``frozenset`` of clonotype keys from a
  :class:`LocusRepertoire`.  Pass ``allow_1mm=True`` to include all single
  amino-acid substitution variants; ``match_v=False`` or ``match_j=False`` to
  ignore gene requirements.
* :func:`make_query_index` — build a ``{key: duplicate_count}`` lookup.
* :func:`count_overlap` — count matching clonotypes and their total
  ``duplicate_count``.  Returns an :class:`OverlapCounts`.
* :func:`compute_overlaps` — batch :func:`count_overlap` over a list of
  reference key sets, optionally dispatched across multiple processes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire

# Standard 20 amino acids used for 1-mismatch expansion.
_AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Clonotype key: (junction_aa, v_base, j_base).  Gene fields are "" when
# the corresponding match requirement is disabled.
_Key = tuple[str, str, str]


# ---------------------------------------------------------------------------
# Public result object
# ---------------------------------------------------------------------------

@dataclass
class OverlapCounts:
    """Clonotype and cell overlap between a query and a reference set.

    Attributes
    ----------
    n:
        Number of unique query clonotypes that match at least one reference
        clonotype.
    dc:
        Sum of ``duplicate_count`` for the matching query clonotypes.
    """

    n: int
    dc: int


# ---------------------------------------------------------------------------
# Module-level state shared with ProcessPoolExecutor worker processes.
# ---------------------------------------------------------------------------

_worker_qi: dict | None = None
_worker_1mm: bool = False


def _overlap_worker_init(qi: dict[_Key, int], allow_1mm: bool) -> None:
    global _worker_qi, _worker_1mm
    _worker_qi = qi
    _worker_1mm = allow_1mm


def _overlap_worker_call(ref_keys: frozenset[_Key]) -> OverlapCounts:
    return count_overlap(ref_keys, _worker_qi, allow_1mm=_worker_1mm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gene_base(gene: str) -> str:
    """Strip allele suffix: ``"TRBV6-2*01"`` → ``"TRBV6-2"``."""
    return gene.split("*")[0] if gene else ""


def _clonotype_key(
    clone: Clonotype,
    *,
    match_v: bool = True,
    match_j: bool = True,
) -> _Key:
    """Return a ``(junction_aa, v_base, j_base)`` key for *clone*.

    When *match_v* or *match_j* is ``False``, the corresponding field is
    set to ``""`` so that overlap matching ignores that gene.
    """
    return (
        clone.junction_aa,
        _gene_base(clone.v_gene) if match_v else "",
        _gene_base(clone.j_gene) if match_j else "",
    )


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
    match_v: bool = True,
    match_j: bool = True,
) -> frozenset[_Key]:
    """Build a ``frozenset`` of ``(junction_aa, v_base, j_base)`` keys.

    Parameters
    ----------
    repertoire:
        Reference repertoire.
    allow_1mm:
        When ``True``, each junction_aa is expanded to all single amino-acid
        substitution variants before inserting into the key set.  Use this for
        the *real* reference so that :func:`count_overlap` can match query
        clonotypes that differ by one substitution using the fast exact path.
        For mock key sets pass them compact to :func:`count_overlap` with
        ``allow_1mm=True`` — equivalent result, much lower memory cost.
    match_v:
        When ``False``, V-gene is ignored (key field set to ``""``).
    match_j:
        When ``False``, J-gene is ignored (key field set to ``""``).

    Returns
    -------
    frozenset
        One key per unique (possibly expanded) clonotype entry.
    """
    keys: set[_Key] = set()
    for c in repertoire.clonotypes:
        if not c.junction_aa:
            continue
        v = _gene_base(c.v_gene) if match_v else ""
        j = _gene_base(c.j_gene) if match_j else ""
        if allow_1mm:
            for variant in expand_1mm(c.junction_aa):
                keys.add((variant, v, j))
        else:
            keys.add((c.junction_aa, v, j))
    return frozenset(keys)


def make_query_index(
    repertoire: LocusRepertoire,
    *,
    match_v: bool = True,
    match_j: bool = True,
) -> dict[_Key, int]:
    """Build a ``{(junction_aa, v_base, j_base): duplicate_count}`` index.

    When the same key appears more than once (unlikely in a typical
    repertoire), ``duplicate_count`` values are summed.

    Parameters
    ----------
    repertoire:
        Query repertoire.
    match_v:
        When ``False``, V-gene is ignored (key field set to ``""``).
    match_j:
        When ``False``, J-gene is ignored (key field set to ``""``).

    Returns
    -------
    dict
        Maps each unique clonotype key to its total ``duplicate_count``.
    """
    index: dict[_Key, int] = {}
    for c in repertoire.clonotypes:
        if not c.junction_aa:
            continue
        key = _clonotype_key(c, match_v=match_v, match_j=match_j)
        index[key] = index.get(key, 0) + c.duplicate_count
    return index


def count_overlap(
    reference_keys: frozenset[_Key],
    query_index: dict[_Key, int],
    *,
    allow_1mm: bool = False,
) -> OverlapCounts:
    """Count matching clonotypes and their aggregate ``duplicate_count``.

    Parameters
    ----------
    reference_keys:
        Set of clonotype keys (from :func:`make_reference_keys` or
        :func:`~mir.biomarkers.vdjbet.generate_mock_key_sets_from_pool`).
        Gene fields should already be ``""`` when the corresponding match
        requirement is disabled (controlled by ``match_v/match_j`` when
        building keys).
    query_index:
        Clonotype → duplicate_count mapping (from :func:`make_query_index`).
        Must be built with the same ``match_v/match_j`` flags.
    allow_1mm:
        When ``True``, expand each reference ``junction_aa`` to all single
        amino-acid substitution variants and count unique query clonotypes
        within Hamming distance 1.  A query clonotype within 1mm of multiple
        reference clonotypes is counted only once.

    Returns
    -------
    OverlapCounts
        ``n`` — unique matching query clonotypes;
        ``dc`` — sum of ``duplicate_count`` for those clonotypes.
    """
    if not allow_1mm:
        n = 0
        total_dc = 0
        for key in reference_keys:
            dc = query_index.get(key)
            if dc is not None:
                n += 1
                total_dc += dc
        return OverlapCounts(n=n, dc=total_dc)

    matched: set[_Key] = set()
    for jaa, v, j in reference_keys:
        for variant in expand_1mm(jaa):
            cand = (variant, v, j)
            if cand not in matched and cand in query_index:
                matched.add(cand)
    return OverlapCounts(
        n=len(matched),
        dc=sum(query_index[k] for k in matched),
    )


def compute_overlaps(
    reference_key_sets: list[frozenset[_Key]],
    query_index: dict[_Key, int],
    *,
    allow_1mm: bool = False,
    n_jobs: int = 1,
) -> list[OverlapCounts]:
    """Compute :func:`count_overlap` for every key set in *reference_key_sets*.

    Parameters
    ----------
    reference_key_sets:
        List of reference key sets — typically the mock null distribution from
        :func:`~mir.biomarkers.vdjbet.generate_mock_key_sets_from_pool`.
    query_index:
        Query clonotype index from :func:`make_query_index`.
    allow_1mm:
        Passed through to :func:`count_overlap`.
    n_jobs:
        Number of parallel worker processes.  ``1`` (default) runs
        single-threaded.  Values > 1 use
        :class:`concurrent.futures.ProcessPoolExecutor` with an initializer
        that sends *query_index* to each worker once.

    Returns
    -------
    list[OverlapCounts]
        One :class:`OverlapCounts` per key set, in the same order.
    """
    if n_jobs == 1 or len(reference_key_sets) <= 1:
        return [
            count_overlap(k, query_index, allow_1mm=allow_1mm)
            for k in reference_key_sets
        ]

    chunksize = max(1, len(reference_key_sets) // (n_jobs * 4))
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_overlap_worker_init,
        initargs=(query_index, allow_1mm),
    ) as pool:
        return list(pool.map(
            _overlap_worker_call,
            reference_key_sets,
            chunksize=chunksize,
        ))
