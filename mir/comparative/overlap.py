"""Repertoire overlap utilities.

Functions for computing clonotype overlap between a query repertoire and
one or more reference sets.  Matching is by CDR3 junction_aa and, optionally,
by V-gene and/or J-gene (allele suffixes are stripped before comparison).

Normalization
~~~~~~~~~~~~~
Overlap counts can be normalized by the target (reference) repertoire size
by passing ``target_n`` and ``target_dc`` parameters. When provided:

* ``n_normalized = matched_clonotypes / target_n``
* ``dc_normalized = matched_duplicate_count / target_dc``

This allows comparison of overlap across repertoires of different sizes.
The interpretation is: "Among all clonotypes in the target repertoire,
what fraction matched the query?"

API
---
* :class:`OverlapCounts` — named result object with raw and normalized counts:
  ``n``, ``dc`` (absolute), and ``n_normalized``, ``dc_normalized`` (relative).
* :func:`expand_1mm` — expand a junction_aa to all single-substitution variants.
* :func:`make_reference_keys` — build a ``frozenset`` of clonotype keys from a
  :class:`LocusRepertoire`.  Pass ``allow_1mm=True`` to include all single
  amino-acid substitution variants; ``match_v=False`` or ``match_j=False`` to
  ignore gene requirements.
* :func:`make_query_index` — build a ``{key: duplicate_count}`` lookup.
* :func:`count_overlap` — count matching clonotypes and their total
  ``duplicate_count`` with optional normalization.  Returns an :class:`OverlapCounts`.
* :func:`compute_overlaps` — batch :func:`count_overlap` over a list of
  reference key sets, optionally dispatched across multiple processes.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import weakref
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

_MP_CTX = multiprocessing.get_context("spawn")

from mir.basic.alphabets import AA_STANDARD_CHARS
from mir.basic.mirseq_compat import is_coding as is_coding_aa
from mir.common.alleles import allele_to_major
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire
from mir.graph._trie_utils import hit_index, search_limits

try:
    from tcrtrie import Trie
except Exception:  # pragma: no cover - optional runtime dependency guard
    Trie = None

_VALID_OVERLAP_SPACES = {"ntvj", "nt", "aavj", "aa"}
_AA_OVERLAP_SPACES = {"aavj", "aa"}
# Backward-compatible alias used by tests and external callers.
_AA20 = AA_STANDARD_CHARS
_MAX_NONCODING_WARNINGS = 10
_noncoding_warning_count = 0


@dataclass(slots=True)
class _PreparedTarget:
    rep_ref: weakref.ReferenceType | None
    qi2: dict[tuple[str, str, str], int]
    keys2: list[tuple[str, str, str]]
    dc2: list[int]
    total_dc2: int
    trie: object | None
    simpson_d2: float


_PAIRWISE_TARGET_CACHE: dict[tuple[int, str, str, int, bool, bool], _PreparedTarget] = {}
_PAIRWISE_TARGET_CACHE_MAX = max(0, int(os.getenv("MIRPY_OVERLAP_TARGET_CACHE_SIZE", "8")))


def clear_pairwise_target_cache() -> None:
    """Clear prepared-target cache used by :func:`pairwise_overlap`.

    In long-lived notebook sessions this can release multiple GB of memory.
    """
    _PAIRWISE_TARGET_CACHE.clear()

# Clonotype key: (junction_aa, v_base, j_base).  Gene fields are "" when
# the corresponding match requirement is disabled.
_Key = tuple[str, str, str]


# ---------------------------------------------------------------------------
# Public result object
# ---------------------------------------------------------------------------

@dataclass
class OverlapCounts:
    """Clonotype overlap summary between query and reference sets."""

    n: int
    dc: int
    n_normalized: float | None = None
    dc_normalized: float | None = None


# ---------------------------------------------------------------------------
# Module-level state shared with ProcessPoolExecutor worker processes.
# ---------------------------------------------------------------------------

_worker_qi: dict | None = None
_worker_1mm: bool = False
_worker_target_n: int | None = None
_worker_target_dc: int | None = None


def _overlap_worker_init(
    qi: dict[_Key, int],
    allow_1mm: bool,
    target_n: int | None = None,
    target_dc: int | None = None,
) -> None:
    global _worker_qi, _worker_1mm, _worker_target_n, _worker_target_dc
    _worker_qi = qi
    _worker_1mm = allow_1mm
    _worker_target_n = target_n
    _worker_target_dc = target_dc


def _overlap_worker_call(ref_keys: frozenset[_Key]) -> OverlapCounts:
    return count_overlap(
        ref_keys,
        _worker_qi,
        allow_1mm=_worker_1mm,
        target_n=_worker_target_n,
        target_dc=_worker_target_dc,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gene_base(gene: str) -> str:
    """Map allele naming to stable major allele form ``*01``."""
    return allele_to_major(gene)


def _resolve_overlap_space(
    overlap_space: str | None,
    *,
    match_v: bool,
    match_j: bool,
) -> str:
    if overlap_space is None:
        if match_v and match_j:
            return "aavj"
        if not match_v and not match_j:
            return "aa"
        # Preserve legacy behavior for asymmetric match flags.
        return "aavj"
    if overlap_space not in _VALID_OVERLAP_SPACES:
        raise ValueError(
            f"overlap_space must be one of {sorted(_VALID_OVERLAP_SPACES)}; got {overlap_space!r}"
        )
    return overlap_space


def _allow_mismatches(overlap_space: str, metric: str, threshold: int) -> None:
    if threshold <= 0 or metric == "exact":
        return
    if overlap_space not in _AA_OVERLAP_SPACES:
        raise ValueError(
            "Approximate overlap is only supported for overlap_space='aa' or 'aavj'. "
            f"Got overlap_space={overlap_space!r}, metric={metric!r}, threshold={threshold}."
        )


def _emit_noncoding_warning(dropped: int, context: str) -> None:
    global _noncoding_warning_count
    if dropped <= 0 or _noncoding_warning_count >= _MAX_NONCODING_WARNINGS:
        return

    remaining = _MAX_NONCODING_WARNINGS - _noncoding_warning_count
    warn_now = min(remaining, 1)
    if warn_now <= 0:
        return

    suffix = ""
    if _noncoding_warning_count + warn_now >= _MAX_NONCODING_WARNINGS:
        suffix = " Further non-coding overlap warnings are suppressed."
    warnings.warn(
        f"Excluded {dropped} non-coding clonotypes from amino-acid overlap ({context})." + suffix,
        RuntimeWarning,
        stacklevel=2,
    )
    _noncoding_warning_count += warn_now


def _sequence_for_space(
    clone: Clonotype,
    overlap_space: str,
) -> str:
    if overlap_space in _AA_OVERLAP_SPACES:
        return clone.junction_aa
    return clone.junction


def _count_overlap_1mm_trie(
    reference_keys: frozenset[_Key],
    query_index: dict[_Key, int],
) -> tuple[int, int]:
    if not reference_keys or not query_index or Trie is None:
        return _count_overlap_1mm_expansion(reference_keys, query_index)

    q_keys = list(query_index.keys())
    q_jaa = [k[0] for k in q_keys]
    q_v = [k[1] for k in q_keys]
    q_j = [k[2] for k in q_keys]
    q_dc = [query_index[k] for k in q_keys]

    try:
        trie = Trie(
            sequences=q_jaa,
            vGenes=q_v,
            jGenes=q_j,
            with_counts=False,
            with_indices=True,
        )
        max_sub, max_ins, max_del, max_edits = search_limits("hamming", 1)
    except Exception:
        return _count_overlap_1mm_expansion(reference_keys, query_index)

    matched_idx: set[int] = set()
    for jaa, v, j in reference_keys:
        if not jaa:
            continue
        try:
            hits = trie.SearchIndices(
                cdr3=jaa,
                maxSub=max_sub,
                maxIns=max_ins,
                maxDel=max_del,
                maxEdits=max_edits,
            )
        except Exception:
            return _count_overlap_1mm_expansion(reference_keys, query_index)
        for hit in hits:
            idx = hit_index(hit)
            if idx in matched_idx:
                continue
            # Preserve exact V/J key semantics used by dict-based matching.
            if q_v[idx] == v and q_j[idx] == j:
                matched_idx.add(idx)

    return len(matched_idx), sum(q_dc[i] for i in matched_idx)


def _count_overlap_1mm_expansion(
    reference_keys: frozenset[_Key],
    query_index: dict[_Key, int],
) -> tuple[int, int]:
    matched: set[_Key] = set()
    for jaa, v, j in reference_keys:
        for variant in expand_1mm(jaa):
            cand = (variant, v, j)
            if cand not in matched and cand in query_index:
                matched.add(cand)
    return len(matched), sum(query_index[k] for k in matched)


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
        for aa in AA_STANDARD_CHARS:
            if aa != orig:
                result.append(prefix + aa + suffix)
    return result


def make_reference_keys(
    repertoire: LocusRepertoire,
    *,
    allow_1mm: bool = False,
    match_v: bool = True,
    match_j: bool = True,
    overlap_space: str | None = None,
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
    overlap_space:
        Explicit key space: ``"ntvj"``, ``"nt"``, ``"aavj"``, or ``"aa"``.
        When ``None``, legacy ``match_v/match_j`` behavior is used.

    Returns
    -------
    frozenset
        One key per unique (possibly expanded) clonotype entry.
    """
    overlap_space = _resolve_overlap_space(overlap_space, match_v=match_v, match_j=match_j)
    use_v = overlap_space in {"ntvj", "aavj"}
    use_j = overlap_space in {"ntvj", "aavj"}
    use_aa = overlap_space in _AA_OVERLAP_SPACES

    keys: set[_Key] = set()
    dropped_noncoding = 0

    pending = getattr(repertoire, "_pending_cols", None)
    if pending is not None:
        junctions = pending.get("junctions", [])
        junction_aas = pending.get("junction_aas", [])
        v_genes = pending.get("v_genes", [])
        j_genes = pending.get("j_genes", [])
        for jnt, jaa, v_gene, j_gene in zip(junctions, junction_aas, v_genes, j_genes):
            seq = jaa if use_aa else jnt
            if not seq:
                continue
            if use_aa and not is_coding_aa(seq):
                dropped_noncoding += 1
                continue
            v = _gene_base(v_gene) if use_v else ""
            j = _gene_base(j_gene) if use_j else ""
            if allow_1mm and use_aa:
                for variant in expand_1mm(seq):
                    keys.add((variant, v, j))
            else:
                keys.add((seq, v, j))
        _emit_noncoding_warning(dropped_noncoding, context=f"reference:{overlap_space}")
        return frozenset(keys)

    for c in repertoire.clonotypes:
        seq = _sequence_for_space(c, overlap_space)
        if not seq:
            continue
        if use_aa and not c.is_coding():
            dropped_noncoding += 1
            continue
        v = _gene_base(c.v_gene) if use_v else ""
        j = _gene_base(c.j_gene) if use_j else ""
        if allow_1mm and use_aa:
            for variant in expand_1mm(seq):
                keys.add((variant, v, j))
        else:
            keys.add((seq, v, j))
    _emit_noncoding_warning(dropped_noncoding, context=f"reference:{overlap_space}")
    return frozenset(keys)


def make_query_index(
    repertoire: LocusRepertoire,
    *,
    match_v: bool = True,
    match_j: bool = True,
    overlap_space: str | None = None,
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
    overlap_space:
        Explicit key space: ``"ntvj"``, ``"nt"``, ``"aavj"``, or ``"aa"``.
        When ``None``, legacy ``match_v/match_j`` behavior is used.

    Returns
    -------
    dict
        Maps each unique clonotype key to its total ``duplicate_count``.
    """
    overlap_space = _resolve_overlap_space(overlap_space, match_v=match_v, match_j=match_j)
    use_v = overlap_space in {"ntvj", "aavj"}
    use_j = overlap_space in {"ntvj", "aavj"}
    use_aa = overlap_space in _AA_OVERLAP_SPACES

    index: dict[_Key, int] = {}
    dropped_noncoding = 0

    pending = getattr(repertoire, "_pending_cols", None)
    if pending is not None:
        junctions = pending.get("junctions", [])
        junction_aas = pending.get("junction_aas", [])
        v_genes = pending.get("v_genes", [])
        j_genes = pending.get("j_genes", [])
        dups = pending.get("dup_counts", [])
        for jnt, jaa, v_gene, j_gene, dc in zip(junctions, junction_aas, v_genes, j_genes, dups):
            seq = jaa if use_aa else jnt
            if not seq:
                continue
            if use_aa and not is_coding_aa(seq):
                dropped_noncoding += 1
                continue
            key = (
                seq,
                _gene_base(v_gene) if use_v else "",
                _gene_base(j_gene) if use_j else "",
            )
            index[key] = index.get(key, 0) + int(dc or 0)
        _emit_noncoding_warning(dropped_noncoding, context=f"query:{overlap_space}")
        return index

    for c in repertoire.clonotypes:
        seq = _sequence_for_space(c, overlap_space)
        if not seq:
            continue
        if use_aa and not c.is_coding():
            dropped_noncoding += 1
            continue
        key = (
            seq,
            _gene_base(c.v_gene) if use_v else "",
            _gene_base(c.j_gene) if use_j else "",
        )
        index[key] = index.get(key, 0) + c.duplicate_count
    _emit_noncoding_warning(dropped_noncoding, context=f"query:{overlap_space}")
    return index


def count_overlap(
    reference_keys: frozenset[_Key],
    query_index: dict[_Key, int],
    *,
    allow_1mm: bool = False,
    target_n: int | None = None,
    target_dc: int | None = None,
) -> OverlapCounts:
    """Count matching clonotypes and their aggregate ``duplicate_count``.

    Parameters
    ----------
    reference_keys:
        Set of clonotype keys (from :func:`make_reference_keys` or mock key
        sets built by :class:`mir.comparative.vdjbet.VDJBetOverlapAnalysis`
        / :class:`mir.comparative.vdjbet.PgenBinPool`).
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
    target_n:
        Total number of clonotypes in the target (reference) repertoire.
        When provided, normalized overlap count is computed as
        ``matched_n / target_n``.
    target_dc:
        Total duplicate_count in the target (reference) repertoire.
        When provided, normalized duplicate_count is computed as
        ``matched_dc / target_dc``.

    Returns
    -------
    OverlapCounts
        ``n`` — unique matching query clonotypes;
        ``dc`` — sum of ``duplicate_count`` for those clonotypes;
        ``n_normalized`` — matched clonotypes / target_n (if target_n provided);
        ``dc_normalized`` — matched duplicate_count / target_dc (if target_dc provided).
    """
    if not allow_1mm:
        n = 0
        total_dc = 0
        for key in reference_keys:
            dc = query_index.get(key)
            if dc is not None:
                n += 1
                total_dc += dc
    else:
        n, total_dc = _count_overlap_1mm_trie(reference_keys, query_index)

    n_normalized = (n / target_n) if target_n is not None and target_n > 0 else None
    dc_normalized = (
        (total_dc / target_dc) if target_dc is not None and target_dc > 0 else None
    )

    return OverlapCounts(
        n=n,
        dc=total_dc,
        n_normalized=n_normalized,
        dc_normalized=dc_normalized,
    )


def compute_overlaps(
    reference_key_sets: list[frozenset[_Key]],
    query_index: dict[_Key, int],
    *,
    allow_1mm: bool = False,
    target_n: int | None = None,
    target_dc: int | None = None,
    n_jobs: int = -1,
) -> list[OverlapCounts]:
    """Compute :func:`count_overlap` for every key set in *reference_key_sets*.

    Parameters
    ----------
    reference_key_sets:
        List of reference key sets — typically the mock null distribution from
        :class:`mir.comparative.vdjbet.VDJBetOverlapAnalysis` /
        :class:`mir.comparative.vdjbet.PgenBinPool`.
    query_index:
        Query clonotype index from :func:`make_query_index`.
    allow_1mm:
        Passed through to :func:`count_overlap`.
    target_n:
        Total number of clonotypes in target (reference) repertoire.
        Passed through to :func:`count_overlap` for normalization.
    target_dc:
        Total duplicate_count in target (reference) repertoire.
        Passed through to :func:`count_overlap` for normalization.
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
    n_jobs = _n_jobs_resolve(n_jobs)

    # Process startup on macOS can dominate runtime for small exact-match batches.
    estimated_keys = sum(len(keys) for keys in reference_key_sets)
    should_run_serial = (
        n_jobs == 1
        or len(reference_key_sets) <= 1
        or (not allow_1mm and estimated_keys <= 2_000_000)
    )
    if should_run_serial:
        return [
            count_overlap(
                k,
                query_index,
                allow_1mm=allow_1mm,
                target_n=target_n,
                target_dc=target_dc,
            )
            for k in reference_key_sets
        ]

    chunksize = max(1, len(reference_key_sets) // (n_jobs * 4))
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=_MP_CTX,
        initializer=_overlap_worker_init,
        initargs=(query_index, allow_1mm, target_n, target_dc),
    ) as pool:
        return list(pool.map(
            _overlap_worker_call,
            reference_key_sets,
            chunksize=chunksize,
        ))


# ---------------------------------------------------------------------------
# Pairwise sample overlap — result dataclass
# ---------------------------------------------------------------------------

_nan = float("nan")


@dataclass
class PairwiseOverlapResult:
    """Pairwise overlap metrics between two repertoires.

    In approximate modes (Hamming/Levenshtein with threshold > 0), matching is
    many-to-many and ``correlation`` / ``f2_similarity`` are reported as ``nan``.
    """

    n1: int
    n2: int
    n1_matched: int
    n2_matched: int
    f1_overlap: float
    f2_overlap: float
    jaccard: float
    szymkiewicz_simpson: float
    d_similarity: float
    f_similarity: float
    morisita_horn: float
    correlation: float
    f2_similarity: float
    mode: str
    is_approximate: bool

    def as_dict(self) -> dict:
        """Return all fields as a plain ``dict``."""
        return {
            "n1": self.n1, "n2": self.n2,
            "n1_matched": self.n1_matched, "n2_matched": self.n2_matched,
            "f1_overlap": self.f1_overlap, "f2_overlap": self.f2_overlap,
            "jaccard": self.jaccard,
            "szymkiewicz_simpson": self.szymkiewicz_simpson,
            "d_similarity": self.d_similarity,
            "f_similarity": self.f_similarity,
            "morisita_horn": self.morisita_horn,
            "correlation": self.correlation,
            "f2_similarity": self.f2_similarity,
            "mode": self.mode, "is_approximate": self.is_approximate,
        }


# ---------------------------------------------------------------------------
# Internal helpers for pairwise metrics
# ---------------------------------------------------------------------------

def _n_jobs_resolve(n_jobs: int) -> int:
    """Resolve ``-1`` to all physical cores; clamp to ≥ 1."""
    if n_jobs == -1:
        try:
            import psutil
            n = psutil.cpu_count(logical=False)
            if n:
                return n
        except ImportError:
            pass
        return os.cpu_count() or 1
    return max(1, n_jobs)


def _simpson_lambda(dc_values: list[int], total: int) -> float:
    """Σ(dc_i / total)² — Simpson's diversity index."""
    if total == 0:
        return 0.0
    return sum((dc / total) ** 2 for dc in dc_values)


def _empty_pairwise(n1: int, n2: int, mode: str, is_approx: bool) -> PairwiseOverlapResult:
    return PairwiseOverlapResult(
        n1=n1, n2=n2, n1_matched=0, n2_matched=0,
        f1_overlap=0.0, f2_overlap=0.0,
        jaccard=0.0, szymkiewicz_simpson=0.0, d_similarity=0.0, f_similarity=0.0,
        morisita_horn=0.0, correlation=_nan, f2_similarity=_nan,
        mode=mode, is_approximate=is_approx,
    )


def _compute_exact_pairwise(
    qi1: dict[_Key, int],
    qi2: dict[_Key, int],
    *,
    total_dc1: int | None = None,
    total_dc2: int | None = None,
    D1: float | None = None,
    D2: float | None = None,
) -> PairwiseOverlapResult:
    """Exact 1:1 matching — all metrics defined."""
    n1, n2 = len(qi1), len(qi2)
    if n1 == 0 or n2 == 0:
        return _empty_pairwise(n1, n2, "exact", False)

    total_dc1 = (sum(qi1.values()) or 1) if total_dc1 is None else total_dc1
    total_dc2 = (sum(qi2.values()) or 1) if total_dc2 is None else total_dc2

    D1 = _simpson_lambda(list(qi1.values()), total_dc1) if D1 is None else D1
    D2 = _simpson_lambda(list(qi2.values()), total_dc2) if D2 is None else D2

    shared = qi1.keys() & qi2.keys()
    n12 = len(shared)

    if n12 == 0:
        return _empty_pairwise(n1, n2, "exact", False)

    ov_f1 = [qi1[k] / total_dc1 for k in shared]
    ov_f2 = [qi2[k] / total_dc2 for k in shared]

    f1_overlap = sum(ov_f1)
    f2_overlap = sum(ov_f2)

    jaccard = n12 / (n1 + n2 - n12)
    d_similarity = n12 / math.sqrt(n1 * n2)
    f_similarity = math.sqrt(f1_overlap * f2_overlap)
    f2_similarity = sum(math.sqrt(a * b) for a, b in zip(ov_f1, ov_f2))
    szymkiewicz_simpson = n12 / min(n1, n2)

    mh_num = 2.0 * sum(a * b for a, b in zip(ov_f1, ov_f2))
    morisita_horn = mh_num / (D1 + D2) if (D1 + D2) > 0 else _nan

    if n12 >= 2:
        from scipy.stats import pearsonr
        try:
            # Avoid scipy ConstantInputWarning spam on degenerate vectors.
            if max(ov_f1) == min(ov_f1) or max(ov_f2) == min(ov_f2):
                correlation = _nan
            else:
                r, _ = pearsonr(ov_f1, ov_f2)
                correlation = float(r) if not math.isnan(r) else _nan
        except Exception:
            correlation = _nan
    else:
        correlation = _nan

    return PairwiseOverlapResult(
        n1=n1, n2=n2, n1_matched=n12, n2_matched=n12,
        f1_overlap=f1_overlap, f2_overlap=f2_overlap,
        jaccard=jaccard,
        szymkiewicz_simpson=szymkiewicz_simpson,
        d_similarity=d_similarity,
        f_similarity=f_similarity,
        morisita_horn=morisita_horn,
        correlation=correlation,
        f2_similarity=f2_similarity,
        mode="exact", is_approximate=False,
    )


def _trie_search_serial(
    keys1: list[_Key],
    dc1: list[int],
    total_dc1: int,
    trie: object,
    v2: list[str],
    j2: list[str],
    dc2: list[int],
    total_dc2: int,
    max_sub: int,
    max_ins: int,
    max_del: int,
    max_edits: int,
) -> tuple[set[int], set[int], int, float]:
    """Search all s1 clones against a pre-built s2 trie.

    Returns ``(s1_matched_idx, s2_matched_idx, dc1_matched_total, mh_pairs_sum)``.
    """
    s1_matched: set[int] = set()
    s2_matched: set[int] = set()
    dc1_matched = 0
    mh_sum = 0.0

    for i1, ((jaa, v, j), dc) in enumerate(zip(keys1, dc1)):
        v_filter = v or None
        j_filter = j or None
        try:
            hits = trie.SearchIndices(
                query=jaa,
                maxSubstitution=max_sub,
                maxInsertion=max_ins,
                maxDeletion=max_del,
                maxEdits=max_edits,
                vGeneFilter=v_filter,
                jGeneFilter=j_filter,
            )
        except Exception:
            continue

        any_hit = False
        p_i = dc / total_dc1
        for hit in hits:
            i2 = hit_index(hit)
            if not any_hit:
                s1_matched.add(i1)
                dc1_matched += dc
                any_hit = True
            s2_matched.add(i2)
            mh_sum += p_i * (dc2[i2] / total_dc2)

    return s1_matched, s2_matched, dc1_matched, mh_sum


# ---------------------------------------------------------------------------
# Parallel trie search — within a single pair (chunk workers)
# ---------------------------------------------------------------------------

_PW_TRIE_STATE: dict = {}


def _pw_trie_worker_init(
    jaa2: list[str],
    v2: list[str],
    j2: list[str],
    dc2: list[int],
    total_dc1: int,
    total_dc2: int,
    limits: tuple[int, int, int, int],
) -> None:
    global _PW_TRIE_STATE
    from tcrtrie import Trie as _Trie
    _PW_TRIE_STATE = {
        "trie": _Trie(sequences=jaa2, vGenes=v2, jGenes=j2),
        "v2": v2, "j2": j2, "dc2": dc2,
        "total_dc1": total_dc1, "total_dc2": total_dc2,
        "limits": limits,
    }


def _pw_trie_worker_call(
    chunk: list[tuple[int, str, str, str, int]],
) -> tuple[list[int], list[int], int, float]:
    """Process a chunk of s1 clones. Returns (i1_matched, i2_matched, dc1_matched, mh_sum)."""
    st = _PW_TRIE_STATE
    trie = st["trie"]
    v2, j2, dc2 = st["v2"], st["j2"], st["dc2"]
    total_dc1, total_dc2 = st["total_dc1"], st["total_dc2"]
    max_sub, max_ins, max_del, max_edits = st["limits"]

    s1_matched: set[int] = set()
    s2_matched: set[int] = set()
    dc1_matched = 0
    mh_sum = 0.0

    for i1_global, jaa, v, j, dc in chunk:
        v_filter = v or None
        j_filter = j or None
        try:
            hits = trie.SearchIndices(
                query=jaa,
                maxSubstitution=max_sub,
                maxInsertion=max_ins,
                maxDeletion=max_del,
                maxEdits=max_edits,
                vGeneFilter=v_filter,
                jGeneFilter=j_filter,
            )
        except Exception:
            continue

        any_hit = False
        p_i = dc / total_dc1
        for hit in hits:
            i2 = hit_index(hit)
            if not any_hit:
                s1_matched.add(i1_global)
                dc1_matched += dc
                any_hit = True
            s2_matched.add(i2)
            mh_sum += p_i * (dc2[i2] / total_dc2)

    return list(s1_matched), list(s2_matched), dc1_matched, mh_sum


def _compute_trie_pairwise(
    qi1: dict[_Key, int],
    qi2: dict[_Key, int],
    *,
    metric: str,
    threshold: int,
    n_jobs: int = 1,
    prepared_target: _PreparedTarget | None = None,
) -> PairwiseOverlapResult:
    """Approximate pairwise overlap via tcrtrie.

    Builds the trie from *qi2* and searches all clones from *qi1* against it.
    Returns ``n1_matched`` (qi1 clones with ≥ 1 hit) and ``n2_matched``
    (qi2 clones hit by ≥ 1 qi1 clone) independently for symmetric metrics.
    """
    mode = f"{metric}:{threshold}"
    n1, n2 = len(qi1), len(qi2)

    if n1 == 0 or n2 == 0 or Trie is None:
        return _empty_pairwise(n1, n2, mode, True)

    if prepared_target is not None:
        keys2 = prepared_target.keys2
        jaa2 = [k[0] for k in keys2]
        v2 = [k[1] for k in keys2]
        j2 = [k[2] for k in keys2]
        dc2 = prepared_target.dc2
        total_dc2 = prepared_target.total_dc2
        trie = prepared_target.trie
    else:
        keys2 = list(qi2.keys())
        jaa2 = [k[0] for k in keys2]
        v2 = [k[1] for k in keys2]
        j2 = [k[2] for k in keys2]
        dc2 = [qi2[k] for k in keys2]
        total_dc2 = sum(dc2) or 1
        try:
            trie = Trie(sequences=jaa2, vGenes=v2, jGenes=j2)
        except Exception:
            return _empty_pairwise(n1, n2, mode, True)

    keys1 = list(qi1.keys())
    dc1 = [qi1[k] for k in keys1]
    total_dc1 = sum(dc1) or 1

    limits = search_limits(metric, threshold)

    # Process startup can dominate runtime for modest query sizes.
    if n_jobs > 1 and len(keys1) < 50_000:
        n_jobs = 1

    if n_jobs == 1:
        s1_idx, s2_idx, dc1_matched, mh_sum = _trie_search_serial(
            keys1, dc1, total_dc1, trie, v2, j2, dc2, total_dc2, *limits,
        )
        if prepared_target is None:
            del trie  # free before spawning (not needed here but consistent)
    else:
        if prepared_target is None:
            del trie  # rebuilt inside each worker via initializer
        chunk_size = max(1, len(keys1) // n_jobs)
        chunks = [
            [(i, keys1[i][0], keys1[i][1], keys1[i][2], dc1[i])
             for i in range(start, min(start + chunk_size, len(keys1)))]
            for start in range(0, len(keys1), chunk_size)
        ]
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=_MP_CTX,
            initializer=_pw_trie_worker_init,
            initargs=(jaa2, v2, j2, dc2, total_dc1, total_dc2, limits),
        ) as pool:
            partial = list(pool.map(_pw_trie_worker_call, chunks))

        s1_idx: set[int] = set()
        s2_idx: set[int] = set()
        dc1_matched = 0
        mh_sum = 0.0
        for i1_list, i2_list, dc1m, mh in partial:
            s1_idx.update(i1_list)
            s2_idx.update(i2_list)
            dc1_matched += dc1m
            mh_sum += mh
        s2_idx = s2_idx  # keep as set

    n1_matched = len(s1_idx)
    n2_matched = len(s2_idx)

    if n1_matched == 0:
        return _empty_pairwise(n1, n2, mode, True)

    f1_overlap = dc1_matched / total_dc1
    f2_overlap = sum(dc2[i] for i in s2_idx) / total_dc2

    # Symmetric n12 via geometric mean
    n12_eff = math.sqrt(n1_matched * n2_matched)
    denom = n1 + n2 - n12_eff
    jaccard = n12_eff / denom if denom > 0 else 0.0
    d_similarity = n12_eff / math.sqrt(n1 * n2)
    f_similarity = math.sqrt(f1_overlap * f2_overlap)
    szymkiewicz_simpson = min(n1_matched, n2_matched) / min(n1, n2)

    D1 = _simpson_lambda(dc1, total_dc1)
    D2 = _simpson_lambda(dc2, total_dc2)
    morisita_horn = 2.0 * mh_sum / (D1 + D2) if (D1 + D2) > 0 else _nan

    return PairwiseOverlapResult(
        n1=n1, n2=n2, n1_matched=n1_matched, n2_matched=n2_matched,
        f1_overlap=f1_overlap, f2_overlap=f2_overlap,
        jaccard=jaccard,
        szymkiewicz_simpson=szymkiewicz_simpson,
        d_similarity=d_similarity,
        f_similarity=f_similarity,
        morisita_horn=morisita_horn,
        correlation=_nan,
        f2_similarity=_nan,
        mode=mode, is_approximate=True,
    )


# ---------------------------------------------------------------------------
# Public pairwise overlap API
# ---------------------------------------------------------------------------

def pairwise_overlap(
    rep1: LocusRepertoire,
    rep2: LocusRepertoire,
    *,
    metric: str = "exact",
    threshold: int = 0,
    match_v: bool = True,
    match_j: bool = True,
    overlap_space: str | None = None,
    n_jobs: int = -1,
) -> PairwiseOverlapResult:
    """Compute pairwise overlap metrics between two repertoires.

    Parameters
    ----------
    rep1, rep2 :
        Repertoires to compare.
    metric :
        ``"exact"`` (default), ``"hamming"``, or ``"levenshtein"``.
        ``"exact"`` ignores *threshold*.
    threshold :
        Maximum edit distance (substitutions for hamming, any edit for
        levenshtein).  ``0`` is equivalent to ``metric="exact"``.
    match_v, match_j :
        When ``False``, the corresponding gene is ignored for matching.
    overlap_space :
        Identity space: ``"ntvj"``, ``"nt"``, ``"aavj"``, or ``"aa"``.
        When set, this overrides legacy ``match_v/match_j`` matching semantics.
        Approximate matching (``threshold > 0``) is allowed only for
        ``"aa"`` and ``"aavj"``.
    n_jobs :
        Worker processes for parallel trie search within this pair.
        ``-1`` → all physical cores.  ``1`` (default) → serial.

    Returns
    -------
    PairwiseOverlapResult
        All overlap metrics.  See :class:`PairwiseOverlapResult` for details.

    Examples
    --------
    >>> result = pairwise_overlap(rep1, rep2)
    >>> result.f_similarity
    0.0123
    >>> result = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)
    >>> result.d_similarity
    0.045
    """
    overlap_space = _resolve_overlap_space(overlap_space, match_v=match_v, match_j=match_j)
    _allow_mismatches(overlap_space, metric, threshold)

    n_jobs = _n_jobs_resolve(n_jobs)
    qi1 = make_query_index(
        rep1,
        match_v=match_v,
        match_j=match_j,
        overlap_space=overlap_space,
    )

    cache_key = (id(rep2), overlap_space, metric, threshold, match_v, match_j)
    prepared_target = _PAIRWISE_TARGET_CACHE.get(cache_key)
    if (
        prepared_target is not None
        and prepared_target.rep_ref is not None
        and prepared_target.rep_ref() is not rep2
    ):
        prepared_target = None
        _PAIRWISE_TARGET_CACHE.pop(cache_key, None)
    if prepared_target is None:
        qi2 = make_query_index(
            rep2,
            match_v=match_v,
            match_j=match_j,
            overlap_space=overlap_space,
        )
        keys2 = list(qi2.keys())
        dc2 = [qi2[k] for k in keys2]
        total_dc2 = sum(dc2) or 1

        trie = None
        if metric != "exact" and threshold > 0 and qi2 and Trie is not None:
            try:
                trie = Trie(
                    sequences=[k[0] for k in keys2],
                    vGenes=[k[1] for k in keys2],
                    jGenes=[k[2] for k in keys2],
                )
            except Exception:
                trie = None

        rep_ref = None
        try:
            rep_ref = weakref.ref(rep2)
        except TypeError:
            rep_ref = None

        prepared_target = _PreparedTarget(
            rep_ref=rep_ref,
            qi2=qi2,
            keys2=keys2,
            dc2=dc2,
            total_dc2=total_dc2,
            trie=trie,
            simpson_d2=_simpson_lambda(dc2, total_dc2),
        )
        # Keep cache bounded or disabled (size 0).
        if _PAIRWISE_TARGET_CACHE_MAX > 0 and rep_ref is not None:
            while len(_PAIRWISE_TARGET_CACHE) >= _PAIRWISE_TARGET_CACHE_MAX:
                _PAIRWISE_TARGET_CACHE.pop(next(iter(_PAIRWISE_TARGET_CACHE)))
            _PAIRWISE_TARGET_CACHE[cache_key] = prepared_target

    if metric == "exact" or threshold == 0:
        total_dc1 = sum(qi1.values()) or 1
        D1 = _simpson_lambda(list(qi1.values()), total_dc1)
        return _compute_exact_pairwise(
            qi1,
            prepared_target.qi2,
            total_dc1=total_dc1,
            total_dc2=prepared_target.total_dc2,
            D1=D1,
            D2=prepared_target.simpson_d2,
        )
    return _compute_trie_pairwise(
        qi1,
        prepared_target.qi2,
        metric=metric,
        threshold=threshold,
        n_jobs=n_jobs,
        prepared_target=prepared_target,
    )


# ---------------------------------------------------------------------------
# Parallel pairwise matrix — worker state
# ---------------------------------------------------------------------------

_MATRIX_QI_LIST: list | None = None
_MATRIX_DISPATCH_PARAMS: dict | None = None


def _matrix_worker_init(qi_list: list, params: dict) -> None:
    global _MATRIX_QI_LIST, _MATRIX_DISPATCH_PARAMS
    _MATRIX_QI_LIST = qi_list
    _MATRIX_DISPATCH_PARAMS = params


def _matrix_worker_call(pair: tuple[int, int]) -> dict:
    i, j = pair
    qi1 = _MATRIX_QI_LIST[i]
    qi2 = _MATRIX_QI_LIST[j]
    p = _MATRIX_DISPATCH_PARAMS
    if p["metric"] == "exact" or p["threshold"] == 0:
        r = _compute_exact_pairwise(qi1, qi2)
    else:
        r = _compute_trie_pairwise(qi1, qi2, metric=p["metric"], threshold=p["threshold"])
    return r.as_dict()


def pairwise_overlap_matrix(
    repertoires: list[LocusRepertoire],
    sample_ids: list[str] | None = None,
    *,
    metric: str = "exact",
    threshold: int = 0,
    match_v: bool = True,
    match_j: bool = True,
    overlap_space: str | None = None,
    n_jobs: int = -1,
) -> "pandas.DataFrame":
    """Compute all pairwise overlap metrics for a list of repertoires.

    Parameters
    ----------
    repertoires :
        List of :class:`~mir.common.repertoire.LocusRepertoire` objects.
    sample_ids :
        Optional sample identifiers (same length as *repertoires*).
        Defaults to ``"s0"``, ``"s1"``, … when ``None``.
    metric :
        ``"exact"``, ``"hamming"``, or ``"levenshtein"``.
    threshold :
        Edit-distance threshold (0 = exact).
    match_v, match_j :
        Gene-matching flags passed through to :func:`pairwise_overlap`.
    overlap_space :
        Identity space: ``"ntvj"``, ``"nt"``, ``"aavj"``, or ``"aa"``.
        Approximate matching is supported only for ``"aa"``/``"aavj"``.
    n_jobs :
        Parallel worker processes.  ``-1`` → all physical cores.
        Parallelism is across *pairs* (not within a single pair).

    Returns
    -------
    pandas.DataFrame
        Long-format table with one row per ordered pair (i, j) with i < j,
        columns: ``sample_id_1``, ``sample_id_2``, then all
        :class:`PairwiseOverlapResult` metric fields.

    Examples
    --------
    >>> df = pairwise_overlap_matrix(reps, sample_ids=ids, n_jobs=-1)
    >>> dist = df.pivot(index="sample_id_1", columns="sample_id_2", values="f_similarity")
    """
    import pandas as pd

    n = len(repertoires)
    if n < 2:
        raise ValueError("Need at least 2 repertoires for a pairwise matrix.")

    ids = sample_ids if sample_ids is not None else [f"s{i}" for i in range(n)]
    if len(ids) != n:
        raise ValueError("sample_ids length must match repertoires length.")

    overlap_space = _resolve_overlap_space(overlap_space, match_v=match_v, match_j=match_j)
    _allow_mismatches(overlap_space, metric, threshold)
    n_jobs = _n_jobs_resolve(n_jobs)

    # Serialize all repertoires to plain dicts once (picklable for workers).
    qi_list = [
        make_query_index(
            r,
            match_v=match_v,
            match_j=match_j,
            overlap_space=overlap_space,
        )
        for r in repertoires
    ]

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    params = {"metric": metric, "threshold": threshold}

    if n_jobs == 1 or len(pairs) <= 1:
        results_raw = []
        for i, j in pairs:
            if metric == "exact" or threshold == 0:
                r = _compute_exact_pairwise(qi_list[i], qi_list[j])
            else:
                r = _compute_trie_pairwise(qi_list[i], qi_list[j], metric=metric, threshold=threshold)
            results_raw.append((i, j, r.as_dict()))
    else:
        chunksize = max(1, len(pairs) // (n_jobs * 4))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=_MP_CTX,
            initializer=_matrix_worker_init,
            initargs=(qi_list, params),
        ) as pool:
            raw_dicts = list(pool.map(_matrix_worker_call, pairs, chunksize=chunksize))
        results_raw = [(i, j, d) for (i, j), d in zip(pairs, raw_dicts)]

    rows = []
    for i, j, d in results_raw:
        rows.append({"sample_id_1": ids[i], "sample_id_2": ids[j], **d})

    return pd.DataFrame(rows)


