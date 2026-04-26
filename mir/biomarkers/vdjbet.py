"""VDJBet: Pgen-matched mock repertoire generation and overlap analysis.

Generate a null-model (mock) repertoire that mirrors the log₁₀ Pgen
histogram of an input repertoire.

Algorithm
---------
1. Compute Pgen for every clonotype; bin by ``⌊log₁₀ Pgen⌋``.
2. Sample from the OLGA null model; accept each generated sequence when
   its bin is not yet full, discard otherwise.
3. The :class:`VDJBetOverlapAnalysis` class manages a growing cache of
   OLGA-generated sequences.  For each mock the cache is traversed first;
   if a bin still needs sequences, new ones are generated on-the-fly and
   added to the cache for future use.  The cache grows up to
   *max_cache_size* sequences in total.

V/J gene bias correction
------------------------
Pass a :class:`~mir.basic.pgen.PgenGeneUsageAdjustment` to
:class:`VDJBetOverlapAnalysis` to re-weight each generated sequence's
Pgen by its V-J factor (target usage / OLGA usage).  This calibrates the
mock null to match the target repertoire's V/J gene usage and eliminates
protocol-induced V/J bias without explicit V/J stratification of the
histogram.  This is the recommended approach; the old *fix_v_usage* /
*fix_j_usage* options in the lower-level functions are retained for
backward compatibility only.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union

import numpy as np
from scipy.stats import norm as _scipy_norm

from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.repertoire import Repertoire
from mir.comparative.overlap import (
    compute_overlaps,
    count_overlap,
    make_query_index,
    make_reference_keys,
)

# Pool record: dict with junction_aa, v_gene, j_gene, pgen (log10 Pgen),
# junction (nt), v_end, j_start — as returned by generate_sequences_with_meta.
_PoolRecord = dict

# Bin key is either a plain int or a tuple of (gene(s), int).
_BinKey = Union[int, tuple]

_MIN_BATCH: int = 100
_MAX_BATCH: int = 10_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_allele(gene: str) -> str:
    """``"TRBV1*01"`` → ``"TRBV1"``."""
    return gene.split("*")[0]


def _make_key(
    v_gene: str,
    j_gene: str,
    bin_val: int,
    fix_v: bool,
    fix_j: bool,
) -> _BinKey:
    """Build a histogram bin key from gene names and the log₁₀ Pgen bin."""
    if fix_v and fix_j:
        return (v_gene, j_gene, bin_val)
    if fix_v:
        return (v_gene, bin_val)
    if fix_j:
        return (j_gene, bin_val)
    return bin_val


def _resolve_locus(repertoire: Repertoire) -> str:
    """Return the IMGT locus code for *repertoire*.

    Checks ``repertoire.locus`` first, then inspects clonotype fields.

    Raises
    ------
    ValueError
        When the locus cannot be determined.
    """
    if repertoire.locus:
        return repertoire.locus
    from mir.common.repertoire import infer_locus
    for clone in repertoire.clonotypes:
        if clone.locus:
            return clone.locus
        if clone.v_gene:
            loc = infer_locus(clone.v_gene)
            if loc:
                return loc
    raise ValueError(
        "Cannot determine locus from repertoire. "
        "Set Repertoire.locus or populate Clonotype.locus / v_gene fields."
    )


def compute_pgen_histogram(
    clonotypes: list[Clonotype],
    model: OlgaModel,
    *,
    fix_v: bool = False,
    fix_j: bool = False,
) -> dict[_BinKey, int]:
    """Build a Pgen histogram for *clonotypes*.

    Clonotypes with zero or undefined Pgen are silently skipped.

    Parameters
    ----------
    clonotypes:
        Source clonotypes.
    model:
        OLGA model used to compute Pgen.
    fix_v:
        Include V-gene in bin keys.
    fix_j:
        Include J-gene in bin keys.

    Returns
    -------
    dict
        Maps each bin key to the number of clonotypes in that bin.
    """
    hist: dict[_BinKey, int] = defaultdict(int)
    for clone in clonotypes:
        pgen_val = model.compute_pgen_junction_aa(clone.junction_aa)
        if pgen_val is None or pgen_val <= 0:
            continue
        bin_val = int(math.floor(math.log10(pgen_val)))
        key = _make_key(
            _strip_allele(clone.v_gene),
            _strip_allele(clone.j_gene),
            bin_val,
            fix_v,
            fix_j,
        )
        hist[key] += 1
    return dict(hist)


# ---------------------------------------------------------------------------
# Public API — lower-level functions (retained for backward compat)
# ---------------------------------------------------------------------------

def generate_mock_repertoire(
    repertoire: Repertoire,
    *,
    fix_v_usage: bool = False,
    fix_j_usage: bool = False,
    max_sequences: int = 10_000_000,
    seed: int = 42,
    pgen_adjustment=None,
) -> Repertoire:
    """Generate a Pgen-matched mock repertoire for *repertoire*.

    Parameters
    ----------
    repertoire:
        Input repertoire.  Its locus is used to select the OLGA model.
    fix_v_usage:
        When ``True``, match V-gene usage within each Pgen bin.
        Not recommended when *pgen_adjustment* is supplied — use V/J
        matching in :func:`~mir.comparative.overlap.count_overlap` instead.
    fix_j_usage:
        When ``True``, match J-gene usage within each Pgen bin.
    max_sequences:
        Cap on OLGA sequences to generate before giving up.
    seed:
        numpy RNG seed for reproducibility.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.  When
        supplied, Pgen values are multiplied by the V-J factor before binning
        so the mock reflects the target V-J gene usage distribution.

    Returns
    -------
    Repertoire
        Mock repertoire with the same Pgen distribution as *repertoire*.

    Warns
    -----
    UserWarning
        When *max_sequences* is exhausted before all bins are filled.
    """
    if not repertoire.clonotypes:
        return Repertoire(clonotypes=[], locus=repertoire.locus,
                          repertoire_id=f"mock_{repertoire.repertoire_id}")

    locus = _resolve_locus(repertoire)
    model = OlgaModel(locus=locus, seed=seed)

    # --- build target histogram and collect duplicate_counts in one pass ----
    target: dict[_BinKey, int] = {}
    valid_duplicate_counts: list[int] = []

    for clone in repertoire.clonotypes:
        pgen_val = model.compute_pgen_junction_aa(clone.junction_aa)
        if pgen_val is None or pgen_val <= 0:
            continue
        if pgen_adjustment is not None:
            pgen_val = pgen_adjustment.adjust_pgen(locus, clone.v_gene or "", clone.j_gene or "", pgen_val)
            if pgen_val <= 0:
                continue
        bin_val = int(math.floor(math.log10(pgen_val)))
        key = _make_key(
            _strip_allele(clone.v_gene),
            _strip_allele(clone.j_gene),
            bin_val, fix_v_usage, fix_j_usage,
        )
        target[key] = target.get(key, 0) + 1
        valid_duplicate_counts.append(clone.duplicate_count)

    if not target:
        warnings.warn(
            "generate_mock_repertoire: all clonotypes have zero or undefined "
            "Pgen; returning empty mock repertoire.",
            UserWarning,
            stacklevel=2,
        )
        return Repertoire(clonotypes=[], locus=locus,
                          repertoire_id=f"mock_{repertoire.repertoire_id}")

    np.random.default_rng(seed).shuffle(valid_duplicate_counts)
    count_iter = iter(valid_duplicate_counts)

    remaining: dict[_BinKey, int] = dict(target)
    mock_clonotypes: list[Clonotype] = []
    total_generated = 0

    while remaining and total_generated < max_sequences:
        still_needed = sum(remaining.values())
        batch_n = min(
            max(still_needed * 5, _MIN_BATCH),
            _MAX_BATCH,
            max_sequences - total_generated,
        )
        batch = model.generate_sequences_with_meta(
            batch_n, pgens=True, seed=None, pgen_adjustment=pgen_adjustment
        )
        total_generated += len(batch)

        for rec in batch:
            if not remaining:
                break
            log_pgen = rec["pgen"]
            if math.isinf(log_pgen):
                continue
            bin_val = int(math.floor(log_pgen))
            v = _strip_allele(rec["v_gene"]) if fix_v_usage else ""
            j = _strip_allele(rec["j_gene"]) if fix_j_usage else ""
            key = _make_key(v, j, bin_val, fix_v_usage, fix_j_usage)
            if key not in remaining:
                continue
            remaining[key] -= 1
            if remaining[key] == 0:
                del remaining[key]
            mock_clonotypes.append(
                Clonotype(
                    sequence_id=str(len(mock_clonotypes)),
                    locus=locus,
                    junction_aa=rec["junction_aa"],
                    junction=rec["junction"],
                    v_gene=rec["v_gene"],
                    j_gene=rec["j_gene"],
                    v_sequence_end=rec["v_end"],
                    j_sequence_start=rec["j_start"],
                    duplicate_count=next(count_iter),
                    _validate=False,
                )
            )

    if remaining:
        n_missing = sum(remaining.values())
        warnings.warn(
            f"generate_mock_repertoire: exhausted {max_sequences:,} generated "
            f"sequences but {n_missing} mock clonotype(s) are still missing "
            f"({len(remaining)} unfilled bin(s)).  "
            "Consider raising max_sequences or relaxing fix_v/fix_j constraints.",
            UserWarning,
            stacklevel=2,
        )

    return Repertoire(
        clonotypes=mock_clonotypes,
        locus=locus,
        repertoire_id=f"mock_{repertoire.repertoire_id}",
    )


def generate_mock_from_pool(
    repertoire: Repertoire,
    pool: list[_PoolRecord],
    *,
    fix_v_usage: bool = False,
    fix_j_usage: bool = False,
    seed: int = 42,
    pgen_adjustment=None,
) -> Repertoire:
    """Generate a Pgen-matched mock repertoire by sampling a pre-computed pool.

    Unlike :func:`generate_mock_repertoire`, which runs OLGA sampling on the
    fly, this function draws from *pool* — a list of pre-generated sequences
    with ``pgen`` (log₁₀) values.  This is faster when many mocks are needed
    for the same target but requires a large enough pool to avoid replacement
    sampling.

    For the primary analysis pipeline prefer :class:`VDJBetOverlapAnalysis`,
    which manages a growing cache automatically.

    Parameters
    ----------
    repertoire:
        Input repertoire whose Pgen histogram is matched.
    pool:
        Pre-generated OLGA sequences (each a dict with at least
        ``junction_aa``, ``v_gene``, ``j_gene``, ``pgen`` (log₁₀)).
    fix_v_usage, fix_j_usage:
        Match V/J gene usage within each Pgen bin (not recommended when
        *pgen_adjustment* is supplied).
    seed:
        Seed for the NumPy RNG used when drawing from the pool.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.
    """
    if not repertoire.clonotypes:
        return Repertoire(clonotypes=[], locus=repertoire.locus,
                          repertoire_id=f"mock_{repertoire.repertoire_id}")

    locus = _resolve_locus(repertoire)
    model = OlgaModel(locus=locus, seed=seed)

    pool_by_key: dict[_BinKey, list[_PoolRecord]] = defaultdict(list)
    for rec in pool:
        log_pgen = rec.get("pgen", float("-inf"))
        if math.isinf(log_pgen):
            continue
        bin_val = int(math.floor(log_pgen))
        key = _make_key(
            _strip_allele(rec.get("v_gene", "")),
            _strip_allele(rec.get("j_gene", "")),
            bin_val, fix_v_usage, fix_j_usage,
        )
        pool_by_key[key].append(rec)

    target: dict[_BinKey, int] = {}
    valid_duplicate_counts: list[int] = []

    for clone in repertoire.clonotypes:
        pgen_val = model.compute_pgen_junction_aa(clone.junction_aa)
        if pgen_val is None or pgen_val <= 0:
            continue
        if pgen_adjustment is not None:
            pgen_val = pgen_adjustment.adjust_pgen(locus, clone.v_gene or "", clone.j_gene or "", pgen_val)
            if pgen_val <= 0:
                continue
        bin_val = int(math.floor(math.log10(pgen_val)))
        key = _make_key(
            _strip_allele(clone.v_gene),
            _strip_allele(clone.j_gene),
            bin_val, fix_v_usage, fix_j_usage,
        )
        target[key] = target.get(key, 0) + 1
        valid_duplicate_counts.append(clone.duplicate_count)

    if not target:
        warnings.warn(
            "generate_mock_from_pool: all clonotypes have zero or undefined "
            "Pgen; returning empty mock repertoire.",
            UserWarning,
            stacklevel=2,
        )
        return Repertoire(clonotypes=[], locus=locus,
                          repertoire_id=f"mock_{repertoire.repertoire_id}")

    np.random.default_rng(seed).shuffle(valid_duplicate_counts)
    count_iter = iter(valid_duplicate_counts)

    rng = np.random.default_rng(seed)
    mock_clonotypes: list[Clonotype] = []
    missing_keys: list[_BinKey] = []

    for key, n_needed in target.items():
        available = pool_by_key.get(key, [])
        if not available:
            missing_keys.append(key)
            for _ in range(n_needed):
                next(count_iter, None)
            continue
        replace = len(available) < n_needed
        if replace:
            warnings.warn(
                f"generate_mock_from_pool: bin {key!r} needs {n_needed} sequences "
                f"but pool only has {len(available)}; sampling with replacement.",
                UserWarning,
                stacklevel=2,
            )
        idxs = rng.choice(len(available), n_needed, replace=replace)
        for idx in idxs:
            rec = available[int(idx)]
            mock_clonotypes.append(
                Clonotype(
                    sequence_id=str(len(mock_clonotypes)),
                    locus=locus,
                    junction_aa=rec["junction_aa"],
                    junction=rec.get("junction", ""),
                    v_gene=rec.get("v_gene", ""),
                    j_gene=rec.get("j_gene", ""),
                    v_sequence_end=rec.get("v_end", -1),
                    j_sequence_start=rec.get("j_start", -1),
                    duplicate_count=next(count_iter, 1),
                    _validate=False,
                )
            )

    if missing_keys:
        n_missing = sum(target[k] for k in missing_keys)
        warnings.warn(
            f"generate_mock_from_pool: {n_missing} mock clonotype(s) could not be "
            f"filled ({len(missing_keys)} bin(s) absent from pool).  "
            "Consider using a larger pool.",
            UserWarning,
            stacklevel=2,
        )

    return Repertoire(
        clonotypes=mock_clonotypes,
        locus=locus,
        repertoire_id=f"mock_{repertoire.repertoire_id}",
    )


def build_olga_pool(
    locus: str,
    n: int,
    *,
    seed: int = 42,
    pgen_adjustment=None,
) -> list[_PoolRecord]:
    """Generate a pool of OLGA sequences with pre-computed log₁₀ Pgen values.

    Parameters
    ----------
    locus:
        IMGT locus code (e.g. ``"TRB"``, ``"TRA"``, ``"IGH"``).
    n:
        Number of sequences to generate.
    seed:
        RNG seed for reproducibility.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.

    Returns
    -------
    list[dict]
        Each record contains at least ``junction_aa``, ``v_gene``, ``j_gene``,
        ``pgen`` (log₁₀ Pgen), ``junction`` (nucleotide), ``v_end``, ``j_start``.
    """
    model = OlgaModel(locus=locus, seed=seed)
    return model.generate_sequences_with_meta(n, pgens=True, seed=seed, pgen_adjustment=pgen_adjustment)


def generate_mock_key_sets_from_pool(
    reference_repertoire: Repertoire,
    pool: list[_PoolRecord],
    n_mocks: int,
    *,
    fix_v_usage: bool = False,
    fix_j_usage: bool = False,
    seed: int = 42,
    pgen_adjustment=None,
) -> list[frozenset[tuple[str, str, str]]]:
    """Generate *n_mocks* Pgen-matched mock reference key sets from a pool.

    This is the batch counterpart of :func:`generate_mock_from_pool`.  Instead
    of building full :class:`Clonotype` / :class:`Repertoire` objects, it
    returns frozensets of ``(junction_aa, v_base, j_base)`` keys — the exact
    format consumed by :func:`~mir.comparative.overlap.count_overlap`.

    For the primary analysis pipeline prefer :class:`VDJBetOverlapAnalysis`,
    which manages a growing cache and pgen_adjustment automatically.

    Parameters
    ----------
    reference_repertoire:
        Reference repertoire whose Pgen histogram is matched.
    pool:
        Pre-generated OLGA sequences from :func:`build_olga_pool`.
    n_mocks:
        Number of mock key sets to generate.
    fix_v_usage, fix_j_usage:
        Match V/J gene usage within each Pgen bin.
    seed:
        Seed for the NumPy RNG.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.
    """
    locus = _resolve_locus(reference_repertoire)
    model = OlgaModel(locus=locus, seed=seed)

    pool_by_key: dict[_BinKey, list[_PoolRecord]] = defaultdict(list)
    for rec in pool:
        log_pgen = rec.get("pgen", float("-inf"))
        if math.isinf(log_pgen):
            continue
        bin_val = int(math.floor(log_pgen))
        key = _make_key(
            _strip_allele(rec.get("v_gene", "")),
            _strip_allele(rec.get("j_gene", "")),
            bin_val, fix_v_usage, fix_j_usage,
        )
        pool_by_key[key].append(rec)

    ref_bins: dict[_BinKey, list[Clonotype]] = defaultdict(list)
    for clone in reference_repertoire.clonotypes:
        pgen_val = model.compute_pgen_junction_aa(clone.junction_aa)
        if pgen_val is None or pgen_val <= 0:
            continue
        if pgen_adjustment is not None:
            pgen_val = pgen_adjustment.adjust_pgen(locus, clone.v_gene or "", clone.j_gene or "", pgen_val)
            if pgen_val <= 0:
                continue
        bin_val = int(math.floor(math.log10(pgen_val)))
        key = _make_key(
            _strip_allele(clone.v_gene or ""),
            _strip_allele(clone.j_gene or ""),
            bin_val, fix_v_usage, fix_j_usage,
        )
        ref_bins[key].append(clone)

    if not ref_bins:
        warnings.warn(
            "generate_mock_key_sets_from_pool: all reference clonotypes have "
            "zero or undefined Pgen; returning empty mock key sets.",
            UserWarning,
            stacklevel=2,
        )
        return [frozenset() for _ in range(n_mocks)]

    absent_bins = [key for key in ref_bins if key not in pool_by_key]
    replacement_bins = [
        key for key in ref_bins
        if key in pool_by_key and len(pool_by_key[key]) < len(ref_bins[key])
    ]
    if absent_bins:
        warnings.warn(
            f"generate_mock_key_sets_from_pool: {len(absent_bins)} bin(s) are "
            f"entirely absent from the pool and will be skipped.  "
            "Consider using a larger pool.",
            UserWarning,
            stacklevel=2,
        )
    elif replacement_bins:
        warnings.warn(
            f"generate_mock_key_sets_from_pool: {len(replacement_bins)} bin(s) have "
            f"fewer pool sequences than needed; sampling with replacement.  "
            "Consider using a larger pool.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    mock_key_sets: list[frozenset[tuple[str, str, str]]] = []

    for _ in range(n_mocks):
        keys: set[tuple[str, str, str]] = set()
        for bin_key, clones in ref_bins.items():
            available = pool_by_key.get(bin_key, [])
            if not available:
                continue
            n_needed = len(clones)
            replace = len(available) < n_needed
            idxs = rng.choice(len(available), n_needed, replace=replace)
            for idx in idxs:
                rec = available[int(idx)]
                keys.add((
                    rec["junction_aa"],
                    rec.get("v_gene", "").split("*")[0],
                    rec.get("j_gene", "").split("*")[0],
                ))
        mock_key_sets.append(frozenset(keys))

    return mock_key_sets


# ---------------------------------------------------------------------------
# Internal helper for mock key normalisation
# ---------------------------------------------------------------------------

def _normalize_mock_keys(
    mock_key_sets: list[frozenset],
    *,
    match_v: bool,
    match_j: bool,
) -> list[frozenset]:
    """Strip V/J gene fields from mock keys when the corresponding match flag is off."""
    if match_v and match_j:
        return mock_key_sets
    return [
        frozenset(
            (jaa, v if match_v else "", j if match_j else "")
            for jaa, v, j in ks
        )
        for ks in mock_key_sets
    ]


# ---------------------------------------------------------------------------
# Overlap result
# ---------------------------------------------------------------------------

@dataclass
class OverlapResult:
    """Overlap statistics for one query sample under one set of match options.

    Produced by :meth:`VDJBetOverlapAnalysis.score`.  Per-mock distributions
    are stored so the object is self-contained; z/p-scores are computed lazily.

    Attributes
    ----------
    n_total, dc_total:
        Total unique clonotypes and total duplicate count in the query.
    n, dc:
        Unique clonotypes and cells overlapping the reference.
    mock_n, mock_dc:
        Per-mock overlap counts (length == n_mocks).
    allow_1mm:
        Whether 1-substitution CDR3 matching was used.
    match_v, match_j:
        Whether V-gene / J-gene matching was required for the overlap.
    """

    n_total: int
    dc_total: int
    n: int
    dc: int
    mock_n: list[int]
    mock_dc: list[int]
    allow_1mm: bool = False
    match_v: bool = True
    match_j: bool = True

    @staticmethod
    def _z_p(real: float, mocks: list) -> tuple[float, float]:
        arr = np.asarray(mocks, dtype=float)
        mean, std = arr.mean(), arr.std()
        if std == 0:
            return (float("inf") if real > mean else 0.0), (
                0.0 if real > mean else 1.0
            )
        z = float((real - mean) / std)
        return z, float(1.0 - _scipy_norm.cdf(z))

    @property
    def frac_n(self) -> float:
        """Fraction of query clonotypes overlapping the reference."""
        return self.n / self.n_total if self.n_total else 0.0

    @property
    def frac_dc(self) -> float:
        """Fraction of query cells overlapping the reference."""
        return self.dc / self.dc_total if self.dc_total else 0.0

    @cached_property
    def _zp_n(self) -> tuple[float, float]:
        return self._z_p(self.n, self.mock_n)

    @property
    def z_n(self) -> float:
        """Z-score for unique-clonotype overlap vs the null distribution."""
        return self._zp_n[0]

    @property
    def p_n(self) -> float:
        """One-sided p-value for unique-clonotype overlap (upper tail)."""
        return self._zp_n[1]

    @cached_property
    def _dc_log2(self) -> float:
        return math.log2(self.dc + 1)

    @cached_property
    def _mock_dc_log2(self) -> list[float]:
        return [math.log2(x + 1) for x in self.mock_dc]

    @cached_property
    def _zp_dc(self) -> tuple[float, float]:
        return self._z_p(self._dc_log2, self._mock_dc_log2)

    @property
    def z_dc(self) -> float:
        """Z-score for duplicate-count overlap (log₂-transformed)."""
        return self._zp_dc[0]

    @property
    def p_dc(self) -> float:
        """One-sided p-value for duplicate-count overlap (log₂, upper tail)."""
        return self._zp_dc[1]


# ---------------------------------------------------------------------------
# Analysis class — growing cache
# ---------------------------------------------------------------------------

class VDJBetOverlapAnalysis:
    """Manages an epitope-specific reference with Pgen-matched mock null sets.

    On the first :meth:`score` call the class builds a growing cache of
    OLGA-generated sequences.  For each mock it first draws from the cache; if
    a Pgen bin still needs sequences, new ones are generated on-the-fly and
    added to the cache for future use.  The cache never exceeds
    *max_cache_size* sequences.

    V/J gene bias (e.g. from sequencing protocol differences) should be
    corrected via *pgen_adjustment* rather than V/J histogram stratification.
    With adjustment, the mock null reflects the target's V/J distribution and
    a plain Pgen histogram (no V/J fixing) is sufficient.

    Parameters
    ----------
    reference:
        Epitope-specific reference repertoire (e.g. VDJdb LLW TRB clonotypes).
    n_mocks:
        Number of Pgen-matched mock key sets to generate.  100 is usually
        sufficient for reliable z-scores.
    cache_size:
        When given, pre-generate this many OLGA sequences into the cache
        before the first :meth:`score` call.  ``None`` (default) lets the
        cache grow lazily on demand.
    max_cache_size:
        Hard cap on total OLGA sequences cached.  On-the-fly generation stops
        when this limit is reached even if some histogram bins are still
        under-represented.
    seed:
        RNG seed for reproducibility.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.  When
        supplied, each generated sequence's Pgen is multiplied by its V-J
        factor before binning so that the null distribution matches the target
        V/J gene usage.  **Recommended** when analysing samples from a
        specific sequencing protocol.

    Examples
    --------
    >>> from mir.basic.gene_usage import GeneUsage
    >>> from mir.basic.pgen import PgenGeneUsageAdjustment
    >>> # Build adjustment from target (e.g. all query samples combined)
    >>> gu = GeneUsage.from_list(query_samples)
    >>> adj = PgenGeneUsageAdjustment(gu)
    >>> analysis = VDJBetOverlapAnalysis(llw_ref, n_mocks=100, pgen_adjustment=adj)
    >>> result = analysis.score(query_sample)
    >>> print(result.z_n, result.p_n)
    """

    def __init__(
        self,
        reference: Repertoire,
        *,
        n_mocks: int = 100,
        cache_size: int | None = None,
        max_cache_size: int = 10_000_000,
        seed: int = 42,
        pgen_adjustment=None,
    ) -> None:
        self._reference = reference
        self._n_mocks = n_mocks
        self._cache_size = cache_size
        self._max_cache_size = max_cache_size
        self._seed = seed
        self._pgen_adjustment = pgen_adjustment

        # Lazy state — populated on first use
        self._locus: str | None = None
        self._model: OlgaModel | None = None
        self._cache: list[_PoolRecord] = []
        # bin value → list of indices into _cache (includes valid pgen records only)
        self._cache_by_bin: dict[int, list[int]] = defaultdict(list)
        self._mock_key_sets: list[frozenset] | None = None

        if cache_size is not None:
            self._ensure_cache(cache_size)

    # ------------------------------------------------------------------
    # Internal lazy helpers
    # ------------------------------------------------------------------

    def _get_locus(self) -> str:
        if self._locus is None:
            self._locus = _resolve_locus(self._reference)
        return self._locus

    def _get_model(self) -> OlgaModel:
        if self._model is None:
            self._model = OlgaModel(locus=self._get_locus(), seed=self._seed)
        return self._model

    def _add_records(self, records: list[_PoolRecord]) -> None:
        """Append records to the cache and update the bin index."""
        for rec in records:
            log_pgen = rec.get("pgen", float("-inf"))
            if math.isinf(log_pgen):
                continue
            idx = len(self._cache)
            self._cache.append(rec)
            bin_val = int(math.floor(log_pgen))
            self._cache_by_bin[bin_val].append(idx)

    def _grow_cache(self, n: int) -> None:
        """Generate *n* more OLGA sequences and add to cache."""
        model = self._get_model()
        records = model.generate_sequences_with_meta(
            n, pgens=True, seed=None,
            pgen_adjustment=self._pgen_adjustment,
        )
        self._add_records(records)

    def _ensure_cache(self, target_size: int) -> None:
        """Pre-fill cache up to *target_size* (bounded by max_cache_size)."""
        need = min(target_size, self._max_cache_size) - len(self._cache)
        if need > 0:
            self._grow_cache(need)

    def _compute_ref_bins(self) -> dict[int, list[Clonotype]]:
        """Bin reference clonotypes by ⌊log₁₀ Pgen⌋ (pgen-only, no V/J)."""
        locus = self._get_locus()
        model = self._get_model()
        ref_bins: dict[int, list[Clonotype]] = defaultdict(list)
        for clone in self._reference.clonotypes:
            pgen_val = model.compute_pgen_junction_aa(clone.junction_aa)
            if pgen_val is None or pgen_val <= 0:
                continue
            if self._pgen_adjustment is not None:
                pgen_val = self._pgen_adjustment.adjust_pgen(
                    locus, clone.v_gene or "", clone.j_gene or "", pgen_val
                )
                if pgen_val <= 0:
                    continue
            bin_val = int(math.floor(math.log10(pgen_val)))
            ref_bins[bin_val].append(clone)
        return dict(ref_bins)

    def _grow_for_bins(self, ref_bins: dict[int, list]) -> None:
        """Grow cache until every bin can serve *n_mocks* draws without replacement.

        Stops early if *max_cache_size* is reached.
        """
        batch = 10_000
        while True:
            if len(self._cache) >= self._max_cache_size:
                break
            all_ok = all(
                len(self._cache_by_bin.get(b, [])) >= len(clones) * self._n_mocks
                for b, clones in ref_bins.items()
            )
            if all_ok:
                break
            self._grow_cache(min(batch, self._max_cache_size - len(self._cache)))

    def _build_mock_key_sets(self) -> list[frozenset]:
        """Build *n_mocks* mock key sets using the growing cache."""
        ref_bins = self._compute_ref_bins()

        if not ref_bins:
            warnings.warn(
                "VDJBetOverlapAnalysis: all reference clonotypes have zero or "
                "undefined Pgen; mock key sets will be empty.",
                UserWarning,
                stacklevel=3,
            )
            return [frozenset() for _ in range(self._n_mocks)]

        # Grow cache as needed
        self._grow_for_bins(ref_bins)

        # Warn about bins that still lack sufficient sequences
        absent = [b for b in ref_bins if b not in self._cache_by_bin]
        under = [
            b for b in ref_bins
            if b in self._cache_by_bin
            and len(self._cache_by_bin[b]) < len(ref_bins[b]) * self._n_mocks
        ]
        if absent:
            warnings.warn(
                f"VDJBetOverlapAnalysis: {len(absent)} Pgen bin(s) absent from "
                f"cache after {len(self._cache):,} total sequences; those bins "
                "will be skipped.  Consider increasing max_cache_size.",
                UserWarning,
                stacklevel=3,
            )
        elif under:
            warnings.warn(
                f"VDJBetOverlapAnalysis: {len(under)} Pgen bin(s) have fewer cache "
                f"sequences than needed for {self._n_mocks} replacement-free mocks; "
                "sampling with replacement.  Consider increasing max_cache_size.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self._seed)
        mock_key_sets: list[frozenset] = []

        for _ in range(self._n_mocks):
            keys: set[tuple[str, str, str]] = set()
            for bin_val, clones in ref_bins.items():
                indices = self._cache_by_bin.get(bin_val, [])
                if not indices:
                    continue
                n_needed = len(clones)
                replace = len(indices) < n_needed
                chosen = rng.choice(len(indices), n_needed, replace=replace)
                for ci in chosen:
                    rec = self._cache[indices[int(ci)]]
                    keys.add((
                        rec["junction_aa"],
                        rec.get("v_gene", "").split("*")[0],
                        rec.get("j_gene", "").split("*")[0],
                    ))
            mock_key_sets.append(frozenset(keys))

        return mock_key_sets

    def _get_mock_key_sets(self) -> list[frozenset]:
        if self._mock_key_sets is None:
            self._mock_key_sets = self._build_mock_key_sets()
        return self._mock_key_sets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        sample: Repertoire,
        *,
        allow_1mm: bool = False,
        match_v: bool = True,
        match_j: bool = True,
        n_jobs: int = 1,
    ) -> OverlapResult:
        """Compute overlap statistics for *sample* against the reference.

        The mock null distribution is built once (and cached) the first time
        this method is called.  Subsequent calls with different *allow_1mm*,
        *match_v*, or *match_j* values reuse the same mock key sets — only the
        query normalization differs.

        Parameters
        ----------
        sample:
            Query repertoire.
        allow_1mm:
            Count clonotypes within one amino-acid substitution of the
            reference CDR3 (in addition to exact matches).
        match_v:
            Require V-gene match for the query↔reference overlap.
            V/J matching in the overlap is preferred over V/J stratification
            of the mock null.
        match_j:
            Require J-gene match for the query↔reference overlap.
        n_jobs:
            Parallel worker processes for mock overlap computation.

        Returns
        -------
        OverlapResult
            Overlap statistics with z/p-scores computed from the mock null.
        """
        qi = make_query_index(sample, match_v=match_v, match_j=match_j)
        n_total  = len(qi)
        dc_total = sum(qi.values())

        ref_keys = make_reference_keys(
            self._reference, allow_1mm=False, match_v=match_v, match_j=match_j,
        )
        real = count_overlap(ref_keys, qi, allow_1mm=allow_1mm)

        raw_mocks  = self._get_mock_key_sets()
        norm_mocks = _normalize_mock_keys(raw_mocks, match_v=match_v, match_j=match_j)
        mock_res   = compute_overlaps(norm_mocks, qi, allow_1mm=allow_1mm, n_jobs=n_jobs)

        return OverlapResult(
            n_total=n_total,
            dc_total=dc_total,
            n=real.n,
            dc=real.dc,
            mock_n=[r.n for r in mock_res],
            mock_dc=[r.dc for r in mock_res],
            allow_1mm=allow_1mm,
            match_v=match_v,
            match_j=match_j,
        )
