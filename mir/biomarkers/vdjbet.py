"""VDJBet: Pgen-matched mock repertoire generation.

Generate a null-model (mock) repertoire that mirrors the log₁₀ Pgen
histogram of an input repertoire.  Optionally the mock can also reproduce
V-gene and/or J-gene usage within each Pgen bin.

Algorithm
---------
1. Compute Pgen for every clonotype; bin by ``⌊log₁₀ Pgen⌋``.
2. Optionally extend each bin key with the V-gene, J-gene, or both.
3. Sample from the OLGA null model in batches; accept each generated
   sequence when its bin is not yet full, discard otherwise.
4. Stop when all bins reach their target count or *max_sequences* have
   been generated.  Emit a :class:`UserWarning` for any unfilled bin.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
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
# Public API
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
    fix_j_usage:
        When ``True``, match J-gene usage within each Pgen bin.
    max_sequences:
        Cap on OLGA sequences to generate before giving up.
        Defaults to 10 000 000.
    seed:
        numpy RNG seed for reproducibility.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.  When
        supplied, Pgen values are multiplied by the V-J factor before binning
        so the mock reflects the target V-J gene usage distribution.

    Returns
    -------
    Repertoire
        Mock repertoire with the same Pgen (and optionally V/J) distribution
        as *repertoire*.  ``duplicate_count`` values from the original
        clonotypes (those with valid Pgen) are randomly shuffled and
        re-assigned to the mock clonotypes so that the count distribution is
        also preserved.  The repertoire may be smaller than *repertoire* if
        some clonotypes have zero Pgen or if *max_sequences* is exhausted.

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

    # Shuffle duplicate_counts using a dedicated RNG so OLGA's global
    # numpy state (used in acceptance-rejection sampling) is unaffected.
    np.random.default_rng(seed).shuffle(valid_duplicate_counts)
    count_iter = iter(valid_duplicate_counts)

    remaining: dict[_BinKey, int] = dict(target)

    # --- acceptance-rejection sampling ------------------------------------
    mock_clonotypes: list[Clonotype] = []
    total_generated = 0

    while remaining and total_generated < max_sequences:
        # Scale batch to remaining need: over-shoot by 5× so we fill bins
        # quickly without wasting pgen computation on fully-filled bins.
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
                )
            )

    # --- warn if histogram not fully filled --------------------------------
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

    Unlike :func:`generate_mock_repertoire`, which runs OLGA acceptance-rejection
    sampling on the fly, this function draws from *pool* — a list of pre-generated
    OLGA sequences that already carry ``pgen`` (log₁₀) values.  This is orders of
    magnitude faster when many mock repertoires are needed for the same target
    because no new sequence generation is required.

    The pool should be large enough to fill every Pgen bin without replacement.
    When a bin has fewer pool entries than required, sampling is done *with*
    replacement and a :class:`UserWarning` is emitted.  Bins entirely absent from
    the pool are silently skipped (same behaviour as
    :func:`generate_mock_repertoire` when budget is exhausted).

    Parameters
    ----------
    repertoire:
        Input repertoire whose Pgen histogram is matched.
    pool:
        Pre-generated OLGA sequences.  Each element must be a ``dict`` with at
        least ``junction_aa``, ``v_gene``, ``j_gene``, and ``pgen`` (log₁₀ Pgen
        as returned by
        :meth:`~mir.basic.pgen.OlgaModel.generate_sequences_with_meta`).
    fix_v_usage:
        Match V-gene usage within each Pgen bin.
    fix_j_usage:
        Match J-gene usage within each Pgen bin.
    seed:
        Seed for the NumPy RNG used when drawing from the pool.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.  When
        supplied, reference clonotype Pgen values are multiplied by the V-J
        factor before binning.  The *pool* is assumed to have been built with
        the same adjustment (via :func:`build_olga_pool`).

    Returns
    -------
    Repertoire
        Mock repertoire with the same Pgen distribution as *repertoire*.
    """
    if not repertoire.clonotypes:
        return Repertoire(clonotypes=[], locus=repertoire.locus,
                          repertoire_id=f"mock_{repertoire.repertoire_id}")

    locus = _resolve_locus(repertoire)
    model = OlgaModel(locus=locus, seed=seed)

    # --- index pool by bin key -----------------------------------------------
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

    # --- build target histogram and collect duplicate_counts -----------------
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

    # --- sample from pool by bin ---------------------------------------------
    rng = np.random.default_rng(seed)
    mock_clonotypes: list[Clonotype] = []
    missing_keys: list[_BinKey] = []

    for key, n_needed in target.items():
        available = pool_by_key.get(key, [])
        if not available:
            missing_keys.append(key)
            # consume duplicate_counts to keep count_iter in sync
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

    The returned records are used by :func:`generate_mock_from_pool` and
    :func:`generate_mock_key_sets_from_pool`.  Building the pool once and
    reusing it across many mock generations avoids repeated OLGA sampling.

    Parameters
    ----------
    locus:
        IMGT locus code (e.g. ``"TRB"``, ``"TRA"``, ``"IGH"``).
    n:
        Number of sequences to generate.
    seed:
        RNG seed for reproducibility.
    pgen_adjustment:
        Optional :class:`~mir.basic.pgen.PgenGeneUsageAdjustment`.  When
        supplied, each pool record's ``pgen`` field stores the V-J-adjusted
        log₁₀ Pgen.  Pass the same object to :func:`generate_mock_from_pool`
        or :func:`generate_mock_key_sets_from_pool` so that pool and reference
        bins are consistent.

    Returns
    -------
    list[dict]
        Each record contains at least ``junction_aa``, ``v_gene``, ``j_gene``,
        ``pgen`` (log₁₀ Pgen — ``-inf`` for zero-probability sequences),
        ``junction`` (nucleotide), ``v_end``, and ``j_start``.
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

    The pool is binned once; subsequent mock generations are pure NumPy array
    index operations, so generating 1 000 mocks from a 50k-sequence pool takes
    milliseconds rather than minutes.

    .. note::

       When ``fix_v_usage=True`` or ``fix_j_usage=True`` the pool is indexed
       by ``(v_base, bin)`` or ``(v_base, j_base, bin)`` cells.  These cells
       are much sparser than pgen-only bins, so sampling with replacement is
       common.  Use a larger pool (≥ 200k) or accept the :class:`UserWarning`.

    Parameters
    ----------
    reference_repertoire:
        Reference repertoire whose Pgen histogram is matched (e.g. a VDJdb
        epitope-specific set).  Its locus selects the OLGA model.
    pool:
        Pre-generated OLGA sequences from :func:`build_olga_pool`.
    n_mocks:
        Number of mock key sets to generate.
    fix_v_usage:
        Match V-gene usage within each Pgen bin.
    fix_j_usage:
        Match J-gene usage within each Pgen bin.
    seed:
        Seed for the NumPy RNG.

    Returns
    -------
    list[frozenset[tuple[str, str, str]]]
        *n_mocks* frozensets, each the same format as
        :func:`~mir.comparative.overlap.make_reference_keys` output.

    Warns
    -----
    UserWarning
        When pool bins have fewer sequences than needed (sampling with
        replacement) or are entirely absent from the pool.
    """
    locus = _resolve_locus(reference_repertoire)
    model = OlgaModel(locus=locus, seed=seed)

    # --- bin the pool by (optionally V/J-stratified) Pgen bin ----------------
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

    # --- bin the reference clonotypes by the same key scheme -----------------
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

    # --- warn once about bins that will need replacement sampling -------------
    replacement_bins = [
        key for key in ref_bins
        if len(pool_by_key.get(key, [])) < len(ref_bins[key])
    ]
    absent_bins = [key for key in ref_bins if key not in pool_by_key]
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

    # --- sample n_mocks times ------------------------------------------------
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
    """Strip V/J gene fields from mock keys when the corresponding match flag is off.

    Mock key sets always carry full ``(junction_aa, v_base, j_base)`` tuples.
    When ``match_v`` or ``match_j`` is ``False``, the query index has ``""`` in
    that field; we strip the mock keys to match so that :func:`count_overlap`
    sees consistent keys on both sides.
    """
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
# Overlap result (single-config)
# ---------------------------------------------------------------------------

@dataclass
class OverlapResult:
    """Overlap statistics for one query sample under one set of match options.

    Produced by :meth:`VDJBetOverlapAnalysis.score`.  Per-mock distributions
    are stored so the object is self-contained; z/p-scores are computed lazily
    via :func:`functools.cached_property`.

    Attributes
    ----------
    n_total, dc_total:
        Total unique clonotypes and total duplicate count in the query sample.
    n, dc:
        Unique clonotypes and cells overlapping the reference under the
        chosen match options.
    mock_n, mock_dc:
        Per-mock overlap counts (length == n_mocks).
    allow_1mm:
        Whether 1-substitution CDR3 matching was used.
    match_v, match_j:
        Whether V-gene / J-gene matching was required for the real overlap.
    mock_v_fixed, mock_j_fixed:
        Whether V-gene / J-gene usage was fixed when generating the mock null.
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
    mock_v_fixed: bool = False
    mock_j_fixed: bool = False

    # ---- internal z/p helper ---------------------------------------------

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

    # ---- fractions -------------------------------------------------------

    @property
    def frac_n(self) -> float:
        """Fraction of query clonotypes overlapping the reference."""
        return self.n / self.n_total if self.n_total else 0.0

    @property
    def frac_dc(self) -> float:
        """Fraction of query cells overlapping the reference."""
        return self.dc / self.dc_total if self.dc_total else 0.0

    # ---- clonotype-count z/p ---------------------------------------------

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

    # ---- duplicate-count z/p (log₂-transformed) -------------------------

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
# Analysis class
# ---------------------------------------------------------------------------

class VDJBetOverlapAnalysis:
    """Manages an epitope-specific reference with Pgen-matched mock null sets.

    Builds and caches the OLGA pool and mock null distributions internally so
    that :meth:`score` can be called repeatedly with different match options
    without redundant sequence generation.

    Parameters
    ----------
    reference:
        Real epitope-specific reference repertoire (e.g. VDJdb LLWNGPMAV TRB
        clonotypes).
    n_mocks:
        Number of Pgen-matched mock key sets per null type.
    pool_size:
        OLGA pool size used when *method* is ``"pool"``.  Ignored when
        *method* is ``"on_the_fly"``.
    seed:
        RNG seed for pool generation and mock sampling.
    method:
        ``"pool"`` (default) — build an OLGA pool once and draw from it for
        each mock set (orders of magnitude faster for large *n_mocks*).
        ``"on_the_fly"`` — run acceptance-rejection OLGA sampling for each
        mock independently (no pool memory, slower for many mocks).

    Examples
    --------
    >>> analysis = VDJBetOverlapAnalysis(ref_repertoire, n_mocks=1000)
    >>> result = analysis.score(query_sample)
    >>> print(result.z_n, result.p_n)
    >>> result_pvj = analysis.score(query_sample, mock_v_fixed=True, mock_j_fixed=True)
    >>> print(result_pvj.z_n, result_pvj.p_n)
    """

    def __init__(
        self,
        reference: Repertoire,
        *,
        n_mocks: int = 1000,
        pool_size: int = 50_000,
        seed: int = 42,
        method: str = "pool",
        pgen_adjustment=None,
    ) -> None:
        if method not in ("pool", "on_the_fly"):
            raise ValueError(f"method must be 'pool' or 'on_the_fly', got {method!r}")
        self._reference = reference
        self._n_mocks = n_mocks
        self._pool_size = pool_size
        self._seed = seed
        self._method = method
        self._pgen_adjustment = pgen_adjustment
        self._pool: list | None = None
        self._mock_cache: dict[tuple[bool, bool], list[frozenset]] = {}

    def _get_pool(self) -> list:
        if self._pool is None:
            locus = _resolve_locus(self._reference)
            self._pool = build_olga_pool(
                locus, self._pool_size, seed=self._seed,
                pgen_adjustment=self._pgen_adjustment,
            )
        return self._pool

    def _get_mocks(self, v_fixed: bool, j_fixed: bool) -> list[frozenset]:
        key = (v_fixed, j_fixed)
        if key not in self._mock_cache:
            if self._method == "pool":
                mocks = generate_mock_key_sets_from_pool(
                    self._reference,
                    self._get_pool(),
                    self._n_mocks,
                    fix_v_usage=v_fixed,
                    fix_j_usage=j_fixed,
                    seed=self._seed,
                    pgen_adjustment=self._pgen_adjustment,
                )
            else:
                mocks = []
                for i in range(self._n_mocks):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        mock_rep = generate_mock_repertoire(
                            self._reference,
                            fix_v_usage=v_fixed,
                            fix_j_usage=j_fixed,
                            seed=self._seed + i,
                            pgen_adjustment=self._pgen_adjustment,
                        )
                    mocks.append(make_reference_keys(mock_rep))
            self._mock_cache[key] = mocks
        return self._mock_cache[key]

    def score(
        self,
        sample: Repertoire,
        *,
        allow_1mm: bool = False,
        match_v: bool = True,
        match_j: bool = True,
        mock_v_fixed: bool = False,
        mock_j_fixed: bool = False,
        n_jobs: int = 1,
    ) -> OverlapResult:
        """Compute overlap statistics for *sample* against the reference.

        Parameters
        ----------
        sample:
            Query repertoire.
        allow_1mm:
            Count clonotypes within one amino-acid substitution of the
            reference (in addition to exact matches).
        match_v:
            Require V-gene match for the real query↔reference overlap.
        match_j:
            Require J-gene match for the real query↔reference overlap.
        mock_v_fixed:
            Use V-gene-stratified Pgen mocks to control for V-gene bias.
        mock_j_fixed:
            Use J-gene-stratified Pgen mocks to control for J-gene bias.
        n_jobs:
            Parallel worker processes for mock overlap computation.

        Returns
        -------
        OverlapResult
            Single-configuration overlap statistics with z/p-scores.
        """
        qi = make_query_index(sample, match_v=match_v, match_j=match_j)
        n_total  = len(qi)
        dc_total = sum(qi.values())

        ref_keys = make_reference_keys(
            self._reference, allow_1mm=False, match_v=match_v, match_j=match_j,
        )
        real = count_overlap(ref_keys, qi, allow_1mm=allow_1mm)

        raw_mocks  = self._get_mocks(mock_v_fixed, mock_j_fixed)
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
            mock_v_fixed=mock_v_fixed,
            mock_j_fixed=mock_j_fixed,
        )
