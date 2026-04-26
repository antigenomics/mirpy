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
        batch = model.generate_sequences_with_meta(batch_n, pgens=True, seed=None)
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

    Returns
    -------
    list[dict]
        Each record contains at least ``junction_aa``, ``v_gene``, ``j_gene``,
        ``pgen`` (log₁₀ Pgen — ``-inf`` for zero-probability sequences),
        ``junction`` (nucleotide), ``v_end``, and ``j_start``.
    """
    model = OlgaModel(locus=locus, seed=seed)
    return model.generate_sequences_with_meta(n, pgens=True, seed=seed)


def generate_mock_key_sets_from_pool(
    reference_repertoire: Repertoire,
    pool: list[_PoolRecord],
    n_mocks: int,
    *,
    fix_v_usage: bool = False,
    fix_j_usage: bool = False,
    seed: int = 42,
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
# Overlap result and analysis class
# ---------------------------------------------------------------------------

@dataclass
class OverlapResult:
    """Overlap counts, fractions, and z/p-scores for one query sample.

    Produced by :meth:`VDJBetOverlapAnalysis.score`.  Raw mock distributions
    are stored so the object is self-contained; fractions and z-scores are
    computed lazily via :func:`functools.cached_property`.

    Attributes
    ----------
    n_total, dc_total:
        Total unique clonotypes and total duplicate count in the query sample.
    n_exact, dc_exact:
        Clonotypes / cells overlapping the reference by exact match.
    n_1mm, dc_1mm:
        Clonotypes / cells overlapping the reference within 1 substitution.
    mock_pgen_n_exact, ...:
        Per-mock overlap counts under the pgen-only null (list length = n_mocks).
    mock_pvj_n_exact, ...:
        Per-mock overlap counts under the pgen+V+J null; ``None`` when not
        provided to :class:`VDJBetOverlapAnalysis`.
    """

    n_total: int
    dc_total: int
    n_exact: int
    dc_exact: int
    n_1mm: int
    dc_1mm: int
    mock_pgen_n_exact: list
    mock_pgen_dc_exact: list
    mock_pgen_n_1mm: list
    mock_pgen_dc_1mm: list
    mock_pvj_n_exact: list | None = None
    mock_pvj_dc_exact: list | None = None
    mock_pvj_n_1mm: list | None = None
    mock_pvj_dc_1mm: list | None = None

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
    def frac_n_exact(self) -> float:
        """Fraction of query clonotypes matching the reference (exact)."""
        return self.n_exact / self.n_total if self.n_total else 0.0

    @property
    def frac_dc_exact(self) -> float:
        """Fraction of query cells matching the reference (exact)."""
        return self.dc_exact / self.dc_total if self.dc_total else 0.0

    @property
    def frac_n_1mm(self) -> float:
        """Fraction of query clonotypes matching within 1 substitution."""
        return self.n_1mm / self.n_total if self.n_total else 0.0

    @property
    def frac_dc_1mm(self) -> float:
        """Fraction of query cells matching within 1 substitution."""
        return self.dc_1mm / self.dc_total if self.dc_total else 0.0

    # ---- log2 dc helpers (for dc z-scores and box plots) -----------------

    @cached_property
    def dc_exact_log2(self) -> float:
        return math.log2(self.dc_exact + 1)

    @cached_property
    def dc_1mm_log2(self) -> float:
        return math.log2(self.dc_1mm + 1)

    @cached_property
    def mock_pgen_dc_exact_log2(self) -> list:
        return [math.log2(x + 1) for x in self.mock_pgen_dc_exact]

    @cached_property
    def mock_pgen_dc_1mm_log2(self) -> list:
        return [math.log2(x + 1) for x in self.mock_pgen_dc_1mm]

    # ---- pgen-only z/p ---------------------------------------------------

    @cached_property
    def _zp_n_exact_pgen(self) -> tuple[float, float]:
        return self._z_p(self.n_exact, self.mock_pgen_n_exact)

    @property
    def z_n_exact_pgen(self) -> float:
        return self._zp_n_exact_pgen[0]

    @property
    def p_n_exact_pgen(self) -> float:
        return self._zp_n_exact_pgen[1]

    @cached_property
    def _zp_n_1mm_pgen(self) -> tuple[float, float]:
        return self._z_p(self.n_1mm, self.mock_pgen_n_1mm)

    @property
    def z_n_1mm_pgen(self) -> float:
        return self._zp_n_1mm_pgen[0]

    @property
    def p_n_1mm_pgen(self) -> float:
        return self._zp_n_1mm_pgen[1]

    @cached_property
    def _zp_dc_exact_pgen(self) -> tuple[float, float]:
        return self._z_p(self.dc_exact_log2, self.mock_pgen_dc_exact_log2)

    @property
    def z_dc_exact_pgen(self) -> float:
        return self._zp_dc_exact_pgen[0]

    @property
    def p_dc_exact_pgen(self) -> float:
        return self._zp_dc_exact_pgen[1]

    @cached_property
    def _zp_dc_1mm_pgen(self) -> tuple[float, float]:
        return self._z_p(self.dc_1mm_log2, self.mock_pgen_dc_1mm_log2)

    @property
    def z_dc_1mm_pgen(self) -> float:
        return self._zp_dc_1mm_pgen[0]

    @property
    def p_dc_1mm_pgen(self) -> float:
        return self._zp_dc_1mm_pgen[1]

    # ---- pgen+V+J z/p (None when pvj mocks not provided) ----------------

    @cached_property
    def _zp_n_exact_pvj(self) -> tuple[float, float] | None:
        if self.mock_pvj_n_exact is None:
            return None
        return self._z_p(self.n_exact, self.mock_pvj_n_exact)

    @property
    def z_n_exact_pvj(self) -> float | None:
        r = self._zp_n_exact_pvj
        return r[0] if r is not None else None

    @property
    def p_n_exact_pvj(self) -> float | None:
        r = self._zp_n_exact_pvj
        return r[1] if r is not None else None

    @cached_property
    def _zp_n_1mm_pvj(self) -> tuple[float, float] | None:
        if self.mock_pvj_n_1mm is None:
            return None
        return self._z_p(self.n_1mm, self.mock_pvj_n_1mm)

    @property
    def z_n_1mm_pvj(self) -> float | None:
        r = self._zp_n_1mm_pvj
        return r[0] if r is not None else None

    @property
    def p_n_1mm_pvj(self) -> float | None:
        r = self._zp_n_1mm_pvj
        return r[1] if r is not None else None


class VDJBetOverlapAnalysis:
    """Holds an epitope-specific reference + Pgen-matched mock null distributions.

    Pre-builds exact and 1mm reference key sets at construction time so that
    :meth:`score` can be called repeatedly across many query samples without
    rebuilding them.

    Parameters
    ----------
    reference:
        Real epitope-specific reference repertoire (e.g. VDJdb LLWNGPMAV TRB
        clonotypes).
    mock_pgen:
        Pgen-only mock key sets from :func:`generate_mock_key_sets_from_pool`.
    mock_pvj:
        Optional Pgen+V+J mock key sets.  When provided, :class:`OverlapResult`
        objects will also expose pvj z/p-scores.

    Examples
    --------
    >>> pool = build_olga_pool("TRB", 50_000, seed=42)
    >>> mocks = generate_mock_key_sets_from_pool(ref_rep, pool, 1000)
    >>> analysis = VDJBetOverlapAnalysis(ref_rep, mocks)
    >>> result = analysis.score(query_sample)
    >>> print(result.z_n_exact_pgen, result.p_n_exact_pgen)
    """

    def __init__(
        self,
        reference: Repertoire,
        mock_pgen: list,
        mock_pvj: list | None = None,
    ) -> None:
        self._ref_exact = make_reference_keys(reference, allow_1mm=False)
        self._ref_1mm   = make_reference_keys(reference, allow_1mm=True)
        self._mock_pgen = mock_pgen
        self._mock_pvj  = mock_pvj

    def score(
        self,
        sample: Repertoire,
        *,
        n_jobs: int = 1,
    ) -> OverlapResult:
        """Compute overlap statistics for *sample* against the reference.

        Parameters
        ----------
        sample:
            Query repertoire.
        n_jobs:
            Number of parallel worker processes for mock overlap computation.
            ``1`` (default) runs single-threaded.

        Returns
        -------
        OverlapResult
            All overlap counts, fractions, and z/p-scores.
        """
        qi       = make_query_index(sample)
        n_total  = len(qi)
        dc_total = sum(qi.values())

        n_exact, dc_exact = count_overlap(self._ref_exact, qi)
        n_1mm,   dc_1mm   = count_overlap(self._ref_1mm,   qi)

        pgen_exact = compute_overlaps(
            self._mock_pgen, qi, allow_1mm=False, n_jobs=n_jobs,
        )
        pgen_1mm = compute_overlaps(
            self._mock_pgen, qi, allow_1mm=True, n_jobs=n_jobs,
        )

        pvj_exact = pvj_1mm = None
        if self._mock_pvj is not None:
            pvj_exact = compute_overlaps(
                self._mock_pvj, qi, allow_1mm=False, n_jobs=n_jobs,
            )
            pvj_1mm = compute_overlaps(
                self._mock_pvj, qi, allow_1mm=True, n_jobs=n_jobs,
            )

        return OverlapResult(
            n_total=n_total,
            dc_total=dc_total,
            n_exact=n_exact,
            dc_exact=dc_exact,
            n_1mm=n_1mm,
            dc_1mm=dc_1mm,
            mock_pgen_n_exact=[r[0] for r in pgen_exact],
            mock_pgen_dc_exact=[r[1] for r in pgen_exact],
            mock_pgen_n_1mm=[r[0] for r in pgen_1mm],
            mock_pgen_dc_1mm=[r[1] for r in pgen_1mm],
            mock_pvj_n_exact=[r[0] for r in pvj_exact] if pvj_exact else None,
            mock_pvj_dc_exact=[r[1] for r in pvj_exact] if pvj_exact else None,
            mock_pvj_n_1mm=[r[0] for r in pvj_1mm] if pvj_1mm else None,
            mock_pvj_dc_1mm=[r[1] for r in pvj_1mm] if pvj_1mm else None,
        )
