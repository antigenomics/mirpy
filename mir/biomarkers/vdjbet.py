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

The technique is inspired by the VDJbeta / VDJBet approach:
    Isacchini et al. (2021) PNAS — *Deep generative selection models …*
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Union

import numpy as np

from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.repertoire import Repertoire

# Pool record type: dict with junction_aa, v_gene, j_gene, pgen (log10), junction, v_end, j_start
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
