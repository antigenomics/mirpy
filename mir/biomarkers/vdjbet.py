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

from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.repertoire import Repertoire

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
        as *repertoire*.  May be smaller than *repertoire* if some clonotypes
        have zero Pgen or if *max_sequences* is exhausted.

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

    # --- build target histogram -------------------------------------------
    target = compute_pgen_histogram(
        repertoire.clonotypes, model,
        fix_v=fix_v_usage, fix_j=fix_j_usage,
    )
    if not target:
        warnings.warn(
            "generate_mock_repertoire: all clonotypes have zero or undefined "
            "Pgen; returning empty mock repertoire.",
            UserWarning,
            stacklevel=2,
        )
        return Repertoire(clonotypes=[], locus=locus,
                          repertoire_id=f"mock_{repertoire.repertoire_id}")

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
                    duplicate_count=1,
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
