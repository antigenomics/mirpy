"""The two halves are separate tools' output, and splitting them must change nothing but cost.

``mir.signature.signature`` is ``vsig`` (vdjtools' statistics) joined to ``rsig`` (mirpy's
geometry). They are separable because the scale reference standardises **per column** — a column's
value does not depend on whether the other half was computed beside it — so a split-and-rejoin on
``sample_id`` is exactly the un-split vector.

What that separation buys is not symmetric. At ``tier="standard"`` mirpy's half is **528 of the
689 columns** and costs about **8% of the runtime**; vdjtools' half is 161 columns and ~94% of the
cost, nearly all of it the Pgen block. So ``mir signature`` emitting both by default meant paying
twelve times over for a half that another tool already owns.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from mir.signature import rsig, rsig_cohort, signature, signature_cohort
from vdjtools.signature.layout import columns

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
_NT = np.array(list("ACGT"))


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    n, ln = 600, None
    ln = rng.integers(10, 18, n)
    j = ["C" + "".join(_AA[rng.integers(0, 20, k)]) + "F" for k in ln]
    return {"TRB": pl.DataFrame({
        "junction_aa": j,
        "junction_nt": ["".join(_NT[rng.integers(0, 4, 3 * len(x))]) for x in j],
        "v_call": [f"TRBV{v}*01" for v in rng.integers(1, 30, n)],
        "j_call": [f"TRBJ{v}*01" for v in rng.integers(1, 13, n)],
        "duplicate_count": rng.integers(1, 200, n),
    })}


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


@pytest.mark.parametrize("tier", ["core", "standard", "full"])
def test_each_tool_emits_exactly_its_own_columns(sample, tier):
    from vdjtools.signature import vsig

    assert set(rsig(sample, tier=tier)) == set(columns(tier, "rsig"))
    assert set(vsig(sample, tier=tier)) == set(columns(tier, "vsig"))


def test_the_halves_partition_the_tier_exactly(sample):
    """No column belongs to both, and none to neither -- otherwise a join would not reconstruct."""
    from vdjtools.signature import vsig

    both = signature(sample, tier="standard")
    r = rsig(sample, tier="standard")
    v = vsig(sample, tier="standard")
    assert set(r) | set(v) == set(both)
    assert not (set(r) & set(v))
    assert len(r) == 528 and len(v) == 161 and len(both) == 689


def test_splitting_does_not_move_a_single_value(sample):
    """The whole justification for splitting. Per-column standardisation is what makes it true.

    Compared at **cohort** level on purpose: bare :func:`rsig` returns raw values while
    :func:`signature` standardises against the reference, so comparing those two directly would
    be comparing different quantities. ``rsig_cohort`` and ``signature_cohort`` both standardise,
    which is what a caller actually gets.
    """
    c = {"S1": sample, "S2": sample}
    mine = rsig_cohort(c, tier="standard")
    joined = signature_cohort(c, tier="standard")
    assert mine["sample_id"].to_list() == joined["sample_id"].to_list()
    for col in mine.columns[1:]:
        for a, b in zip(mine[col].to_list(), joined[col].to_list()):
            if a is None and b is None:
                continue
            assert _same(a, b), f"{col} moved when the half was computed alone"


def test_each_cohort_function_emits_its_own_half(sample):
    """``rsig_cohort`` is this tool's; ``signature_cohort`` is the join the scale fitting needs."""
    c = {"S1": sample, "S2": sample}
    assert rsig_cohort(c, tier="standard").width - 1 == 528
    assert signature_cohort(c, tier="standard").width - 1 == 689


def test_an_explicit_column_list_is_intersected_with_the_half_not_overridden(sample):
    """A preset plus a half must not ask for columns that were never computed."""
    want = columns("standard")[:200]
    got = rsig_cohort({"S1": sample}, tier="standard", columns=want)
    assert set(got.columns[1:]) == {c for c in want if c.startswith("rsig:")}


def test_the_cli_emits_the_rsig_half_and_offers_no_way_to_ask_for_anything_else():
    """One tool per half. There is no ``--half`` flag at all -- not even to opt back in.

    ``mir signature`` emits ``rsig``; ``vdjtools signature`` emits ``vsig``; the caller joins them
    on ``sample_id``. A combined mode here would duplicate the other tool and re-introduce the
    cost it exists to avoid -- vdjtools' Pgen block is ~94% of computing the pair.
    """
    from mir.cli import build_parser

    sig = build_parser()._subparsers._group_actions[0].choices["signature"]
    flags = {o for a in sig._actions for o in getattr(a, "option_strings", [])}
    assert "--half" not in flags, "a combined mode was re-introduced; one tool per half"
    assert "--tier" in flags and "--preset" in flags       # the flags that do belong here


def test_a_preset_keeps_only_its_rsig_columns(sample):
    """The mirror image of `vdjtools signature --preset`, which keeps only the vsig ones."""
    from vdjtools.signature import presets as P

    spec = P.get("classify")
    mine = [c for c in spec.columns() if c.startswith("rsig:")]
    theirs = [c for c in spec.columns() if c.startswith("vsig:")]
    assert len(mine) + len(theirs) == spec.n_columns       # together they are the preset
    assert mine and theirs                                 # classify genuinely spans both
    got = rsig_cohort({"S1": sample}, tier=spec.tier, columns=spec.columns())
    assert set(got.columns[1:]) == set(mine)
