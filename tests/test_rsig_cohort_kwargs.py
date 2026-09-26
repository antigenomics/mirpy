"""``rsig_cohort`` must pass its own keywords through intact, on every locus.

Both bugs here came from one line: two ``kw.pop`` calls sitting *inside* a dict comprehension.
``pop`` is destructive, so the first locus consumed the caller's value and every locus after it
silently got the default -- a one-locus sample honoured ``on_duplicate="sum"`` and a two-locus
one raised on its second locus. The same placement meant that with ``sanitise=False`` the pop
never happened at all and ``on_duplicate`` was forwarded to ``rsig()``, which has no such
parameter.

Both tests therefore use a **two-locus** sample whose *second* locus carries the duplicate. A
single-locus test passes against the broken code and proves nothing.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mir.signature import assemble


def _clean(n, v, j, seed=0):
    r = np.random.default_rng(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    return pl.DataFrame({
        "v_call": [v] * n, "j_call": [j] * n, "c_call": [None] * n,
        "junction_aa": ["C" + "".join(r.choice(aa, 12)) + "F" for _ in range(n)],
        "duplicate_count": np.ceil(r.zipf(1.5, n).clip(1, 500)).astype(int).tolist(),
    }, schema_overrides={"c_call": pl.Utf8})


def _with_duplicate_key(n, v, j, seed=0):
    """A frame with no ``junction_nt`` that repeats one (junction_aa, v, j, c) key."""
    df = _clean(n, v, j, seed)
    return pl.concat([df, df.head(1)], how="vertical")


@pytest.fixture
def two_loci():
    """TRB clean, IGH duplicated -- so only a per-locus-correct pop can get this right."""
    return {"TRB": _clean(200, "TRBV20-1", "TRBJ2-2", seed=1),
            "IGH": _with_duplicate_key(150, "IGHV1-2", "IGHJ4", seed=2)}


def test_on_duplicate_reaches_the_second_locus(two_loci):
    """The bug: the first locus ate the value, the second fell back to "error" and raised."""
    out = assemble.rsig_cohort({"s1": two_loci}, tier="core", on_duplicate="sum",
                               standardize="none")
    assert out.height == 1


def test_on_duplicate_error_still_raises_on_the_second_locus(two_loci):
    """The other direction: the default must not be silently softened either."""
    with pytest.raises(ValueError, match="more than once"):
        assemble.rsig_cohort({"s1": two_loci}, tier="core", on_duplicate="error",
                             standardize="none")


def test_sanitise_false_does_not_forward_on_duplicate_to_rsig(two_loci):
    """``on_duplicate`` was popped only inside ``if sanitise:``, so it leaked into ``rsig()``."""
    out = assemble.rsig_cohort({"s1": two_loci}, tier="core", sanitise=False,
                               on_duplicate="sum", standardize="none")
    assert out.height == 1


def test_the_first_locus_is_not_privileged(two_loci):
    """Order must not matter: the same sample with the duplicate first behaves identically."""
    flipped = {"IGH": two_loci["IGH"], "TRB": two_loci["TRB"]}
    a = assemble.rsig_cohort({"s1": two_loci}, tier="core", on_duplicate="sum",
                             standardize="none")
    b = assemble.rsig_cohort({"s1": flipped}, tier="core", on_duplicate="sum",
                             standardize="none")
    assert a.columns == b.columns and a.height == b.height == 1


def test_every_sample_gets_the_keyword_not_just_the_first(two_loci):
    """``kw`` is shared across samples in one worker; a destructive pop would drain it."""
    out = assemble.rsig_cohort({f"s{i}": two_loci for i in range(4)}, tier="core",
                               on_duplicate="sum", standardize="none")
    assert out.height == 4


def test_a_deferred_sample_is_read_inside_the_worker(two_loci):
    """The documented ``O(n_jobs)`` memory path, on the joined half as well as the geometry one."""
    calls = []

    def deferred():
        calls.append(1)
        return two_loci

    out = assemble.signature_cohort({"s1": deferred}, tier="core", on_duplicate="sum",
                                    standardize="none")
    assert out.height == 1 and calls == [1]
