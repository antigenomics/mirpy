"""Resolving the prototype panel once must be exactly equivalent to resolving it every call.

`GermlineDistances.matrix` used to run `resolve()` over the prototype panel on every call --
3,584 regex-backed calls per sample, 10.4% of `rsig`. The panel is fixed at construction, so this
was a loop invariant inside the loop. Hoisting it is worth ~1.5x on `rsig` and is only legitimate
if it changes nothing, which is what these pin: not the speed (a stopwatch flakes on a loaded
box), the *equality*.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mir.distances.germline import load_germline_distances

GENES = ["TRBV20-1*01", "TRBV6-5", "TRBV28*01", "TRBVNOPE*99", "TRBV20-1*01"]
PROTO = ["TRBV20-1*01", "TRBV6-5*01", "TRBV28*01", "TRBV5-1*01"]


@pytest.fixture(scope="module")
def gd():
    return load_germline_distances("human", "TRB")


def test_proto_idx_reproduces_the_unhoisted_matrix(gd):
    a = gd.matrix("V", GENES, PROTO)
    b = gd.matrix("V", GENES, PROTO, proto_idx=gd.resolve_all("V", PROTO))
    assert np.array_equal(a, b)


def test_proto_idx_ignores_the_allele_list_it_replaces(gd):
    """Once resolved, the strings are not consulted -- pinning what `proto_idx` means."""
    idx = gd.resolve_all("V", PROTO)
    good = gd.matrix("V", GENES, PROTO, proto_idx=idx)
    junk = gd.matrix("V", GENES, ["nonsense"] * len(PROTO), proto_idx=idx)
    assert np.array_equal(good, junk)


def test_resolve_all_matches_resolve_one_at_a_time(gd):
    c = gd._component("V")
    assert list(gd.resolve_all("V", PROTO)) == [c.resolve(a) for a in PROTO]


def test_an_unknown_component_still_raises_through_resolve_all(gd):
    with pytest.raises(KeyError, match="unavailable"):
        gd.resolve_all("NOPE", PROTO)
    with pytest.raises(KeyError, match="unavailable"):
        gd.matrix("NOPE", GENES, PROTO)


def test_a_null_allele_is_still_handled(gd):
    """The dict factorize exists so a null `v_call` resolves to its fallback; `np.unique` cannot.

    Measured 2026-09-26: np.unique(return_inverse=True) is 1.12x here -- 0.05 ms of a 32.8 ms
    sample -- and raises TypeError on this input. The dict stays.
    """
    D = gd.matrix("V", [None, "TRBV20-1*01"], PROTO)
    assert D.shape == (2, len(PROTO)) and np.isfinite(D).all()
    with pytest.raises(TypeError):
        np.unique(np.array([None, "TRBV20-1*01"], dtype=object), return_inverse=True)


def test_the_embedder_hoist_changes_no_rsig_column():
    """End to end: the whole signature half, hoisted against un-hoisted, in one process."""
    from mir.embedding.tcremp import TCREmp
    from mir.signature.assemble import rsig

    r = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    frames = {"TRB": pl.DataFrame({
        "v_call": ["TRBV20-1"] * 120, "j_call": ["TRBJ2-2"] * 120, "c_call": [None] * 120,
        "junction_aa": ["C" + "".join(r.choice(aa, 12)) + "F" for _ in range(120)],
        "duplicate_count": np.ceil(r.zipf(1.5, 120).clip(1, 500)).astype(int).tolist(),
    }, schema_overrides={"c_call": pl.Utf8})}

    hoisted = rsig(frames, tier="standard")
    orig = TCREmp._proto_idx
    try:
        TCREmp._proto_idx = lambda self, comp: None          # the pre-3.19.0 path
        plain = rsig(frames, tier="standard")
    finally:
        TCREmp._proto_idx = orig

    moved = [k for k in hoisted
             if not (hoisted[k] == plain[k]
                     or (np.isnan(hoisted[k]) and np.isnan(plain[k])))]
    assert moved == [], moved[:5]
