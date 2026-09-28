"""The embedding-diversity channel family: the four ways to get it wrong.

Each test here fails under one plausible implementation mistake, which is why these four and not a
coverage sweep. Written from the work plan in ``ISSUES.md`` item 13.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import mir.signature  # noqa: F401  -- registers the rsig contract into vdjtools' layout
from mir.embedding.tcremp import TCREmp
from mir.signature import features as F
from vdjtools.signature import layout as L

AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def _frame(n, seed, *, counts=None, v=None, j=None):
    r = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "v_call": list(v) if v is not None else r.choice(
                ["TRBV20-1", "TRBV5-1", "TRBV9", "TRBV28"], n).tolist(),
            "j_call": list(j) if j is not None else r.choice(
                ["TRBJ2-2", "TRBJ1-1", "TRBJ2-7"], n).tolist(),
            "c_call": [None] * n,
            "junction_aa": ["C" + "".join(r.choice(AA, int(r.integers(9, 15)))) + "F"
                            for _ in range(n)],
            "junction_nt": ["A" * 39] * n,
            "duplicate_count": (list(counts) if counts is not None
                                else r.integers(1, 60, n).tolist()),
        },
        schema_overrides={"c_call": pl.Utf8},
    ).unique(subset=["junction_aa"], maintain_order=True)


@pytest.fixture(scope="module")
def model():
    return TCREmp.from_defaults("human", "TRB", n_prototypes=48)


def _acc(df, model, weight="log2p1"):
    w = F.weights(df["duplicate_count"].to_numpy(), weight)
    counts = df["duplicate_count"].to_numpy()
    masks = {k: np.asarray(p(counts), dtype=bool) for k, p in F.BANDS.items()}
    return F.dispersion_pass(df, model, w, band_masks=masks)


def test_the_slot_rao_values_sum_to_the_total_rao():
    """``Phi``'s V/J/junction parts are the strides ``[0::3]``, ``[1::3]``, ``[2::3]``.

    Reading them as the blocks ``[0:K]``, ``[K:2K]``, ``[2K:3K]`` is the likeliest single bug in
    this family and produces entirely plausible-looking numbers -- the only thing that catches it
    is that the three no longer partition the total.

    The tolerance is **relative**: Rao here is order 1e6, so an absolute 1e-9 would be asking for
    more than float64 carries.
    """
    m = TCREmp.from_defaults("human", "TRB", n_prototypes=48)
    d = F.diversity_channels(_acc(_frame(300, 3), m), "TRB")
    raw = {k.split(":")[-1]: np.expm1(v) for k, v in d.items()}
    total = raw["q_v"] + raw["q_j"] + raw["q_c"]
    assert total == pytest.approx(raw["rao"], rel=1e-9)


def test_evenness_is_one_for_equal_clone_sizes_and_inside_the_unit_interval_otherwise(model):
    """``evenness = Q(w)/Q(uniform)``. Swapping the two calls, or applying the self-pair
    correction twice, moves this off 1.0 for a flat repertoire."""
    n = 200
    flat = _frame(n, 11, counts=[7] * n)
    d = F.diversity_channels(_acc(flat, model), "TRB")
    assert d["rsig:div:TRB:evenness"] == pytest.approx(1.0, abs=1e-9)

    skew = _frame(n, 11, counts=[1] * (n - 3) + [5000, 9000, 20000])
    ev = F.diversity_channels(_acc(skew, model), "TRB")["rsig:div:TRB:evenness"]
    assert 0.0 < ev < 1.0, ev


def test_the_metrics_are_scale_free_in_clone_size(model):
    """Duplicating every clonotype with halved weight must not move anything.

    A weight normalisation that is not scale-free -- dividing by a count rather than by the weight
    sum, say -- shows up here and nowhere else.
    """
    df = _frame(150, 21)
    a = F.diversity_channels(_acc(df, model), "TRB")
    doubled = df.with_columns((pl.col("duplicate_count") * 2).alias("duplicate_count"))
    b = F.diversity_channels(_acc(doubled, model, weight="duplicate_count"), "TRB")
    ref = F.diversity_channels(_acc(df, model, weight="duplicate_count"), "TRB")
    for k in ("eff_dim", "eff_dim_pr", "q_v", "q_j", "q_c", "rao"):
        col = f"rsig:div:TRB:{k}"
        assert b[col] == pytest.approx(ref[col], rel=1e-9), k
    assert np.isfinite(a["rsig:div:TRB:eff_dim"])


def test_a_locus_below_the_floor_is_a_hole_and_the_mask_says_so(tmp_path):
    """Item 11. Rao of a one-clonotype locus is arithmetically ``0.0`` and is not a diversity
    measurement -- it is ``mask:present`` in different units, sitting far below every real value.
    Emitted as a number it gets read as one."""
    from vdjtools.signature.corpus import fit_cohort

    from mir.signature import rsig

    samples = {f"S{i}": {"TRB": _frame(90, i)} for i in range(12)}
    corpus = fit_cohort(samples, sig="rsig", name="t", loci=("TRB",), n_components=3, n_jobs=1)

    deep = rsig(samples["S0"], corpus, n_components=3, min_clonotypes=5)
    assert np.isfinite(deep["rsig:div:TRB:rao"])
    assert deep["rsig:mask:TRB:present"] == 1.0
    assert deep["rsig:mask:TRB:estimable"] == 1.0

    shallow = rsig({"TRB": _frame(4, 77)}, corpus, n_components=3, min_clonotypes=5)
    for c in L.channel_columns("rsig"):
        _s, block, locus, _f = L.parse(c)
        if locus == "TRB" and block in ("div", "disp"):
            assert np.isnan(shallow[c]), f"{c} is a number on a sub-floor locus"
    assert shallow["rsig:mask:TRB:present"] == 1.0
    assert shallow["rsig:mask:TRB:estimable"] == 0.0

    absent = rsig({"TRB": _frame(90, 5)}, corpus, n_components=3)
    assert absent["rsig:mask:TRD:present"] == 0.0


def test_adding_channels_does_not_invalidate_a_fitted_corpus(tmp_path):
    """The rotation is indexed by ``raw_columns`` and nothing else, so a new channel appears in the
    output of an artifact fitted before it existed. This is what makes the family shippable without
    a refit, and the claim is worth a test rather than a sentence."""
    from vdjtools.signature.corpus import Corpus, fit_cohort

    from mir.signature import rsig

    samples = {f"S{i}": {"TRB": _frame(90, i)} for i in range(12)}
    corpus = fit_cohort(samples, sig="rsig", name="t", loci=("TRB",), n_components=3, n_jobs=1)
    path = tmp_path / "c.npz"
    corpus.save(path)
    reloaded = Corpus.load(path)
    # Every registered channel is filled from the sample, not from the artifact.
    row = rsig(samples["S1"], reloaded, n_components=3)
    for c in L.channel_columns("rsig"):
        assert c in row, c
