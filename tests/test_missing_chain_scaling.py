"""A chain present in few donors must not gain weight from its own sparsity.

Regression test for the impute-then-standardize order in ChannelBuilder.build and
build_donor_cohort: computing sd over the imputed matrix deflates it in proportion to the hole
fraction, so the least-observed block dominates every downstream distance.
"""

import numpy as np

from mir.explain import ChannelBuilder


def _cohort_like(n=400, width=8, present=0.3, seed=0):
    """Two blocks with identical real distributions; the second is observed in `present` of rows."""
    rng = np.random.default_rng(seed)
    full = rng.normal(0.0, 1.0, (n, width))
    sparse = rng.normal(0.0, 1.0, (n, width))
    sparse[int(n * present):] = np.nan
    return full, sparse


def test_sparse_block_is_not_upweighted():
    full, sparse = _cohort_like()
    b = ChannelBuilder()
    b.add("full", full)
    b.add("sparse", sparse)
    X, spec = b.build(standardize=True, impute=True)

    cf = spec.columns("full")
    cs = spec.columns("sparse")
    obs = np.isfinite(sparse[:, 0])

    # the two blocks are drawn from the same distribution, so their OBSERVED entries must end up on
    # the same scale -- that is the whole claim
    sd_full = X[:, cf].std()
    sd_sparse_obs = X[np.ix_(obs, cs)].std()
    assert abs(sd_sparse_obs - sd_full) < 0.15, (
        f"sparse block observed sd {sd_sparse_obs:.3f} vs full {sd_full:.3f}: "
        "sparsity is still inflating the block")

    # an imputed hole must carry no information: it sits at the column mean, i.e. ~0
    assert abs(X[np.ix_(~obs, cs)].mean()) < 0.05
    assert X[np.ix_(~obs, cs)].std() < 1e-9, "holes must all land on one value"


def test_no_holes_is_unchanged():
    """With nothing missing, this must be exactly a plain z-score."""
    rng = np.random.default_rng(1)
    A = rng.normal(3.0, 7.0, (50, 4))
    b = ChannelBuilder()
    b.add("a", A)
    X, _ = b.build(standardize=True, impute=True)
    assert np.allclose(X, (A - A.mean(0)) / A.std(0))


def test_all_nan_column_is_inert():
    rng = np.random.default_rng(2)
    A = rng.normal(0, 1, (30, 2))
    dead = np.full((30, 1), np.nan)
    b = ChannelBuilder()
    b.add("a", A)
    b.add("dead", dead)
    X, spec = b.build(standardize=True, impute=True)
    assert np.isfinite(X).all()
    assert np.allclose(X[:, spec.columns("dead")], 0.0)


# --------------------------------------------------------------- align_loci

import pytest

from mir.cohort import align_loci, missingness_report


def _two_loci(n_a=100, n_b=30, seed=0):
    rng = np.random.default_rng(seed)
    ids_a = [f"s{i}" for i in range(n_a)]
    ids_b = [f"s{i}" for i in range(n_b)]
    return {"TRB": (ids_a, rng.normal(0, 1, (n_a, 4))),
            "TRD": (ids_b, rng.normal(0, 1, (n_b, 3)))}


def test_union_keeps_every_sample_inner_does_not():
    b = _two_loci()
    assert len(align_loci(b, how="union").ids) == 100
    assert len(align_loci(b, how="inner").ids) == 30


def test_absent_locus_is_nan_then_zero_after_build():
    al = align_loci(_two_loci())
    absent = ~al.mask[:, al.loci.index("TRD")]
    assert np.isnan(al.blocks["TRD"][absent]).all(), "absent rows must be nan, not 0"
    X, spec = al.build()
    cols = spec.columns("TRD")
    assert np.isfinite(X).all()
    assert np.allclose(X[np.ix_(absent, cols)], 0.0), "an absent locus must contribute nothing"


def test_sparse_locus_does_not_dominate_the_geometry():
    """The bug this feature exists to prevent: the thinnest locus owning the distances."""
    al = align_loci(_two_loci(n_a=400, n_b=120))
    X, spec = al.build()
    present = al.mask[:, al.loci.index("TRD")]
    sd_trb = X[:, spec.columns("TRB")].std()
    sd_trd_obs = X[np.ix_(present, spec.columns("TRD"))].std()
    assert abs(sd_trd_obs - sd_trb) < 0.15, (
        f"TRD observed sd {sd_trd_obs:.3f} vs TRB {sd_trb:.3f}")


def test_missingness_report_flags_a_coverage_clustering():
    al = align_loci(_two_loci())
    j = al.loci.index("TRD")
    by_coverage = al.mask[:, j].astype(int)          # clusters ARE the presence pattern
    rep = missingness_report(by_coverage, al.mask)
    assert rep["ami_vs_pattern"] > 0.9, rep
    assert rep["n_patterns"] == 2

    rng = np.random.default_rng(7)
    rep2 = missingness_report(rng.integers(0, 4, len(al.ids)), al.mask)
    assert rep2["ami_vs_pattern"] < 0.1, rep2


def test_require_and_min_loci():
    b = _two_loci()
    assert len(align_loci(b, require=["TRD"]).ids) == 30
    assert len(align_loci(b, min_loci=2).ids) == 30
    assert len(align_loci(b, min_loci=1).ids) == 100


def test_align_loci_rejects_bad_input():
    with pytest.raises(ValueError, match="no blocks"):
        align_loci({})
    with pytest.raises(ValueError, match="union.*inner"):
        align_loci(_two_loci(), how="outer")
    with pytest.raises(ValueError, match="rows for"):
        align_loci({"TRB": (["a", "b"], np.zeros((3, 2)))})
    with pytest.raises(ValueError, match="not in blocks"):
        align_loci(_two_loci(), require=["IGH"])
    with pytest.raises(ValueError, match="min_loci"):
        align_loci(_two_loci(), min_loci=5)


def test_pattern_and_n_loci():
    al = align_loci(_two_loci(n_a=10, n_b=4))
    assert al.n_loci.tolist().count(2) == 4
    assert set(al.pattern.tolist()) == {"11", "10"}
