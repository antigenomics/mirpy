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
