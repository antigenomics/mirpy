"""Tests for mir.track (PhenoPath-style covariate-disentangled repertoire trajectory)."""

import numpy as np
import pytest

from mir.track import fit_exposure_trajectory


def _synthetic(n=100, g=10, seed=0):
    rng = np.random.default_rng(seed)
    tau_true = rng.standard_normal(n)
    x = rng.integers(0, 2, size=n).astype(np.float64)
    kappa_true = rng.standard_normal(g) * 0.3
    gamma_true = np.zeros(g)
    gamma_true[2] = 1.5
    gamma_true[6] = -1.3
    alpha_true = rng.standard_normal(g) * 0.2
    Y = (alpha_true[None, :] * x[:, None]
         + (kappa_true[None, :] + gamma_true[None, :] * x[:, None]) * tau_true[:, None]
         + 0.2 * rng.standard_normal((n, g)))
    return Y, x, tau_true, gamma_true


def test_tau_is_standardized():
    Y, x, _, _ = _synthetic()
    fit = fit_exposure_trajectory(Y, x, seed=0)
    assert fit.tau.shape == (Y.shape[0],)
    assert abs(fit.tau.mean()) < 1e-6
    assert abs(fit.tau.std() - 1.0) < 1e-6


def test_recovers_planted_trajectory_and_interactions():
    Y, x, tau_true, gamma_true = _synthetic(n=1500, seed=1)   # ample n for a tight recovery check
    fit = fit_exposure_trajectory(Y, x, seed=0)
    corr = abs(np.corrcoef(fit.tau, tau_true)[0, 1])
    assert corr > 0.9, f"tau correlation too low: {corr:.3f}"

    top = fit.top_interactions(top=3)   # default channel_names=None -> "channel{i}"
    top_names = set(top["channel"].to_list())
    assert {"channel2", "channel6"} <= top_names


def test_channel_names_and_top_interactions_frame():
    Y, x, _, _ = _synthetic(seed=2)
    names = [f"c{i}" for i in range(Y.shape[1])]
    fit = fit_exposure_trajectory(Y, x, channel_names=names, seed=0)
    top = fit.top_interactions(top=4)
    assert top.height == 4
    assert set(top.columns) == {"channel", "interaction_score", "kappa"}
    assert list(top["interaction_score"]) == sorted(top["interaction_score"], reverse=True)


def test_ard_shrinks_interactions_more_than_no_ard():
    Y, x, _, _ = _synthetic(n=60, g=15, seed=3)   # modest n: ARD's extra shrinkage should show
    fit_ard = fit_exposure_trajectory(Y, x, ard=True, seed=0)
    fit_noard = fit_exposure_trajectory(Y, x, ard=False, seed=0)
    assert np.linalg.norm(fit_ard.gamma) <= np.linalg.norm(fit_noard.gamma) + 1e-6


def test_no_covariate_reduces_to_plain_trajectory():
    # a single all-zero "covariate" column: gamma has nothing to load on, kappa carries everything
    Y, _, tau_true, _ = _synthetic(seed=4)
    zero_cov = np.zeros(Y.shape[0])
    fit = fit_exposure_trajectory(Y, zero_cov, seed=0)
    assert fit.gamma.shape == (Y.shape[1], 1)
    corr = abs(np.corrcoef(fit.tau, tau_true)[0, 1])
    assert corr > 0.5   # trajectory still recoverable from kappa alone


def test_multi_covariate_shapes():
    n, g, p = 80, 6, 3
    rng = np.random.default_rng(5)
    Y = rng.standard_normal((n, g))
    cov = rng.standard_normal((n, p))
    fit = fit_exposure_trajectory(Y, cov, covariate_names=["a", "b", "c"], seed=0)
    assert fit.alpha.shape == (g, p)
    assert fit.gamma.shape == (g, p)
    assert fit.covariate_names == ["a", "b", "c"]


def test_row_count_mismatch_raises():
    Y = np.zeros((10, 4))
    with pytest.raises(ValueError, match="samples"):
        fit_exposure_trajectory(Y, np.zeros(9))


def test_too_few_samples_raises():
    with pytest.raises(ValueError, match=">= 3"):
        fit_exposure_trajectory(np.zeros((2, 4)), np.zeros(2))


def test_bad_label_lengths_raise():
    Y = np.zeros((10, 4))
    x = np.zeros(10)
    with pytest.raises(ValueError, match="channel_names"):
        fit_exposure_trajectory(Y, x, channel_names=["a", "b"])
    with pytest.raises(ValueError, match="covariate_names"):
        fit_exposure_trajectory(Y, x, covariate_names=["a", "b"])
