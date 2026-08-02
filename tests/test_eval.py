"""Tests for mir.bench.eval (channel_report scorers). Needs [bench]: sklearn (+ lifelines for survival)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mir.bench.eval import cv_auc, km_logrank


def test_cv_auc_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    signal = y + rng.normal(0, 0.6, n)
    noise = rng.normal(0, 1, n)
    m_sig, s_sig = cv_auc(signal, y, n_repeats=3)
    m_noise, _ = cv_auc(noise, y, n_repeats=3)
    assert m_sig > 0.7 > m_noise
    assert 0.0 <= s_sig < 0.2                          # returns (mean, std) — a CI, not a point


def test_survival_scorers_recover_risk():
    # not marked integration: lifelines ships in [bench], which CI installs — the fast tier
    # should cover the survival scorers too (~1 s).
    pytest.importorskip("lifelines")
    from mir.bench.eval import cv_cindex

    rng = np.random.default_rng(0)
    n = 240
    risk = rng.normal(0, 1, n)
    base = rng.normal(0, 1, (n, 2))
    dur = rng.exponential(np.exp(-0.9 * risk))
    evt = (rng.random(n) < 0.7).astype(float)
    c_base = cv_cindex(dur, evt, base=base, block=None)
    c_full = cv_cindex(dur, evt, base=base, block=risk)
    assert c_full > c_base and c_full > 0.6            # the risk block adds concordance
    p = km_logrank(dur, evt, (risk > np.median(risk)).astype(int))
    assert p < 0.05


def test_recovery_report_scores_carried_statistics_high_and_absent_ones_low():
    from mir.bench.eval import recovery_report

    rng = np.random.default_rng(0)
    n, d = 200, 10
    X = rng.normal(0, 1, (n, d))
    carried = 2.0 * X[:, 0] + X[:, 3] + rng.normal(0, 0.1, n)   # linear in the embedding
    absent = rng.normal(0, 1, n)                                # nowhere in it
    r = recovery_report(X, {"carried": carried, "absent": absent}, n_pc=0)
    assert r["carried"] > 0.9 > r["absent"]

    # grouped folds (a subject never spans train/test) + per-stat non-finite dropping
    groups = np.repeat(np.arange(10), 20)
    holed = carried.copy()
    holed[:5] = np.nan
    assert recovery_report(X, {"carried": holed}, groups=groups, n_pc=0)["carried"] > 0.8

    with pytest.raises(ValueError, match="values for"):
        recovery_report(X, {"bad": np.ones(5)})
