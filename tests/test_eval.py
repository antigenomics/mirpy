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


@pytest.mark.integration
def test_survival_scorers_recover_risk():
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
