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


def test_cv_cindex_degenerate_inputs_return_nan_not_a_crash():
    """The survival scorer's guard rails: no covariates, and a fold Cox cannot fit."""
    pytest.importorskip("lifelines")
    from mir.bench.eval import cv_cindex

    rng = np.random.default_rng(0)
    n = 60
    dur = rng.exponential(10, n)
    evt = rng.integers(0, 2, n).astype(float)

    # a Cox with nothing to regress on is undefined, not an error
    assert np.isnan(cv_cindex(dur, evt))

    # a constant covariate block carries no information; _nondegen strips it and the score
    # falls back to chance rather than raising
    assert cv_cindex(dur, evt, block=np.ones((n, 3))) == pytest.approx(0.5)

    # a real covariate scores finitely, and n_pc reduces a wide block rather than failing
    risk = rng.standard_normal(n)
    dur_r = rng.exponential(1.0 / np.exp(risk * 0.8))
    wide = np.column_stack([risk, rng.standard_normal((n, 30))])
    c = cv_cindex(dur_r, np.ones(n), block=wide, n_pc=5)
    assert np.isfinite(c) and 0.0 <= c <= 1.0


def test_recovery_report_handles_missing_targets_and_too_few_groups():
    """Non-finite targets are dropped per-statistic; a statistic with < 2 usable groups is nan."""
    from mir.bench.eval import recovery_report

    rng = np.random.default_rng(1)
    n, d = 80, 6
    X = rng.standard_normal((n, d))
    carried = X[:, 0] * 3.0                      # trivially recoverable from the embedding
    groups = np.repeat(np.arange(8), n // 8)

    holed = carried.copy()
    holed[:10] = np.nan                          # dropped for this statistic only
    out = recovery_report(X, {"carried": carried, "holed": holed}, groups=groups)
    assert out["carried"] > 0.9 and out["holed"] > 0.9

    # a statistic present for only one group cannot be cross-validated across groups
    one_group = np.full(n, np.nan)
    one_group[groups == 0] = carried[groups == 0]
    assert np.isnan(recovery_report(X, {"one_group": one_group}, groups=groups)["one_group"])


def test_kmer_matrix_is_a_fixed_width_baseline():
    """The k-mer baseline channel_report compares embeddings against."""
    import polars as pl

    from mir.bench.eval import kmer_matrix

    def _frame(junctions, counts):
        c = np.asarray(counts, dtype=float)
        return pl.DataFrame({
            "junction_aa": junctions, "v_call": ["TRBV20-1*01"] * len(junctions),
            "j_call": ["TRBJ2-7*01"] * len(junctions), "duplicate_count": c,
            "frequency": c / c.sum(), "locus": ["TRB"] * len(junctions),
        })

    frames = [
        _frame(["CASSLGQAYEQFF", "CASSIRSSYEQYF"], [3, 1]),
        _frame(["CASSLGQAYEQFF"], [5]),
        _frame(["CSARVSGYYGYTF", "CASSISGGADTQYF"], [2, 2]),
    ]
    M = kmer_matrix(frames, k=3)
    # one row per sample, one column per k-mer in the POOLED vocabulary (so every sample is
    # expressed on a common basis, which is what makes the rows comparable)
    assert M.shape[0] == 3
    assert M.shape[1] == len({j[i:i + 3] for f in frames for j in f["junction_aa"]
                              for i in range(len(j) - 2)})
    assert np.isfinite(M).all() and (M >= 0).all()

    # the two samples sharing a junction are closer than either is to the third
    d01 = np.linalg.norm(M[0] - M[1])
    assert d01 < np.linalg.norm(M[0] - M[2]) and d01 < np.linalg.norm(M[1] - M[2])
