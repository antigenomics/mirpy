from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

from mir.biomarkers.token_stats import compare_kmer_counts


def test_compare_kmer_counts_binom_uses_control_background_probability() -> None:
    # total sample kmers = 20 + 10 = 30; total control kmers = 8 + 12 = 20
    counts_sample = {"AAA": 20, "BBB": 10}
    counts_control = {"AAA": 8, "BBB": 12}

    df = compare_kmer_counts(counts_sample, counts_control, test="binom", p_adj_method="fdr_bh")

    row = df.loc["AAA"]
    expected_p_background = 8 / 20
    expected_pval = float(binom.sf(20 - 1, 30, expected_p_background))

    assert math.isclose(float(row["p_background"]), expected_p_background, rel_tol=1e-12)
    assert math.isclose(float(row["p_val"]), expected_pval, rel_tol=1e-12)
    assert math.isclose(float(row["odds_ratio"]), (20 / 30) / expected_p_background, rel_tol=1e-12)
    assert np.isfinite(float(row["p_val"]))
    assert np.isfinite(float(row["p_val_adj"]))


def test_compare_kmer_counts_unknown_test_raises() -> None:
    counts_sample = {"AAA": 3}
    counts_control = {"AAA": 1}

    try:
        compare_kmer_counts(counts_sample, counts_control, test="bad")
    except ValueError as exc:
        assert "Unknown test" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown test name")
