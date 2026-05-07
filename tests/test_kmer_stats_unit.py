from __future__ import annotations

import math

import numpy as np

from mir.biomarkers.kmer_stats import compare_kmer_counts


def test_compare_kmer_counts_fisher_uses_total_kmer_denominator() -> None:
    # total sample kmers = 20 + 10 = 30; total control kmers = 8 + 12 = 20
    counts_sample = {"AAA": 20, "BBB": 10}
    counts_control = {"AAA": 8, "BBB": 12}

    df = compare_kmer_counts(counts_sample, counts_control, test="fisher", p_adj_method="fdr_bh")

    row = df.loc["AAA"]
    # Odds for [[20,10],[8,12]] is (20*12)/(10*8)=3.0
    assert math.isclose(float(row["odds_ratio"]), 3.0, rel_tol=1e-12)
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
