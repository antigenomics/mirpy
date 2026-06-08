"""Unit tests for mir.utils.stats (Benjamini-Hochberg FDR helper)."""

from __future__ import annotations

import numpy as np
import pytest

from mir.utils.stats import bh_fdr


class TestBhFdr:
    def test_empty_input(self):
        out = bh_fdr([])
        assert out.shape == (0,)

    def test_all_nan_preserved(self):
        out = bh_fdr([np.nan, np.nan])
        assert np.isnan(out).all()

    def test_nan_entries_preserved_alongside_finite(self):
        out = bh_fdr([0.01, np.nan, 0.5])
        assert np.isnan(out[1])
        assert np.isfinite(out[0]) and np.isfinite(out[2])

    def test_output_clipped_to_unit_interval(self):
        out = bh_fdr([0.9, 0.95, 0.99])
        assert np.all(out[np.isfinite(out)] >= 0.0)
        assert np.all(out[np.isfinite(out)] <= 1.0)

    def test_adjusted_geq_raw(self):
        pvals = np.array([0.001, 0.01, 0.02, 0.5])
        out = bh_fdr(pvals)
        # BH adjustment never decreases a p-value.
        assert np.all(out >= pvals - 1e-12)

    def test_matches_statsmodels_reference(self):
        sm = pytest.importorskip("statsmodels.stats.multitest")
        pvals = [0.001, 0.008, 0.02, 0.04, 0.2, 0.7]
        expected = sm.multipletests(pvals, method="fdr_bh")[1]
        out = bh_fdr(pvals)
        np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)

    def test_monotonic_in_rank_order(self):
        pvals = np.array([0.001, 0.01, 0.03, 0.04, 0.05])
        out = bh_fdr(pvals)
        # adjusted q-values are non-decreasing along the sorted p-value order
        order = np.argsort(pvals)
        assert np.all(np.diff(out[order]) >= -1e-12)
