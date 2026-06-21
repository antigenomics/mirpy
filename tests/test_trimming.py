"""Unit tests for OLGA-derived germline-retention profiles and Pgen-lite.

These build a real OLGA TRB model once (shared across tests) and exercise the
retention profiles, anchor-depth helper, and the PgenLite approximation.
"""
from __future__ import annotations

import numpy as np
import pytest

from mir.basic.trimming import (
    PgenLite,
    retention_anchor_depths,
    retention_profiles,
)


@pytest.fixture(scope="module")
def profiles():
    return retention_profiles(locus="TRB", species="human")


@pytest.fixture(scope="module")
def lite():
    # Small calibration sample — enough for monotonicity checks, fast enough for CI.
    return PgenLite.calibrate(locus="TRB", species="human", n_calib=1500, seed=7)


def test_retention_profiles_bounded_and_monotone(profiles):
    retV, retJ = profiles
    assert "TRBV9*01" in retV and "TRBJ2-7*01" in retJ
    for prof in (retV["TRBV9*01"], retJ["TRBJ2-7*01"]):
        assert all(0.0 <= p <= 1.0 for p in prof)
        # Retention is non-increasing with offset from the anchor (more inner
        # positions need more germline survival).
        assert all(prof[i] >= prof[i + 1] - 1e-9 for i in range(len(prof) - 1))
    # A functional V is strongly germline at the anchor.
    assert retV["TRBV9*01"][0] > 0.8


def test_retention_anchor_depths(profiles):
    retV, retJ = profiles
    n_term, c_term = retention_anchor_depths(retV, retJ, "TRBV9*01", "TRBJ2-7*01", cutoff=0.5)
    assert 0 < n_term <= len(retV["TRBV9*01"])
    assert 0 < c_term <= len(retJ["TRBJ2-7*01"])
    # Unknown genes -> zero depth, no crash.
    assert retention_anchor_depths(retV, retJ, "NOPE", "NOPE") == (0, 0)


def test_pgen_lite_runs_and_penalizes_insertions(lite):
    base = "CASSLGETQYF"
    longer = "CASSLGGGGGGGETQYF"  # same anchors, larger insert core
    s_base, s_long = lite.score([base, longer], ["TRBV9*01", "TRBV9*01"],
                                ["TRBJ2-7*01", "TRBJ2-7*01"])
    assert np.isfinite(s_base) and np.isfinite(s_long)
    # More insert positions -> lower (more negative) log10 Pgen-lite.
    assert s_long < s_base


def test_pgen_lite_unknown_gene_uses_fallback(lite):
    (s,) = lite.score(["CASSLGETQYF"], ["TRBV_UNKNOWN"], ["TRBJ_UNKNOWN"])
    assert np.isfinite(s) and s < 0
