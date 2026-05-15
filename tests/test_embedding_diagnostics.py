"""Unit tests for shared embedding diagnostics helpers."""

from __future__ import annotations

import numpy as np

from mir.utils.embedding_diagnostics import analyze_embedding_dbscan, cluster_purity_consistency


def test_cluster_purity_consistency_handles_noise_and_clusters() -> None:
    labels = np.array(["A", "A", "B", "B", "B"])
    clusters = np.array([0, 0, 1, -1, 1])

    purity, consistency, n_clusters, retention = cluster_purity_consistency(labels, clusters)

    assert n_clusters == 2
    assert retention == 0.8
    assert 0.0 <= purity <= 1.0
    assert 0.0 <= consistency <= 1.0


def test_analyze_embedding_dbscan_returns_expected_keys() -> None:
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.2, size=(40, 8)),
            rng.normal(loc=2.0, scale=0.2, size=(40, 8)),
        ]
    )
    labels = np.array(["A"] * 40 + ["B"] * 40)

    result = analyze_embedding_dbscan(X, labels, seed=42)

    expected = {
        "n_comp",
        "eps",
        "n_clusters",
        "retention",
        "purity",
        "consistency",
        "median_4nn",
        "kth",
        "knee_idx",
        "cum",
        "X_pca",
        "clusters",
    }
    assert expected.issubset(result.keys())
    assert result["n_comp"] >= 1
    assert result["eps"] > 0.0
    assert 0.0 <= result["retention"] <= 1.0
    assert 0.0 <= result["purity"] <= 1.0
    assert 0.0 <= result["consistency"] <= 1.0
