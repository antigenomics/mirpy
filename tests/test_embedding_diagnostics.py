"""Unit tests for shared embedding diagnostics helpers."""

from __future__ import annotations

import numpy as np

from mir.utils.embedding_diagnostics import (
    analyze_embedding_dbscan,
    classification_scores_by_label,
    cluster_purity_consistency,
    majority_vote_cluster_predictions,
)


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
        "eps_selection_mode",
        "eps_selector_meta",
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


def test_analyze_embedding_dbscan_stable_mode_returns_selector_meta() -> None:
    rng = np.random.default_rng(7)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.25, size=(30, 6)),
            rng.normal(loc=1.5, scale=0.25, size=(30, 6)),
        ]
    )
    labels = np.array(["A"] * 30 + ["B"] * 30)

    result = analyze_embedding_dbscan(
        X,
        labels,
        seed=7,
        eps_selection_mode="stable_kneedle",
    )

    assert result["eps"] > 0.0
    assert result["eps_selection_mode"] == "stable_kneedle"
    selector_meta = result["eps_selector_meta"]
    assert isinstance(selector_meta, dict)
    assert "knee_found" in selector_meta
    assert "eps_floor" in selector_meta
    assert "eps_cap" in selector_meta
    assert isinstance(selector_meta["knee_found"], bool)


def test_majority_vote_predictions_and_scores() -> None:
    labels = np.array(["E1", "E1", "E2", "E2", "E3", "E3"])
    clusters = np.array([0, 0, 1, 1, -1, 1])

    predicted = majority_vote_cluster_predictions(labels, clusters)
    scores = classification_scores_by_label(labels, predicted)

    assert predicted.shape == labels.shape
    # With the constructed clusters (majority vote maps cluster 0→E1, cluster 1→E2),
    # at least 4 of 6 items are correctly classified → accuracy > 0.5.
    assert float(scores["accuracy"]) > 0.5
    assert len(scores["per_label"]) == len(np.unique(labels))
