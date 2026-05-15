"""Shared embedding diagnostics for PCA, kneedle, and DBSCAN quality metrics."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize as l2normalize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from kneed import KneeLocator
except Exception:  # pragma: no cover - optional dependency fallback
    KneeLocator = None


def select_eps_kneedle(
    X_pca: np.ndarray,
    *,
    k: int = 4,
    q_floor: float = 0.10,
    q_cap: float = 0.40,
) -> tuple[np.ndarray, float, int | None]:
    """Select DBSCAN eps from sorted k-NN distances with quantile bounds."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_pca)
    dists, _ = nn.kneighbors(X_pca)
    kth = np.sort(dists[:, -1])

    eps_floor = float(np.quantile(kth, q_floor))
    eps_cap = float(np.quantile(kth, q_cap))

    eps_floor = max(eps_floor, 1e-6)
    eps_cap = max(eps_cap, eps_floor)
    if KneeLocator is None:
        return kth, eps_floor, None

    knee = KneeLocator(
        np.arange(len(kth)),
        kth,
        curve="convex",
        direction="increasing",
        interp_method="polynomial",
    )
    if knee.knee is None:
        return kth, eps_floor, None

    eps = min(float(kth[knee.knee]), eps_cap)
    return kth, max(eps, eps_floor, 1e-6), int(knee.knee)


def cluster_purity_consistency(
    labels: np.ndarray,
    clusters: np.ndarray,
    *,
    consistency_threshold: float = 0.70,
) -> tuple[float, float, int, float]:
    """Return (purity, consistency, n_clusters, retention) for clustered points."""
    mask = clusters != -1
    retained = labels[mask]
    cluster_ids = np.unique(clusters[mask])

    if len(cluster_ids) == 0:
        return 0.0, 0.0, 0, float(mask.mean())

    per_cluster_purity: list[float] = []
    for cid in cluster_ids:
        cl_labels = retained[clusters[mask] == cid]
        _, counts = np.unique(cl_labels, return_counts=True)
        per_cluster_purity.append(float(counts.max() / counts.sum()))

    purity = float(np.mean(per_cluster_purity))
    consistency = float(np.mean(np.array(per_cluster_purity) >= consistency_threshold))
    return purity, consistency, int(len(cluster_ids)), float(mask.mean())


def analyze_embedding_dbscan(
    X_raw: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 42,
    pca_variance_threshold: float = 0.90,
    min_samples: int = 3,
    k_neighbors: int = 4,
    consistency_threshold: float = 0.70,
) -> dict[str, float | int | np.ndarray | None]:
    """Standardize embedding, select PCA rank, choose eps, and compute DBSCAN metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca_full = PCA(random_state=seed).fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, pca_variance_threshold)) + 1

    X_pca = l2normalize(PCA(n_components=n_comp, random_state=seed).fit_transform(X_scaled))
    kth, eps, knee_idx = select_eps_kneedle(X_pca, k=k_neighbors)
    clusters = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1).fit_predict(X_pca)

    purity, consistency, n_clusters, retention = cluster_purity_consistency(
        labels,
        clusters,
        consistency_threshold=consistency_threshold,
    )

    return {
        "n_comp": int(n_comp),
        "eps": float(eps),
        "n_clusters": int(n_clusters),
        "retention": float(retention),
        "purity": float(purity),
        "consistency": float(consistency),
        "median_4nn": float(np.median(kth)),
        "kth": kth,
        "knee_idx": knee_idx,
        "cum": cum,
        "X_pca": X_pca,
        "clusters": clusters,
    }


def majority_vote_cluster_predictions(
    labels: np.ndarray,
    clusters: np.ndarray,
    *,
    noise_label: str = "noise",
) -> np.ndarray:
    """Map cluster ids to majority labels and return per-point predicted labels."""
    labels = np.asarray(labels)
    clusters = np.asarray(clusters)
    preds = np.full(labels.shape, noise_label, dtype=object)

    cluster_ids = np.unique(clusters[clusters != -1])
    for cid in cluster_ids:
        idx = clusters == cid
        cl_labels = labels[idx]
        uniq, counts = np.unique(cl_labels, return_counts=True)
        preds[idx] = uniq[np.argmax(counts)]
    return preds


def classification_scores_by_label(
    labels: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | list[dict[str, float | str | int]]]:
    """Compute per-label precision/recall/F1/support and global summary scores."""
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)

    unique_labels = np.unique(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        labels=unique_labels,
        average=None,
        zero_division=0,
    )

    per_label = [
        {
            "label": str(label),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(unique_labels)
    ]

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predicted,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_label": per_label,
    }