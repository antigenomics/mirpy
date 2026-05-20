"""Shared embedding diagnostics for PCA, bootstrap-knee eps selection, and DBSCAN metrics.

The default eps selector uses bootstrap-stable knee detection on k-NN distance
curves and is designed for reproducibility and scalability:

1. Build a sorted k-th NN distance curve on the full PCA embedding.
2. Draw multiple random subsets of rows and recompute subset-specific curves.
3. Detect knees using two kneed interpolation methods (polynomial, interp1d).
4. Aggregate candidate eps values and derive bootstrap-informed bounds.
5. Select the final eps as a robust location estimate (median) within bounds.

For very large datasets, subset size is bounded so total work scales with the
subset cap instead of full dataset size per bootstrap iteration.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize as l2normalize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from kneed import KneeLocator


def _fallback_curvature_knee_index(sorted_curve: np.ndarray) -> int:
    """Estimate a knee index from maximum log-curve curvature."""
    n = len(sorted_curve)
    if n < 5:
        return max(0, int(round((n - 1) * 0.5)))

    y = np.log(np.maximum(sorted_curve, 1e-12))
    d1 = np.gradient(y)
    d2 = np.gradient(d1)
    return int(np.argmax(d2))


def _compute_kth_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Return sorted k-th neighbor distances for rows in X."""
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    return np.sort(dists[:, -1])


def select_eps_kneedle(
    X_pca: np.ndarray,
    *,
    k: int = 4,
    q_floor: float = 0.40,
    q_cap: float = 0.65,
) -> tuple[np.ndarray, float, int | None]:
    """Select DBSCAN eps from sorted k-NN distances with quantile bounds."""
    kth = _compute_kth_distances(X_pca, k)

    eps_floor = float(np.quantile(kth, q_floor))
    eps_cap = float(np.quantile(kth, q_cap))

    eps_floor = max(eps_floor, 1e-6)
    eps_cap = max(eps_cap, eps_floor)
    knee = KneeLocator(
        np.arange(len(kth)),
        kth,
        curve="convex",
        direction="increasing",
        interp_method="polynomial",
    )
    if knee.knee is None:
        return kth, eps_floor, None

    # Clamp the knee-derived eps into a practical operating band so DBSCAN
    # does not collapse into a low-retention regime on broad real datasets.
    eps = float(np.clip(float(kth[knee.knee]), eps_floor, eps_cap))
    return kth, max(eps, 1e-6), int(knee.knee)


def select_eps_kneedle_stable(
    X_pca: np.ndarray,
    *,
    k: int = 4,
    n_bootstrap: int | None = None,
    min_bootstrap: int = 10,
    max_bootstrap: int = 100,
    subset_fraction: float = 0.70,
    large_dataset_threshold: int = 100_000,
    max_subset_size: int = 35_000,
    bounds_quantiles: tuple[float, float] = (0.20, 0.80),
    operational_quantiles: tuple[float, float] = (0.40, 0.65),
    final_quantiles: tuple[float, float] = (0.20, 0.60),
    final_selection_quantile: float = 0.35,
    interp_methods: tuple[str, ...] = ("polynomial", "interp1d"),
    random_state: int = 42,
) -> tuple[np.ndarray, float, int | None, dict[str, float | int]]:
    """Select DBSCAN eps from bootstrap-stable knees over random row subsets.

    This function recomputes k-NN curves on random subsets of the input sample,
    aggregates knee candidates across interpolation methods, and derives eps
    bounds directly from bootstrap statistics.
    """
    kth = _compute_kth_distances(X_pca, k)

    rng = np.random.default_rng(random_state)
    n = len(X_pca)
    if n_bootstrap is None:
        auto_bootstrap = int(round(math.sqrt(max(1, n)) / 5.0))
        n_bootstrap = max(min_bootstrap, min(max_bootstrap, auto_bootstrap))
    else:
        n_bootstrap = max(min_bootstrap, min(max_bootstrap, int(n_bootstrap)))

    subset_size = max(k + 2, int(round(n * subset_fraction)))
    if n > large_dataset_threshold:
        subset_size = min(subset_size, max_subset_size)
    subset_size = min(subset_size, n)

    eps_candidates: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=subset_size, replace=False)
        kth_subset = _compute_kth_distances(X_pca[idx], k)
        x = np.arange(len(kth_subset))
        found = False
        for interp_method in interp_methods:
            try:
                knee = KneeLocator(
                    x,
                    kth_subset,
                    curve="convex",
                    direction="increasing",
                    interp_method=interp_method,
                )
            except Exception:
                continue
            if knee.knee is None:
                continue
            denom = max(1, len(kth_subset) - 1)
            p = float(int(knee.knee) / denom)
            eps_candidates.append(float(np.quantile(kth, p)))
            found = True

        if not found:
            knee_idx = _fallback_curvature_knee_index(kth_subset)
            denom = max(1, len(kth_subset) - 1)
            p = float(knee_idx / denom)
            eps_candidates.append(float(np.quantile(kth, p)))

    eps_candidates_arr = np.asarray(eps_candidates, dtype=float)
    b_low_q, b_high_q = bounds_quantiles
    eps_floor_boot = max(float(np.quantile(eps_candidates_arr, b_low_q)), 1e-6)
    eps_cap_boot = max(float(np.quantile(eps_candidates_arr, b_high_q)), eps_floor_boot)

    op_low_q, op_high_q = operational_quantiles
    eps_floor_op = max(float(np.quantile(kth, op_low_q)), 1e-6)
    eps_cap_op = max(float(np.quantile(kth, op_high_q)), eps_floor_op)

    eps_floor = max(eps_floor_boot, eps_floor_op)
    eps_cap = max(min(eps_cap_boot, eps_cap_op), eps_floor)

    f_low_q, f_high_q = final_quantiles
    eps_center = float(np.quantile(eps_candidates_arr, final_selection_quantile))
    eps = float(np.clip(eps_center, eps_floor, eps_cap))
    knee_idx = int(np.searchsorted(kth, eps, side="left"))

    q25, q75 = np.quantile(eps_candidates_arr, [f_low_q, f_high_q])
    return kth, max(eps, 1e-6), min(knee_idx, n - 1), {
        "eps_floor": float(eps_floor),
        "eps_cap": float(eps_cap),
        "eps_floor_boot": float(eps_floor_boot),
        "eps_cap_boot": float(eps_cap_boot),
        "eps_floor_operational": float(eps_floor_op),
        "eps_cap_operational": float(eps_cap_op),
        "n_bootstrap": int(n_bootstrap),
        "subset_size": int(subset_size),
        "subset_fraction": float(subset_fraction),
        "large_dataset_threshold": int(large_dataset_threshold),
        "max_subset_size": int(max_subset_size),
        "n_candidates": int(len(eps_candidates)),
        "eps_iqr": float(q75 - q25),
        "final_selection_quantile": float(final_selection_quantile),
        "final_quantile_low": float(f_low_q),
        "final_quantile_high": float(f_high_q),
    }


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
    eps_selection_mode: str = "stable_kneedle",
    n_bootstrap: int | None = None,
    subset_fraction: float = 0.70,
    large_dataset_threshold: int = 100_000,
    max_subset_size: int = 35_000,
    bounds_quantiles: tuple[float, float] = (0.20, 0.80),
    operational_quantiles: tuple[float, float] = (0.40, 0.65),
    final_quantiles: tuple[float, float] = (0.20, 0.60),
    final_selection_quantile: float = 0.35,
    interp_methods: tuple[str, ...] = ("polynomial", "interp1d"),
) -> dict[str, float | int | np.ndarray | None]:
    """Standardize embedding, select PCA rank, choose eps, and compute DBSCAN metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca_full = PCA(random_state=seed).fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cum, pca_variance_threshold)) + 1

    X_pca = l2normalize(PCA(n_components=n_comp, random_state=seed).fit_transform(X_scaled))
    selector_meta: dict[str, float | int] = {}
    if eps_selection_mode == "stable_kneedle":
        kth, eps, knee_idx, selector_meta = select_eps_kneedle_stable(
            X_pca,
            k=k_neighbors,
            n_bootstrap=n_bootstrap,
            subset_fraction=subset_fraction,
            large_dataset_threshold=large_dataset_threshold,
            max_subset_size=max_subset_size,
            bounds_quantiles=bounds_quantiles,
            operational_quantiles=operational_quantiles,
            final_quantiles=final_quantiles,
            final_selection_quantile=final_selection_quantile,
            interp_methods=interp_methods,
            random_state=seed,
        )
    else:
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
        "eps_selection_mode": eps_selection_mode,
        "eps_selector_meta": selector_meta,
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