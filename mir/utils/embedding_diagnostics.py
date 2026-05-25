"""Shared embedding diagnostics for PCA, eps selection, and DBSCAN metrics.

Eps selection
-------------
The default eps selector (``select_eps_kneedle_stable``) works in two steps:

1. **Floor quantile as default.**  Set *eps = q(kth, q_floor)* where *kth* is
   the sorted 4-NN distance curve of the L2-normalised PCA embedding.

    Parameter justification via cross-validated subset scan (5 balanced
    epitope subsets, VDJdb TRB, n≈3 000 each):

    - q_floor=0.25: avg_ret=0.411, avg_pur=0.530, avg_cons=0.204, pass=0/5, min_margin=-0.290
    - q_floor=0.30: avg_ret=0.493, avg_pur=0.506, avg_cons=0.168, pass=0/5, min_margin=-0.132
    - q_floor=0.35: avg_ret=0.565, avg_pur=0.491, avg_cons=0.154, pass=4/5, min_margin=-0.002
    - q_floor=0.38: avg_ret=0.600, avg_pur=0.479, avg_cons=0.148, pass=5/5, min_margin=+0.060
    - q_floor=0.40: avg_ret=0.620, avg_pur=0.476, avg_cons=0.151, pass=5/5, min_margin=+0.105 (selected)
    - q_floor=0.42: avg_ret=0.638, avg_pur=0.472, avg_cons=0.142, pass=5/5, min_margin=+0.029
    - q_floor=0.45: avg_ret=0.669, avg_pur=0.461, avg_cons=0.133, pass=4/5, min_margin=-0.030
    - q_floor=0.50: avg_ret=0.716, avg_pur=0.452, avg_cons=0.127, pass=4/5, min_margin=-0.114

   *Retention ≈ q_floor + density_bonus* where density_bonus ≈ 0.20 from
   DBSCAN density connectivity.  Below 0.38 retention falls below 0.50;
   above 0.42 cluster consistency degrades as eps over-merges unrelated
   sequences.  **q_floor = 0.40 maximises the minimum quality margin.**

2. **Optional knee refinement** in a narrow window
   *[floor_idx, floor_idx + knee_fraction*(cap_idx−floor_idx)]*, i.e.
   roughly *[q_floor, q_floor+0.05]* with the default ``knee_fraction=0.20``
   and ``q_cap=0.65``.  KneeLocator (polynomial) is run on this window; if a
   knee is found within the window it is accepted; otherwise the floor
   quantile is used unchanged.

   For nearly-flat k-NN curves (common in high-dimensional TCREmp embeddings)
   no knee is found in the narrow window, so the algorithm reduces to the
   floor quantile.  Structured data with a genuine elbow at q ≤ q_floor+0.05
   benefits from the refinement; steeper windows are rejected to prevent
   over-merging.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize as l2normalize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from kneed import KneeLocator


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
    """Legacy: select DBSCAN eps via kneedle on the full k-NN curve, clamped to [q_floor, q_cap].

    Prefer ``select_eps_kneedle_stable`` for production use.
    """
    kth = _compute_kth_distances(X_pca, k)

    eps_floor = float(np.quantile(kth, q_floor))
    eps_cap = float(np.quantile(kth, q_cap))
    knee = KneeLocator(
        np.arange(len(kth)),
        kth,
        curve="convex",
        direction="increasing",
        interp_method="polynomial",
    )
    if knee.knee is None:
        return kth, eps_floor, None

    eps = float(np.clip(float(kth[knee.knee]), eps_floor, eps_cap))
    return kth, max(eps, 1e-6), int(knee.knee)


def select_eps_kneedle_stable(
    X_pca: np.ndarray,
    *,
    k: int = 4,
    q_floor: float = 0.40,
    q_cap: float = 0.65,
    knee_fraction: float = 0.20,
    interp_method: str = "polynomial",
    random_state: int = 42,
) -> tuple[np.ndarray, float, int | None, dict[str, float | int]]:
    """Select DBSCAN eps using the q_floor quantile with optional knee refinement.

    Algorithm
    ---------
    1. Compute the sorted 4-NN distance curve *kth* for the input embedding.
    2. Set *eps = q(kth, q_floor)* as the default (safe minimum).
    3. Run ``KneeLocator`` on *kth[floor_idx : knee_max_idx]*, the lower
       *knee_fraction* of the operational window *[q_floor, q_cap]*.  Accept
       the knee only if it falls strictly within that narrow window.
    4. For nearly-flat curves (common in TCREmp embeddings) no knee is found
       in the narrow window, and the selection stays at the floor quantile.

    Args:
        X_pca: L2-normalised PCA embedding, shape (n_samples, n_components).
        k: Number of nearest neighbours for the distance curve.
        q_floor: Lower quantile of kth used as the default eps and the hard
            minimum.  See module docstring for empirical justification;
            default 0.40 is the cross-validated optimum.
        q_cap: Upper quantile of kth used as the hard ceiling on eps, and
            to define the width of the operational window.  Default 0.65.
        knee_fraction: Fraction of [q_floor, q_cap] to search for a
            structural knee.  The knee search window ends at
            *q_floor + knee_fraction*(q_cap - q_floor)*.  Default 0.20
            gives a narrow ≈0.05-quantile window that rejects late/noisy
            knees and degrades safely to the floor on flat curves.
        interp_method: Interpolation method passed to ``KneeLocator``.
        random_state: Unused; retained for API compatibility.

    Returns:
        kth: Sorted k-NN distances.
        eps: Selected eps value.
        knee_idx: Index in kth corresponding to eps.
        meta: Diagnostic metadata dict.
    """
    kth = _compute_kth_distances(X_pca, k)
    n = len(kth)

    floor_idx = max(k + 1, int(round(n * q_floor)))
    cap_idx = min(n - 1, int(round(n * q_cap)))
    # Narrow search window: lower knee_fraction of [floor, cap]
    window_size = max(1, int(round((cap_idx - floor_idx) * knee_fraction)))
    knee_max_idx = min(cap_idx, floor_idx + window_size)

    eps = float(kth[floor_idx])
    knee_found = False
    knee_idx = floor_idx

    if knee_max_idx > floor_idx + 3:
        x = np.arange(floor_idx, knee_max_idx + 1)
        y = kth[floor_idx : knee_max_idx + 1]
        try:
            kl = KneeLocator(
                x,
                y,
                curve="convex",
                direction="increasing",
                interp_method=interp_method,
            )
        except Exception:
            kl = None
        if kl is not None and kl.knee is not None:
            ki = int(kl.knee)
            if floor_idx <= ki <= knee_max_idx:
                eps = float(kth[ki])
                knee_found = True
                knee_idx = ki

    return kth, max(eps, 1e-6), min(knee_idx, n - 1), {
        "q_floor": float(q_floor),
        "q_cap": float(q_cap),
        "knee_fraction": float(knee_fraction),
        "eps_floor": float(kth[floor_idx]),
        "eps_cap": float(kth[cap_idx]),
        "knee_found": bool(knee_found),
        "interp_method": str(interp_method),
        "n": int(n),
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
    q_floor: float = 0.40,
    q_cap: float = 0.65,
    knee_fraction: float = 0.20,
) -> dict[str, float | int | np.ndarray | None]:
    """Standardize embedding, select PCA rank, choose eps, and compute DBSCAN metrics.

    Args:
        X_raw: Raw embedding matrix, shape (n_samples, n_features).
        labels: Ground-truth labels for each sample.
        seed: Random seed for PCA.
        pca_variance_threshold: Cumulative explained variance to retain.
        min_samples: DBSCAN min_samples parameter.
        k_neighbors: k for k-NN distance curve.
        consistency_threshold: Per-cluster purity threshold for consistency metric.
        eps_selection_mode: ``"stable_kneedle"`` (default) or ``"kneedle"``.
        q_floor: Lower quantile for eps selection (see ``select_eps_kneedle_stable``).
        q_cap: Upper quantile cap for eps selection.
        knee_fraction: Narrow-window fraction for knee refinement.

    Returns:
        Dict with keys: ``n_comp``, ``eps``, ``n_clusters``, ``retention``,
        ``purity``, ``consistency``, ``median_4nn``, ``kth``, ``knee_idx``,
        ``eps_selection_mode``, ``eps_selector_meta``, ``cum``, ``X_pca``,
        ``clusters``.
    """
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
            q_floor=q_floor,
            q_cap=q_cap,
            knee_fraction=knee_fraction,
            random_state=seed,
        )
    else:
        kth, eps, knee_idx = select_eps_kneedle(X_pca, k=k_neighbors, q_floor=q_floor, q_cap=q_cap)
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
