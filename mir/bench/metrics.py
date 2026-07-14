"""Antigen-cluster metrics for the TCREMP benchmark (paper Table 1 / S1).

DBSCAN clustering with a kneedle-selected ``eps`` on embedded TCRs, then per-antigen
**F1** (on clustered TCRs, via cluster→majority-label assignment) and **retention**
(fraction of an antigen's TCRs that land in a cluster), following the decoupling used
in the paper: F1 measures clustering *quality*, retention measures *coverage*.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def estimate_dbscan_eps(X: np.ndarray, k: int = 4) -> float:
    """Kneedle ``eps``: the knee of the sorted k-th nearest-neighbour distance curve."""
    from kneed import KneeLocator

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    kdist = np.sort(nn.kneighbors(X)[0][:, k - 1])
    kl = KneeLocator(
        np.arange(kdist.size), kdist, curve="convex", direction="increasing",
        interp_method="polynomial", polynomial_degree=7,
    )
    if kl.knee is None:
        return float(np.median(kdist))
    return float(kdist[int(kl.knee)])


def cluster(X: np.ndarray, eps: float | None = None, min_samples: int = 3,
            k: int = 4, eps_factor: float = 0.4, method: str = "dbscan", **kwargs) -> np.ndarray:
    """Density-cluster an embedding; noise = ``-1`` for every method.

    Args:
        method: ``"dbscan"`` (default), ``"hdbscan"``, or ``"optics"``. All three are density
            estimators that emit ``-1`` noise, so :func:`cluster_metrics` and
            :func:`mir.density.denoise_and_cluster` work unchanged across them. ``"dbscan"`` is
            the default for reproducibility of the published Table-S1 numbers.
        eps / eps_factor / k: DBSCAN only. When ``eps is None`` it is ``kneedle_knee × eps_factor``
            — the paper's ``eps := (distance at knee) × (dataset-specific factor)`` (Fig 1); the
            raw knee over-merges in the PCA embedding, so ~0.3–0.4 recovers tight antigen clusters.
        min_samples: core-point / density threshold, shared by all three methods.
        **kwargs: passed to the underlying estimator (e.g. ``min_cluster_size`` for HDBSCAN,
            ``cluster_method`` / ``xi`` for OPTICS).

    HDBSCAN is a variable-density estimator — the persistent cluster tree (Hartigan level sets)
    rather than DBSCAN's single global ``eps`` slice — so it is the natural choice when local
    density spans orders of magnitude across antigen ridges (appendix ``sec:dens-depth``).
    """
    if method == "dbscan":
        if eps is None:
            eps = estimate_dbscan_eps(X, k=k) * eps_factor
        return DBSCAN(eps=eps, min_samples=min_samples, **kwargs).fit_predict(X)
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN

        kwargs.setdefault("min_cluster_size", max(min_samples, 2))
        kwargs.setdefault("copy", True)  # don't mutate the caller's array (silences FutureWarning)
        return HDBSCAN(min_samples=min_samples, **kwargs).fit_predict(X)
    if method == "optics":
        from sklearn.cluster import OPTICS

        return OPTICS(min_samples=min_samples, **kwargs).fit_predict(X)
    raise ValueError(f"method must be 'dbscan', 'hdbscan', or 'optics', got {method!r}")


@dataclass
class AntigenMetric:
    epitope: str
    n: int
    f1: float
    precision: float
    recall: float
    retention: float


def cluster_metrics(labels: np.ndarray, antigens) -> dict[str, AntigenMetric]:
    """Per-antigen F1 (on clustered TCRs) and retention.

    Args:
        labels: DBSCAN cluster labels (``-1`` = noise/unclustered).
        antigens: True antigen (epitope) label per TCR, same length as *labels*.

    Returns:
        ``epitope -> AntigenMetric``.
    """
    labels = np.asarray(labels)
    antigens = np.asarray(antigens, dtype=object)
    clustered = labels >= 0

    # cluster -> majority antigen (over clustered members)
    majority: dict[int, str] = {}
    for lab in np.unique(labels[clustered]):
        members = antigens[clustered & (labels == lab)]
        majority[int(lab)] = Counter(members).most_common(1)[0][0]

    predicted = np.array(
        [majority.get(int(l), None) if l >= 0 else None for l in labels], dtype=object
    )

    out: dict[str, AntigenMetric] = {}
    for ag in sorted(set(antigens.tolist())):
        true_ag = antigens == ag
        n = int(true_ag.sum())
        pred_ag = predicted == ag
        tp = int((true_ag & pred_ag).sum())
        n_pred = int(pred_ag.sum())
        n_true_clustered = int((true_ag & clustered).sum())
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true_clustered if n_true_clustered else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        retention = n_true_clustered / n if n else 0.0
        out[ag] = AntigenMetric(ag, n, f1, precision, recall, retention)
    return out
