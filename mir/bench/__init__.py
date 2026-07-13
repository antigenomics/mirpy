"""VDJdb / 10X benchmark harness, clustering metrics, and theory-validation experiments.

* :mod:`mir.bench.vdjdb` — load VDJdb dumps into AIRR polars frames.
* :mod:`mir.bench.metrics` — DBSCAN (+ kneedle eps) and per-antigen F1 / retention
  (reproduces the paper's Table 1 / S1).
* :mod:`mir.bench.theory` — reproduce supplementary S1–S3 (distribution laws,
  dissimilarity↔distance correlation, real-vs-model prototype robustness).

Requires the ``[bench]`` extra (kneed, matplotlib, BioPython for the SW baseline).
"""

from mir.bench.metrics import AntigenMetric, cluster, cluster_metrics, estimate_dbscan_eps
from mir.bench.vdjdb import antigen_subset, load_vdjdb

__all__ = [
    "load_vdjdb",
    "antigen_subset",
    "cluster",
    "cluster_metrics",
    "estimate_dbscan_eps",
    "AntigenMetric",
]
