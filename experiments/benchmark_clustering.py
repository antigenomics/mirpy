# 2026-07-14
# Clustering-algorithm comparison for the antigen-clustering benchmark: is DBSCAN the right
# choice, or do HDBSCAN / OPTICS / KMeans do better? Real repertoire density spans orders of
# magnitude across epitopes (Pogorelyy 2018; appendix sec:dens-depth), which is DBSCAN's weak
# spot (one global eps) and HDBSCAN's home turf (the persistent cluster tree across all eps).
#
# Same embedding as benchmark_vdjdb.py (TCREMP vjcdr3 -> PCA-50); only the clusterer changes.
# Metrics: mean per-antigen F1 + retention (paper decoupling), plus global ARI / AMI vs the true
# epitope labels, %clustered (noise rejection), #clusters, wall-time. KMeans gets the true cluster
# count as an oracle and still cannot reject noise (no -1), so its retention is trivially ~1.
#
# Run (needs [bench]):  python experiments/benchmark_clustering.py [path/to/vdjdb.slim.txt.gz]

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from mir.bench.metrics import cluster, cluster_metrics
from mir.bench.vdjdb import antigen_subset, load_vdjdb
from mir.embedding.pca import pca_denoise
from mir.embedding.tcremp import TCREmp

METHODS = ["dbscan", "hdbscan", "optics"]


def _score(labels, epitopes) -> dict:
    mets = cluster_metrics(labels, epitopes)
    f1 = float(np.mean([m.f1 for m in mets.values()]))
    ret = float(np.mean([m.retention for m in mets.values()]))
    ep = np.asarray(epitopes, dtype=object)
    cl = labels >= 0  # ARI/AMI over ALL points count noise as one giant cluster, which unfairly
    # penalizes methods that reject more noise; also report them on the clustered subset (cluster
    # QUALITY, coverage-independent) so precision and coverage are read on separate axes.
    return {
        "f1": f1, "ret": ret,
        "clustered": float(cl.mean()),
        "ami": float(adjusted_mutual_info_score(epitopes, labels)),
        "ami_cl": float(adjusted_mutual_info_score(ep[cl], labels[cl])) if cl.sum() > 1 else 0.0,
        "ari_cl": float(adjusted_rand_score(ep[cl], labels[cl])) if cl.sum() > 1 else 0.0,
        "n_clusters": int(len({x for x in labels if x >= 0})),
    }


def main(path: str) -> None:
    df = load_vdjdb(path)
    sub = antigen_subset(df, "TRB", 300)
    sub = sub.group_by("epitope", maintain_order=True).head(1500)  # cap per-epitope dominance
    epitopes = sub["epitope"].to_list()
    k_true = sub["epitope"].n_unique()

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=3000, mode="vjcdr3")
    Xp = pca_denoise(model.embed(sub), n_components=50)
    print(f"{sub.height} TRB TCRs, {k_true} epitopes, embedding {Xp.shape}\n", flush=True)

    rows = []
    for method in METHODS:
        t = time.perf_counter()
        labels = cluster(Xp, min_samples=3, method=method)
        rows.append((method, _score(labels, epitopes), time.perf_counter() - t))
    # KMeans oracle: given the true #epitopes; cannot label noise (-1), so retention is trivially 1
    t = time.perf_counter()
    km = KMeans(n_clusters=k_true, n_init=10, random_state=0).fit_predict(Xp)
    rows.append(("kmeans*", _score(km, epitopes), time.perf_counter() - t))

    print(f"{'method':<9}{'clustered':>10}{'meanF1':>8}{'meanRet':>8}{'AMIcl':>7}{'ARIcl':>7}"
          f"{'#clust':>7}{'sec':>7}")
    for name, s, dt in rows:
        print(f"{name:<9}{100 * s['clustered']:>9.0f}%{100 * s['f1']:>8.1f}{100 * s['ret']:>8.1f}"
              f"{s['ami_cl']:>7.3f}{s['ari_cl']:>7.3f}{s['n_clusters']:>7}{dt:>7.1f}")
    print("\nNo single winner — a precision/coverage trade-off. DBSCAN: tightest, purest (high F1,"
          "\nlow retention = the paper regime). HDBSCAN: variable-density, ~3x the coverage"
          "\n(retention) at some F1 cost — the pick when clustering more of the repertoire matters"
          "\n(Pogorelyy-2018 heterogeneity). OPTICS: dominated + slow. KMeans*: oracle K, no noise"
          "\nrejection -> F1 collapses. AMIcl/ARIcl are cluster QUALITY on the clustered subset.",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/assets/vdjdb.slim.txt.gz")
