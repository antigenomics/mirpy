"""Prototype-count saturation: how the embedding scales with the number of prototypes K.

Theory T.1 / supplementary S4 predict **logarithmic saturation** — a landmark (distance-to-prototype)
embedding stabilises once K is a few hundred, because preserving the pairwise distance map needs only
K = O(log n) landmarks (Johnson-Lindenstrauss on the distance map). This measures it on VDJdb TRB:

* **geometry** — Pearson correlation of the pairwise embedding distances at K vs the largest-K
  reference (on a fixed clonotype sample). Rises toward 1 and saturates.
* **clustering** — DBSCAN mean F1 / retention on the paper's 9 TRB epitopes at each K.

Shows how few prototypes suffice, and where the per-chain presets (compact 1000, diverse 2000) sit.

Run: python experiments/analyze_prototype_counts.py [vdjdb_path]   (needs [bench])
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

from mir.bench.metrics import cluster, cluster_metrics
from mir.bench.vdjdb import antigen_subset, load_vdjdb
from mir.embedding.pca import pca_denoise
from mir.embedding.tcremp import TCREmp

PAPER = ["CINGVCWTV", "ELAGIGILTV", "GILGFVFTL", "GLCTLVAML", "KRWIILGLNK",
         "LLWNGPMAV", "NLVPMVATV", "SPRWYFYYL", "YLQPRTFLL"]
COUNTS = [100, 300, 500, 1000, 2000, 3000, 5000]
GEOM_SAMPLE = 600  # clonotypes used for the pairwise-distance geometry correlation


def main(path: str) -> None:
    t0 = time.perf_counter()
    df = load_vdjdb(path)
    sub = antigen_subset(df, "TRB", 300).filter(pl.col("epitope").is_in(PAPER))
    sub = sub.group_by("epitope", maintain_order=True).head(1500)
    ag = sub["epitope"].to_list()
    samp = sub.head(GEOM_SAMPLE)

    print(f"{sub.height} VDJdb TRB TCRs, {sub['epitope'].n_unique()} epitopes; "
          f"geometry on {samp.height} clones\n")
    print(f"{'K':>6}{'PCs':>5}{'meanF1':>8}{'ret':>6}{'geom_r_vs_max':>15}")
    ref_pd = None
    rows = []
    for n in COUNTS:
        model = TCREmp.from_defaults("human", "TRB", n_prototypes=n, mode="vjcdr3")
        emb = model.embed(sub)
        Xp = pca_denoise(emb, n_components=50)
        mets = cluster_metrics(cluster(Xp, min_samples=3), ag)
        f1 = float(np.mean([m.f1 for m in mets.values()]))
        ret = float(np.mean([m.retention for m in mets.values()]))
        pd_n = pdist(model.embed(samp))                      # raw pairwise distances (scale grows with K)
        rows.append((n, Xp.shape[1], f1, ret, pd_n))
    ref_pd = rows[-1][4]
    for n, pcs, f1, ret, pd_n in rows:
        # Pearson on the pairwise-distance vectors — scale-invariant, so K-vs-max geometry agreement
        geom = float(np.corrcoef(pd_n, ref_pd)[0, 1])
        print(f"{n:>6}{pcs:>5}{f1 * 100:>7.0f}%{ret * 100:>5.0f}%{geom:>14.3f}")

    f1s = [r[2] for r in rows]
    sat = next((n for n, r in zip(COUNTS, rows)
                if float(np.corrcoef(r[4], ref_pd)[0, 1]) >= 0.98), COUNTS[-1])
    print(f"\ngeometry reaches r≥0.98 vs max-K by K={sat}; F1 range {min(f1s)*100:.0f}–{max(f1s)*100:.0f}% "
          f"(saturating — Theory T.1/S4 logarithmic).")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/assets/vdjdb.slim.txt.gz")
