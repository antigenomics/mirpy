"""Reproduce the TCREMP VDJdb antigen-clustering benchmark (paper Table 1 / S1).

Embeds VDJdb TRB TCRs (TCREMP vjcdr3), PCA-denoises, DBSCAN-clusters with a
kneedle-selected eps (times the paper's dataset-specific factor), and reports
per-antigen F1 + retention against Table S1.

Needs a VDJdb slim dump (default: tests/assets/vdjdb.slim.txt.gz; see SOURCES.md).
Run (needs the [bench] extra):
    python experiments/benchmark_vdjdb.py [path/to/vdjdb.slim.txt.gz]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl

from mir.bench.metrics import cluster, cluster_metrics
from mir.bench.vdjdb import antigen_subset, load_vdjdb
from mir.embedding.pca import pca_denoise
from mir.embedding.tcremp import TCREmp

# Table S1 (TRB, tcremp): epitope -> (f1%, retention%)
PAPER = {
    "CINGVCWTV": (86, 7), "ELAGIGILTV": (82, 6), "GILGFVFTL": (99, 34),
    "GLCTLVAML": (99, 21), "KRWIILGLNK": (96, 26), "LLWNGPMAV": (83, 8),
    "NLVPMVATV": (93, 10), "SPRWYFYYL": (78, 6), "YLQPRTFLL": (100, 42),
}


def main(path: str) -> None:
    t0 = time.perf_counter()
    df = load_vdjdb(path)
    sub = antigen_subset(df, "TRB", 300).filter(pl.col("epitope").is_in(list(PAPER)))
    sub = sub.group_by("epitope", maintain_order=True).head(1500)  # cap dominance

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=3000, mode="vjcdr3")
    Xp = pca_denoise(model.embed(sub), n_components=50)
    labels = cluster(Xp, min_samples=3)  # eps = kneedle knee x 0.4
    mets = cluster_metrics(labels, sub["epitope"].to_list())

    print(f"{sub.height} TCRs, {sub['epitope'].n_unique()} antigens, "
          f"{100 * (labels >= 0).mean():.0f}% clustered")
    print(f"\n{'epitope':<14}{'n':>6}{'f1':>6}{'ret':>6}   {'paper':>12}")
    f1s, rets = [], []
    for ag, m in sorted(mets.items(), key=lambda kv: -kv[1].n):
        p = PAPER[ag]
        print(f"{ag:<14}{m.n:>6}{m.f1 * 100:>5.0f}%{m.retention * 100:>5.0f}%"
              f"   {p[0]:>4}% /{p[1]:>3}%")
        f1s.append(m.f1); rets.append(m.retention)
    print(f"\nmirpy mean F1={100 * np.mean(f1s):.0f}% ret={100 * np.mean(rets):.0f}%  |  "
          f"paper mean F1={np.mean([v[0] for v in PAPER.values()]):.0f}% "
          f"ret={np.mean([v[1] for v in PAPER.values()]):.0f}%")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/assets/vdjdb.slim.txt.gz")
