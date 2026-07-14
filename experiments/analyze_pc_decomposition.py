"""What the TCREMP embedding encodes: V vs J vs CDR3(junction), and CDR3 length (Theory T3/T.4).

The raw embedding interleaves three distance blocks per prototype — V (cols 0::3), J (1::3),
junction (2::3). Theory T.4 says the germline (V/J) blocks are **low rank** (rank ≤ #distinct
genes ≪ K), so a few PCs absorb V/J and the rest carry the high-rank junction (CDR3) signal. This
quantifies it on a diverse VDJdb TRB sample:

* **block variance shares** — fraction of total embedding variance in the V / J / junction blocks.
* **per-PC composition** — each PC's squared-loading fraction from V / J / junction (early PCs
  should be V/J-dominated, later PCs junction).
* **ANOVA η²** — fraction of embedding variance explained by V-gene identity, J-gene identity, and
  CDR3-length bin.
* **length recoverability** — R² of CDR3 length regressed on the PCs (how strongly length is encoded).

Run: python experiments/analyze_pc_decomposition.py [vdjdb_path]   (needs [bench])
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mir.bench.vdjdb import load_vdjdb
from mir.embedding.tcremp import TCREmp

N_PROTO = 1000
N_PC = 100
SAMPLE = 4000


def _eta2(X: np.ndarray, labels) -> float:
    """Multivariate η²: fraction of total (summed-over-dims) variance explained by group membership."""
    labels = np.asarray(labels)
    grand = X.mean(0)
    ss_tot = float(((X - grand) ** 2).sum())
    ss_between = 0.0
    for g in np.unique(labels):
        Xg = X[labels == g]
        ss_between += Xg.shape[0] * float(((Xg.mean(0) - grand) ** 2).sum())
    return ss_between / ss_tot if ss_tot else 0.0


def main(path: str) -> None:
    df = load_vdjdb(path).filter(pl.col("locus") == "TRB")
    sub = df.sample(min(SAMPLE, df.height), seed=0) if df.height > SAMPLE else df
    v_gene = [str(x).split("*")[0] for x in sub["v_call"]]
    j_gene = [str(x).split("*")[0] for x in sub["j_call"]]
    length = np.array([len(s) for s in sub["junction_aa"]])

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO, mode="vjcdr3")
    emb = model.embed(sub)                                   # (N, 3K): [V, J, junction] interleaved
    v_b, j_b, jun_b = emb[:, 0::3], emb[:, 1::3], emb[:, 2::3]
    tot = float(emb.var(0).sum())
    print(f"{sub.height} VDJdb TRB TCRs, K={model.n_prototypes} prototypes, "
          f"{len(set(v_gene))} V / {len(set(j_gene))} J genes\n")
    print("raw block variance shares (magnitude, not info — V genes are long so their distances are "
          "numerically huge; hence StandardScaler before PCA, Theory T3):")
    print(f"    V {v_b.var(0).sum() / tot:>5.0%}   J {j_b.var(0).sum() / tot:>5.0%}   "
          f"junction {jun_b.var(0).sum() / tot:>5.0%}")

    Xs = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=min(N_PC, sub.height, emb.shape[1]), random_state=0).fit(Xs)
    comps, scores = pca.components_, pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    # each PC's loading energy split across the three interleaved blocks
    v_load = (comps[:, 0::3] ** 2).sum(1)
    j_load = (comps[:, 1::3] ** 2).sum(1)
    jun_load = (comps[:, 2::3] ** 2).sum(1)

    print(f"\n{'PC':>4}{'var%':>7}{'V%':>6}{'J%':>6}{'junc%':>7}{'|corr(len)|':>12}")
    for k in range(min(12, comps.shape[0])):
        rlen = abs(float(np.corrcoef(scores[:, k], length)[0, 1]))
        print(f"{k + 1:>4}{evr[k] * 100:>6.1f}%{v_load[k] * 100:>5.0f}%{j_load[k] * 100:>5.0f}%"
              f"{jun_load[k] * 100:>6.0f}%{rlen:>12.2f}")

    # how many PCs to absorb V/J (their low-rank prediction) vs total
    v_pc = int((v_load > 0.5).sum())
    j_pc = int((j_load > 0.5).sum())
    # length recoverability: R² of length ~ all PCs (multiple regression)
    A = np.column_stack([scores, np.ones(scores.shape[0])])
    coef, *_ = np.linalg.lstsq(A, length, rcond=None)
    r2_len = 1.0 - float(((length - A @ coef) ** 2).sum()) / float(((length - length.mean()) ** 2).sum())

    print(f"\nANOVA η² (share of embedding variance): "
          f"V-gene {_eta2(Xs, v_gene):.2f}   J-gene {_eta2(Xs, j_gene):.2f}   "
          f"CDR3-length {_eta2(Xs, length):.2f}")
    print(f"PCs that are majority V-loaded: {v_pc}; majority J-loaded: {j_pc}  "
          f"(≪ {comps.shape[0]} — germline blocks are low-rank, Theory T.4)")
    print(f"CDR3 length recoverable from PCs: R²={r2_len:.2f}  "
          f"(length is a strong axis of the junction block)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/assets/vdjdb.slim.txt.gz")
