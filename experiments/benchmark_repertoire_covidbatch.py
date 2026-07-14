"""Batch effects are real and large — and a within-batch (batch-orthogonal) contrast cancels them.

A sample-level embedding Φ(S) inevitably carries a **sequencing-batch nuisance** (run, protocol, date):
two repertoires from the same NovaSeq run look more alike than their biology warrants. Prop. ``prop:batch``
(Theory §T.7) says this offset is *first-order* — it shifts a whole batch by a common vector — so it
**cancels in a within-batch contrast**, and residualizing Φ on the batch indicator removes it while leaving
the batch-orthogonal biological signal intact.

This benchmark validates that on a **real 9-batch cohort** (FMBA DNA-multiplex TCRβ; Vlasova et al.,
Genome Med 2026, HF ``isalgo/airr_covid19``) with a built-in natural experiment:

* **COVID status is confounded with batch** — some runs are ~all-healthy (NovaSeq4) or all-precovid
  (NovaSeq9); only NovaSeq5/6/7 mix COVID⁺ and healthy. So a batch-blind COVID classifier can *ride the
  batch* → inflated. The honest COVID signal is the batch-residualized / within-mixed-batch number.
* **HLA is NOT confounded with batch** — HLA-A*02 carriage is donor genetics, sprinkled evenly across
  runs. So the HLA (clonotype-identity) signal is *already* batch-orthogonal → residualizing batch barely
  moves it. That contrast (COVID collapses under batch-correction, HLA survives) is the clean demonstration
  of ``prop:batch``.

We report: (1) batch is strongly encoded in Φ (multiclass OvR AUC ≫ chance); (2) same-status-cross-batch MMD
(pure batch offset) vs cross-status-within-batch MMD (biology); (3) COVID & HLA AUC, naive vs batch-residualized;
(4) batch AUC after residualization → drops to chance, confirming the cancellation.

Data: ``~/hf/airr_covid19`` local git-LFS checkout, else HF ``isalgo/airr_covid19`` (auto-fallback). Needs ``[bench]``.
Run:  python experiments/benchmark_repertoire_covidbatch.py [n_donors] [downsample_reads]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _cohort import cv_auc, pooled_clonotypes
from _covid import batch_of, load_covid, residualize

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, mmd_matrix, sample_embedding

N_PROTO, N_COMPONENTS, N_RFF, N_RFF_SECOND = 1000, 20, 2048, 256
MIXED = ("NovaSeq5", "NovaSeq6", "NovaSeq7")   # the batches carrying both COVID⁺ and healthy


def _multiclass_auc(X: np.ndarray, labels: np.ndarray, *, seed: int = 0) -> float:
    """Macro one-vs-rest AUC for predicting a categorical (batch) from Φ — how strongly Φ encodes it."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    keep = np.isin(labels, [b for b, c in zip(*np.unique(labels, return_counts=True)) if c >= 10])
    X, labels = X[keep], labels[keep]
    if len(np.unique(labels)) < 2:
        return float("nan")
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, labels):
        clf = make_pipeline(StandardScaler(), PCA(0.9, svd_solver="full", random_state=0),
                            LogisticRegression(max_iter=2000)).fit(X[tr], labels[tr])
        proba = clf.predict_proba(X[te])
        aucs.append(roc_auc_score(labels[te], proba[:, 1]) if len(clf.classes_) == 2 else
                    roc_auc_score(labels[te], proba, multi_class="ovr", average="macro", labels=clf.classes_))
    return float(np.mean(aucs))


def _mean_offdiag(D, row_mask, col_mask, same=False):
    sub = D[np.ix_(row_mask, col_mask)]
    return float(np.nanmean(sub[~np.eye(sub.shape[0], sub.shape[1], dtype=bool)] if same else sub))


def main(n_donors: int = 300, downsample_to: int = 20_000) -> None:
    t0 = time.perf_counter()
    rows, frames = load_covid(n_donors, downsample_to)
    covid = np.array([r["COVID_status"] == "COVID" for r in rows], dtype=int)
    a02 = np.array(["A*02" in (r["HLA-A_1"] or "") or "A*02" in (r["HLA-A_2"] or "") for r in rows], dtype=int)
    batch = batch_of(rows)
    mixed = np.isin(batch, MIXED)
    print(f"{len(rows)} donors ≤{downsample_to} reads: {covid.sum()} COVID⁺, {(~covid.astype(bool)).sum()} healthy; "
          f"{a02.sum()} A*02⁺; {len(np.unique(batch))} batches; {mixed.sum()} in mixed COVID/healthy batches\n")

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes([(None, f) for f in frames]),
                                 n_rff=N_RFF, n_rff_second=N_RFF_SECOND, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, f, blocks=("mean", "second")) for f in frames]
    X = np.hstack([np.stack([e.mean for e in embs]), np.stack([e.second for e in embs])])
    Xr = residualize(X, batch)                                      # batch-residualized Φ (prop:batch op)

    # (1) batch detectability, before/after residualization
    b_auc = _multiclass_auc(X, batch)
    b_auc_r = _multiclass_auc(Xr, batch)

    # (2) MMD decomposition: pure batch offset vs biological contrast
    D = mmd_matrix(embs)
    cov_b, heal_b = covid == 1, covid == 0
    same_cross = np.mean([  # same status, DIFFERENT batch = batch nuisance
        _mean_offdiag(D, (grp & (batch == bb)), (grp & (batch != bb)))
        for grp in (cov_b, heal_b) for bb in np.unique(batch) if (grp & (batch == bb)).sum() >= 3])
    cross_within = np.mean([  # different status, SAME batch = biology, batch cancels
        _mean_offdiag(D, (cov_b & (batch == bb)), (heal_b & (batch == bb)))
        for bb in np.unique(batch) if (cov_b & (batch == bb)).sum() >= 3 and (heal_b & (batch == bb)).sum() >= 3])

    # (3) COVID (confounded) & HLA (not confounded): naive vs batch-residualized
    cov_naive = cv_auc(X, covid, pca_cols=X.shape[1])
    cov_resid = cv_auc(Xr, covid, pca_cols=X.shape[1])
    cov_mixed = cv_auc(X[mixed], covid[mixed], pca_cols=X.shape[1])  # honest: within mixed batches only
    hla_naive = cv_auc(X, a02, pca_cols=X.shape[1])
    hla_resid = cv_auc(Xr, a02, pca_cols=X.shape[1])

    print(f"{'signal':<26}{'naive AUC':>16}{'batch-resid AUC':>18}{'Δ':>8}")
    print(f"{'batch identity (OvR)':<26}{b_auc:>16.3f}{b_auc_r:>18.3f}{b_auc_r - b_auc:>+8.3f}   (must collapse → cancellation)")
    print(f"{'COVID status (⟂̸ batch)':<26}{cov_naive[0]:>10.3f}±{cov_naive[1]:.3f}{cov_resid[0]:>12.3f}±{cov_resid[1]:.3f}"
          f"{cov_resid[0] - cov_naive[0]:>+8.3f}")
    print(f"{'HLA-A*02 (⟂ batch)':<26}{hla_naive[0]:>10.3f}±{hla_naive[1]:.3f}{hla_resid[0]:>12.3f}±{hla_resid[1]:.3f}"
          f"{hla_resid[0] - hla_naive[0]:>+8.3f}")
    print(f"\nCOVID within mixed batches only (batch-balanced, honest): {cov_mixed[0]:.3f}±{cov_mixed[1]:.3f}")
    print(f"MMD  same-status / cross-batch (batch offset) = {same_cross:.4f}")
    print(f"MMD  cross-status / within-batch (biology)     = {cross_within:.4f}   "
          f"ratio batch:biology = {same_cross / cross_within:.2f}")

    cancels = b_auc_r < 0.6 and b_auc > 0.7                          # batch encoded, then removed
    hla_survives = hla_resid[0] > hla_naive[0] - hla_naive[1]        # HLA (batch-⟂) preserved
    verdict = "PASS" if cancels and hla_survives else "PARTIAL" if cancels else "FAIL"
    print(f"\n[{verdict}] batch strongly encoded (OvR {b_auc:.2f}) then cancelled by residualization "
          f"({b_auc_r:.2f}); the batch-⟂ HLA signal survives ({hla_resid[0]:.2f} vs {hla_naive[0]:.2f}) while the "
          f"batch-confounded COVID signal {'drops toward its honest within-batch value' if cov_resid[0] < cov_naive[0] else 'holds'} "
          f"({cov_naive[0]:.2f}→{cov_resid[0]:.2f}; within-mixed {cov_mixed[0]:.2f}). prop:batch: within-batch contrasts are batch-free.")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 300, int(args[2]) if len(args) > 2 else 20_000)
