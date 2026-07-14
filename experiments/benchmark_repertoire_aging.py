"""Repertoire embedding (mir.repertoire) predicts donor age better than diversity/k-mer baselines.

The aging cohort (Britanova 2014/2016; HF ``isalgo/airr_benchmark`` ``vdjtools/metadata_aging.txt``,
79 donors aged 6–90) shows the classic immune-age signal: repertoire diversity falls with age. The
sample-level embedding ``Φ(S)`` (RFF kernel mean ‖ coverage/observed Hill profile, Theory §T.7) should
predict age *better* than the ¹D-diversity-only and k-mer-profile baselines — i.e. the kernel-mean
backbone adds structure beyond a scalar diversity summary (Prop. ``prop:kme``; distribution regression,
Szabó 2016).

Each donor is downsampled to a common shallow depth (``--downsample``, the RNA-seq regime — cost is
per-sample depth, not the 79-sample count), one ``RepertoireSpace`` is fit on the pooled clonotype
cloud, and age is regressed off each block with 5-fold CV (Spearman of out-of-fold predictions). A
**batch check** (leave-one-batch-out over the ``A2/A3/A4`` sequencing batches, Prop. ``prop:batch``)
confirms the signal is not a batch artifact.

Data: HF isalgo/airr_benchmark (vdjtools format). Downloaded + cached on first run (needs [bench]).
Run:  python experiments/benchmark_repertoire_aging.py [cap_samples] [downsample_reads]
Full cohort: python experiments/benchmark_repertoire_aging.py 0 50000   (all 79, ~50k reads each)
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _cohort import kmer_matrix, load_cohort, pooled_clonotypes

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding

REPO = "isalgo/airr_benchmark"
META = "vdjtools/metadata_aging.txt"
PREFIX, SUFFIX = "vdjtools/", ".gz"          # metadata file_name "A3-i101.txt" -> vdjtools/A3-i101.txt.gz
N_PROTO, N_COMPONENTS, N_RFF = 1000, 20, 2048
ALPHAS = np.logspace(-2, 4, 13)


def _cv_spearman(X: np.ndarray, y: np.ndarray, *, pca_cols: int = 0, groups=None) -> float:
    """|Spearman| of out-of-fold RidgeCV predictions with leakage-safe in-fold reduction.

    The first ``pca_cols`` columns (the high-dim kernel-mean block) get ``PCA(0.9 var)`` so the
    distribution-regression head doesn't overfit; remaining columns (the few precious diversity
    features) pass through *raw* so their signal isn't buried in the mean block's PCA. All
    reduction is fit inside the CV fold via the pipeline.
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    n = X.shape[1]
    pca = make_pipeline(StandardScaler(), PCA(n_components=0.9, svd_solver="full", random_state=0))
    if pca_cols and pca_cols < n:
        pre = ColumnTransformer([("mean", pca, list(range(pca_cols))),
                                 ("rest", StandardScaler(), list(range(pca_cols, n)))])
    elif pca_cols:                       # PCA the whole (wide) block
        pre = pca
    else:
        pre = StandardScaler()
    model = make_pipeline(pre, RidgeCV(alphas=ALPHAS))
    cv, kw = (LeaveOneGroupOut(), {"groups": groups}) if groups is not None else (5, {})
    return abs(spearmanr(y, cross_val_predict(model, X, y, cv=cv, **kw)).correlation)


def main(cap: int = 20, downsample_to: int = 5000) -> None:
    t0 = time.perf_counter()
    cap = cap or None
    meta, samples = load_cohort(REPO, META, prefix=PREFIX, suffix=SUFFIX,
                                downsample_to=downsample_to, cap_samples=cap)
    ages = np.array([int(r["age"]) for r, _ in samples], dtype=float)
    batch = np.array([r["sample_id"].split("-")[0] for r, _ in samples])   # A2 / A3 / A4
    print(f"{len(samples)} donors, ages {ages.min():.0f}–{ages.max():.0f}, "
          f"batches {sorted(set(batch))}, ≤{downsample_to} reads/donor")

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes(samples),
                                 n_rff=N_RFF, n_components=N_COMPONENTS, seed=0)
    embs = [sample_embedding(space, df, blocks=("mean", "diversity")) for _, df in samples]
    mean = np.stack([e.mean for e in embs])
    div = np.stack([e.diversity for e in embs])
    Phi = np.hstack([mean, div])                       # mean block PCA'd, diversity kept raw
    n_mean = mean.shape[1]

    r = {
        "Phi (mean+div)": _cv_spearman(Phi, ages, pca_cols=n_mean),
        "mean only": _cv_spearman(mean, ages, pca_cols=n_mean),
        "diversity (4)": _cv_spearman(div, ages),
        "¹D only": _cv_spearman(div[:, 1:2], ages),
        "kmer_profile": _cv_spearman(kmer_matrix(samples), ages, pca_cols=10**9),
    }
    r_phi_lobo = _cv_spearman(Phi, ages, pca_cols=n_mean, groups=batch)

    print(f"\n{'features':<18}{'|Spearman(age)|':>16}")
    for k, v in r.items():
        print(f"{k:<18}{v:>16.3f}")
    print(f"{'Phi leave-1-batch':<18}{r_phi_lobo:>16.3f}   (batch-artifact check, prop:batch)")

    best_base = max(r["diversity (4)"], r["¹D only"], r["kmer_profile"])
    verdict = "PASS" if r["Phi (mean+div)"] >= best_base else "FAIL"
    print(f"\n[{verdict}] Φ={r['Phi (mean+div)']:.3f} vs best baseline={best_base:.3f} "
          f"(div={r['diversity (4)']:.3f}, ¹D={r['¹D only']:.3f}, kmer={r['kmer_profile']:.3f}); "
          f"survives batch = {r_phi_lobo:.3f}")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]) if len(args) > 1 else 20, int(args[2]) if len(args) > 2 else 5000)
