"""TCGA tumor-infiltrating repertoire embedding — tumor type, stage, and prognosis (survival).

TCGA bulk-RNA-seq carries a tumor-infiltrating immune repertoire across all 7 chains (``airr_tcga``,
9 591 samples, 33 cancer types, overall-survival + clinical). It is **IG-dominant** (~97% IG; TR chains
are ~tens of clonotypes/sample), so a central question is *which chains carry signal*. We embed each chain
separately AND concatenate all chains (per the locked design), and test three readouts:

1. **Tumor type** — does Φ(S) separate the cancer types (macro one-vs-rest AUC) beyond a diversity summary?
2. **Stage** — ordinal cancer_stage (S1–S4) association.
3. **Survival (headline)** — per cancer type, a CoxPH base model on clinical covariates
   (age, sex, stage, log total_reads) vs base+Φ (PCA-reduced); we report the cross-validated **C-index
   gain** — the added prognostic value of the repertoire embedding over clinical covariates alone. This is
   the "coxph of survival residuals minus clinical covariates" test: whether Φ explains survival variation
   the covariates do not.

Data: ``~/hf/airr_tcga`` local checkout, else HF ``isalgo/airr_tcga``. Needs ``[bench]`` (incl. lifelines).
Run:  python experiments/benchmark_repertoire_tcga.py [chains] [cancers] [downsample] [cap]
      e.g. python experiments/benchmark_repertoire_tcga.py TRB,IGH BRCA,LUAD,KIRC 50000 0
Validate on a few large cohorts first (scaling discipline); full 7-chain × 33-type is a documented sweep.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _cohort import pooled_clonotypes
from _tcga import CHAINS, clinical_matrix, load_tcga

from mir.embedding.tcremp import TCREmp
from mir.repertoire import fit_repertoire_space, sample_embedding

N_PROTO, N_COMPONENTS, N_RFF, N_RFF_SECOND, N_PC = 1000, 20, 2048, 128, 10
MIN_COHORT = 120                                     # min samples to fit a per-cancer-type Cox model


def embed_chain(chain: str, cancers, downsample_to: int | None):
    """Fit ONE RepertoireSpace for this chain, embed every sample. Returns {sample_id: (row, Φ vector)}."""
    rows, frames = load_tcga(chain, cancers, downsample_to=downsample_to, min_clonotypes=5)
    if len(frames) < MIN_COHORT:
        return {}
    model = TCREmp.from_defaults("human", chain, n_prototypes=N_PROTO)
    space = fit_repertoire_space(model, pooled_clonotypes([(None, f) for f in frames]),
                                 n_rff=N_RFF, n_rff_second=N_RFF_SECOND, n_components=N_COMPONENTS, seed=0)
    out = {}
    for r, f in zip(rows, frames):
        e = sample_embedding(space, f, blocks=("mean", "diversity", "second"))
        out[r["sample_id"]] = (r, np.concatenate([e.mean, e.diversity, e.second]))
    return out


def tumortype_auc(Phi, labels) -> float:
    """Macro one-vs-rest AUC of a PCA+logistic multiclass head (5-fold CV)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    y = LabelEncoder().fit_transform(labels)
    if len(set(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), PCA(n_components=min(50, Phi.shape[1]), random_state=0),
                        LogisticRegression(max_iter=2000, C=1.0))
    proba = cross_val_predict(clf, Phi, y, method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
    if len(set(y)) == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def stage_spearman(Phi, rows) -> float:
    """|Spearman| of a PCA+ridge stage predictor vs ordinal cancer_stage (5-fold CV)."""
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from _tcga import _STAGE
    y = np.array([_STAGE.get(r["cancer_stage"], np.nan) for r in rows], dtype=float)
    m = np.isfinite(y)
    if m.sum() < 50 or len(set(y[m])) < 2:
        return float("nan")
    reg = make_pipeline(StandardScaler(), PCA(n_components=min(30, Phi.shape[1]), random_state=0),
                        RidgeCV(alphas=np.logspace(-2, 4, 13)))
    pred = cross_val_predict(reg, Phi[m], y[m], cv=5)
    return abs(spearmanr(y[m], pred).correlation)


def cox_cv_cindex(rows, Phi=None, *, n_pc: int = N_PC, n_splits: int = 5, seed: int = 0):
    """Cross-validated Cox C-index on clinical covariates, optionally + PCA-reduced Φ. Returns (mean, std).

    In-fold PCA on Φ (unsupervised — never sees survival) keeps the added-value test leakage-safe.
    """
    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    X, names = clinical_matrix(rows)
    os_ = np.array([float(r["OS"]) for r in rows])
    ev = np.array([float(r["OS_event"]) for r in rows])
    base = pd.DataFrame(X, columns=names)
    base["OS"], base["OS_event"] = os_, ev

    scores = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(base):
        dtr, dte = base.iloc[tr].copy(), base.iloc[te].copy()
        if Phi is not None:
            sc = StandardScaler().fit(Phi[tr])
            pca = PCA(n_components=min(n_pc, len(tr) - 1, Phi.shape[1]), random_state=0).fit(sc.transform(Phi[tr]))
            Ptr, Pte = pca.transform(sc.transform(Phi[tr])), pca.transform(sc.transform(Phi[te]))
            for j in range(Ptr.shape[1]):
                dtr[f"pc{j}"], dte[f"pc{j}"] = Ptr[:, j], Pte[:, j]
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(dtr, "OS", "OS_event")
            risk = cph.predict_partial_hazard(dte)
            scores.append(concordance_index(dte["OS"], -risk, dte["OS_event"]))
        except Exception:
            scores.append(float("nan"))
    s = np.array(scores, dtype=float)
    return float(np.nanmean(s)), float(np.nanstd(s))


def analyze(name: str, embed: dict, cancers) -> None:
    """Run the three readouts for one representation (a single chain, or the concat)."""
    rows = [v[0] for v in embed.values()]
    Phi = np.stack([v[1] for v in embed.values()])
    labels = np.array([r["study_id"] for r in rows])
    tt = tumortype_auc(Phi, labels) if len(cancers) > 1 else float("nan")
    st = stage_spearman(Phi, rows)
    print(f"\n[{name}]  {len(rows)} samples, dim {Phi.shape[1]}")
    print(f"  tumor-type macro-OvR AUC : {tt:.3f}   stage |Spearman| : {st:.3f}")
    print(f"  {'cancer':<12}{'n(OS)':>7}{'C base':>9}{'C base+Φ':>11}{'ΔC':>8}")
    for ct in cancers:
        idx = [i for i, r in enumerate(rows)
               if r["study_id"] == ct and r["OS"] is not None and float(r["OS"]) > 0
               and r["OS_event"] in (0.0, 1.0)]
        if len(idx) < MIN_COHORT:
            continue
        sub = [rows[i] for i in idx]
        P = Phi[idx]
        cb, _ = cox_cv_cindex(sub)
        cf, _ = cox_cv_cindex(sub, P)
        print(f"  {ct.replace('TCGA-', ''):<12}{len(idx):>7}{cb:>9.3f}{cf:>11.3f}{cf - cb:>+8.3f}")


def main(chains=("TRB", "IGH"), cancers=("TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC"),
         downsample_to: int | None = 50_000, cap: int | None = None) -> None:
    t0 = time.perf_counter()
    print(f"chains={list(chains)} cancers={[c.replace('TCGA-', '') for c in cancers]} "
          f"≤{downsample_to} reads/sample\n")

    embeds = {}
    for chain in chains:
        e = embed_chain(chain, cancers, downsample_to)
        print(f"{chain}: {len(e)} samples embedded")
        if e:
            embeds[chain] = e

    for chain, e in embeds.items():
        analyze(chain, e, cancers)

    # all-chain concat: samples present in EVERY embedded chain, Φ hstacked
    if len(embeds) > 1:
        common = set.intersection(*[set(e) for e in embeds.values()])
        first = next(iter(embeds))
        concat = {sid: (embeds[first][sid][0],
                        np.concatenate([embeds[c][sid][1] for c in embeds]))
                  for sid in common}
        if len(concat) >= MIN_COHORT:
            analyze(f"concat[{'+'.join(embeds)}]", concat, cancers)

    print(f"\n[read] ΔC>0 = the embedding adds prognostic value over clinical covariates (age/sex/stage/depth); "
          f"contrast chains (IG deep vs TR shallow) and concat. Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    ch = tuple(a[1].split(",")) if len(a) > 1 else ("TRB", "IGH")
    cn = tuple(f"TCGA-{c}" if not c.startswith("TCGA-") else c for c in a[2].split(",")) if len(a) > 2 \
        else ("TCGA-BRCA", "TCGA-LUAD", "TCGA-KIRC")
    ds = (int(a[3]) or None) if len(a) > 3 else 50_000
    cp = (int(a[4]) or None) if len(a) > 4 else None
    main(ch, cn, ds, cp)
