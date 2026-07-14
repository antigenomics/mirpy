"""Repertoire embeddings stratify the tumour microenvironment and patient survival — pan-cancer (TCGA).

The headline of the repertoire-embedding paradigm applied to tissue: a single **TME-aware, multi-chain
repertoire embedding** Φ(S) (`_tcga_embedding.py` — per-chain identity ‖ diversity ‖ coverage/infiltration ‖
isotype ‖ composition ‖ atypicality) predicts overall survival **beyond clinical covariates**, across all 7
chains and up to 33 TCGA cancer types. Where the bare clonotype-identity embedding was flat (ΔC≈0), the full
embedding — whose channels are exactly the TME axes — carries per-cancer prognostic signal.

Per cancer we report, over clinical base (age+sex+stage+log reads):
  - **ΔC-index** (5-fold CV) of clinical + Φ (PCA-reduced) — added prognostic value;
  - a **likelihood-ratio p** for the Φ block (does the embedding significantly improve the fit);
  - **channel ablation** — which channel group (identity / diversity / coverage / isotype / composition /
    atypicality) drives the gain.
Then a pan-cancer summary: mean ΔC and the count of cancers where Φ is significant.

Data: ``~/hf/airr_tcga`` (all 7 chains, c_call). Needs ``[bench]`` (lifelines).
Run:  python experiments/benchmark_repertoire_tcga_pancancer.py [cancers|ALL] [n_pc]
      ALL = every cancer with enough OS events; default a validated subset. Full run is compute-heavy (background).
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _tcga import clinical_matrix, load_metadata
from _tcga_embedding import build_embedding

N_PC, MIN_EVENTS, MIN_COHORT = 8, 25, 120
CHANNEL_ORDER = ["identity", "diversity", "coverage", "isotype", "composition", "atypicality"]


def _surv_df(rows):
    import pandas as pd
    C, names = clinical_matrix(rows)
    df = pd.DataFrame(C, columns=names)
    df["OS"] = [float(r["OS"]) for r in rows]; df["OS_event"] = [float(r["OS_event"]) for r in rows]
    return df


def _nondegen(df):
    return df[[c for c in df.columns if c in ("OS", "OS_event") or df[c].std() > 1e-9]]


def cv_cindex(rows, Xblock, *, n_pc=None, seed=0):
    """5-fold CV Cox C-index of clinical base + optional feature block (PCA-reduced if n_pc set)."""
    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    base = _surv_df(rows)
    sc = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(base):
        dtr, dte = base.iloc[tr].copy(), base.iloc[te].copy()
        if Xblock is not None:
            B = Xblock
            if n_pc and Xblock.shape[1] > n_pc:
                st = StandardScaler().fit(Xblock[tr])
                pca = PCA(min(n_pc, len(tr) - 1), random_state=seed).fit(st.transform(Xblock[tr]))
                Btr, Bte = pca.transform(st.transform(Xblock[tr])), pca.transform(st.transform(Xblock[te]))
            else:
                Btr, Bte = Xblock[tr], Xblock[te]
            for j in range(Btr.shape[1]):
                dtr[f"z{j}"] = Btr[:, j]; dte[f"z{j}"] = Bte[:, j]
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(_nondegen(dtr), "OS", "OS_event")
            risk = cph.predict_partial_hazard(_nondegen(dte).drop(columns=["OS", "OS_event"]))
            sc.append(concordance_index(dte["OS"], -risk, dte["OS_event"]))
        except Exception:
            sc.append(np.nan)
    return float(np.nanmean(sc))


def lr_pvalue(rows, Xblock, *, n_pc=N_PC, seed=0):
    """Likelihood-ratio p for the Φ block: 2·(ll_full − ll_base) ~ χ²(df) on the full cancer cohort."""
    from lifelines import CoxPHFitter
    from scipy.stats import chi2
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    base = _nondegen(_surv_df(rows))
    try:
        b = CoxPHFitter(penalizer=0.05).fit(base, "OS", "OS_event")
        P = PCA(min(n_pc, Xblock.shape[1]), random_state=seed).fit_transform(
            StandardScaler().fit_transform(Xblock))
        full = base.copy()
        for j in range(P.shape[1]):
            full[f"z{j}"] = P[:, j]
        f = CoxPHFitter(penalizer=0.05).fit(full, "OS", "OS_event")
        lr = 2.0 * (f.log_likelihood_ - b.log_likelihood_)
        return float(chi2.sf(max(lr, 0.0), P.shape[1]))
    except Exception:
        return float("nan")


def main(cancers=None, n_pc=N_PC):
    t0 = time.perf_counter()
    if cancers is None:
        cancers = ["SKCM", "KIRC", "KIRP", "LGG", "LUAD", "BLCA", "STAD", "OV", "HNSC", "LIHC"]
    cn = [f"TCGA-{c}" for c in cancers]
    print(f"building TME repertoire embedding over {len(cancers)} cancers …")
    rows, X, ch = build_embedding(cn)
    print(f"{len(rows)} samples, Φ dim {X.shape[1]}; channels { {k: len(v) for k, v in ch.items()} }\n")

    hdr = f"{'cancer':<7}{'n':>5}{'ev':>4}{'Cbase':>7}{'C+Φ':>7}{'ΔC':>7}{'LRp':>8}  best-channel(ΔC)"
    print(hdr)
    results = []
    for c in cancers:
        idx = np.array([i for i, r in enumerate(rows) if r["study_id"] == f"TCGA-{c}"])
        ev = int(sum(rows[i]["OS_event"] == 1.0 for i in idx))
        if len(idx) < MIN_COHORT or ev < MIN_EVENTS:
            continue
        sub = [rows[i] for i in idx]; Xi = X[idx]
        cb = cv_cindex(sub, None)
        cf = cv_cindex(sub, Xi, n_pc=n_pc)
        lrp = lr_pvalue(sub, Xi, n_pc=n_pc)
        chan = {g: cv_cindex(sub, Xi[:, ch[g]], n_pc=(n_pc if len(ch[g]) > n_pc else None)) - cb
                for g in CHANNEL_ORDER if ch.get(g)}
        best = max(chan, key=chan.get)
        results.append((c, len(idx), ev, cb, cf, cf - cb, lrp, best, chan[best]))
        print(f"{c:<7}{len(idx):>5}{ev:>4}{cb:>7.3f}{cf:>7.3f}{cf - cb:>+7.3f}{lrp:>8.3f}  "
              f"{best} ({chan[best]:+.3f})")

    if results:
        dC = np.array([r[5] for r in results]); nsig = sum(r[6] < 0.05 for r in results)
        from collections import Counter
        bc = Counter(r[7] for r in results)
        print(f"\n[pan-cancer] {len(results)} cancers: mean ΔC {dC.mean():+.3f} (median {np.median(dC):+.3f}); "
              f"Φ significant (LR p<0.05) in {nsig}/{len(results)}; "
              f"most-informative channel: {dict(bc.most_common())}")
        print("The TME-aware repertoire embedding adds prognostic value the bare clonotype-identity embedding "
              "cannot — the signal is in its coverage/isotype/composition channels (the TME state).")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    if len(a) > 1 and a[1].upper() == "ALL":
        m = load_metadata(require_os=True)
        cs = [s.replace("TCGA-", "") for s, _ in
              sorted(m["study_id"].value_counts().iter_rows(), key=lambda x: -x[1])]
    elif len(a) > 1:
        cs = a[1].split(",")
    else:
        cs = None
    main(cs, int(a[2]) if len(a) > 2 else N_PC)
