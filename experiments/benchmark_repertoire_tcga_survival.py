"""TCGA survival from biologically-grounded AIRR features — where the raw clonotype embedding failed.

The clonotype embedding added no prognostic value over clinical covariates (benchmark_repertoire_tcga.py,
ΔC≈0). The tissue-repertoire prognosis lives instead in interpretable, biology-grounded axes (from the
internal BostonGene AIRR-tissue EDA): **isotype** (IgG/IgA class-switching — mucosal / plasma-cell
responses), **infiltration** magnitude / hot-vs-cold (receptor read load, T-vs-B balance), **atypicality**
(gene-usage divergence from what is typical for the tumour type), and clonal expansion. We test whether these
add C-index over a clinical Cox (age+sex+stage+log reads), **per tumour type**, and report the direction
(HR) of each feature against known biology.

Known-biology anchors (source EDA): KIRP/UVM/CESC = **IgA / mucosal**; STAD, SKCM = **typical→better**;
OV, LGG = **atypical→better**; KIRC/LUAD/LUSC/BLCA carry ICI-validation cohorts.

Data: ``~/hf/airr_tcga`` (all 7 chains, c_call isotype). Needs ``[bench]`` (lifelines).
Run:  python experiments/benchmark_repertoire_tcga_survival.py [cancers] [cap]
      e.g. python experiments/benchmark_repertoire_tcga_survival.py KIRC,LUAD,SKCM,KIRP,STAD,OV 0
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _tcga import _samples_dir, clinical_matrix, load_metadata
from _tcga_features import atypicality, sample_airr_features

FEATS = ["infiltration", "infiltration_frac", "tb_balance", "igg_frac", "iga_frac", "igm_frac",
         "switch_frac", "clonality", "top_clone", "atypicality"]
GROUPS = {
    "isotype": ["igg_frac", "iga_frac", "switch_frac"],
    "infiltration": ["infiltration", "tb_balance"],
    "atypicality": ["atypicality"],
    "clonality": ["clonality", "top_clone"],
    "all-AIRR": ["infiltration", "tb_balance", "igg_frac", "iga_frac", "switch_frac",
                 "clonality", "top_clone", "atypicality"],
}
MIN_EVENTS, MIN_COHORT = 25, 120


def load_features(cancers, cap=None):
    """Per-sample AIRR feature matrix (z-scored) + rows, for samples with usable OS."""
    rows = load_metadata(cancers, require_os=True).to_dicts()
    if cap:
        rows = rows[:cap]
    sd = _samples_dir()
    keep, scal, vus = [], [], []
    import os
    for r in rows:
        p = f"{sd}/{r['sample_id']}.tsv"
        if not os.path.exists(p):
            continue
        s, v = sample_airr_features(p, float(r["total_reads"]) if r["total_reads"] else None)
        keep.append(r); scal.append(s); vus.append(v)
    aty = atypicality(vus, [r["study_id"] for r in keep])
    X = np.array([[s[k] for k in FEATS[:-1]] for s in scal], dtype=float)
    X = np.column_stack([X, aty])                                   # append atypicality as last col
    # median-impute + z-score each column (interpretable, comparable coefficients)
    for j in range(X.shape[1]):
        col = X[:, j]; col[~np.isfinite(col)] = np.nanmedian(col[np.isfinite(col)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return keep, X


def _base_df(rows):
    import pandas as pd
    C, names = clinical_matrix(rows)
    df = pd.DataFrame(C, columns=names)
    df["OS"] = [float(r["OS"]) for r in rows]
    df["OS_event"] = [float(r["OS_event"]) for r in rows]
    return df, names


def _drop_degenerate(df):
    """Drop constant covariates (e.g. sex in an all-female cohort) — lifelines can't fit them."""
    keep = [c for c in df.columns if c in ("OS", "OS_event") or df[c].std() > 1e-9]
    return df[keep]


def cv_cindex(rows, X, cols, *, n_splits=5, seed=0):
    """5-fold CV Cox C-index on clinical base + the given feature columns (None = base only)."""
    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.model_selection import KFold

    base, names = _base_df(rows)
    full = base.copy()
    if cols:
        for j, c in cols:
            full[c] = X[:, j]
    full = _drop_degenerate(full)
    sc = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(full):
        try:
            cph = CoxPHFitter(penalizer=0.1).fit(full.iloc[tr], "OS", "OS_event")
            risk = cph.predict_partial_hazard(full.iloc[te])
            sc.append(concordance_index(full.iloc[te]["OS"], -risk, full.iloc[te]["OS_event"]))
        except Exception:
            sc.append(np.nan)
    return float(np.nanmean(sc))


def full_cox_dirs(rows, X):
    """Full-data penalized Cox on base + all AIRR features; return {feature: (HR, p)} for AIRR terms."""
    from lifelines import CoxPHFitter

    base, _ = _base_df(rows)
    for j, c in enumerate(FEATS):
        base[c] = X[:, j]
    base = _drop_degenerate(base)
    try:
        cph = CoxPHFitter(penalizer=0.1).fit(base, "OS", "OS_event")
        s = cph.summary
        return {c: (float(np.exp(s.loc[c, "coef"])), float(s.loc[c, "p"])) for c in FEATS if c in s.index}
    except Exception:
        return {}


def logrank_p(rows, x) -> float:
    """Median-split log-rank p on a single raw AIRR feature (the EDA's KM-stratification test)."""
    from lifelines.statistics import logrank_test
    os_ = np.array([float(r["OS"]) for r in rows]); ev = np.array([float(r["OS_event"]) for r in rows])
    hi = x > np.median(x)
    if hi.sum() < 10 or (~hi).sum() < 10:
        return float("nan")
    r = logrank_test(os_[hi], os_[~hi], ev[hi], ev[~hi])
    return float(r.p_value)


def main(cancers=("KIRC", "LUAD", "SKCM", "KIRP", "STAD", "OV"), cap=None):
    t0 = time.perf_counter()
    cn = tuple(f"TCGA-{c}" for c in cancers)
    rows, X = load_features(cn, cap)
    print(f"{len(rows)} samples across {list(cancers)} with usable OS; {len(FEATS)} AIRR features\n")

    print(f"{'cancer':<8}{'n':>5}{'ev':>5}{'C base':>8}" + "".join(f"{g:>13}" for g in GROUPS))
    for c in cancers:
        idx = [i for i, r in enumerate(rows) if r["study_id"] == f"TCGA-{c}"]
        ev = sum(rows[i]["OS_event"] == 1.0 for i in idx)
        if len(idx) < MIN_COHORT or ev < MIN_EVENTS:
            continue
        sub = [rows[i] for i in idx]
        Xi = X[idx]
        cb = cv_cindex(sub, Xi, None)
        gains = {g: cv_cindex(sub, Xi, [(FEATS.index(f), f) for f in cols]) - cb for g, cols in GROUPS.items()}
        print(f"{c:<8}{len(idx):>5}{ev:>5}{cb:>8.3f}" + "".join(f"{gains[g]:>+13.3f}" for g in GROUPS))

    print("\nDirection of AIRR features per cancer (full-data Cox; HR>1 worse, HR<1 protective; * p<0.05):")
    for c in cancers:
        idx = [i for i, r in enumerate(rows) if r["study_id"] == f"TCGA-{c}"]
        ev = sum(rows[i]["OS_event"] == 1.0 for i in idx)
        if len(idx) < MIN_COHORT or ev < MIN_EVENTS:
            continue
        dirs = full_cox_dirs([rows[i] for i in idx], X[idx])
        sig = [f"{f} HR={hr:.2f}{'*' if p < 0.05 else ''}" for f, (hr, p) in dirs.items() if p < 0.1]
        print(f"  {c:<6}: {', '.join(sig) if sig else '(no AIRR term p<0.1)'}")

    print("\nKM stratification (median-split log-rank p; the EDA's test — surfaces threshold effects "
          "isotype/atypicality show but linear C-index misses):")
    print(f"  {'cancer':<8}{'infiltration':>14}{'IgA frac':>11}{'atypicality':>13}")
    for c in cancers:
        idx = [i for i, r in enumerate(rows) if r["study_id"] == f"TCGA-{c}"]
        ev = sum(rows[i]["OS_event"] == 1.0 for i in idx)
        if len(idx) < MIN_COHORT or ev < MIN_EVENTS:
            continue
        sub = [rows[i] for i in idx]; Xi = X[idx]
        p_inf = logrank_p(sub, Xi[:, FEATS.index("infiltration")])
        p_iga = logrank_p(sub, Xi[:, FEATS.index("iga_frac")])
        p_aty = logrank_p(sub, Xi[:, FEATS.index("atypicality")])
        mark = lambda p: f"{p:.3f}{'*' if p < 0.05 else ''}"
        print(f"  {c:<8}{mark(p_inf):>14}{mark(p_iga):>11}{mark(p_aty):>13}")

    print(f"\n[read] ΔC>0 for a group = that biology axis adds prognostic value over clinical covariates. "
          f"Contrast with the embedding's flat ΔC. Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    cs = tuple(a[1].split(",")) if len(a) > 1 else ("KIRC", "LUAD", "SKCM", "KIRP", "STAD", "OV")
    cp = (int(a[2]) or None) if len(a) > 2 else None
    main(cs, cp)
