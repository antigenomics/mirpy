"""In-silico evolution of the tumour microenvironment — perturb infiltration, read the coupled response.

Because the TME-aware repertoire embedding makes every metric a smooth coordinate (`_tcga_embedding.py`;
`mir.repertoire.sample_descriptor`), the cohort's *joint distribution* of those coordinates is a learnable
**manifold**, and moving a sample along one coordinate (infiltration = hot↔cold) while staying on the manifold
predicts how the *others* respond. We fit a Gaussian manifold over the interpretable metric coordinates
[infiltration, diversity, IgG, IgA, class-switch, T-vs-B, atypicality] per cancer, then:

  1. **coupled metric response** — the conditional slope d(metric)/d(infiltration): as a tumour gets hotter,
     what happens to diversity / class-switching / T-vs-B balance (on the empirical manifold);
  2. **survival response** — fit CoxPH(OS ~ metrics + clinical); move a sample cold→hot *on-manifold* (others
     follow the conditional expectation) and read the predicted risk (HR) change — the in-silico "make this
     tumour hotter" experiment.

This is the substrate for embedding simulation: the coordinates are decodable metrics, the manifold is a
generative model, and perturbation + decode = in-silico evolution.

Data: ``~/hf/airr_tcga``. Needs ``[bench]`` (lifelines). Reuses the cached pan-cancer embedding.
Run:  python experiments/benchmark_repertoire_tcga_insilico.py [cancers|ALL]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from _tcga import clinical_matrix, load_metadata
from _tcga_embedding import build_embedding

CACHE = os.path.join(os.path.dirname(__file__), "..", "tmp", "tcga_emb_cache")
METRICS = ["infiltration", "diversity", "IgG", "IgA", "switch", "T-vs-B", "atypicality"]
MIN_COHORT, MIN_EVENTS = 120, 25


def _metric_matrix(X, ch):
    """The interpretable metric coordinates (z-scored) — infiltration first (the perturbation axis)."""
    return np.column_stack([
        X[:, ch["coverage"]].mean(1),            # infiltration (hot/cold)
        X[:, ch["diversity"]].mean(1),           # diversity
        X[:, ch["isotype"][0]],                  # IgG
        X[:, ch["isotype"][1]],                  # IgA
        X[:, ch["isotype"][3]],                  # class-switch
        X[:, ch["composition"][0]],              # T-vs-B balance
        X[:, ch["atypicality"][0]],              # atypicality
    ])


def _conditional_slope(M):
    """Gaussian manifold: slope of each metric wrt infiltration (col 0) = Σ_{j,0}/Σ_{0,0}."""
    C = np.cov(M, rowvar=False)
    return C[:, 0] / C[0, 0]                      # d E[metric_j | infiltration] / d infiltration


def _survival_response(rows, M):
    """CoxPH(OS ~ metrics + clinical); predicted log-HR moving infiltration cold(−1)→hot(+1) ON-MANIFOLD."""
    import pandas as pd
    from lifelines import CoxPHFitter

    C, names = clinical_matrix(rows)
    df = pd.DataFrame(C, columns=names)
    for j, k in enumerate(METRICS):
        df[k] = M[:, j]
    df["OS"] = [float(r["OS"]) for r in rows]; df["OS_event"] = [float(r["OS_event"]) for r in rows]
    keep = [c for c in df.columns if c in ("OS", "OS_event") or df[c].std() > 1e-9]
    try:
        cph = CoxPHFitter(penalizer=0.1).fit(df[keep], "OS", "OS_event")
    except Exception:
        return float("nan"), float("nan")
    slope = _conditional_slope(M)                # on-manifold direction: infiltration + coupled response
    coef = np.array([cph.params_.get(k, 0.0) for k in METRICS])
    # cold (infil=-1) -> hot (+1): metrics move by 2*slope along the manifold; Δ log-HR = coefᵀ·Δmetric
    dloghr = float(coef @ (2.0 * slope))
    return dloghr, float(cph.params_.get("infiltration", 0.0))


def main(cancers=None):
    t0 = time.perf_counter()
    if cancers is None:
        cancers = ["SKCM", "KIRC", "LGG", "HNSC", "BLCA", "KIRP", "LUAD", "SARC", "STAD", "LIHC"]
    cn = [f"TCGA-{c}" for c in cancers]
    print(f"building / loading TME repertoire embedding over {len(cancers)} cancers …")
    rows, X, ch = build_embedding(cn, cache_dir=CACHE)
    M = _metric_matrix(X, ch)
    print(f"{len(rows)} samples; manifold over {METRICS}\n")

    print("Coupled metric response to rising infiltration (on-manifold slope d·/d-infiltration):")
    print(f"  {'cancer':<7}" + "".join(f"{m[:7]:>9}" for m in METRICS[1:]) + f"{'hot→ ΔlogHR':>12}{'':>4}")
    coup = []
    for c in cancers:
        idx = np.array([i for i, r in enumerate(rows) if r["study_id"] == f"TCGA-{c}"])
        ev = int(sum(rows[i]["OS_event"] == 1.0 for i in idx))
        if len(idx) < MIN_COHORT or ev < MIN_EVENTS:
            continue
        Mi = M[idx]; sub = [rows[i] for i in idx]
        slope = _conditional_slope(Mi)
        dloghr, _ = _survival_response(sub, Mi)
        coup.append((c, slope, dloghr))
        arrow = "protective" if dloghr < -0.05 else ("adverse" if dloghr > 0.05 else "~neutral")
        print(f"  {c:<7}" + "".join(f"{slope[j]:>+9.2f}" for j in range(1, len(METRICS)))
              + f"{dloghr:>+12.2f}  {arrow}")

    if coup:
        S = np.array([s for _, s, _ in coup])
        print(f"\n[pan-cancer manifold] mean coupling to infiltration: "
              + ", ".join(f"{METRICS[j]} {S[:, j].mean():+.2f}" for j in range(1, len(METRICS))))
        hot_protective = sum(d < -0.05 for _, _, d in coup)
        print(f"  in-silico 'make hotter': predicted protective in {hot_protective}/{len(coup)} cancers "
              f"(ΔlogHR<−0.05) — the embedding simulates the hot→cold survival axis.")
    print(f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    if len(a) > 1 and a[1].upper() == "ALL":
        m = load_metadata(require_os=True)
        cancers = [s.replace("TCGA-", "") for s, _ in m["study_id"].value_counts().iter_rows()]
    elif len(a) > 1:
        cancers = a[1].split(",")
    else:
        cancers = None
    main(cancers)
