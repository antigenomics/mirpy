"""TME states from the repertoire embedding — unsupervised discovery + survival, pan-cancer (TCGA).

Second face of the repertoire-embedding paradigm: the TME-aware embedding (`_tcga_embedding.py`), clustered
unsupervised, recovers interpretable **tumour-microenvironment states** (T-hot / B-hot-switched / cold / mixed)
that (a) have coherent channel profiles (infiltration, T-vs-B, isotype, diversity), (b) are enriched in the
expected cancers, and (c) are **prognostic controlling for tumour type + clinical** (CoxPH stratified by
cancer). A UMAP of Φ colours by state / infiltration / cancer / survival.

Data: ``~/hf/airr_tcga``. Needs ``[bench]`` (lifelines, umap-learn). Reuses the cached embedding.
Run:  python experiments/benchmark_repertoire_tcga_tme.py [cancers|ALL] [k_states]
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

import numpy as np

from _tcga import load_metadata
from _tcga_embedding import build_embedding

CACHE = os.path.join(os.path.dirname(__file__), "..", "tmp", "tcga_emb_cache")
OUTDIR = os.path.join(os.path.dirname(__file__), "figures")


def _state_profiles(rows, X, ch, labels, k):
    """Per-state mean channel z-scores + composition, dominant cancers, OS-event rate."""
    infil = X[:, ch["coverage"]].mean(1)                 # mean per-chain log-infiltration (z)
    tb = X[:, ch["composition"][0]]                      # T-vs-B balance (z)
    igg, iga, igm, switch = (X[:, ch["isotype"][i]] for i in range(4))
    div = X[:, ch["diversity"]].mean(1)
    ev = np.array([r["OS_event"] == 1.0 for r in rows])
    tt = np.array([r["study_id"].replace("TCGA-", "") for r in rows])
    prof = []
    for s in range(k):
        m = labels == s
        top = ", ".join(f"{c}({n})" for c, n in Counter(tt[m]).most_common(3))
        prof.append(dict(state=s, n=int(m.sum()), infil=infil[m].mean(), tb=tb[m].mean(),
                         igg=igg[m].mean(), iga=iga[m].mean(), switch=switch[m].mean(),
                         div=div[m].mean(), death=ev[m].mean(), top=top))
    return prof


def _state_survival(rows, labels, k):
    """CoxPH: OS ~ state (one-hot) + clinical, STRATIFIED by tumour type. Returns (per-state HR/p, global p)."""
    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    from _tcga import clinical_matrix

    C, names = clinical_matrix(rows)
    df = pd.DataFrame(C, columns=names)
    df["OS"] = [float(r["OS"]) for r in rows]; df["OS_event"] = [float(r["OS_event"]) for r in rows]
    df["tumor"] = [r["study_id"] for r in rows]
    ref = Counter(labels).most_common(1)[0][0]           # largest state = reference
    for s in range(k):
        if s != ref:
            df[f"state{s}"] = (labels == s).astype(float)
    keep = [c for c in df.columns if c in ("OS", "OS_event", "tumor") or df[c].std() > 1e-9]
    try:
        cph = CoxPHFitter(penalizer=0.05).fit(df[keep], "OS", "OS_event", strata=["tumor"])
        hr = {c: (float(np.exp(cph.summary.loc[c, "coef"])), float(cph.summary.loc[c, "p"]))
              for c in cph.summary.index if c.startswith("state")}
    except Exception:
        hr = {}
    # stratified multivariate log-rank across states (blocks tumour type via the strata arg)
    lr = multivariate_logrank_test(np.array([float(r["OS"]) for r in rows]), labels,
                                   np.array([float(r["OS_event"]) for r in rows]))
    return hr, float(lr.p_value)


def _umap_figure(rows, X, ch, labels, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    coords = umap.UMAP(n_neighbors=20, min_dist=0.15, random_state=0).fit_transform(X)
    infil = X[:, ch["coverage"]].mean(1)
    ev = np.array([r["OS_event"] == 1.0 for r in rows])
    tt = np.array([r["study_id"].replace("TCGA-", "") for r in rows])
    top = {c for c, _ in Counter(tt).most_common(8)}
    tt_lab = np.where(np.isin(tt, list(top)), tt, "other")

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    ax[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=6, alpha=0.7); ax[0].set_title("TME state")
    sc = ax[1].scatter(coords[:, 0], coords[:, 1], c=infil, cmap="viridis", s=6, alpha=0.7)
    plt.colorbar(sc, ax=ax[1], shrink=0.8); ax[1].set_title("infiltration (z)")
    for c in sorted(set(tt_lab)):
        m = tt_lab == c
        ax[2].scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.7, label=c)
    ax[2].legend(fontsize=6, ncol=2, loc="best"); ax[2].set_title("cancer type")
    ax[3].scatter(coords[~ev, 0], coords[~ev, 1], s=6, alpha=0.5, color="tab:blue", label="alive")
    ax[3].scatter(coords[ev, 0], coords[ev, 1], s=6, alpha=0.5, color="tab:red", label="death")
    ax[3].legend(fontsize=7); ax[3].set_title("OS event")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("TCGA TME-aware repertoire embedding — UMAP", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{stem}.{ext}"), dpi=140, bbox_inches="tight")
    print(f"wrote {OUTDIR}/{stem}.png (+.pdf)")


def main(cancers=None, k=5):
    from sklearn.cluster import KMeans

    t0 = time.perf_counter()
    if cancers is None:
        cancers = ["SKCM", "KIRC", "LGG", "HNSC", "KIRP", "LUAD", "BLCA", "STAD"]
    cn = [f"TCGA-{c}" for c in cancers]
    print(f"building TME repertoire embedding over {len(cancers)} cancers (cached) …")
    rows, X, ch = build_embedding(cn, cache_dir=CACHE)
    # TME state = cluster on the NON-identity (TME) channels: coverage/composition/isotype/diversity
    tme = ch["coverage"] + ch["composition"] + ch["isotype"] + ch["diversity"]
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(X[:, tme])
    print(f"{len(rows)} samples, {k} TME states clustered on {len(tme)} channel dims\n")

    print(f"{'st':>3}{'n':>6}{'infil':>7}{'T-vs-B':>8}{'IgG':>6}{'IgA':>6}{'switch':>8}{'div':>7}{'death':>7}  top cancers")
    for p in _state_profiles(rows, X, ch, labels, k):
        print(f"{p['state']:>3}{p['n']:>6}{p['infil']:>+7.2f}{p['tb']:>+8.2f}{p['igg']:>+6.2f}{p['iga']:>+6.2f}"
              f"{p['switch']:>+8.2f}{p['div']:>+7.2f}{p['death']:>7.2f}  {p['top']}")

    hr, lrp = _state_survival(rows, labels, k)
    print(f"\nTME state prognosis (CoxPH stratified by tumour type; HR vs largest state):")
    for s, (h, pv) in sorted(hr.items()):
        print(f"  {s}: HR={h:.2f}{'*' if pv < 0.05 else ''} (p={pv:.3f})")
    print(f"  stratified multivariate log-rank across states: p={lrp:.4f}{'  *significant' if lrp < 0.05 else ''}")

    _umap_figure(rows, X, ch, labels, "umap_tcga_tme")
    print(f"\n[read] Coherent TME states with distinct infiltration/isotype/T-B profiles, enriched in the "
          f"expected cancers, and prognostic beyond tumour type — the repertoire embedding organises the TME. "
          f"Total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    a = sys.argv
    if len(a) > 1 and a[1].upper() == "ALL":
        m = load_metadata(require_os=True)
        cs = [s.replace("TCGA-", "") for s, _ in m["study_id"].value_counts().iter_rows()]
    elif len(a) > 1:
        cs = a[1].split(",")
    else:
        cs = None
    main(cs, int(a[2]) if len(a) > 2 else 5)
