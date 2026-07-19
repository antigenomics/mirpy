"""Scorers for the explainable cohort readout (:mod:`mir.explain`) — the closures you hand to
:func:`mir.explain.channel_report`.

``mir.explain`` is deliberately scorer-agnostic: it slices channels and never sees ``y``, so the
*model choice* lives here, in the caller's closure. This module collects the recurring choices —
cross-validated classification AUC and Cox survival C-index — as small, reusable functions so every
cohort benchmark stops re-gluing them. A scorer takes a feature block and returns a float where
**higher is better**::

    from mir.bench.eval import cv_auc, cv_cindex
    from mir.explain import stack_embeddings, channel_report

    X, spec = stack_embeddings(embs)
    rep = channel_report(X, spec, lambda B: cv_auc(B, y)[0], base=0.5)              # classification
    rep = channel_report(X, spec, lambda B: cv_cindex(dur, evt, base=C, block=B, n_pc=8),
                         base=cv_cindex(dur, evt, base=C, block=None), mode="both")  # survival

Needs the ``[bench]`` extra (scikit-learn always; lifelines for the survival scorers; vdjtools for
the k-mer baseline). Everything is lazily imported so importing this module stays cheap.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- classification


def held_out_auc(Xtr, ytr, Xte, yte, *, pca_cols: int = 0) -> float:
    """Held-out ROC-AUC of a logistic head; the first ``pca_cols`` columns get in-fold PCA(0.9 var).

    Wide blocks (kernel mean, second moment, k-mer) overfit unless reduced *inside* the fold; the few
    diversity/coverage features pass through raw so their signal isn't buried. ``pca_cols=0`` = no PCA.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr = Xtr.reshape(-1, 1) if Xtr.ndim == 1 else Xtr
    Xte = Xte.reshape(-1, 1) if Xte.ndim == 1 else Xte
    pca = make_pipeline(StandardScaler(), PCA(n_components=0.9, svd_solver="full", random_state=0))
    n = Xtr.shape[1]
    if pca_cols and pca_cols < n:
        pre = ColumnTransformer([("m", pca, list(range(pca_cols))),
                                 ("r", StandardScaler(), list(range(pca_cols, n)))])
    elif pca_cols:
        pre = pca
    else:
        pre = StandardScaler()
    clf = make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0)).fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def cv_auc(X, y, *, pca_cols: int = 0, n_splits: int = 5, n_repeats: int = 10, seed: int = 0):
    """Repeated stratified k-fold AUC as ``(mean, std)`` — a CI, not a single-split point estimate.

    A single 70/30 split at small ``n`` has AUC SD ≈ 0.5/√n_test (~0.1 for n_test≈30), so point
    estimates are near-meaningless; repeated CV exposes whether two methods' intervals separate.
    Pass ``cv_auc(B, y)[0]`` as the ``channel_report`` scorer.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    X = X.reshape(-1, 1) if X.ndim == 1 else X
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs = [held_out_auc(X[tr], y[tr], X[te], y[te], pca_cols=pca_cols) for tr, te in rskf.split(X, y)]
    return float(np.mean(aucs)), float(np.std(aucs))


# --------------------------------------------------------------------- survival


def _design(base, block, rows):
    """Assemble a pandas design frame from a base covariate matrix and a feature block."""
    import pandas as pd

    cols = {}
    if base is not None:
        base = np.asarray(base, dtype=np.float64)
        base = base.reshape(-1, 1) if base.ndim == 1 else base
        for j in range(base.shape[1]):
            cols[f"c{j}"] = base[:, j]
    if block is not None:
        block = np.asarray(block, dtype=np.float64)
        block = block.reshape(-1, 1) if block.ndim == 1 else block
        for j in range(block.shape[1]):
            cols[f"z{j}"] = block[:, j]
    return pd.DataFrame(cols, index=np.arange(rows))


def _nondegen(df):
    """Drop zero-variance columns (excluding the survival targets) so Cox converges."""
    keep = [c for c in df.columns if c in ("_T", "_E") or df[c].std() > 1e-9]
    return df[keep]


def cv_cindex(durations, events, *, base=None, block=None, n_pc: int = 0,
              n_splits: int = 5, seed: int = 0, penalizer: float = 0.1) -> float:
    """5-fold CV Cox C-index of ``base`` covariates + an optional feature ``block``.

    The survival analog of :func:`cv_auc`, and the scorer for survival channel reports. ``base`` is
    the clinical design (e.g. age+sex+stage+log-reads — whatever the study built; this function is
    schema-agnostic) and ``block`` is the feature block being scored; when ``block`` is wider than
    ``n_pc`` it is PCA-reduced to ``n_pc`` components **inside each fold** (train-only fit) so a wide
    kernel-mean block doesn't overfit. Returns the mean concordance across folds (nan-robust).

    Use as ``lambda B: cv_cindex(dur, evt, base=C, block=B, n_pc=8)`` with base score
    ``cv_cindex(dur, evt, base=C, block=None)``.
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    dur = np.asarray(durations, dtype=np.float64)
    evt = np.asarray(events, dtype=np.float64)
    n = dur.shape[0]
    if base is None and block is None:
        return float("nan")                       # a Cox with no covariates is undefined

    block = None if block is None else (
        np.asarray(block).reshape(-1, 1) if np.asarray(block).ndim == 1 else np.asarray(block))
    sc = []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n)):
        Btr = Bte = None
        if block is not None:
            if n_pc and block.shape[1] > n_pc:
                st = StandardScaler().fit(block[tr])
                pca = PCA(min(n_pc, len(tr) - 1), random_state=seed).fit(st.transform(block[tr]))
                Btr, Bte = pca.transform(st.transform(block[tr])), pca.transform(st.transform(block[te]))
            else:
                Btr, Bte = block[tr], block[te]
        dtr = _design(None if base is None else np.asarray(base)[tr], Btr, len(tr))
        dte = _design(None if base is None else np.asarray(base)[te], Bte, len(te))
        dtr["_T"], dtr["_E"] = dur[tr], evt[tr]
        dtr = _nondegen(dtr)
        dte = dte[[c for c in dtr.columns if c not in ("_T", "_E")]]
        try:
            cph = CoxPHFitter(penalizer=penalizer).fit(dtr, "_T", "_E")
            risk = cph.predict_partial_hazard(dte)
            sc.append(concordance_index(dur[te], -risk, evt[te]))
        except Exception:
            sc.append(np.nan)
    return float(np.nanmean(sc)) if np.isfinite(sc).any() else float("nan")


def km_logrank(durations, events, groups) -> float:
    """Multivariate log-rank p-value across ``groups`` — do the KM survival curves differ?

    The test behind a Kaplan–Meier stratification (e.g. TME states from ``cluster_samples``).
    Returns the p-value.
    """
    from lifelines.statistics import multivariate_logrank_test

    return float(multivariate_logrank_test(durations, groups, events).p_value)


# --------------------------------------------------------------------- baseline


def kmer_matrix(frames, k: int = 3) -> np.ndarray:
    """Sample × k-mer frequency matrix (vdjtools ``kmer_profile`` long-form → wide) — the classic baseline.

    Args:
        frames: One clonotype :class:`polars.DataFrame` per sample (with ``junction_aa`` + counts).
        k: k-mer length.

    Returns:
        ``(len(frames), n_kmers)`` frequency matrix over the pooled vocabulary.
    """
    from vdjtools.features import kmer_profile

    dicts = []
    for df in frames:
        kp = kmer_profile(df, k=k, weight="freq", by_locus=False)
        dicts.append(dict(zip(kp["kmer"].to_list(), kp["weight"].to_list())))
    vocab = sorted({x for d in dicts for x in d})
    idx = {x: i for i, x in enumerate(vocab)}
    M = np.zeros((len(dicts), len(vocab)))
    for i, d in enumerate(dicts):
        for x, v in d.items():
            M[i, idx[x]] = v
    return M


def _demo() -> None:
    """Self-check on synthetic data: the scorers recover a planted class / survival signal."""
    rng = np.random.default_rng(0)
    n = 240

    # classification: a signal feature carries y, a noise feature does not
    y = rng.integers(0, 2, n).astype(float)
    signal = y + rng.normal(0, 0.6, n)
    noise = rng.normal(0, 1, n)
    auc_sig = cv_auc(signal, y, n_repeats=3)[0]
    auc_noise = cv_auc(noise, y, n_repeats=3)[0]
    assert auc_sig > 0.7 > auc_noise, (auc_sig, auc_noise)

    # survival: risk drives the hazard; C-index of base+risk beats base-only, and the log-rank splits
    risk = rng.normal(0, 1, n)
    base = rng.normal(0, 1, (n, 2))                                  # uninformative clinical covariates
    dur = rng.exponential(np.exp(-0.9 * risk))                       # higher risk -> shorter survival
    evt = (rng.random(n) < 0.7).astype(float)
    c_base = cv_cindex(dur, evt, base=base, block=None)
    c_full = cv_cindex(dur, evt, base=base, block=risk)
    assert c_full > c_base and c_full > 0.6, (c_base, c_full)
    p = km_logrank(dur, evt, (risk > np.median(risk)).astype(int))
    assert p < 0.05, p

    print(f"mir.bench.eval OK; AUC signal {auc_sig:.2f} > noise {auc_noise:.2f}; "
          f"C-index base {c_base:.2f} -> +risk {c_full:.2f}; log-rank p {p:.1e}")


if __name__ == "__main__":
    _demo()
