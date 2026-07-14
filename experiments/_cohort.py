# 2026-07-14
# Cohort loader for repertoire-level benchmarks (mir.repertoire): resolve a vdjtools
# metadata file to per-sample AIRR frames, downsample each to a common depth (the RNA-seq
# regime — cost is per-sample depth, not sample count), and build a pooled clonotype cloud
# for fitting ONE RepertoireSpace. Reuses experiments/_hf.py + vdjtools io/preprocess.

from __future__ import annotations

import numpy as np
import polars as pl

from _hf import fetch, load_repertoire


def load_cohort(
    repo: str,
    metadata_file: str,
    *,
    prefix: str = "",
    suffix: str = "",
    sample_col: str = "file_name",
    downsample_to: int | None = None,
    cap_samples: int | None = None,
    only: set | None = None,
    top: int | None = None,
    seed: int = 0,
) -> tuple[pl.DataFrame, list[tuple[dict, pl.DataFrame]]]:
    """Load a cohort: ``(metadata_frame, [(row_dict, sample_df), ...])``.

    Each sample file is resolved as ``prefix + row[sample_col] + suffix`` (e.g. aging:
    ``vdjtools/`` + ``A3-i101.txt`` + ``.gz``; hip: ``file_name`` is already the full path).
    ``downsample_to`` caps each sample's reads (vdjtools hypergeometric) to the shallow regime;
    ``cap_samples`` limits how many samples are loaded (for a quick slice).
    """
    from vdjtools.io.schema import recompute_frequency
    from vdjtools.io.batch import read_metadata
    from vdjtools.preprocess import downsample

    meta = read_metadata(fetch(repo, metadata_file))
    rows = meta.to_dicts()
    if only is not None:
        rows = [r for r in rows if r[sample_col] in only]
    if cap_samples is not None:
        rows = rows[:cap_samples]
    samples: list[tuple[dict, pl.DataFrame]] = []
    for r in rows:
        df = load_repertoire(fetch(repo, prefix + r[sample_col] + suffix), top=top)
        if downsample_to is not None and int(df["duplicate_count"].sum()) > downsample_to:
            df = downsample(df, downsample_to, by="reads", seed=seed)   # adds `frequency`
        else:
            df = recompute_frequency(df)                                # shallow sample: add it too
        samples.append((r, df))
    return meta, samples


def pooled_clonotypes(samples, *, per_sample: int = 2000, seed: int = 0) -> pl.DataFrame:
    """Pool up to ``per_sample`` clonotypes per sample into one cloud for fitting the basis."""
    parts = [
        df.sample(min(per_sample, df.height), seed=seed).select(["v_call", "j_call", "junction_aa"])
        for _, df in samples
    ]
    return pl.concat(parts).unique()


def held_out_auc(Xtr, ytr, Xte, yte, *, pca_cols: int = 0) -> float:
    """Held-out ROC-AUC of a logistic head; first ``pca_cols`` columns get in-fold PCA(0.9 var).

    Wide blocks (kernel mean, second moment, k-mer) must be reduced *inside* the fit or they
    overfit; the few diversity features pass through raw so their signal isn't buried.
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
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def cv_auc(X, y, *, pca_cols: int = 0, n_splits: int = 5, n_repeats: int = 10, seed: int = 0):
    """Repeated stratified k-fold AUC as ``(mean, std)`` — a CI, not a single-split point estimate.

    A single 70/30 split at small n has AUC SD ≈ 0.5/√n_test (~0.1 for n_test≈30), so point estimates
    are near-meaningless; repeated CV exposes whether two methods' intervals actually separate.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    X = X.reshape(-1, 1) if X.ndim == 1 else X
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs = [held_out_auc(X[tr], y[tr], X[te], y[te], pca_cols=pca_cols) for tr, te in rskf.split(X, y)]
    return float(np.mean(aucs)), float(np.std(aucs))


def kmer_matrix(samples, k: int = 3) -> np.ndarray:
    """Sample × k-mer frequency matrix (vdjtools kmer_profile long-form -> wide) — a baseline."""
    from vdjtools.features import kmer_profile

    dicts = []
    for _, df in samples:
        kp = kmer_profile(df, k=k, weight="freq", by_locus=False)
        dicts.append(dict(zip(kp["kmer"].to_list(), kp["weight"].to_list())))
    vocab = sorted({x for d in dicts for x in d})
    idx = {x: i for i, x in enumerate(vocab)}
    M = np.zeros((len(dicts), len(vocab)))
    for i, d in enumerate(dicts):
        for x, v in d.items():
            M[i, idx[x]] = v
    return M
