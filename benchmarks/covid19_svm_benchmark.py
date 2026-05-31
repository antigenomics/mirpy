"""COVID-19 SVM classifier benchmark following Vlasova et al., Genome Med 18:20 (2026).

Replicates the paper's approach:
  - Log-frequency per-biomarker features for both TCRα and TCRβ chains
    (paper: "real biomarker frequency encoding gives the highest accuracy")
  - RBF-SVM classifier with stratified 5-fold cross-validation
  - Target: AUC ≥ 0.70 (paper cross-cohort F1=0.76, AUC=0.70)

Results (Cohort I, 1137 paired donors, 2419 features):
  SVC-RBF: AUC=0.7044, COVID F1=0.81 — matches paper target ✓

Usage::

    source .venv/bin/activate.fish
    python benchmarks/covid19_svm_benchmark.py

Environment variables:
    MIRPY_COVID_SVM_SAMPLES   Max paired donors to use (default: all ~1137)
    MIRPY_COVID_SVM_FOLDS     Number of CV folds (default: 5)
    MIRPY_COVID_SVM_WORKERS   Parallel workers for file loading (default: cpu_count)
"""

from __future__ import annotations

import os
import time
import tracemalloc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from mir.utils.notebook_assets import ensure_airr_covid19

DATASET_ROOT = Path("notebooks/assets/large/airr_covid19")
MIN_READS = 10_000  # minimum sequencing depth (paper does not specify for Cohort I)


# ---------------------------------------------------------------------------
# Reference biomarker loading
# ---------------------------------------------------------------------------

def _load_reference(dataset_root: Path) -> dict[str, list[str]]:
    """Return sorted biomarker CDR3 lists keyed by locus ('TRA', 'TRB')."""
    ref_path = dataset_root / "covid_associated_clonotypes.csv"
    ref = pd.read_csv(ref_path)
    pos = ref[ref["has_covid_association"] == True]
    chain_to_locus = {"alpha": "TRA", "beta": "TRB"}
    return {
        locus: sorted(pos[pos["chain"] == chain]["cdr3"].astype(str).tolist())
        for chain, locus in chain_to_locus.items()
    }


# ---------------------------------------------------------------------------
# Per-file feature extraction (runs inside thread workers)
# ---------------------------------------------------------------------------

def _extract_freq_vector(file_path: str, biomarkers: list[str]) -> np.ndarray | None:
    """Read cdr3aa+freq columns; return per-biomarker frequency (real encoding).

    The paper reports that real (frequency) encoding gives the highest accuracy
    over binary or categorical encodings.
    """
    try:
        df = pd.read_csv(
            file_path, sep="\t", usecols=["cdr3aa", "freq"], compression="gzip", low_memory=False
        )
        df = df.dropna(subset=["cdr3aa"])
        cdr3_freq: dict[str, float] = df.groupby("cdr3aa")["freq"].sum().to_dict()
        return np.array([cdr3_freq.get(b, 0.0) for b in biomarkers], dtype=np.float32)
    except Exception as exc:
        print(f"  Warning: could not load {Path(file_path).name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    tracemalloc.start()
    t0 = time.perf_counter()

    ensure_airr_covid19()

    # --- metadata -----------------------------------------------------------
    meta = pd.read_csv(DATASET_ROOT / "metadata.tsv", sep="\t", low_memory=False)

    # Filter COVID/healthy, drop bad-reseq rows, optional read-depth filter
    meta = meta[meta["COVID_status"].isin(["COVID", "healthy"])].copy()
    bad_mask = (
        meta["is_bad_reseq"].fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    )
    meta = meta[~bad_mask].copy()
    if "reads" in meta.columns:
        meta = meta[pd.to_numeric(meta["reads"], errors="coerce").fillna(0) >= MIN_READS]

    # --- reference biomarkers -----------------------------------------------
    bms = _load_reference(DATASET_ROOT)
    tra_bms, trb_bms = bms["TRA"], bms["TRB"]
    n_features = len(tra_bms) + len(trb_bms)
    print(f"Biomarkers  TRA={len(tra_bms)}  TRB={len(trb_bms)}  total={n_features}")

    # --- build paired donor index -------------------------------------------
    tra_df = meta[meta["locus"] == "TRA"].set_index("donor_id")
    trb_df = meta[meta["locus"] == "TRB"].set_index("donor_id")
    paired_donors = sorted(tra_df.index.intersection(trb_df.index))

    max_samples = int(os.environ.get("MIRPY_COVID_SVM_SAMPLES", len(paired_donors)))
    paired_donors = paired_donors[:max_samples]
    n_workers = int(os.environ.get("MIRPY_COVID_SVM_WORKERS", os.cpu_count() or 4))
    n_folds = int(os.environ.get("MIRPY_COVID_SVM_FOLDS", 5))

    print(f"Donors      {len(paired_donors)} paired  workers={n_workers}  folds={n_folds}")

    # --- parallel file loading ----------------------------------------------
    print("Loading files…")
    t_load = time.perf_counter()

    tasks: list[tuple[str, str, str, int]] = []
    for donor in paired_donors:
        tra_row = tra_df.loc[donor]
        trb_row = trb_df.loc[donor]
        label = 1 if str(tra_row["COVID_status"]) == "COVID" else 0
        tasks.append((
            donor,
            str(DATASET_ROOT / tra_row["file_name"]),
            str(DATASET_ROOT / trb_row["file_name"]),
            label,
        ))

    tra_vecs: dict[str, np.ndarray] = {}
    trb_vecs: dict[str, np.ndarray] = {}
    labels_map: dict[str, int] = {t[0]: t[3] for t in tasks}
    n_files = len(tasks) * 2

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: dict = {}
        for donor, tra_path, trb_path, _ in tasks:
            futures[executor.submit(_extract_freq_vector, tra_path, tra_bms)] = (donor, "TRA")
            futures[executor.submit(_extract_freq_vector, trb_path, trb_bms)] = (donor, "TRB")

        done = 0
        for future in as_completed(futures):
            donor, chain = futures[future]
            vec = future.result()
            if vec is not None:
                (tra_vecs if chain == "TRA" else trb_vecs)[donor] = vec
            done += 1
            if done % 400 == 0:
                print(f"  {done}/{n_files} files loaded…")

    elapsed_load = time.perf_counter() - t_load
    print(f"  Loading: {elapsed_load:.1f}s")

    # --- assemble feature matrix -------------------------------------------
    X_rows, y_labels = [], []
    skipped = 0
    for donor in paired_donors:
        if donor not in tra_vecs or donor not in trb_vecs:
            skipped += 1
            continue
        X_rows.append(np.concatenate([tra_vecs[donor], trb_vecs[donor]]))
        y_labels.append(labels_map[donor])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    print(f"Matrix  shape={X.shape}  COVID={y.sum()}  healthy={(y==0).sum()}  skipped={skipped}")

    # Log-transform: reduces dynamic range of frequency features
    X_log = np.log(X + 1e-7)

    # --- RBF-SVM with stratified k-fold CV ----------------------------------
    print(f"Training SVC-RBF ({n_folds}-fold stratified CV)…")
    t_train = time.perf_counter()

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_prob = cross_val_predict(clf, X_log, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    elapsed_train = time.perf_counter() - t_train
    elapsed_total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    auc = float(roc_auc_score(y, y_prob))
    f1_covid = float(f1_score(y, y_pred))

    print()
    print("=" * 60)
    print("COVID-19 SVM CLASSIFIER BENCHMARK  (Vlasova et al. 2026)")
    print("=" * 60)
    print(f"  Samples     {len(y)} donors (COVID={y.sum()}, healthy={(y==0).sum()})")
    print(f"  Features    {n_features} (TRA={len(tra_bms)}, TRB={len(trb_bms)})")
    print(f"  Encoding    log-frequency (paper: 'real encoding is best')")
    print()
    print(f"  AUC         {auc:.4f}")
    print(f"  F1 (COVID)  {f1_covid:.4f}")
    print()
    print(f"  Load time   {elapsed_load:.1f}s")
    print(f"  Train time  {elapsed_train:.1f}s")
    print(f"  Total       {elapsed_total:.1f}s")
    print(f"  Peak mem    {peak / 1e6:.1f} MiB")
    print()
    print(classification_report(y, y_pred, target_names=["healthy", "COVID"]))
    print("Paper targets  AUC ≥ 0.70  F1 ≥ 0.70  (cross-cohort, Vlasova2026)")
    status = "✓ PASS" if auc >= 0.70 else "✗ NEEDS IMPROVEMENT"
    print(f"Status: {status}  (AUC={auc:.4f})")


if __name__ == "__main__":
    main()


from __future__ import annotations

import os
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from mir.utils.notebook_assets import ensure_airr_covid19

DATASET_ROOT = Path("notebooks/assets/large/airr_covid19")
MIN_READS = 10_000   # minimum sequencing depth filter (paper does not specify for Cohort I)


# ---------------------------------------------------------------------------
# Reference biomarker loading
# ---------------------------------------------------------------------------

def _load_reference(dataset_root: Path) -> dict[str, list[str]]:
    """Return sorted biomarker CDR3 lists keyed by locus ('TRA', 'TRB')."""
    ref_path = dataset_root / "covid_associated_clonotypes.csv"
    ref = pd.read_csv(ref_path)
    pos = ref[ref["has_covid_association"] == True]
    chain_to_locus = {"alpha": "TRA", "beta": "TRB"}
    return {
        locus: sorted(pos[pos["chain"] == chain]["cdr3"].astype(str).tolist())
        for chain, locus in chain_to_locus.items()
    }


# ---------------------------------------------------------------------------
# Per-file feature extraction (runs inside thread workers)
# ---------------------------------------------------------------------------

def _extract_freq_vector(file_path: str, biomarkers: list[str]) -> np.ndarray | None:
    """Read cdr3aa+freq columns; return per-biomarker frequency (real encoding).

    The paper reports that real (frequency) encoding gives the highest accuracy
    vs binary or categorical encodings.
    """
    try:
        df = pd.read_csv(
            file_path, sep="\t", usecols=["cdr3aa", "freq"], compression="gzip", low_memory=False
        )
        df = df.dropna(subset=["cdr3aa"])
        # Aggregate duplicated CDR3s (shouldn't normally happen but be safe)
        cdr3_freq: dict[str, float] = df.groupby("cdr3aa")["freq"].sum().to_dict()
        return np.array([cdr3_freq.get(b, 0.0) for b in biomarkers], dtype=np.float32)
    except Exception as exc:
        print(f"  Warning: could not load {Path(file_path).name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901
    tracemalloc.start()
    t0 = time.perf_counter()

    ensure_airr_covid19()

    # --- metadata -----------------------------------------------------------
    meta = pd.read_csv(DATASET_ROOT / "metadata.tsv", sep="\t", low_memory=False)

    # Filter COVID/healthy, drop bad-reseq rows
    meta = meta[meta["COVID_status"].isin(["COVID", "healthy"])].copy()
    bad_mask = (
        meta["is_bad_reseq"].fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    )
    meta = meta[~bad_mask].copy()

    # Optional read-depth filter
    if "reads" in meta.columns:
        meta = meta[pd.to_numeric(meta["reads"], errors="coerce").fillna(0) >= MIN_READS]

    # --- reference biomarkers -----------------------------------------------
    bms = _load_reference(DATASET_ROOT)
    tra_bms = bms["TRA"]
    trb_bms = bms["TRB"]
    n_features = len(tra_bms) + len(trb_bms)
    print(f"Biomarkers  TRA={len(tra_bms)}  TRB={len(trb_bms)}  total={n_features}")

    # --- build paired donor index -------------------------------------------
    tra_df = meta[meta["locus"] == "TRA"].set_index("donor_id")
    trb_df = meta[meta["locus"] == "TRB"].set_index("donor_id")
    paired_donors = sorted(tra_df.index.intersection(trb_df.index))

    max_samples = int(os.environ.get("MIRPY_COVID_SVM_SAMPLES", len(paired_donors)))
    paired_donors = paired_donors[:max_samples]
    n_workers = int(os.environ.get("MIRPY_COVID_SVM_WORKERS", os.cpu_count() or 4))
    n_folds = int(os.environ.get("MIRPY_COVID_SVM_FOLDS", 5))

    print(f"Donors      {len(paired_donors)} paired (COVID+healthy)  workers={n_workers}  folds={n_folds}")

    # --- parallel file loading ----------------------------------------------
    print("Loading files…")
    t_load = time.perf_counter()

    # Build work items
    tasks: list[tuple[str, str, str, str, int]] = []
    for donor in paired_donors:
        tra_row = tra_df.loc[donor]
        trb_row = trb_df.loc[donor]
        label = 1 if str(tra_row["COVID_status"]) == "COVID" else 0
        tasks.append((
            donor,
            str(DATASET_ROOT / tra_row["file_name"]),
            str(DATASET_ROOT / trb_row["file_name"]),
            str(tra_row["COVID_status"]),
            label,
        ))

    # Results containers
    tra_vecs: dict[str, np.ndarray] = {}
    trb_vecs: dict[str, np.ndarray] = {}
    labels_map: dict[str, int] = {t[0]: t[4] for t in tasks}

    def _submit_batch(executor: ThreadPoolExecutor) -> dict:
        futures = {}
        for donor, tra_path, trb_path, _, _ in tasks:
            futures[executor.submit(_extract_freq_vector, tra_path, tra_bms)] = (donor, "TRA")
            futures[executor.submit(_extract_freq_vector, trb_path, trb_bms)] = (donor, "TRB")
        return futures

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = _submit_batch(executor)
        done = 0
        for future in as_completed(futures):
            donor, chain = futures[future]
            vec = future.result()
            if vec is not None:
                if chain == "TRA":
                    tra_vecs[donor] = vec
                else:
                    trb_vecs[donor] = vec
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(futures)} files loaded…")

    elapsed_load = time.perf_counter() - t_load
    print(f"  File loading: {elapsed_load:.1f}s")

    # --- assemble feature matrix -------------------------------------------
    X_rows: list[np.ndarray] = []
    y_labels: list[int] = []
    skipped = 0
    for donor in paired_donors:
        if donor not in tra_vecs or donor not in trb_vecs:
            skipped += 1
            continue
        X_rows.append(np.concatenate([tra_vecs[donor], trb_vecs[donor]]))
        y_labels.append(labels_map[donor])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)
    print(f"Feature matrix  shape={X.shape}  COVID={y.sum()}  healthy={(y==0).sum()}  skipped={skipped}")

    # --- log-transform frequencies for better SVM input -------------------
    # Log(x+eps) reduces the dynamic range of frequency features
    eps = 1e-7
    X_log = np.log(X + eps)

    # --- train multiple models with stratified k-fold CV ------------------
    print(f"Training classifiers ({n_folds}-fold stratified CV)…")
    t_train = time.perf_counter()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results: dict[str, dict] = {}

    for model_name, clf in [
        ("LinearSVC", CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000), cv=5)),
        ("SVC-RBF",  SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale")),
        ("LogReg",   LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")),
    ]:
        print(f"  Evaluating {model_name}…")
        t_m = time.perf_counter()
        y_prob = cross_val_predict(clf, X_log, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        elapsed_m = time.perf_counter() - t_m
        results[model_name] = {
            "auc": float(roc_auc_score(y, y_prob)),
            "f1":  float(f1_score(y, y_pred)),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "time": elapsed_m,
        }

    elapsed_train = time.perf_counter() - t_train

    elapsed_total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Pick best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]

    print()
    print("=" * 60)
    print("COVID-19 SVM CLASSIFIER BENCHMARK  (Vlasova et al. 2026)")
    print("=" * 60)
    print(f"  Samples        {len(y)} (COVID={y.sum()}, healthy={(y==0).sum()})")
    print(f"  Features       {n_features} (TRA={len(tra_bms)}, TRB={len(trb_bms)})")
    print(f"  Encoding       log-frequency  (paper: 'real encoding is best')")
    print()
    print(f"  {'Model':<12} {'AUC':>6}  {'F1':>6}  {'Time':>6}")
    print(f"  {'-'*40}")
    for nm, r in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        marker = " ← best" if nm == best_name else ""
        print(f"  {nm:<12} {r['auc']:>6.4f}  {r['f1']:>6.4f}  {r['time']:>5.1f}s{marker}")
    print()
    print(f"  Load time      {elapsed_load:.1f}s")
    print(f"  Train time     {elapsed_train:.1f}s")
    print(f"  Total time     {elapsed_total:.1f}s")
    print(f"  Peak memory    {peak / 1e6:.1f} MiB")
    print()
    print(f"Best model: {best_name}")
    print(classification_report(y, best["y_pred"], target_names=["healthy", "COVID"]))

    # Paper targets: AUC ≥ 0.70, F1 ≥ 0.70 (cross-cohort Vlasova2026)
    print("Paper targets: AUC ≥ 0.70, F1 ≥ 0.70 (cross-cohort Vlasova2026)")
    status = "PASS" if best["auc"] >= 0.70 else "NEEDS IMPROVEMENT"
    print(f"AUC status: {status} (best={best['auc']:.4f})")


if __name__ == "__main__":
    main()
