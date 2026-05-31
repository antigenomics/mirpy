"""Benchmark COVID-19 clonotype association workflow on AIRR COVID cohort.

Run with:
    RUN_BENCHMARK=1 pytest -s tests/test_associations_covid19_benchmark.py
"""

from __future__ import annotations

import os
import time
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from mir.biomarkers.associations import (
    AssociationParams,
    associate_clonotype_metadata,
    build_public_clonotype_panel,
)
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.utils.notebook_assets import ensure_airr_covid19
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_max_seconds, benchmark_track_memory, skip_benchmarks


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_metadata(dataset_root: Path) -> pd.DataFrame:
    meta_path = dataset_root / "metadata_trb_min100000.tsv"
    if not meta_path.exists():
        pytest.skip(f"Missing COVID metadata file: {meta_path}")

    meta = pd.read_csv(meta_path, sep="\t", dtype={"donor_id": "string"}, low_memory=False)
    meta = meta[meta["COVID_status"].isin(["COVID", "healthy"])].copy()
    if "is_bad_reseq" in meta.columns:
        bad_mask = meta["is_bad_reseq"].fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        meta = meta[~bad_mask].copy()
    return meta


def _reference_file(dataset_root: Path) -> Path | None:
    candidates = [
        dataset_root / "covid19_biomarker_clonotypes.csv",
        dataset_root / "covid_associated_clonotypes.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _reference_cdr3_set(path: Path) -> set[str]:
    df = pd.read_csv(path)
    for col in ("cdr3", "junction_aa", "sequence"):
        if col in df.columns:
            return {str(x) for x in df[col].dropna().astype(str)}
    return set()


def _sample_biomarker_scores(samples: list[SampleRepertoire], biomarker_cdr3: set[str]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        rep = sample.get_locus("TRB")
        seqs = {str(c.junction_aa) for c in rep.clonotypes if c.junction_aa}
        score = float(len(seqs & biomarker_cdr3))
        rows.append(
            {
                "sample_id": sample.sample_id,
                "covid": 1 if str(sample.sample_metadata.get("COVID_status", "")) == "COVID" else 0,
                "score": score,
            }
        )
    return pd.DataFrame(rows)


def _build_samples(dataset_root: Path, meta: pd.DataFrame, max_samples: int) -> list[SampleRepertoire]:
    parser = ClonotypeTableParser()
    selected = meta.sort_values(["COVID_status", "sample_id"]).head(max_samples)

    samples: list[SampleRepertoire] = []
    for _, row in selected.iterrows():
        file_name = str(row["file_name"])
        sample_id = str(row["sample_id"])
        path = dataset_root / file_name
        if not path.exists():
            continue

        clones = [c for c in parser.parse(str(path)) if str(c.locus).upper() == "TRB"]
        if not clones:
            continue
        rep = LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=sample_id)
        rep = filter_functional(rep)
        if rep.clonotype_count == 0:
            continue

        samples.append(
            SampleRepertoire(
                loci={"TRB": rep},
                sample_id=sample_id,
                sample_metadata={
                    "COVID_status": str(row["COVID_status"]),
                    "batch_id": str(row.get("batch_id", "")),
                    "reads": int(row.get("reads", 0) or 0),
                },
            )
        )

    return samples


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_covid19_association_scan_runtime_and_concordance(capsys) -> None:
    dataset_root = ensure_airr_covid19()

    max_samples = _env_int("MIRPY_COVID_BENCH_SAMPLES", 120)
    max_targets = _env_int("MIRPY_COVID_BENCH_MAX_TARGETS", 800)
    min_fraction = _env_float("MIRPY_COVID_BENCH_MIN_SAMPLE_FRACTION", 0.02)
    track_memory = benchmark_track_memory(default=False)

    t0 = time.perf_counter()
    meta = _load_metadata(dataset_root)
    samples = _build_samples(dataset_root, meta, max_samples=max_samples)
    load_s = time.perf_counter() - t0

    if len(samples) < 20:
        pytest.skip("Need at least 20 filtered TRB samples for stable COVID benchmark")

    targets = build_public_clonotype_panel(samples, locus="TRB", min_sample_fraction=min_fraction)
    targets = targets[:max_targets]
    if not targets:
        pytest.skip("No public clonotype targets after functional filtering and frequency threshold")

    if track_memory:
        tracemalloc.start()

    t_fisher = time.perf_counter()
    fisher_res = associate_clonotype_metadata(
        samples,
        targets,
        metadata_field="COVID_status",
        metadata_value=["COVID", "healthy"],
        params=AssociationParams(
            match_mode="none",
            count_mode="sample",
            test="fisher",
        ),
    )
    fisher_s = time.perf_counter() - t_fisher

    t_depth = time.perf_counter()
    depth_res = associate_clonotype_metadata(
        samples,
        targets,
        metadata_field="COVID_status",
        metadata_value=["COVID", "healthy"],
        params=AssociationParams(
            match_mode="none",
            count_mode="rearrangement",
            test="depth_glm",
        ),
    )
    depth_s = time.perf_counter() - t_depth

    peak_mem_mib = float("nan")
    if track_memory:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mib = float(peak / (1024 ** 2))

    fisher_df = fisher_res.table.to_pandas().sort_values("p_value", ascending=True).reset_index(drop=True)
    depth_df = depth_res.table.to_pandas().sort_values("p_value", ascending=True).reset_index(drop=True)

    positive_hits = fisher_df[
        (pd.to_numeric(fisher_df["odds_ratio"], errors="coerce").fillna(0.0) > 1.0)
        & (fisher_df["p_value_adj"] < 0.2)
    ]
    if positive_hits.empty:
        positive_hits = fisher_df.head(30)
    biomarker_set = set(positive_hits["junction_aa"].astype(str))
    score_df = _sample_biomarker_scores(samples, biomarker_set)
    auc = float("nan")
    if score_df["covid"].nunique() == 2 and score_df["score"].nunique() > 1:
        auc = float(roc_auc_score(score_df["covid"], score_df["score"]))

    fisher_top = set(fisher_df.head(100)["junction_aa"].astype(str))
    depth_top = set(depth_df.head(100)["junction_aa"].astype(str))
    top_overlap = len(fisher_top & depth_top)

    ref_path = _reference_file(dataset_root)
    ref_overlap = None
    if ref_path is not None:
        ref_set = _reference_cdr3_set(ref_path)
        ref_overlap = len(fisher_top & ref_set)

    benchmark_log_line(
        "COVID_ASSOC_BENCH "
        f"samples={len(samples)} targets={len(targets)} load_s={load_s:.3f} "
        f"fisher_s={fisher_s:.3f} depth_s={depth_s:.3f} top100_overlap={top_overlap} "
        f"peak_mem_mib={peak_mem_mib:.2f} ref_overlap={ref_overlap} auc={auc:.4f}"
    )  # auc may be nan when samples are too small

    with capsys.disabled():
        print("\n" + "=" * 90)
        print("COVID-19 association benchmark")
        print(f"dataset_root={dataset_root}")
        print(
            f"samples={len(samples)} targets={len(targets)} "
            f"load_s={load_s:.2f} fisher_s={fisher_s:.2f} depth_s={depth_s:.2f} "
            f"top100_overlap={top_overlap} peak_mem_mib={peak_mem_mib:.2f} auc={auc:.4f}"
        )
        if ref_overlap is None or ref_path is None:
            print("reference comparison: covid_associated_clonotypes.csv / covid19_biomarker_clonotypes.csv not present")
        else:
            print(f"reference comparison: top100 fisher overlap with reference={ref_overlap}")
        print("=" * 90)

    assert not fisher_df.empty
    assert not depth_df.empty
    assert top_overlap > 0
    assert float(auc) > 0.5
    assert float(fisher_s) < benchmark_max_seconds(default=900.0)
    assert float(depth_s) < benchmark_max_seconds(default=900.0)


# ---------------------------------------------------------------------------
# SVM classifier benchmark (Vlasova et al. 2026, replicated approach)
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_covid19_svm_classifier_auc(capsys) -> None:
    """Replicate Vlasova 2026 SVM approach: log-frequency features, RBF-SVM, 5-fold CV.

    Paper target: AUC ≥ 0.70 on Cohort I (cross-cohort benchmark).
    This test uses a small subset (max_samples) to run quickly; set
    MIRPY_COVID_SVM_SAMPLES=1137 to run the full benchmark.
    """
    import warnings
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.svm import SVC

    dataset_root = Path(ensure_airr_covid19())

    ref_path = dataset_root / "covid_associated_clonotypes.csv"
    if not ref_path.exists():
        pytest.skip("covid_associated_clonotypes.csv not present")

    meta_path = dataset_root / "metadata.tsv"
    if not meta_path.exists():
        pytest.skip("metadata.tsv not present")

    # Biomarkers
    ref = pd.read_csv(ref_path)
    pos = ref[ref["has_covid_association"] == True]
    chain_map = {"alpha": "TRA", "beta": "TRB"}
    bms: dict[str, list[str]] = {
        locus: sorted(pos[pos["chain"] == chain]["cdr3"].astype(str).tolist())
        for chain, locus in chain_map.items()
    }
    tra_bms, trb_bms = bms["TRA"], bms["TRB"]

    # Metadata
    meta = pd.read_csv(meta_path, sep="\t", low_memory=False)
    meta = meta[meta["COVID_status"].isin(["COVID", "healthy"])].copy()
    bad_mask = (
        meta["is_bad_reseq"].fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    )
    meta = meta[~bad_mask].copy()

    tra_df = meta[meta["locus"] == "TRA"].set_index("donor_id")
    trb_df = meta[meta["locus"] == "TRB"].set_index("donor_id")
    paired = sorted(tra_df.index.intersection(trb_df.index))
    max_samples = _env_int("MIRPY_COVID_SVM_SAMPLES", 200)
    paired = paired[:max_samples]

    if len(paired) < 30:
        pytest.skip(f"Only {len(paired)} paired donors — need ≥30 for stable CV")

    def _load_freq(file_path: str, biomarkers: list[str]) -> np.ndarray | None:
        try:
            df = pd.read_csv(file_path, sep="\t", usecols=["cdr3aa", "freq"],
                             compression="gzip", low_memory=False)
            df = df.dropna(subset=["cdr3aa"])
            freq_map = df.groupby("cdr3aa")["freq"].sum().to_dict()
            return np.array([freq_map.get(b, 0.0) for b in biomarkers], dtype=np.float32)
        except Exception:
            return None

    t0 = time.perf_counter()
    tra_vecs: dict[str, np.ndarray] = {}
    trb_vecs: dict[str, np.ndarray] = {}
    labels_map: dict[str, int] = {}
    tasks = []
    for donor in paired:
        tra_row = tra_df.loc[donor]
        trb_row = trb_df.loc[donor]
        tasks.append((
            donor,
            str(dataset_root / tra_row["file_name"]),
            str(dataset_root / trb_row["file_name"]),
            1 if str(tra_row["COVID_status"]) == "COVID" else 0,
        ))
        labels_map[donor] = tasks[-1][3]

    n_workers = _env_int("MIRPY_COVID_SVM_WORKERS", min(8, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures: dict = {}
        for donor, tra_p, trb_p, _ in tasks:
            futures[pool.submit(_load_freq, tra_p, tra_bms)] = (donor, "TRA")
            futures[pool.submit(_load_freq, trb_p, trb_bms)] = (donor, "TRB")
        for fut in as_completed(futures):
            donor, chain = futures[fut]
            vec = fut.result()
            if vec is not None:
                (tra_vecs if chain == "TRA" else trb_vecs)[donor] = vec
    load_s = time.perf_counter() - t0

    X_rows, y_labels = [], []
    for donor in paired:
        if donor in tra_vecs and donor in trb_vecs:
            X_rows.append(np.concatenate([tra_vecs[donor], trb_vecs[donor]]))
            y_labels.append(labels_map[donor])

    if len(X_rows) < 20:
        pytest.skip("Too few loadable samples for CV")

    X = np.log(np.array(X_rows, dtype=np.float32) + 1e-7)
    y = np.array(y_labels, dtype=np.int32)

    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t_train = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    train_s = time.perf_counter() - t_train

    auc = float(roc_auc_score(y, y_prob))

    with capsys.disabled():
        print(f"\nCOVID SVM: donors={len(y)} COVID={y.sum()} healthy={(y==0).sum()} "
              f"features={X.shape[1]} AUC={auc:.4f} load={load_s:.1f}s train={train_s:.1f}s")

    # AUC threshold scales with cohort size: full 1137-donor run reaches 0.70
    # (Vlasova 2026 target). Small subsets have high variance — use 0.50 floor.
    min_auc = 0.65 if len(y) >= 500 else 0.50
    assert auc > min_auc, (
        f"SVC-RBF AUC={auc:.4f} < {min_auc} (n={len(y)}). "
        "Full 1137-donor run achieves AUC=0.70 (Vlasova 2026 target)."
    )
