"""Standalone benchmark runner for COVID-19 clonotype associations.

Usage:
    source .venv/bin/activate.fish
    /Users/mikesh/vcs/code/mirpy/.venv/bin/python benchmarks/covid19_associations_benchmark.py
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from mir.biomarkers.associations import AssociationParams, associate_clonotype_metadata, build_public_clonotype_panel
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.utils.notebook_assets import ensure_airr_covid19


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


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


def main() -> int:
    dataset_root = ensure_airr_covid19()
    metadata = pd.read_csv(dataset_root / "metadata_trb_min100000.tsv", sep="\t", dtype={"donor_id": "string"}, low_memory=False)
    metadata = metadata[metadata["COVID_status"].isin(["COVID", "healthy"])].copy()

    max_samples = _env_int("MIRPY_COVID_BENCH_SAMPLES", 120)
    max_targets = _env_int("MIRPY_COVID_BENCH_MAX_TARGETS", 800)
    min_fraction = float(os.getenv("MIRPY_COVID_BENCH_MIN_SAMPLE_FRACTION", "0.02"))

    parser = ClonotypeTableParser()
    samples: list[SampleRepertoire] = []

    t0 = time.perf_counter()
    tracemalloc.start()
    for _, row in metadata.sort_values(["COVID_status", "sample_id"]).head(max_samples).iterrows():
        path = Path(dataset_root) / str(row["file_name"])
        if not path.exists():
            continue
        clones = [c for c in parser.parse(str(path)) if str(c.locus).upper() == "TRB"]
        if not clones:
            continue
        rep = filter_functional(LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=str(row["sample_id"])))
        if rep.clonotype_count == 0:
            continue
        samples.append(
            SampleRepertoire(
                loci={"TRB": rep},
                sample_id=str(row["sample_id"]),
                sample_metadata={"COVID_status": str(row["COVID_status"]), "batch_id": str(row.get("batch_id", ""))},
            )
        )

    load_s = time.perf_counter() - t0
    _, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    targets = build_public_clonotype_panel(samples, locus="TRB", min_sample_fraction=min_fraction)[:max_targets]

    tracemalloc.start()
    t1 = time.perf_counter()
    fisher_res = associate_clonotype_metadata(
        samples,
        targets,
        metadata_field="COVID_status",
        metadata_value=["COVID", "healthy"],
        params=AssociationParams(test="fisher", count_mode="sample", match_mode="none"),
    )
    fisher_s = time.perf_counter() - t1
    _, fisher_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t2 = time.perf_counter()
    depth_res = associate_clonotype_metadata(
        samples,
        targets,
        metadata_field="COVID_status",
        metadata_value=["COVID", "healthy"],
        params=AssociationParams(test="depth_glm", count_mode="rearrangement", match_mode="none"),
    )
    depth_s = time.perf_counter() - t2
    _, depth_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    fisher_df = fisher_res.table.to_pandas().sort_values(["p_value_adj", "p_value"]).reset_index(drop=True)
    positive_hits = fisher_df[(fisher_df["odds_ratio"].fillna(0.0) > 1.0) & (fisher_df["p_value_adj"] < 0.2)]
    if positive_hits.empty:
        positive_hits = fisher_df.head(30)
    biomarker_set = set(positive_hits["junction_aa"].astype(str))
    score_df = _sample_biomarker_scores(samples, biomarker_set)
    auc = float("nan")
    if score_df["covid"].nunique() == 2 and score_df["score"].nunique() > 1:
        auc = float(roc_auc_score(score_df["covid"], score_df["score"]))

    ref = _reference_file(Path(dataset_root))
    ref_overlap_top100 = None
    ref_overlap_biomarkers = None
    if ref is not None:
        ref_set = _reference_cdr3_set(ref)
        top100 = set(fisher_df.head(100)["junction_aa"].astype(str))
        ref_overlap_top100 = len(top100 & ref_set)
        ref_overlap_biomarkers = len(biomarker_set & ref_set)

    out = {
        "dataset_root": str(dataset_root),
        "samples": len(samples),
        "targets": len(targets),
        "load_seconds": load_s,
        "load_peak_mib": float(load_peak / (1024 ** 2)),
        "fisher_seconds": fisher_s,
        "fisher_peak_mib": float(fisher_peak / (1024 ** 2)),
        "depth_glm_seconds": depth_s,
        "depth_glm_peak_mib": float(depth_peak / (1024 ** 2)),
        "fisher_rows": int(fisher_res.table.height),
        "depth_rows": int(depth_res.table.height),
        "biomarker_count": int(len(biomarker_set)),
        "separation_auc": auc,
        "reference_csv": str(ref) if ref is not None else None,
        "reference_overlap_top100": ref_overlap_top100,
        "reference_overlap_biomarkers": ref_overlap_biomarkers,
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
