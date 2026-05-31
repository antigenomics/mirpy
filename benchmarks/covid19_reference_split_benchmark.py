"""Benchmark alpha/beta separation using split reference biomarker clonotype file.

This benchmark does not infer biomarkers; it scores samples by counting presence
of reference biomarker clonotypes from ``covid_associated_clonotypes.csv`` split
by chain and reports runtime, memory footprint, and AUC.

Usage:
    source .venv/bin/activate.fish
    /Users/mikesh/vcs/code/mirpy/.venv/bin/python benchmarks/covid19_reference_split_benchmark.py
"""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire
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


def _load_reference(dataset_root: Path) -> dict[str, set[str]]:
    ref_path = dataset_root / "covid_associated_clonotypes.csv"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file is missing: {ref_path}")

    ref = pd.read_csv(ref_path)
    required = {"cdr3", "chain", "has_covid_association"}
    missing = sorted(required - set(ref.columns))
    if missing:
        raise ValueError(f"Reference file missing required columns: {missing}")

    ref = ref[ref["has_covid_association"] == True].copy()
    chain_to_locus = {"alpha": "TRA", "beta": "TRB"}
    out: dict[str, set[str]] = {}
    for chain, locus in chain_to_locus.items():
        out[locus] = set(ref[ref["chain"] == chain]["cdr3"].astype(str))
    return out


def _score_chain(
    *,
    dataset_root: Path,
    metadata: pd.DataFrame,
    locus: str,
    biomarkers: set[str],
    max_samples: int,
) -> dict[str, float | int | str]:
    parser = ClonotypeTableParser()
    selected = metadata[
        (metadata["locus"].astype(str).str.upper() == locus)
        & (metadata["COVID_status"].isin(["COVID", "healthy"]))
    ].copy()

    if "is_bad_reseq" in selected.columns:
        bad_mask = selected["is_bad_reseq"].fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
        selected = selected[~bad_mask].copy()

    selected = selected.sort_values(["sample_id"])
    half = max_samples // 2
    covid_part = selected[selected["COVID_status"] == "COVID"].head(half)
    healthy_part = selected[selected["COVID_status"] == "healthy"].head(half)
    selected = pd.concat([covid_part, healthy_part], axis=0).sort_values(["COVID_status", "sample_id"]).reset_index(drop=True)

    labels: list[int] = []
    scores: list[float] = []
    n_loaded = 0

    tracemalloc.start()
    t0 = time.perf_counter()

    for _, row in selected.iterrows():
        sample_path = dataset_root / str(row["file_name"])
        if not sample_path.exists():
            continue
        clones = [c for c in parser.parse(str(sample_path)) if str(c.locus).upper() == locus]
        if not clones:
            continue

        rep = filter_functional(LocusRepertoire(clonotypes=clones, locus=locus, repertoire_id=str(row["sample_id"])))
        if rep.clonotype_count == 0:
            continue

        seqs = {str(c.junction_aa) for c in rep.clonotypes if c.junction_aa}
        score = float(len(seqs & biomarkers))

        labels.append(1 if str(row["COVID_status"]) == "COVID" else 0)
        scores.append(score)
        n_loaded += 1

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    auc = float("nan")
    if len(set(labels)) == 2 and len(set(scores)) > 1:
        auc = float(roc_auc_score(labels, scores))

    return {
        "locus": locus,
        "samples_loaded": int(n_loaded),
        "covid_loaded": int(sum(labels)),
        "healthy_loaded": int(len(labels) - sum(labels)),
        "biomarker_count": int(len(biomarkers)),
        "elapsed_seconds": float(elapsed),
        "peak_mib": float(peak / (1024 ** 2)),
        "auc": float(auc),
        "score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
    }


def main() -> int:
    dataset_root = ensure_airr_covid19()
    metadata = pd.read_csv(dataset_root / "metadata.tsv", sep="\t", dtype={"donor_id": "string"}, low_memory=False)

    max_samples = _env_int("MIRPY_COVID_REF_BENCH_SAMPLES", 160)
    refs = _load_reference(dataset_root)

    out = {
        "dataset_root": str(dataset_root),
        "max_samples_per_chain": int(max_samples),
        "results": [],
    }

    for locus in ("TRA", "TRB"):
        out["results"].append(
            _score_chain(
                dataset_root=dataset_root,
                metadata=metadata,
                locus=locus,
                biomarkers=refs.get(locus, set()),
                max_samples=max_samples,
            )
        )

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
