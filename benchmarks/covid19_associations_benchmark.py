"""Standalone benchmark runner for COVID-19 clonotype associations.

Usage:
    source .venv/bin/activate.fish
    /Users/mikesh/vcs/code/mirpy/.venv/bin/python benchmarks/covid19_associations_benchmark.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

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
    for _, row in metadata.sort_values(["COVID_status", "sample_id"]).head(max_samples).iterrows():
        path = Path(dataset_root) / str(row["file_name"])
        if not path.exists():
            continue
        clones = parser.parse(str(path))
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
    targets = build_public_clonotype_panel(samples, locus="TRB", min_sample_fraction=min_fraction)[:max_targets]

    t1 = time.perf_counter()
    fisher_res = associate_clonotype_metadata(
        samples,
        targets,
        metadata_field="COVID_status",
        metadata_value=["COVID", "healthy"],
        params=AssociationParams(test="fisher", count_mode="sample", match_mode="none"),
    )
    fisher_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    depth_res = associate_clonotype_metadata(
        samples,
        targets,
        metadata_field="COVID_status",
        metadata_value=["COVID", "healthy"],
        params=AssociationParams(test="depth_glm", count_mode="rearrangement", match_mode="none"),
    )
    depth_s = time.perf_counter() - t2

    out = {
        "dataset_root": str(dataset_root),
        "samples": len(samples),
        "targets": len(targets),
        "load_seconds": load_s,
        "fisher_seconds": fisher_s,
        "depth_glm_seconds": depth_s,
        "fisher_rows": int(fisher_res.table.height),
        "depth_rows": int(depth_res.table.height),
        "reference_csv_exists": bool((Path(dataset_root) / "covid_associated_clonotypes.csv").exists()),
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
