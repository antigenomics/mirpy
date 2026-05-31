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

        clones = parser.parse(str(path))
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

    fisher_top = set(fisher_df.head(100)["junction_aa"].astype(str))
    depth_top = set(depth_df.head(100)["junction_aa"].astype(str))
    top_overlap = len(fisher_top & depth_top)

    ref_path = dataset_root / "covid_associated_clonotypes.csv"
    ref_overlap = None
    if ref_path.exists():
        ref_df = pd.read_csv(ref_path)
        if "junction_aa" in ref_df.columns:
            ref_set = set(ref_df["junction_aa"].astype(str))
            ref_overlap = len(fisher_top & ref_set)

    benchmark_log_line(
        "COVID_ASSOC_BENCH "
        f"samples={len(samples)} targets={len(targets)} load_s={load_s:.3f} "
        f"fisher_s={fisher_s:.3f} depth_s={depth_s:.3f} top100_overlap={top_overlap} "
        f"peak_mem_mib={peak_mem_mib:.2f} ref_overlap={ref_overlap}"
    )

    with capsys.disabled():
        print("\n" + "=" * 90)
        print("COVID-19 association benchmark")
        print(f"dataset_root={dataset_root}")
        print(
            f"samples={len(samples)} targets={len(targets)} "
            f"load_s={load_s:.2f} fisher_s={fisher_s:.2f} depth_s={depth_s:.2f} "
            f"top100_overlap={top_overlap} peak_mem_mib={peak_mem_mib:.2f}"
        )
        if ref_overlap is None:
            print("reference comparison: covid_associated_clonotypes.csv not present in local dataset")
        else:
            print(f"reference comparison: top100 fisher overlap with reference={ref_overlap}")
        print("=" * 90)

    assert not fisher_df.empty
    assert not depth_df.empty
    assert top_overlap > 0
    assert float(fisher_s) < benchmark_max_seconds(default=900.0)
    assert float(depth_s) < benchmark_max_seconds(default=900.0)
