"""Benchmarks for single-cell imputation and cleanup timing."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mir.common.single_cell_parser import load_10x_vdj_v1_cell_clonotypes
from mir.common.single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains
from tests.prepare_airr_benchmark_data import ensure_test_data

RUN_BENCHMARK = os.getenv("RUN_BENCHMARK") == "1"
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_BENCHMARK, reason="set RUN_BENCHMARK=1 to run benchmark tests"),
]


def _first_sample_pair() -> tuple[str, Path, Path]:
    dcode_root = Path(__file__).resolve().parents[1] / "airr_benchmark" / "dcode"
    all_contig_files = sorted(dcode_root.glob("*_all_contig_annotations.csv.gz"))
    consensus_files = {
        p.name.replace("_consensus_annotations.csv.gz", ""): p
        for p in dcode_root.glob("*_consensus_annotations.csv.gz")
    }
    for all_contig in all_contig_files:
        key = all_contig.name.replace("_all_contig_annotations.csv.gz", "")
        cons = consensus_files.get(key)
        if cons is not None:
            return key, cons, all_contig
    raise RuntimeError("No sample files found for single-cell repair benchmark")


def test_impute_and_cleanup_runtime() -> None:
    ensure_test_data(force=False, verbose=False)
    sample_id, consensus, all_contig = _first_sample_pair()

    raw = load_10x_vdj_v1_cell_clonotypes(
        consensus_annotations_path=consensus,
        all_contig_annotations_path=all_contig,
        sample_id=sample_id,
    )

    t0 = time.perf_counter()
    imputed = impute_missing_chains(raw, seed=42, reuse_slave_per_master=True)
    t_impute = time.perf_counter() - t0

    t1 = time.perf_counter()
    cleaned = cleanup_cell_clonotypes(
        imputed,
        enforce_consistent_slave_per_master=True,
        max_slave_edges_per_master=10,
    )
    t_cleanup = time.perf_counter() - t1

    print(f"impute_seconds={t_impute:.3f}")
    print(f"cleanup_seconds={t_cleanup:.3f}")
    print(f"raw_rows={raw.height} imputed_rows={imputed.height} cleaned_rows={cleaned.height}")

    assert imputed.height >= raw.height
    assert cleaned.height <= imputed.height
    assert t_impute < 30.0
    assert t_cleanup < 30.0
