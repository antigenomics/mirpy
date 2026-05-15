"""Benchmark CITE-seq matrix loading and VDJdb-10x sanity checks on all dcode donors."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from mir.common.single_cell import (
    load_10x_vdj_v1_citeseq_sample,
    validate_citeseq_binders_against_vdjdb_10x,
)
from tests.prepare_airr_benchmark_data import ensure_test_data

RUN_BENCHMARK = os.getenv("RUN_BENCHMARK") == "1"
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_BENCHMARK, reason="set RUN_BENCHMARK=1 to run benchmark tests"),
]


def _discover_donors(dcode_root: Path) -> list[tuple[str, Path, Path, Path]]:
    all_contig_files = sorted(dcode_root.glob("*_all_contig_annotations.csv.gz"))
    consensus_map = {
        p.name.replace("_consensus_annotations.csv.gz", ""): p
        for p in dcode_root.glob("*_consensus_annotations.csv.gz")
    }
    matrix_map = {
        p.name.replace("_binarized_matrix.csv.gz", ""): p
        for p in dcode_root.glob("*_binarized_matrix.csv.gz")
    }

    donors: list[tuple[str, Path, Path, Path]] = []
    for all_contig in all_contig_files:
        key = all_contig.name.replace("_all_contig_annotations.csv.gz", "")
        consensus = consensus_map.get(key)
        matrix = matrix_map.get(key)
        if consensus is not None and matrix is not None:
            donors.append((key, all_contig, consensus, matrix))
    return donors


def test_citeseq_load_and_vdjdb_10x_sanity_all_donors() -> None:
    ensure_test_data(force=False, verbose=False)
    root = Path(__file__).resolve().parents[1] / "airr_benchmark"
    dcode_root = root / "dcode"
    vdjdb_full = root / "vdjdb" / "vdjdb-2025-12-29" / "vdjdb_full.txt.gz"
    if not dcode_root.exists() or not vdjdb_full.exists():
        pytest.skip("required benchmark assets are missing")

    donors = _discover_donors(dcode_root)
    if not donors:
        pytest.skip("no donor triplets with all_contig/consensus/binarized_matrix found")

    for donor_id, all_contig, consensus, matrix in donors:
        t0 = time.perf_counter()
        sample = load_10x_vdj_v1_citeseq_sample(
            consensus_annotations_path=consensus,
            all_contig_annotations_path=all_contig,
            binarized_matrix_path=matrix,
            sample_id=donor_id,
        )
        dt = time.perf_counter() - t0

        missing = validate_citeseq_binders_against_vdjdb_10x(
            sample.cite_seq_binder_columns,
            vdjdb_full,
        )
        expected_residual = {
            ("A0201", "CLGGLLTMV"),
            ("A0201", "LLMGTLGIVC"),
        }
        observed = {(row["hla"], row["antigen.epitope"]) for row in missing.to_dicts()}

        print(
            f"\n{donor_id}: cells={sample.paired_repertoire.loaded_cell_count:,} "
            f"pairs={sample.paired_repertoire.paired_locus_repertoires['TRA_TRB'].clonotype_count:,} "
            f"matrix_rows={sample.cite_seq_matrix.height:,} binder_cols={sample.cite_seq_binder_columns.height:,} "
            f"time={dt:.2f}s"
        )

        assert sample.paired_repertoire.loaded_cell_count > 0
        assert sample.cite_seq_matrix.height > 0
        assert sample.cite_seq_binder_columns.height > 0
        assert dt < 45.0
        assert observed.issubset(expected_residual)
