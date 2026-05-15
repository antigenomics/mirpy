"""Sanity tests for single-cell repair and pairing graph utilities."""

from __future__ import annotations

import polars as pl
import pytest

from mir.common.single_cell import build_tenx_sample_from_cell_clonotypes
from mir.common.single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains
from mir.graph.single_cell_pairing import build_pairing_graph


def _cell_table(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
    "sample_id": pl.Utf8,
        "barcode": pl.Utf8,
        "raw_pair_id": pl.Utf8,
        "sequence_id": pl.Utf8,
        "locus": pl.Utf8,
        "duplicate_count": pl.Int64,
        "umi_count": pl.Int64,
        "junction": pl.Utf8,
        "junction_aa": pl.Utf8,
        "v_gene": pl.Utf8,
        "d_gene": pl.Utf8,
        "j_gene": pl.Utf8,
        "c_gene": pl.Utf8,
    }
    return pl.from_dicts(rows, schema=schema)


def test_impute_missing_chains_adds_synthetic_rows() -> None:
    raw = _cell_table(
        [
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "tra1",
                "locus": "TRA",
                "duplicate_count": 10,
                "umi_count": 5,
                "junction": "AAA",
                "junction_aa": "CAAAF",
                "v_gene": "TRAV1*01",
                "d_gene": "",
                "j_gene": "TRAJ1*01",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p2",
                "sequence_id": "trb1",
                "locus": "TRB",
                "duplicate_count": 9,
                "umi_count": 4,
                "junction": "BBB",
                "junction_aa": "CASSF",
                "v_gene": "TRBV1*01",
                "d_gene": "",
                "j_gene": "TRBJ1*01",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc3",
                "raw_pair_id": "p3",
                "sequence_id": "igh1",
                "locus": "IGH",
                "duplicate_count": 8,
                "umi_count": 3,
                "junction": "CCC",
                "junction_aa": "CARDR",
                "v_gene": "IGHV1*01",
                "d_gene": "",
                "j_gene": "IGHJ1*01",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc4",
                "raw_pair_id": "p4",
                "sequence_id": "igk1",
                "locus": "IGK",
                "duplicate_count": 7,
                "umi_count": 3,
                "junction": "DDD",
                "junction_aa": "CQQY",
                "v_gene": "IGKV1*01",
                "d_gene": "",
                "j_gene": "IGKJ1*01",
                "c_gene": "",
            },
        ]
    )

    imputed = impute_missing_chains(raw, seed=11)
    assert imputed.height == raw.height + 4

    synthetic = imputed.filter(pl.col("sequence_id").str.starts_with("synthetic_"))
    assert synthetic.height == 4
    assert set(synthetic["locus"].to_list()) == {"TRB", "TRA", "IGK", "IGH"}
    assert synthetic["duplicate_count"].to_list() == [1, 1, 1, 1]
    assert synthetic["umi_count"].to_list() == [1, 1, 1, 1]


def test_cleanup_cell_clonotypes_keeps_expected_chain_counts() -> None:
    raw = _cell_table(
        [
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "trb_top",
                "locus": "TRB",
                "duplicate_count": 100,
                "umi_count": 50,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "trb_low",
                "locus": "TRB",
                "duplicate_count": 20,
                "umi_count": 10,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "tra_top",
                "locus": "TRA",
                "duplicate_count": 100,
                "umi_count": 50,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "tra_second",
                "locus": "TRA",
                "duplicate_count": 12,
                "umi_count": 6,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "igh_top",
                "locus": "IGH",
                "duplicate_count": 100,
                "umi_count": 30,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "igk_top",
                "locus": "IGK",
                "duplicate_count": 100,
                "umi_count": 30,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "igl_low",
                "locus": "IGL",
                "duplicate_count": 9,
                "umi_count": 3,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
        ]
    )

    cleaned = cleanup_cell_clonotypes(raw)
    assert cleaned.filter(pl.col("locus") == "TRB").height == 1
    assert cleaned.filter(pl.col("locus") == "TRA").height == 2
    assert cleaned.filter(pl.col("locus") == "IGH").height == 1
    assert cleaned.filter(pl.col("locus").is_in(["IGK", "IGL"])).height == 1


def test_build_pairing_graph_counts_cells_per_edge() -> None:
    cell_df = _cell_table(
        [
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "tra1",
                "locus": "TRA",
                "duplicate_count": 10,
                "umi_count": 5,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "trb1",
                "locus": "TRB",
                "duplicate_count": 12,
                "umi_count": 6,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p1",
                "sequence_id": "tra1",
                "locus": "TRA",
                "duplicate_count": 8,
                "umi_count": 4,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p1",
                "sequence_id": "trb1",
                "locus": "TRB",
                "duplicate_count": 11,
                "umi_count": 5,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc3",
                "raw_pair_id": "p2",
                "sequence_id": "tra2",
                "locus": "TRA",
                "duplicate_count": 5,
                "umi_count": 3,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc3",
                "raw_pair_id": "p2",
                "sequence_id": "trb2",
                "locus": "TRB",
                "duplicate_count": 6,
                "umi_count": 3,
                "junction": "",
                "junction_aa": "",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
        ]
    )

    sample = build_tenx_sample_from_cell_clonotypes(cell_df, sample_id="s1")
    graph = build_pairing_graph(sample)

    assert graph.nodes.height == 4
    assert graph.edges.height == 2

    dominant = graph.edges.filter(
        (pl.col("source") == "TRA:tra1") & (pl.col("target") == "TRB:trb1")
    )
    assert dominant.height == 1
    assert dominant["cell_count"][0] == 2

    filtered = build_pairing_graph(sample, min_shared_cells=2)
    assert filtered.edges.height == 1


def test_cleanup_enforces_consistent_synthetic_slave_per_master() -> None:
    rows = _cell_table(
        [
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 50,
                "umi_count": 20,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "synthetic_TRA_A",
                "locus": "TRA",
                "duplicate_count": 1,
                "umi_count": 1,
                "junction": "",
                "junction_aa": "CAVRNNNARLMF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p2",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 45,
                "umi_count": 15,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p2",
                "sequence_id": "synthetic_TRA_B",
                "locus": "TRA",
                "duplicate_count": 1,
                "umi_count": 1,
                "junction": "",
                "junction_aa": "CAVRNNNARLMF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
        ]
    )

    cleaned = cleanup_cell_clonotypes(
        rows,
        enforce_consistent_slave_per_master=True,
        consistency_only_on_synthetic_slave=True,
    )
    tra_ids = sorted(cleaned.filter(pl.col("locus") == "TRA")["sequence_id"].to_list())
    assert len(set(tra_ids)) == 1


def test_cleanup_prunes_master_with_too_many_slave_edges() -> None:
    rows_list: list[dict[str, object]] = []
    for i in range(12):
        rows_list.append(
            {
                "sample_id": "s1",
                "barcode": f"bc{i}",
                "raw_pair_id": f"p{i}",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 30,
                "umi_count": 10,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            }
        )
        rows_list.append(
            {
                "sample_id": "s1",
                "barcode": f"bc{i}",
                "raw_pair_id": f"p{i}",
                "sequence_id": f"synthetic_TRA_{i}",
                "locus": "TRA",
                "duplicate_count": 1,
                "umi_count": 1,
                "junction": "",
                "junction_aa": "CAVRNNNARLMF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            }
        )

    rows = _cell_table(rows_list)
    pruned = cleanup_cell_clonotypes(rows, max_slave_edges_per_master=10)
    assert pruned.filter(pl.col("locus").is_in(["TRB", "TRA"])).height == 0


def test_impute_reuses_synthetic_slave_per_master_when_enabled() -> None:
    rows = _cell_table(
        [
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 20,
                "umi_count": 8,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p2",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 22,
                "umi_count": 9,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
        ]
    )

    imputed = impute_missing_chains(rows, seed=7, reuse_slave_per_master=True)
    tra_ids = sorted(imputed.filter(pl.col("locus") == "TRA")["sequence_id"].to_list())
    assert len(set(tra_ids)) == 1


def test_cleanup_consistency_can_apply_to_non_synthetic_slaves() -> None:
    rows = _cell_table(
        [
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 40,
                "umi_count": 14,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "tra_real_a",
                "locus": "TRA",
                "duplicate_count": 12,
                "umi_count": 6,
                "junction": "",
                "junction_aa": "CAVRNNNARLMF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p2",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 35,
                "umi_count": 12,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
            {
                "sample_id": "s1",
                "barcode": "bc2",
                "raw_pair_id": "p2",
                "sequence_id": "tra_real_b",
                "locus": "TRA",
                "duplicate_count": 11,
                "umi_count": 5,
                "junction": "",
                "junction_aa": "CAVRNNNARLMF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            },
        ]
    )

    cleaned = cleanup_cell_clonotypes(
        rows,
        enforce_consistent_slave_per_master=True,
        consistency_only_on_synthetic_slave=False,
    )
    tra_ids = cleaned.filter(pl.col("locus") == "TRA")["sequence_id"].to_list()
    assert len(set(tra_ids)) == 1


def test_cleanup_rejects_non_positive_max_slave_edges() -> None:
    rows = _cell_table(
        [
            {
                "sample_id": "s1",
                "barcode": "bc1",
                "raw_pair_id": "p1",
                "sequence_id": "trb_master",
                "locus": "TRB",
                "duplicate_count": 20,
                "umi_count": 8,
                "junction": "",
                "junction_aa": "CASSQETQYF",
                "v_gene": "",
                "d_gene": "",
                "j_gene": "",
                "c_gene": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="must be positive"):
        cleanup_cell_clonotypes(rows, max_slave_edges_per_master=0)
