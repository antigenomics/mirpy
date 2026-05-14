"""Sanity tests for single-cell repair and pairing graph utilities."""

from __future__ import annotations

import polars as pl

from mir.common.single_cell import build_tenx_donor_from_cell_clonotypes
from mir.common.single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains
from mir.common.single_cell_util import build_pairing_graph


def _cell_table(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "donor_id": pl.Utf8,
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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
                "donor_id": "d1",
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

    donor = build_tenx_donor_from_cell_clonotypes(cell_df, donor_id="d1")
    graph = build_pairing_graph(donor)

    assert graph.nodes.height == 4
    assert graph.edges.height == 2

    dominant = graph.edges.filter(
        (pl.col("source") == "TRA:tra1") & (pl.col("target") == "TRB:trb1")
    )
    assert dominant.height == 1
    assert dominant["cell_count"][0] == 2

    filtered = build_pairing_graph(donor, min_shared_cells=2)
    assert filtered.edges.height == 1
