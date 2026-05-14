"""Tests for paired single-cell repertoire structures and 10x_vdj_v1 loading."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from mir.common.single_cell import LOCUS_PAIR_TO_LOCI, load_10x_vdj_v1_donor
from tests.prepare_airr_benchmark_data import ensure_test_data


def _write_csv_gz(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(df.write_csv(separator=","))


def test_multi_tra_single_trb_expands_to_two_pairs(tmp_path: Path) -> None:
    consensus = pl.DataFrame(
        {
            "clonotype_id": ["pairA", "pairA", "pairA"],
            "consensus_id": ["consA1", "consA2", "consB1"],
            "chain": ["TRA", "TRA", "TRB"],
            "cdr3_nt": ["AAA", "CCC", "GGG"],
            "cdr3_aa": ["CAAAF", "CCAAF", "CGGGF"],
            "reads": [10, 8, 15],
            "umis": [5, 4, 7],
        }
    )
    all_contig = pl.DataFrame(
        {
            "is_cell": [True, True, True],
            "barcode": ["bc1", "bc1", "bc1"],
            "raw_clonotype_id": ["pairA", "pairA", "pairA"],
            "raw_consensus_id": ["consA1", "consA2", "consB1"],
        }
    )

    consensus_path = tmp_path / "consensus.csv.gz"
    all_contig_path = tmp_path / "all_contig.csv.gz"
    _write_csv_gz(consensus_path, consensus)
    _write_csv_gz(all_contig_path, all_contig)

    donor = load_10x_vdj_v1_donor(consensus_path, all_contig_path, donor_id="d1")

    tra_trb = donor.paired_locus_repertoires["TRA_TRB"]
    assert tra_trb.clonotype_count == 2
    assert [p.pair_id for p in tra_trb.paired_clonotypes] == ["pairA_1", "pairA_2"]

    links = donor.single_cell_repertoire.to_polars().sort(["barcode", "pair_id"])
    assert links.height == 2
    assert links["barcode"].to_list() == ["bc1", "bc1"]
    assert links["pair_id"].to_list() == ["pairA_1", "pairA_2"]


def test_incomplete_pairs_counted_but_not_emitted(tmp_path: Path) -> None:
    consensus = pl.DataFrame(
        {
            "clonotype_id": ["pairX"],
            "consensus_id": ["consX1"],
            "chain": ["TRA"],
            "cdr3_nt": ["AAA"],
            "cdr3_aa": ["CAAAF"],
            "reads": [7],
            "umis": [3],
        }
    )
    all_contig = pl.DataFrame(
        {
            "is_cell": [True],
            "barcode": ["bcX"],
            "raw_clonotype_id": ["pairX"],
            "raw_consensus_id": ["consX1"],
        }
    )

    consensus_path = tmp_path / "consensus.csv.gz"
    all_contig_path = tmp_path / "all_contig.csv.gz"
    _write_csv_gz(consensus_path, consensus)
    _write_csv_gz(all_contig_path, all_contig)

    donor = load_10x_vdj_v1_donor(consensus_path, all_contig_path, donor_id="d2")

    assert donor.paired_locus_repertoires["TRA_TRB"].clonotype_count == 0
    assert donor.single_cell_repertoire.to_polars().height == 0

    counts = donor.chain_multiplicity.filter(
        (pl.col("locus_pair") == "TRA_TRB")
        & (pl.col("n_chain1") == 1)
        & (pl.col("m_chain2") == 0)
    )
    assert counts.height == 1
    assert counts["cell_count"][0] == 1


def test_mixed_locus_pair_routed_to_correct_family(tmp_path: Path) -> None:
    consensus = pl.DataFrame(
        {
            "clonotype_id": ["pairB", "pairB"],
            "consensus_id": ["consH", "consK"],
            "chain": ["IGH", "IGK"],
            "cdr3_nt": ["AAA", "CCC"],
            "cdr3_aa": ["CHHHF", "CKKKF"],
            "reads": [21, 17],
            "umis": [11, 9],
        }
    )
    all_contig = pl.DataFrame(
        {
            "is_cell": [True, True],
            "barcode": ["bcB", "bcB"],
            "raw_clonotype_id": ["pairB", "pairB"],
            "raw_consensus_id": ["consH", "consK"],
        }
    )

    consensus_path = tmp_path / "consensus.csv.gz"
    all_contig_path = tmp_path / "all_contig.csv.gz"
    _write_csv_gz(consensus_path, consensus)
    _write_csv_gz(all_contig_path, all_contig)

    donor = load_10x_vdj_v1_donor(consensus_path, all_contig_path, donor_id="d3")

    assert donor.paired_locus_repertoires["IGH_IGK"].clonotype_count == 1
    assert donor.paired_locus_repertoires["TRA_TRB"].clonotype_count == 0


def test_locus_pair_registry_is_stable() -> None:
    assert set(LOCUS_PAIR_TO_LOCI) == {"TRA_TRB", "TRG_TRD", "IGH_IGK", "IGH_IGL"}


@pytest.mark.integration
def test_load_10x_vdj_v1_from_airr_benchmark_donor1_like() -> None:
    ensure_test_data(force=False, verbose=False)
    dcode_root = Path(__file__).resolve().parents[1] / "airr_benchmark" / "dcode"
    if not dcode_root.exists():
        pytest.skip("dcode assets not found in local airr_benchmark")

    all_contig_files = sorted(dcode_root.glob("*_all_contig_annotations.csv.gz"))
    consensus_files = sorted(dcode_root.glob("*_consensus_annotations.csv.gz"))
    if not all_contig_files or not consensus_files:
        pytest.skip("No donor all_contig/consensus files found under dcode")

    # Prefer donor 1 naming variants when present; otherwise fallback to first donor.
    donor1_all = next((p for p in all_contig_files if "donor1" in p.name or "_1_" in p.name), all_contig_files[0])
    donor1_consensus = next(
        (
            p
            for p in consensus_files
            if p.name.replace("_consensus_annotations", "")
            == donor1_all.name.replace("_all_contig_annotations", "")
        ),
        None,
    )
    if donor1_consensus is None:
        pytest.skip("Could not find matching consensus/all_contig donor pair")

    donor = load_10x_vdj_v1_donor(
        consensus_annotations_path=donor1_consensus,
        all_contig_annotations_path=donor1_all,
    )

    assert donor.chain_multiplicity.height > 0
    assert set(donor.chain_multiplicity.columns) == {
        "donor_id",
        "locus_pair",
        "n_chain1",
        "m_chain2",
        "cell_count",
    }
