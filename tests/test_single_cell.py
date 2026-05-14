"""Tests for paired single-cell repertoire structures and 10x_vdj_v1 loading."""

from __future__ import annotations

import gzip
import warnings
from pathlib import Path

import polars as pl
import pytest

from mir.common.clonotype import Clonotype
from mir.common.single_cell import (
    LOCUS_PAIR_TO_LOCI,
    PairedClonotype,
    PairedLocusRepertoire,
    SingleCellRepertoire,
    TenXVdjV1DonorData,
    load_10x_vdj_v1_donor,
)
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


# ---------------------------------------------------------------------------
# Object-construction unit tests
# ---------------------------------------------------------------------------


def _make_clonotype(seq_id: str = "c1", locus: str = "TRA", junction_aa: str = "CAASTGTF") -> Clonotype:
    return Clonotype(
        sequence_id=seq_id,
        locus=locus,
        junction_aa=junction_aa,
    )


def test_paired_clonotype_construction() -> None:
    c1 = _make_clonotype("tra1", "TRA", "CAASTGTF")
    c2 = _make_clonotype("trb1", "TRB", "CASSGQPF")
    pc = PairedClonotype(pair_id="p1", clonotype1=c1, clonotype2=c2)
    assert pc.pair_id == "p1"
    assert pc.clonotype1 is c1
    assert pc.clonotype2 is c2


def test_paired_locus_repertoire_construction_and_count() -> None:
    c1 = _make_clonotype("tra1", "TRA")
    c2 = _make_clonotype("trb1", "TRB")
    pc = PairedClonotype(pair_id="p1", clonotype1=c1, clonotype2=c2)
    repo = PairedLocusRepertoire(locus_pair="TRA_TRB", paired_clonotypes=[pc])
    assert repo.clonotype_count == 1
    assert repo.paired_clonotypes[0].pair_id == "p1"


def test_paired_locus_repertoire_rejects_invalid_locus_pair() -> None:
    with pytest.raises(ValueError, match="Unsupported locus_pair"):
        PairedLocusRepertoire(locus_pair="TRA_IGH", paired_clonotypes=[])


def test_single_cell_repertoire_to_polars_schema() -> None:
    scr = SingleCellRepertoire(barcode_pair_ids=[("bc1", "p1"), ("bc2", "p2")])
    df = scr.to_polars()
    assert set(df.columns) == {"barcode", "pair_id"}
    assert df.height == 2
    assert df["barcode"].to_list() == ["bc1", "bc2"]
    assert df["pair_id"].to_list() == ["p1", "p2"]


def test_tenx_vdj_v1_donor_data_construction() -> None:
    c1 = _make_clonotype("tra1", "TRA")
    c2 = _make_clonotype("trb1", "TRB")
    pc = PairedClonotype(pair_id="p1", clonotype1=c1, clonotype2=c2)
    repos = {
        name: PairedLocusRepertoire(locus_pair=name, paired_clonotypes=[])
        for name in LOCUS_PAIR_TO_LOCI
    }
    repos["TRA_TRB"] = PairedLocusRepertoire(locus_pair="TRA_TRB", paired_clonotypes=[pc])
    scr = SingleCellRepertoire(barcode_pair_ids=[("bc1", "p1")])
    cm = pl.DataFrame({
        "donor_id": ["d1"], "locus_pair": ["TRA_TRB"],
        "n_chain1": [1], "m_chain2": [1], "cell_count": [1],
    })
    d = TenXVdjV1DonorData(
        donor_id="d1",
        single_cell_repertoire=scr,
        paired_locus_repertoires=repos,
        chain_multiplicity=cm,
        loaded_cell_count=1,
        loaded_clonotype_count=1,
    )
    assert d.donor_id == "d1"
    assert d.loaded_cell_count == 1
    assert d.paired_locus_repertoires["TRA_TRB"].clonotype_count == 1


def test_donor_id_defaults_to_consensus_filename(tmp_path: Path) -> None:
    consensus = pl.DataFrame({
        "clonotype_id": ["pairA", "pairA"],
        "consensus_id": ["consA_TRA", "consA_TRB"],
        "chain": ["TRA", "TRB"],
        "cdr3_nt": ["AAA", "GGG"],
        "cdr3_aa": ["CAAAF", "CGGGF"],
        "reads": [10, 15],
        "umis": [5, 7],
    })
    all_contig = pl.DataFrame({
        "is_cell": [True, True],
        "barcode": ["bc1", "bc1"],
        "raw_clonotype_id": ["pairA", "pairA"],
        "raw_consensus_id": ["consA_TRA", "consA_TRB"],
        "cdr3_nt": ["AAA", "GGG"],
    })
    cp = tmp_path / "mydonor_consensus.csv.gz"
    ap = tmp_path / "mydonor_all_contig.csv.gz"
    _write_csv_gz(cp, consensus)
    _write_csv_gz(ap, all_contig)
    donor = load_10x_vdj_v1_donor(cp, ap)
    assert donor.donor_id == cp.name


# ---------------------------------------------------------------------------
# Safety / erroneous-input tests
# ---------------------------------------------------------------------------


def test_non_cell_rows_are_excluded(tmp_path: Path) -> None:
    consensus = pl.DataFrame({
        "clonotype_id": ["pairA", "pairA"],
        "consensus_id": ["consA_TRA", "consA_TRB"],
        "chain": ["TRA", "TRB"],
        "cdr3_nt": ["AAA", "GGG"],
        "cdr3_aa": ["CAAAF", "CGGGF"],
        "reads": [10, 15],
        "umis": [5, 7],
    })
    all_contig = pl.DataFrame({
        "is_cell": [False, False],  # all non-cell
        "barcode": ["bc1", "bc1"],
        "raw_clonotype_id": ["pairA", "pairA"],
        "raw_consensus_id": ["consA_TRA", "consA_TRB"],
        "cdr3_nt": ["AAA", "GGG"],
    })
    cp = tmp_path / "consensus.csv.gz"
    ap = tmp_path / "all_contig.csv.gz"
    _write_csv_gz(cp, consensus)
    _write_csv_gz(ap, all_contig)
    donor = load_10x_vdj_v1_donor(cp, ap, donor_id="d_noncell")
    assert donor.loaded_cell_count == 0
    assert donor.paired_locus_repertoires["TRA_TRB"].clonotype_count == 0


def test_all_contig_missing_consensus_reference_is_skipped(tmp_path: Path) -> None:
    """Rows referencing a consensus_id absent in consensus file are silently skipped."""
    consensus = pl.DataFrame({
        "clonotype_id": ["pairA"],
        "consensus_id": ["consA_TRA"],
        "chain": ["TRA"],
        "cdr3_nt": ["AAA"],
        "cdr3_aa": ["CAAAF"],
        "reads": [10],
        "umis": [5],
    })
    all_contig = pl.DataFrame({
        "is_cell": [True, True],
        "barcode": ["bc1", "bc1"],
        "raw_clonotype_id": ["pairA", "pairA"],
        "raw_consensus_id": ["consA_TRA", "GHOST_CONSENSUS"],  # second doesn't exist
        "cdr3_nt": ["AAA", "TTT"],
    })
    cp = tmp_path / "consensus.csv.gz"
    ap = tmp_path / "all_contig.csv.gz"
    _write_csv_gz(cp, consensus)
    _write_csv_gz(ap, all_contig)
    donor = load_10x_vdj_v1_donor(cp, ap, donor_id="d_ghost")
    # TRA-only → no complete pair, but no crash
    assert donor.paired_locus_repertoires["TRA_TRB"].clonotype_count == 0


def test_donor_mismatch_warning_emitted(tmp_path: Path) -> None:
    """Loading files from different donors (junction conflicts) should raise UserWarning."""
    consensus = pl.DataFrame({
        "clonotype_id": ["pairA", "pairA"],
        "consensus_id": ["consA_TRA", "consA_TRB"],
        "chain": ["TRA", "TRB"],
        "cdr3_nt": ["AAA", "GGG"],
        "cdr3_aa": ["CAAAF", "CGGGF"],
        "reads": [10, 15],
        "umis": [5, 7],
    })
    # all_contig references same consensus IDs but different junctions — different donor
    all_contig = pl.DataFrame({
        "is_cell": [True, True],
        "barcode": ["bc2", "bc2"],
        "raw_clonotype_id": ["pairA", "pairA"],
        "raw_consensus_id": ["consA_TRA", "consA_TRB"],
        "cdr3_nt": ["TTTTTT", "CCCCCC"],
    })
    cp = tmp_path / "consensus.csv.gz"
    ap = tmp_path / "all_contig.csv.gz"
    _write_csv_gz(cp, consensus)
    _write_csv_gz(ap, all_contig)
    with pytest.warns(UserWarning, match="Possible donor mismatch"):
        load_10x_vdj_v1_donor(cp, ap, donor_id="d_mismatch")


def test_no_donor_mismatch_warning_when_junctions_agree(tmp_path: Path) -> None:
    """No warning should be emitted when junctions match across both files."""
    consensus = pl.DataFrame({
        "clonotype_id": ["pairA", "pairA"],
        "consensus_id": ["consA_TRA", "consA_TRB"],
        "chain": ["TRA", "TRB"],
        "cdr3_nt": ["AAA", "GGG"],
        "cdr3_aa": ["CAAAF", "CGGGF"],
        "reads": [10, 15],
        "umis": [5, 7],
    })
    all_contig = pl.DataFrame({
        "is_cell": [True, True],
        "barcode": ["bc1", "bc1"],
        "raw_clonotype_id": ["pairA", "pairA"],
        "raw_consensus_id": ["consA_TRA", "consA_TRB"],
        "cdr3_nt": ["AAA", "GGG"],  # junctions match
    })
    cp = tmp_path / "consensus.csv.gz"
    ap = tmp_path / "all_contig.csv.gz"
    _write_csv_gz(cp, consensus)
    _write_csv_gz(ap, all_contig)
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        donor = load_10x_vdj_v1_donor(cp, ap, donor_id="d_ok")
    assert donor.paired_locus_repertoires["TRA_TRB"].clonotype_count == 1
