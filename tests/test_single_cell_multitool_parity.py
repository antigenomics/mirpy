"""Integration test comparing mirpy, scirpy, and dandelion single-cell multiplicities.

This regression test locks the discrepancy pattern observed on AIRR benchmark
10x donor files: mirpy's consensus-linked pairing is stricter than raw all-contig
views, while scirpy and dandelion retain more multi-TRA / multi-TRB cells.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mir.common.single_cell import load_10x_vdj_v1_donor
from mir.common.single_cell_parser import load_10x_vdj_v1_cell_clonotypes
from tests.prepare_airr_benchmark_data import ensure_test_data


def _discover_donor_pair() -> tuple[Path, Path] | None:
    dcode_root = Path(__file__).resolve().parents[1] / "airr_benchmark" / "dcode"
    if not dcode_root.exists():
        return None

    all_contig_files = sorted(dcode_root.glob("*_all_contig_annotations.csv.gz"))
    consensus_files = sorted(dcode_root.glob("*_consensus_annotations.csv.gz"))
    if not all_contig_files or not consensus_files:
        return None

    all_contig = all_contig_files[0]
    consensus = next(
        (
            p
            for p in consensus_files
            if p.name.replace("_consensus_annotations", "")
            == all_contig.name.replace("_all_contig_annotations", "")
        ),
        None,
    )
    if consensus is None:
        return None
    return all_contig, consensus


def _barcode_multiplicity_from_mirpy(df: pl.DataFrame) -> dict[tuple[str, str], int]:
    counts = (
        df.filter(pl.col("locus").is_in(["TRA", "TRB"]))
        .group_by(["barcode", "locus"])
        .agg(pl.n_unique("sequence_id").alias("n"))
        .pivot(index="barcode", on="locus", values="n", aggregate_function="first")
        .with_columns(
            pl.col("TRA").fill_null(0).cast(pl.Int64),
            pl.col("TRB").fill_null(0).cast(pl.Int64),
        )
        .with_columns(
            pl.when(pl.col("TRA") > 0).then(pl.lit("TRA+")).otherwise(pl.lit("TRA-")).alias("tra"),
            pl.when(pl.col("TRB") > 0).then(pl.lit("TRB+")).otherwise(pl.lit("TRB-")).alias("trb"),
        )
        .group_by(["tra", "trb"])
        .len()
        .rename({"len": "cells"})
    )
    return {(row["tra"], row["trb"]): int(row["cells"]) for row in counts.to_dicts()}


def _multi_chain_counts_from_mirpy(df: pl.DataFrame) -> dict[str, int]:
    counts = df.filter(pl.col("locus").is_in(["TRA", "TRB"]))
    return {
        "multi_tra": (
            counts.group_by("barcode")
            .agg(pl.col("sequence_id").filter(pl.col("locus") == "TRA").n_unique().alias("tra"))
            .filter(pl.col("tra") >= 2)
            .height
        ),
        "multi_trb": (
            counts.group_by("barcode")
            .agg(pl.col("sequence_id").filter(pl.col("locus") == "TRB").n_unique().alias("trb"))
            .filter(pl.col("trb") >= 2)
            .height
        ),
        "multi_both": (
            counts.group_by("barcode")
            .agg(
                pl.col("sequence_id").filter(pl.col("locus") == "TRA").n_unique().alias("tra"),
                pl.col("sequence_id").filter(pl.col("locus") == "TRB").n_unique().alias("trb"),
            )
            .filter((pl.col("tra") >= 2) & (pl.col("trb") >= 2))
            .height
        ),
    }


def _multi_chain_counts_from_raw_all_contig(all_contig: Path) -> dict[str, int]:
    raw = pl.read_csv(all_contig)
    if "is_cell" in raw.columns:
        raw = raw.filter(pl.col("is_cell").cast(pl.Utf8).str.to_lowercase().is_in(["true", "1", "t", "yes", "y"]))

    counts = (
        raw.filter(pl.col("chain").is_in(["TRA", "TRB"]))
        .group_by(["barcode", "chain"])
        .agg(pl.n_unique("raw_consensus_id").alias("n"))
    )
    return {
        "multi_tra": (
            counts.filter(pl.col("chain") == "TRA")
            .group_by("barcode")
            .agg(pl.col("n").max().alias("tra"))
            .filter(pl.col("tra") >= 2)
            .height
        ),
        "multi_trb": (
            counts.filter(pl.col("chain") == "TRB")
            .group_by("barcode")
            .agg(pl.col("n").max().alias("trb"))
            .filter(pl.col("trb") >= 2)
            .height
        ),
        "multi_both": (
            counts.group_by("barcode")
            .agg(
                pl.col("n").filter(pl.col("chain") == "TRA").max().alias("tra"),
                pl.col("n").filter(pl.col("chain") == "TRB").max().alias("trb"),
            )
            .with_columns(pl.col("tra").fill_null(0), pl.col("trb").fill_null(0))
            .filter((pl.col("tra") >= 2) & (pl.col("trb") >= 2))
            .height
        ),
    }


def _multi_chain_counts_from_scirpy(all_contig: Path, *, filtered: bool) -> dict[str, int]:
    scirpy = pytest.importorskip("scirpy")
    _ = pytest.importorskip("scanpy")
    _ = pytest.importorskip("awkward")
    import awkward as ak

    adata = scirpy.io.read_10x_vdj(all_contig, filtered=filtered)
    airr = adata.obsm["airr"]
    n_tra = ak.to_numpy(ak.sum(airr.locus == "TRA", axis=1))
    n_trb = ak.to_numpy(ak.sum(airr.locus == "TRB", axis=1))
    df = pl.DataFrame({"tra": n_tra, "trb": n_trb})
    return {
        "multi_tra": df.filter(pl.col("tra") >= 2).height,
        "multi_trb": df.filter(pl.col("trb") >= 2).height,
        "multi_both": df.filter((pl.col("tra") >= 2) & (pl.col("trb") >= 2)).height,
    }


def _multi_chain_counts_from_dandelion(all_contig: Path) -> dict[str, int]:
    import os
    import warnings

    os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0")
    ddl = pytest.importorskip("dandelion")
    pandas = pytest.importorskip("pandas")

    df = pandas.read_csv(all_contig)
    df = df[df["is_cell"] == True].copy()  # noqa: E712
    df = df.rename(
        columns={
            "contig_id": "sequence_id",
            "barcode": "cell_id",
            "chain": "locus",
            "cdr3": "junction_aa",
            "cdr3_nt": "junction",
            "v_gene": "v_call",
            "d_gene": "d_call",
            "j_gene": "j_call",
            "c_gene": "c_call",
            "reads": "consensus_count",
            "umis": "umi_count",
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vdj = ddl.Dandelion(data=df)

    meta = vdj.metadata
    locus_vj = meta["locus_VJ"].fillna("").astype(str)
    locus_vdj = meta["locus_VDJ"].fillna("").astype(str)

    return {
        "multi_tra": (locus_vj.str.count("TRA") >= 2).sum(),
        "multi_trb": (locus_vdj.str.count("TRB") >= 2).sum(),
        "multi_both": ((locus_vj.str.contains("TRA")) & (locus_vdj.str.contains("TRB"))).sum(),
    }


@pytest.mark.integration
def test_multi_chain_discrepancy_is_consistent_across_loaders() -> None:
    """Track the known multi-chain gap between mirrored loaders on one donor.

    mirpy is stricter because it requires consensus-linked pairing rows and,
    optionally, is_cell filtering. Raw all-contig, scirpy, and dandelion all
    retain more multi-TRA / multi-TRB evidence on this benchmark donor.
    """
    ensure_test_data(force=False, verbose=False)
    donor_files = _discover_donor_pair()
    if donor_files is None:
        pytest.skip("No local dcode/10x_vdj_v1 donor assets found")

    all_contig, consensus = donor_files

    mirpy_cell_rows = load_10x_vdj_v1_cell_clonotypes(
        consensus_annotations_path=consensus,
        all_contig_annotations_path=all_contig,
        sample_id=all_contig.name,
        check_is_cell=True,
    )
    mirpy_quads = _barcode_multiplicity_from_mirpy(mirpy_cell_rows)
    mirpy_multi = _multi_chain_counts_from_mirpy(mirpy_cell_rows)
    raw_multi = _multi_chain_counts_from_raw_all_contig(all_contig)
    scirpy_false = _multi_chain_counts_from_scirpy(all_contig, filtered=False)
    scirpy_true = _multi_chain_counts_from_scirpy(all_contig, filtered=True)
    dandelion_multi = _multi_chain_counts_from_dandelion(all_contig)

    donor = load_10x_vdj_v1_donor(consensus, all_contig, donor_id="probe")
    assert donor.sample_id == "probe"
    assert donor.to_sample_repertoire().sample_id == "probe"

    # Mirpy should retain the major TRA/TRB-positive population but be stricter
    # than the raw all-contig, scirpy, and dandelion views.
    assert mirpy_quads.get(("TRA+", "TRB+"), 0) > 0
    assert raw_multi["multi_both"] > mirpy_multi["multi_both"]
    assert scirpy_false["multi_both"] > mirpy_multi["multi_both"]
    assert dandelion_multi["multi_both"] > mirpy_multi["multi_both"]

    # filtered=True scirpy removes some cells, but still keeps more multi-chain
    # evidence than the consensus-linked mirpy loader for this donor.
    assert scirpy_true["multi_both"] >= mirpy_multi["multi_both"]

    # The consensus-linked loader should never exceed the raw all-contig pool.
    assert mirpy_multi["multi_tra"] < raw_multi["multi_tra"]
    assert mirpy_multi["multi_trb"] < raw_multi["multi_trb"]

    # Keep the relative gap visible: on this donor, mirpy is much stricter than
    # scirpy filtered=False for the TRA+/TRB+ quadrant.
    assert scirpy_false["multi_both"] >= scirpy_true["multi_both"]
