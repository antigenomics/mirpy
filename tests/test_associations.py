from __future__ import annotations

import polars as pl
import pytest
from scipy.stats import fisher_exact

from mir.biomarkers.associations import (
    AssociationParams,
    associate_clonotype_cooccurrence,
    associate_clonotype_metadata,
    associate_paired_clonotype_metadata,
    build_public_clonotype_panel,
)
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.single_cell import PairedClonotype, PairedLocusRepertoire, PairedRepertoire, SingleCellRepertoire
from tests.factories import make_trb_clone


def _sample(
    sample_id: str,
    rows: list[tuple[str, str]],
    *,
    metadata: dict,
    v: str = "TRBV5-1*01",
    j: str = "TRBJ2-7*01",
) -> SampleRepertoire:
    clones = [make_trb_clone(sid, aa, v=v, j=j) for sid, aa in rows]
    return SampleRepertoire(
        loci={"TRB": LocusRepertoire(clones, locus="TRB", repertoire_id=sample_id)},
        sample_id=sample_id,
        sample_metadata=metadata,
    )


def _paired_sample(
    sample_id: str,
    pairs: list[PairedClonotype],
) -> PairedRepertoire:
    return PairedRepertoire(
        sample_id=sample_id,
        single_cell_repertoire=SingleCellRepertoire(barcode_pair_ids=[]),
        paired_locus_repertoires={"TRA_TRB": PairedLocusRepertoire(locus_pair="TRA_TRB", paired_clonotypes=pairs)},
        chain_multiplicity=pl.DataFrame(),
        loaded_cell_count=0,
        loaded_clonotype_count=len(pairs),
    )


def test_binary_sample_association_uses_fisher() -> None:
    target = make_trb_clone("target", "CASSLGQETQYF")
    samples = [
        _sample("s1", [("a", "CASSLGQETQYF")], metadata={"condition": "positive"}),
        _sample("s2", [("b", "CASSLGQETQYF")], metadata={"condition": "positive"}),
        _sample("s3", [("c", "CASSPGQETQYF")], metadata={"condition": "negative"}),
        _sample("s4", [("d", "CASSPGQETQYF")], metadata={"condition": "negative"}),
    ]

    result = associate_clonotype_metadata(
        samples,
        [target],
        metadata_field="condition",
        params=AssociationParams(match_mode="vj", count_mode="sample"),
    )

    row = result.table.row(0, named=True)
    assert row["levels"] == ["negative", "positive"]
    assert row["detected_counts"] == [0, 2]
    assert row["background_counts"] == [2, 0]
    assert row["test"] == "fisher"
    expected_p = fisher_exact([[0, 2], [2, 0]], alternative="two-sided")[1]
    assert float(row["p_value"]) == pytest.approx(float(expected_p))


def test_binary_label_presence_supports_comma_separated_hla() -> None:
    target = make_trb_clone("target", "CASSLGQETQYF")
    samples = [
        _sample("s1", [("a", "CASSLGQETQYF")], metadata={"hlas": "HLA-A*02,HLA-B*07"}),
        _sample("s2", [("b", "CASSLGQETQYF")], metadata={"hlas": ["HLA-A*02", "HLA-C*02"]}),
        _sample("s3", [("c", "CASSPGQETQYF")], metadata={"hlas": "HLA-B*08"}),
    ]

    result = associate_clonotype_metadata(
        samples,
        [target],
        metadata_field="hlas",
        metadata_value="HLA-A*02",
    )

    row = result.table.row(0, named=True)
    assert row["levels"] == ["HLA-A*02", "not_HLA-A*02"]
    assert row["detected_counts"] == [2, 0]


def test_multiclass_association_builds_one_vs_rest_contrasts() -> None:
    target = make_trb_clone("target", "CASSLGQETQYF")
    samples = [
        _sample("s1", [("a", "CASSLGQETQYF")], metadata={"condition": "convalescent"}),
        _sample("s2", [("b", "CASSLGQETQYF")], metadata={"condition": "severe"}),
        _sample("s3", [("c", "CASSPGQETQYF")], metadata={"condition": "healthy"}),
        _sample("s4", [("d", "CASSPGQETQYF")], metadata={"condition": "healthy"}),
    ]

    result = associate_clonotype_metadata(samples, [target], metadata_field="condition")

    summary = result.table.row(0, named=True)
    assert summary["test"] == "chi2"
    assert summary["levels"] == ["convalescent", "healthy", "severe"]
    assert result.contrast_table.height == 3
    severe = result.contrast_table.filter(pl.col("level") == "severe").row(0, named=True)
    assert severe["detected_in_level"] == 1
    assert severe["background_in_level"] == 0


def test_rearrangement_count_mode_uses_row_counts() -> None:
    target = make_trb_clone("target", "CASSLGQETQYF")
    params = AssociationParams(count_mode="rearrangement", max_distance=0)
    samples = [
        _sample(
            "s1",
            [("a", "CASSLGQETQYF"), ("b", "CASSLGQETQFF"), ("c", "CASSPGQETQYF")],
            metadata={"condition": "positive"},
        ),
        _sample(
            "s2",
            [("d", "CASSPGQETQYF"), ("e", "CASSPGQETQFF")],
            metadata={"condition": "negative"},
        ),
    ]

    result = associate_clonotype_metadata(samples, [target], metadata_field="condition", params=params)

    row = result.table.row(0, named=True)
    assert row["detected_counts"] == [0, 1]
    assert row["background_counts"] == [2, 2]


def test_cooccurrence_counts_both_only_and_neither() -> None:
    left = make_trb_clone("left", "CASSLGQETQYF")
    right = make_trb_clone("right", "CASSPGQETQYF")
    samples = [
        _sample("s1", [("a", "CASSLGQETQYF"), ("b", "CASSPGQETQYF")], metadata={"condition": "x"}),
        _sample("s2", [("c", "CASSLGQETQYF")], metadata={"condition": "x"}),
        _sample("s3", [("d", "CASSPGQETQYF")], metadata={"condition": "x"}),
        _sample("s4", [("e", "CASSQGQETQYF")], metadata={"condition": "x"}),
    ]

    result = associate_clonotype_cooccurrence(samples, [left], [right])
    row = result.table.row(0, named=True)

    assert row["both"] == 1
    assert row["left_only"] == 1
    assert row["right_only"] == 1
    assert row["neither"] == 1


def test_paired_association_matches_both_chains() -> None:
    tra_target = make_trb_clone("tra", "CAVRNTGNQFYF", v="TRAV1-2*01", j="TRAJ33*01")
    tra_target.locus = "TRA"
    trb_target = make_trb_clone("trb", "CASSLGQETQYF")
    target = PairedClonotype(pair_id="pair", clonotype1=tra_target, clonotype2=trb_target)
    matching_pair = PairedClonotype(pair_id="p1", clonotype1=tra_target, clonotype2=trb_target)
    non_matching_pair = PairedClonotype(
        pair_id="p2",
        clonotype1=tra_target,
        clonotype2=make_trb_clone("x", "CASSPGQETQYF"),
    )
    samples = [
        _paired_sample("s1", [matching_pair]),
        _paired_sample("s2", [matching_pair]),
        _paired_sample("s3", [non_matching_pair]),
    ]
    metadata = [
        {"condition": "positive"},
        {"condition": "positive"},
        {"condition": "negative"},
    ]

    result = associate_paired_clonotype_metadata(
        samples,
        [target],
        sample_metadata=metadata,
        metadata_field="condition",
    )

    row = result.table.row(0, named=True)
    assert row["detected_counts"] == [0, 2]
    assert row["background_counts"] == [1, 0]


def test_build_public_clonotype_panel_filters_by_sample_fraction() -> None:
    samples = [
        _sample("s1", [("a", "CASSLGQETQYF"), ("b", "CASSLGQETQFF")], metadata={"condition": "x"}),
        _sample("s2", [("c", "CASSLGQETQYF")], metadata={"condition": "x"}),
        _sample("s3", [("d", "CASSLGQETQFF")], metadata={"condition": "x"}),
    ]

    panel = build_public_clonotype_panel(samples, locus="TRB", min_sample_fraction=0.66)
    seqs = sorted(clone.junction_aa for clone in panel)
    assert seqs == ["CASSLGQETQFF", "CASSLGQETQYF"]