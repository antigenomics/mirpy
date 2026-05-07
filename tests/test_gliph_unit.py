from __future__ import annotations

from mir.biomarkers.gliph import (
    deduplicate_clonotype_rows,
    extract_u4mer_artifacts,
    extract_v3mer_artifacts,
    extract_vpos3mer_artifacts,
)
import pandas as pd


def test_deduplicate_clonotype_rows_sums_duplicate_count() -> None:
    df = pd.DataFrame(
        {
            "reference_id": ["study", "study", "study"],
            "junction_aa": ["AAAA", "AAAA", "CCCC"],
            "v_gene": ["TRBV1*01", "TRBV1*01", "TRBV2*01"],
            "j_gene": ["TRBJ1-1*01", "TRBJ1-1*01", "TRBJ2-1*01"],
            "duplicate_count": [2, 3, 1],
            "stimulus": ["x", "x", "y"],
            "row_id": ["0", "1", "2"],
        }
    )

    dedup = deduplicate_clonotype_rows(df)

    assert len(dedup) == 2
    aaaa = dedup[dedup["junction_aa"] == "AAAA"].iloc[0]
    assert int(aaaa["duplicate_count"]) == 5


def test_extract_v3mer_artifacts_supports_clonotype_count_mode() -> None:
    df = pd.DataFrame(
        {
            "reference_id": ["study", "study"],
            "junction_aa": ["AAAA", "AAAA"],
            "v_gene": ["TRBV1*01", "TRBV1*01"],
            "j_gene": ["TRBJ1-1*01", "TRBJ1-1*01"],
            "duplicate_count": [1, 1],
            "row_id": ["0", "1"],
        }
    )

    art = extract_v3mer_artifacts(
        df,
        count_mode="clonotype",
        unique_clonotypes=True,
    )

    assert art.clonotype_counts["v3::TRBV1::AAA"] == 1
    assert art.occurrence_counts["v3::TRBV1::AAA"] == 2
    assert art.counts["v3::TRBV1::AAA"] == 1


def test_extract_vpos3_and_u4_artifacts_expose_new_token_families() -> None:
    df = pd.DataFrame(
        {
            "reference_id": ["study"],
            "junction_aa": ["AAAA"],
            "v_gene": ["TRBV1*01"],
            "j_gene": ["TRBJ1-1*01"],
            "duplicate_count": [1],
            "row_id": ["0"],
        }
    )

    vpos = extract_vpos3mer_artifacts(df, count_mode="clonotype")
    u4 = extract_u4mer_artifacts(df, count_mode="clonotype")

    assert "vpos3::TRBV1::0::AAA" in vpos.counts
    assert "vpos3::TRBV1::1::AAA" in vpos.counts
    assert vpos.counts["vpos3::TRBV1::0::AAA"] == 1
    assert vpos.counts["vpos3::TRBV1::1::AAA"] == 1
    assert u4.counts["u4::AAAA"] == 1