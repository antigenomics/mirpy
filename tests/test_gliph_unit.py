from __future__ import annotations

import warnings

import pandas as pd
import polars as pl

from mir.biomarkers.gliph import (
    deduplicate_clonotype_rows,
    extract_g5mer_artifacts,
    extract_pos3mer_artifacts,
    extract_u4mer_artifacts,
    extract_v3mer_artifacts,
    extract_vpos3mer_artifacts,
    normalize_control_v,
)


def test_deduplicate_clonotype_rows_sums_duplicate_count() -> None:
    df = pl.DataFrame(
        {
            "reference_id": ["study", "study", "study"],
            "junction_aa": ["AAAA", "AAAA", "CCCC"],
            "v_call": ["TRBV1*01", "TRBV1*01", "TRBV2*01"],
            "j_call": ["TRBJ1-1*01", "TRBJ1-1*01", "TRBJ2-1*01"],
            "duplicate_count": [2, 3, 1],
            "stimulus": ["x", "x", "y"],
            "row_id": ["0", "1", "2"],
        }
    )

    dedup = deduplicate_clonotype_rows(df)

    assert len(dedup) == 2
    aaaa = dedup.filter(pl.col("junction_aa") == "AAAA").row(0, named=True)
    assert int(aaaa["duplicate_count"]) == 5


def test_extract_v3mer_artifacts_supports_clonotype_count_mode() -> None:
    df = pl.DataFrame(
        {
            "reference_id": ["study", "study"],
            "junction_aa": ["CASSLGQETQYF", "CASSLGQETQYF"],
            "v_call": ["TRBV1*01", "TRBV1*01"],
            "j_call": ["TRBJ1-1*01", "TRBJ1-1*01"],
            "duplicate_count": [1, 1],
            "row_id": ["0", "1"],
        }
    )

    art = extract_v3mer_artifacts(
        df,
        count_mode="clonotype",
        unique_clonotypes=True,
    )

    assert art.clonotype_counts["v3::TRBV1::SLG"] == 1
    assert art.occurrence_counts["v3::TRBV1::SLG"] == 1
    assert art.counts["v3::TRBV1::SLG"] == 1


def test_extract_pos3_and_u4_artifacts_expose_new_token_families() -> None:
    df = pl.DataFrame(
        {
            "reference_id": ["study"],
            "junction_aa": ["CASSLGQETQYF"],
            "v_call": ["TRBV1*01"],
            "j_call": ["TRBJ1-1*01"],
            "duplicate_count": [1],
            "row_id": ["0"],
        }
    )

    pos3 = extract_pos3mer_artifacts(df, count_mode="clonotype")
    u4 = extract_u4mer_artifacts(df, count_mode="clonotype")

    assert "pos3::TRBV1::3::SLG" in pos3.counts
    assert "pos3::TRBV1::4::LGQ" in pos3.counts
    assert pos3.counts["pos3::TRBV1::3::SLG"] == 1
    assert pos3.counts["pos3::TRBV1::4::LGQ"] == 1
    assert u4.counts["u4::SLGQ"] == 1


def test_extract_g5mer_artifacts_exposes_gapped_5mer_family() -> None:
    df = pl.DataFrame(
        {
            "reference_id": ["study"],
            "junction_aa": ["CASSLGQETQYF"],
            "v_call": ["TRBV1*01"],
            "j_call": ["TRBJ1-1*01"],
            "duplicate_count": [1],
            "row_id": ["0"],
        }
    )

    g5 = extract_g5mer_artifacts(df, count_mode="clonotype")

    assert "g5::SLGQX" in g5.counts
    assert any(token.startswith("g5::") and "X" in token for token in g5.counts)
    assert g5.counts["g5::SLGQX"] == 1


def test_extract_vpos3_alias_still_returns_pos3_tokens() -> None:
    df = pl.DataFrame(
        {
            "reference_id": ["study"],
            "junction_aa": ["CASSLGQETQYF"],
            "v_call": ["TRBV1*01"],
            "j_call": ["TRBJ1-1*01"],
            "duplicate_count": [1],
            "row_id": ["0"],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vpos_alias = extract_vpos3mer_artifacts(df, count_mode="clonotype")

    assert any("deprecated" in str(w.message).lower() for w in caught)
    assert "pos3::TRBV1::3::SLG" in vpos_alias.counts


def test_extract_family_artifacts_can_disable_trimming() -> None:
    df = pl.DataFrame(
        {
            "reference_id": ["study"],
            "junction_aa": ["CASSLGQETQYF"],
            "v_call": ["TRBV1*01"],
            "j_call": ["TRBJ1-1*01"],
            "duplicate_count": [1],
            "row_id": ["0"],
        }
    )

    untrimmed = extract_v3mer_artifacts(df, count_mode="clonotype", trim_first=0, trim_last=0)
    assert "v3::TRBV1::CAS" in untrimmed.counts


def test_normalize_control_v_matches_v_usage_only() -> None:
    sample_df = pd.DataFrame(
        {
            "row_id": ["0", "1", "2"],
            "junction_aa": ["AAAAA", "CCCCC", "GGGGG"],
            "v_call": ["TRBV1*01", "TRBV1*01", "TRBV2*01"],
            "j_call": ["TRBJ1-1*01", "TRBJ2-1*01", "TRBJ2-3*01"],
            "duplicate_count": [1, 1, 1],
        }
    )
    control_df = pd.DataFrame(
        {
            "junction_aa": ["VVVVV", "WWWWW", "XXXXX", "YYYYY"],
            "v_call": ["TRBV1", "TRBV1", "TRBV2", "TRBV3"],
            "j_call": ["TRBJ1-1", "TRBJ2-1", "TRBJ2-3", "TRBJ1-2"],
            "duplicate_count": [1, 1, 1, 1],
        }
    )

    norm = normalize_control_v(sample_df, control_df, n=30, seed=1)
    v_freq = norm["v_call"].astype(str).str.split("*").str[0].value_counts(normalize=True)

    assert set(v_freq.index).issubset({"TRBV1", "TRBV2"})
    assert v_freq["TRBV1"] > v_freq["TRBV2"]
