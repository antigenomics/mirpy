from __future__ import annotations

import pandas as pd
import pytest

from mir.biomarkers.tcrnet import add_tcrnet_metadata, compute_tcrnet, tcrnet_table
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire


def _clone(
    sid: str,
    aa: str,
    *,
    v: str = "TRBV5-1*01",
    j: str = "TRBJ2-7*01",
    dup: int = 1,
) -> Clonotype:
    return Clonotype(
        sequence_id=sid,
        locus="TRB",
        junction_aa=aa,
        v_gene=v,
        j_gene=j,
        duplicate_count=dup,
        _validate=False,
    )


def _toy_target() -> LocusRepertoire:
    return LocusRepertoire(
        [
            _clone("0", "CASSLGQETQYF"),
            _clone("1", "CASSLGQETQFF"),
            _clone("2", "CASSLGQDTQYF"),
            _clone("3", "CASSQGQETQYF"),
            _clone("4", "CATSRGQETQYF"),
        ],
        locus="TRB",
    )


def _toy_control() -> LocusRepertoire:
    return LocusRepertoire(
        [
            _clone("a", "CASSLGQETQYF"),
            _clone("b", "CASSPGQETQYF"),
            _clone("c", "CASSLGQATQYF"),
            _clone("d", "CARDRGQETQYF", v="TRBV20-1*01"),
            _clone("e", "CASSLGQETQRF"),
            _clone("f", "CASSLGRETQYF"),
        ],
        locus="TRB",
    )


def test_compute_tcrnet_binomial_basic() -> None:
    result = compute_tcrnet(
        _toy_target(),
        control=_toy_control(),
        metric="hamming",
        threshold=1,
        match_mode="vj",
        pvalue_mode="binomial",
    )

    assert not result.table.empty
    required = {
        "sequence_id",
        "n_neighbors",
        "N_possible",
        "m_control_neighbors",
        "M_control_possible",
        "fold_enrichment",
        "p_value",
    }
    assert required.issubset(set(result.table.columns))
    assert (result.table["N_possible"] >= 1).all()
    assert (result.table["M_control_possible"] >= 0).all()
    assert ((result.table["p_value"] >= 0.0) & (result.table["p_value"] <= 1.0)).all()


def test_compute_tcrnet_beta_binomial_runs() -> None:
    rep = _toy_target()
    out = compute_tcrnet(
        rep,
        control=_toy_control(),
        metric="levenshtein",
        threshold=1,
        match_mode="v",
        pvalue_mode="beta-binomial",
        as_table=False,
    )
    assert out is rep
    for c in rep.clonotypes:
        p = float(c.clone_metadata["tcrnet_p_value"])
        assert 0.0 <= p <= 1.0


def test_add_tcrnet_metadata_inplace() -> None:
    rep = _toy_target()
    out = add_tcrnet_metadata(rep, control=_toy_control(), match_mode="none")
    assert out is rep
    for c in rep.clonotypes:
        assert "tcrnet_n" in c.clone_metadata
        assert "tcrnet_fold" in c.clone_metadata
        assert "tcrnet_p_value" in c.clone_metadata


def test_tcrnet_table_from_metadata() -> None:
    rep = _toy_target()
    add_tcrnet_metadata(rep, control=_toy_control(), match_mode="none")
    table = tcrnet_table(rep)
    assert not table.empty
    assert len(table) == len(rep.clonotypes)
    assert ((table["p_value"] >= 0.0) & (table["p_value"] <= 1.0)).all()


def test_threshold_gt_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_tcrnet(
            _toy_target(),
            control=_toy_control(),
            threshold=2,
        )


def test_control_type_loading_via_manager(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "duplicate_count": [1, 1],
            "junction": ["ATG", "GCT"],
            "junction_aa": ["CASSLGQETQYF", "CASSLGQATQYF"],
            "v_gene": ["TRBV5-1*01", "TRBV5-1*01"],
            "j_gene": ["TRBJ2-7*01", "TRBJ2-7*01"],
        }
    )

    monkeypatch.setattr(
        ControlManager,
        "ensure_and_load_control_df",
        lambda self, control_type, species, locus, **kwargs: df,
    )

    result = compute_tcrnet(
        _toy_target(),
        control_type="real",
        species="human",
        control_manager=ControlManager(control_dir="/tmp/mirpy_tcrnet_test_controls"),
    )
    assert not result.table.empty


def test_tcrnet_parallel_matches_single_worker() -> None:
    serial = compute_tcrnet(
        _toy_target(),
        control=_toy_control(),
        metric="hamming",
        threshold=1,
        match_mode="none",
        pvalue_mode="binomial",
        n_jobs=1,
    )
    parallel = compute_tcrnet(
        _toy_target(),
        control=_toy_control(),
        metric="hamming",
        threshold=1,
        match_mode="none",
        pvalue_mode="binomial",
        n_jobs=4,
    )

    pd.testing.assert_frame_equal(serial.table, parallel.table)
