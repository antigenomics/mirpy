from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import poisson

from mir.biomarkers.alice import add_alice_metadata, compute_alice
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire


class _FakeOlgaModel:
    def __init__(self, *, locus: str, species: str, seed: int | None = 42) -> None:
        self.locus = locus
        self.species = species
        self.seed = seed

    def compute_pgen_junction_aa(self, junction_aa: str) -> float:
        return {
            "CASSLGQETQYF": 0.2,
            "CASSLGQETQFF": 0.1,
            "CASSQGQETQYF": 0.3,
        }.get(junction_aa, 0.05)

    def compute_pgen_junction_aa_1mm(self, junction_aa: str) -> float:
        return {
            "CASSLGQETQYF": 0.5,
            "CASSLGQETQFF": 0.4,
            "CASSQGQETQYF": 0.6,
        }.get(junction_aa, 0.2)


def _clone(
    sid: str,
    aa: str,
    *,
    v: str = "TRBV5-1*01",
    j: str = "TRBJ2-7*01",
) -> Clonotype:
    return Clonotype(
        sequence_id=sid,
        locus="TRB",
        junction_aa=aa,
        v_gene=v,
        j_gene=j,
        duplicate_count=1,
        _validate=False,
    )


def test_compute_alice_basic_formulae(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire(
        [
            _clone("0", "CASSLGQETQYF"),
            _clone("1", "CASSLGQETQFF"),
        ],
        locus="TRB",
    )

    result = compute_alice(
        rep,
        threshold=1,
        match_mode="none",
        pgen_mode="exact",
        n_jobs=1,
    )

    assert not result.table.empty
    row0 = result.table[result.table["sequence_id"] == "0"].iloc[0]

    assert int(row0["n_neighbors"]) == 2
    assert int(row0["N_possible"]) == 2
    assert float(row0["pgen"]) == pytest.approx(0.2)
    assert float(row0["expected_neighbors"]) == pytest.approx(0.4)
    assert float(row0["fold_enrichment"]) == pytest.approx(5.0)
    assert float(row0["p_value"]) == pytest.approx(float(poisson.sf(1, 0.4)))


def test_compute_alice_v_matching_uses_gene_usage_divisor(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    control_df = pd.DataFrame(
        {
            "v_gene": ["TRBV5-1*01", "TRBV5-1*01", "TRBV5-1*01", "TRBV6-1*01"],
            "j_gene": ["TRBJ2-7*01", "TRBJ2-1*01", "TRBJ2-7*01", "TRBJ2-7*01"],
        }
    )

    monkeypatch.setattr(
        ControlManager,
        "ensure_and_load_control_df",
        lambda self, control_type, species, locus, **kwargs: control_df,
    )

    rep = LocusRepertoire([_clone("0", "CASSQGQETQYF")], locus="TRB")

    result = compute_alice(
        rep,
        threshold=0,
        match_mode="v",
        pgen_mode="exact",
        control_manager=ControlManager(control_dir="/tmp/mirpy_alice_test_controls"),
        gene_usage_synthetic_n=1_000,
        n_jobs=1,
    )

    row = result.table.iloc[0]
    p_v = 3.0 / 4.0
    expected_pgen = 0.3 / (p_v + 1e-6)
    assert float(row["pgen_raw"]) == pytest.approx(0.3)
    assert float(row["pgen"]) == pytest.approx(expected_pgen)


def test_compute_alice_1mm_mode(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    result = compute_alice(rep, threshold=0, pgen_mode="1mm", n_jobs=1)
    row = result.table.iloc[0]
    assert float(row["pgen_raw"]) == pytest.approx(0.5)


def test_add_alice_metadata_inplace(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    out = add_alice_metadata(rep, threshold=0, n_jobs=1)
    assert out is rep

    md = rep.clonotypes[0].clone_metadata
    assert "alice_n" in md
    assert "alice_N" in md
    assert "alice_pgen" in md
    assert "alice_fold" in md
    assert "alice_p_value" in md


def test_threshold_gt_one_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    with pytest.raises(ValueError):
        compute_alice(rep, threshold=2)


def test_only_hamming_metric_is_supported(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    with pytest.raises(ValueError):
        compute_alice(rep, metric="levenshtein")
