from __future__ import annotations

from pathlib import Path

import pandas as pd

from mir.common.control import ControlManager, build_real_control_from_ntvj


def test_control_aliases_species_and_locus(tmp_path: Path) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    assert mgr.canonical_species("human") == "human"
    assert mgr.canonical_species("hsa") == "human"
    assert mgr.canonical_species("HomoSapiens") == "human"

    assert mgr.canonical_species("mouse") == "mouse"
    assert mgr.canonical_species("mmu") == "mouse"
    assert mgr.canonical_species("MusMusculus") == "mouse"

    assert mgr.canonical_locus("TRA") == "TRA"
    assert mgr.canonical_locus("Talpha") == "TRA"
    assert mgr.canonical_locus("t_beta") == "TRB"
    assert mgr.canonical_locus("IGH") == "IGH"
    assert mgr.canonical_locus("Bkappa") == "IGK"


def test_list_available_olga_models_from_model_root(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    (model_root / "human_T_alpha").mkdir(parents=True)
    (model_root / "human_T_beta").mkdir(parents=True)
    (model_root / "mouse_T_alpha").mkdir(parents=True)
    (model_root / "mouse_T_beta").mkdir(parents=True)
    (model_root / "human_unknown").mkdir(parents=True)

    got = ControlManager.list_available_olga_models(model_root=model_root)

    assert ("human", "TRA") in got
    assert ("human", "TRB") in got
    assert ("mouse", "TRA") in got
    assert ("mouse", "TRB") in got
    assert ("human", "TRG") not in got


def test_ensure_synthetic_control_registers_manifest(tmp_path: Path, monkeypatch) -> None:
    control_dir = tmp_path / "controls"
    mgr = ControlManager(control_dir=control_dir)

    monkeypatch.setattr(
        ControlManager,
        "list_available_olga_models",
        staticmethod(lambda model_root=None: [("human", "TRB")]),
    )

    def _fake_generator(*, species: str, locus: str, n: int, seed: int, chunk_size: int, progress: bool) -> pd.DataFrame:
        assert species == "human"
        assert locus == "TRB"
        assert n == 3
        return pd.DataFrame(
            {
                "junction": ["ATG", "GTA", "CCC"],
                "junction_aa": ["M", "V", "P"],
                "v_gene": ["TRBV1*01", "TRBV2*01", "TRBV3*01"],
                "j_gene": ["TRBJ1*01", "TRBJ2*01", "TRBJ1*01"],
            }
        )

    monkeypatch.setattr("mir.common.control.generate_synthetic_olga_control", _fake_generator)

    rec = mgr.ensure_synthetic_control("hsa", "Tbeta", n=3, overwrite=True, progress=False)

    assert rec.control_type == "synthetic"
    assert rec.species == "human"
    assert rec.locus == "TRB"
    out_path = Path(rec.path)
    assert out_path.exists()

    loaded = mgr.load_control_df("synthetic", "human", "TRB")
    assert list(loaded.columns) == ["junction", "junction_aa", "v_gene", "j_gene"]
    assert len(loaded) == 3

    manifest = mgr.load_manifest()
    assert "synthetic:human:TRB" in manifest["records"]


def test_build_real_control_from_ntvj_appends_alleles(tmp_path: Path) -> None:
    src = tmp_path / "human_trb.ntvj"
    src.write_text(
        "count\tcdr3nt\tcdr3aa\tv\tj\n"
        "1\tATG\tM\tTRBV1\tTRBJ1\n"
        "2\tGTA\tV\tTRBV2*02\tTRBJ2\n",
        encoding="utf-8",
    )

    df = build_real_control_from_ntvj(src)

    assert list(df.columns) == ["junction", "junction_aa", "v_gene", "j_gene"]
    assert df.iloc[0]["v_gene"] == "TRBV1*01"
    assert df.iloc[1]["v_gene"] == "TRBV2*02"
    assert df.iloc[1]["j_gene"] == "TRBJ2*01"


def test_ensure_real_control_download_and_register(tmp_path: Path, monkeypatch) -> None:
    control_dir = tmp_path / "controls"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir(parents=True)

    ntvj = snapshot_dir / "hsa_trb_controls.ntvj"
    ntvj.write_text(
        "count\tcdr3nt\tcdr3aa\tv\tj\n"
        "1\tATG\tM\tTRBV1\tTRBJ1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("mir.common.control._download_hf_snapshot", lambda repo_id, cache_dir=None: str(snapshot_dir))

    mgr = ControlManager(control_dir=control_dir)
    rec = mgr.ensure_real_control("human", "TRB", overwrite=True)

    assert rec.control_type == "real"
    assert rec.species == "human"
    assert rec.locus == "TRB"
    assert Path(rec.path).exists()

    df = mgr.load_control_df("real", "hsa", "Tbeta")
    assert len(df) == 1
    assert df.iloc[0]["v_gene"] == "TRBV1*01"


def test_ensure_and_load_control_df(tmp_path: Path, monkeypatch) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    monkeypatch.setattr(
        ControlManager,
        "list_available_olga_models",
        staticmethod(lambda model_root=None: [("human", "TRA")]),
    )

    monkeypatch.setattr(
        "mir.common.control.generate_synthetic_olga_control",
        lambda **kwargs: pd.DataFrame(
            {
                "junction": ["ATG"],
                "junction_aa": ["M"],
                "v_gene": ["TRAV1*01"],
                "j_gene": ["TRAJ1*01"],
            }
        ),
    )

    df = mgr.ensure_and_load_control_df("synthetic", "hsa", "Talpha", n=1, overwrite=True, progress=False)
    assert len(df) == 1
    assert df.iloc[0]["v_gene"] == "TRAV1*01"
