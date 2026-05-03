from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import pandas as pd

from mir.common.control import (
    ControlManager,
    ControlRecord,
    build_real_control_from_ntvj,
    compute_control_pgen_records,
)


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
                "duplicate_count": [1, 2, 1],
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

    loaded = mgr.load_control_df("synthetic", "human", "TRB", n=3)
    assert list(loaded.columns) == ["duplicate_count", "junction", "junction_aa", "v_gene", "j_gene"]
    assert len(loaded) == 3
    assert int(loaded["duplicate_count"].sum()) == 4

    manifest = mgr.load_manifest()
    assert "synthetic:human:TRB:n=3" in manifest["records"]


def test_build_real_control_from_ntvj_appends_alleles(tmp_path: Path) -> None:
    src = tmp_path / "human_trb.ntvj"
    src.write_text(
        "count\tcdr3nt\tcdr3aa\tv\tj\n"
        "1\tATG\tM\tTRBV1\tTRBJ1\n"
        "2\tGTA\tV\tTRBV2*02\tTRBJ2\n",
        encoding="utf-8",
    )

    df = build_real_control_from_ntvj(src)

    assert list(df.columns) == ["duplicate_count", "junction", "junction_aa", "v_gene", "j_gene"]
    assert int(df.iloc[0]["duplicate_count"]) == 1
    assert int(df.iloc[1]["duplicate_count"]) == 2
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
    assert int(df.iloc[0]["duplicate_count"]) == 1
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
                "duplicate_count": [1],
                "junction": ["ATG"],
                "junction_aa": ["M"],
                "v_gene": ["TRAV1*01"],
                "j_gene": ["TRAJ1*01"],
            }
        ),
    )

    df = mgr.ensure_and_load_control_df("synthetic", "hsa", "Talpha", n=1, overwrite=True, progress=False)
    assert len(df) == 1
    assert int(df.iloc[0]["duplicate_count"]) == 1
    assert df.iloc[0]["v_gene"] == "TRAV1*01"


def test_ensure_synthetic_control_rebuilds_unreadable_cache(tmp_path: Path, monkeypatch) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    monkeypatch.setattr(
        ControlManager,
        "list_available_olga_models",
        staticmethod(lambda model_root=None: [("human", "TRB")]),
    )

    path = mgr.synthetic_control_path("human", "TRB", 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-pickle")

    monkeypatch.setattr(
        "mir.common.control.generate_synthetic_olga_control",
        lambda **kwargs: pd.DataFrame(
            {
                "duplicate_count": [1],
                "junction": ["ATG"],
                "junction_aa": ["M"],
                "v_gene": ["TRBV1*01"],
                "j_gene": ["TRBJ1*01"],
                "log2_pgen": [-12.5],
            }
        ),
    )

    rec = mgr.ensure_synthetic_control("human", "TRB", n=3, overwrite=False, progress=False)
    loaded = mgr.load_control_df("synthetic", "human", "TRB", n=3)

    assert Path(rec.path).exists()
    assert len(loaded) == 1
    assert float(loaded.iloc[0]["log2_pgen"]) == -12.5


def test_compute_control_pgen_records_uses_precomputed_log2_pgen_with_adjustment(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "junction_aa": ["CASSIRSSYEQYF"],
            "v_gene": ["TRBV1*01"],
            "j_gene": ["TRBJ1*01"],
            "log2_pgen": [-10.0],
        }
    )

    class _Adj:
        def factor(self, locus: str, v: str, j: str) -> float:
            assert locus == "TRB"
            assert v == "TRBV1*01"
            assert j == "TRBJ1*01"
            return 4.0

    def _fail(*args, **kwargs):
        raise AssertionError("OlgaModel should not be instantiated when log2_pgen is precomputed")

    monkeypatch.setattr("mir.common.control.OlgaModel", _fail)

    records = compute_control_pgen_records(
        df,
        locus="TRB",
        species="human",
        pgen_adjustment=_Adj(),
    )

    assert len(records) == 1
    assert math.isclose(records[0]["log2_pgen"], -8.0, rel_tol=0.0, abs_tol=1e-9)


def test_ensure_synthetic_control_waits_for_existing_lock(tmp_path: Path, monkeypatch) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    monkeypatch.setattr(
        ControlManager,
        "list_available_olga_models",
        staticmethod(lambda model_root=None: [("human", "TRB")]),
    )
    monkeypatch.setattr(
        "mir.common.control.generate_synthetic_olga_control",
        lambda **kwargs: pd.DataFrame(
            {
                "duplicate_count": [1],
                "junction": ["ATG"],
                "junction_aa": ["M"],
                "v_gene": ["TRBV1*01"],
                "j_gene": ["TRBJ1*01"],
            }
        ),
    )

    lock_path = mgr._control_lock_path("synthetic", "human", "TRB", n=1)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("hold\n", encoding="utf-8")

    def _release_lock_later() -> None:
        time.sleep(0.3)
        lock_path.unlink(missing_ok=True)

    t = threading.Thread(target=_release_lock_later, daemon=True)
    t.start()

    t0 = time.perf_counter()
    rec = mgr.ensure_synthetic_control("human", "TRB", n=1, overwrite=True, progress=False)
    elapsed = time.perf_counter() - t0

    assert Path(rec.path).exists()
    # Should have waited for lock release by peer process/thread.
    assert elapsed >= 0.25


def test_load_control_df_waits_if_building_lock_present(tmp_path: Path) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    path = mgr.synthetic_control_path("human", "TRB", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "duplicate_count": [1],
            "junction": ["ATG"],
            "junction_aa": ["M"],
            "v_gene": ["TRBV1*01"],
            "j_gene": ["TRBJ1*01"],
        }
    )
    df.to_pickle(path)
    mgr.register_record(
        ControlRecord(
            control_type="synthetic",
            species="human",
            locus="TRB",
            path=str(path),
            format="pickle",
            source="olga",
            n=1,
            created_at_utc=None,
        )
    )

    lock_path = mgr._control_lock_path("synthetic", "human", "TRB", n=1)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("building\n", encoding="utf-8")

    def _release_lock_later() -> None:
        time.sleep(0.25)
        lock_path.unlink(missing_ok=True)

    t = threading.Thread(target=_release_lock_later, daemon=True)
    t.start()

    t0 = time.perf_counter()
    loaded = mgr.load_control_df("synthetic", "human", "TRB", n=1, wait_if_building=True, wait_timeout_s=5.0)
    elapsed = time.perf_counter() - t0

    assert len(loaded) == 1
    assert elapsed >= 0.2


def test_synthetic_controls_with_different_n_have_distinct_manifest_records(tmp_path: Path) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    path_small = mgr.synthetic_control_path("human", "TRB", 10)
    path_large = mgr.synthetic_control_path("human", "TRB", 20)
    path_small.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "duplicate_count": [1],
            "junction": ["ATG"],
            "junction_aa": ["M"],
            "v_gene": ["TRBV1*01"],
            "j_gene": ["TRBJ1*01"],
        }
    ).to_pickle(path_small)
    pd.DataFrame(
        {
            "duplicate_count": [2],
            "junction": ["GTA"],
            "junction_aa": ["V"],
            "v_gene": ["TRBV2*01"],
            "j_gene": ["TRBJ2*01"],
        }
    ).to_pickle(path_large)

    mgr.register_record(
        ControlRecord(
            control_type="synthetic",
            species="human",
            locus="TRB",
            path=str(path_small),
            format="pickle",
            source="olga",
            n=10,
            created_at_utc=None,
        )
    )
    mgr.register_record(
        ControlRecord(
            control_type="synthetic",
            species="human",
            locus="TRB",
            path=str(path_large),
            format="pickle",
            source="olga",
            n=20,
            created_at_utc=None,
        )
    )

    manifest = mgr.load_manifest()
    assert "synthetic:human:TRB:n=10" in manifest["records"]
    assert "synthetic:human:TRB:n=20" in manifest["records"]

    loaded_small = mgr.load_control_df("synthetic", "human", "TRB", n=10)
    loaded_large = mgr.load_control_df("synthetic", "human", "TRB", n=20)
    assert int(loaded_small.iloc[0]["duplicate_count"]) == 1
    assert int(loaded_large.iloc[0]["duplicate_count"]) == 2


def test_loading_synthetic_control_without_n_is_rejected_when_multiple_sizes_exist(tmp_path: Path) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    for n, dup in [(10, 1), (20, 2)]:
        path = mgr.synthetic_control_path("human", "TRB", n)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "duplicate_count": [dup],
                "junction": ["ATG"],
                "junction_aa": ["M"],
                "v_gene": ["TRBV1*01"],
                "j_gene": ["TRBJ1*01"],
            }
        ).to_pickle(path)
        mgr.register_record(
            ControlRecord(
                control_type="synthetic",
                species="human",
                locus="TRB",
                path=str(path),
                format="pickle",
                source="olga",
                n=n,
                created_at_utc=None,
            )
        )

    try:
        mgr.load_control_df("synthetic", "human", "TRB")
    except ValueError as exc:
        assert "specify n explicitly" in str(exc)
    else:
        raise AssertionError("Expected ambiguous synthetic cache load to require explicit n")
