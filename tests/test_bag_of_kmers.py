from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import pandas as pd

from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset
from mir.embedding.bag_of_kmers import (
    BagOfKmersParams,
    build_control_kmer_profile,
    control_kmer_profile_name,
    ensure_control_kmer_profile,
    load_control_kmer_profile,
    tokenize_dataset_by_sample_and_locus,
    tokenize_locus_repertoire_to_table,
    tokenize_sample_repertoire_by_locus,
)


def _clone(
    seq_id: str,
    jaa: str,
    *,
    locus: str = "TRB",
    v_gene: str = "TRBV5-1*01",
    duplicate_count: int = 1,
) -> Clonotype:
    return Clonotype(
        sequence_id=seq_id,
        locus=locus,
        v_gene=v_gene,
        j_gene="TRBJ1-1*01",
        c_gene="TRBC1",
        junction_aa=jaa,
        duplicate_count=duplicate_count,
        _validate=False,
    )


def test_tokenize_locus_repertoire_plain_and_reduced() -> None:
    rep = LocusRepertoire(
        [
            _clone("1", "CASS", duplicate_count=2),
            _clone("2", "CASA", duplicate_count=1),
        ],
        locus="TRB",
    )

    plain = tokenize_locus_repertoire_to_table(rep, params=BagOfKmersParams(k=3))
    counts = dict(zip(plain["token"], plain["n"]))
    assert counts["CAS"] == 3

    reduced = tokenize_locus_repertoire_to_table(
        rep,
        params=BagOfKmersParams(k=3, reduced_alphabet=True),
    )
    assert int(reduced["T"].iloc[0]) == int(plain["T"].iloc[0])


def test_tokenize_locus_repertoire_with_v_annotation() -> None:
    rep = LocusRepertoire(
        [
            _clone("1", "CASS", v_gene="TRBV5-1*01", duplicate_count=1),
            _clone("2", "CASS", v_gene="TRBV6-1*01", duplicate_count=1),
        ],
        locus="TRB",
    )
    table = tokenize_locus_repertoire_to_table(rep, params=BagOfKmersParams(use_v=True, k=3))
    tokens = set(table["token"])
    assert "TRBV5-1|CAS" in tokens
    assert "TRBV6-1|CAS" in tokens


def test_tokenize_locus_repertoire_gapped_total_kmers() -> None:
    rep = LocusRepertoire([_clone("1", "CASSL", duplicate_count=1)], locus="TRB")
    table = tokenize_locus_repertoire_to_table(
        rep,
        params=BagOfKmersParams(k=3, gapped=True),
    )
    total = int(table["T"].iloc[0])
    # len=5, k=3 -> (5-3+1)*3 = 9 gapped tokens
    assert total == 9


def test_tokenize_sample_and_dataset_by_locus() -> None:
    s1 = SampleRepertoire(
        loci={
            "TRA": LocusRepertoire([_clone("1", "CASS", locus="TRA", v_gene="TRAV1*01")], locus="TRA"),
            "TRB": LocusRepertoire([_clone("2", "CASS", locus="TRB", v_gene="TRBV5-1*01")], locus="TRB"),
        },
        sample_id="S1",
    )
    s2 = SampleRepertoire(
        loci={
            "TRB": LocusRepertoire([_clone("3", "CATS", locus="TRB", v_gene="TRBV7-2*01")], locus="TRB"),
        },
        sample_id="S2",
    )
    ds = RepertoireDataset(samples={"S1": s1, "S2": s2})

    by_locus = tokenize_sample_repertoire_by_locus(s1, params=BagOfKmersParams(k=3))
    assert set(by_locus) == {"TRA", "TRB"}

    by_sample = tokenize_dataset_by_sample_and_locus(ds, params=BagOfKmersParams(k=3))
    assert set(by_sample) == {"S1", "S2"}
    assert set(by_sample["S1"]) == {"TRA", "TRB"}
    assert set(by_sample["S2"]) == {"TRB"}


def test_control_profile_name_shape() -> None:
    name = control_kmer_profile_name(
        "real",
        "human",
        "TRB",
        params=BagOfKmersParams(use_v=True, k=3, gapped=True, reduced_alphabet=True),
    )
    assert name == "real_human_TRB_v_3mer_gapped_reduced"


def test_build_control_kmer_profile_in_memory_default(tmp_path: Path, monkeypatch) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    control_df = pd.DataFrame(
        {
            "duplicate_count": [2, 1],
            "junction": ["ATG", "GCT"],
            "junction_aa": ["CASS", "CASA"],
            "v_gene": ["TRBV5-1*01", "TRBV5-1*01"],
            "j_gene": ["TRBJ1*01", "TRBJ1*01"],
        }
    )

    monkeypatch.setattr(
        ControlManager,
        "ensure_control",
        lambda self, control_type, species, locus, **kwargs: None,
    )
    monkeypatch.setattr(
        ControlManager,
        "load_control_df",
        lambda self, control_type, species, locus, **kwargs: control_df,
    )

    params = BagOfKmersParams(k=3)
    profile = build_control_kmer_profile(
        mgr,
        control_type="real",
        species="human",
        locus="TRB",
        params=params,
    )

    assert profile.metadata["profile_name"] == "real_human_TRB_kmer_3mer_plain_full"
    assert profile.metadata["total_kmers"] > 0
    assert profile.metadata["cache_enabled"] is False

    row = profile.token_stats[profile.token_stats["token"] == "CAS"].iloc[0]
    assert int(row["n"]) == 3
    assert math.isclose(float(row["idf"]), -math.log(float(row["n"]) / float(row["T"])))


def test_ensure_and_load_control_kmer_profile_with_cache(tmp_path: Path, monkeypatch) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    control_df = pd.DataFrame(
        {
            "duplicate_count": [2, 1],
            "junction": ["ATG", "GCT"],
            "junction_aa": ["CASS", "CASA"],
            "v_gene": ["TRBV5-1*01", "TRBV5-1*01"],
            "j_gene": ["TRBJ1*01", "TRBJ1*01"],
        }
    )

    monkeypatch.setattr(
        ControlManager,
        "ensure_control",
        lambda self, control_type, species, locus, **kwargs: None,
    )
    monkeypatch.setattr(
        ControlManager,
        "load_control_df",
        lambda self, control_type, species, locus, **kwargs: control_df,
    )

    params = BagOfKmersParams(k=3)
    profile = ensure_control_kmer_profile(
        mgr,
        control_type="real",
        species="human",
        locus="TRB",
        params=params,
        cache=True,
    )

    assert profile.metadata["cache_enabled"] is True

    loaded = load_control_kmer_profile(
        mgr,
        control_type="real",
        species="human",
        locus="TRB",
        params=params,
    )

    assert len(loaded.token_stats) > 0
    assert len(loaded.position_stats) > 0
    assert loaded.metadata["profile_name"] == profile.metadata["profile_name"]


def test_control_kmer_profile_waits_for_lock(tmp_path: Path, monkeypatch) -> None:
    mgr = ControlManager(control_dir=tmp_path / "controls")

    control_df = pd.DataFrame(
        {
            "duplicate_count": [1],
            "junction": ["ATG"],
            "junction_aa": ["CASS"],
            "v_gene": ["TRBV1*01"],
            "j_gene": ["TRBJ1*01"],
        }
    )

    monkeypatch.setattr(
        ControlManager,
        "ensure_control",
        lambda self, control_type, species, locus, **kwargs: None,
    )
    monkeypatch.setattr(
        ControlManager,
        "load_control_df",
        lambda self, control_type, species, locus, **kwargs: control_df,
    )

    params = BagOfKmersParams(k=3)
    lock_path = (
        mgr.control_dir
        / ".locks"
        / "kmer_profile_real_human_TRB_kmer_3mer_plain_full.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("held\n", encoding="utf-8")

    def _release() -> None:
        time.sleep(0.3)
        lock_path.unlink(missing_ok=True)

    t = threading.Thread(target=_release, daemon=True)
    t.start()

    t0 = time.perf_counter()
    ensure_control_kmer_profile(
        mgr,
        control_type="real",
        species="human",
        locus="TRB",
        params=params,
        cache=True,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.25
