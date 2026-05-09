from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

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

    def compute_pgen_junction_aa_bulk(
        self,
        junction_aas,
        *,
        max_mismatches: int = 0,
        n_jobs: int = 1,
    ) -> list[float]:
        compute_one = self.compute_pgen_junction_aa_1mm if max_mismatches == 1 else self.compute_pgen_junction_aa
        return [float(compute_one(seq)) for seq in junction_aas]


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


def test_compute_alice_uses_bulk_pgen_path(monkeypatch) -> None:
    thread_names: set[str] = set()
    thread_names_lock = threading.Lock()

    class _ThreadTrackingOlgaModel(_FakeOlgaModel):
        def compute_pgen_junction_aa(self, junction_aa: str) -> float:
            with thread_names_lock:
                thread_names.add(threading.current_thread().name)
            time.sleep(0.001)
            return super().compute_pgen_junction_aa(junction_aa)

        def compute_pgen_junction_aa_bulk(
            self,
            junction_aas,
            *,
            max_mismatches: int = 0,
            n_jobs: int = 1,
        ) -> list[float]:
            if n_jobs <= 1:
                return super().compute_pgen_junction_aa_bulk(
                    junction_aas,
                    max_mismatches=max_mismatches,
                    n_jobs=n_jobs,
                )
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                return list(executor.map(self.compute_pgen_junction_aa, junction_aas))

    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _ThreadTrackingOlgaModel)

    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = [
        f"CASSLGQETQ{aa[i % len(aa)]}{aa[(i // len(aa)) % len(aa)]}{aa[(i // (len(aa) * len(aa))) % len(aa)]}"
        for i in range(320)
    ]
    clones = [_clone(str(i), seqs[i]) for i in range(len(seqs))]
    rep = LocusRepertoire(clones, locus="TRB")

    compute_alice(rep, threshold=0, pgen_mode="exact", n_jobs=8)

    # Pgen is intentionally computed in one thread; p-value stage is parallelized separately.
    assert len(thread_names) == 1


def test_compute_alice_parallelizes_pvalue_calls(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    from mir.biomarkers import alice as alice_mod

    thread_names: set[str] = set()
    thread_names_lock = threading.Lock()
    original = alice_mod._poisson_pvalue

    def _tracking_poisson_pvalue(n: int, N: int, pgen: float) -> float:
        with thread_names_lock:
            thread_names.add(threading.current_thread().name)
        time.sleep(0.001)
        return original(n, N, pgen)

    monkeypatch.setattr("mir.biomarkers.alice._poisson_pvalue", _tracking_poisson_pvalue)

    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = [
        f"CASSLGQETQ{aa[i % len(aa)]}{aa[(i // len(aa)) % len(aa)]}{aa[(i // (len(aa) * len(aa))) % len(aa)]}"
        for i in range(320)
    ]
    rep = LocusRepertoire([_clone(str(i), seqs[i]) for i in range(len(seqs))], locus="TRB")

    compute_alice(rep, threshold=0, pgen_mode="exact", n_jobs=8)

    assert len(thread_names) > 1
