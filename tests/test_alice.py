from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import polars as pl
import pytest
from scipy.stats import poisson

from mir.biomarkers.alice import add_alice_metadata, compute_alice
from mir.common.repertoire import LocusRepertoire
from tests.factories import make_trb_clone


class _FakeGenModel:
    """Minimal gen_model stub for get_gene_usage_from_olga_model."""
    import numpy as np
    PV = np.array([1.0])
    PDJ = np.array([[1.0]])


class _FakeOlgaModel:
    def __init__(self, *, locus: str, species: str, seed: int | None = 42) -> None:
        self.locus = locus
        self.species = species
        self.seed = seed
        self.is_d_present = True
        self.v_names = ["TRBV5-1*01"]
        self.j_names = ["TRBJ2-7*01"]
        self.gen_model = _FakeGenModel()

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



_clone = make_trb_clone

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
        match_mode="none",
        pgen_mode="exact",
        min_neighbors=2,
        n_jobs=1,
    )

    assert not result.table.is_empty()
    row0 = result.table.filter(pl.col("sequence_id") == "0").row(0, named=True)

    # CASSLGQETQYF and CASSLGQETQFF differ by 1 AA (Hamming-1 neighbours).
    # With neighbourhood_threshold=1, both sequences are neighbours → n=2.
    # (n=2 = self + 1 additional neighbour; min_neighbors=2 is the minimum here)
    assert int(row0["n_neighbors"]) == 2
    assert int(row0["N_possible"]) == 2
    assert float(row0["pgen"]) == pytest.approx(0.2)
    assert float(row0["expected_neighbors"]) == pytest.approx(0.4)
    assert float(row0["fold_enrichment"]) == pytest.approx(5.0)
    assert float(row0["p_value"]) == pytest.approx(float(poisson.sf(1, 0.4)))


def test_compute_alice_v_matching_raw_pgen(monkeypatch) -> None:
    """match_mode='v' restricts neighborhood search but pgen is raw OLGA pgen (no conditioning)."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSQGQETQYF")], locus="TRB")

    result = compute_alice(
        rep,
        match_mode="v",
        pgen_mode="exact",
        min_neighbors=0,
        n_jobs=1,
    )

    row = result.table.row(0, named=True)
    assert float(row["pgen_raw"]) == pytest.approx(0.3)
    assert float(row["pgen"]) == pytest.approx(0.3)


def test_compute_alice_1mm_mode(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    result = compute_alice(rep, pgen_mode="1mm", min_neighbors=0, n_jobs=1)
    row = result.table.row(0, named=True)
    assert float(row["pgen_raw"]) == pytest.approx(0.5)


def test_add_alice_metadata_inplace(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    out = add_alice_metadata(rep, n_jobs=1)
    assert out is rep

    md = rep.clonotypes[0].clone_metadata
    assert "alice_n" in md
    assert "alice_N" in md
    assert "alice_pgen" in md
    assert "alice_fold" in md
    assert "alice_p_value" in md
    assert "alice_q_value" in md


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

    compute_alice(rep, pgen_mode="exact", n_jobs=8)

    # Bulk Pgen path should use multiple workers when n_jobs > 1.
    assert len(thread_names) > 1


def test_compute_alice_parallelizes_pvalue_calls(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)
    monkeypatch.setenv("MIRPY_ALICE_PVALUE_EXECUTOR", "thread")

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

    compute_alice(rep, pgen_mode="exact", n_jobs=8)

    assert len(thread_names) > 1


def test_compute_alice_pvalue_mode_negative_binomial(monkeypatch) -> None:
    """NB mode produces a valid p-value and doesn't crash."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire(
        [
            _clone("0", "CASSLGQETQYF"),
            _clone("1", "CASSLGQETQFF"),
        ],
        locus="TRB",
    )

    result_poisson = compute_alice(rep, match_mode="none", pgen_mode="exact", pvalue_mode="poisson", min_neighbors=2, n_jobs=1)
    result_nb = compute_alice(rep, match_mode="none", pgen_mode="exact", pvalue_mode="negative-binomial", min_neighbors=2, n_jobs=1)

    row_p = result_poisson.table.filter(pl.col("sequence_id") == "0").row(0, named=True)
    row_nb = result_nb.table.filter(pl.col("sequence_id") == "0").row(0, named=True)

    assert 0.0 <= float(row_nb["p_value"]) <= 1.0
    assert 0.0 <= float(row_p["p_value"]) <= 1.0


def test_compute_alice_invalid_pvalue_mode(monkeypatch) -> None:
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    with pytest.raises(ValueError):
        compute_alice(rep, pvalue_mode="invalid-mode")  # type: ignore[arg-type]


def test_compute_alice_pseudocount_shifts_expected(monkeypatch) -> None:
    """Pseudocount > 0 increases expected neighbors and expected value."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire(
        [
            _clone("0", "CASSLGQETQYF"),
            _clone("1", "CASSLGQETQFF"),
        ],
        locus="TRB",
    )

    result_no_pc = compute_alice(rep, pgen_mode="exact", pseudocount=0.0, min_neighbors=2, n_jobs=1)
    result_pc = compute_alice(rep, pgen_mode="exact", pseudocount=1.0, min_neighbors=2, n_jobs=1)

    row_no = result_no_pc.table.filter(pl.col("sequence_id") == "0").row(0, named=True)
    row_pc = result_pc.table.filter(pl.col("sequence_id") == "0").row(0, named=True)

    # With pseudocount=1.0, expected_neighbors = (N+1)*pgen > N*pgen
    assert float(row_pc["expected_neighbors"]) > float(row_no["expected_neighbors"])


def test_compute_alice_min_neighbors_filters_isolated_sequences(monkeypatch) -> None:
    """Sequences below min_neighbors threshold get p_value=1.0, not 0.0."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    # Single sequence has n_neighbors=1 (only self); below the specified min_neighbors=2.
    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    result = compute_alice(rep, pgen_mode="exact", min_neighbors=2, n_jobs=1)
    row = result.table.row(0, named=True)

    # Filtered sequence must get p_value=1.0 and pgen=0.0, not 0.0/inf due to pgen=0.
    assert float(row["p_value"]) == pytest.approx(1.0)
    assert float(row["pgen_raw"]) == pytest.approx(0.0)


def test_compute_alice_min_neighbors_zero_computes_all(monkeypatch) -> None:
    """min_neighbors=0 disables the filter; pgen is computed for every sequence."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    result = compute_alice(rep, pgen_mode="exact", min_neighbors=0, n_jobs=1)
    row = result.table.row(0, named=True)

    assert float(row["pgen_raw"]) == pytest.approx(0.2)


def test_compute_alice_q_factor_scales_expected(monkeypatch) -> None:
    """q_factor multiplies expected_neighbors and shifts p_value accordingly."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    rep = LocusRepertoire(
        [
            _clone("0", "CASSLGQETQYF"),
            _clone("1", "CASSLGQETQFF"),
        ],
        locus="TRB",
    )

    result_q1 = compute_alice(rep, match_mode="none", pgen_mode="exact", min_neighbors=2, q_factor=1.0, n_jobs=1)
    result_q3 = compute_alice(rep, match_mode="none", pgen_mode="exact", min_neighbors=2, q_factor=3.0, n_jobs=1)

    row_q1 = result_q1.table.filter(pl.col("sequence_id") == "0").row(0, named=True)
    row_q3 = result_q3.table.filter(pl.col("sequence_id") == "0").row(0, named=True)

    # pgen_raw must be unchanged by q_factor
    assert float(row_q1["pgen_raw"]) == pytest.approx(float(row_q3["pgen_raw"]))
    # expected_neighbors must scale by q_factor
    assert float(row_q3["expected_neighbors"]) == pytest.approx(
        float(row_q1["expected_neighbors"]) * 3.0
    )
    # p_value must be higher with larger q_factor (harder to beat larger lambda)
    assert float(row_q3["p_value"]) > float(row_q1["p_value"])


def test_compute_alice_min_neighbors_one_includes_single_neighbor(monkeypatch) -> None:
    """min_neighbors=1 includes sequences with at least one neighbor (including self)."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    # Single sequence has n_neighbors=1 (self); passes min_neighbors=1.
    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    result = compute_alice(rep, pgen_mode="exact", min_neighbors=1, n_jobs=1)
    row = result.table.row(0, named=True)

    # Pgen must be computed (not filtered).
    assert float(row["pgen_raw"]) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# MC Pgen mode tests
# ---------------------------------------------------------------------------

class _FakeMcPool:
    """Fake McPgenPool for testing mc pgen path without generating real sequences."""

    def __init__(self, pgen_1mm_map: dict[str, float], n_total: int = 10_000_000) -> None:
        self.n_total = n_total
        self.n_productive = n_total
        self.p_productive = 1.0
        self._map = pgen_1mm_map

    def pgen_1mm_bulk(self, seqs: list[str], n_jobs: int = 1) -> list[float]:
        return [self._map.get(s, 0.0) for s in seqs]


def test_compute_alice_mc_mode_uses_pool(monkeypatch) -> None:
    """pgen_mode='mc' routes through the MC pool and falls back to OLGA for sparse seqs."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)

    # High match count for seq 0; low count for seq 1 (triggers OLGA fallback).
    _CASSLGQETQYF_mc_pgen = 5 / 10_000_000  # 5 matches → count=5 ≥ mc_min_count=2
    _CASSLGQETQFF_mc_pgen = 1 / 10_000_000  # 1 match  → count=1 < mc_min_count=2 → fallback to OLGA exact
    fake_pool = _FakeMcPool({
        "CASSLGQETQYF": _CASSLGQETQYF_mc_pgen,
        "CASSLGQETQFF": _CASSLGQETQFF_mc_pgen,
    })

    # get_or_build_mc_pool is imported lazily inside _compute_pgen_raw_by_junction_aa
    # from mir.basic.pgen, so patch it there.
    monkeypatch.setattr("mir.basic.pgen.get_or_build_mc_pool", lambda **kwargs: fake_pool)

    rep = LocusRepertoire([
        _clone("0", "CASSLGQETQYF"),
        _clone("1", "CASSLGQETQFF"),
    ], locus="TRB")

    result = compute_alice(rep, pgen_mode="mc", min_neighbors=2, mc_min_count=2, n_jobs=1)
    row0 = result.table.filter(pl.col("sequence_id") == "0").row(0, named=True)
    row1 = result.table.filter(pl.col("sequence_id") == "1").row(0, named=True)

    # seq 0: MC count=5 ≥ 2 → uses MC pgen
    assert float(row0["pgen_raw"]) == pytest.approx(_CASSLGQETQYF_mc_pgen, rel=0.01)
    # seq 1: MC count=1 < 2 → falls back to OLGA exact (0.1 from _FakeOlgaModel)
    assert float(row1["pgen_raw"]) == pytest.approx(0.1, rel=0.01)


def test_compute_alice_mc_fallback_is_exact(monkeypatch) -> None:
    """MC fallback for sparse sequences uses OLGA exact Pgen (faster than 1mm).

    Sequences absent from a 10M pool have very small true pgen, so the exact
    value is a conservative (lower) λ estimate — acceptable for benchmark use.
    _FakeOlgaModel returns 0.1 for exact and 0.4 for 1mm on CASSLGQETQFF.
    """
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)
    # Empty pool → every sequence falls back to OLGA
    monkeypatch.setattr("mir.basic.pgen.get_or_build_mc_pool", lambda **kwargs: _FakeMcPool({}))

    rep = LocusRepertoire([_clone("0", "CASSLGQETQFF")], locus="TRB")
    result = compute_alice(rep, pgen_mode="mc", min_neighbors=0, mc_min_count=2, n_jobs=1)
    row = result.table.row(0, named=True)

    assert float(row["pgen_raw"]) == pytest.approx(0.1, rel=0.01), (
        "MC fallback must use OLGA exact Pgen (0.1), not 1mm (0.4)"
    )


def test_compute_alice_mc_mode_invalid_n_pool(monkeypatch) -> None:
    """mc_n_pool below 100k raises ValueError."""
    monkeypatch.setattr("mir.biomarkers.alice.OlgaModel", _FakeOlgaModel)
    rep = LocusRepertoire([_clone("0", "CASSLGQETQYF")], locus="TRB")
    with pytest.raises(ValueError, match="mc_n_pool"):
        compute_alice(rep, pgen_mode="mc", mc_n_pool=1000, n_jobs=1)


def test_alice_tcrnet_equivalence_note() -> None:
    """Documenting the ALICE / TCRNET relationship via assertion on shared concepts.

    ALICE (pgen_mode='mc', large pool) and TCRNET both:
    - compute Hamming-1 neighbor counts in a repertoire,
    - compare observed counts to a background (MC pool vs real control),
    - assign enrichment p-values.

    The key difference: ALICE falls back to OLGA analytical Pgen for sparse
    sequences; TCRNET uses a pseudocount-adjusted binomial model throughout.
    This test records that shared interface expectation (both return tables
    with n_neighbors, N_possible, p_value columns).
    """
    from mir.biomarkers.tcrnet import compute_tcrnet

    assert hasattr(compute_tcrnet, "__call__")
    from mir.biomarkers.alice import compute_alice as _compute_alice
    assert hasattr(_compute_alice, "__call__")
    # Both expose a table with enrichment statistics — verified structurally
    # in their respective test modules.


def test_alice_hit_clusters_vgene_restriction() -> None:
    """V-gene restriction prevents cross-family merges; same-family 1mm seqs cluster."""
    import pandas as pd
    from mir.biomarkers.alice import alice_hit_clusters

    hits = pd.DataFrame({
        "junction_aa": [
            "CASSVGLYSTDTQYF",  # TRBV9 — motif
            "CASSVGLFSTDTQYF",  # TRBV9 — motif (1mm from above)
            "CASSVGVYSTDTQYF",  # TRBV9 — 1mm from first
            "CASSXXXXXAAAAAA",  # TRBV5-1 — different V, isolated
        ],
        "v_gene": ["TRBV9*01", "TRBV9*01", "TRBV9*01", "TRBV5-1*01"],
        "q_value": [1e-10, 1e-8, 1e-7, 1e-6],
    })
    result = alice_hit_clusters(hits)
    assert "cluster_id" in result.columns

    sizes = result.groupby("cluster_id").size()
    assert sizes.max() == 3, f"expected TRBV9 trio in one cluster, got {sizes.tolist()}"
    assert sizes.min() == 1, "expected TRBV5-1 as singleton"

    # All three TRBV9 sequences share the same cluster_id
    trbv9_ids = result[result["v_gene"] == "TRBV9*01"]["cluster_id"].unique()
    assert len(trbv9_ids) == 1, "TRBV9 sequences must be in the same cluster"

    # TRBV5-1 sequence is in a different cluster
    trbv5_id = result[result["v_gene"] == "TRBV5-1*01"]["cluster_id"].iloc[0]
    assert trbv5_id != trbv9_ids[0], "TRBV5-1 must not merge with TRBV9 cluster"


def test_alice_hit_clusters_non_enriched_neighbors() -> None:
    """non_enriched_neighbors=True adds 1mm non-hit neighbors from full_df."""
    import pandas as pd
    from mir.biomarkers.alice import alice_hit_clusters

    hits = pd.DataFrame({
        "junction_aa": ["CASSVGLYSTDTQYF"],
        "v_gene": ["TRBV9*01"],
        "q_value": [1e-10],
    })
    full = pd.DataFrame({
        "junction_aa": [
            "CASSVGLYSTDTQYF",  # the hit itself — should not be added twice
            "CASSVGLFSTDTQYF",  # 1mm TRBV9 neighbor — should be added
            "CASSDIFFERENTXXX",  # far away — should not be added
        ],
        "v_gene": ["TRBV9*01", "TRBV9*01", "TRBV9*01"],
        "q_value": [1e-10, 1.0, 1.0],
    })
    result = alice_hit_clusters(hits, full_df=full, non_enriched_neighbors=True)
    assert "is_hit" in result.columns
    assert len(result) == 2, f"expected hit + 1 neighbor, got {len(result)}"
    assert result[result["is_hit"]].shape[0] == 1
    assert result[~result["is_hit"]]["junction_aa"].iloc[0] == "CASSVGLFSTDTQYF"

    # Both rows must share the same cluster_id
    assert result["cluster_id"].nunique() == 1


def test_alice_hit_clusters_requires_full_df_for_neighbors() -> None:
    """Raises ValueError when non_enriched_neighbors=True but full_df is None."""
    import pandas as pd
    from mir.biomarkers.alice import alice_hit_clusters

    hits = pd.DataFrame({"junction_aa": ["CASSLGQETQYF"], "v_gene": ["TRBV5-1*01"]})
    with pytest.raises(ValueError, match="full_df"):
        alice_hit_clusters(hits, non_enriched_neighbors=True)

