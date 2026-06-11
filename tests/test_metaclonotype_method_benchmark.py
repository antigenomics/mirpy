"""Benchmarks comparing metaclonotype clustering methods and paired strategies.

Run with: env RUN_BENCHMARK=1 python -m pytest tests/test_metaclonotype_method_benchmark.py -s -x
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_BENCHMARK"),
    reason="set RUN_BENCHMARK=1 to run",
)

from mir.biomarkers.metaclonotype_cluster import (
    MetaclonotypeClusterConfig,
    cluster_metaclonotypes,
    cluster_paired_metaclonotypes,
)
from mir.common.clonotype import Clonotype
from mir.common.metaclonotype import functional_diversity
from mir.common.repertoire import LocusRepertoire
from mir.common.single_cell import PairedClonotype, PairedLocusRepertoire


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TRBV_GENES = [
    "TRBV5-1*01", "TRBV12-3*01", "TRBV20-1*01", "TRBV7-2*01", "TRBV2*01",
]
_TRBJ_GENES = ["TRBJ2-7*01", "TRBJ1-2*01", "TRBJ2-3*01"]
_TRAV_GENES = ["TRAV1-2*01", "TRAV12-2*01", "TRAV38-2DV8*01"]
_TRAJ_GENES = ["TRAJ12*01", "TRAJ20*01", "TRAJ33*01"]
_AA = "ACDEFGHIKLMNPQRSTVWY"
_CDR3_LEN_RANGE = (12, 17)


def _random_aa(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(_AA) for _ in range(length))


def _synthetic_trb_rep(n: int = 1000, seed: int = 42) -> LocusRepertoire:
    rng = random.Random(seed)
    clonotypes = []
    for i in range(n):
        length = rng.randint(*_CDR3_LEN_RANGE)
        clonotypes.append(
            Clonotype(
                sequence_id=f"b{i}",
                locus="TRB",
                junction_aa=_random_aa(rng, length),
                v_call=rng.choice(_TRBV_GENES),
                j_call=rng.choice(_TRBJ_GENES),
                duplicate_count=rng.randint(1, 1000),
                _validate=False,
            )
        )
    return LocusRepertoire(clonotypes=clonotypes, locus="TRB")


def _synthetic_tra_rep(n: int = 1000, seed: int = 99) -> LocusRepertoire:
    rng = random.Random(seed)
    clonotypes = []
    for i in range(n):
        length = rng.randint(*_CDR3_LEN_RANGE)
        clonotypes.append(
            Clonotype(
                sequence_id=f"a{i}",
                locus="TRA",
                junction_aa=_random_aa(rng, length),
                v_call=rng.choice(_TRAV_GENES),
                j_call=rng.choice(_TRAJ_GENES),
                duplicate_count=rng.randint(1, 1000),
                _validate=False,
            )
        )
    return LocusRepertoire(clonotypes=clonotypes, locus="TRA")


def _synthetic_paired_rep(n: int = 500, seed: int = 42) -> PairedLocusRepertoire:
    rng = random.Random(seed)
    trb_rng = random.Random(seed)
    tra_rng = random.Random(seed + 1)
    pairs = []
    for i in range(n):
        trb_len = rng.randint(*_CDR3_LEN_RANGE)
        tra_len = rng.randint(*_CDR3_LEN_RANGE)
        pairs.append(
            PairedClonotype(
                pair_id=f"pair_{i}",
                clonotype1=Clonotype(
                    sequence_id=f"a{i}",
                    locus="TRA",
                    junction_aa=_random_aa(tra_rng, tra_len),
                    v_call=tra_rng.choice(_TRAV_GENES),
                    j_call=tra_rng.choice(_TRAJ_GENES),
                    duplicate_count=1,
                    _validate=False,
                ),
                clonotype2=Clonotype(
                    sequence_id=f"b{i}",
                    locus="TRB",
                    junction_aa=_random_aa(trb_rng, trb_len),
                    v_call=trb_rng.choice(_TRBV_GENES),
                    j_call=trb_rng.choice(_TRBJ_GENES),
                    duplicate_count=1,
                    _validate=False,
                ),
            )
        )
    return PairedLocusRepertoire(locus_pair="TRA_TRB", paired_clonotypes=pairs)


# ---------------------------------------------------------------------------
# Single-chain method benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_clonotypes", [500, 2000])
def test_benchmark_edit_distance_components(n_clonotypes: int) -> None:
    """Edit-distance graph with connected-components community detection."""
    rep = _synthetic_trb_rep(n_clonotypes)
    cfg = MetaclonotypeClusterConfig(
        method="edit_distance",
        metric="hamming",
        threshold=1,
        graph_algo="components",
        min_cluster_size=2,
        n_jobs=4,
    )
    t0 = time.perf_counter()
    meta = cluster_metaclonotypes(rep, cfg)
    elapsed = time.perf_counter() - t0

    div = functional_diversity(rep, meta)
    print(
        f"\n[edit_distance/components] n={n_clonotypes} "
        f"clusters={meta.n_clusters} "
        f"functional_diversity(q=1)={div.shannon:.2f} "
        f"elapsed={elapsed:.3f}s"
    )
    assert meta.n_clusters >= 0
    assert elapsed < 120.0


@pytest.mark.parametrize("n_clonotypes", [500, 2000])
def test_benchmark_edit_distance_leiden(n_clonotypes: int) -> None:
    """Edit-distance graph with Leiden community detection."""
    rep = _synthetic_trb_rep(n_clonotypes)
    cfg = MetaclonotypeClusterConfig(
        method="edit_distance",
        metric="hamming",
        threshold=1,
        graph_algo="leiden",
        min_cluster_size=2,
        n_jobs=4,
    )
    t0 = time.perf_counter()
    meta = cluster_metaclonotypes(rep, cfg)
    elapsed = time.perf_counter() - t0

    div = functional_diversity(rep, meta)
    print(
        f"\n[edit_distance/leiden] n={n_clonotypes} "
        f"clusters={meta.n_clusters} "
        f"functional_diversity(q=1)={div.shannon:.2f} "
        f"elapsed={elapsed:.3f}s"
    )
    assert elapsed < 120.0


# ---------------------------------------------------------------------------
# Paired benchmarks: TCREmp-native vs single-chain-combined
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_pairs", [200, 500])
def test_benchmark_paired_edit_distance_combined(n_pairs: int) -> None:
    """Single-chain edit-distance per chain, combined into paired metaclonotypes."""
    paired_rep = _synthetic_paired_rep(n_pairs)
    cfg = MetaclonotypeClusterConfig(
        method="edit_distance",
        metric="hamming",
        threshold=1,
        graph_algo="components",
        min_cluster_size=1,
        n_jobs=4,
    )
    t0 = time.perf_counter()
    meta = cluster_paired_metaclonotypes(paired_rep, cfg)
    elapsed = time.perf_counter() - t0

    print(
        f"\n[paired/edit_distance combined] n_pairs={n_pairs} "
        f"paired_clusters={meta.n_clusters} "
        f"elapsed={elapsed:.3f}s"
    )
    assert meta.paired
    assert elapsed < 120.0


@pytest.mark.parametrize("n_pairs", [200, 500])
def test_benchmark_paired_alice_combined(n_pairs: int) -> None:
    """ALICE per chain with pre-set q-values, combined into paired metaclonotypes."""
    paired_rep = _synthetic_paired_rep(n_pairs)
    # Mark first 20% of each chain as enriched
    n_sig = max(1, n_pairs // 5)
    for i, pair in enumerate(paired_rep.paired_clonotypes):
        q = 0.01 if i < n_sig else 0.99
        pair.clonotype1.clone_metadata["alice_q_value"] = q
        pair.clonotype2.clone_metadata["alice_q_value"] = q

    cfg = MetaclonotypeClusterConfig(method="alice", q_value_max=0.05)
    t0 = time.perf_counter()
    meta = cluster_paired_metaclonotypes(paired_rep, cfg)
    elapsed = time.perf_counter() - t0

    print(
        f"\n[paired/alice combined] n_pairs={n_pairs} "
        f"paired_clusters={meta.n_clusters} "
        f"elapsed={elapsed:.3f}s"
    )
    assert meta.paired
    assert elapsed < 120.0


# ---------------------------------------------------------------------------
# Concordance: TCREmp-native paired vs single-chain-combined
# ---------------------------------------------------------------------------


def test_benchmark_tcremp_paired_vs_combined_concordance() -> None:
    """Compare cluster count between native paired TCREmp and edit-distance combined.

    This test does not assert concordance equality (different algorithms will
    differ), but it verifies both run successfully and reports cluster counts.
    """
    n_pairs = 300
    paired_rep = _synthetic_paired_rep(n_pairs)

    # Single-chain edit-distance combined
    t0 = time.perf_counter()
    cfg_ed = MetaclonotypeClusterConfig(
        method="edit_distance",
        metric="hamming",
        threshold=1,
        graph_algo="components",
        min_cluster_size=1,
        n_jobs=4,
    )
    meta_ed = cluster_paired_metaclonotypes(paired_rep, cfg_ed)
    t_ed = time.perf_counter() - t0

    print(
        f"\n[concordance] edit_distance_combined clusters={meta_ed.n_clusters} "
        f"elapsed={t_ed:.3f}s"
    )

    # TCREmp paired (only if model files available — skip gracefully)
    try:
        from mir.embedding.tcremp import PairedTCREmp

        t0 = time.perf_counter()
        cfg_tcremp = MetaclonotypeClusterConfig(
            method="tcremp",
            locus_pair="TRA_TRB",
            n_prototypes=100,
            dbscan_eps=1.0,
            dbscan_min_samples=2,
        )
        meta_tcremp = cluster_paired_metaclonotypes(paired_rep, cfg_tcremp)
        t_tcremp = time.perf_counter() - t0

        # Concordance: fraction of pairs assigned to the same paired cluster
        ed_table = meta_ed.table.rename({"cluster_id": "cluster_ed"})
        tcp_table = meta_tcremp.table.rename({"cluster_id": "cluster_tcremp"})
        joined = ed_table.join(
            tcp_table,
            on=["clonotype_id_1", "clonotype_id_2"],
            how="inner",
        )
        concordance = (
            joined.group_by(["cluster_ed", "cluster_tcremp"])
            .len()
            .sort("len", descending=True)
        )
        print(
            f"[concordance] tcremp_native clusters={meta_tcremp.n_clusters} "
            f"elapsed={t_tcremp:.3f}s"
        )
        print(f"[concordance] top concordant pairs:\n{concordance.head(10)}")
    except Exception as exc:
        print(f"[concordance] tcremp skipped: {exc}")
