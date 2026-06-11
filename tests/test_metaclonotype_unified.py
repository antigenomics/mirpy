"""Tests for the unified metaclonotype clustering interface."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import polars as pl
import pytest

from mir.biomarkers.metaclonotype_cluster import (
    MetaclonotypeClusterConfig,
    cluster_metaclonotypes,
    cluster_paired_metaclonotypes,
)
from mir.common.clonotype import Clonotype
from mir.common.metaclonotype import MetaClonotypeClustering, metaclonotypes_from_labels
from mir.common.repertoire import LocusRepertoire
from mir.common.single_cell import PairedClonotype, PairedLocusRepertoire


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone(
    seq_id: str,
    junction_aa: str,
    locus: str = "TRB",
    *,
    v: str = "TRBV5-1*01",
    j: str = "TRBJ2-7*01",
    dup: int = 1,
) -> Clonotype:
    return Clonotype(
        sequence_id=seq_id,
        locus=locus,
        junction_aa=junction_aa,
        v_call=v,
        j_call=j,
        duplicate_count=dup,
        _validate=False,
    )


def _toy_trb() -> LocusRepertoire:
    return LocusRepertoire(
        [
            _clone("c1", "CASSLGQETQYF", dup=10),
            _clone("c2", "CASSLGQETQFF", dup=5),
            _clone("c3", "CASSLGQATQYF", dup=2),
            _clone("c4", "CATSLGQETQYF", dup=1),
        ],
        locus="TRB",
    )


def _toy_tra() -> LocusRepertoire:
    return LocusRepertoire(
        [
            _clone("a1", "CAASDTGNQFYF", locus="TRA", v="TRAV1-1*01", j="TRAJ12*01"),
            _clone("a2", "CAASDTGNQFFF", locus="TRA", v="TRAV1-1*01", j="TRAJ12*01"),
            _clone("a3", "CGTSGTYKYIF", locus="TRA", v="TRAV2-1*01", j="TRAJ13*01"),
            _clone("a4", "CGTSGTYKFIF", locus="TRA", v="TRAV2-1*01", j="TRAJ13*01"),
        ],
        locus="TRA",
    )


def _toy_paired_rep() -> PairedLocusRepertoire:
    return PairedLocusRepertoire(
        locus_pair="TRA_TRB",
        paired_clonotypes=[
            PairedClonotype(
                "p1",
                _clone("a1", "CAASDTGNQFYF", locus="TRA", v="TRAV1-1*01", j="TRAJ12*01"),
                _clone("b1", "CASSLGQETQYF"),
            ),
            PairedClonotype(
                "p2",
                _clone("a2", "CAASDTGNQFFF", locus="TRA", v="TRAV1-1*01", j="TRAJ12*01"),
                _clone("b2", "CASSLGQETQFF"),
            ),
            PairedClonotype(
                "p3",
                _clone("a3", "CGTSGTYKYIF", locus="TRA", v="TRAV2-1*01", j="TRAJ13*01"),
                _clone("b3", "CASSVGGSSYEQYF"),
            ),
        ],
    )


def _make_meta(ids: list[str], labels: list[int]) -> MetaClonotypeClustering:
    return metaclonotypes_from_labels(ids, labels)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_invalid_method_raises() -> None:
    with pytest.raises(ValueError, match="method"):
        MetaclonotypeClusterConfig(method="bogus")


def test_config_invalid_embed_algo_raises() -> None:
    with pytest.raises(ValueError, match="embed_cluster_algo"):
        MetaclonotypeClusterConfig(method="tcremp", embed_cluster_algo="kmeans")


def test_config_invalid_graph_algo_raises() -> None:
    with pytest.raises(ValueError, match="graph_algo"):
        MetaclonotypeClusterConfig(method="edit_distance", graph_algo="spectral")


def test_config_prefix_auto() -> None:
    cfg = MetaclonotypeClusterConfig(method="tcrdist")
    assert cfg._prefix == "tcrdist_mc"

    cfg_custom = MetaclonotypeClusterConfig(method="tcrdist", cluster_prefix="my_mc")
    assert cfg_custom._prefix == "my_mc"


# ---------------------------------------------------------------------------
# ALICE / TCRNET (metadata pre-set on clonotypes)
# ---------------------------------------------------------------------------


def _annotated_rep(method: str) -> LocusRepertoire:
    """Return a toy repertoire with fake enrichment metadata pre-set."""
    rep = _toy_trb()
    for i, c in enumerate(rep.clonotypes):
        c.clone_metadata[f"{method}_q_value"] = 0.01 if i < 2 else 0.99
    return rep


def test_cluster_alice_uses_metadata() -> None:
    rep = _annotated_rep("alice")
    cfg = MetaclonotypeClusterConfig(method="alice", q_value_max=0.05)
    meta = cluster_metaclonotypes(rep, cfg)
    assert isinstance(meta, MetaClonotypeClustering)
    assert not meta.paired
    # c1 and c2 are significant seeds; c3/c4 are Hamming-1 neighbors
    assert meta.n_clusters >= 1


def test_cluster_tcrnet_uses_metadata() -> None:
    rep = _annotated_rep("tcrnet")
    cfg = MetaclonotypeClusterConfig(method="tcrnet", q_value_max=0.05)
    meta = cluster_metaclonotypes(rep, cfg)
    assert isinstance(meta, MetaClonotypeClustering)
    assert not meta.paired


def test_cluster_alice_no_significant_clonotypes() -> None:
    """When no clonotype passes q_value_max, the result has zero clusters."""
    rep = _toy_trb()
    for c in rep.clonotypes:
        c.clone_metadata["alice_q_value"] = 1.0
    cfg = MetaclonotypeClusterConfig(method="alice", q_value_max=0.05)
    meta = cluster_metaclonotypes(rep, cfg)
    assert meta.n_clusters == 0


# ---------------------------------------------------------------------------
# Edit-distance graph
# ---------------------------------------------------------------------------


def test_cluster_edit_distance_components() -> None:
    rep = _toy_trb()
    cfg = MetaclonotypeClusterConfig(
        method="edit_distance",
        metric="hamming",
        threshold=1,
        graph_algo="components",
        min_cluster_size=1,
    )
    meta = cluster_metaclonotypes(rep, cfg)
    assert isinstance(meta, MetaClonotypeClustering)
    assert not meta.paired
    assert meta.n_clusters >= 1


def test_cluster_edit_distance_min_cluster_size() -> None:
    rep = _toy_trb()
    # min_cluster_size=10 should drop all clusters from a 4-clone toy rep
    cfg = MetaclonotypeClusterConfig(
        method="edit_distance",
        metric="hamming",
        threshold=1,
        graph_algo="components",
        min_cluster_size=10,
    )
    meta = cluster_metaclonotypes(rep, cfg)
    assert meta.n_clusters == 0


# ---------------------------------------------------------------------------
# TCRdist (mock the heavy TcrDist object)
# ---------------------------------------------------------------------------


def test_cluster_tcrdist_dispatches() -> None:
    rep = _toy_trb()
    cfg = MetaclonotypeClusterConfig(method="tcrdist", locus="TRB", max_distance=20.0)

    stub_meta = _make_meta(["c1", "c2"], [0, 0])

    with patch("mir.distances.tcrdist.TcrDist") as MockTcrDist:
        instance = MockTcrDist.from_defaults.return_value
        instance.find_metaclonotypes.return_value = stub_meta
        result = cluster_metaclonotypes(rep, cfg)

    assert result is stub_meta
    MockTcrDist.from_defaults.assert_called_once_with("TRB", "human")
    instance.find_metaclonotypes.assert_called_once()


# ---------------------------------------------------------------------------
# TCREmp single-chain (mock embedding)
# ---------------------------------------------------------------------------


def test_cluster_tcremp_single_dispatches() -> None:
    rep = _toy_trb()
    cfg = MetaclonotypeClusterConfig(
        method="tcremp",
        n_prototypes=10,
        dbscan_eps=1.0,
        dbscan_min_samples=2,
    )

    n = len(rep.clonotypes)
    # Fake embedding: two tight clusters
    fake_X = np.zeros((n, 30), dtype=np.float32)
    fake_X[0, 0] = 0.01
    fake_X[1, 0] = 0.02
    fake_X[2, 1] = 0.01
    fake_X[3, 1] = 0.02

    with patch("mir.embedding.tcremp.TCREmp") as MockTCREmp:
        instance = MockTCREmp.from_defaults.return_value
        instance.embed.return_value = fake_X
        result = cluster_metaclonotypes(rep, cfg)

    assert isinstance(result, MetaClonotypeClustering)
    assert not result.paired


# ---------------------------------------------------------------------------
# GLIPH (requires extra dict)
# ---------------------------------------------------------------------------


def test_cluster_gliph_missing_extra_raises() -> None:
    rep = _toy_trb()
    cfg = MetaclonotypeClusterConfig(method="gliph")
    with pytest.raises(ValueError, match="extra"):
        cluster_metaclonotypes(rep, cfg)


def test_cluster_gliph_dispatches() -> None:
    rep = _toy_trb()
    cfg = MetaclonotypeClusterConfig(method="gliph", graph_algo="components")
    stub_meta = _make_meta(["c1"], [0])
    extra = {"study_df": MagicMock(), "token_to_clones": {"tok": {"c1"}}}

    with patch(
        "mir.graph.token_graph.build_gliph_metaclonotypes",
        return_value=stub_meta,
    ) as mock_fn:
        result = cluster_metaclonotypes(rep, cfg, extra=extra)

    assert result is stub_meta
    mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Paired clustering — single-chain combine
# ---------------------------------------------------------------------------


def test_cluster_paired_edit_distance() -> None:
    paired_rep = _toy_paired_rep()
    cfg = MetaclonotypeClusterConfig(
        method="edit_distance",
        metric="hamming",
        threshold=1,
        graph_algo="components",
        min_cluster_size=1,
    )
    result = cluster_paired_metaclonotypes(paired_rep, cfg)
    assert result.paired
    assert result.n_clusters >= 1


def test_cluster_paired_alice_precomputed() -> None:
    """Paired ALICE: annotate per-chain reps then run combined clustering."""
    paired_rep = _toy_paired_rep()

    # Annotate chain 1 and chain 2 clonotypes with alice metadata
    for pair in paired_rep.paired_clonotypes:
        pair.clonotype1.clone_metadata["alice_q_value"] = 0.01
        pair.clonotype2.clone_metadata["alice_q_value"] = 0.01

    cfg = MetaclonotypeClusterConfig(method="alice", q_value_max=0.05)
    result = cluster_paired_metaclonotypes(paired_rep, cfg)
    assert result.paired


def test_cluster_paired_chain_configs_override() -> None:
    """Per-chain config overrides are respected for chain 1 vs chain 2."""
    paired_rep = _toy_paired_rep()

    for pair in paired_rep.paired_clonotypes:
        pair.clonotype1.clone_metadata["alice_q_value"] = 0.01
        pair.clonotype2.clone_metadata["tcrnet_q_value"] = 0.01

    cfg_base = MetaclonotypeClusterConfig(method="edit_distance", min_cluster_size=1)
    cfg_chain1 = MetaclonotypeClusterConfig(method="alice", q_value_max=0.05)
    cfg_chain2 = MetaclonotypeClusterConfig(method="tcrnet", q_value_max=0.05)

    result = cluster_paired_metaclonotypes(
        paired_rep,
        cfg_base,
        config_chain1=cfg_chain1,
        config_chain2=cfg_chain2,
    )
    assert result.paired


def test_cluster_paired_tcremp_uses_paired_embedding() -> None:
    """When method=tcremp and no chain overrides, PairedTCREmp is used."""
    paired_rep = _toy_paired_rep()
    cfg = MetaclonotypeClusterConfig(
        method="tcremp",
        locus_pair="TRA_TRB",
        n_prototypes=10,
        dbscan_eps=2.0,
        dbscan_min_samples=2,
    )

    n = len(paired_rep.paired_clonotypes)
    fake_X = np.zeros((n, 60), dtype=np.float32)
    fake_X[0, 0] = 0.01
    fake_X[1, 0] = 0.02

    with patch("mir.embedding.tcremp.PairedTCREmp") as MockPaired:
        instance = MockPaired.from_defaults.return_value
        instance.embed.return_value = fake_X
        result = cluster_paired_metaclonotypes(paired_rep, cfg)

    assert result.paired
    MockPaired.from_defaults.assert_called_once()
