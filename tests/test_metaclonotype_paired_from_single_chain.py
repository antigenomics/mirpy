"""Tests for paired_metaclonotypes_from_single_chain."""

from __future__ import annotations

import pytest

from mir.common.clonotype import Clonotype
from mir.common.metaclonotype import MetaClonotypeDefinition, metaclonotypes_from_labels
from mir.common.single_cell import PairedClonotype
from mir.utils.metaclonotype_clustering import paired_metaclonotypes_from_single_chain


def _make_clone(
    seq_id: str,
    junction_aa: str,
    locus: str = "TRB",
    *,
    v: str = "TRBV5-1*01",
    j: str = "TRBJ2-7*01",
) -> Clonotype:
    return Clonotype(
        sequence_id=seq_id,
        locus=locus,
        junction_aa=junction_aa,
        v_gene=v,
        j_gene=j,
        duplicate_count=1,
        _validate=False,
    )


def _toy_pairs() -> list[PairedClonotype]:
    """Three TRA/TRB pairs sharing two TRB clusters and two TRA clusters."""
    return [
        PairedClonotype(
            pair_id="p1",
            clonotype1=_make_clone("a1", "CAASDTGNQFYF", locus="TRA"),
            clonotype2=_make_clone("b1", "CASSLGQETQYF"),
        ),
        PairedClonotype(
            pair_id="p2",
            clonotype1=_make_clone("a2", "CAASDTGNQFFF", locus="TRA"),
            clonotype2=_make_clone("b2", "CASSLGQETQFF"),
        ),
        PairedClonotype(
            pair_id="p3",
            clonotype1=_make_clone("a3", "CGTSGTYKYIF", locus="TRA"),
            clonotype2=_make_clone("b3", "CASSVGGSSYEQYF"),
        ),
    ]


def _meta_trb() -> MetaClonotypeDefinition:
    # b1 and b2 are in cluster "trb_0"; b3 is alone in "trb_1"
    return metaclonotypes_from_labels(
        ["b1", "b2", "b3"],
        [0, 0, 1],
        include_noise=False,
    )


def _meta_tra() -> MetaClonotypeDefinition:
    # a1 and a2 cluster together ("0"), a3 alone ("2")
    return metaclonotypes_from_labels(
        ["a1", "a2", "a3"],
        [0, 0, 2],
        include_noise=False,
    )


def test_basic_combination() -> None:
    pairs = _toy_pairs()
    meta1 = _meta_tra()
    meta2 = _meta_trb()
    result = paired_metaclonotypes_from_single_chain(pairs, meta1, meta2)

    assert result.paired
    # p1 → "0.0", p2 → "0.0", p3 → "2.1" (after cluster renaming in _meta_trb)
    # _meta_trb labels: b1→0, b2→0, b3→1  → cluster "0" and "1"
    # _meta_tra labels: a1→0, a2→0, a3→2  → cluster "0" and "2"
    cluster_ids = set(result.table["cluster_id"].to_list())
    assert "0.0" in cluster_ids   # p1 and p2
    assert "2.1" in cluster_ids   # p3
    assert len(cluster_ids) == 2


def test_pair_count_and_representative() -> None:
    pairs = _toy_pairs()
    result = paired_metaclonotypes_from_single_chain(pairs, _meta_tra(), _meta_trb())

    # cluster "0.0" has two members (p1, p2)
    members_00 = result.members_of("0.0")
    assert len(members_00) == 2
    # Exactly one representative per cluster
    reps = result.representatives()
    assert reps["cluster_id"].to_list().count("0.0") == 1
    assert reps["cluster_id"].to_list().count("2.1") == 1


def test_unassigned_excluded_by_default() -> None:
    """Pairs where a chain has no cluster assignment are dropped by default."""
    pairs = _toy_pairs()
    # Only b1 is assigned in chain 2
    meta2_partial = metaclonotypes_from_labels(["b1"], [0])
    result = paired_metaclonotypes_from_single_chain(pairs, _meta_tra(), meta2_partial)
    # Only p1 survives (a1 → "0", b1 → "0")
    assert result.n_clusters == 1
    assert len(result.table) == 1


def test_unassigned_included_when_requested() -> None:
    """Unassigned chain uses the placeholder label when include_unassigned=True."""
    pairs = _toy_pairs()
    meta2_partial = metaclonotypes_from_labels(["b1"], [0])
    result = paired_metaclonotypes_from_single_chain(
        pairs,
        _meta_tra(),
        meta2_partial,
        include_unassigned=True,
        unassigned_label="X",
    )
    cluster_ids = set(result.table["cluster_id"].to_list())
    # p2: a2→"0", b2→"X" → "0.X"
    # p3: a3→"2", b3→"X" → "2.X"
    assert "0.X" in cluster_ids
    assert "2.X" in cluster_ids


def test_custom_separator() -> None:
    pairs = _toy_pairs()
    result = paired_metaclonotypes_from_single_chain(
        pairs, _meta_tra(), _meta_trb(), cluster_separator="|"
    )
    assert all("|" in cid for cid in result.table["cluster_id"].to_list())


def test_raises_on_paired_input() -> None:
    import polars as pl

    paired_meta = MetaClonotypeDefinition(
        pl.DataFrame(
            {
                "cluster_id": ["x"],
                "clonotype_id_1": ["a1"],
                "clonotype_id_2": ["b1"],
            }
        ),
        paired=True,
    )
    with pytest.raises(ValueError, match="single-chain"):
        paired_metaclonotypes_from_single_chain([], paired_meta, _meta_trb())


def test_empty_input() -> None:
    result = paired_metaclonotypes_from_single_chain([], _meta_tra(), _meta_trb())
    assert result.paired
    assert result.n_clusters == 0
    assert len(result.table) == 0
