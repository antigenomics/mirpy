"""Unit tests for neighborhood enrichment statistics."""

from __future__ import annotations

import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.graph.neighborhood_enrichment import (
    add_neighborhood_enrichment_metadata,
    add_neighborhood_metadata,
    compute_neighborhood_stats,
)


def _clonotype(
    seq_id: str,
    nt_seq: str,
    aa_seq: str,
    v_gene: str = "TRBV1",
    j_gene: str = "TRBJ1",
    locus: str = "TRB",
    dup: int = 1,
) -> Clonotype:
    """Create a test clonotype."""
    return Clonotype(
        sequence_id=seq_id,
        junction=nt_seq,
        junction_aa=aa_seq,
        v_gene=v_gene,
        j_gene=j_gene,
        locus=locus,
        duplicate_count=dup,
    )


def test_neighborhood_single_clonotype_hamming() -> None:
    """Single clonotype should have itself as the only neighbor."""
    rep = LocusRepertoire(
        clonotypes=[_clonotype("c1", "ATG", "M", v_gene="TRBV1", j_gene="TRBJ1")],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1)

    assert len(stats) == 1
    assert stats["c1"]["neighbor_count"] == 1
    assert stats["c1"]["potential_neighbors"] == 1


def test_neighborhood_identical_sequences_hamming() -> None:
    """Identical sequences should be neighbors with hamming distance 0."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c3", "CCG", "PRK", v_gene="TRBV1", j_gene="TRBJ1"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1)

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2
    assert stats["c2"]["neighbor_count"] == 2  # c1 and c2
    assert stats["c3"]["neighbor_count"] == 1  # only itself


def test_neighborhood_one_hamming_difference() -> None:
    """Sequences with 1 AA difference should be neighbors."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA"),
            _clonotype("c2", "ATG", "AVA"),  # 1 difference (M vs A at pos 0)
            _clonotype("c3", "ATG", "MLA"),  # 1 difference (V vs L at pos 1)
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1)

    assert stats["c1"]["neighbor_count"] == 3  # neighbors with c1, c2, c3
    assert stats["c2"]["neighbor_count"] == 2  # neighbors with c1, c3 (2 diff from c3)
    assert stats["c3"]["neighbor_count"] == 2  # neighbors with c1, c2 (2 diff from c2)


def test_neighborhood_hamming_unequal_length_no_neighbors() -> None:
    """Hamming distance: unequal length sequences cannot be neighbors."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA"),
            _clonotype("c2", "ATG", "MV"),  # shorter
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1)

    assert stats["c1"]["neighbor_count"] == 1  # only itself
    assert stats["c2"]["neighbor_count"] == 1  # only itself


def test_neighborhood_levenshtein_allows_unequal_length() -> None:
    """Levenshtein distance: unequal length sequences can be neighbors."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA"),
            _clonotype("c2", "ATG", "MV"),  # 1 edit (deletion of A)
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="levenshtein", threshold=1)

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2
    assert stats["c2"]["neighbor_count"] == 2  # c1 and c2


def test_neighborhood_match_v_gene() -> None:
    """With match_v_gene=True, neighbors must have same v_gene."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c3", "ATG", "MVA", v_gene="TRBV2", j_gene="TRBJ1"),  # different V
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, match_v_gene=True)

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2 (same V)
    assert stats["c1"]["potential_neighbors"] == 2
    assert stats["c3"]["neighbor_count"] == 1  # only itself
    assert stats["c3"]["potential_neighbors"] == 1


def test_neighborhood_match_j_gene() -> None:
    """With match_j_gene=True, neighbors must have same j_gene."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_gene="TRBV2", j_gene="TRBJ1"),
            _clonotype("c3", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ2"),  # different J
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, match_j_gene=True)

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2 (same J)
    assert stats["c1"]["potential_neighbors"] == 2
    assert stats["c3"]["neighbor_count"] == 1  # only itself
    assert stats["c3"]["potential_neighbors"] == 1


def test_neighborhood_match_v_and_j_gene() -> None:
    """With both match_v_gene and match_j_gene=True, both must match."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c3", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ2"),
            _clonotype("c4", "ATG", "MVA", v_gene="TRBV2", j_gene="TRBJ1"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(
        rep, metric="hamming", threshold=1, match_v_gene=True, match_j_gene=True
    )

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2 (both match)
    assert stats["c1"]["potential_neighbors"] == 2


def test_neighborhood_sample_repertoire() -> None:
    """Compute neighborhood stats for multi-locus SampleRepertoire."""
    trb = LocusRepertoire(
        clonotypes=[
            _clonotype("b1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1", locus="TRB"),
            _clonotype("b2", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1", locus="TRB"),
        ],
        locus="TRB",
    )
    tra = LocusRepertoire(
        clonotypes=[
            _clonotype("a1", "TTT", "FFF", v_gene="TRAV1", j_gene="TRAJ1", locus="TRA"),
        ],
        locus="TRA",
    )

    rep = SampleRepertoire(loci={"TRB": trb, "TRA": tra}, sample_id="s1")

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1)

    assert len(stats) == 3
    assert stats["b1"]["neighbor_count"] == 2
    assert stats["a1"]["neighbor_count"] == 1


def test_add_neighborhood_metadata() -> None:
    """Test in-place metadata addition to clonotypes."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA"),
            _clonotype("c2", "ATG", "MVA"),
        ],
        locus="TRB",
    )

    add_neighborhood_metadata(rep, metric="hamming", threshold=1)

    assert rep.clonotypes[0].clone_metadata["neighborhood_count"] == 2
    assert rep.clonotypes[0].clone_metadata["neighborhood_potential"] == 2
    assert rep.clonotypes[1].clone_metadata["neighborhood_count"] == 2
    assert rep.clonotypes[1].clone_metadata["neighborhood_potential"] == 2


def test_neighborhood_threshold() -> None:
    """Test that threshold correctly filters neighbors."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA"),
            _clonotype("c2", "ATG", "LVA"),  # 1 difference from c1
            _clonotype("c3", "ATG", "LLA"),  # 2 differences from c1, 1 from c2
        ],
        locus="TRB",
    )

    stats_t1 = compute_neighborhood_stats(rep, metric="hamming", threshold=1)
    stats_t2 = compute_neighborhood_stats(rep, metric="hamming", threshold=2)

    # At threshold=1:
    # c1: neighbors are c1 (self, 0 diff) and c2 (1 diff) = 2
    # c2: neighbors are c1 (1 diff), c2 (self, 0 diff), c3 (1 diff) = 3
    # c3: neighbors are c2 (1 diff), c3 (self, 0 diff) = 2
    assert stats_t1["c1"]["neighbor_count"] == 2
    assert stats_t1["c2"]["neighbor_count"] == 3
    assert stats_t1["c3"]["neighbor_count"] == 2

    # At threshold=2: all are neighbors of all
    assert stats_t2["c1"]["neighbor_count"] == 3
    assert stats_t2["c2"]["neighbor_count"] == 3
    assert stats_t2["c3"]["neighbor_count"] == 3


def test_neighborhood_empty_repertoire() -> None:
    """Empty repertoire should return empty stats."""
    rep = LocusRepertoire(clonotypes=[], locus="TRB")

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1)

    assert stats == {}


def test_neighborhood_invalid_metric() -> None:
    """Invalid metric should raise ValueError."""
    rep = LocusRepertoire(
        clonotypes=[_clonotype("c1", "ATG", "MVA")],
        locus="TRB",
    )

    with pytest.raises(ValueError, match="metric must be"):
        compute_neighborhood_stats(rep, metric="invalid")


def test_neighborhood_convergent_rearrangements() -> None:
    """Convergent rearrangements (same junction_aa, different junction_nt) are neighbors.
    
    Three clonotypes with the same junction_aa but different junction_nt should
    all be neighbors of each other (distance 0 in junction_aa space).
    Each should have neighbor_count = 3 (itself + 2 others).
    """
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c2", "GTA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),  # different nt, same aa
            _clonotype("c3", "AAA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),  # different nt, same aa
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1)

    # All three should be neighbors of each other (same junction_aa = distance 0)
    assert stats["c1"]["neighbor_count"] == 3  # c1, c2, c3 (2 + 1 original)
    assert stats["c2"]["neighbor_count"] == 3  # c1, c2, c3 (2 + 1 original)
    assert stats["c3"]["neighbor_count"] == 3  # c1, c2, c3 (2 + 1 original)


def test_neighborhood_stats_background_pseudocount() -> None:
    """Background mode adds +1 pseudocount for query clonotype membership."""
    query = LocusRepertoire(
        clonotypes=[
            _clonotype("q1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("q2", "GTA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
        ],
        locus="TRB",
    )
    background = LocusRepertoire(
        clonotypes=[
            _clonotype("b1", "CCC", "CCC", v_gene="TRBV2", j_gene="TRBJ2"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(query, background=background, metric="hamming", threshold=1)

    # No true background neighbors within threshold, but pseudocount adds one.
    assert stats["q1"]["neighbor_count"] == 1
    assert stats["q1"]["potential_neighbors"] == 2
    assert stats["q2"]["neighbor_count"] == 1
    assert stats["q2"]["potential_neighbors"] == 2


def test_neighborhood_stats_background_same_as_query_is_equivalent() -> None:
    """Syntax sugar: explicit background=self must equal self-mode stats."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c2", "GTA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c3", "AAA", "MLA", v_gene="TRBV1", j_gene="TRBJ1"),
        ],
        locus="TRB",
    )

    stats_self = compute_neighborhood_stats(rep, metric="hamming", threshold=1)
    stats_explicit = compute_neighborhood_stats(
        rep,
        background=rep,
        metric="hamming",
        threshold=1,
    )

    assert stats_explicit == stats_self


def test_add_neighborhood_enrichment_metadata_background_equals_self() -> None:
    """Parent/background stats and enrichment are consistent when background=self."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
            _clonotype("c2", "GTA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
        ],
        locus="TRB",
    )

    add_neighborhood_enrichment_metadata(
        rep,
        background=rep,
        metric="hamming",
        threshold=1,
        metadata_prefix="nbr",
    )

    for clonotype in rep.clonotypes:
        assert clonotype.clone_metadata["nbr_parent_count"] == clonotype.clone_metadata["nbr_background_count"]
        assert clonotype.clone_metadata["nbr_parent_potential"] == clonotype.clone_metadata["nbr_background_potential"]
        assert clonotype.clone_metadata["nbr_enrichment"] == 1.0
