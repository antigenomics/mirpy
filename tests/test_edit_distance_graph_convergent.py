"""Tests for edit distance graph with convergent rearrangements."""

from __future__ import annotations

from mir.common.clonotype import Clonotype
from mir.graph.edit_distance_graph import build_edit_distance_graph


def _clonotype(
    seq_id: str,
    nt_seq: str,
    aa_seq: str,
    v_gene: str = "TRBV1",
    j_gene: str = "TRBJ1",
    dup: int = 1,
) -> Clonotype:
    """Create a test clonotype."""
    return Clonotype(
        sequence_id=seq_id,
        junction=nt_seq,
        junction_aa=aa_seq,
        v_gene=v_gene,
        j_gene=j_gene,
        locus="TRB",
        duplicate_count=dup,
    )


def test_edit_distance_graph_convergent_rearrangements() -> None:
    """Convergent rearrangements (same junction_aa, different junction_nt) have edges.
    
    Three clonotypes with the same junction_aa but different junction_nt should
    all be connected by edges in the edit distance graph (distance 0 in junction_aa).
    """
    clonotypes = [
        _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
        _clonotype("c2", "GTA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),  # different nt, same aa
        _clonotype("c3", "AAA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),  # different nt, same aa
    ]

    g = build_edit_distance_graph(clonotypes, metric="hamming", threshold=1, nproc=1)

    assert g.vcount() == 3
    # All three should be connected to each other: c0-c1, c0-c2, c1-c2
    assert g.ecount() == 3  # complete triangle
    
    # Verify edges exist
    assert g.are_adjacent(0, 1)
    assert g.are_adjacent(0, 2)
    assert g.are_adjacent(1, 2)


def test_edit_distance_graph_convergent_with_v_gene_match() -> None:
    """With v_gene_match=True, convergent clonotypes with different V genes should not be connected.
    
    Two convergent clonotypes (same junction_aa) but with different V genes
    should not have an edge when v_gene_match=True.
    """
    clonotypes = [
        _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
        _clonotype("c2", "GTA", "MVA", v_gene="TRBV2", j_gene="TRBJ1"),  # different nt, same aa, different V
    ]

    g = build_edit_distance_graph(
        clonotypes, metric="hamming", threshold=1, v_gene_match=True, nproc=1
    )

    assert g.vcount() == 2
    assert g.ecount() == 0  # no edge because V genes don't match


def test_edit_distance_graph_convergent_with_matching_v_and_j() -> None:
    """Convergent clonotypes with matching V and J should be connected.
    
    Two convergent clonotypes (same junction_aa) with same V and J genes
    should have an edge even with matching constraints.
    """
    clonotypes = [
        _clonotype("c1", "ATG", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),
        _clonotype("c2", "GTA", "MVA", v_gene="TRBV1", j_gene="TRBJ1"),  # same V and J, different nt/aa coding
    ]

    g = build_edit_distance_graph(
        clonotypes,
        metric="hamming",
        threshold=1,
        v_gene_match=True,
        nproc=1,
    )

    assert g.vcount() == 2
    assert g.ecount() == 1  # edge exists because V and J match and junction_aa distance is 0
    assert g.are_adjacent(0, 1)
