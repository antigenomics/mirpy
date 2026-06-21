"""Unit tests for neighborhood enrichment statistics."""

from __future__ import annotations

import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
import mir.graph.neighborhood_enrichment as _ne_mod
from mir.graph.neighborhood_enrichment import (
    add_neighborhood_enrichment_metadata,
    add_neighborhood_metadata,
    compute_neighborhood_stats,
    compute_neighborhood_stats_by_locus,
)


def _clonotype(
    seq_id: str,
    nt_seq: str,
    aa_seq: str,
    v_call: str = "TRBV1",
    j_call: str = "TRBJ1",
    locus: str = "TRB",
    dup: int = 1,
) -> Clonotype:
    """Create a test clonotype."""
    return Clonotype(
        sequence_id=seq_id,
        junction=nt_seq,
        junction_aa=aa_seq,
        v_call=v_call,
        j_call=j_call,
        locus=locus,
        duplicate_count=dup,
    )


def test_neighborhood_single_clonotype_hamming() -> None:
    """Single clonotype should have itself as the only neighbor."""
    rep = LocusRepertoire(
        clonotypes=[_clonotype("c1", "ATG", "M", v_call="TRBV1", j_call="TRBJ1")],
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
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c3", "CCG", "PRK", v_call="TRBV1", j_call="TRBJ1"),
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


def test_neighborhood_match_v_call() -> None:
    """With match_v_call=True, neighbors must have same v_call."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c3", "ATG", "MVA", v_call="TRBV2", j_call="TRBJ1"),  # different V
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, match_v_call=True)

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2 (same V)
    assert stats["c1"]["potential_neighbors"] == 2
    assert stats["c3"]["neighbor_count"] == 1  # only itself
    assert stats["c3"]["potential_neighbors"] == 1


def test_neighborhood_match_v_gene_ignores_allele_suffix() -> None:
    """V-gene matching should treat TRBVx and TRBVx*01 as the same key."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_call="TRBV1*01", j_call="TRBJ2"),
            _clonotype("c3", "ATG", "MVA", v_call="TRBV2*01", j_call="TRBJ1"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=0, match_v_call=True)

    assert stats["c1"]["neighbor_count"] == 2
    assert stats["c1"]["potential_neighbors"] == 2


def test_neighborhood_specific_allele_does_not_match_different_allele() -> None:
    """*01 and *02 are distinct alleles — should not count as neighbors."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1*01", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_call="TRBV1*02", j_call="TRBJ1"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=0, match_v_call=True)

    # c1 (*01) should NOT count c2 (*02) as a neighbor.
    assert stats["c1"]["neighbor_count"] == 1  # only self
    assert stats["c1"]["potential_neighbors"] == 1


def test_neighborhood_bare_query_matches_all_alleles() -> None:
    """Bare-gene query is a wildcard and finds neighbors in all allele groups."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c_bare", "ATG", "MVA", v_call="TRBV1",    j_call="TRBJ1"),
            _clonotype("c_01",   "ATG", "MVA", v_call="TRBV1*01", j_call="TRBJ1"),
            _clonotype("c_02",   "ATG", "MVA", v_call="TRBV1*02", j_call="TRBJ1"),
            _clonotype("c_other","ATG", "MVA", v_call="TRBV2*01", j_call="TRBJ1"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=0, match_v_call=True)

    # c_bare (wildcard) should find c_01, c_02, and itself; not c_other.
    assert stats["c_bare"]["neighbor_count"] == 3
    assert stats["c_bare"]["potential_neighbors"] == 3


def test_neighborhood_match_j_call() -> None:
    """With match_j_call=True, neighbors must have same j_call."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_call="TRBV2", j_call="TRBJ1"),
            _clonotype("c3", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ2"),  # different J
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, match_j_call=True)

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2 (same J)
    assert stats["c1"]["potential_neighbors"] == 2
    assert stats["c3"]["neighbor_count"] == 1  # only itself
    assert stats["c3"]["potential_neighbors"] == 1


def test_neighborhood_match_v_and_j_gene() -> None:
    """With both match_v_call and match_j_call=True, both must match."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c3", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ2"),
            _clonotype("c4", "ATG", "MVA", v_call="TRBV2", j_call="TRBJ1"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(
        rep, metric="hamming", threshold=1, match_v_call=True, match_j_call=True
    )

    assert stats["c1"]["neighbor_count"] == 2  # c1 and c2 (both match)
    assert stats["c1"]["potential_neighbors"] == 2


def test_neighborhood_sample_repertoire() -> None:
    """Compute neighborhood stats for multi-locus SampleRepertoire."""
    trb = LocusRepertoire(
        clonotypes=[
            _clonotype("b1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1", locus="TRB"),
            _clonotype("b2", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1", locus="TRB"),
        ],
        locus="TRB",
    )
    tra = LocusRepertoire(
        clonotypes=[
            _clonotype("a1", "TTT", "FFF", v_call="TRAV1", j_call="TRAJ1", locus="TRA"),
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
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "GTA", "MVA", v_call="TRBV1", j_call="TRBJ1"),  # different nt, same aa
            _clonotype("c3", "AAA", "MVA", v_call="TRBV1", j_call="TRBJ1"),  # different nt, same aa
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
            _clonotype("q1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("q2", "GTA", "MVA", v_call="TRBV1", j_call="TRBJ1"),
        ],
        locus="TRB",
    )
    background = LocusRepertoire(
        clonotypes=[
            _clonotype("b1", "CCC", "CCC", v_call="TRBV2", j_call="TRBJ2"),
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
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "GTA", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c3", "AAA", "MLA", v_call="TRBV1", j_call="TRBJ1"),
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


def test_neighborhood_parallel_matches_single_worker() -> None:
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "CASSLGQETQYF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "CASSLGQETQFF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c3", "ATG", "CASSLGQDTQYF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c4", "ATG", "CASSPGQETQYF", v_call="TRBV2", j_call="TRBJ1"),
            _clonotype("c5", "ATG", "CASSPGQETQYF", v_call="TRBV2", j_call="TRBJ2"),
        ],
        locus="TRB",
    )

    serial = compute_neighborhood_stats(
        rep,
        metric="hamming",
        threshold=1,
        match_v_call=False,
        match_j_call=False,
        n_jobs=1,
    )
    parallel = compute_neighborhood_stats(
        rep,
        metric="hamming",
        threshold=1,
        match_v_call=False,
        match_j_call=False,
        n_jobs=4,
    )

    assert parallel == serial


def test_add_neighborhood_enrichment_metadata_background_equals_self() -> None:
    """Parent/background stats and enrichment are consistent when background=self."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "MVA", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "GTA", "MVA", v_call="TRBV1", j_call="TRBJ1"),
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


def test_neighborhood_long_queries_over_33_and_64_aa() -> None:
    rep_h = LocusRepertoire(
        clonotypes=[
            _clonotype("h1", "ATG", "C" * 40),
            _clonotype("h2", "ATG", ("C" * 39) + "A"),
        ],
        locus="TRB",
    )
    rep_l = LocusRepertoire(
        clonotypes=[
            _clonotype("l1", "ATG", "G" * 70),
            _clonotype("l2", "ATG", ("G" * 70) + "A"),
        ],
        locus="TRB",
    )

    stats_h = compute_neighborhood_stats(rep_h, metric="hamming", threshold=1)
    stats_l = compute_neighborhood_stats(rep_l, metric="levenshtein", threshold=1)

    assert stats_h["h1"]["neighbor_count"] == 2
    assert stats_h["h2"]["neighbor_count"] == 2
    assert stats_l["l1"]["neighbor_count"] == 2
    assert stats_l["l2"]["neighbor_count"] == 2


def test_neighborhood_grouped_vj_explicit_values() -> None:
    """Grouped-trie VJ search counts only same-V+J neighbours; potential = group size."""
    # c1/c2/c3 share TRBV1/TRBJ1; c4 has TRBV2/TRBJ1; c5 has TRBV2/TRBJ2.
    # c1 vs c2: Hamming-1 (pos 10 Y→F). c1 vs c3: Hamming-1 (pos 7 E→D).
    # c2 vs c3: Hamming-2 (pos 7 E→D, pos 10 Y→F) — NOT neighbours.
    # c4 is Hamming-1 from c1 (pos 4 L→P) but different V — invisible in vj mode.
    # c4 and c5 share the same CDR3 but differ in J — each isolated in its own group.
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "CASSLGQETQYF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "CASSLGQETQFF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c3", "ATG", "CASSLGQDTQYF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c4", "ATG", "CASSPGQETQYF", v_call="TRBV2", j_call="TRBJ1"),
            _clonotype("c5", "ATG", "CASSPGQETQYF", v_call="TRBV2", j_call="TRBJ2"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, match_v_call=True, match_j_call=True)
    trb = stats

    # (TRBV1, TRBJ1) group has 3 members; c1 sees c2 and c3 (both 1mm)
    assert trb["c1"]["neighbor_count"] == 3
    assert trb["c1"]["potential_neighbors"] == 3
    # c2 sees only c1 (c3 is 2mm away); c3 sees only c1
    assert trb["c2"]["neighbor_count"] == 2
    assert trb["c2"]["potential_neighbors"] == 3
    assert trb["c3"]["neighbor_count"] == 2
    assert trb["c3"]["potential_neighbors"] == 3
    # c4 is Hamming-1 from c1 but in a different V group — must be invisible
    assert trb["c4"]["neighbor_count"] == 1   # self only
    assert trb["c4"]["potential_neighbors"] == 1
    # c5 shares CDR3 with c4 but different J — isolated group
    assert trb["c5"]["neighbor_count"] == 1   # self only
    assert trb["c5"]["potential_neighbors"] == 1


def test_neighborhood_grouped_v_only_explicit_values() -> None:
    """match_v_call=True, match_j_call=False: groups by V only; J is ignored."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "CASSLGQETQYF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "CASSLGQETQFF", v_call="TRBV1", j_call="TRBJ2"),
            _clonotype("c3", "ATG", "CASSPGQETQYF", v_call="TRBV2", j_call="TRBJ1"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(rep, match_v_call=True, match_j_call=False)

    # c1 and c2 share TRBV1 (different J, but J not required); c1↔c2 are 1mm
    assert stats["c1"]["neighbor_count"] == 2
    assert stats["c1"]["potential_neighbors"] == 2
    assert stats["c2"]["neighbor_count"] == 2
    assert stats["c2"]["potential_neighbors"] == 2
    # c3 is in a separate V group
    assert stats["c3"]["neighbor_count"] == 1
    assert stats["c3"]["potential_neighbors"] == 1


def test_neighborhood_grouped_cross_background_query_absent() -> None:
    """Cross-background: query V/J absent from background → pseudocount only."""
    query = LocusRepertoire(
        clonotypes=[
            _clonotype("q1", "ATG", "CASSLGQETQYF", v_call="TRBV1", j_call="TRBJ1"),
        ],
        locus="TRB",
    )
    # Background has no TRBV1/TRBJ1 sequences
    background = LocusRepertoire(
        clonotypes=[
            _clonotype("b1", "CCC", "CASSLGQETQYF", v_call="TRBV2", j_call="TRBJ2"),
        ],
        locus="TRB",
    )

    stats = compute_neighborhood_stats(
        query, background=background, metric="hamming", threshold=1,
        match_v_call=True, match_j_call=True,
    )

    # No background group for TRBV1/TRBJ1 → only the pseudocount for self
    assert stats["q1"]["neighbor_count"] == 1
    assert stats["q1"]["potential_neighbors"] == 1


def test_neighborhood_parallel_grouped_matches_serial_grouped() -> None:
    """Parallel grouped-trie path produces identical results to the serial path."""
    rep = LocusRepertoire(
        clonotypes=[
            _clonotype("c1", "ATG", "CASSLGQETQYF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c2", "ATG", "CASSLGQETQFF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c3", "ATG", "CASSLGQDTQYF", v_call="TRBV1", j_call="TRBJ1"),
            _clonotype("c4", "ATG", "CASSPGQETQYF", v_call="TRBV2", j_call="TRBJ1"),
            _clonotype("c5", "ATG", "CASSPGQETQYF", v_call="TRBV2", j_call="TRBJ2"),
        ],
        locus="TRB",
    )

    original_min = _ne_mod._NEIGHBOR_PARALLEL_MIN_CLONOTYPES
    try:
        _ne_mod._NEIGHBOR_PARALLEL_MIN_CLONOTYPES = 2  # force parallel path for small rep

        serial = compute_neighborhood_stats_by_locus(
            rep, match_v_call=True, match_j_call=True, n_jobs=1,
        )
        parallel = compute_neighborhood_stats_by_locus(
            rep, match_v_call=True, match_j_call=True, n_jobs=4,
        )
    finally:
        _ne_mod._NEIGHBOR_PARALLEL_MIN_CLONOTYPES = original_min

    # Parallel and serial must agree exactly
    assert parallel == serial
    # Spot-check values: c1 sees c2 and c3 (both 1mm, same V+J group)
    trb_serial = serial["TRB"]
    assert trb_serial["c1"]["neighbor_count"] == 3
    assert trb_serial["c1"]["potential_neighbors"] == 3
    # c4 and c5 share the same CDR3 but are in different (V, J) groups
    assert trb_serial["c4"]["neighbor_count"] == 1
    assert trb_serial["c5"]["neighbor_count"] == 1
