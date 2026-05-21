from __future__ import annotations

import igraph as ig
import polars as pl

from mir.biomarkers.alice import metaclonotypes_from_alice
from mir.biomarkers.tcrnet import metaclonotypes_from_tcrnet
from mir.common.clonotype import Clonotype
from mir.common.metaclonotype import (
    MetaClonotypeDefinition,
    functional_diversity,
    functional_overlap_1,
    metaclonotype_count_vector,
    metaclonotype_junctions,
    metaclonotypes_from_components,
    metaclonotypes_from_igraph,
    metaclonotypes_from_labels,
    metaclonotypes_from_seed_neighbors,
    summarize_metaclonotypes,
    summarize_paired_metaclonotypes,
)
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.single_cell import (
    PairedClonotype,
    PairedLocusRepertoire,
    PairedRepertoire,
    SingleCellRepertoire,
)


def _clone(
    seq_id: str,
    junction_aa: str,
    *,
    v: str = "TRBV5-1*01",
    j: str = "TRBJ2-7*01",
    dup: int = 1,
    umi: int = 0,
) -> Clonotype:
    return Clonotype(
        sequence_id=seq_id,
        locus="TRB",
        junction_aa=junction_aa,
        v_gene=v,
        j_gene=j,
        duplicate_count=dup,
        umi_count=umi,
        _validate=False,
    )


def _toy_rep() -> LocusRepertoire:
    return LocusRepertoire(
        [
            _clone("c1", "CASSLGQETQYF", dup=10, umi=4),
            _clone("c2", "CASSLGQETQFF", dup=5, umi=3),
            _clone("c3", "CASSLGQATQYF", dup=2, umi=1),
            _clone("c4", "CATSLGQETQYF", dup=1, umi=1),
        ],
        locus="TRB",
    )


def test_metaclonotype_definition_normalizes_and_deduplicates() -> None:
    table = pl.DataFrame(
        {
            "cluster_id": ["a", "a", "a"],
            "clonotype_id": ["c1", "c1", "c2"],
        }
    )
    m = MetaClonotypeDefinition(table, paired=False)
    assert m.n_clusters == 1
    assert len(m.table) == 2
    assert "is_representative" in m.table.columns


def test_builders_from_labels_components_and_igraph() -> None:
    m_labels = metaclonotypes_from_labels(["c1", "c2", "c3"], [0, 0, 1])
    assert m_labels.n_clusters == 2

    m_comp = metaclonotypes_from_components([["c1", "c2"], ["c3"]])
    assert m_comp.n_clusters == 2

    g = ig.Graph(n=3, edges=[(0, 1)], directed=False)
    g.vs["r_id"] = ["c1", "c2", "c3"]
    m_graph = metaclonotypes_from_igraph(g)
    assert m_graph.n_clusters == 2


def test_seed_neighbor_builder_and_summary() -> None:
    rep = _toy_rep()
    m = metaclonotypes_from_seed_neighbors(
        rep,
        seed_clonotype_ids=["c1"],
        metric="hamming",
        threshold=1,
        match_v_gene=True,
        match_j_gene=True,
    )
    # All clonotypes in this toy set are within Hamming-1 from c1.
    assert set(m.table["clonotype_id"].to_list()) == {"c1", "c2", "c3", "c4"}

    summary = summarize_metaclonotypes(rep, m)
    row = summary.row(0, named=True)
    assert int(row["n_members"]) == 4
    assert int(row["duplicate_count"]) == 18
    assert int(row["umi_count"]) == 9


def test_functional_diversity_and_counts() -> None:
    rep = _toy_rep()
    m = metaclonotypes_from_components([["c1", "c2"], ["c3"], ["c4"]])
    counts = metaclonotype_count_vector(rep, m, count_field="duplicate_count")
    assert sorted(counts, reverse=True) == [15, 2, 1]

    div = functional_diversity(rep, m, count_field="duplicate_count")
    assert div.abundance == 18
    assert div.diversity == 3


def test_functional_overlap_1() -> None:
    rep_a = LocusRepertoire([_clone("a1", "CASSLGQETQYF"), _clone("a2", "CASSLGQATQYF")], locus="TRB")
    rep_b = LocusRepertoire([_clone("b1", "CASSLGQETQYF"), _clone("b2", "CASSXXXXTQYF")], locus="TRB")

    m_a = metaclonotypes_from_components([["a1", "a2"]])
    m_b = metaclonotypes_from_components([["b1"], ["b2"]])

    overlap = functional_overlap_1(rep_a, m_a, rep_b, m_b)
    row = overlap.row(0, named=True)
    assert int(row["a_shared_clusters"]) == 1
    assert int(row["b_shared_clusters"]) == 1


def test_motif_logo_sequence_extraction() -> None:
    rep = _toy_rep()
    m = metaclonotypes_from_components([["c1", "c2"], ["c3"]])
    seqs = metaclonotype_junctions(rep, m, cluster_id="mc_0")
    assert set(seqs) == {"CASSLGQETQYF", "CASSLGQETQFF"}


def test_repertoire_attachment_apis() -> None:
    rep = _toy_rep()
    m = metaclonotypes_from_components([["c1", "c2"]])
    rep.set_metaclonotypes(m)
    assert rep.metaclonotypes is m

    sample = SampleRepertoire(loci={"TRB": rep}, sample_id="s1")
    sample.set_metaclonotypes("TRB", m)
    assert sample.get_metaclonotypes("TRB") is m


def _make_paired_repertoire() -> PairedRepertoire:
    c1a = Clonotype(sequence_id="tra1", locus="TRA", junction_aa="CAVAAA", duplicate_count=3, umi_count=2, _validate=False)
    c1b = Clonotype(sequence_id="trb1", locus="TRB", junction_aa="CASSAAA", duplicate_count=4, umi_count=1, _validate=False)
    c2a = Clonotype(sequence_id="tra2", locus="TRA", junction_aa="CAVDDD", duplicate_count=5, umi_count=2, _validate=False)
    c2b = Clonotype(sequence_id="trb2", locus="TRB", junction_aa="CASSDDD", duplicate_count=6, umi_count=3, _validate=False)

    p1 = PairedClonotype(pair_id="p1", clonotype1=c1a, clonotype2=c1b)
    p2 = PairedClonotype(pair_id="p2", clonotype1=c2a, clonotype2=c2b)

    by_pair = {
        "TRA_TRB": PairedLocusRepertoire(locus_pair="TRA_TRB", paired_clonotypes=[p1, p2]),
        "TRG_TRD": PairedLocusRepertoire(locus_pair="TRG_TRD", paired_clonotypes=[]),
        "IGH_IGK": PairedLocusRepertoire(locus_pair="IGH_IGK", paired_clonotypes=[]),
        "IGH_IGL": PairedLocusRepertoire(locus_pair="IGH_IGL", paired_clonotypes=[]),
    }

    links = SingleCellRepertoire(barcode_pair_ids=[("bc1", "p1"), ("bc2", "p2")], barcode_metadata={})
    return PairedRepertoire(
        sample_id="d1",
        single_cell_repertoire=links,
        paired_locus_repertoires=by_pair,
        chain_multiplicity=pl.DataFrame({"sample_id": ["d1"], "locus_pair": ["TRA_TRB"], "n_chain1": [1], "m_chain2": [1], "cell_count": [2]}),
        loaded_cell_count=2,
        loaded_clonotype_count=4,
    )


def test_summarize_paired_metaclonotypes() -> None:
    paired = _make_paired_repertoire()
    meta = MetaClonotypeDefinition(
        pl.DataFrame(
            {
                "cluster_id": ["p_mc_0", "p_mc_1"],
                "clonotype_id_1": ["tra1", "tra2"],
                "clonotype_id_2": ["trb1", "trb2"],
                "is_representative": [True, True],
            }
        ),
        paired=True,
    )

    paired.set_metaclonotypes(meta)
    summary = summarize_paired_metaclonotypes(paired, meta, count_field="duplicate_count")
    assert summary.height == 2
    assert sorted(summary["duplicate_count"].to_list()) == [7, 11]


def test_alice_and_tcrnet_meta_builders_from_metadata() -> None:
    rep = _toy_rep()
    rep.clonotypes[0].clone_metadata["alice_q_value"] = 0.01
    rep.clonotypes[1].clone_metadata["alice_q_value"] = 0.2
    rep.clonotypes[0].clone_metadata["tcrnet_q_value"] = 0.01

    alice_meta = metaclonotypes_from_alice(rep, q_value_max=0.05)
    tcrnet_meta = metaclonotypes_from_tcrnet(rep, q_value_max=0.05)

    assert alice_meta.n_clusters == 1
    assert tcrnet_meta.n_clusters == 1
