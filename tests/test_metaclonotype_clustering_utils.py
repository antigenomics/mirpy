from __future__ import annotations

import igraph as ig
import polars as pl

from mir.common.clonotype import Clonotype
from mir.common.single_cell import PairedClonotype
from mir.embedding.tcremp import (
    metaclonotypes_from_tcremp_labels,
    paired_metaclonotypes_from_tcremp_labels,
)
from mir.graph.token_graph import (
    build_gliph_metaclonotypes,
    metaclonotypes_from_token_clonotype_graph,
)
from mir.utils.metaclonotype_clustering import (
    metaclonotypes_from_cluster_labels,
    metaclonotypes_from_graph_communities,
    metaclonotypes_from_search_scope,
)


def _clone(seq_id: str, locus: str, aa: str) -> Clonotype:
    return Clonotype(
        sequence_id=seq_id,
        locus=locus,
        junction_aa=aa,
        duplicate_count=1,
        _validate=False,
    )


def test_labels_to_metaclonotypes_noise_filtering() -> None:
    meta = metaclonotypes_from_cluster_labels(
        ["c1", "c2", "c3", "c4"],
        [0, 0, 1, -1],
        include_noise=False,
    )
    assert meta.n_clusters == 2
    assert set(meta.table["clonotype_id"].to_list()) == {"c1", "c2", "c3"}


def test_graph_community_conversion_components_and_leiden() -> None:
    g = ig.Graph(n=4, edges=[(0, 1), (2, 3)], directed=False)
    g.es["weight"] = [1.0, 1.0]
    g.vs["r_id"] = ["c1", "c2", "c3", "c4"]

    comps = metaclonotypes_from_graph_communities(g, method="components", vertex_id_attr="r_id")
    assert comps.n_clusters == 2

    leiden = metaclonotypes_from_graph_communities(g, method="leiden", vertex_id_attr="r_id")
    assert set(leiden.table["clonotype_id"].to_list()) == {"c1", "c2", "c3", "c4"}


def test_tcremp_label_wrappers_single_and_paired() -> None:
    clonotypes = [
        _clone("c1", "TRB", "CASSLGQETQYF"),
        _clone("c2", "TRB", "CASSLGQETQFF"),
        _clone("c3", "TRB", "CASSLGQATQYF"),
    ]
    meta = metaclonotypes_from_tcremp_labels(clonotypes, [0, 0, -1], include_noise=False)
    assert meta.n_clusters == 1

    p1 = PairedClonotype("p1", _clone("tra1", "TRA", "CAVAAA"), _clone("trb1", "TRB", "CASSAAA"))
    p2 = PairedClonotype("p2", _clone("tra2", "TRA", "CAVDDD"), _clone("trb2", "TRB", "CASSDDD"))
    pmeta = paired_metaclonotypes_from_tcremp_labels(
        [p1, p2],
        [0, -1],
        include_noise=True,
        mock_chain_1_by_pair={"p2": True},
    )
    assert pmeta.paired
    assert set(pmeta.table["cluster_id"].to_list()) == {"0", "-1"}
    assert int(pmeta.table.filter(pl.col("clonotype_id_1") == "tra2")["mock_chain_1"][0]) == 1


def test_metaclonotypes_from_search_scope() -> None:
    neigh = {
        "c1": ["c1", "c2", "c3"],
        "c3": ["c3", "c4"],
    }
    meta = metaclonotypes_from_search_scope(
        ["c1", "c3"],
        neighbor_selector=lambda x: neigh.get(x, []),
        cluster_prefix="scope",
    )
    assert meta.n_clusters == 2
    c1 = set(meta.table.filter(pl.col("cluster_id") == "scope_0")["clonotype_id"].to_list())
    assert c1 == {"c1", "c2", "c3"}


def test_token_graph_metaclonotype_wrappers() -> None:
    g = ig.Graph(n=3, edges=[(0, 1)], directed=False)
    g.es["weight"] = [1.0]
    g.vs["name"] = ["c1", "c2", "c3"]

    meta = metaclonotypes_from_token_clonotype_graph(
        g,
        method="components",
        min_cluster_size=2,
        vertex_id_attr="name",
    )
    assert set(meta.table["clonotype_id"].to_list()) == {"c1", "c2"}


def test_build_gliph_metaclonotypes_end_to_end_small() -> None:
    study_df = pl.DataFrame(
        {
            "row_id": ["c1", "c2", "c3"],
            "sequence_id": ["c1", "c2", "c3"],
            "junction_aa": ["CASSLGQETQYF", "CASSLGQETQFF", "TTTTLGQETQFF"],
            "v_call": ["TRBV5-1*01", "TRBV5-1*01", "TRBV5-1*01"],
            "duplicate_count": [10, 5, 1],
        }
    )
    token_to_clones = {"tok1": {"c1", "c2"}}
    meta = build_gliph_metaclonotypes(
        study_df,
        token_to_clones,
        method="components",
        min_cluster_size=1,
        hamming_threads=1,
    )
    assert meta.n_clusters >= 1
