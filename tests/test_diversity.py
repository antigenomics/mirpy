"""Unit tests for repertoire diversity metrics and curve APIs."""

from __future__ import annotations

import numpy as np
import polars as pl

from mir.common.clonotype import Clonotype
from mir.common.diversity import build_abundance_table, hill_curve, rarefaction_curve, summarize_counts
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.single_cell import (
    PairedClonotype,
    PairedLocusRepertoire,
    PairedRepertoire,
    SingleCellRepertoire,
    SingleCellSample,
)


def _clone(
    seq_id: str,
    locus: str,
    duplicate_count: int,
    umi_count: int,
    junction_aa: str,
) -> Clonotype:
    return Clonotype(
        sequence_id=seq_id,
        locus=locus,
        duplicate_count=duplicate_count,
        umi_count=umi_count,
        junction_aa=junction_aa,
        v_gene=f"{locus}V1",
        j_gene=f"{locus}J1",
    )


def test_diversity_summary_core_metrics() -> None:
    summary = summarize_counts([10, 5, 2, 1, 1, 1])
    assert summary.abundance == 20
    assert summary.diversity == 6
    assert summary.singletons == 3
    assert summary.doubletons == 1
    assert summary.expanded == 6
    assert summary.hyperexpanded == 6
    assert summary.chao1 > summary.diversity
    assert 0.0 < summary.gini_simpson < 1.0
    assert summary.shannon > 0.0


def test_locus_repertoire_count_modes() -> None:
    rep = LocusRepertoire(
        [
            _clone("c1", "TRB", 10, 2, "CASSAAA"),
            _clone("c2", "TRB", 5, 4, "CASSDDD"),
            _clone("c3", "TRB", 1, 1, "CASSEEE"),
        ],
        locus="TRB",
    )

    dup = rep.diversity(count_field="duplicate_count")
    umi = rep.diversity(count_field="umi_count")

    assert dup.abundance == 16
    assert umi.abundance == 7
    assert dup.diversity == umi.diversity == 3


def test_sample_repertoire_per_locus_metrics() -> None:
    tra = LocusRepertoire([_clone("a1", "TRA", 3, 2, "CAVR")], locus="TRA")
    trb = LocusRepertoire([_clone("b1", "TRB", 7, 3, "CASS"), _clone("b2", "TRB", 2, 1, "CASR")], locus="TRB")
    sample = SampleRepertoire(loci={"TRA": tra, "TRB": trb}, sample_id="s1")

    per_locus = sample.diversity(per_locus=True)
    pooled = sample.diversity(per_locus=False)

    assert set(per_locus) == {"TRA", "TRB"}
    assert per_locus["TRA"].abundance == 3
    assert per_locus["TRB"].diversity == 2
    assert pooled.abundance == 12


def test_hill_curve_reference_values() -> None:
    counts = [8, 4, 2, 1]
    profile = hill_curve(counts, q_values=[0.0, 1.0, 2.0])

    q = profile["q"].to_list()
    h = profile["hill"].to_list()
    assert q == [0.0, 1.0, 2.0]
    assert h[0] == 4.0

    p = np.array(counts, dtype=float) / sum(counts)
    expected_q1 = float(np.exp(-np.sum(p * np.log(p))))
    expected_q2 = float(1.0 / np.sum(p * p))
    assert np.isclose(h[1], expected_q1)
    assert np.isclose(h[2], expected_q2)


def test_rarefaction_curve_includes_exact_point() -> None:
    counts = [5, 3, 2, 1]
    n = sum(counts)
    curve = rarefaction_curve(counts, m_steps=[2, 4, 6], include_exact=True)

    assert {"m", "s_est", "coverage", "s_lwr", "s_upr"}.issubset(set(curve.columns))
    assert n in curve["m"].to_list()

    at = build_abundance_table(counts)
    assert at.columns == ["k", "f_k"]
    assert at.height >= 1


def _make_paired_repertoire() -> PairedRepertoire:
    tra1 = _clone("tra1", "TRA", 10, 5, "CAVAAA")
    trb1 = _clone("trb1", "TRB", 8, 4, "CASSAAA")
    tra2 = _clone("tra2", "TRA", 4, 2, "CAVDDD")
    trb2 = _clone("trb2", "TRB", 3, 1, "CASSEEE")

    p1 = PairedClonotype(pair_id="p1", clonotype1=tra1, clonotype2=trb1)
    p2 = PairedClonotype(pair_id="p2", clonotype1=tra2, clonotype2=trb2)

    by_pair = {
        "TRA_TRB": PairedLocusRepertoire(locus_pair="TRA_TRB", paired_clonotypes=[p1, p2]),
        "TRG_TRD": PairedLocusRepertoire(locus_pair="TRG_TRD", paired_clonotypes=[]),
        "IGH_IGK": PairedLocusRepertoire(locus_pair="IGH_IGK", paired_clonotypes=[]),
        "IGH_IGL": PairedLocusRepertoire(locus_pair="IGH_IGL", paired_clonotypes=[]),
    }

    links = SingleCellRepertoire(
        barcode_pair_ids=[("bc1", "p1"), ("bc2", "p1"), ("bc3", "p2")],
        barcode_metadata={},
    )
    return PairedRepertoire(
        sample_id="donor1",
        single_cell_repertoire=links,
        paired_locus_repertoires=by_pair,
        chain_multiplicity=pl.DataFrame(
            {
                "sample_id": ["donor1"],
                "locus_pair": ["TRA_TRB"],
                "n_chain1": [1],
                "m_chain2": [1],
                "cell_count": [3],
            }
        ),
        loaded_cell_count=3,
        loaded_clonotype_count=4,
    )


def test_paired_repertoire_barcode_default_and_locus_breakdown() -> None:
    paired = _make_paired_repertoire()

    pair_level = paired.diversity()
    chain_level = paired.diversity_by_locus()

    assert pair_level["TRA_TRB"].abundance == 3
    assert pair_level["TRA_TRB"].diversity == 2
    assert chain_level["TRA"].abundance == 3
    assert chain_level["TRB"].abundance == 3


def test_single_cell_sample_delegates_diversity_and_curves() -> None:
    paired = _make_paired_repertoire()
    sc = SingleCellSample(
        sample_id="donor1",
        paired_repertoire=paired,
        cite_seq_matrix=pl.DataFrame({"barcode": ["bc1", "bc2", "bc3"]}),
        cite_seq_binder_columns=pl.DataFrame({"column": []}),
    )

    div = sc.diversity(per_locus=True)
    hill = sc.hill_curve(per_locus=True)
    rare = sc.rarefaction_curve(per_locus=True, m_steps=[1, 2, 3], include_exact=True)

    assert set(div) >= {"TRA", "TRB"}
    assert set(hill["TRA"].columns) == {"q", "hill"}
    assert "coverage" in rare["TRA"].columns
