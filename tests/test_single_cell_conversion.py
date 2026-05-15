"""Tests for PairedRepertoire <-> SampleRepertoire conversions."""

from __future__ import annotations

import polars as pl

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.single_cell import PairedRepertoire, PairedClonotype, PairedLocusRepertoire, SingleCellRepertoire


def _clono(sequence_id: str, locus: str, aa: str) -> Clonotype:
    return Clonotype(sequence_id=sequence_id, locus=locus, junction_aa=aa)


def test_paired_to_sample_repertoire_collapse() -> None:
    tra1 = _clono("tra1", "TRA", "CAAAAF")
    trb1 = _clono("trb1", "TRB", "CASSFF")
    trb2 = _clono("trb2", "TRB", "CASSTF")

    paired = PairedRepertoire(
        sample_id="sample_1",
        single_cell_repertoire=SingleCellRepertoire(barcode_pair_ids=[("bc1", "p1"), ("bc2", "p2")]),
        paired_locus_repertoires={
            "TRA_TRB": PairedLocusRepertoire(
                locus_pair="TRA_TRB",
                paired_clonotypes=[
                    PairedClonotype(pair_id="p1", clonotype1=tra1, clonotype2=trb1),
                    PairedClonotype(pair_id="p2", clonotype1=tra1, clonotype2=trb2),
                ],
            ),
            "TRG_TRD": PairedLocusRepertoire(locus_pair="TRG_TRD", paired_clonotypes=[]),
            "IGH_IGK": PairedLocusRepertoire(locus_pair="IGH_IGK", paired_clonotypes=[]),
            "IGH_IGL": PairedLocusRepertoire(locus_pair="IGH_IGL", paired_clonotypes=[]),
        },
        chain_multiplicity=pl.DataFrame(
            {
                "sample_id": ["sample_1"],
                "locus_pair": ["TRA_TRB"],
                "n_chain1": [1],
                "m_chain2": [2],
                "cell_count": [1],
            }
        ),
        loaded_cell_count=2,
        loaded_clonotype_count=3,
    )

    sample = paired.to_sample_repertoire()
    assert sample.sample_id == "sample_1"
    assert set(sample.loci) == {"TRA", "TRB"}
    assert sample.loci["TRA"].clonotype_count == 1
    assert sample.loci["TRB"].clonotype_count == 2


def test_sample_to_paired_repertoire_and_lazy_lookup() -> None:
    sample = SampleRepertoire(
        loci={
            "TRA": LocusRepertoire([_clono("tra1", "TRA", "CAAAAF")], locus="TRA"),
            "TRB": LocusRepertoire([_clono("trb1", "TRB", "CASSFF")], locus="TRB"),
        },
        sample_id="sample_2",
    )

    paired = PairedRepertoire.from_sample_repertoire(
        sample,
        pairing_rows=[("pair_1", "TRA", "TRB", "tra1", "trb1")],
        barcode_pair_ids=[("bc1", "pair_1")],
    )

    assert paired.sample_id == "sample_2"
    assert paired.loaded_cell_count == 1
    assert paired.paired_locus_repertoires["TRA_TRB"].clonotype_count == 1

    resolved = paired.get_clonotype("TRA", "tra1")
    assert resolved is not None
    assert resolved.sequence_id == "tra1"

    rebuilt = paired.to_sample_repertoire()
    assert set(rebuilt.loci) == {"TRA", "TRB"}
    assert rebuilt.loci["TRA"].clonotype_count == 1
    assert rebuilt.loci["TRB"].clonotype_count == 1
