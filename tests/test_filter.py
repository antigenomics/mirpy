import pytest

from mir.common.clonotype import Clonotype
from mir.common.filter import filter_canonical, filter_functional
from mir.common.gene_library import GeneEntry, GeneLibrary
from mir.common.repertoire import LocusRepertoire, SampleRepertoire


def _make_lib() -> GeneLibrary:
    entries = {
        "TRBV1*01": GeneEntry("TRBV1*01", species="human", locus="TRB", gene="V", sequence="ATG", functionality="F"),
        "TRBV2*01": GeneEntry("TRBV2*01", species="human", locus="TRB", gene="V", sequence="ATG", functionality="P"),
        "TRAV1*01": GeneEntry("TRAV1*01", species="human", locus="TRA", gene="V", sequence="ATG", functionality="F"),
    }
    return GeneLibrary(entries=entries, complete=True)


def test_filter_functional_locus_repertoire() -> None:
    lib = _make_lib()
    rep = LocusRepertoire(
        clonotypes=[
            Clonotype(sequence_id="1", locus="TRB", junction_aa="CASSF", v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1),
            Clonotype(sequence_id="2", locus="TRB", junction_aa="CAS*F", v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1, _validate=False),
            Clonotype(sequence_id="3", locus="TRB", junction_aa="CASSF", v_gene="TRBV2", j_gene="TRBJ1-1", duplicate_count=1),
        ],
        locus="TRB",
    )

    out = filter_functional(rep, gene_library=lib)
    assert isinstance(out, LocusRepertoire)
    assert [c.sequence_id for c in out.clonotypes] == ["1"]


def test_filter_canonical_locus_repertoire() -> None:
    lib = _make_lib()
    rep = LocusRepertoire(
        clonotypes=[
            Clonotype(sequence_id="1", locus="TRB", junction_aa="CASSF", v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1),
            Clonotype(sequence_id="2", locus="TRB", junction_aa="ASSSF", v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1),
        ],
        locus="TRB",
    )

    out = filter_canonical(rep, gene_library=lib)
    assert isinstance(out, LocusRepertoire)
    assert [c.sequence_id for c in out.clonotypes] == ["1"]


def test_filter_functional_sample_repertoire() -> None:
    lib = _make_lib()
    trb = LocusRepertoire(
        clonotypes=[
            Clonotype(sequence_id="1", locus="TRB", junction_aa="CASSF", v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1),
            Clonotype(sequence_id="2", locus="TRB", junction_aa="CAS*F", v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1, _validate=False),
        ],
        locus="TRB",
    )
    tra = LocusRepertoire(
        clonotypes=[
            Clonotype(sequence_id="3", locus="TRA", junction_aa="CAVRDSNYQLIW", v_gene="TRAV1", j_gene="TRAJ33", duplicate_count=1),
        ],
        locus="TRA",
    )
    sample = SampleRepertoire(loci={"TRB": trb, "TRA": tra}, sample_id="s1")

    out = filter_functional(sample, gene_library=lib)
    assert isinstance(out, SampleRepertoire)
    assert len(out.loci["TRB"].clonotypes) == 1
    assert len(out.loci["TRA"].clonotypes) == 1
    assert out.sample_id == "s1"
