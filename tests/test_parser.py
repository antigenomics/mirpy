"""Unit tests for mir.common.parser auxiliary parsers.

Tests use module-level functions (not TestCase classes) with pytest.
No files are written; all assertions operate on in-memory objects only.
"""

from __future__ import annotations

import gzip
import io
import tempfile
from pathlib import Path

import pytest

from mir.basic.alphabets import back_translate, _MOST_LIKELY_CODON
from mir.common.clonotype import Clonotype
from mir.common.parser import OldMiXCRParser, VDJdbSlimParser, OlgaParser
from mir.common.repertoire import SampleRepertoire, LocusRepertoire

ASSETS = Path(__file__).parent / "assets"

_OLD_MIXCR_FILE = ASSETS / "old_mixcr.gz"
_VDJDB_FILE     = ASSETS / "vdjdb.slim.txt.gz"
_OLGA_FILE      = ASSETS / "olga_humanTRB_1000.txt.gz"


# ---------------------------------------------------------------------------
# back_translate
# ---------------------------------------------------------------------------

def test_back_translate_length():
    seq = "CASSEGF"
    assert len(back_translate(seq)) == len(seq) * 3


def test_back_translate_known_codon():
    # C → TGC, A → GCC
    assert back_translate("CA") == "TGC" + "GCC"


def test_back_translate_all_standard_aa():
    aa = "".join(_MOST_LIKELY_CODON)
    nt = back_translate(aa)
    assert len(nt) == len(aa) * 3
    # All returned triplets must be known codons
    for i in range(0, len(nt), 3):
        assert nt[i:i+3] in _MOST_LIKELY_CODON.values()


def test_back_translate_unknown_residue_default():
    assert back_translate("X") == "NNN"


def test_back_translate_unknown_residue_custom():
    assert back_translate("X", unknown_codon="AAA") == "AAA"


def test_back_translate_empty():
    assert back_translate("") == ""


# ---------------------------------------------------------------------------
# OldMiXCRParser
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def old_mixcr_sample() -> SampleRepertoire:
    return OldMiXCRParser().parse_file(_OLD_MIXCR_FILE)


def test_old_mixcr_returns_sample_repertoire(old_mixcr_sample):
    assert isinstance(old_mixcr_sample, SampleRepertoire)


def test_old_mixcr_sample_id(old_mixcr_sample):
    assert old_mixcr_sample.sample_id == "old_mixcr"


def test_old_mixcr_has_trb_locus(old_mixcr_sample):
    assert "TRB" in old_mixcr_sample


def test_old_mixcr_clonotype_count(old_mixcr_sample):
    assert old_mixcr_sample["TRB"].clonotype_count == 999


def test_old_mixcr_clonotypes_have_junction_aa(old_mixcr_sample):
    for c in old_mixcr_sample["TRB"].clonotypes[:10]:
        assert c.junction_aa


def test_old_mixcr_v_gene_normalized(old_mixcr_sample):
    # allele must be *01, not *00
    for c in old_mixcr_sample["TRB"].clonotypes[:20]:
        if c.v_gene:
            assert "*00" not in c.v_gene
            assert c.v_gene.endswith("*01") or "*" not in c.v_gene


def test_old_mixcr_locus_on_clonotypes(old_mixcr_sample):
    for c in old_mixcr_sample["TRB"].clonotypes[:20]:
        assert c.locus == "TRB"


def test_old_mixcr_ref_points_v_end(old_mixcr_sample):
    # first clone: CASSNSDRTYGDNEQFF, v_end = 11
    c = old_mixcr_sample["TRB"].clonotypes[0]
    assert c.v_sequence_end == 11


def test_old_mixcr_ref_points_j_start(old_mixcr_sample):
    # first clone: j_start = 34
    c = old_mixcr_sample["TRB"].clonotypes[0]
    assert c.j_sequence_start == 34


def test_old_mixcr_d_absent_gives_minus_one(old_mixcr_sample):
    # first clone has no D gene — d_sequence_start should be -1
    c = old_mixcr_sample["TRB"].clonotypes[0]
    assert c.d_sequence_start == -1
    assert c.d_sequence_end == -1


def test_old_mixcr_custom_sample_id():
    sample = OldMiXCRParser().parse_file(_OLD_MIXCR_FILE, sample_id="custom_id")
    assert sample.sample_id == "custom_id"


# ---------------------------------------------------------------------------
# VDJdbSlimParser
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vdjdb_sample() -> SampleRepertoire:
    return VDJdbSlimParser().parse_file(_VDJDB_FILE)


def test_vdjdb_returns_sample_repertoire(vdjdb_sample):
    assert isinstance(vdjdb_sample, SampleRepertoire)


def test_vdjdb_has_trb_locus(vdjdb_sample):
    assert "TRB" in vdjdb_sample


def test_vdjdb_has_tra_locus(vdjdb_sample):
    assert "TRA" in vdjdb_sample


def test_vdjdb_trb_nonempty(vdjdb_sample):
    assert vdjdb_sample["TRB"].clonotype_count > 0


def test_vdjdb_junction_aa_from_cdr3(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    # First TRB row in the file is CASSTSRLSNQPQYF
    assert c.junction_aa == "CASSTSRLSNQPQYF"


def test_vdjdb_junction_back_translated(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    expected_nt = back_translate(c.junction_aa)
    assert c.junction == expected_nt
    assert len(c.junction) == len(c.junction_aa) * 3


def test_vdjdb_v_gene_set(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    assert c.v_gene  # non-empty


def test_vdjdb_v_sequence_end(vdjdb_sample):
    # First TRB row: v.end = 4 → v_sequence_end = 12
    c = vdjdb_sample["TRB"].clonotypes[0]
    assert c.v_sequence_end == 4 * 3


def test_vdjdb_j_sequence_start(vdjdb_sample):
    # First TRB row: j.start = 8 → j_sequence_start = 24
    c = vdjdb_sample["TRB"].clonotypes[0]
    assert c.j_sequence_start == 8 * 3


def test_vdjdb_metadata_keys(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    for key in ("mhc.a", "mhc.b", "mhc.class", "antigen.species",
                "antigen.gene", "antigen.epitope"):
        assert key in c.clone_metadata


def test_vdjdb_metadata_epitope(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    assert c.clone_metadata["antigen.epitope"] == "STPESANL"


def test_vdjdb_species_filter():
    sample = VDJdbSlimParser().parse_file(_VDJDB_FILE, species="HomoSapiens")
    # All loci should only contain HomoSapiens entries (no MacacaMulatta etc.)
    total = sum(lr.clonotype_count for lr in sample)
    assert total > 0
    # Unfiltered has more entries
    total_all = sum(
        lr.clonotype_count
        for lr in VDJdbSlimParser().parse_file(_VDJDB_FILE)
    )
    assert total < total_all


def test_vdjdb_locus_from_gene_column(vdjdb_sample):
    for c in vdjdb_sample["TRB"].clonotypes[:5]:
        assert c.locus == "TRB"
    for c in vdjdb_sample["TRA"].clonotypes[:5]:
        assert c.locus == "TRA"


# ---------------------------------------------------------------------------
# OlgaParser
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def olga_sample() -> SampleRepertoire:
    return OlgaParser().parse_file(_OLGA_FILE)


def test_olga_returns_sample_repertoire(olga_sample):
    assert isinstance(olga_sample, SampleRepertoire)


def test_olga_has_trb_locus(olga_sample):
    assert "TRB" in olga_sample


def test_olga_clonotype_count(olga_sample):
    assert olga_sample["TRB"].clonotype_count == 1000


def test_olga_junction_nonempty(olga_sample):
    for c in olga_sample["TRB"].clonotypes[:10]:
        assert c.junction
        assert c.junction_aa


def test_olga_v_gene_set(olga_sample):
    for c in olga_sample["TRB"].clonotypes[:10]:
        assert c.v_gene.startswith("TRB")


def test_olga_j_gene_set(olga_sample):
    for c in olga_sample["TRB"].clonotypes[:10]:
        assert c.j_gene.startswith("TRB")


def test_olga_locus_inferred_from_j_gene(olga_sample):
    for c in olga_sample["TRB"].clonotypes[:10]:
        assert c.locus == "TRB"


def test_olga_custom_sample_id():
    sample = OlgaParser().parse_file(_OLGA_FILE, sample_id="my_olga")
    assert sample.sample_id == "my_olga"


# ---------------------------------------------------------------------------
# csv field size limit
# ---------------------------------------------------------------------------

def _make_vdjdb_slim_gz(oversized_reference_id: str) -> bytes:
    """Return gzipped VDJdb slim TSV bytes with one TRB row containing an
    intentionally large reference.id field (to trigger the 131072-byte limit
    unless the module-level csv.field_size_limit fix is in place)."""
    header = (
        "gene\tcdr3\tspecies\tantigen.epitope\tantigen.gene\tantigen.species"
        "\tcomplex.id\tv.segm\tj.segm\tmhc.a\tmhc.b\tmhc.class"
        "\treference.id\tvdjdb.score\tvdjdb.pgen.score\tTCR_hash\tj.start\tv.end"
    )
    row = "\t".join([
        "TRB",
        "CASSEGFTGELFF",
        "HomoSapiens",
        "GILGFVFTL",
        "M1",
        "InfluenzaA",
        "0",
        "TRBV12-3*01",
        "TRBJ2-2*01",
        "HLA-A*02:01",
        "B2M",
        "MHCI",
        oversized_reference_id,
        "3",
        "0",
        "abc123",
        "10",
        "5",
    ])
    content = header + "\n" + row + "\n"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(content.encode())
    return buf.getvalue()


def test_vdjdb_large_field_no_csv_error():
    """VDJdbSlimParser must handle fields larger than csv's default 131072-byte
    limit without raising csv.Error."""
    big_ref = "PUBMED:" + "X" * 200_000
    gz_bytes = _make_vdjdb_slim_gz(big_ref)

    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as tmp:
        tmp.write(gz_bytes)
        tmp_path = Path(tmp.name)

    try:
        sample = VDJdbSlimParser().parse_file(tmp_path)
        assert isinstance(sample, SampleRepertoire)
        assert sample["TRB"].clonotype_count == 1
    finally:
        tmp_path.unlink(missing_ok=True)


def test_olga_locus_override():
    sample = OlgaParser().parse_file(_OLGA_FILE, locus="TRB")
    assert "TRB" in sample
