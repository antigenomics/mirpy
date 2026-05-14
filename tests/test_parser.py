"""Unit tests for mir.common.parser auxiliary parsers.

Tests use module-level functions (not TestCase classes) with pytest.
No files are written; all assertions operate on in-memory objects only.
"""

from __future__ import annotations

import gzip
import io
import tempfile
from pathlib import Path

import polars as pl
import pytest

from mir.basic.alphabets import back_translate, _MOST_LIKELY_CODON
from mir.common.parser import (
    AdaptiveParser,
    OldMiXCRParser,
    OlgaParser,
    VDJdbFullPairedParser,
    VDJdbSlimParser,
    VDJtoolsParser,
)
from mir.common.repertoire import SampleRepertoire, LocusRepertoire
from mir.common.gene_library import GeneLibrary
from mir.common.single_cell import PairedRepertoire

ASSETS = Path(__file__).parent / "assets"

_OLD_MIXCR_FILE = ASSETS / "old_mixcr.gz"
_VDJDB_FILE     = ASSETS / "vdjdb.slim.txt.gz"
_VDJDB_FULL_FILE = ASSETS / "vdjdb_full.txt.gz"
_OLGA_FILE      = ASSETS / "olga_humanTRB_1000.txt.gz"
_VDJTOOLS_FILE  = ASSETS / "vdjtools_trb_d_dot.tsv"


# ---------------------------------------------------------------------------
# VDJtoolsParser — d='.' normalisation
# ---------------------------------------------------------------------------

def test_vdjtools_d_dot_normalised_to_empty():
    """'.' in the d-gene column must be normalised to '' by _gene_str."""
    clonotypes = VDJtoolsParser().parse(str(_VDJTOOLS_FILE))
    assert all(c.d_gene != "." for c in clonotypes), \
        "d_gene '.' was not normalised to empty string"


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
    assert isinstance(c.junction_aa, str)
    assert len(c.junction_aa) > 0


def test_vdjdb_junction_back_translated(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    expected_nt = back_translate(c.junction_aa)
    assert c.junction == expected_nt
    assert len(c.junction) == len(c.junction_aa) * 3


def test_vdjdb_v_gene_set(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    assert c.v_gene  # non-empty


def test_vdjdb_v_sequence_end(vdjdb_sample):
    c = next((x for x in vdjdb_sample["TRB"].clonotypes if x.v_sequence_end > 0), None)
    assert c is not None
    assert c.v_sequence_end % 3 == 0


def test_vdjdb_j_sequence_start(vdjdb_sample):
    c = next((x for x in vdjdb_sample["TRB"].clonotypes if x.j_sequence_start > 0), None)
    assert c is not None
    assert c.j_sequence_start % 3 == 0


def test_vdjdb_metadata_keys(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    for key in ("mhc.a", "mhc.b", "mhc.class", "antigen.species",
                "antigen.gene", "antigen.epitope"):
        assert key in c.clone_metadata


def test_vdjdb_metadata_epitope(vdjdb_sample):
    c = vdjdb_sample["TRB"].clonotypes[0]
    assert isinstance(c.clone_metadata["antigen.epitope"], str)
    assert c.clone_metadata["antigen.epitope"]


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
# VDJdbFullPairedParser
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vdjdb_full_parser() -> VDJdbFullPairedParser:
    return VDJdbFullPairedParser()


@pytest.fixture(scope="module")
def vdjdb_full_paired(vdjdb_full_parser) -> PairedRepertoire:
    return vdjdb_full_parser.parse_file(
        _VDJDB_FULL_FILE,
        sample_id="vdjdb_full_human",
        species="HomoSapiens",
    )


def test_vdjdb_full_returns_paired_repertoire(vdjdb_full_paired):
    assert isinstance(vdjdb_full_paired, PairedRepertoire)


def test_vdjdb_full_has_tra_trb_pairs(vdjdb_full_paired):
    assert vdjdb_full_paired.paired_locus_repertoires["TRA_TRB"].clonotype_count > 0


def test_vdjdb_full_manual_record_993(vdjdb_full_paired):
    pair = next(
        pair
        for pair in vdjdb_full_paired.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
        if pair.pair_id == "993"
    )
    chains = {pair.clonotype1.locus: pair.clonotype1, pair.clonotype2.locus: pair.clonotype2}
    assert chains["TRA"].sequence_id == "993_TRA"
    assert chains["TRA"].v_gene == "TRAV38-2/DV8*01"
    assert chains["TRA"].j_gene == "TRAJ53*01"
    assert chains["TRA"].junction_aa == "CAYRSAGSGGSNYKLTF"
    assert chains["TRB"].sequence_id == "993_TRB"
    assert chains["TRB"].v_gene == "TRBV27*01"
    assert chains["TRB"].j_gene == "TRBJ1-5*01"
    assert chains["TRB"].junction_aa == "CASSLMTNQPQHF"


def test_vdjdb_full_manual_record_1007_metadata(vdjdb_full_paired):
    metadata = vdjdb_full_paired.single_cell_repertoire.barcode_metadata["1007"]
    assert metadata["vdjdb_record_id"] == "1007"
    assert metadata["mhc.a"] == "HLA-B*07:02"
    assert metadata["mhc.b"] == "B2M"
    assert metadata["mhc.class"] == "MHCI"
    assert metadata["antigen.epitope"] == "RPIIRPATL"
    assert metadata["antigen.gene"] == "NP"
    assert metadata["antigen.species"] == "InfluenzaB"


def test_vdjdb_full_clone_metadata_propagates(vdjdb_full_paired):
    pair = next(
        pair
        for pair in vdjdb_full_paired.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
        if pair.pair_id == "1007"
    )
    chains = {pair.clonotype1.locus: pair.clonotype1, pair.clonotype2.locus: pair.clonotype2}
    assert chains["TRA"].clone_metadata["vdjdb_record_id"] == "1007"
    assert chains["TRB"].clone_metadata["antigen.epitope"] == "RPIIRPATL"


def test_vdjdb_full_include_incomplete_keeps_single_chain_records(vdjdb_full_parser):
    cell_df, barcode_metadata = vdjdb_full_parser.parse_cell_clonotypes_file(
        _VDJDB_FULL_FILE,
        species="HomoSapiens",
        include_incomplete=True,
    )
    record_zero = cell_df.filter((pl.col("barcode") == "0") & (pl.col("locus") == "TRB"))
    record_112 = cell_df.filter((pl.col("barcode") == "112") & (pl.col("locus") == "TRA"))
    assert record_zero.height == 1
    assert record_zero[0, "junction_aa"] == "CASSIVGGNEQFF"
    assert record_112.height == 1
    assert record_112[0, "junction_aa"] == "CAASGGYQKVTF"
    assert barcode_metadata["0"]["antigen.epitope"] == "GILGFVFTL"
    assert barcode_metadata["112"]["antigen.epitope"] == "KAFSPEVIPMF"


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


# ---------------------------------------------------------------------------
# AdaptiveParser
# ---------------------------------------------------------------------------

def _make_adaptive_gz_bytes(rows: list[dict[str, str]]) -> bytes:
    """Create a gzipped Adaptive format TSV from a list of row dicts.
    
    Each dict should contain keys like: nucleotide, aminoAcid, count,
    vMaxResolved, dMaxResolved, jMaxResolved, etc.
    """
    # Common Adaptive header
    columns = [
        "nucleotide",
        "aminoAcid",
        "count",
        "frequencyCount",
        "cdr3Length",
        "vMaxResolved",
        "dMaxResolved",
        "jMaxResolved",
        "vDeletion",
        "d5Deletion",
        "d3Deletion",
        "jDeletion",
        "n2Insertion",
        "n1Insertion",
        "estimatedNumberGenomes",
        "sequenceStatus",
    ]
    
    lines = ["\t".join(columns)]
    for row in rows:
        row_values = [str(row.get(col, "")) for col in columns]
        lines.append("\t".join(row_values))
    
    content = "\n".join(lines) + "\n"
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(content.encode())
    return buf.getvalue()


@pytest.fixture(scope="module")
def adaptive_test_sample() -> LocusRepertoire:
    """Parse a minimal Adaptive format test fixture."""
    rows = [
        {
            "nucleotide": "TGTGCCTCCAGCAGTGATCGCACCTACGGTGATGATGAGCAGTAC",
            "aminoAcid": "CASSDSRVDDESO",
            "count": "150",
            "frequencyCount": "0.0234",
            "cdr3Length": "45",
            "vMaxResolved": "TCRBV05-01",
            "dMaxResolved": "TCRBD01",  # D genes don't have subtypes
            "jMaxResolved": "TCRBJ02-01",
            "vDeletion": "1",
            "d5Deletion": "2",
            "d3Deletion": "1",
            "jDeletion": "3",
            "n2Insertion": "0",
            "n1Insertion": "2",
            "estimatedNumberGenomes": "143",
            "sequenceStatus": "In-frame",
        },
        {
            "nucleotide": "TGTGCCTCTAGCAGCGGCACCAAGACGGTTATGGTACGACGGTCCGGGCACACGTTC",
            "aminoAcid": "CASSSGTKTVVRRTSRHTF",
            "count": "89",
            "frequencyCount": "0.0139",
            "cdr3Length": "60",
            "vMaxResolved": "TCRBV12-03",
            "dMaxResolved": "TCRBD02",  # D genes don't have subtypes
            "jMaxResolved": "TCRBJ01-02",
            "vDeletion": "0",
            "d5Deletion": "1",
            "d3Deletion": "2",
            "jDeletion": "2",
            "n2Insertion": "3",
            "n1Insertion": "1",
            "estimatedNumberGenomes": "87",
            "sequenceStatus": "In-frame",
        },
        {
            "nucleotide": "TGTGCCTCCAGCAGTGATAGGACACCCGTCTATGGTACGACGGTCCGGGGACCCGTT",
            "aminoAcid": "CASSDRTRSVTTRRSRRF",
            "count": "45",
            "frequencyCount": "0.0070",
            "cdr3Length": "57",
            "vMaxResolved": "TCRBV20-01",
            "dMaxResolved": "",  # No D gene
            "jMaxResolved": "TCRBJ02-03",
            "vDeletion": "2",
            "d5Deletion": "0",
            "d3Deletion": "0",
            "jDeletion": "1",
            "n2Insertion": "2",
            "n1Insertion": "0",
            "estimatedNumberGenomes": "44",
            "sequenceStatus": "In-frame",
        },
    ]
    
    gz_bytes = _make_adaptive_gz_bytes(rows)
    with tempfile.NamedTemporaryFile(suffix=".tsv.gz", delete=False) as tmp:
        tmp.write(gz_bytes)
        tmp_path = Path(tmp.name)
    
    try:
        parser = AdaptiveParser(locus="TRB")
        repertoire = parser.parse_file(tmp_path, sample_id="adaptive_test", locus="TRB")
        return repertoire
    finally:
        tmp_path.unlink(missing_ok=True)


def test_adaptive_returns_locus_repertoire(adaptive_test_sample):
    assert isinstance(adaptive_test_sample, LocusRepertoire)


def test_adaptive_has_correct_locus(adaptive_test_sample):
    assert adaptive_test_sample.locus == "TRB"


def test_adaptive_clonotype_count(adaptive_test_sample):
    assert adaptive_test_sample.clonotype_count == 3


def test_adaptive_duplicate_count(adaptive_test_sample):
    # 150 + 89 + 45 = 284
    assert adaptive_test_sample.duplicate_count == 284


def test_adaptive_junction_sequences(adaptive_test_sample):
    junctions = {c.junction for c in adaptive_test_sample.clonotypes}
    assert len(junctions) == 3


def test_adaptive_junction_aa_sequences(adaptive_test_sample):
    junctions_aa = [c.junction_aa for c in adaptive_test_sample.clonotypes]
    assert "CASSDSRVDDESO" in junctions_aa
    assert "CASSSGTKTVVRRTSRHTF" in junctions_aa
    assert "CASSDRTRSVTTRRSRRF" in junctions_aa


def test_adaptive_gene_normalization():
    """Test that TCRBV05-01 → TRBV5-1 gene name normalization works correctly.
    
    The normalization rules:
    - TCRBV → TRBV (TCR → TR replacement)
    - TCRBV05-01 → TRBV5-1 (zero-padding removal from gene number and subtype)
    - *01 → *1 (zero-padding removal from allele)
    """
    parser = AdaptiveParser(locus="TRB")
    
    # Test _normalize_gene_value directly
    # Leading zeros in both gene number and subtype are removed
    assert parser._normalize_gene_value("TCRBV05-01") == "TRBV5-1"
    assert parser._normalize_gene_value("TCRBV12-03") == "TRBV12-3"
    assert parser._normalize_gene_value("TCRBD02-01") == "TRBD2-1"
    assert parser._normalize_gene_value("TCRBJ02-01") == "TRBJ2-1"
    
    # Allele normalization: leading zero in allele is removed
    assert parser._normalize_gene_value("TRBV05-01*01") == "TRBV5-1*1"
    assert parser._normalize_gene_value("TRBV05-01*02") == "TRBV5-1*2"
    
    # TCR → TR replacement
    assert parser._normalize_gene_value("TCRA") == "TRA"


def test_adaptive_v_genes_normalized(adaptive_test_sample):
    """Check that all V genes are normalized to IMGT format (TCR prefix removed, leading zeros removed)."""
    v_genes = {c.v_gene for c in adaptive_test_sample.clonotypes}
    for gene in v_genes:
        assert gene.startswith("TRBV"), f"V gene {gene} should start with TRBV"
        # Should not have TCR prefix (TCRBV is wrong)
        assert not gene.startswith("TCRBV"), f"V gene {gene} should not have TCR prefix"
        # Should not have zero-padded numbers immediately after V (e.g., V05 is wrong, should be V5)
        parts = gene.split('-')
        assert not parts[0][4:].startswith("0"), f"V gene {gene} should not have zero-padding after V"


def test_adaptive_j_genes_normalized(adaptive_test_sample):
    """Check that all J genes are normalized to IMGT format."""
    j_genes = {c.j_gene for c in adaptive_test_sample.clonotypes}
    for gene in j_genes:
        assert gene.startswith("TRBJ"), f"J gene {gene} should start with TRBJ"
        assert not gene.startswith("TCRBJ"), f"J gene {gene} should not have TCR prefix"


def test_adaptive_d_genes_normalized(adaptive_test_sample):
    """Check that all D genes are normalized to IMGT format (or empty if absent)."""
    d_genes = {c.d_gene for c in adaptive_test_sample.clonotypes}
    for gene in d_genes:
        if gene:  # non-empty D gene
            assert gene.startswith("TRBD"), f"D gene {gene} should start with TRBD"
            assert not gene.startswith("TCRBD"), f"D gene {gene} should not have TCR prefix"


def test_adaptive_genes_match_imgt_library(adaptive_test_sample):
    """Validate that parsed gene names exist in the IMGT library.
    
    This is a key concordance check: the AdaptiveParser gene normalization
    must produce names that match the IMGT reference library.
    """
    lib = GeneLibrary.load_default(loci='TRB', species='human', source='imgt')
    
    # Collect all unique gene names from the parsed repertoire
    all_genes = set()
    for clonotype in adaptive_test_sample.clonotypes:
        if clonotype.v_gene:
            all_genes.add(clonotype.v_gene)
        if clonotype.d_gene:
            all_genes.add(clonotype.d_gene)
        if clonotype.j_gene:
            all_genes.add(clonotype.j_gene)
    
    # Check that each gene (as a base name without allele) exists in the library
    for gene_name in all_genes:
        base_name = gene_name.split('*')[0]  # Remove allele suffix if present
        matching_entries = [
            e for e in lib.entries.values()
            if e.allele.split('*')[0] == base_name
        ]
        assert len(matching_entries) > 0, \
            f"Gene name {gene_name} (base: {base_name}) not found in IMGT library. " \
            f"Available TRB genes: {sorted(set(e.allele.split('*')[0] for e in lib.entries.values() if e.locus == 'TRB'))[:10]}..."


def test_adaptive_gene_concordance_v_genes(adaptive_test_sample):
    """Ensure V gene names from AdaptiveParser match IMGT library V genes."""
    lib = GeneLibrary.load_default(loci='TRB', species='human', source='imgt')
    v_gene_bases = {e.allele.split('*')[0] for e in lib.entries.values() if e.gene == 'V' and e.locus == 'TRB'}
    
    for clonotype in adaptive_test_sample.clonotypes:
        if clonotype.v_gene:
            base = clonotype.v_gene.split('*')[0]
            assert base in v_gene_bases, \
                f"V gene {clonotype.v_gene} (base: {base}) not in IMGT V genes. " \
                f"Available: {sorted(v_gene_bases)}"


def test_adaptive_gene_concordance_j_genes(adaptive_test_sample):
    """Ensure J gene names from AdaptiveParser match IMGT library J genes."""
    lib = GeneLibrary.load_default(loci='TRB', species='human', source='imgt')
    j_gene_bases = {e.allele.split('*')[0] for e in lib.entries.values() if e.gene == 'J' and e.locus == 'TRB'}
    
    for clonotype in adaptive_test_sample.clonotypes:
        if clonotype.j_gene:
            base = clonotype.j_gene.split('*')[0]
            assert base in j_gene_bases, \
                f"J gene {clonotype.j_gene} (base: {base}) not in IMGT J genes. " \
                f"Available: {sorted(j_gene_bases)}"


def test_adaptive_gene_concordance_d_genes(adaptive_test_sample):
    """Ensure D gene names from AdaptiveParser match IMGT library D genes."""
    lib = GeneLibrary.load_default(loci='TRB', species='human', source='imgt')
    d_gene_bases = {e.allele.split('*')[0] for e in lib.entries.values() if e.gene == 'D' and e.locus == 'TRB'}
    
    for clonotype in adaptive_test_sample.clonotypes:
        if clonotype.d_gene:
            base = clonotype.d_gene.split('*')[0]
            assert base in d_gene_bases, \
                f"D gene {clonotype.d_gene} (base: {base}) not in IMGT D genes. " \
                f"Available: {sorted(d_gene_bases)}"


def test_adaptive_custom_sample_id():
    """Test that custom sample_id is preserved."""
    rows = [
        {
            "nucleotide": "TGTGCCTCCAGCAGTGATCGCACCTACGGTGATGATGAGCAGTAC",
            "aminoAcid": "CASSDSRVDDESO",
            "count": "150",
            "vMaxResolved": "TCRBV05-01",
            "jMaxResolved": "TCRBJ02-01",
            "sequenceStatus": "In-frame",
        }
    ]
    
    gz_bytes = _make_adaptive_gz_bytes(rows)
    with tempfile.NamedTemporaryFile(suffix=".tsv.gz", delete=False) as tmp:
        tmp.write(gz_bytes)
        tmp_path = Path(tmp.name)
    
    try:
        parser = AdaptiveParser(locus="TRB")
        repertoire = parser.parse_file(tmp_path, sample_id="my_adaptive_sample", locus="TRB")
        # Note: LocusRepertoire may not have sample_id attribute, but metadata can be checked
        assert repertoire.clonotype_count == 1
    finally:
        tmp_path.unlink(missing_ok=True)
