"""Tests for mir/resources/build_gene_library.py.

Unit tests exercise parsing and building functions with local fixtures.
Integration tests validate the structure of generated library files on disk;
they are skipped when the corresponding file is absent.

Run unit tests only::

    pytest tests/test_gene_library.py -v

Run all tests including integration::

    RUN_BENCHMARK=1 pytest tests/test_gene_library.py -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mir.resources.gene_library.build_gene_library import (
    IMGT_LOCI,
    LOCI_WITH_D,
    OLGA_MODEL_MAP,
    Row,
    _parse_imgt_fasta,
    _parse_olga_model_params,
    build_imgt_library,
    build_olga_library,
    check_library_consistency,
    compute_stats,
)

RESOURCES    = Path(__file__).parent.parent / "mir" / "resources" / "gene_library"
OLGA_LIB     = RESOURCES / "olga_gene_library.txt"
IMGT_LIB     = RESOURCES / "imgt_gene_library.txt"

# ---------------------------------------------------------------------------
# Expected coverage definitions
# ---------------------------------------------------------------------------

#: (species, locus) → required gene types for the OLGA library.
# Mouse B-cell (IG*) OLGA models use synthetic non-IMGT allele names and are excluded.
OLGA_EXPECTED: dict[tuple[str, str], set[str]] = {
    ("human", "TRB"): {"V", "D", "J"},
    ("human", "TRA"): {"V", "J"},
    ("human", "TRG"): {"V", "J"},
    ("human", "TRD"): {"V", "D", "J"},
    ("human", "IGH"): {"V", "D", "J"},
    ("human", "IGK"): {"V", "J"},
    ("human", "IGL"): {"V", "J"},
    ("mouse", "TRB"): {"V", "D", "J"},
    ("mouse", "TRA"): {"V", "J"},
}

#: (species, locus) → required gene types for the IMGT library.
IMGT_EXPECTED: dict[tuple[str, str], set[str]] = {
    (species, locus): ({"V", "D", "J"} if locus in LOCI_WITH_D else {"V", "J"})
    for species in IMGT_LOCI
    for locus in IMGT_LOCI[species]
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_library(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 5:
                rows.append(tuple(parts))  # type: ignore[arg-type]
    return rows


def _imgt_fasta_entry(allele: str, sequence: str) -> str:
    """Build a minimal well-formed IMGT FASTA entry (16 pipe-separated fields)."""
    fields = [""] * 16
    fields[1]  = allele
    fields[15] = sequence
    return ">" + "|".join(fields) + "\n"


# ---------------------------------------------------------------------------
# Unit: _parse_olga_model_params
# ---------------------------------------------------------------------------

class TestParseOlgaModelParams(unittest.TestCase):

    def _write(self, content: str) -> Path:
        d = tempfile.mkdtemp()
        p = Path(d) / "model_params.txt"
        p.write_text(content, encoding="utf-8")
        return p

    def test_parses_v_gene_section(self):
        p = self._write(
            "@Event_list\n"
            "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
            "%TRBV1*01;ATGCATGC;10\n"
            "%TRBV2*01;GCTAGCTA;5\n"
        )
        records = _parse_olga_model_params(p)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], ("V", "TRBV1*01", "ATGCATGC"))
        self.assertEqual(records[1], ("V", "TRBV2*01", "GCTAGCTA"))

    def test_parses_d_gene_with_leading_space(self):
        """D-gene lines use '% ' (percent then space)."""
        p = self._write(
            "@Event_list\n"
            "#GeneChoice;D_gene;Undefined_side;6;d_gene\n"
            "% TRBD1*01;GGGACAGGGGGC;0\n"
        )
        records = _parse_olga_model_params(p)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0], ("D", "TRBD1*01", "GGGACAGGGGGC"))

    def test_parses_j_gene_section(self):
        p = self._write(
            "@Event_list\n"
            "#GeneChoice;J_gene;Undefined_side;7;j_choice\n"
            "%TRBJ1-1*01;TGAACACT;0\n"
        )
        records = _parse_olga_model_params(p)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][0], "J")
        self.assertEqual(records[0][1], "TRBJ1-1*01")

    def test_ignores_entries_outside_gene_sections(self):
        """Lines before any #GeneChoice header must be ignored."""
        p = self._write(
            "@Event_list\n"
            "#DinucMarkov;VD_dinucl;...\n"
            "%stray;ATGC;0\n"
            "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
            "%TRBV1*01;ATGC;1\n"
        )
        records = _parse_olga_model_params(p)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1], "TRBV1*01")

    def test_skips_malformed_lines_with_no_semicolons(self):
        p = self._write(
            "@Event_list\n"
            "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
            "%TRBV1*01\n"
        )
        records = _parse_olga_model_params(p)
        self.assertEqual(len(records), 0)

    def test_parses_all_three_gene_types(self):
        p = self._write(
            "@Event_list\n"
            "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
            "%TRBV1*01;AAAA;1\n"
            "#GeneChoice;D_gene;Undefined_side;6;d_gene\n"
            "% TRBD1*01;GGGG;0\n"
            "#GeneChoice;J_gene;Undefined_side;7;j_choice\n"
            "%TRBJ1*01;TTTT;1\n"
        )
        records = _parse_olga_model_params(p)
        gene_types = {r[0] for r in records}
        self.assertEqual(gene_types, {"V", "D", "J"})


# ---------------------------------------------------------------------------
# Unit: _parse_imgt_fasta
# ---------------------------------------------------------------------------

class TestParseImgtFasta(unittest.TestCase):

    def test_parses_allele_and_sequence(self):
        fasta = _imgt_fasta_entry("TRBV1*01", "ATGCATGC")
        rows = _parse_imgt_fasta(fasta, "human", "TRB", "V")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("human", "TRB", "V", "TRBV1*01", "ATGCATGC"))

    def test_appends_star_01_when_allele_has_no_asterisk(self):
        fields = [""] * 16
        fields[1], fields[15] = "TRBV1", "ATGC"
        fasta = ">" + "|".join(fields) + "\n"
        rows = _parse_imgt_fasta(fasta, "human", "TRB", "V")
        self.assertEqual(rows[0][3], "TRBV1*01")

    def test_uppercases_sequence(self):
        fasta = _imgt_fasta_entry("TRBV1*01", "atgcatgc")
        rows = _parse_imgt_fasta(fasta, "human", "TRB", "V")
        self.assertEqual(rows[0][4], "ATGCATGC")

    def test_removes_gap_dots_from_sequence(self):
        fasta = _imgt_fasta_entry("TRBV1*01", "ATG.CAT.GC")
        rows = _parse_imgt_fasta(fasta, "human", "TRB", "V")
        self.assertEqual(rows[0][4], "ATGCATGC")

    def test_skips_entries_with_fewer_than_16_fields(self):
        fasta = ">TRBV1*01|ATGC\n"
        rows = _parse_imgt_fasta(fasta, "human", "TRB", "V")
        self.assertEqual(len(rows), 0)

    def test_skips_empty_entries(self):
        rows = _parse_imgt_fasta(">\n>\n", "human", "TRB", "V")
        self.assertEqual(len(rows), 0)

    def test_parses_multiple_entries(self):
        fasta = "".join(
            _imgt_fasta_entry(f"TRBV{i}*01", "ATGC") for i in range(1, 4)
        )
        rows = _parse_imgt_fasta(fasta, "human", "TRB", "V")
        self.assertEqual(len(rows), 3)
        alleles = [r[3] for r in rows]
        self.assertIn("TRBV1*01", alleles)
        self.assertIn("TRBV3*01", alleles)

    def test_species_locus_gene_forwarded_correctly(self):
        fasta = _imgt_fasta_entry("TRAJ1*01", "TTTT")
        rows = _parse_imgt_fasta(fasta, "mouse", "TRA", "J")
        self.assertEqual(rows[0][:3], ("mouse", "TRA", "J"))


# ---------------------------------------------------------------------------
# Unit: build_olga_library
# ---------------------------------------------------------------------------

class TestBuildOlgaLibrary(unittest.TestCase):

    def _make_model(self, root: Path, model_name: str, content: str) -> None:
        model_dir = root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_params.txt").write_text(content, encoding="utf-8")

    def test_returns_rows_for_present_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._make_model(d, "human_T_beta",
                "@Event_list\n"
                "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
                "%TRBV1*01;ATGC;1\n"
                "#GeneChoice;J_gene;Undefined_side;7;j_choice\n"
                "%TRBJ1*01;TTTT;1\n"
            )
            rows = build_olga_library(models_dirs=[d])

        locus_rows = [(s, l, g, a) for s, l, g, a, _ in rows if l == "TRB"]
        self.assertIn(("human", "TRB", "V", "TRBV1*01"), locus_rows)
        self.assertIn(("human", "TRB", "J", "TRBJ1*01"), locus_rows)

    def test_skips_absent_models_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = build_olga_library(models_dirs=[Path(tmp)])
        self.assertEqual(rows, [])

    def test_first_matching_directory_takes_priority(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            d1, d2 = Path(t1), Path(t2)
            self._make_model(d1, "human_T_beta",
                "@Event_list\n"
                "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
                "%TRBV_D1*01;AAAA;1\n"
            )
            self._make_model(d2, "human_T_beta",
                "@Event_list\n"
                "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
                "%TRBV_D2*01;CCCC;1\n"
            )
            rows = build_olga_library(models_dirs=[d1, d2])

        alleles = [a for _, _, _, a, _ in rows]
        self.assertIn("TRBV_D1*01", alleles)
        self.assertNotIn("TRBV_D2*01", alleles)

    def test_species_and_locus_set_from_model_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._make_model(d, "mouse_T_beta",
                "@Event_list\n"
                "#GeneChoice;V_gene;Undefined_side;7;v_choice\n"
                "%TRBV1*01;ATGC;1\n"
            )
            rows = build_olga_library(models_dirs=[d])

        self.assertTrue(all(s == "mouse" and l == "TRB" for s, l, *_ in rows))


# ---------------------------------------------------------------------------
# Unit: build_imgt_library
# ---------------------------------------------------------------------------

class TestBuildImgtLibrary(unittest.TestCase):

    def _urlopen_side_effect(self, url_suffix_map: dict[str, str]):
        """Return a side-effect function that matches URL suffixes."""
        def side_effect(url: str):
            for suffix, fasta_text in url_suffix_map.items():
                if url.endswith(suffix):
                    mock = MagicMock()
                    mock.read.return_value = fasta_text.encode("utf-8")
                    return mock
            raise AssertionError(f"Unexpected URL: {url}")
        return side_effect

    def test_fetches_v_and_j_only_for_vj_locus(self):
        url_map = {
            "TRAV.fasta": _imgt_fasta_entry("TRAV1*01", "ATGC"),
            "TRAJ.fasta": _imgt_fasta_entry("TRAJ1*01", "TTTT"),
        }
        with patch("mir.resources.gene_library.build_gene_library.urllib.request.urlopen",
                   side_effect=self._urlopen_side_effect(url_map)):
            rows = build_imgt_library(
                species_list=["human"],
                loci_map={"human": ["TRA"]},
            )
        genes = {g for _, _, g, _, _ in rows}
        self.assertEqual(genes, {"V", "J"})

    def test_fetches_v_d_j_for_locus_with_d(self):
        url_map = {
            "TRBV.fasta": _imgt_fasta_entry("TRBV1*01", "ATGC"),
            "TRBJ.fasta": _imgt_fasta_entry("TRBJ1*01", "TTTT"),
            "TRBD.fasta": _imgt_fasta_entry("TRBD1*01", "GGGG"),
        }
        with patch("mir.resources.gene_library.build_gene_library.urllib.request.urlopen",
                   side_effect=self._urlopen_side_effect(url_map)):
            rows = build_imgt_library(
                species_list=["human"],
                loci_map={"human": ["TRB"]},
            )
        genes = {g for _, _, g, _, _ in rows}
        self.assertEqual(genes, {"V", "D", "J"})

    def test_uses_correct_imgt_species_in_url(self):
        seen_urls: list[str] = []

        def capture_url(url: str):
            seen_urls.append(url)
            mock = MagicMock()
            mock.read.return_value = b""
            return mock

        with patch("mir.resources.gene_library.build_gene_library.urllib.request.urlopen",
                   side_effect=capture_url):
            build_imgt_library(
                species_list=["human"],
                loci_map={"human": ["TRA"]},
            )

        self.assertTrue(all("Homo_sapiens" in u for u in seen_urls))
        self.assertFalse(any("human" in u for u in seen_urls))

    def test_continues_on_network_error(self):
        def raise_error(url: str):
            raise OSError("network down")

        with patch("mir.resources.gene_library.build_gene_library.urllib.request.urlopen",
                   side_effect=raise_error):
            rows = build_imgt_library(
                species_list=["human"],
                loci_map={"human": ["TRA"]},
            )
        self.assertEqual(rows, [])

    def test_multi_species_rows_tagged_correctly(self):
        fasta_h = _imgt_fasta_entry("TRAV1*01", "ATGC")
        fasta_m = _imgt_fasta_entry("TRAV1*01", "GCTA")

        def side_effect(url: str):
            mock = MagicMock()
            if "Homo_sapiens" in url:
                mock.read.return_value = fasta_h.encode()
            else:
                mock.read.return_value = fasta_m.encode()
            return mock

        with patch("mir.resources.gene_library.build_gene_library.urllib.request.urlopen",
                   side_effect=side_effect):
            rows = build_imgt_library(
                species_list=["human", "mouse"],
                loci_map={"human": ["TRA"], "mouse": ["TRA"]},
            )

        species_in_rows = {r[0] for r in rows}
        self.assertIn("human", species_in_rows)
        self.assertIn("mouse", species_in_rows)


# ---------------------------------------------------------------------------
# Unit: compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats(unittest.TestCase):

    def test_counts_total_alleles_per_key(self):
        rows: list[Row] = [
            ("human", "TRB", "V", "TRBV1*01", "ATGC"),
            ("human", "TRB", "V", "TRBV1*02", "ATGC"),
            ("human", "TRB", "J", "TRBJ1*01", "TTTT"),
        ]
        total, _ = compute_stats(rows)
        self.assertEqual(total[("human", "TRB", "V")], 2)
        self.assertEqual(total[("human", "TRB", "J")], 1)

    def test_counts_only_star01_as_major(self):
        rows: list[Row] = [
            ("human", "TRB", "V", "TRBV1*01", "ATGC"),
            ("human", "TRB", "V", "TRBV1*02", "ATGC"),
            ("human", "TRB", "V", "TRBV1*11", "ATGC"),
        ]
        _, major = compute_stats(rows)
        self.assertEqual(major[("human", "TRB", "V")], 1)

    def test_empty_input_returns_empty_dicts(self):
        total, major = compute_stats([])
        self.assertEqual(len(total), 0)
        self.assertEqual(len(major), 0)

    def test_multiple_species_tracked_separately(self):
        rows: list[Row] = [
            ("human", "TRB", "V", "TRBV1*01", "ATGC"),
            ("mouse", "TRB", "V", "TRBV1*01", "ATGC"),
        ]
        total, _ = compute_stats(rows)
        self.assertEqual(total[("human", "TRB", "V")], 1)
        self.assertEqual(total[("mouse", "TRB", "V")], 1)


# ---------------------------------------------------------------------------
# Integration: OLGA library file (always present after first script run)
# ---------------------------------------------------------------------------

_OLGA_PRESENT = pytest.mark.skipif(
    not OLGA_LIB.exists(), reason="olga_gene_library.txt not found — run build_gene_library.py"
)

_IMGT_PRESENT = pytest.mark.skipif(
    not IMGT_LIB.exists(), reason="imgt_gene_library.txt not found — run build_gene_library.py without --olga"
)


@_OLGA_PRESENT
def test_olga_library_not_empty():
    rows = _load_library(OLGA_LIB)
    assert len(rows) > 0


@_OLGA_PRESENT
@pytest.mark.parametrize("species,locus,expected_genes", [
    (s, l, g) for (s, l), g in OLGA_EXPECTED.items()
])
def test_olga_species_locus_not_empty(species, locus, expected_genes):
    rows = _load_library(OLGA_LIB)
    subset = [r for r in rows if r[0] == species and r[1] == locus]
    assert len(subset) > 0, f"No rows for ({species}, {locus}) in OLGA library"


@_OLGA_PRESENT
@pytest.mark.parametrize("species,locus,expected_genes", [
    (s, l, g) for (s, l), g in OLGA_EXPECTED.items()
])
def test_olga_required_genes_present(species, locus, expected_genes):
    rows = _load_library(OLGA_LIB)
    found = {r[2] for r in rows if r[0] == species and r[1] == locus}
    missing = expected_genes - found
    assert not missing, (
        f"({species}, {locus}): expected genes {expected_genes}, "
        f"found {found}, missing {missing}"
    )


@_OLGA_PRESENT
def test_olga_allele_names_contain_asterisk():
    rows = _load_library(OLGA_LIB)
    bad = [r[3] for r in rows if "*" not in r[3]]
    assert not bad, f"Alleles without '*': {bad[:5]}"


@_OLGA_PRESENT
def test_olga_sequences_are_uppercase_dna():
    import re
    rows = _load_library(OLGA_LIB)
    # Allow standard IUPAC ambiguity codes (N, Y, R, …) present in a few OLGA alleles.
    bad = [r[3] for r in rows if not re.fullmatch(r"[ACGTNRYWSKMBDHV]+", r[4])]
    assert not bad, f"Non-uppercase DNA sequences for: {bad[:5]}"


# ---------------------------------------------------------------------------
# Integration: IMGT library file (optional, network-dependent)
# ---------------------------------------------------------------------------

@_IMGT_PRESENT
def test_imgt_library_not_empty():
    rows = _load_library(IMGT_LIB)
    assert len(rows) > 0


@_IMGT_PRESENT
@pytest.mark.parametrize("species,locus,expected_genes", [
    (s, l, g) for (s, l), g in IMGT_EXPECTED.items()
])
def test_imgt_species_locus_not_empty(species, locus, expected_genes):
    rows = _load_library(IMGT_LIB)
    subset = [r for r in rows if r[0] == species and r[1] == locus]
    assert len(subset) > 0, f"No rows for ({species}, {locus}) in IMGT library"


@_IMGT_PRESENT
@pytest.mark.parametrize("species,locus,expected_genes", [
    (s, l, g) for (s, l), g in IMGT_EXPECTED.items()
])
def test_imgt_required_genes_present(species, locus, expected_genes):
    rows = _load_library(IMGT_LIB)
    found = {r[2] for r in rows if r[0] == species and r[1] == locus}
    missing = expected_genes - found
    assert not missing, (
        f"({species}, {locus}): expected genes {expected_genes}, "
        f"found {found}, missing {missing}"
    )


@_IMGT_PRESENT
def test_imgt_allele_names_contain_asterisk():
    rows = _load_library(IMGT_LIB)
    bad = [r[3] for r in rows if "*" not in r[3]]
    assert not bad, f"Alleles without '*': {bad[:5]}"


# ---------------------------------------------------------------------------
# Unit: check_library_consistency
# ---------------------------------------------------------------------------

class TestCheckLibraryConsistency(unittest.TestCase):

    def _rows(self, entries: list[tuple[str, str, str, str]]) -> list[Row]:
        """Build Row list from (species, locus, gene, allele) tuples."""
        return [(s, l, g, a, "ATGC") for s, l, g, a in entries]

    def test_returns_string(self):
        result = check_library_consistency([], [])
        self.assertIsInstance(result, str)

    def test_shared_key_appears_in_table(self):
        olga = self._rows([("human", "TRB", "V", "TRBV1*01")])
        imgt = self._rows([("human", "TRB", "V", "TRBV1*01")])
        report = check_library_consistency(olga, imgt)
        self.assertIn("human", report)
        self.assertIn("TRB", report)
        self.assertIn("V", report)

    def test_only_olga_key_reported(self):
        olga = self._rows([("human", "TRG", "V", "TRGV1*01")])
        report = check_library_consistency(olga, [])
        self.assertIn("[keys in OLGA, absent in IMGT]", report)
        self.assertIn("TRG", report)

    def test_only_imgt_key_reported(self):
        imgt = self._rows([("mouse", "TRD", "D", "TRDD1*01")])
        report = check_library_consistency([], imgt)
        self.assertIn("[keys in IMGT, absent in OLGA]", report)
        self.assertIn("TRD", report)

    def test_shared_counts_correct(self):
        olga = self._rows([
            ("human", "TRB", "V", "TRBV1*01"),
            ("human", "TRB", "V", "TRBV2*01"),
        ])
        imgt = self._rows([
            ("human", "TRB", "V", "TRBV1*01"),
            ("human", "TRB", "V", "TRBV3*01"),
        ])
        report = check_library_consistency(olga, imgt)
        # 2 in OLGA, 2 in IMGT, 1 shared, 1 only-OLGA, 1 only-IMGT
        self.assertIn("allele counts", report)
        lines = [l for l in report.splitlines() if "human" in l and "TRB" in l and "V" in l]
        self.assertEqual(len(lines), 1)
        parts = lines[0].split()
        olga_n, imgt_n, shared, only_o, only_i = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
        self.assertEqual(olga_n, 2)
        self.assertEqual(imgt_n, 2)
        self.assertEqual(shared, 1)
        self.assertEqual(only_o, 1)
        self.assertEqual(only_i, 1)

    def test_identical_libraries_no_exclusive_alleles(self):
        rows = self._rows([("human", "TRB", "V", "TRBV1*01")])
        report = check_library_consistency(rows, rows)
        self.assertNotIn("[keys in OLGA, absent in IMGT]", report)
        self.assertNotIn("[keys in IMGT, absent in OLGA]", report)

    def test_no_section_headers_when_no_gaps(self):
        rows = self._rows([
            ("human", "TRB", "V", "TRBV1*01"),
            ("human", "TRB", "J", "TRBJ1*01"),
        ])
        report = check_library_consistency(rows, rows)
        self.assertNotIn("absent", report)


# ---------------------------------------------------------------------------
# Integration: build log
# ---------------------------------------------------------------------------

_LOG_PATH = RESOURCES / "build_gene_library.log"

_LOG_PRESENT = pytest.mark.skipif(
    not _LOG_PATH.exists(),
    reason="build_gene_library.log not found — run build_gene_library.py",
)


@_LOG_PRESENT
def test_log_contains_olga_section():
    text = _LOG_PATH.read_text(encoding="utf-8")
    assert "OLGA gene library" in text


@_LOG_PRESENT
def test_log_contains_imgt_section():
    text = _LOG_PATH.read_text(encoding="utf-8")
    assert "IMGT gene library" in text


@_LOG_PRESENT
def test_log_contains_consistency_section():
    text = _LOG_PATH.read_text(encoding="utf-8")
    assert "Consistency: OLGA vs IMGT" in text


@_LOG_PRESENT
def test_log_contains_commit_hash():
    text = _LOG_PATH.read_text(encoding="utf-8")
    assert "commit" in text


if __name__ == "__main__":
    unittest.main()
