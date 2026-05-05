"""Tests for mir.common.gene_library (GeneEntry / GeneLibrary)."""

import unittest
from pathlib import Path

import pytest

from mir.common.gene_library import GeneEntry, GeneLibrary

RESOURCES = Path(__file__).parent.parent / "mir" / "resources" / "gene_library"
_OLGA_LIB = RESOURCES / "olga_gene_library.txt"
_IMGT_LIB = RESOURCES / "imgt_gene_library.txt"

_OLGA_PRESENT = pytest.mark.skipif(
    not _OLGA_LIB.exists(),
    reason="olga_gene_library.txt not present — run build_gene_library.py",
)
_IMGT_PRESENT = pytest.mark.skipif(
    not _IMGT_LIB.exists(),
    reason="imgt_gene_library.txt not present — run build_gene_library.py without --olga",
)


class TestGeneEntry(unittest.TestCase):

    def test_allele_inferred_from_name(self):
        e = GeneEntry("TRBV1*01")
        self.assertEqual(e.allele, "TRBV1*01")
        self.assertEqual(e.locus, "TRB")
        self.assertEqual(e.gene, "V")

    def test_explicit_locus_and_gene(self):
        e = GeneEntry("TRBV1*01", locus="TRB", gene="V")
        self.assertEqual(e.locus, "TRB")
        self.assertEqual(e.gene, "V")

    def test_bad_locus_raises(self):
        with self.assertRaises(ValueError):
            GeneEntry("XXXV1*01")

    def test_bad_gene_type_raises(self):
        with self.assertRaises(ValueError):
            GeneEntry("TRBV1*01", gene="X")

    def test_sequence_aa_auto_translated(self):
        e = GeneEntry("TRBV1*01", sequence="ATGCAG")
        self.assertIsNotNone(e.sequence_aa)
        self.assertIsInstance(e.sequence_aa, str)
        self.assertGreater(len(e.sequence_aa), 0)

    def test_explicit_sequence_aa_not_overwritten(self):
        e = GeneEntry("TRBV1*01", sequence="ATGCAG", sequence_aa="MQ")
        self.assertEqual(e.sequence_aa, "MQ")

    def test_no_sequence_sequence_aa_is_none(self):
        e = GeneEntry("TRBV1*01")
        self.assertIsNone(e.sequence_aa)

    def test_str_returns_allele(self):
        e = GeneEntry("TRBV1*01")
        self.assertEqual(str(e), "TRBV1*01")

    def test_slash_in_allele_preserved(self):
        e = GeneEntry("TRAV29/DV5*01", locus="TRA", gene="V")
        self.assertEqual(str(e), "TRAV29/DV5*01")


class TestGeneLibraryGetOrCreate(unittest.TestCase):

    def _make_lib(self, entries: dict[str, GeneEntry], complete: bool = True) -> GeneLibrary:
        return GeneLibrary(entries, complete=complete)

    def test_get_or_create_noallele_returns_minimum_allele(self):
        lib = self._make_lib({
            "IGHV3-43D*03": GeneEntry("IGHV3-43D*03", locus="IGH", gene="V"),
            "IGHV3-43D*06": GeneEntry("IGHV3-43D*06", locus="IGH", gene="V"),
        })
        result = lib.get_or_create_noallele("IGHV3-43D")
        self.assertEqual(result.allele, "IGHV3-43D*03")

    def test_get_or_create_noallele_with_allele_passthrough(self):
        lib = self._make_lib({
            "TRBV1*01": GeneEntry("TRBV1*01", locus="TRB", gene="V"),
        })
        result = lib.get_or_create_noallele("TRBV1*01")
        self.assertEqual(result.allele, "TRBV1*01")

    def test_get_or_create_preserves_slash_in_allele(self):
        lib = self._make_lib({
            "TRAV29/DV5*01": GeneEntry("TRAV29/DV5*01", locus="TRA", gene="V"),
        })
        self.assertEqual(lib.get_or_create("TRAV29/DV5*01").allele, "TRAV29/DV5*01")
        self.assertEqual(lib.get_or_create_noallele("TRAV29/DV5").allele, "TRAV29/DV5*01")

    def test_get_or_create_incomplete_lib_creates_placeholder(self):
        lib = GeneLibrary({}, complete=False)
        e = lib.get_or_create("TRBV1*01")
        self.assertEqual(e.allele, "TRBV1*01")

    def test_get_or_create_complete_lib_raises_for_unknown(self):
        lib = GeneLibrary({}, complete=True)
        with self.assertRaises(ValueError):
            lib.get_or_create("TRBV1*01")

    def test_get_or_create_entry_object(self):
        e = GeneEntry("TRBV1*01", locus="TRB", gene="V")
        lib = self._make_lib({"TRBV1*01": e})
        self.assertIs(lib.get_or_create(e), e)


class TestGeneLibraryQueries(unittest.TestCase):

    def setUp(self):
        self.lib = GeneLibrary({
            "TRBV1*01": GeneEntry("TRBV1*01", species="human", locus="TRB", gene="V", sequence="ATGCAGATG"),
            "TRBJ1*01": GeneEntry("TRBJ1*01", species="human", locus="TRB", gene="J", sequence="TTTTGG"),
            "TRAV1*01": GeneEntry("TRAV1*01", species="human", locus="TRA", gene="V", sequence="ATGATGATG"),
        }, complete=True)

    def test_get_entries_by_locus(self):
        entries = self.lib.get_entries(locus="TRB")
        self.assertEqual(len(entries), 2)

    def test_get_entries_by_gene(self):
        entries = self.lib.get_entries(gene="V")
        self.assertEqual(len(entries), 2)

    def test_get_entries_locus_and_gene(self):
        entries = self.lib.get_entries(locus="TRB", gene="V")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].allele, "TRBV1*01")

    def test_get_sequences_aa_returns_pairs(self):
        pairs = self.lib.get_sequences_aa(locus="TRB", gene="V")
        self.assertEqual(len(pairs), 1)
        allele, aa = pairs[0]
        self.assertEqual(allele, "TRBV1*01")
        self.assertIsNotNone(aa)

    def test_get_sequences_returns_nt_pairs(self):
        pairs = self.lib.get_sequences(locus="TRB", gene="V")
        self.assertEqual(pairs[0][1], "ATGCAGATG")

    def test_get_species(self):
        self.assertEqual(self.lib.get_species(), {"human"})

    def test_get_loci(self):
        self.assertEqual(self.lib.get_loci(), {"TRB", "TRA"})

    def test_get_genes(self):
        self.assertIn("V", self.lib.get_genes())
        self.assertIn("J", self.lib.get_genes())

    def test_get_summary_counts(self):
        summary = self.lib.get_summary()
        self.assertEqual(summary[("human", "TRB", "V")], 1)
        self.assertEqual(summary[("human", "TRB", "J")], 1)

    def test_is_functional_by_allele(self):
        self.assertTrue(self.lib.is_functional("TRBV1*01"))

    def test_is_functional_by_gene_base(self):
        self.assertTrue(self.lib.is_functional("TRBV1"))

    def test_is_coding_v_gene(self):
        self.assertTrue(self.lib.is_coding("TRBV1"))

    def test_nonfunctional_entry_is_not_functional(self):
        self.lib.entries["TRBVX*01"] = GeneEntry(
            "TRBVX*01", species="human", locus="TRB", gene="V", sequence="ATG", functionality="P"
        )
        self.lib._rebuild_coding_v_genes()
        self.assertFalse(self.lib.is_functional("TRBVX*01"))
        self.assertFalse(self.lib.is_coding("TRBVX"))


# ---------------------------------------------------------------------------
# Integration tests — require pre-built library files on disk
# ---------------------------------------------------------------------------

@_OLGA_PRESENT
class TestGeneLibraryLoadOlga(unittest.TestCase):

    def setUp(self):
        self.lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="olga")

    def test_not_empty(self):
        self.assertGreater(len(self.lib.entries), 0)

    def test_complete_flag_set(self):
        self.assertTrue(self.lib.complete)

    def test_only_requested_locus(self):
        self.assertEqual(self.lib.get_loci(), {"TRB"})

    def test_only_requested_species(self):
        self.assertEqual(self.lib.get_species(), {"human"})

    def test_v_d_j_genes_present(self):
        self.assertEqual(self.lib.get_genes(), {"V", "D", "J"})

    def test_alleles_have_asterisk(self):
        bad = [a for a in self.lib.entries if "*" not in a]
        self.assertEqual(bad, [])

    def test_sequence_aa_auto_translated(self):
        entries_with_seq = [e for e in self.lib.entries.values() if e.sequence_aa]
        self.assertGreater(len(entries_with_seq), 0)

    def test_get_or_create_noallele_resolves(self):
        result = self.lib.get_or_create_noallele("TRBV10-3")
        self.assertIn("*", result.allele)
        self.assertTrue(result.allele.startswith("TRBV10-3*"))

    def test_all_v_entries_are_functional(self):
        v_entries = self.lib.get_entries(gene="V")
        self.assertGreater(len(v_entries), 0)
        self.assertTrue(all(e.is_functional for e in v_entries))

    def test_all_v_genes_marked_functional_by_default(self):
        v_entries = self.lib.get_entries(gene="V")
        v_bases = {e.allele.split("*", 1)[0] for e in v_entries}
        self.assertTrue(all(self.lib.is_functional(v) for v in v_bases))

    def test_olga_is_coding_set_contains_all_v_bases(self):
        v_entries = self.lib.get_entries(gene="V")
        v_bases = {e.allele.split("*", 1)[0] for e in v_entries}
        self.assertEqual(v_bases, self.lib.coding_v_genes)


@_IMGT_PRESENT
class TestGeneLibraryLoadImgt(unittest.TestCase):

    def setUp(self):
        self.lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="imgt")

    def test_not_empty(self):
        self.assertGreater(len(self.lib.entries), 0)

    def test_only_requested_locus(self):
        self.assertEqual(self.lib.get_loci(), {"TRB"})

    def test_functionality_annotations_present(self):
        vals = {e.functionality for e in self.lib.entries.values()}
        self.assertTrue(vals)
        self.assertTrue(vals.issubset({"F", "ORF", "P"}))


if __name__ == "__main__":
    unittest.main()
