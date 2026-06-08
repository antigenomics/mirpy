"""Tests for germline region annotations (companion file + arda integration)."""

import shutil
import unittest
from pathlib import Path

import pytest

from mir.common.gene_library import GeneLibrary

RESOURCES = Path(__file__).parent.parent / "mir" / "resources" / "gene_library"
_REGION_FILE = RESOURCES / "region_annotations.txt"

_REGION_PRESENT = pytest.mark.skipif(
    not _REGION_FILE.exists(),
    reason="region_annotations.txt not present — run build_region_annotations.py",
)


def _arda_available() -> bool:
    try:
        import arda.annotate.mapper  # noqa: F401
    except ImportError:
        return False
    return shutil.which("mmseqs") is not None


_ARDA_PRESENT = pytest.mark.skipif(
    not _arda_available(),
    reason="arda + mmseqs2 not installed (optional 'arda' extra)",
)

_V_REGIONS = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3")


@_REGION_PRESENT
class TestCompanionRegionFile(unittest.TestCase):
    """Validate the committed region_annotations.txt as loaded via load_default."""

    def test_v_gene_has_all_v_side_regions(self):
        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
        entry = lib.entries["TRBV9*01"]
        for region in _V_REGIONS:
            seq = entry.region_aa.get(region)
            self.assertTrue(seq, f"TRBV9*01 missing {region}")
            self.assertTrue(seq.isalpha(), f"{region} not amino acids: {seq!r}")
        # CDRs are short loops; FRs are longer scaffolds.
        self.assertLess(len(entry.region_aa["cdr1"]), len(entry.region_aa["fwr1"]))
        self.assertLess(len(entry.region_aa["cdr2"]), len(entry.region_aa["fwr3"]))

    def test_j_gene_has_fwr4_and_jcdr3(self):
        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
        # TRBJ FR4 is the conserved FG.G stretch; jcdr3 is the J's CDR3 part.
        entry = lib.entries["TRBJ1-6*01"]
        self.assertTrue(entry.region_aa.get("fwr4", "").startswith("F"))
        self.assertTrue(entry.region_aa.get("jcdr3"))

    def test_get_region_sequences_aa_matches_v_entries(self):
        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
        v_entries = lib.get_entries("TRB", "V")
        cdr1 = lib.get_region_sequences_aa("TRB", "V", "cdr1")
        # Every functional V allele in the OLGA library should be annotated.
        self.assertEqual(len(cdr1), len(v_entries))
        self.assertTrue(all(seq for _, seq in cdr1))

    def test_mouse_and_multi_locus_coverage(self):
        lib = GeneLibrary.load_default(loci={"TRA"}, species={"mouse"})
        self.assertGreater(len(lib.get_region_sequences_aa("TRA", "V", "cdr1")), 50)

    def test_with_regions_false_leaves_region_aa_empty(self):
        lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, with_regions=False)
        self.assertEqual(lib.entries["TRBV9*01"].region_aa, {})


@_ARDA_PRESENT
class TestArdaAnnotation(unittest.TestCase):
    """Live arda annotation of a tiny custom library (build-time path)."""

    def test_annotate_mini_library(self):
        from mir.common.region_annotation import annotate_gene_library

        full = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, with_regions=False)
        v = full.get_entries("TRB", "V")[:2]
        j = full.get_entries("TRB", "J")[:2]
        mini = GeneLibrary({e.allele: e for e in v + j}, complete=True)

        ann = annotate_gene_library(mini, "human")
        for entry in v:
            self.assertIn(entry.allele, ann)
            for region in _V_REGIONS:
                self.assertIn(region, ann[entry.allele])
        for entry in j:
            self.assertIn("fwr4", ann[entry.allele])
            self.assertIn("jcdr3", ann[entry.allele])


if __name__ == "__main__":
    unittest.main()
