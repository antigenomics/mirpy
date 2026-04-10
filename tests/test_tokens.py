"""Unit tests for :mod:`mir.basic.tokens`."""

import unittest

import numpy as np

from mir.basic.sequence import (
    AminoAcidSequence,
    NucleotideSequence,
    ReducedAminoAcidSequence,
)
from mir.basic.tokens import tokenize


def _strs(seqs):
    """Helper: list of sequences → list of str."""
    return [s.to_string() for s in seqs]


class TestTokenizePlain(unittest.TestCase):
    """Plain (non-gapped) k-mer extraction."""

    def test_amino_acid_k3(self) -> None:
        """CASSL → CAS ASS SSL."""
        aa = AminoAcidSequence.from_string("CASSL")
        kmers = tokenize(aa, k=3)
        self.assertEqual(_strs(kmers), ["CAS", "ASS", "SSL"])
        self.assertIsInstance(kmers[0], AminoAcidSequence)

    def test_nucleotide_k4(self) -> None:
        nt = NucleotideSequence.from_string("ATCGAT")
        kmers = tokenize(nt, k=4)
        self.assertEqual(_strs(kmers), ["ATCG", "TCGA", "CGAT"])
        self.assertIsInstance(kmers[0], NucleotideSequence)

    def test_reduced_k2(self) -> None:
        red = ReducedAminoAcidSequence.from_string("slhh")
        kmers = tokenize(red, k=2)
        self.assertEqual(_strs(kmers), ["sl", "lh", "hh"])

    def test_k_equals_length(self) -> None:
        """When k == len, a single k-mer equal to the sequence is returned."""
        aa = AminoAcidSequence.from_string("CAST")
        kmers = tokenize(aa, k=4)
        self.assertEqual(len(kmers), 1)
        self.assertEqual(kmers[0], aa)

    def test_k_equals_one(self) -> None:
        nt = NucleotideSequence.from_string("ATG")
        kmers = tokenize(nt, k=1)
        self.assertEqual(_strs(kmers), ["A", "T", "G"])

    def test_invalid_k(self) -> None:
        aa = AminoAcidSequence.from_string("CAST")
        with self.assertRaises(ValueError):
            tokenize(aa, k=0)
        with self.assertRaises(ValueError):
            tokenize(aa, k=5)

    def test_kmers_are_independent_copies(self) -> None:
        """Returned k-mers own their data and don't share buffers."""
        aa = AminoAcidSequence.from_string("CASSL")
        kmers = tokenize(aa, k=3)
        self.assertFalse(np.shares_memory(kmers[0].data, kmers[1].data))


class TestTokenizeGapped(unittest.TestCase):
    """Gapped k-mer extraction (single-position mask variants)."""

    def test_amino_acid_gapped_k3(self) -> None:
        """CASSL → 3 windows × 3 gap positions = 9 gapped k-mers."""
        aa = AminoAcidSequence.from_string("CASSL")
        gapped = tokenize(aa, k=3, gapped=True)
        self.assertEqual(len(gapped), 9)
        expected = [
            # window CAS
            "XAS", "CXS", "CAX",
            # window ASS
            "XSS", "AXS", "ASX",
            # window SSL
            "XSL", "SXL", "SSX",
        ]
        self.assertEqual(_strs(gapped), expected)
        self.assertIsInstance(gapped[0], AminoAcidSequence)

    def test_nucleotide_gapped_k2(self) -> None:
        nt = NucleotideSequence.from_string("ATG")
        gapped = tokenize(nt, k=2, gapped=True)
        expected = [
            "NT", "AN",  # AT
            "NG", "TN",  # TG
        ]
        self.assertEqual(_strs(gapped), expected)

    def test_reduced_gapped_k2(self) -> None:
        red = ReducedAminoAcidSequence.from_string("slh")
        gapped = tokenize(red, k=2, gapped=True)
        expected = ["Xl", "sX", "Xh", "lX"]
        self.assertEqual(_strs(gapped), expected)

    def test_gapped_k1(self) -> None:
        """With k=1, each gapped k-mer is just the mask character."""
        aa = AminoAcidSequence.from_string("CA")
        gapped = tokenize(aa, k=1, gapped=True)
        self.assertEqual(_strs(gapped), ["X", "X"])

    def test_gapped_invalid_k(self) -> None:
        aa = AminoAcidSequence.from_string("CAST")
        with self.assertRaises(ValueError):
            tokenize(aa, k=0, gapped=True)
        with self.assertRaises(ValueError):
            tokenize(aa, k=5, gapped=True)

    def test_gapped_kmers_match_plain_kmers(self) -> None:
        """Each gapped k-mer should wildcard-match its corresponding plain k-mer."""
        aa = AminoAcidSequence.from_string("CASSL")
        plain = tokenize(aa, k=3)
        gapped = tokenize(aa, k=3, gapped=True)
        for i, kmer in enumerate(plain):
            variants = gapped[i * 3 : (i + 1) * 3]
            for var in variants:
                self.assertTrue(
                    kmer.matches(var),
                    f"{kmer} should match {var}",
                )


if __name__ == "__main__":
    unittest.main()
