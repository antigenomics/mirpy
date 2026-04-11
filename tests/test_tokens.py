"""Unit tests for :mod:`mir.basic.tokens` functions."""

import unittest

from mir.basic.sequence import AA_MASK, NT_MASK, REDUCED_AA_MASK, matches
from mir.basic.tokens import tokenize, tokenize_gapped, tokenize_gapped_str, tokenize_str


class TestTokenize(unittest.TestCase):
    """Plain k-mer extraction (bytes output)."""

    def test_aa_k3(self) -> None:
        self.assertEqual(tokenize("CASSL", 3), [b"CAS", b"ASS", b"SSL"])

    def test_nt_k4(self) -> None:
        self.assertEqual(tokenize("ATCGAT", 4), [b"ATCG", b"TCGA", b"CGAT"])

    def test_reduced_k2(self) -> None:
        self.assertEqual(tokenize("slhh", 2), [b"sl", b"lh", b"hh"])

    def test_k_equals_length(self) -> None:
        self.assertEqual(tokenize("CAST", 4), [b"CAST"])

    def test_k_equals_one(self) -> None:
        self.assertEqual(tokenize("ATG", 1), [b"A", b"T", b"G"])

    def test_bytes_input(self) -> None:
        self.assertEqual(tokenize(b"CASSL", 3), [b"CAS", b"ASS", b"SSL"])

    def test_bytearray_input(self) -> None:
        self.assertEqual(tokenize(bytearray(b"ATG"), 1), [b"A", b"T", b"G"])

    def test_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            tokenize("CAST", 0)
        with self.assertRaises(ValueError):
            tokenize("CAST", 5)


class TestTokenizeStr(unittest.TestCase):
    """Plain k-mer extraction (str output)."""

    def test_basic(self) -> None:
        self.assertEqual(tokenize_str("CASSL", 3), ["CAS", "ASS", "SSL"])

    def test_bytes_input(self) -> None:
        self.assertEqual(tokenize_str(b"ATG", 1), ["A", "T", "G"])


class TestTokenizeGapped(unittest.TestCase):
    """Gapped k-mer extraction (bytes output)."""

    def test_aa_gapped_k3(self) -> None:
        gapped = tokenize_gapped("CASSL", 3, AA_MASK)
        self.assertEqual(len(gapped), 9)
        expected = [
            b"XAS", b"CXS", b"CAX",
            b"XSS", b"AXS", b"ASX",
            b"XSL", b"SXL", b"SSX",
        ]
        self.assertEqual(gapped, expected)

    def test_nt_gapped_k2(self) -> None:
        gapped = tokenize_gapped("ATG", 2, NT_MASK)
        self.assertEqual(gapped, [b"NT", b"AN", b"NG", b"TN"])

    def test_reduced_gapped_k2(self) -> None:
        gapped = tokenize_gapped("slh", 2, REDUCED_AA_MASK)
        self.assertEqual(gapped, [b"Xl", b"sX", b"Xh", b"lX"])

    def test_gapped_k1(self) -> None:
        gapped = tokenize_gapped("CA", 1, AA_MASK)
        self.assertEqual(gapped, [b"X", b"X"])

    def test_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            tokenize_gapped("CAST", 0, AA_MASK)
        with self.assertRaises(ValueError):
            tokenize_gapped("CAST", 5, AA_MASK)

    def test_bytes_input(self) -> None:
        gapped = tokenize_gapped(b"ATG", 2, NT_MASK)
        self.assertEqual(gapped, [b"NT", b"AN", b"NG", b"TN"])

    def test_gapped_match_plain(self) -> None:
        """Each gapped k-mer should wildcard-match its corresponding plain k-mer."""
        plain = tokenize("CASSL", 3)
        gapped = tokenize_gapped("CASSL", 3, AA_MASK)
        for i, kmer in enumerate(plain):
            variants = gapped[i * 3 : (i + 1) * 3]
            for var in variants:
                self.assertTrue(
                    matches(kmer, var, AA_MASK),
                    f"{kmer} should match {var}",
                )


class TestTokenizeGappedStr(unittest.TestCase):
    """Gapped k-mer extraction (str output)."""

    def test_basic(self) -> None:
        gapped = tokenize_gapped_str("CASSL", 3, "X")
        self.assertEqual(len(gapped), 9)
        self.assertEqual(gapped[0], "XAS")
        self.assertIsInstance(gapped[0], str)


if __name__ == "__main__":
    unittest.main()
