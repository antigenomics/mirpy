"""Unit tests for the ``seqdist_c`` C extension and ``seqdist`` wrapper.

Covers: Hamming distance, Levenshtein distance.

Run with ``python -m pytest tests/test_seqdist.py -v``.
"""

import unittest

from mir.distances import seqdist_c
from mir.distances.seqdist import hamming, levenshtein


# ── Hamming distance ──────────────────────────────────────────────

class TestHamming(unittest.TestCase):

    def test_identical(self) -> None:
        self.assertEqual(seqdist_c.hamming("CAST", "CAST"), 0)

    def test_one_mismatch(self) -> None:
        self.assertEqual(seqdist_c.hamming("CAST", "CAAT"), 1)

    def test_all_mismatch(self) -> None:
        self.assertEqual(seqdist_c.hamming("AAAA", "TTTT"), 4)

    def test_empty(self) -> None:
        self.assertEqual(seqdist_c.hamming("", ""), 0)

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(Exception):
            seqdist_c.hamming("ABC", "AB")

    def test_bytes_input(self) -> None:
        self.assertEqual(seqdist_c.hamming(b"CAST", b"CAAT"), 1)

    def test_wrapper(self) -> None:
        self.assertEqual(hamming("CAST", "CAAT"), 1)


# ── Levenshtein distance ─────────────────────────────────────────

class TestLevenshtein(unittest.TestCase):

    def test_classic(self) -> None:
        self.assertEqual(seqdist_c.levenshtein("kitten", "sitting"), 3)

    def test_identical(self) -> None:
        self.assertEqual(seqdist_c.levenshtein("CAST", "CAST"), 0)

    def test_insertion(self) -> None:
        self.assertEqual(seqdist_c.levenshtein("ABC", "ABCD"), 1)

    def test_deletion(self) -> None:
        self.assertEqual(seqdist_c.levenshtein("ABCD", "ABC"), 1)

    def test_substitution(self) -> None:
        self.assertEqual(seqdist_c.levenshtein("ABC", "AXC"), 1)

    def test_empty_vs_nonempty(self) -> None:
        self.assertEqual(seqdist_c.levenshtein("", "ABC"), 3)
        self.assertEqual(seqdist_c.levenshtein("ABC", ""), 3)

    def test_both_empty(self) -> None:
        self.assertEqual(seqdist_c.levenshtein("", ""), 0)

    def test_bytes_input(self) -> None:
        self.assertEqual(seqdist_c.levenshtein(b"kitten", b"sitting"), 3)

    def test_wrapper(self) -> None:
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_symmetric(self) -> None:
        self.assertEqual(
            seqdist_c.levenshtein("CASSL", "CASSQL"),
            seqdist_c.levenshtein("CASSQL", "CASSL"))


if __name__ == "__main__":
    unittest.main()
