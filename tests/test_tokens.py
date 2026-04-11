"""Unit tests for ``mir.basic.tokens`` wrapper functions.

These delegates to the ``mirseq`` C extension; focus is on the wrapper
API, input normalisation, and agreement with ``mirseq`` direct calls.

Run with ``python -m pytest tests/test_tokens.py -v``.
"""

import unittest

from mir.basic.alphabets import AA_MASK, NT_MASK
from mir.basic.tokens import (
    tokenize,
    tokenize_str,
    tokenize_gapped,
    tokenize_gapped_str,
)
from mir.basic import mirseq


# ── tokenize (returns list[bytes]) ────────────────────────────────

class TestTokenize(unittest.TestCase):

    def test_basic_aa(self) -> None:
        self.assertEqual(tokenize("CASSL", 3), [b"CAS", b"ASS", b"SSL"])

    def test_basic_nt(self) -> None:
        self.assertEqual(tokenize("ATCGAT", 4),
                         [b"ATCG", b"TCGA", b"CGAT"])

    def test_k1(self) -> None:
        self.assertEqual(tokenize("ATG", 1), [b"A", b"T", b"G"])

    def test_k_eq_len(self) -> None:
        self.assertEqual(tokenize("CAST", 4), [b"CAST"])

    def test_str_input(self) -> None:
        self.assertEqual(tokenize("CAST", 2), [b"CA", b"AS", b"ST"])

    def test_bytes_input(self) -> None:
        self.assertEqual(tokenize(b"CAST", 2), [b"CA", b"AS", b"ST"])

    def test_agrees_with_c(self) -> None:
        for seq in ["CASSL", "ATCGATCGATCG"]:
            for k in [1, 2, 3]:
                with self.subTest(seq=seq, k=k):
                    self.assertEqual(tokenize(seq, k),
                                     mirseq.tokenize_bytes(seq, k))


# ── tokenize_str (returns list[str]) ─────────────────────────────

class TestTokenizeStr(unittest.TestCase):

    def test_basic(self) -> None:
        self.assertEqual(tokenize_str("CASSL", 3), ["CAS", "ASS", "SSL"])

    def test_type(self) -> None:
        result = tokenize_str("CAST", 2)
        self.assertIsInstance(result[0], str)

    def test_agrees_with_c(self) -> None:
        self.assertEqual(tokenize_str("CASSL", 3),
                         mirseq.tokenize_str("CASSL", 3))


# ── tokenize_gapped (returns list[bytes]) ────────────────────────

class TestTokenizeGapped(unittest.TestCase):

    def test_basic(self) -> None:
        expected = [
            b"XAS", b"CXS", b"CAX",
            b"XSS", b"AXS", b"ASX",
            b"XSL", b"SXL", b"SSX",
        ]
        self.assertEqual(tokenize_gapped("CASSL", 3, AA_MASK), expected)

    def test_nt(self) -> None:
        self.assertEqual(tokenize_gapped("ATG", 2, NT_MASK),
                         [b"NT", b"AN", b"NG", b"TN"])

    def test_agrees_with_c(self) -> None:
        self.assertEqual(tokenize_gapped("CASSL", 3, AA_MASK),
                         mirseq.tokenize_gapped_bytes("CASSL", 3, AA_MASK))


# ── tokenize_gapped_str (returns list[str]) ──────────────────────

class TestTokenizeGappedStr(unittest.TestCase):

    def test_basic(self) -> None:
        result = tokenize_gapped_str("CASSL", 3, "X")
        self.assertEqual(len(result), 9)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], "XAS")

    def test_agrees_with_c(self) -> None:
        self.assertEqual(
            tokenize_gapped_str("CASSL", 3, "X"),
            mirseq.tokenize_gapped_str("CASSL", 3, AA_MASK))


if __name__ == "__main__":
    unittest.main()
