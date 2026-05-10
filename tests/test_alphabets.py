"""Unit tests for ``mir.basic.alphabets``.

Covers: Seq helpers, alphabet LUTs, aa_to_reduced (Python path),
validate, mask, matches, matches_aa_reduced.

Run with ``python -m pytest tests/test_alphabets.py -v``.
"""

import unittest

from mir.basic.alphabets import (
    _to_bytes,
    make_alphabet,
    NT_ALPHABET,
    AA_ALPHABET,
    REDUCED_AA_ALPHABET,
    NT_CHARS,
    AA_CHARS,
    REDUCED_AA_CHARS,
    AA_MASK,
    AA_TO_REDUCED,
    AA_TO_REDUCED_TABLE,
    _AA_TO_REDUCED_LUT,
    aa_to_reduced,
    validate,
    mask,
    matches,
    matches_aa_reduced,
)


# ── _to_bytes ─────────────────────────────────────────────────────

class TestToBytes(unittest.TestCase):

    def test_str(self) -> None:
        self.assertEqual(_to_bytes("CAST"), b"CAST")

    def test_bytes(self) -> None:
        self.assertEqual(_to_bytes(b"CAST"), b"CAST")

    def test_bytearray(self) -> None:
        self.assertEqual(_to_bytes(bytearray(b"CAST")), b"CAST")

    def test_empty(self) -> None:
        self.assertEqual(_to_bytes(""), b"")


# ── Alphabet LUTs ─────────────────────────────────────────────────

class TestAlphabets(unittest.TestCase):

    def test_nt_lut_size(self) -> None:
        self.assertEqual(len(NT_ALPHABET), 256)

    def test_nt_chars_allowed(self) -> None:
        for ch in NT_CHARS:
            self.assertEqual(NT_ALPHABET[ord(ch)], 1, ch)

    def test_nt_lowercase_disallowed(self) -> None:
        for ch in "atgcn":
            self.assertEqual(NT_ALPHABET[ord(ch)], 0, ch)

    def test_aa_chars_allowed(self) -> None:
        for ch in AA_CHARS:
            self.assertEqual(AA_ALPHABET[ord(ch)], 1, ch)

    def test_reduced_chars_allowed(self) -> None:
        for ch in REDUCED_AA_CHARS:
            self.assertEqual(REDUCED_AA_ALPHABET[ord(ch)], 1, ch)

    def test_make_alphabet_custom(self) -> None:
        lut = make_alphabet("AB")
        self.assertEqual(lut[ord("A")], 1)
        self.assertEqual(lut[ord("B")], 1)
        self.assertEqual(lut[ord("C")], 0)


# ── AA → reduced ─────────────────────────────────────────────────

class TestAaToReduced(unittest.TestCase):

    def test_str_input(self) -> None:
        self.assertEqual(aa_to_reduced("CASTIVGGLSQDKIVW"), b"slhhllGGlhmcbllW")

    def test_bytes_input(self) -> None:
        self.assertEqual(aa_to_reduced(b"CASTIVGGLSQDKIVW"), b"slhhllGGlhmcbllW")

    def test_empty(self) -> None:
        self.assertEqual(aa_to_reduced(""), b"")

    def test_specials(self) -> None:
        self.assertEqual(aa_to_reduced("*_X"), b"*_X")

    def test_each_aa(self) -> None:
        for aa, exp in AA_TO_REDUCED.items():
            with self.subTest(aa=aa):
                self.assertEqual(aa_to_reduced(aa), exp.encode())

    def test_table_consistency(self) -> None:
        for aa, exp in AA_TO_REDUCED.items():
            with self.subTest(aa=aa):
                self.assertEqual(AA_TO_REDUCED_TABLE[ord(aa)], ord(exp))

    def test_lut_consistency(self) -> None:
        for aa, exp in AA_TO_REDUCED.items():
            with self.subTest(aa=aa):
                self.assertEqual(_AA_TO_REDUCED_LUT[ord(aa)], ord(exp))


# ── validate ──────────────────────────────────────────────────────

class TestValidate(unittest.TestCase):

    def test_valid_nt(self) -> None:
        self.assertEqual(validate("ATGCN", NT_ALPHABET), b"ATGCN")

    def test_valid_aa(self) -> None:
        self.assertEqual(validate("CASTIVW*_X", AA_ALPHABET), b"CASTIVW*_X")

    def test_invalid_nt_lowercase(self) -> None:
        with self.assertRaises(ValueError):
            validate("atgc", NT_ALPHABET)

    def test_invalid_aa_number(self) -> None:
        with self.assertRaises(ValueError):
            validate("CAST1", AA_ALPHABET)

    def test_empty(self) -> None:
        self.assertEqual(validate("", NT_ALPHABET), b"")


# ── mask ──────────────────────────────────────────────────────────

class TestMask(unittest.TestCase):

    def test_single_position(self) -> None:
        self.assertEqual(mask("CAST", 0, AA_MASK), b"XAST")
        self.assertEqual(mask("CAST", 3, AA_MASK), b"CASX")

    def test_negative_position(self) -> None:
        self.assertEqual(mask("CAST", -1, AA_MASK), b"CASX")

    def test_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            mask("CA", 5, AA_MASK)

    def test_slice_position(self) -> None:
        self.assertEqual(mask("CASTIV", slice(1, 3), AA_MASK), b"CXXTIV")

    def test_tuple_position(self) -> None:
        self.assertEqual(mask("CASTIV", (1, 3), AA_MASK), b"CXXTIV")

    def test_bad_position_type(self) -> None:
        with self.assertRaises(TypeError):
            mask("CAST", [0], AA_MASK)  # type: ignore[arg-type]


# ── matches ───────────────────────────────────────────────────────

class TestMatches(unittest.TestCase):

    def test_identical(self) -> None:
        self.assertTrue(matches("CAST", "CAST", AA_MASK))

    def test_wildcard_on_a(self) -> None:
        self.assertTrue(matches("XAST", "CAST", AA_MASK))

    def test_wildcard_on_b(self) -> None:
        self.assertTrue(matches("CAST", "XAST", AA_MASK))

    def test_mismatch(self) -> None:
        self.assertFalse(matches("CAST", "GAST", AA_MASK))

    def test_length_mismatch(self) -> None:
        self.assertFalse(matches("CAST", "CAS", AA_MASK))

    def test_empty(self) -> None:
        self.assertTrue(matches("", "", AA_MASK))

    def test_all_wildcards(self) -> None:
        self.assertTrue(matches("XXX", "CAS", AA_MASK))


# ── matches_aa_reduced ────────────────────────────────────────────

class TestMatchesAaReduced(unittest.TestCase):

    def test_matching_pair(self) -> None:
        aa = "CASTIVGGLSQDKIVW"
        reduced = aa_to_reduced(aa).decode()
        self.assertTrue(matches_aa_reduced(aa, reduced))

    def test_mismatch(self) -> None:
        self.assertFalse(matches_aa_reduced("C", "G"))

    def test_wildcard_aa_side(self) -> None:
        self.assertTrue(matches_aa_reduced("X", "s"))

    def test_wildcard_reduced_side(self) -> None:
        self.assertTrue(matches_aa_reduced("C", "X"))

    def test_length_mismatch(self) -> None:
        self.assertFalse(matches_aa_reduced("CA", "s"))

    def test_empty(self) -> None:
        self.assertTrue(matches_aa_reduced("", ""))


if __name__ == "__main__":
    unittest.main()
