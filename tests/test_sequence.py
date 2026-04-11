"""Unit tests for :mod:`mir.basic.sequence` functions.

Coverage:
    make_alphabet / validate  — alphabet construction and validation.
    aa_to_reduced / translate — byte-level translation.
    mask                      — single-index, range, and slice masking.
    matches                   — wildcard-aware comparison.
    matches_aa_reduced        — cross-alphabet wildcard match.
    str / bytes duality       — every function accepts both types.
"""

import unittest

from mir.basic.sequence import (
    AA_ALPHABET,
    AA_MASK,
    AA_TO_REDUCED_TABLE,
    NT_ALPHABET,
    NT_MASK,
    REDUCED_AA_ALPHABET,
    REDUCED_AA_MASK,
    aa_to_reduced,
    make_alphabet,
    mask,
    matches,
    matches_aa_reduced,
    translate,
    validate,
)


class TestMakeAlphabet(unittest.TestCase):

    def test_custom_alphabet(self) -> None:
        lut = make_alphabet("AB")
        self.assertEqual(len(lut), 256)
        self.assertEqual(lut[ord("A")], 1)
        self.assertEqual(lut[ord("B")], 1)
        self.assertEqual(lut[ord("C")], 0)

    def test_predefined_nt(self) -> None:
        for ch in "ATGCN":
            self.assertEqual(NT_ALPHABET[ord(ch)], 1)
        self.assertEqual(NT_ALPHABET[ord("U")], 0)

    def test_predefined_aa(self) -> None:
        for ch in "ACDEFGHIKLMNPQRSTVWYX*_":
            self.assertEqual(AA_ALPHABET[ord(ch)], 1)
        self.assertEqual(AA_ALPHABET[ord("B")], 0)

    def test_predefined_reduced(self) -> None:
        for ch in "lbmcshGFPWYX*_":
            self.assertEqual(REDUCED_AA_ALPHABET[ord(ch)], 1)
        self.assertEqual(REDUCED_AA_ALPHABET[ord("Z")], 0)


class TestValidate(unittest.TestCase):

    def test_valid_nt_str(self) -> None:
        self.assertEqual(validate("ATTAGACA", NT_ALPHABET), b"ATTAGACA")

    def test_valid_nt_bytes(self) -> None:
        self.assertEqual(validate(b"ATN", NT_ALPHABET), b"ATN")

    def test_valid_aa_bytearray(self) -> None:
        self.assertEqual(validate(bytearray(b"CAST"), AA_ALPHABET), b"CAST")

    def test_empty(self) -> None:
        self.assertEqual(validate("", NT_ALPHABET), b"")
        self.assertEqual(validate(b"", AA_ALPHABET), b"")

    def test_invalid_nt(self) -> None:
        with self.assertRaises(ValueError):
            validate("ATU", NT_ALPHABET)

    def test_invalid_aa(self) -> None:
        with self.assertRaises(ValueError):
            validate("B", AA_ALPHABET)

    def test_invalid_reduced(self) -> None:
        with self.assertRaises(ValueError):
            validate("Z", REDUCED_AA_ALPHABET)


class TestTranslateAndReduce(unittest.TestCase):

    def test_aa_to_reduced_str(self) -> None:
        self.assertEqual(aa_to_reduced("CASTIVGGLSQDKIVW"), b"slhhllGGlhmcbllW")

    def test_aa_to_reduced_bytes(self) -> None:
        self.assertEqual(aa_to_reduced(b"CASTIVGGLSQDKIVW"), b"slhhllGGlhmcbllW")

    def test_generic_translate(self) -> None:
        self.assertEqual(translate("CAST", AA_TO_REDUCED_TABLE), b"slhh")

    def test_empty_translate(self) -> None:
        self.assertEqual(aa_to_reduced(""), b"")


class TestMask(unittest.TestCase):

    def test_single_nt(self) -> None:
        self.assertEqual(mask("ATCGAT", 1, NT_MASK), b"ANCGAT")

    def test_range_nt(self) -> None:
        self.assertEqual(mask("ATCGAT", (2, 5), NT_MASK), b"ATNNNT")

    def test_slice_nt(self) -> None:
        self.assertEqual(mask("ATCGAT", slice(0, 3), NT_MASK), b"NNNGAT")

    def test_aa_single(self) -> None:
        self.assertEqual(mask("CASTIV", 0, AA_MASK), b"XASTIV")

    def test_aa_range(self) -> None:
        self.assertEqual(mask("CASTIV", (1, 4), AA_MASK), b"CXXXIV")

    def test_reduced_slice(self) -> None:
        self.assertEqual(mask("slhhll", slice(2, 5), REDUCED_AA_MASK), b"slXXXl")

    def test_bytes_input(self) -> None:
        self.assertEqual(mask(b"ATCG", 0, NT_MASK), b"NTCG")

    def test_out_of_range(self) -> None:
        with self.assertRaises(IndexError):
            mask("AT", 5, NT_MASK)


class TestMatches(unittest.TestCase):

    def test_identical(self) -> None:
        self.assertTrue(matches("ATCG", "ATCG", NT_MASK))

    def test_wildcard_match(self) -> None:
        self.assertTrue(matches("ATCG", "ANNG", NT_MASK))

    def test_no_match(self) -> None:
        self.assertFalse(matches("ATCG", "ANNA", NT_MASK))

    def test_length_mismatch(self) -> None:
        self.assertFalse(matches("ATC", "ATCG", NT_MASK))

    def test_empty(self) -> None:
        self.assertTrue(matches("", "", NT_MASK))

    def test_aa_wildcard(self) -> None:
        self.assertTrue(matches("CAST", "XASX", AA_MASK))
        self.assertFalse(matches("CAST", "XATX", AA_MASK))

    def test_reduced_wildcard(self) -> None:
        self.assertTrue(matches("slhh", "sXXh", REDUCED_AA_MASK))
        self.assertFalse(matches("slhh", "sXXY", REDUCED_AA_MASK))

    def test_bytes_input(self) -> None:
        self.assertTrue(matches(b"ATCG", b"ANNG", NT_MASK))

    def test_mixed_str_bytes(self) -> None:
        self.assertTrue(matches("ATCG", b"ANNG", NT_MASK))


class TestMatchesAaReduced(unittest.TestCase):

    def test_match(self) -> None:
        reduced = aa_to_reduced("CASTIVGGLSQDKIVW")
        self.assertTrue(matches_aa_reduced("CASTIVGGLSQDKIVW", reduced))

    def test_mismatch(self) -> None:
        self.assertFalse(matches_aa_reduced("CASTIVGGLSQDKIVW", b"slhhllGGlhmcbllY"))

    def test_masked_aa(self) -> None:
        reduced = aa_to_reduced("CASTIVGGLSQDKIVW")
        masked_aa = mask("CASTIVGGLSQDKIVW", 2, AA_MASK)
        self.assertTrue(matches_aa_reduced(masked_aa, reduced))

    def test_masked_reduced(self) -> None:
        reduced = aa_to_reduced("CASTIVGGLSQDKIVW")
        masked_red = mask(reduced, (2, 5), REDUCED_AA_MASK)
        self.assertTrue(matches_aa_reduced("CASTIVGGLSQDKIVW", masked_red))

    def test_empty(self) -> None:
        self.assertTrue(matches_aa_reduced("", ""))

    def test_length_mismatch(self) -> None:
        self.assertFalse(matches_aa_reduced("CAS", "sl"))

    def test_bytes_input(self) -> None:
        reduced = aa_to_reduced(b"CAST")
        self.assertTrue(matches_aa_reduced(b"CAST", reduced))


if __name__ == "__main__":
    unittest.main()
