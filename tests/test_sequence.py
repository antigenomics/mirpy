"""Unit tests for :mod:`mir.basic.sequence`.

Coverage:
    SequenceAlphabet         -- singleton caching.
    AlphabetSequence         -- construction, string round-trip, substring,
                                immutability, ``__eq__``, ``__hash__``.
    NucleotideSequence       -- parsing, slicing, masking.
    AminoAcidSequence        -- parsing, slicing, reduced conversion, matching.
    ReducedAminoAcidSequence -- parsing, slicing, masking, matching.
    Equality vs matching     -- ``matches()`` is wildcard-aware, ``==`` is not.
"""

import unittest

import numpy as np

from mir.basic.sequence import (
    AminoAcidSequence,
    NucleotideSequence,
    ReducedAminoAcidSequence,
    SequenceAlphabet,
)


class TestAlphabetSequence(unittest.TestCase):
    """Construction, round-trip, substring, immutability."""

    def test_create_convert_and_substring(self) -> None:
        self.assertIs(
            NucleotideSequence.DEFAULT_ALPHABET,
            SequenceAlphabet(("A", "T", "G", "C", "N")),
        )

        nt = NucleotideSequence.from_string("ATTAGACA")
        self.assertEqual(nt.to_string(), "ATTAGACA")
        self.assertEqual(nt.data.dtype, np.dtype("S1"))
        self.assertEqual(nt.data.tobytes(), b"ATTAGACA")
        self.assertEqual(nt.substring(2, 6).to_string(), "TAGA")

        aa = AminoAcidSequence.from_string("CASSLAPGATNEKLFF")
        self.assertEqual(aa.to_string(), "CASSLAPGATNEKLFF")
        self.assertEqual(aa.substring(4, 9).to_string(), "LAPGA")

    def test_empty_or_invalid_sequence(self) -> None:
        empty_nt = NucleotideSequence.from_string("")
        self.assertEqual(len(empty_nt), 0)
        self.assertEqual(empty_nt.to_string(), "")

        empty_aa = AminoAcidSequence.from_string("")
        self.assertEqual(len(empty_aa), 0)
        self.assertEqual(empty_aa.to_string(), "")

        self.assertEqual(
            NucleotideSequence.from_string("ATTAGACA").substring(0, 0).to_string(), ""
        )

        self.assertEqual(NucleotideSequence.from_string("ATN").to_string(), "ATN")

        with self.assertRaises(ValueError):
            NucleotideSequence.from_string("ATU")

        with self.assertRaises(ValueError):
            AminoAcidSequence.from_string("B")

    def test_immutability(self) -> None:
        """The underlying byte array is read-only."""
        nt = NucleotideSequence.from_string("ATCG")
        with self.assertRaises(ValueError):
            nt.data[0] = b"G"

    def test_no_extra_attributes(self) -> None:
        """__slots__ prevents adding arbitrary instance attributes."""
        nt = NucleotideSequence.from_string("ATCG")
        with self.assertRaises(AttributeError):
            nt.foo = 42  # type: ignore[attr-defined]

    def test_content_backward_compat(self) -> None:
        """The .content property still works."""
        nt = NucleotideSequence.from_string("ATCG")
        np.testing.assert_array_equal(nt.content, nt.data)

    def test_repr(self) -> None:
        nt = NucleotideSequence.from_string("ATCG")
        self.assertEqual(repr(nt), "NucleotideSequence('ATCG')")


class TestEqualityAndHashing(unittest.TestCase):
    """``__eq__`` and ``__hash__`` use raw bytes, not wildcard matching."""

    def test_equal_sequences(self) -> None:
        a = NucleotideSequence.from_string("ATCG")
        b = NucleotideSequence.from_string("ATCG")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_unequal_sequences(self) -> None:
        a = NucleotideSequence.from_string("ATCG")
        b = NucleotideSequence.from_string("ATNG")
        self.assertNotEqual(a, b)

    def test_masked_not_equal_but_matches(self) -> None:
        """A masked sequence matches the original but is not equal."""
        orig = NucleotideSequence.from_string("ATCG")
        masked = NucleotideSequence.from_string("ANNG")
        self.assertNotEqual(orig, masked)
        self.assertTrue(orig.matches(masked))

    def test_set_and_dict_storage(self) -> None:
        a = AminoAcidSequence.from_string("CAST")
        b = AminoAcidSequence.from_string("CAST")
        c = AminoAcidSequence.from_string("XAST")
        s = {a, b, c}
        self.assertEqual(len(s), 2)
        d = {a: 1}
        self.assertEqual(d[b], 1)
        self.assertNotIn(c, d)

    def test_cross_type_not_equal(self) -> None:
        """Different types with identical bytes are not equal."""
        aa = AminoAcidSequence.from_string("X")
        red = ReducedAminoAcidSequence.from_string("X")
        self.assertNotEqual(aa, red)


class TestReducedAminoAcidSequence(unittest.TestCase):
    """Reduced-alphabet conversion and matching."""

    def test_conversion_via_byte_lut(self) -> None:
        aa = AminoAcidSequence.from_string("CASTIVGGLSQDKIVW")
        reduced = aa.to_reduced_amino_acid()
        self.assertEqual(reduced.to_string(), "slhhllGGlhmcbllW")

    def test_match_and_mismatch(self) -> None:
        aa = AminoAcidSequence.from_string("CASTIVGGLSQDKIVW")
        reduced = aa.to_reduced_amino_acid()
        self.assertTrue(aa.matches_reduced_amino_acid(reduced))
        self.assertFalse(
            aa.matches_reduced_amino_acid(
                ReducedAminoAcidSequence.from_string("slhhllGGlhmcbllY")
            )
        )

    def test_masked_aa_matches_reduced(self) -> None:
        aa = AminoAcidSequence.from_string("CASTIVGGLSQDKIVW")
        reduced = aa.to_reduced_amino_acid()
        self.assertTrue(aa.mask(2).matches_reduced_amino_acid(reduced))

    def test_masked_reduced_matches_aa(self) -> None:
        aa = AminoAcidSequence.from_string("CASTIVGGLSQDKIVW")
        reduced = aa.to_reduced_amino_acid()
        self.assertTrue(aa.matches_reduced_amino_acid(reduced.mask((2, 5))))

    def test_backwards_compatible_aliases(self) -> None:
        aa = AminoAcidSequence.from_string("CAST")
        reduced = aa.to_simple_amino_acid()
        self.assertIsInstance(reduced, ReducedAminoAcidSequence)
        self.assertTrue(aa.matches_simple_amino_acid(reduced))

    def test_reduced_substrings(self) -> None:
        reduced = ReducedAminoAcidSequence.from_string("slhhllGGlhmcbllW")
        self.assertEqual(reduced.substring(0, 4).to_string(), "slhh")
        self.assertEqual(reduced.substring(6, 8).to_string(), "GG")
        self.assertEqual(reduced.substring(11, None).to_string(), "cbllW")

        with self.assertRaises(ValueError):
            ReducedAminoAcidSequence.from_string("Z")


class TestMaskAndMatch(unittest.TestCase):
    """Masking and wildcard-aware matching."""

    def test_nucleotide_mask_single_and_range(self) -> None:
        seq = NucleotideSequence.from_string("ATCGAT")
        self.assertEqual(seq.mask(1).to_string(), "ANCGAT")
        self.assertEqual(seq.mask((2, 5)).to_string(), "ATNNNT")
        self.assertEqual(seq.mask(slice(0, 3)).to_string(), "NNNGAT")

    def test_amino_and_reduced_mask(self) -> None:
        aa = AminoAcidSequence.from_string("CASTIV")
        reduced = ReducedAminoAcidSequence.from_string("slhhll")
        self.assertEqual(aa.mask(0).to_string(), "XASTIV")
        self.assertEqual(aa.mask((1, 4)).to_string(), "CXXXIV")
        self.assertEqual(reduced.mask(slice(2, 5)).to_string(), "slXXXl")

    def test_matching_ignores_mask_symbols(self) -> None:
        nt1 = NucleotideSequence.from_string("ATCG")
        nt2 = NucleotideSequence.from_string("ANNG")
        self.assertTrue(nt1.matches(nt2))
        self.assertFalse(nt1.matches(NucleotideSequence.from_string("ANNA")))

        aa1 = AminoAcidSequence.from_string("CAST")
        aa2 = AminoAcidSequence.from_string("XASX")
        self.assertTrue(aa1.matches(aa2))
        self.assertFalse(aa1.matches(AminoAcidSequence.from_string("XATX")))

        red1 = ReducedAminoAcidSequence.from_string("slhh")
        red2 = ReducedAminoAcidSequence.from_string("sXXh")
        self.assertTrue(red1.matches(red2))
        self.assertFalse(red1.matches(ReducedAminoAcidSequence.from_string("sXXY")))

    def test_length_mismatch_does_not_match(self) -> None:
        a = NucleotideSequence.from_string("ATC")
        b = NucleotideSequence.from_string("ATCG")
        self.assertFalse(a.matches(b))

    def test_empty_sequences_match(self) -> None:
        a = NucleotideSequence.from_string("")
        b = NucleotideSequence.from_string("")
        self.assertTrue(a.matches(b))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
