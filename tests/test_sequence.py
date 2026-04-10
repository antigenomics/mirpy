"""Unit tests for :mod:`mir.basic.sequence`.

Coverage:
    SequenceAlphabet  -- singleton caching behaviour.
    AlphabetSequence  -- construction, round-trip string conversion,
                         substring slicing, length, and alphabet
                         rejection.
    NucleotideSequence  -- DNA string parsing and slicing.
    AminoAcidSequence     -- protein string parsing, slicing, reduction,
                 and mask-aware matching.
    ReducedAminoAcidSequence -- reduced-alphabet parsing, slicing,
                  masking, and matching.
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
    """Tests for :class:`~mir.basic.sequence.AlphabetSequence` and its subclasses."""

    def test_create_convert_and_substring(self) -> None:
        """Sequences round-trip through ``from_string`` / ``to_string`` and slice correctly.

        Verifies that:
        * ``DEFAULT_ALPHABET`` is the singleton cached by :class:`SequenceAlphabet`.
        * The ``S1`` dtype is preserved after construction.
        * ``to_string`` reconstructs the original string exactly.
        * ``substring(start, stop)`` returns the expected subsequence for both
          :class:`NucleotideSequence` and :class:`AminoAcidSequence`.
        """
        self.assertIs(NucleotideSequence.DEFAULT_ALPHABET, SequenceAlphabet(("A", "T", "G", "C", "N")))

        nt = NucleotideSequence.from_string("ATTAGACA")
        self.assertEqual(nt.to_string(), "ATTAGACA")
        self.assertEqual(nt.content.dtype, np.dtype("S1"))
        self.assertEqual(nt.content.tobytes(), b"ATTAGACA")
        self.assertEqual(nt.substring(2, 6).to_string(), "TAGA")

        aa = AminoAcidSequence.from_string("CASSLAPGATNEKLFF")
        self.assertEqual(aa.to_string(), "CASSLAPGATNEKLFF")
        self.assertEqual(aa.substring(4, 9).to_string(), "LAPGA")

    def test_empty_or_invalid_sequence(self) -> None:
        """Empty sequences are valid; out-of-alphabet characters raise ``ValueError``.

        Verifies that:
        * An empty :class:`NucleotideSequence` and :class:`AminoAcidSequence`
          have length 0 and round-trip to ``""``.
        * ``substring(0, 0)`` on a non-empty sequence returns an empty sequence.
        * Constructing a :class:`NucleotideSequence` from ``"ATU"`` raises
          ``ValueError`` (``U`` is not in the DNA alphabet).
        * Constructing an :class:`AminoAcidSequence` from ``"B"`` raises
          ``ValueError`` (``B`` is not a standard amino acid).
        """
        empty_nt = NucleotideSequence.from_string("")
        self.assertEqual(len(empty_nt), 0)
        self.assertEqual(empty_nt.to_string(), "")

        empty_aa = AminoAcidSequence.from_string("")
        self.assertEqual(len(empty_aa), 0)
        self.assertEqual(empty_aa.to_string(), "")

        self.assertEqual(NucleotideSequence.from_string("ATTAGACA").substring(0, 0).to_string(), "")

        self.assertEqual(NucleotideSequence.from_string("ATN").to_string(), "ATN")

        with self.assertRaises(ValueError):
          NucleotideSequence.from_string("ATU")

        with self.assertRaises(ValueError):
            AminoAcidSequence.from_string("B")


class TestReducedAminoAcidSequence(unittest.TestCase):
    """Tests for :class:`~mir.basic.sequence.ReducedAminoAcidSequence` and AA conversion."""

    def test_amino_acid_to_reduced_conversion_and_match(self) -> None:
        """Reduced conversion and matching respect mapping and masks.

        Verifies that:
        * The reduced string produced by :meth:`AminoAcidSequence.to_reduced_amino_acid`
          matches the expected character-by-character mapping from
          :data:`AMINO_ACID_TO_REDUCED_AMINO_ACID`.
        * :meth:`AminoAcidSequence.matches_reduced_amino_acid` returns ``True``
          for the sequence's own reduced form and ``False`` for an altered one
          at an unmasked position.
        """
        aa: AminoAcidSequence = AminoAcidSequence.from_string("CASTIVGGLSQDKIVW")
        reduced = aa.to_reduced_amino_acid()

        self.assertEqual(reduced.to_string(), "slhhllGGlhmcbllW")
        self.assertTrue(aa.matches_reduced_amino_acid(reduced))
        self.assertFalse(aa.matches_reduced_amino_acid(ReducedAminoAcidSequence.from_string("slhhllGGlhmcbllY")))

        masked_aa = aa.mask(2)
        self.assertTrue(masked_aa.matches_reduced_amino_acid(reduced))

        masked_reduced = reduced.mask((2, 5))
        self.assertTrue(aa.matches_reduced_amino_acid(masked_reduced))

    def test_aa_to_reduced_backwards_compatible_aliases(self) -> None:
        """Legacy simple-amino-acid aliases keep working."""
        aa = AminoAcidSequence.from_string("CAST")
        reduced = aa.to_simple_amino_acid()
        self.assertIsInstance(reduced, ReducedAminoAcidSequence)
        self.assertTrue(aa.matches_simple_amino_acid(reduced))

    def test_reduced_substrings(self) -> None:
        """``substring`` on a :class:`ReducedAminoAcidSequence` slices correctly.

        Verifies that:
        * A half-open slice returns the expected subsequence.
        * ``substring(start, None)`` slices through to the end of the sequence.
        * An out-of-alphabet character (``Z``) raises ``ValueError``.
        """
        reduced = ReducedAminoAcidSequence.from_string("slhhllGGlhmcbllW")
        self.assertEqual(reduced.substring(0, 4).to_string(), "slhh")
        self.assertEqual(reduced.substring(6, 8).to_string(), "GG")
        self.assertEqual(reduced.substring(11, None).to_string(), "cbllW")

        with self.assertRaises(ValueError):
            ReducedAminoAcidSequence.from_string("Z")


class TestMaskAndMatch(unittest.TestCase):
    """Tests for masking and wildcard-aware matching."""

    def test_nucleotide_mask_single_and_range(self) -> None:
        seq = NucleotideSequence.from_string("ATCGAT")
        self.assertEqual(seq.mask(1).to_string(), "ANCGAT")
        self.assertEqual(seq.mask((2, 5)).to_string(), "ATNNNT")
        self.assertEqual(seq.mask(slice(0, 3)).to_string(), "NNNGAT")

    def test_amino_and_reduced_mask_single_and_range(self) -> None:
        aa = AminoAcidSequence.from_string("CASTIV")
        reduced = ReducedAminoAcidSequence.from_string("slhhll")
        self.assertEqual(aa.mask(0).to_string(), "XASTIV")
        self.assertEqual(aa.mask((1, 4)).to_string(), "CXXXIV")
        self.assertEqual(reduced.mask(slice(2, 5)).to_string(), "slXXXl")

    def test_sequence_matching_ignores_mask_symbols(self) -> None:
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


if __name__ == "__main__":
    unittest.main()