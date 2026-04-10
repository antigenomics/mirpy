"""Unit tests for :mod:`mir.basic.sequence`.

Coverage:
    SequenceAlphabet  -- singleton caching behaviour.
    AlphabetSequence  -- construction, round-trip string conversion,
                         substring slicing, length, and alphabet
                         rejection.
    NucleotideSequence  -- DNA string parsing and slicing.
    AminoAcidSequence   -- protein string parsing, slicing, and
                           conversion to the reduced alphabet.
    SimpleAminoAcidSequence -- reduced-alphabet string parsing and
                               slicing.
"""

import unittest

import numpy as np

from mir.basic.sequence import (
    AminoAcidSequence,
    NucleotideSequence,
    SequenceAlphabet,
    SimpleAminoAcidSequence,
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
        self.assertIs(NucleotideSequence.DEFAULT_ALPHABET, SequenceAlphabet(("A", "T", "G", "C")))

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

        with self.assertRaises(ValueError):
            NucleotideSequence.from_string("ATU")

        with self.assertRaises(ValueError):
            AminoAcidSequence.from_string("B")


class TestSimpleAminoAcidSequence(unittest.TestCase):
    """Tests for :class:`~mir.basic.sequence.SimpleAminoAcidSequence` and AA conversion."""

    def test_amino_acid_to_simple_conversion_and_match(self) -> None:
        """``to_simple_amino_acid`` applies the physico-chemical grouping map correctly.

        Verifies that:
        * The reduced string produced by :meth:`AminoAcidSequence.to_simple_amino_acid`
          matches the expected character-by-character mapping from
          :data:`AMINO_ACID_TO_SIMPLE_AMINO_ACID`.
        * :meth:`AminoAcidSequence.matches_simple_amino_acid` returns ``True``
          for the sequence's own reduced form and ``False`` for an altered one.
        """
        aa: AminoAcidSequence = AminoAcidSequence.from_string("CASTIVGGLSQDKIVW")
        simple = aa.to_simple_amino_acid()

        self.assertEqual(simple.to_string(), "slhhllGGlhmcbllW")
        self.assertTrue(aa.matches_simple_amino_acid(simple))
        self.assertFalse(aa.matches_simple_amino_acid(SimpleAminoAcidSequence.from_string("slhhllGGlhmcbllY")))

    def test_simple_substrings(self) -> None:
        """``substring`` on a :class:`SimpleAminoAcidSequence` slices correctly.

        Verifies that:
        * A half-open slice returns the expected subsequence.
        * ``substring(start, None)`` slices through to the end of the sequence.
        * An out-of-alphabet character (``Z``) raises ``ValueError``.
        """
        simple = SimpleAminoAcidSequence.from_string("slhhllGGlhmcbllW")
        self.assertEqual(simple.substring(0, 4).to_string(), "slhh")
        self.assertEqual(simple.substring(6, 8).to_string(), "GG")
        self.assertEqual(simple.substring(11, None).to_string(), "cbllW")

        with self.assertRaises(ValueError):
            SimpleAminoAcidSequence.from_string("Z")


if __name__ == "__main__":
    unittest.main()