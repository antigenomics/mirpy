"""Biological sequence types backed by NumPy byte arrays.

This module defines alphabet-validated sequence classes for nucleotide and
amino-acid data. All sequences are stored as ``np.ndarray`` of dtype ``S1``
(single-byte ASCII characters) so they can be operated on efficiently with
NumPy primitives and ``stringzilla``.

Classes:
    SequenceAlphabet         -- Singleton alphabet definition.
    AlphabetSequence         -- Base class for alphabet-constrained sequences.
    NucleotideSequence       -- DNA sequence (A/T/G/C/N by default).
    AminoAcidSequence        -- Standard 20-AA + stop/unknown sequence.
    ReducedAminoAcidSequence -- Reduced amino-acid alphabet used for fuzzy
                                 matching (groups physico-chemically similar AAs).
"""

from __future__ import annotations

import numpy as np
import stringzilla as sz 
from typing import Self


#: Maps each standard amino-acid one-letter code (plus ``X``, ``*``, ``_``)
#: to a reduced symbol representing its physico-chemical class::
#:
#:   l  aliphatic/hydrophobic  (A, I, L, V)
#:   b  basic                  (R, H, K)
#:   m  amide                  (N, Q)
#:   c  acidic/charged         (D, E)
#:   s  sulphur-containing     (C, M)
#:   h  hydroxyl               (S, T)
#:   G  glycine
#:   F  phenylalanine
#:   P  proline
#:   W  tryptophan
#:   Y  tyrosine
#:   X  unknown
#:   *  stop codon
#:   _  gap
AMINO_ACID_TO_REDUCED_AMINO_ACID: dict[str, str] = {
    "A": "l",
    "R": "b",
    "N": "m",
    "D": "c",
    "C": "s",
    "Q": "m",
    "E": "c",
    "G": "G",
    "H": "b",
    "I": "l",
    "L": "l",
    "K": "b",
    "M": "s",
    "F": "F",
    "P": "P",
    "S": "h",
    "T": "h",
    "W": "W",
    "Y": "Y",
    "V": "l",
    "X": "X",
    "*": "*",
    "_": "_",
}


class SequenceAlphabet:
    """Singleton-like immutable alphabet definition keyed by allowed symbols.

    Instances are cached by their ``allowed_symbols`` tuple so that two
    ``SequenceAlphabet`` objects constructed with identical symbol sets are
    guaranteed to be the *same* object (``is`` comparison holds).

    Attributes:
        allowed_symbols (tuple[str, ...]): Immutable ordered collection of
            single-character symbols that belong to this alphabet.
        allowed_array (np.ndarray): ``S1``-dtype NumPy array of the same
            symbols, pre-built for fast membership testing via ``np.isin``.
    """

    _instances: dict[tuple[str, ...], "SequenceAlphabet"] = {}

    def __new__(cls, allowed_symbols: tuple[str, ...]) -> "SequenceAlphabet":
        """Return the cached instance for *allowed_symbols*, creating it if needed."""
        key = tuple(allowed_symbols)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, allowed_symbols: tuple[str, ...]) -> None:
        """Initialise the alphabet (no-op when the cached instance already exists).

        Args:
            allowed_symbols: Ordered tuple of single-character strings that
                define the legal symbol set for this alphabet.
        """
        if hasattr(self, "allowed_symbols"):
            return
        self.allowed_symbols = tuple(allowed_symbols)
        self.allowed_array = np.array([c.encode("ascii") for c in self.allowed_symbols], dtype="S1")


#: Backwards-compatible alias for older name.
AMINO_ACID_TO_SIMPLE_AMINO_ACID = AMINO_ACID_TO_REDUCED_AMINO_ACID


#: Pre-built :class:`SequenceAlphabet` for the reduced amino-acid symbol set
#: derived from :data:`AMINO_ACID_TO_REDUCED_AMINO_ACID`.
REDUCED_AMINO_ACID_ALPHABET = SequenceAlphabet(tuple(dict.fromkeys(AMINO_ACID_TO_REDUCED_AMINO_ACID.values())))


#: Backwards-compatible alias for older name.
SIMPLE_AMINO_ACID_ALPHABET = REDUCED_AMINO_ACID_ALPHABET


class AlphabetSequence:
    """Compact alphabet-validated sequence backed by a NumPy array.

    This is the abstract base class for all concrete sequence types in this
    module.  Direct instantiation is allowed but callers should prefer one of
    the concrete subclasses (:class:`NucleotideSequence`,
    :class:`AminoAcidSequence`, :class:`SimpleAminoAcidSequence`) which
    provide a sensible default alphabet.

    Class Attributes:
        MASK_SYMBOL (str | None): Symbol used by :meth:`mask`. Subclasses
            that support masking define this symbol (``"N"`` for nucleotides,
            ``"X"`` for amino-acid alphabets).

    Attributes:
        content (np.ndarray): One-dimensional ``S1``-dtype array storing the
            sequence as individual ASCII bytes.
        alphabet (SequenceAlphabet): The alphabet that ``content`` was
            validated against.
    """

    MASK_SYMBOL: str | None = None

    def __init__(self, content: np.ndarray, alphabet: SequenceAlphabet) -> None:
        """Construct a validated sequence from an existing byte array.

        Args:
            content:  One-dimensional NumPy array with ``dtype='S1'``.
            alphabet: Alphabet to validate *content* against.

        Raises:
            ValueError: If *content* does not have dtype ``S1``, is not
                one-dimensional, or contains symbols absent from *alphabet*.
        """
        if content.dtype != np.dtype("S1"):
            raise ValueError("Sequence storage must have dtype S1")
        if content.ndim != 1:
            raise ValueError("Sequence storage must be one-dimensional")

        if not np.isin(content, alphabet.allowed_array).all():
            raise ValueError("Sequence contains symbols outside of alphabet")

        self.content = content
        self.alphabet = alphabet

    @classmethod
    def from_string(cls: type[Self], sequence: str, alphabet: SequenceAlphabet) -> Self:
        """Create an instance by parsing a plain Python string.

        Args:
            sequence: String whose characters must all belong to *alphabet*.
            alphabet: Alphabet to validate the sequence against.

        Returns:
            A new instance of the calling class backed by a freshly allocated
            ``S1`` NumPy array.

        Raises:
            ValueError: If any character in *sequence* is outside *alphabet*.
        """
        sz_sequence = sz.Str(sequence)
        sequence_bytes = bytes(sz_sequence)
        array = np.frombuffer(memoryview(sequence_bytes), dtype="S1").copy()
        return cls(array, alphabet)

    def to_string(self) -> str:
        """Decode the byte array back to a plain Python string.

        Returns:
            The sequence as a ``str``.
        """
        return str(sz.Str(self.content.tobytes()))

    def substring(self, start: int, stop: int | None = None) -> "AlphabetSequence":
        """Return a validated slice of this sequence.

        Uses the same slicing semantics as Python built-in strings: *start* is
        inclusive, *stop* is exclusive, and ``None`` means "to the end".

        Args:
            start: Index of the first character to include (0-based).
            stop:  Index of the first character to *exclude*, or ``None`` to
                   slice through to the end of the sequence.

        Returns:
            A new instance of the same concrete class containing the requested
            subsequence with the same alphabet.
        """
        view = sz.Str(self.content.tobytes())
        part = view[start:stop]
        part_bytes = bytes(part)
        sub_array = np.frombuffer(memoryview(part_bytes), dtype="S1").copy()
        return self.__class__(sub_array, self.alphabet)

    def mask(self, position: int | slice | tuple[int, int]) -> Self:
        """Return a copy with one position or a range replaced by mask symbol.

        Args:
            position: Either a single integer index, a :class:`slice`, or a
                ``(start, stop)`` tuple. ``stop`` is exclusive.

        Returns:
            A new sequence with masked positions replaced by ``MASK_SYMBOL``.

        Raises:
            ValueError: If this sequence type does not define ``MASK_SYMBOL``.
            TypeError: If *position* has unsupported type.
            IndexError: If integer position is out of bounds.
        """
        if self.MASK_SYMBOL is None:
            raise ValueError(f"Masking is not supported for {self.__class__.__name__}")

        masked = self.content.copy()
        mask_byte = np.array(self.MASK_SYMBOL.encode("ascii"), dtype="S1")

        if isinstance(position, int):
            if position < 0:
                position += len(self)
            if position < 0 or position >= len(self):
                raise IndexError("Mask position out of range")
            masked[position] = mask_byte
        elif isinstance(position, slice):
            masked[position] = mask_byte
        elif isinstance(position, tuple) and len(position) == 2:
            start, stop = position
            masked[slice(start, stop)] = mask_byte
        else:
            raise TypeError("position must be int, slice, or (start, stop) tuple")

        return self.__class__(masked, self.alphabet)

    def _is_masked(self, symbol: np.bytes_) -> bool:
        """Return whether a symbol is the sequence-specific mask marker."""
        if self.MASK_SYMBOL is None:
            return False
        return symbol == self.MASK_SYMBOL.encode("ascii")

    def matches(self, other: "AlphabetSequence") -> bool:
        """Compare two sequences, treating mask characters as wildcards.

        Matching requires equal lengths. At each position, symbols match if
        they are equal or if either symbol is masked.
        """
        if len(self) != len(other):
            return False

        for left, right in zip(self.content, other.content):
            if left == right:
                continue
            if self._is_masked(left) or other._is_masked(right):
                continue
            return False
        return True

    def __len__(self) -> int:
        """Return the number of characters in the sequence."""
        return int(self.content.shape[0])
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the sequence."""
        return self.to_string()


class NucleotideSequence(AlphabetSequence):
    """A DNA nucleotide sequence restricted to the standard four-base alphabet.

    The default alphabet is ``("A", "T", "G", "C")``.  A custom
    :class:`SequenceAlphabet` may be supplied to support, for example,
    ambiguity codes.

    Class Attributes:
        DEFAULT_ALPHABET (SequenceAlphabet): Standard DNA alphabet ``{A, T, G, C, N}``.
    """

    MASK_SYMBOL = "N"
    DEFAULT_ALPHABET = SequenceAlphabet(("A", "T", "G", "C", "N"))

    def __init__(self, content: np.ndarray, alphabet: SequenceAlphabet = DEFAULT_ALPHABET) -> None:
        """Construct a nucleotide sequence from a byte array.

        Args:
            content:  One-dimensional ``S1``-dtype NumPy array.
            alphabet: Alphabet to validate against (defaults to ``{A, T, G, C}``).
        """
        super().__init__(content, alphabet)

    @classmethod
    def from_string(
        cls: type[Self],
        sequence: str,
        alphabet: SequenceAlphabet = DEFAULT_ALPHABET,
    ) -> Self:
        """Create a :class:`NucleotideSequence` from a plain string.

        Args:
            sequence: DNA string (e.g. ``"ATCG"``).  All characters must
                belong to *alphabet*.
            alphabet: Defaults to the standard four-base DNA alphabet.

        Returns:
            A new :class:`NucleotideSequence` instance.

        Raises:
            ValueError: If *sequence* contains characters outside *alphabet*.
        """
        sz_sequence = sz.Str(sequence)
        sequence_bytes = bytes(sz_sequence)
        array = np.frombuffer(memoryview(sequence_bytes), dtype="S1").copy()
        return cls(array, alphabet)


class AminoAcidSequence(AlphabetSequence):
    """A standard amino-acid sequence using the 20-letter IUPAC alphabet.

    In addition to the 20 canonical amino acids the alphabet includes:

    * ``*`` — stop codon
    * ``_`` — gap
    * ``X`` — unknown / any amino acid

    Class Attributes:
        DEFAULT_ALPHABET (SequenceAlphabet): The 20 canonical AAs plus
            ``*``, ``_``, and ``X``.
    """

    DEFAULT_ALPHABET = SequenceAlphabet(
        (
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
            "*", "_", "X",
        )
    )
    MASK_SYMBOL = "X"

    def __init__(self, content: np.ndarray, alphabet: SequenceAlphabet = DEFAULT_ALPHABET) -> None:
        """Construct an amino-acid sequence from a byte array.

        Args:
            content:  One-dimensional ``S1``-dtype NumPy array.
            alphabet: Alphabet to validate against (defaults to the standard
                20-AA + stop/gap/unknown alphabet).
        """
        super().__init__(content, alphabet)

    @classmethod
    def from_string(
        cls: type[Self],
        sequence: str,
        alphabet: SequenceAlphabet = DEFAULT_ALPHABET,
    ) -> Self:
        """Create an :class:`AminoAcidSequence` from a plain string.

        Args:
            sequence: Amino-acid string in single-letter code
                (e.g. ``"CASSLAPGATNEKLFF"``).  All characters must belong to
                *alphabet*.
            alphabet: Defaults to the standard amino-acid alphabet.

        Returns:
            A new :class:`AminoAcidSequence` instance.

        Raises:
            ValueError: If *sequence* contains characters outside *alphabet*.
        """
        sz_sequence = sz.Str(sequence)
        sequence_bytes = bytes(sz_sequence)
        array = np.frombuffer(memoryview(sequence_bytes), dtype="S1").copy()
        return cls(array, alphabet)

    def to_reduced_amino_acid(self) -> "ReducedAminoAcidSequence":
        """Convert to :class:`ReducedAminoAcidSequence` using physico-chemical grouping.

        Each amino acid is mapped to a reduced symbol according to
        :data:`AMINO_ACID_TO_SIMPLE_AMINO_ACID`.

        Returns:
            A :class:`ReducedAminoAcidSequence` of the same length whose
            symbols represent physico-chemical classes.
        """
        converted = "".join(AMINO_ACID_TO_REDUCED_AMINO_ACID[s] for s in self.to_string())
        return ReducedAminoAcidSequence.from_string(converted)

    def to_simple_amino_acid(self) -> "ReducedAminoAcidSequence":
        """Backwards-compatible alias for :meth:`to_reduced_amino_acid`."""
        return self.to_reduced_amino_acid()

    def matches_reduced_amino_acid(self, reduced_sequence: "ReducedAminoAcidSequence") -> bool:
        """Check whether this sequence matches a reduced amino-acid sequence.

        Matching is performed position-wise and ignores masked characters
        (``X`` in either sequence).

        Args:
            reduced_sequence: A :class:`ReducedAminoAcidSequence` to compare
                against.

        Returns:
            ``True`` if all non-masked positions are compatible with the
            reduced amino-acid mapping, ``False`` otherwise.
        """
        if len(self) != len(reduced_sequence):
            return False

        for aa_symbol, reduced_symbol in zip(self.content, reduced_sequence.content):
            if self._is_masked(aa_symbol) or reduced_sequence._is_masked(reduced_symbol):
                continue
            mapped = AMINO_ACID_TO_REDUCED_AMINO_ACID[aa_symbol.decode("ascii")].encode("ascii")
            if mapped != reduced_symbol:
                return False
        return True

    def matches_simple_amino_acid(self, simple_sequence: "ReducedAminoAcidSequence") -> bool:
        """Backwards-compatible alias for :meth:`matches_reduced_amino_acid`."""
        return self.matches_reduced_amino_acid(simple_sequence)


class ReducedAminoAcidSequence(AlphabetSequence):
    """A sequence encoded in the reduced physico-chemical amino-acid alphabet.

    Symbols are those produced by :data:`AMINO_ACID_TO_SIMPLE_AMINO_ACID`:
    ``l``, ``b``, ``m``, ``c``, ``s``, ``h``, ``G``, ``F``, ``P``, ``W``,
    ``Y``, ``X``, ``*``, ``_``.

    Instances are typically obtained via
    :meth:`AminoAcidSequence.to_reduced_amino_acid` rather than constructed
    directly.

    Class Attributes:
        DEFAULT_ALPHABET (SequenceAlphabet): :data:`REDUCED_AMINO_ACID_ALPHABET`.
    """

    DEFAULT_ALPHABET = REDUCED_AMINO_ACID_ALPHABET
    MASK_SYMBOL = "X"

    def __init__(self, content: np.ndarray, alphabet: SequenceAlphabet = DEFAULT_ALPHABET) -> None:
        """Construct a reduced amino-acid sequence from a byte array.

        Args:
            content:  One-dimensional ``S1``-dtype NumPy array.
            alphabet: Alphabet to validate against (defaults to
                :data:`REDUCED_AMINO_ACID_ALPHABET`).
        """
        super().__init__(content, alphabet)

    @classmethod
    def from_string(
        cls: type[Self],
        sequence: str,
        alphabet: SequenceAlphabet = DEFAULT_ALPHABET,
    ) -> Self:
        """Create a :class:`ReducedAminoAcidSequence` from a plain string.

        Args:
            sequence: String using the reduced physico-chemical symbols
                (e.g. ``"slhhllGGlhmcbllW"``).  All characters must belong
                to *alphabet*.
            alphabet: Defaults to :data:`REDUCED_AMINO_ACID_ALPHABET`.

        Returns:
            A new :class:`ReducedAminoAcidSequence` instance.

        Raises:
            ValueError: If *sequence* contains characters outside *alphabet*.
        """
        sz_sequence = sz.Str(sequence)
        sequence_bytes = bytes(sz_sequence)
        array = np.frombuffer(memoryview(sequence_bytes), dtype="S1").copy()
        return cls(array, alphabet)



#: Backwards-compatible class alias.
SimpleAminoAcidSequence = ReducedAminoAcidSequence
