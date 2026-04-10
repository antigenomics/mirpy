"""Biological sequence types backed by immutable NumPy byte arrays.

Sequence objects are lightweight and immutable: each instance stores only a
read-only ``np.ndarray`` of dtype ``S1``.  The alphabet is validated on
construction and is defined at the class level (``DEFAULT_ALPHABET``).

**Equality vs. matching:**  ``__eq__`` and ``__hash__`` compare raw byte
content, so sequences work correctly as dictionary keys and set members.
The :meth:`~AlphabetSequence.matches` method is different: it treats mask
characters (``N`` for nucleotides, ``X`` for amino-acid alphabets) as
wildcards, so two sequences that *match* may not be *equal*.

Classes:
    SequenceAlphabet         -- Singleton alphabet definition.
    AlphabetSequence         -- Base class for alphabet-constrained sequences.
    NucleotideSequence       -- DNA sequence (A/T/G/C/N).
    AminoAcidSequence        -- Standard 20-AA + stop/unknown sequence.
    ReducedAminoAcidSequence -- Reduced amino-acid alphabet for fuzzy matching.
"""

from __future__ import annotations

import numpy as np
from typing import Self


# ---------------------------------------------------------------------------
# Amino-acid → reduced-alphabet mapping
# ---------------------------------------------------------------------------

#: Maps each standard amino-acid one-letter code (plus ``X``, ``*``, ``_``)
#: to a reduced symbol representing its physico-chemical class.
AMINO_ACID_TO_REDUCED_AMINO_ACID: dict[str, str] = {
    "A": "l", "R": "b", "N": "m", "D": "c", "C": "s", "Q": "m",
    "E": "c", "G": "G", "H": "b", "I": "l", "L": "l", "K": "b",
    "M": "s", "F": "F", "P": "P", "S": "h", "T": "h", "W": "W",
    "Y": "Y", "V": "l", "X": "X", "*": "*", "_": "_",
}

#: Byte lookup table (128 entries, indexed by ASCII ordinal) for converting
#: amino-acid bytes to reduced-alphabet bytes without string intermediaries.
_AA_TO_REDUCED_LUT = np.zeros(128, dtype=np.uint8)
for _aa, _red in AMINO_ACID_TO_REDUCED_AMINO_ACID.items():
    _AA_TO_REDUCED_LUT[ord(_aa)] = ord(_red)


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

class SequenceAlphabet:
    """Singleton immutable alphabet keyed by allowed symbols.

    Instances are cached so that two objects with identical symbol sets are
    the *same* object (``is`` holds).

    Attributes:
        allowed_symbols: Ordered tuple of single-character symbols.
        allowed_array: ``S1``-dtype NumPy array for fast ``np.isin`` tests.
    """

    _instances: dict[tuple[str, ...], SequenceAlphabet] = {}

    def __new__(cls, allowed_symbols: tuple[str, ...]) -> SequenceAlphabet:
        key = tuple(allowed_symbols)
        if key not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[key] = inst
        return cls._instances[key]

    def __init__(self, allowed_symbols: tuple[str, ...]) -> None:
        if hasattr(self, "allowed_symbols"):
            return
        self.allowed_symbols = tuple(allowed_symbols)
        self.allowed_array = np.array(
            [c.encode("ascii") for c in self.allowed_symbols], dtype="S1",
        )


REDUCED_AMINO_ACID_ALPHABET = SequenceAlphabet(
    tuple(dict.fromkeys(AMINO_ACID_TO_REDUCED_AMINO_ACID.values()))
)


# ---------------------------------------------------------------------------
# Base sequence
# ---------------------------------------------------------------------------

class AlphabetSequence:
    """Immutable alphabet-validated sequence backed by a read-only byte array.

    Each instance stores **only** a read-only ``np.ndarray`` of dtype ``S1``.
    The alphabet and mask symbol live on the class, not on instances.

    **Equality vs matching:**
    ``__eq__`` / ``__hash__`` compare raw bytes (for ``dict`` / ``set`` use).
    :meth:`matches` performs wildcard-aware comparison where mask characters
    (``N`` for nucleotides, ``X`` for amino-acid types) count as matching
    any symbol — so two sequences can *match* without being *equal*.

    Subclass protocol:
        * ``DEFAULT_ALPHABET`` — :class:`SequenceAlphabet` with allowed symbols.
        * ``_MASK_BYTE``       — ``b"N"``, ``b"X"``, or ``b""`` (no masking).
    """

    __slots__ = ("_data",)

    DEFAULT_ALPHABET: SequenceAlphabet  # set by subclasses
    _MASK_BYTE: bytes = b""

    def __init__(self, data: np.ndarray) -> None:
        """Validate *data* against the class alphabet and freeze it.

        Raises:
            ValueError: If dtype is not ``S1``, array is not 1-D, or
                symbols fall outside ``DEFAULT_ALPHABET``.
        """
        if data.dtype != np.dtype("S1"):
            raise ValueError("Sequence storage must have dtype S1")
        if data.ndim != 1:
            raise ValueError("Sequence storage must be one-dimensional")
        if not np.isin(data, self.DEFAULT_ALPHABET.allowed_array).all():
            raise ValueError("Sequence contains symbols outside of alphabet")
        data.flags.writeable = False
        self._data = data

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_string(cls: type[Self], sequence: str) -> Self:
        """Create a sequence from a plain Python string."""
        arr = np.frombuffer(sequence.encode("ascii"), dtype="S1").copy()
        return cls(arr)

    # -- accessors ----------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Read-only ``S1`` byte array backing this sequence."""
        return self._data

    @property
    def content(self) -> np.ndarray:
        """Alias for :attr:`data` (backward compatibility)."""
        return self._data

    def to_string(self) -> str:
        """Decode the byte array to a plain Python string."""
        return self._data.tobytes().decode("ascii")

    def substring(self, start: int, stop: int | None = None) -> Self:
        """Return a new sequence for the half-open range ``[start, stop)``."""
        return type(self)(self._data[start:stop].copy())

    # -- masking ------------------------------------------------------------

    def mask(self, position: int | slice | tuple[int, int]) -> Self:
        """Return a copy with the given position(s) replaced by the mask byte.

        Args:
            position: Integer index, ``slice``, or ``(start, stop)`` tuple.

        Raises:
            ValueError: If this class does not support masking.
            IndexError: If an integer position is out of bounds.
        """
        if not self._MASK_BYTE:
            raise ValueError(f"Masking not supported for {type(self).__name__}")
        buf = self._data.copy()
        buf.flags.writeable = True
        mv = np.array(self._MASK_BYTE, dtype="S1")
        if isinstance(position, int):
            if position < 0:
                position += len(self)
            if position < 0 or position >= len(self):
                raise IndexError("Mask position out of range")
            buf[position] = mv
        elif isinstance(position, slice):
            buf[position] = mv
        elif isinstance(position, tuple) and len(position) == 2:
            buf[position[0]:position[1]] = mv
        else:
            raise TypeError("position must be int, slice, or (start, stop) tuple")
        return type(self)(buf)

    # -- wildcard matching (NOT equality) -----------------------------------

    def matches(self, other: AlphabetSequence) -> bool:
        """Wildcard-aware positional comparison.

        Returns ``True`` when the sequences have the same length and at
        every position the symbols are equal **or** at least one side
        carries a mask character.  This is intentionally **not** the same
        as ``__eq__`` which compares bytes exactly.
        """
        if len(self) != len(other):
            return False
        if len(self) == 0:
            return True
        eq = self._data == other._data
        if eq.all():
            return True
        ok = eq.copy()
        if self._MASK_BYTE:
            ok |= self._data == np.array(self._MASK_BYTE, dtype="S1")
        if other._MASK_BYTE:
            ok |= other._data == np.array(other._MASK_BYTE, dtype="S1")
        return bool(ok.all())

    # -- equality & hashing (byte-exact) ------------------------------------

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._data.tobytes() == other._data.tobytes()

    def __hash__(self) -> int:
        return hash(self._data.tobytes())

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_string()!r})"


# ---------------------------------------------------------------------------
# Concrete sequence types
# ---------------------------------------------------------------------------

class NucleotideSequence(AlphabetSequence):
    """DNA nucleotide sequence (``A``, ``T``, ``G``, ``C``, ``N``).

    ``N`` serves as the mask / ambiguity symbol.
    """

    __slots__ = ()
    DEFAULT_ALPHABET = SequenceAlphabet(("A", "T", "G", "C", "N"))
    _MASK_BYTE = b"N"


class AminoAcidSequence(AlphabetSequence):
    """Standard 20-letter amino-acid sequence.

    The alphabet includes ``*`` (stop), ``_`` (gap), and ``X`` (unknown).
    ``X`` serves as the mask / wildcard symbol.
    """

    __slots__ = ()
    DEFAULT_ALPHABET = SequenceAlphabet((
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
        "*", "_", "X",
    ))
    _MASK_BYTE = b"X"

    def to_reduced_amino_acid(self) -> ReducedAminoAcidSequence:
        """Convert to the reduced physico-chemical alphabet via byte LUT."""
        converted = _AA_TO_REDUCED_LUT[self._data.view(np.uint8)].view("S1").copy()
        return ReducedAminoAcidSequence(converted)

    def to_simple_amino_acid(self) -> ReducedAminoAcidSequence:
        """Backwards-compatible alias for :meth:`to_reduced_amino_acid`."""
        return self.to_reduced_amino_acid()

    def matches_reduced_amino_acid(self, reduced: ReducedAminoAcidSequence) -> bool:
        """Wildcard-aware match against a reduced amino-acid sequence.

        Each position of *self* is first mapped to the reduced alphabet via a
        byte lookup table; then positions are compared treating ``X`` on
        either side as a wildcard.  Like :meth:`matches`, this is **not** an
        equality test.
        """
        if len(self) != len(reduced):
            return False
        if len(self) == 0:
            return True
        converted = _AA_TO_REDUCED_LUT[self._data.view(np.uint8)].view("S1")
        eq = converted == reduced._data
        if eq.all():
            return True
        mask_x = np.array(b"X", dtype="S1")
        return bool((eq | (self._data == mask_x) | (reduced._data == mask_x)).all())

    def matches_simple_amino_acid(self, simple: ReducedAminoAcidSequence) -> bool:
        """Backwards-compatible alias for :meth:`matches_reduced_amino_acid`."""
        return self.matches_reduced_amino_acid(simple)


class ReducedAminoAcidSequence(AlphabetSequence):
    """Sequence in the reduced physico-chemical amino-acid alphabet.

    Symbols: ``l b m c s h G F P W Y X * _``.  ``X`` is the mask / wildcard.
    Instances are typically obtained via
    :meth:`AminoAcidSequence.to_reduced_amino_acid`.
    """

    __slots__ = ()
    DEFAULT_ALPHABET = REDUCED_AMINO_ACID_ALPHABET
    _MASK_BYTE = b"X"
