"""Biological sequence types backed by immutable ``stringzilla.Str`` buffers.

Sequence objects are lightweight and immutable.  Each instance stores only a
``stringzilla.Str`` (a zero-copy, contiguous byte buffer).  The alphabet is
validated on construction and defined at the class level.

**Equality vs. matching:**  ``__eq__`` and ``__hash__`` compare raw bytes so
sequences work correctly as ``dict`` keys and ``set`` members.
:meth:`~AlphabetSequence.matches` is different — it treats mask characters
(``N`` for nucleotides, ``X`` for amino-acid alphabets) as wildcards, so two
sequences that *match* may not be *equal*.

Classes:
    SequenceAlphabet         -- Singleton alphabet definition.
    AlphabetSequence         -- Base class for alphabet-constrained sequences.
    NucleotideSequence       -- DNA sequence (A/T/G/C/N).
    AminoAcidSequence        -- Standard 20-AA + stop/unknown sequence.
    ReducedAminoAcidSequence -- Reduced amino-acid alphabet for fuzzy matching.
"""

from __future__ import annotations

import numpy as np
import stringzilla as sz
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

#: NumPy uint8 LUT (128 entries) for fast vectorised conversion used by
#: :meth:`AminoAcidSequence.matches_reduced_amino_acid`.
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
        _allowed_set: ``frozenset`` of allowed byte values for O(1) lookup.
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
        self._allowed_set = frozenset(
            c.encode("ascii") for c in self.allowed_symbols
        )
        # Kept for any downstream code that uses np.isin against this.
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
    """Immutable alphabet-validated sequence backed by a ``stringzilla.Str``.

    Each instance stores **only** a ``sz.Str`` buffer.  The alphabet and mask
    symbol live on the class, not on instances.

    **Equality vs matching:**
    ``__eq__`` / ``__hash__`` compare raw bytes (for ``dict`` / ``set`` use).
    :meth:`matches` performs wildcard-aware comparison where mask characters
    (``N`` for nucleotides, ``X`` for amino-acid types) count as matching any
    symbol — so two sequences can *match* without being *equal*.

    Subclass protocol:
        * ``DEFAULT_ALPHABET`` — :class:`SequenceAlphabet` with allowed symbols.
        * ``_MASK_BYTE``       — ``b"N"``, ``b"X"``, or ``b""`` (no masking).
    """

    __slots__ = ("_sz",)

    DEFAULT_ALPHABET: SequenceAlphabet  # set by subclasses
    _MASK_BYTE: bytes = b""

    def __init__(self, data: np.ndarray) -> None:
        """Validate *data* against the class alphabet and store as ``sz.Str``.

        Args:
            data: One-dimensional ``S1``-dtype NumPy array.

        Raises:
            ValueError: If dtype is not ``S1``, array is not 1-D, or
                symbols fall outside ``DEFAULT_ALPHABET``.
        """
        if data.dtype != np.dtype("S1"):
            raise ValueError("Sequence storage must have dtype S1")
        if data.ndim != 1:
            raise ValueError("Sequence storage must be one-dimensional")
        allowed = self.DEFAULT_ALPHABET._allowed_set
        raw = data.tobytes()
        for b in raw:
            if b.to_bytes(1, "little") not in allowed:
                raise ValueError("Sequence contains symbols outside of alphabet")
        self._sz = sz.Str(raw)

    @classmethod
    def _from_trusted_bytes(cls: type[Self], raw: bytes) -> Self:
        """Fast-path constructor that skips alphabet validation.

        The caller **must** guarantee that every byte in *raw* belongs to
        ``cls.DEFAULT_ALPHABET``.  This is used internally by
        :func:`~mir.basic.tokens.tokenize` and :meth:`substring` where the
        source data has already been validated.
        """
        inst = object.__new__(cls)
        inst._sz = sz.Str(raw)
        return inst

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_string(cls: type[Self], sequence: str) -> Self:
        """Create a sequence from a plain Python string."""
        arr = np.frombuffer(sequence.encode("ascii"), dtype="S1").copy()
        return cls(arr)

    # -- accessors ----------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        """Read-only ``S1`` NumPy view of the sequence bytes."""
        arr = np.frombuffer(bytes(self._sz), dtype="S1").copy()
        arr.flags.writeable = False
        return arr

    @property
    def content(self) -> np.ndarray:
        """Alias for :attr:`data` (backward compatibility)."""
        return self.data

    def to_string(self) -> str:
        """Decode the sequence to a plain Python string."""
        return str(self._sz)

    def to_bytes(self) -> bytes:
        """Return the raw byte content."""
        return bytes(self._sz)

    def substring(self, start: int, stop: int | None = None) -> Self:
        """Return a new sequence for the half-open range ``[start, stop)``.

        Uses ``sz.Str`` slicing for a zero-copy view, then stores a copy.
        """
        sliced = self._sz[start:stop]
        return type(self)._from_trusted_bytes(bytes(sliced))

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
        buf = bytearray(bytes(self._sz))
        mask_val = self._MASK_BYTE[0]
        if isinstance(position, int):
            n = len(buf)
            if position < 0:
                position += n
            if position < 0 or position >= n:
                raise IndexError("Mask position out of range")
            buf[position] = mask_val
        elif isinstance(position, slice):
            for i in range(*position.indices(len(buf))):
                buf[i] = mask_val
        elif isinstance(position, tuple) and len(position) == 2:
            for i in range(position[0], position[1]):
                buf[i] = mask_val
        else:
            raise TypeError("position must be int, slice, or (start, stop) tuple")
        return type(self)._from_trusted_bytes(bytes(buf))

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
        sb = bytes(self._sz)
        ob = bytes(other._sz)
        if sb == ob:
            return True
        sm = self._MASK_BYTE[0] if self._MASK_BYTE else -1
        om = other._MASK_BYTE[0] if other._MASK_BYTE else -1
        for a, b in zip(sb, ob):
            if a == b or a == sm or b == om:
                continue
            return False
        return True

    # -- equality & hashing (byte-exact) ------------------------------------

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return bytes(self._sz) == bytes(other._sz)

    def __hash__(self) -> int:
        return hash(bytes(self._sz))

    def __len__(self) -> int:
        return len(self._sz)

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
        """Convert to the reduced physico-chemical alphabet via ``sz.translate``.

        Uses the :data:`AMINO_ACID_TO_REDUCED_AMINO_ACID` char→char mapping
        applied through ``stringzilla.Str.translate`` for native-speed
        byte-level translation.
        """
        translated: bytes = self._sz.translate(AMINO_ACID_TO_REDUCED_AMINO_ACID)
        return ReducedAminoAcidSequence._from_trusted_bytes(translated)

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
        # Use numpy LUT for the comparison (avoids creating an intermediate
        # ReducedAminoAcidSequence object).
        self_np = np.frombuffer(bytes(self._sz), dtype=np.uint8)
        converted = _AA_TO_REDUCED_LUT[self_np].view("S1")
        reduced_np = np.frombuffer(bytes(reduced._sz), dtype="S1")
        eq = converted == reduced_np
        if eq.all():
            return True
        mask_x = np.array(b"X", dtype="S1")
        self_s1 = self_np.view("S1")
        return bool((eq | (self_s1 == mask_x) | (reduced_np == mask_x)).all())

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


#: Backwards-compatible class alias.
SimpleAminoAcidSequence = ReducedAminoAcidSequence
