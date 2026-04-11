"""Biological sequence validation, translation, masking, and matching.

All functions operate on plain ``str`` or ``bytes`` — no wrapper classes.
Alphabet membership is checked via 256-byte lookup tables (``bytes``) for
O(1) per-character validation.  Translation uses ``bytes.translate`` with a
pre-built table for native-speed conversion.

Alphabets
---------
Three predefined alphabets are provided as module-level ``bytes`` lookup
tables (256 entries, 1 = allowed, 0 = disallowed):

* ``NT_ALPHABET``       — DNA nucleotides ``ATGCN`` (``N`` = mask).
* ``AA_ALPHABET``       — 20 amino acids + ``*_X`` (``X`` = mask).
* ``REDUCED_AA_ALPHABET`` — Physico-chemical reduced alphabet (``X`` = mask).

Functions
---------
* ``make_alphabet``     — Build a 256-byte LUT from a string of allowed chars.
* ``validate``          — Check every byte belongs to an alphabet.
* ``translate``         — Byte-level translation via ``bytes.translate``.
* ``mask``              — Replace position(s) with a mask character.
* ``matches``           — Wildcard-aware positional comparison.
* ``aa_to_reduced``     — Convert amino-acid sequence to reduced alphabet.
* ``matches_aa_reduced``— Cross-alphabet wildcard match (AA vs reduced).
"""

from __future__ import annotations

Seq = str | bytes | bytearray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_bytes(seq: Seq) -> bytes:
    """Normalise *seq* to ``bytes``.  Strings are ASCII-encoded."""
    return seq.encode("ascii") if isinstance(seq, str) else bytes(seq)


# ---------------------------------------------------------------------------
# Alphabet construction
# ---------------------------------------------------------------------------

def make_alphabet(chars: str) -> bytes:
    """Build a 256-byte lookup table where allowed positions are ``1``.

    Args:
        chars: String of allowed ASCII characters.

    Returns:
        A 256-byte ``bytes`` object usable as a fast membership LUT.
    """
    lut = bytearray(256)
    for ch in chars:
        lut[ord(ch)] = 1
    return bytes(lut)


# ---------------------------------------------------------------------------
# Pre-built alphabets
# ---------------------------------------------------------------------------

NT_CHARS = "ATGCN"
AA_CHARS = "ACDEFGHIKLMNPQRSTVWY*_X"
REDUCED_AA_CHARS = "lbmcshGFPWYX*_"

NT_ALPHABET: bytes = make_alphabet(NT_CHARS)
AA_ALPHABET: bytes = make_alphabet(AA_CHARS)
REDUCED_AA_ALPHABET: bytes = make_alphabet(REDUCED_AA_CHARS)

NT_MASK = ord("N")
AA_MASK = ord("X")
REDUCED_AA_MASK = ord("X")


# ---------------------------------------------------------------------------
# Amino-acid → reduced-alphabet mapping
# ---------------------------------------------------------------------------

#: Per-character mapping from standard amino-acid codes to reduced symbols.
AA_TO_REDUCED: dict[str, str] = {
    "A": "l", "R": "b", "N": "m", "D": "c", "C": "s", "Q": "m",
    "E": "c", "G": "G", "H": "b", "I": "l", "L": "l", "K": "b",
    "M": "s", "F": "F", "P": "P", "S": "h", "T": "h", "W": "W",
    "Y": "Y", "V": "l", "X": "X", "*": "*", "_": "_",
}

#: ``bytes.translate`` table for fast AA → reduced conversion.
AA_TO_REDUCED_TABLE: bytes = bytes.maketrans(
    "".join(AA_TO_REDUCED.keys()).encode(),
    "".join(AA_TO_REDUCED.values()).encode(),
)

#: 256-byte LUT mapping each AA byte to its reduced byte (for matching).
_AA_TO_REDUCED_LUT: bytes
_lut = bytearray(256)
for _aa, _red in AA_TO_REDUCED.items():
    _lut[ord(_aa)] = ord(_red)
_AA_TO_REDUCED_LUT = bytes(_lut)
del _lut, _aa, _red


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(seq: Seq, alphabet: bytes) -> bytes:
    """Validate that every byte of *seq* belongs to *alphabet*.

    Accepts ``str``, ``bytes``, or ``bytearray``.  Strings are
    ASCII-encoded first.

    Args:
        seq: Input sequence.
        alphabet: 256-byte LUT (1 = allowed).

    Returns:
        The validated sequence as ``bytes``.

    Raises:
        ValueError: If any byte falls outside the alphabet.
    """
    raw = _to_bytes(seq)
    for b in raw:
        if not alphabet[b]:
            raise ValueError(
                f"Sequence contains symbol {chr(b)!r} outside of alphabet"
            )
    return raw


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate(seq: Seq, table: bytes) -> bytes:
    """Translate *seq* byte-by-byte using a ``bytes.maketrans`` *table*.

    Args:
        seq: Input sequence (``str``, ``bytes``, or ``bytearray``).
        table: A 256-byte translation table (from ``bytes.maketrans``).

    Returns:
        Translated ``bytes``.
    """
    return _to_bytes(seq).translate(table)


def aa_to_reduced(seq: Seq) -> bytes:
    """Convert an amino-acid sequence to the reduced physico-chemical alphabet.

    Uses ``bytes.translate`` with a pre-built table for native speed.

    Args:
        seq: Amino-acid sequence (``str``, ``bytes``, or ``bytearray``).

    Returns:
        Reduced-alphabet ``bytes``.
    """
    return _to_bytes(seq).translate(AA_TO_REDUCED_TABLE)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask(seq: Seq, position: int | slice | tuple[int, int], mask_byte: int) -> bytes:
    """Return a copy of *seq* with the given position(s) replaced by *mask_byte*.

    Args:
        seq: Input sequence.
        position: Single index, ``slice``, or ``(start, stop)`` half-open range.
        mask_byte: Replacement byte value (e.g. ``ord('N')`` or ``NT_MASK``).

    Returns:
        New ``bytes`` with the specified positions masked.

    Raises:
        IndexError: If a single-index position is out of bounds.
    """
    buf = bytearray(_to_bytes(seq))
    if isinstance(position, int):
        n = len(buf)
        if position < 0:
            position += n
        if position < 0 or position >= n:
            raise IndexError("Mask position out of range")
        buf[position] = mask_byte
    elif isinstance(position, slice):
        for i in range(*position.indices(len(buf))):
            buf[i] = mask_byte
    elif isinstance(position, tuple) and len(position) == 2:
        for i in range(position[0], position[1]):
            buf[i] = mask_byte
    else:
        raise TypeError("position must be int, slice, or (start, stop) tuple")
    return bytes(buf)


# ---------------------------------------------------------------------------
# Wildcard matching
# ---------------------------------------------------------------------------

def matches(a: Seq, b: Seq, mask_byte: int) -> bool:
    """Wildcard-aware positional comparison.

    Returns ``True`` when *a* and *b* have the same length and at every
    position the bytes are equal **or** at least one side carries
    *mask_byte*.  This is **not** the same as ``a == b``.

    Args:
        a: First sequence.
        b: Second sequence.
        mask_byte: The wildcard byte value (e.g. ``NT_MASK``).

    Returns:
        ``True`` if the sequences match, ``False`` otherwise.
    """
    ba = _to_bytes(a)
    bb = _to_bytes(b)
    if len(ba) != len(bb):
        return False
    if ba == bb:
        return True
    for x, y in zip(ba, bb):
        if x == y or x == mask_byte or y == mask_byte:
            continue
        return False
    return True


def matches_aa_reduced(aa_seq: Seq, reduced_seq: Seq) -> bool:
    """Wildcard-aware match between an amino-acid and a reduced-alphabet sequence.

    Each byte of *aa_seq* is first mapped to the reduced alphabet via a
    byte LUT, then compared against *reduced_seq*.  ``X`` (mask) on either
    side counts as a wildcard.

    Args:
        aa_seq: Amino-acid sequence.
        reduced_seq: Reduced-alphabet sequence.

    Returns:
        ``True`` if every position matches (accounting for wildcards).
    """
    ba = _to_bytes(aa_seq)
    br = _to_bytes(reduced_seq)
    if len(ba) != len(br):
        return False
    if len(ba) == 0:
        return True
    lut = _AA_TO_REDUCED_LUT
    mask_x = AA_MASK
    for a, r in zip(ba, br):
        conv = lut[a]
        if conv == r or a == mask_x or r == mask_x:
            continue
        return False
    return True
