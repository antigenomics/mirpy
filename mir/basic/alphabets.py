"""Alphabets, constants, and amino-acid → reduced-alphabet translation.

This module holds the lightweight, GC-friendly parts that are faster in
pure Python (``bytes.translate``) than in C.  Heavy-lifting functions
(codon translation, tokenisation, distances) live in the ``mirseq``
C extension.

Types
-----
* ``Seq`` — Union type ``str | bytes | bytearray``.

Helpers
-------
* ``_to_bytes`` — Normalise *Seq* to ``bytes``.

Alphabets
---------
* ``NT_ALPHABET`` / ``AA_ALPHABET`` / ``REDUCED_AA_ALPHABET`` — 256-byte LUTs.
* ``NT_MASK`` / ``AA_MASK`` / ``REDUCED_AA_MASK`` — Mask byte values.

Translation
-----------
* ``aa_to_reduced``     — AA → reduced via ``bytes.translate`` (fastest path).
* ``validate``          — Check every byte belongs to an alphabet.
* ``mask``              — Replace position(s) with a mask character.
* ``matches``           — Wildcard-aware positional comparison.
* ``matches_aa_reduced``— Cross-alphabet wildcard match (AA vs reduced).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

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
    """Build a 256-byte lookup table where allowed positions are ``1``."""
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

AA_TO_REDUCED: dict[str, str] = {
    "A": "l", "R": "b", "N": "m", "D": "c", "C": "s", "Q": "m",
    "E": "c", "G": "G", "H": "b", "I": "l", "L": "l", "K": "b",
    "M": "s", "F": "F", "P": "P", "S": "h", "T": "h", "W": "W",
    "Y": "Y", "V": "l", "X": "X", "*": "*", "_": "_",
}

AA_TO_REDUCED_TABLE: bytes = bytes.maketrans(
    "".join(AA_TO_REDUCED.keys()).encode(),
    "".join(AA_TO_REDUCED.values()).encode(),
)

_AA_TO_REDUCED_LUT: bytes
_lut = bytearray(256)
for _aa, _red in AA_TO_REDUCED.items():
    _lut[ord(_aa)] = ord(_red)
_AA_TO_REDUCED_LUT = bytes(_lut)
del _lut, _aa, _red


# ---------------------------------------------------------------------------
# Translation (aa_to_reduced — fastest in Python via bytes.translate)
# ---------------------------------------------------------------------------

def aa_to_reduced(seq: Seq) -> bytes:
    """Convert an amino-acid sequence to the reduced physico-chemical alphabet.

    Uses ``bytes.translate`` with a pre-built table — faster than C for
    this particular operation.
    """
    return _to_bytes(seq).translate(AA_TO_REDUCED_TABLE)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(seq: Seq, alphabet: bytes) -> bytes:
    """Validate every byte of *seq* belongs to *alphabet* (256-byte LUT)."""
    raw = _to_bytes(seq)
    for b in raw:
        if not alphabet[b]:
            raise ValueError(
                f"Sequence contains symbol {chr(b)!r} outside of alphabet"
            )
    return raw


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask(seq: Seq, position: int | slice | tuple[int, int], mask_byte: int) -> bytes:
    """Return a copy of *seq* with the given position(s) replaced by *mask_byte*."""
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
    *mask_byte*.
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


# ---------------------------------------------------------------------------
# Back-translation (amino acid → nucleotide)
# ---------------------------------------------------------------------------

# Most likely human codon per amino acid (Kazusa Homo sapiens codon usage table).
_MOST_LIKELY_CODON: dict[str, str] = {
    "A": "GCC", "R": "AGG", "N": "AAC", "D": "GAC",
    "C": "TGC", "Q": "CAG", "E": "GAG", "G": "GGC",
    "H": "CAC", "I": "ATC", "L": "CTG", "K": "AAG",
    "M": "ATG", "F": "TTC", "P": "CCC", "S": "AGC",
    "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
}


def back_translate(aa_seq: str, unknown_codon: str = "NNN") -> str:
    """Back-translate *aa_seq* to a nucleotide sequence.

    Each residue is mapped to the most frequently used human codon
    (Kazusa Homo sapiens codon usage database).  Non-standard residues
    (``X``, ``*``, ``_``, etc.) produce *unknown_codon* (default ``"NNN"``).

    The returned sequence has length ``len(aa_seq) * 3``.

    Examples
    --------
    >>> back_translate("CA")
    'TGCGCC'
    >>> back_translate("X")
    'NNN'
    """
    return "".join(_MOST_LIKELY_CODON.get(aa, unknown_codon) for aa in aa_seq)


def matches_aa_reduced(aa_seq: Seq, reduced_seq: Seq) -> bool:
    """Wildcard-aware match between an amino-acid and a reduced-alphabet sequence."""
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
