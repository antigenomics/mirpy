"""K-mer tokenisation for biological sequences.

Thin wrappers around the ``mirseq`` C extension.  Accepts ``str``,
``bytes``, or ``bytearray`` inputs.

Functions
---------
* ``tokenize``        — Overlapping k-mers as ``list[bytes]``.
* ``tokenize_gapped`` — Gapped (single-position masked) k-mers as ``list[bytes]``.
* ``tokenize_str``        — Same as ``tokenize`` returning ``list[str]``.
* ``tokenize_gapped_str`` — Same as ``tokenize_gapped`` returning ``list[str]``.
"""

from __future__ import annotations

from mir.basic.mirseq import (
    tokenize_bytes as _c_tokenize_bytes,
    tokenize_str as _c_tokenize_str,
    tokenize_gapped_bytes as _c_tokenize_gapped_bytes,
    tokenize_gapped_str as _c_tokenize_gapped_str,
)
from mir.basic.alphabets import Seq


# ---------------------------------------------------------------------------
# Plain k-mers
# ---------------------------------------------------------------------------

def tokenize(seq: Seq, k: int) -> list[bytes]:
    """Extract overlapping k-mers of length *k* from *seq*.

    Delegates to the ``mirseq`` C extension for speed.

    Args:
        seq: Input sequence.
        k:   K-mer length.  Must satisfy ``1 <= k <= len(seq)``.

    Returns:
        List of ``bytes`` k-mers (length ``len(seq) - k + 1``).
    """
    return _c_tokenize_bytes(seq, k)


def tokenize_str(seq: Seq, k: int) -> list[str]:
    """Like :func:`tokenize` but returns ``list[str]``."""
    return _c_tokenize_str(seq, k)


# ---------------------------------------------------------------------------
# Gapped k-mers
# ---------------------------------------------------------------------------

def tokenize_gapped(seq: Seq, k: int, mask_byte: int) -> list[bytes]:
    """Extract gapped k-mers: for each window, *k* variants with one
    position replaced by *mask_byte*.

    Delegates to the ``mirseq`` C extension for speed.

    Args:
        seq: Input sequence.
        k:   K-mer length.  Must satisfy ``1 <= k <= len(seq)``.
        mask_byte: Replacement byte value (e.g. ``ord('X')``).

    Returns:
        List of ``bytes`` gapped k-mers.
        Length is ``(len(seq) - k + 1) * k``.
    """
    return _c_tokenize_gapped_bytes(seq, k, mask_byte)


def tokenize_gapped_str(seq: Seq, k: int, mask_char: str) -> list[str]:
    """Like :func:`tokenize_gapped` but returns ``list[str]``.

    Args:
        seq: Input sequence.
        k:   K-mer length.
        mask_char: Single-character mask string (e.g. ``"X"``).
    """
    return _c_tokenize_gapped_str(seq, k, ord(mask_char))
