"""Thin Python wrappers around the C-native distance functions in ``seqdist_c``.

Functions
---------
* ``hamming(a, b)``      — Hamming distance (equal-length sequences).
* ``levenshtein(a, b)``  — Levenshtein (edit) distance.
"""

from __future__ import annotations

from mir.basic.alphabets import Seq
from mir.distances.seqdist_c import hamming as _c_hamming, levenshtein as _c_levenshtein


def hamming(a: Seq, b: Seq) -> int:
    """Hamming distance between two equal-length sequences.

    Accepts ``str``, ``bytes``, or ``bytearray``.

    Raises:
        ValueError: If the sequences differ in length.
    """
    return _c_hamming(a, b)


def levenshtein(a: Seq, b: Seq) -> int:
    """Levenshtein (edit) distance between two sequences.

    Accepts ``str``, ``bytes``, or ``bytearray``.
    """
    return _c_levenshtein(a, b)
