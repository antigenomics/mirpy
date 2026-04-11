"""K-mer tokenisation for biological sequences.

Provides plain and gapped k-mer extraction operating on ``str`` or ``bytes``
inputs.  Both approaches use bytes slicing internally (``str.encode`` is
virtually free for short ASCII sequences and ``bytes`` slicing is faster than
``str`` slicing in CPython).

Functions
---------
* ``tokenize``        — Overlapping k-mers as a ``list[bytes]``.
* ``tokenize_gapped`` — Gapped (single-position masked) k-mers as ``list[bytes]``.
* ``tokenize_str``        — Same as ``tokenize`` returning ``list[str]``.
* ``tokenize_gapped_str`` — Same as ``tokenize_gapped`` returning ``list[str]``.
"""

from __future__ import annotations

from mir.basic.sequence import Seq, _to_bytes


# ---------------------------------------------------------------------------
# Plain k-mers
# ---------------------------------------------------------------------------

def tokenize(seq: Seq, k: int) -> list[bytes]:
    """Extract overlapping k-mers of length *k* from *seq*.

    Uses ``bytes`` slicing for speed; accepts ``str``, ``bytes``,
    or ``bytearray``.

    Args:
        seq: Input sequence.
        k:   K-mer length.  Must satisfy ``1 <= k <= len(seq)``.

    Returns:
        List of ``bytes`` k-mers (length ``len(seq) - k + 1``).

    Raises:
        ValueError: If *k* < 1 or *k* > ``len(seq)``.
    """
    raw = _to_bytes(seq)
    n = len(raw)
    if k < 1 or k > n:
        raise ValueError(
            f"k must be between 1 and sequence length ({n}), got {k}"
        )
    return [raw[i : i + k] for i in range(n - k + 1)]


def tokenize_str(seq: Seq, k: int) -> list[str]:
    """Like :func:`tokenize` but returns ``list[str]``.

    Internally converts to bytes, tokenizes, then decodes each k-mer.
    """
    return [km.decode("ascii") for km in tokenize(seq, k)]


# ---------------------------------------------------------------------------
# Gapped k-mers
# ---------------------------------------------------------------------------

def tokenize_gapped(seq: Seq, k: int, mask_byte: int) -> list[bytes]:
    """Extract gapped k-mers: for each window, *k* variants with one
    position replaced by *mask_byte*.

    For window ``CAS`` with mask ``X`` (88)::

        XAS  CXS  CAX

    Args:
        seq: Input sequence.
        k:   K-mer length.  Must satisfy ``1 <= k <= len(seq)``.
        mask_byte: Replacement byte value (e.g. ``ord('X')``).

    Returns:
        List of ``bytes`` gapped k-mers.
        Length is ``(len(seq) - k + 1) * k``.

    Raises:
        ValueError: If *k* < 1 or *k* > ``len(seq)``.
    """
    raw = _to_bytes(seq)
    n = len(raw)
    if k < 1 or k > n:
        raise ValueError(
            f"k must be between 1 and sequence length ({n}), got {k}"
        )
    n_windows = n - k + 1
    n_gapped = n_windows * k
    out = bytearray(n_gapped * k)
    offset = 0
    for i in range(n_windows):
        window = raw[i : i + k]
        for j in range(k):
            out[offset : offset + k] = window
            out[offset + j] = mask_byte
            offset += k
    frozen = bytes(out)
    return [frozen[i * k : (i + 1) * k] for i in range(n_gapped)]


def tokenize_gapped_str(seq: Seq, k: int, mask_char: str) -> list[str]:
    """Like :func:`tokenize_gapped` but returns ``list[str]``.

    Args:
        seq: Input sequence.
        k:   K-mer length.
        mask_char: Single-character mask string (e.g. ``"X"``).
    """
    return [
        km.decode("ascii")
        for km in tokenize_gapped(seq, k, ord(mask_char))
    ]
