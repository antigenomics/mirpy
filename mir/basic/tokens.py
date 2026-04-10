"""K-mer tokenisation of :class:`~mir.basic.sequence.AlphabetSequence` objects.

Uses ``stringzilla.Str`` slicing for zero-copy windowing and the fast
:meth:`~mir.basic.sequence.AlphabetSequence._from_trusted_bytes` constructor
to bypass per-k-mer alphabet validation (the source sequence was already
validated on construction).

Functions:
    tokenize -- Extract overlapping k-mers, optionally with gapped variants.
"""

from __future__ import annotations

import stringzilla as sz

from mir.basic.sequence import AlphabetSequence


def tokenize(
    sequence: AlphabetSequence,
    k: int,
    *,
    gapped: bool = False,
) -> list[AlphabetSequence]:
    """Split *sequence* into overlapping k-mers of length *k*.

    Uses ``sz.Str`` slicing (zero-copy view) for each window and
    :meth:`AlphabetSequence._from_trusted_bytes` to construct k-mer objects
    without re-validating the alphabet.

    When *gapped* is ``True``, instead of plain k-mers, each window position
    produces *k* gapped variants where exactly one position within the k-mer
    is replaced by the mask byte (``N`` for nucleotides, ``X`` for amino-acid
    types).  For example, with ``k=3`` and amino-acid sequence ``CASSL``::

        position 0 → XAS  CXS  CAX
        position 1 → XSS  AXS  ASX
        position 2 → XSL  SXL  SSX

    Args:
        sequence: Input sequence to tokenize.
        k: K-mer length.  Must satisfy ``1 <= k <= len(sequence)``.
        gapped: If ``True``, emit gapped (single-position masked) k-mers
            rather than plain k-mers.

    Returns:
        A flat list of k-mer sequences.  Plain mode yields
        ``len(sequence) - k + 1`` items; gapped mode yields
        ``(len(sequence) - k + 1) * k`` items.

    Raises:
        ValueError: If *k* < 1 or *k* > ``len(sequence)``.
    """
    n = len(sequence)
    if k < 1 or k > n:
        raise ValueError(
            f"k must be between 1 and sequence length ({n}), got {k}"
        )

    cls = type(sequence)
    raw_sz = sequence._sz  # stringzilla.Str — slicing is zero-copy

    if not gapped:
        result: list[AlphabetSequence] = []
        for i in range(n - k + 1):
            result.append(cls._from_trusted_bytes(bytes(raw_sz[i : i + k])))
        return result

    # Gapped mode: for each window spawn k variants, each with one
    # position replaced by the mask byte.
    mask_byte = sequence._MASK_BYTE
    if not mask_byte:
        raise ValueError(
            f"Gapped tokenisation requires a mask byte; "
            f"{cls.__name__} does not define one"
        )
    mask_val = mask_byte[0]

    result = []
    for i in range(n - k + 1):
        window = bytes(raw_sz[i : i + k])
        for j in range(k):
            buf = bytearray(window)
            buf[j] = mask_val
            result.append(cls._from_trusted_bytes(bytes(buf)))
    return result

