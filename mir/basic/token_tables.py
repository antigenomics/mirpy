"""Rearrangement-level k-mer indexing.

Provides a lightweight ``Rearrangement`` type and hashable k-mer
named-tuples, together with functions that build inverted indices and
summary statistics from rearrangement lists.

Functions
---------
* ``tokenize_rearrangements`` — ``dict[Kmer, list[KmerMatch]]`` with
  position tracking.
* ``summarize_rearrangements`` — ``dict[Kmer, KmerStats]`` (full key).
* ``summarize_annotations``   — ``dict[KmerSeq, dict[KmerAnnotation, KmerStats]]``
  keyed by (locus, seq) only, mapping to per-(v_gene, c_gene, position)
  counts.

All functions accept an optional *mask_byte* for gapped k-mers.
No runtime type checks — relies on static typing.
"""

from __future__ import annotations

from typing import NamedTuple

from mir.basic.sequence import Seq, _to_bytes


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Rearrangement:
    """Immune receptor rearrangement with minimal annotation.

    Uses ``__slots__`` for memory efficiency.
    """

    __slots__ = ("locus", "id", "v_gene", "c_gene", "junction_aa",
                 "duplicate_count")

    def __init__(
        self,
        locus: str,
        id: str,
        v_gene: str,
        c_gene: str,
        junction_aa: str,
        duplicate_count: int,
    ) -> None:
        self.locus = locus
        self.id = id
        self.v_gene = v_gene
        self.c_gene = c_gene
        self.junction_aa = junction_aa
        self.duplicate_count = duplicate_count


class Kmer(NamedTuple):
    """Annotated k-mer: sequence plus the gene context it was drawn from.

    Hashable by default (NamedTuple), so it can serve as a ``dict`` key.
    """

    locus: str
    v_gene: str
    c_gene: str
    seq: bytes


class KmerMatch(NamedTuple):
    """A single k-mer occurrence linking back to its source."""

    rearrangement: Rearrangement
    position: int


class KmerSeq(NamedTuple):
    """Reduced k-mer key: locus + sequence only (ignores gene annotation)."""

    locus: str
    seq: bytes


class KmerAnnotation(NamedTuple):
    """Parent annotation for a k-mer occurrence."""

    v_gene: str
    c_gene: str
    position: int


class KmerStats(NamedTuple):
    """Aggregate statistics for a single k-mer (or annotation bucket)."""

    rearrangement_count: int
    duplicate_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plain_kmers(raw: bytes, k: int) -> list[tuple[bytes, int]]:
    """Overlapping k-mers from *raw* with their start positions."""
    return [(raw[i : i + k], i) for i in range(len(raw) - k + 1)]


def _gapped_kmers(raw: bytes, k: int, mask_byte: int) -> list[tuple[bytes, int]]:
    """Gapped k-mers (each position masked once) with window start positions."""
    n = len(raw)
    n_windows = n - k + 1
    buf = bytearray(k)
    result: list[tuple[bytes, int]] = []
    for i in range(n_windows):
        window = raw[i : i + k]
        for j in range(k):
            buf[:] = window
            buf[j] = mask_byte
            result.append((bytes(buf), i))
    return result


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def tokenize_rearrangements(
    rearrangements: list[Rearrangement],
    k: int,
    mask_byte: int | None = None,
) -> dict[Kmer, list[KmerMatch]]:
    """Build an inverted index from annotated k-mers to their source
    rearrangements, tracking the position each k-mer was extracted from.

    When *mask_byte* is given, gapped k-mers are produced instead (each
    position masked once per window, as in :func:`tokens.tokenize_gapped`).

    Args:
        rearrangements: Input rearrangements.
        k: K-mer length.
        mask_byte: If not ``None``, replacement byte for gapped k-mers
            (e.g. ``ord('X')``).  ``None`` (default) produces plain k-mers.

    Returns:
        Dict mapping each :class:`Kmer` to a list of :class:`KmerMatch`
        (rearrangement, position) pairs.
    """
    _kmers = _gapped_kmers if mask_byte is not None else _plain_kmers
    _mb = mask_byte
    index: dict[Kmer, list[KmerMatch]] = {}
    for r in rearrangements:
        raw = _to_bytes(r.junction_aa)
        if k > len(raw):
            continue
        locus = r.locus
        v_gene = r.v_gene
        c_gene = r.c_gene
        pairs = _kmers(raw, k, _mb) if _mb is not None else _kmers(raw, k)
        for s, pos in pairs:
            key = Kmer(locus, v_gene, c_gene, s)
            match = KmerMatch(r, pos)
            lst = index.get(key)
            if lst is None:
                index[key] = [match]
            else:
                lst.append(match)
    return index


def summarize_rearrangements(
    rearrangements: list[Rearrangement],
    k: int,
    mask_byte: int | None = None,
) -> dict[Kmer, KmerStats]:
    """Compute per-kmer summary statistics (full :class:`Kmer` key).

    For each :class:`Kmer` the result contains:

    * ``rearrangement_count`` — number of rearrangements contributing
      that k-mer.
    * ``duplicate_count`` — sum of :attr:`Rearrangement.duplicate_count`
      across those rearrangements.

    Args:
        rearrangements: Input rearrangements.
        k: K-mer length.
        mask_byte: If not ``None``, produce gapped k-mers.

    Returns:
        Dict mapping each :class:`Kmer` to its :class:`KmerStats`.
    """
    _kmers = _gapped_kmers if mask_byte is not None else _plain_kmers
    _mb = mask_byte
    counts: dict[Kmer, int] = {}
    dups: dict[Kmer, int] = {}
    for r in rearrangements:
        raw = _to_bytes(r.junction_aa)
        if k > len(raw):
            continue
        locus = r.locus
        v_gene = r.v_gene
        c_gene = r.c_gene
        dc = r.duplicate_count
        pairs = _kmers(raw, k, _mb) if _mb is not None else _kmers(raw, k)
        for s, _pos in pairs:
            key = Kmer(locus, v_gene, c_gene, s)
            counts[key] = counts.get(key, 0) + 1
            dups[key] = dups.get(key, 0) + dc
    return {k: KmerStats(counts[k], dups[k]) for k in counts}


def summarize_annotations(
    rearrangements: list[Rearrangement],
    k: int,
    mask_byte: int | None = None,
) -> dict[KmerSeq, dict[KmerAnnotation, KmerStats]]:
    """Compute per-kmer summary keyed by (locus, seq) only, with
    per-(v_gene, c_gene, position) breakdowns.

    The outer key is a :class:`KmerSeq` — just locus and k-mer bytes,
    ignoring gene annotation.  The inner dict maps each unique
    :class:`KmerAnnotation` (v_gene, c_gene, position) to a
    :class:`KmerStats` holding rearrangement_count and duplicate_count.

    Args:
        rearrangements: Input rearrangements.
        k: K-mer length.
        mask_byte: If not ``None``, produce gapped k-mers.

    Returns:
        Nested dict ``KmerSeq → KmerAnnotation → KmerStats``.
    """
    _kmers = _gapped_kmers if mask_byte is not None else _plain_kmers
    _mb = mask_byte
    # Accumulate into flat (KmerSeq, KmerAnnotation) → (count, dup_sum)
    counts: dict[tuple[KmerSeq, KmerAnnotation], int] = {}
    dups: dict[tuple[KmerSeq, KmerAnnotation], int] = {}
    for r in rearrangements:
        raw = _to_bytes(r.junction_aa)
        if k > len(raw):
            continue
        locus = r.locus
        v_gene = r.v_gene
        c_gene = r.c_gene
        dc = r.duplicate_count
        pairs = _kmers(raw, k, _mb) if _mb is not None else _kmers(raw, k)
        for s, pos in pairs:
            ks = KmerSeq(locus, s)
            ka = KmerAnnotation(v_gene, c_gene, pos)
            flat_key = (ks, ka)
            counts[flat_key] = counts.get(flat_key, 0) + 1
            dups[flat_key] = dups.get(flat_key, 0) + dc
    # Pivot into nested dict
    result: dict[KmerSeq, dict[KmerAnnotation, KmerStats]] = {}
    for (ks, ka), cnt in counts.items():
        inner = result.get(ks)
        if inner is None:
            inner = {}
            result[ks] = inner
        inner[ka] = KmerStats(cnt, dups[(ks, ka)])
    return result
