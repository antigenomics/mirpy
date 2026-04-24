"""Rearrangement-level k-mer indexing.

Provides a lightweight ``Rearrangement`` type and hashable k-mer
named-tuples, together with functions that build inverted indices,
summary statistics, and filtered views of token tables.

Functions
---------
* ``tokenize_rearrangements`` — ``dict[Kmer, list[KmerMatch]]`` with
  position tracking.
* ``filter_token_table``      — filter by regex pattern and/or minimum
  rearrangement count; both criteria may be combined.
* ``summarize_rearrangements`` — ``dict[Kmer, KmerStats]`` (full key).
* ``summarize_annotations``   — ``dict[KmerSeq, dict[KmerAnnotation, KmerStats]]``
  keyed by (locus, seq) only, mapping to per-(v_gene, c_gene, position)
  counts.

All tokenization functions accept an optional *mask_byte* for gapped k-mers.
No runtime type checks — relies on static typing.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from mir.basic.alphabets import Seq, _to_bytes
from mir.basic.mirseq import (
    tokenize_bytes as _c_tokenize_bytes,
    tokenize_gapped_bytes as _c_tokenize_gapped_bytes,
)
from mir.common.clonotype import Clonotype


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Rearrangement is now an alias for Clonotype.  Existing code that constructs
# Rearrangement(locus, id, v_gene, c_gene, junction_aa, duplicate_count) must
# be updated to use Clonotype keyword arguments.
Rearrangement = Clonotype


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
    """Aggregate statistics for a single k-mer (or annotation bucket).

    ``rearrangement_count`` holds the number of unique rearrangement IDs;
    ``duplicate_count`` holds the sum of ``Rearrangement.duplicate_count``.
    """

    rearrangement_count: int
    duplicate_count: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plain_kmers(raw: bytes, k: int) -> list[tuple[bytes, int]]:
    """Overlapping k-mers from *raw* with their start positions.

    Delegates to the C extension for the k-mer extraction.
    """
    kmers = _c_tokenize_bytes(raw, k)
    return [(kmer, i) for i, kmer in enumerate(kmers)]


def _gapped_kmers(raw: bytes, k: int, mask_byte: int) -> list[tuple[bytes, int]]:
    """Gapped k-mers (each position masked once) with window start positions.

    Delegates to the C extension for the gapped k-mer extraction.
    """
    gapped = _c_tokenize_gapped_bytes(raw, k, mask_byte)
    n_windows = len(raw) - k + 1
    result: list[tuple[bytes, int]] = []
    idx = 0
    for i in range(n_windows):
        for _ in range(k):
            result.append((gapped[idx], i))
            idx += 1
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

    * ``rearrangement_count`` — number of *unique* rearrangement IDs
      contributing that k-mer.
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
    ids: dict[Kmer, set[int]] = {}
    dups: dict[Kmer, int] = {}
    for r in rearrangements:
        raw = _to_bytes(r.junction_aa)
        if k > len(raw):
            continue
        locus = r.locus
        v_gene = r.v_gene
        c_gene = r.c_gene
        rid = r.id
        dc = r.duplicate_count
        pairs = _kmers(raw, k, _mb) if _mb is not None else _kmers(raw, k)
        for s, _pos in pairs:
            key = Kmer(locus, v_gene, c_gene, s)
            id_set = ids.get(key)
            if id_set is None:
                ids[key] = {rid}
                dups[key] = dc
            elif rid not in id_set:
                id_set.add(rid)
                dups[key] += dc
    return {k: KmerStats(len(ids[k]), dups[k]) for k in ids}


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
    :class:`KmerStats` holding rearrangement_count (unique IDs) and
    duplicate_count.

    Args:
        rearrangements: Input rearrangements.
        k: K-mer length.
        mask_byte: If not ``None``, produce gapped k-mers.

    Returns:
        Nested dict ``KmerSeq → KmerAnnotation → KmerStats``.
    """
    _kmers = _gapped_kmers if mask_byte is not None else _plain_kmers
    _mb = mask_byte
    ids: dict[tuple[KmerSeq, KmerAnnotation], set[int]] = {}
    dups: dict[tuple[KmerSeq, KmerAnnotation], int] = {}
    for r in rearrangements:
        raw = _to_bytes(r.junction_aa)
        if k > len(raw):
            continue
        locus = r.locus
        v_gene = r.v_gene
        c_gene = r.c_gene
        rid = r.id
        dc = r.duplicate_count
        pairs = _kmers(raw, k, _mb) if _mb is not None else _kmers(raw, k)
        for s, pos in pairs:
            ks = KmerSeq(locus, s)
            ka = KmerAnnotation(v_gene, c_gene, pos)
            flat_key = (ks, ka)
            id_set = ids.get(flat_key)
            if id_set is None:
                ids[flat_key] = {rid}
                dups[flat_key] = dc
            elif rid not in id_set:
                id_set.add(rid)
                dups[flat_key] += dc
    # Pivot into nested dict
    result: dict[KmerSeq, dict[KmerAnnotation, KmerStats]] = {}
    for (ks, ka), id_set in ids.items():
        inner = result.get(ks)
        if inner is None:
            inner = {}
            result[ks] = inner
        inner[ka] = KmerStats(len(id_set), dups[(ks, ka)])
    return result


def filter_token_table(
    table: dict[Kmer, list[KmerMatch]],
    kmer_pattern: str | None = None,
    min_rearrangement_count: int | None = None,
) -> dict[Kmer, list[KmerMatch]]:
    """Return a filtered view of a token table.

    Both filters may be combined; a k-mer must satisfy *all* active criteria
    to be retained.

    Args:
        table: Output of :func:`tokenize_rearrangements`.
        kmer_pattern: Regular expression matched against each k-mer sequence
            decoded as ASCII.  Only k-mers whose sequence matches are kept.
            ``None`` (default) skips pattern filtering.
        min_rearrangement_count: Minimum number of *distinct* rearrangement IDs
            that must contain a k-mer for it to be retained.  ``None`` (default)
            skips count filtering.  Pass ``3`` to discard rare k-mers seen in
            fewer than three rearrangements.

    Returns:
        Filtered ``dict[Kmer, list[KmerMatch]]``.  The original table is
        never modified.
    """
    if kmer_pattern is None and min_rearrangement_count is None:
        return table
    rx = re.compile(kmer_pattern) if kmer_pattern is not None else None
    result: dict[Kmer, list[KmerMatch]] = {}
    for kmer, matches in table.items():
        if rx is not None and not rx.search(kmer.seq.decode("ascii")):
            continue
        if min_rearrangement_count is not None:
            n_distinct = len({m.rearrangement.id for m in matches})
            if n_distinct < min_rearrangement_count:
                continue
        result[kmer] = matches
    return result
