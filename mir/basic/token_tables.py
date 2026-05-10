"""Clonotype-level k-mer indexing.

Provides a lightweight ``Clonotype`` type and hashable k-mer
named-tuples, together with functions that build inverted indices,
summary statistics, and filtered views of token tables.

Functions
---------
* ``tokenize_clonotypes`` — ``dict[Kmer, list[KmerMatch]]`` with
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
import warnings
from itertools import islice
from typing import NamedTuple

from mir.basic.alphabets import _to_bytes
from mir.basic.tokens import (
    tokenize_with_positions,
    tokenize_gapped_with_positions,
)
from mir.common.clonotype import Clonotype


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
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

    rearrangement: Clonotype
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
    ``duplicate_count`` holds the sum of ``Clonotype.duplicate_count``.
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
    return list(tokenize_with_positions(raw, k))


def _gapped_kmers(raw: bytes, k: int, mask_byte: int) -> list[tuple[bytes, int]]:
    """Gapped k-mers (each position masked once) with window start positions.

    Delegates to the C extension for the gapped k-mer extraction.
    """
    return list(tokenize_gapped_with_positions(raw, k, mask_byte))


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def tokenize_clonotypes(
    clonotypes: list[Clonotype],
    k: int,
    mask_byte: int | None = None,
) -> dict[Kmer, list[KmerMatch]]:
    """Build an inverted index from annotated k-mers to their source
    clonotypes, tracking the position each token was extracted from.

    When *mask_byte* is given, gapped k-mers are produced instead (each
    position masked once per window, as in :func:`tokens.tokenize_gapped`).

    Args:
        clonotypes: Input clonotypes.
        k: K-mer length.
        mask_byte: If not ``None``, replacement byte for gapped k-mers
            (e.g. ``ord('X')``).  ``None`` (default) produces plain k-mers.

    Returns:
        Dict mapping each :class:`Kmer` to a list of :class:`KmerMatch`
        (clonotype, position) pairs.
    """
    _kmers = _gapped_kmers if mask_byte is not None else _plain_kmers
    _mb = mask_byte
    index: dict[Kmer, list[KmerMatch]] = {}
    for r in clonotypes:
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


def tokenize_rearrangements(
    rearrangements: list[Clonotype],
    k: int,
    mask_byte: int | None = None,
) -> dict[Kmer, list[KmerMatch]]:
    """Deprecated alias for :func:`tokenize_clonotypes`."""
    warnings.warn(
        "tokenize_rearrangements is deprecated; use tokenize_clonotypes.",
        DeprecationWarning,
        stacklevel=2,
    )
    return tokenize_clonotypes(rearrangements, k=k, mask_byte=mask_byte)


def summarize_rearrangements(
    rearrangements: list[Clonotype],
    k: int,
    mask_byte: int | None = None,
) -> dict[Kmer, KmerStats]:
    """Compute per-kmer summary statistics (full :class:`Kmer` key).

    For each :class:`Kmer` the result contains:

    * ``rearrangement_count`` — number of *unique* rearrangement IDs
      contributing that k-mer.
    * ``duplicate_count`` — sum of :attr:`Clonotype.duplicate_count`
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
        seen: set[Kmer] = set()
        for s, _pos in pairs:
            key = Kmer(locus, v_gene, c_gene, s)
            seen.add(key)
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
            dups[key] = dups.get(key, 0) + dc
    return {kmer: KmerStats(counts[kmer], dups[kmer]) for kmer in counts}


def summarize_annotations(
    rearrangements: list[Clonotype],
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
        seen: set[tuple[KmerSeq, KmerAnnotation]] = set()
        for s, pos in pairs:
            ks = KmerSeq(locus, s)
            ka = KmerAnnotation(v_gene, c_gene, pos)
            flat_key = (ks, ka)
            seen.add(flat_key)
        for flat_key in seen:
            counts[flat_key] = counts.get(flat_key, 0) + 1
            dups[flat_key] = dups.get(flat_key, 0) + dc
    # Pivot into nested dict
    result: dict[KmerSeq, dict[KmerAnnotation, KmerStats]] = {}
    for (ks, ka), rc in counts.items():
        inner = result.get(ks)
        if inner is None:
            inner = {}
            result[ks] = inner
        inner[ka] = KmerStats(rc, dups[(ks, ka)])
    return result


def summarize_rearrangements_chunked(
    rearrangements: list[Clonotype],
    k: int,
    mask_byte: int | None = None,
    *,
    chunk_size: int = 100_000,
) -> dict[Kmer, KmerStats]:
    """Chunked variant of :func:`summarize_rearrangements`.

    Useful for very large repertoires when the input can be iterated in
    batches. Produces the same output as the non-chunked function.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    counts: dict[Kmer, int] = {}
    dups: dict[Kmer, int] = {}
    it = iter(rearrangements)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        part = summarize_rearrangements(chunk, k=k, mask_byte=mask_byte)
        for key, stats in part.items():
            counts[key] = counts.get(key, 0) + stats.rearrangement_count
            dups[key] = dups.get(key, 0) + stats.duplicate_count

    return {kmer: KmerStats(counts[kmer], dups[kmer]) for kmer in counts}


def summarize_annotations_chunked(
    rearrangements: list[Clonotype],
    k: int,
    mask_byte: int | None = None,
    *,
    chunk_size: int = 100_000,
) -> dict[KmerSeq, dict[KmerAnnotation, KmerStats]]:
    """Chunked variant of :func:`summarize_annotations`.

    Useful for very large repertoires when the input can be iterated in
    batches. Produces the same output as the non-chunked function.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    counts: dict[tuple[KmerSeq, KmerAnnotation], int] = {}
    dups: dict[tuple[KmerSeq, KmerAnnotation], int] = {}
    it = iter(rearrangements)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        part = summarize_annotations(chunk, k=k, mask_byte=mask_byte)
        for ks, inner in part.items():
            for ka, stats in inner.items():
                flat = (ks, ka)
                counts[flat] = counts.get(flat, 0) + stats.rearrangement_count
                dups[flat] = dups.get(flat, 0) + stats.duplicate_count

    result: dict[KmerSeq, dict[KmerAnnotation, KmerStats]] = {}
    for (ks, ka), rc in counts.items():
        inner = result.get(ks)
        if inner is None:
            inner = {}
            result[ks] = inner
        inner[ka] = KmerStats(rc, dups[(ks, ka)])
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
        table: Output of :func:`tokenize_clonotypes`.
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


def compute_token_tables_batch(
    rearrangements: list[Clonotype],
    families: list[str] | None = None,
) -> dict[str, dict[Kmer, list[KmerMatch]]]:
    """Compute token tables for multiple k-mer families at once.
    
    This is a convenience function for batch extraction of all requested
    token families from a single set of rearrangements, avoiding redundant
    processing of the input data.
    
    Parameters
    ----------
    rearrangements : list[Clonotype]
        Input rearrangements.
    families : list[str] | None
        Token families to compute. Supported values: 'v3', 'pos3', 'u3',
        'u4', 'g4', 'g5'. If None, all six families are computed.
    
    Returns
    -------
    dict[str, dict[Kmer, list[KmerMatch]]]
        Dictionary mapping family name to its token table.
    
    Notes
    -----
    Each family uses different k-mer parameters:
    - 'v3': plain 3-mers
    - 'pos3': plain 3-mers (position-annotated in token_graph)
    - 'u3': ungapped 3-mers
    - 'u4': ungapped 4-mers
    - 'g4': gapped 4-mers (mask_byte=ord('X'))
    - 'g5': gapped 5-mers (mask_byte=ord('X'))
    """
    if families is None:
        families = ['v3', 'pos3', 'u3', 'u4', 'g4', 'g5']

    normalized_families: list[str] = []
    for family in families:
        fam = 'pos3' if family == 'vpos3' else family
        if fam not in {'v3', 'pos3', 'u3', 'u4', 'g4', 'g5'}:
            raise ValueError(f"Unknown token family: {family}")
        normalized_families.append(fam)

    result: dict[str, dict[Kmer, list[KmerMatch]]] = {}

    # Reuse the same plain 3-mer tokenization across v3/pos3/u3.
    plain3: dict[Kmer, list[KmerMatch]] | None = None
    if any(fam in {'v3', 'pos3', 'u3'} for fam in normalized_families):
        plain3 = tokenize_clonotypes(rearrangements, k=3, mask_byte=None)

    for family in normalized_families:
        if family in {'v3', 'pos3', 'u3'}:
            result[family] = plain3 if plain3 is not None else {}
        elif family == 'u4':
            result[family] = tokenize_clonotypes(rearrangements, k=4, mask_byte=None)
        elif family == 'g4':
            result[family] = tokenize_clonotypes(rearrangements, k=4, mask_byte=ord('X'))
        elif family == 'g5':
            result[family] = tokenize_clonotypes(rearrangements, k=5, mask_byte=ord('X'))

    return result
