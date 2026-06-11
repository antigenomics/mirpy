"""Polars-based rearrangement k-mer indexing and summarisation.

Mirrors the object-based API in :mod:`token_tables` using Polars
DataFrames.  The rearrangement table has columns:

    ``id`` (Int64), ``locus`` (Utf8), ``v_call`` (Utf8),
    ``c_call`` (Utf8), ``junction_aa`` (Utf8), ``duplicate_count`` (Int64).

Functions
---------
* ``expand_kmers``           — Expand each rearrangement row into one row
  per k-mer, adding ``kmer_pos`` and ``kmer_seq`` columns.
* ``summarize_by_gene``      — Group by (locus, v_call, c_call, kmer_seq)
  → rearrangement_count, duplicate_count.
* ``summarize_by_pos``       — Group by (locus, kmer_seq, kmer_pos).
* ``summarize_by_v``         — Group by (locus, kmer_seq, v_call).
* ``summarize_by_c``         — Group by (locus, kmer_seq, c_call).
* ``fetch_by_kmer``          — Rows from the original table matching
  (locus, kmer_seq).
* ``fetch_by_annotated_kmer``— Rows matching (locus, v_call, c_call, kmer_seq).
"""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# K-mer expansion
# ---------------------------------------------------------------------------

def expand_kmers(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """Expand rearrangement table: one row per overlapping k-mer.

    For each rearrangement with ``junction_aa`` of length *n ≥ k*, produces
    *n − k + 1* rows with new columns ``kmer_pos`` (``Int64``) and
    ``kmer_seq`` (``Utf8``).  Clonotypes shorter than *k* are dropped.

    Args:
        df: Clonotype table with at least ``id``, ``locus``,
            ``v_call``, ``c_call``, ``junction_aa``, ``duplicate_count``.
        k:  K-mer length.

    Returns:
        Expanded :class:`polars.DataFrame`.
    """
    jlen = df["junction_aa"].str.len_chars()
    df_valid = df.filter(jlen >= k)
    if df_valid.height == 0:
        return df_valid.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("kmer_pos"),
            pl.lit(None, dtype=pl.Utf8).alias("kmer_seq"),
        )
    n_kmers = df_valid["junction_aa"].str.len_chars() - k + 1
    df_with_n = df_valid.with_columns(n_kmers.alias("_n_kmers"))
    # Repeat each row n_kmers times, then assign positions
    rows = df_with_n.with_columns(
        pl.col("_n_kmers").map_elements(
            lambda n: list(range(n)), return_dtype=pl.List(pl.Int64)
        ).alias("kmer_pos")
    ).explode("kmer_pos").drop("_n_kmers")
    # Extract k-mer at each position
    rows = rows.with_columns(
        pl.col("junction_aa").str.slice(
            pl.col("kmer_pos").cast(pl.UInt32), k
        ).alias("kmer_seq")
    )
    return rows


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def _summarize(expanded: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Group *expanded* by *group_cols* and compute summary stats."""
    unique = expanded.select(group_cols + ["id", "duplicate_count"]).unique()
    return (
        unique
        .group_by(group_cols)
        .agg(
            pl.col("id").n_unique().alias("rearrangement_count"),
            pl.col("duplicate_count").sum().alias("duplicate_count"),
        )
    )


def _summarize_chunked(
    df: pl.DataFrame,
    k: int,
    *,
    group_cols: list[str],
    chunk_size: int,
) -> pl.DataFrame:
    """Chunked summary helper to avoid full expanded-table materialization."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if df.height == 0:
        return pl.DataFrame(
            {
                **{col: [] for col in group_cols},
                "rearrangement_count": [],
                "duplicate_count": [],
            }
        )

    parts: list[pl.DataFrame] = []
    for start in range(0, df.height, chunk_size):
        chunk = df.slice(start, chunk_size)
        expanded = expand_kmers(chunk, k)
        if expanded.height == 0:
            continue
        parts.append(_summarize(expanded, group_cols))

    if not parts:
        return pl.DataFrame(
            {
                **{col: [] for col in group_cols},
                "rearrangement_count": [],
                "duplicate_count": [],
            }
        )

    merged = pl.concat(parts, how="vertical")
    return (
        merged
        .group_by(group_cols)
        .agg(
            pl.col("rearrangement_count").sum().alias("rearrangement_count"),
            pl.col("duplicate_count").sum().alias("duplicate_count"),
        )
    )


def summarize_by_gene(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, v_call, c_call, kmer_seq).

    Returns columns: locus, v_call, c_call, kmer_seq,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "v_call", "c_call", "kmer_seq"])


def summarize_by_gene_chunked(df: pl.DataFrame, k: int, *, chunk_size: int = 100_000) -> pl.DataFrame:
    """Chunked summary by (locus, v_call, c_call, kmer_seq)."""
    return _summarize_chunked(
        df,
        k,
        group_cols=["locus", "v_call", "c_call", "kmer_seq"],
        chunk_size=chunk_size,
    )


def summarize_by_pos(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, kmer_seq, kmer_pos).

    Returns columns: locus, kmer_seq, kmer_pos,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "kmer_seq", "kmer_pos"])


def summarize_by_pos_chunked(df: pl.DataFrame, k: int, *, chunk_size: int = 100_000) -> pl.DataFrame:
    """Chunked summary by (locus, kmer_seq, kmer_pos)."""
    return _summarize_chunked(
        df,
        k,
        group_cols=["locus", "kmer_seq", "kmer_pos"],
        chunk_size=chunk_size,
    )


def summarize_by_v(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, kmer_seq, v_call).

    Returns columns: locus, kmer_seq, v_call,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "kmer_seq", "v_call"])


def summarize_by_v_chunked(df: pl.DataFrame, k: int, *, chunk_size: int = 100_000) -> pl.DataFrame:
    """Chunked summary by (locus, kmer_seq, v_call)."""
    return _summarize_chunked(
        df,
        k,
        group_cols=["locus", "kmer_seq", "v_call"],
        chunk_size=chunk_size,
    )


def summarize_by_c(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, kmer_seq, c_call).

    Returns columns: locus, kmer_seq, c_call,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "kmer_seq", "c_call"])


def summarize_by_c_chunked(df: pl.DataFrame, k: int, *, chunk_size: int = 100_000) -> pl.DataFrame:
    """Chunked summary by (locus, kmer_seq, c_call)."""
    return _summarize_chunked(
        df,
        k,
        group_cols=["locus", "kmer_seq", "c_call"],
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_by_kmer(
    df: pl.DataFrame,
    expanded: pl.DataFrame,
    locus: str,
    kmer_seq: str,
) -> pl.DataFrame:
    """Return rows from the original rearrangement table whose
    ``junction_aa`` contains the given k-mer at the specified locus.

    Args:
        df: Original rearrangement table.
        expanded: Expanded k-mer table (from :func:`expand_kmers`).
        locus: Locus string to match.
        kmer_seq: K-mer sequence string to match.

    Returns:
        Subset of *df* (original columns only, deduplicated by ``id``).
    """
    ids = (
        expanded
        .filter(
            (pl.col("locus") == locus) & (pl.col("kmer_seq") == kmer_seq)
        )
        .select("id")
        .unique()
    )
    return df.join(ids, on="id", how="inner")


def fetch_by_annotated_kmer(
    df: pl.DataFrame,
    expanded: pl.DataFrame,
    locus: str,
    v_call: str,
    c_call: str,
    kmer_seq: str,
) -> pl.DataFrame:
    """Return rows from the original rearrangement table matching a fully
    annotated k-mer query (locus, v_call, c_call, kmer_seq).

    Args:
        df: Original rearrangement table.
        expanded: Expanded k-mer table (from :func:`expand_kmers`).
        locus: Locus string to match.
        v_call: V-gene name to match.
        c_call: C-gene name to match.
        kmer_seq: K-mer sequence string to match.

    Returns:
        Subset of *df* (original columns only, deduplicated by ``id``).
    """
    ids = (
        expanded
        .filter(
            (pl.col("locus") == locus)
            & (pl.col("v_call") == v_call)
            & (pl.col("c_call") == c_call)
            & (pl.col("kmer_seq") == kmer_seq)
        )
        .select("id")
        .unique()
    )
    return df.join(ids, on="id", how="inner")
