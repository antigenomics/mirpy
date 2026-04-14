"""Polars-based rearrangement k-mer indexing and summarisation.

Mirrors the object-based API in :mod:`token_tables` using Polars
DataFrames.  The rearrangement table has columns:

    ``id`` (Int64), ``locus`` (Utf8), ``v_gene`` (Utf8),
    ``c_gene`` (Utf8), ``junction_aa`` (Utf8), ``duplicate_count`` (Int64).

Functions
---------
* ``expand_kmers``           — Expand each rearrangement row into one row
  per k-mer, adding ``kmer_pos`` and ``kmer_seq`` columns.
* ``summarize_by_gene``      — Group by (locus, v_gene, c_gene, kmer_seq)
  → rearrangement_count, duplicate_count.
* ``summarize_by_pos``       — Group by (locus, kmer_seq, kmer_pos).
* ``summarize_by_v``         — Group by (locus, kmer_seq, v_gene).
* ``summarize_by_c``         — Group by (locus, kmer_seq, c_gene).
* ``fetch_by_kmer``          — Rows from the original table matching
  (locus, kmer_seq).
* ``fetch_by_annotated_kmer``— Rows matching (locus, v_gene, c_gene, kmer_seq).
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
    ``kmer_seq`` (``Utf8``).  Rearrangements shorter than *k* are dropped.

    Args:
        df: Rearrangement table with at least ``id``, ``locus``,
            ``v_gene``, ``c_gene``, ``junction_aa``, ``duplicate_count``.
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
    return (
        expanded
        .group_by(group_cols)
        .agg(
            pl.col("id").n_unique().alias("rearrangement_count"),
            pl.col("duplicate_count").sum().alias("duplicate_count"),
        )
    )


def summarize_by_gene(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, v_gene, c_gene, kmer_seq).

    Returns columns: locus, v_gene, c_gene, kmer_seq,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "v_gene", "c_gene", "kmer_seq"])


def summarize_by_pos(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, kmer_seq, kmer_pos).

    Returns columns: locus, kmer_seq, kmer_pos,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "kmer_seq", "kmer_pos"])


def summarize_by_v(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, kmer_seq, v_gene).

    Returns columns: locus, kmer_seq, v_gene,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "kmer_seq", "v_gene"])


def summarize_by_c(expanded: pl.DataFrame) -> pl.DataFrame:
    """Group by (locus, kmer_seq, c_gene).

    Returns columns: locus, kmer_seq, c_gene,
    rearrangement_count, duplicate_count.
    """
    return _summarize(expanded, ["locus", "kmer_seq", "c_gene"])


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
    v_gene: str,
    c_gene: str,
    kmer_seq: str,
) -> pl.DataFrame:
    """Return rows from the original rearrangement table matching a fully
    annotated k-mer query (locus, v_gene, c_gene, kmer_seq).

    Args:
        df: Original rearrangement table.
        expanded: Expanded k-mer table (from :func:`expand_kmers`).
        locus: Locus string to match.
        v_gene: V-gene name to match.
        c_gene: C-gene name to match.
        kmer_seq: K-mer sequence string to match.

    Returns:
        Subset of *df* (original columns only, deduplicated by ``id``).
    """
    ids = (
        expanded
        .filter(
            (pl.col("locus") == locus)
            & (pl.col("v_gene") == v_gene)
            & (pl.col("c_gene") == c_gene)
            & (pl.col("kmer_seq") == kmer_seq)
        )
        .select("id")
        .unique()
    )
    return df.join(ids, on="id", how="inner")
