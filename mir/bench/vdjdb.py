"""Load VDJdb dumps into the AIRR polars frame TCREmp expects.

VDJdb "slim" dumps use dotted column names (``cdr3``, ``v.segm``, ``j.segm``,
``antigen.epitope``, ``gene``, ``complex.id``, ``mhc.class``). This maps them to
the ``vdjtools.io.schema`` names plus an ``epitope`` label column.
"""

from __future__ import annotations

import polars as pl


def load_vdjdb(path: str) -> pl.DataFrame:
    """Return an AIRR frame with ``v_call, j_call, junction_aa, locus, epitope``.

    Args:
        path: VDJdb slim TSV (optionally gzipped).
    """
    raw = pl.read_csv(path, separator="\t", infer_schema_length=0)  # all Utf8
    return (
        raw.select(
            junction_aa=pl.col("cdr3"),
            v_call=pl.col("v.segm"),
            j_call=pl.col("j.segm"),
            locus=pl.col("gene"),
            epitope=pl.col("antigen.epitope"),
            mhc_class=pl.col("mhc.class"),
        )
        .filter(
            pl.col("junction_aa").is_not_null()
            & (pl.col("junction_aa").str.len_chars() >= 5)
            & pl.col("epitope").is_not_null()
            & pl.col("v_call").is_not_null()
            & pl.col("j_call").is_not_null()
        )
        .unique(subset=["junction_aa", "v_call", "j_call", "locus", "epitope"])
    )


def antigen_subset(df: pl.DataFrame, chain: str, min_records: int) -> pl.DataFrame:
    """Rows for *chain* restricted to epitopes with ``>= min_records`` records."""
    sub = df.filter(pl.col("locus") == chain)
    keep = (
        sub.group_by("epitope").len().filter(pl.col("len") >= min_records)["epitope"].to_list()
    )
    return sub.filter(pl.col("epitope").is_in(keep))
