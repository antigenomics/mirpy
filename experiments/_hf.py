"""Shared helpers for the density benchmarks: fetch + normalize isalgo AIRR datasets.

The benchmark repertoires live on HuggingFace (``isalgo/airr_{benchmark,yfv19,ankspond}``)
in five different header conventions (AIRR, VDJtools, MiXCR, ...). :func:`load_repertoire`
sniffs the columns and returns a uniform ``junction_aa / v_call / j_call / duplicate_count``
frame ready for ``TCREmp.embed``. Files are large LFS objects — always subsample with ``top``.
Needs ``huggingface_hub`` (the ``[bench]`` extra).
"""

from __future__ import annotations

import polars as pl

# column synonyms across AIRR / VDJtools / MiXCR exports. Note: the anchored *junction* is
# wanted, not the anchor-excluded AIRR `cdr3` — so bare `cdr3` is deliberately excluded.
_JUNC = ["junction_aa", "cdr3aa", "aaSeqCDR3", "CDR3.amino.acid.sequence", "CDR3aa"]
_V = ["v_call", "v_gene", "v", "V.gene", "bestVGene", "v.segm", "V"]
_J = ["j_call", "j_gene", "j", "J.gene", "bestJGene", "j.segm", "J"]
_CNT = ["duplicate_count", "count", "Read.count", "#count", "cloneCount", "reads", "cloneFraction"]
_CANON = r"^C[ACDEFGHIKLMNPQRSTVWY]*[FW]$"  # canonical CDR3: Cys … Phe/Trp, amino acids only


def fetch(repo_id: str, filename: str) -> str:
    """Download (and cache) one file from a HuggingFace dataset; return its local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


def _pick(cols: list[str], names: list[str]) -> str | None:
    return next((n for n in names if n in cols), None)


def load_repertoire(path: str, *, top: int | None = None) -> pl.DataFrame:
    """Load a repertoire file into a uniform AIRR frame, collapsing duplicate clonotypes.

    Args:
        path: Local path to a ``.tsv.gz`` / ``.txt.gz`` repertoire.
        top: Keep only the ``top`` most-abundant clonotypes (by summed count). ``None``
            keeps all — only do that for small files.

    Returns:
        ``junction_aa / v_call / j_call / duplicate_count`` frame (canonical CDR3s only,
        one row per unique ``(junction_aa, v_call, j_call)``).
    """
    df = pl.read_csv(path, separator="\t", infer_schema_length=0)  # all Utf8, robust to mixed types
    jc = _pick(df.columns, _JUNC)
    vc = _pick(df.columns, _V)
    jj = _pick(df.columns, _J)
    cc = _pick(df.columns, _CNT)
    if jc is None or vc is None or jj is None:
        raise ValueError(f"{path}: could not map junction/v/j columns from {df.columns}")
    count = (
        pl.col(cc).cast(pl.Float64, strict=False).fill_null(1.0)
        if cc is not None else pl.lit(1.0)
    )
    out = (
        df.select(
            pl.col(jc).alias("junction_aa"),
            pl.col(vc).alias("v_call"),
            pl.col(jj).alias("j_call"),
            count.alias("duplicate_count"),
        )
        .filter(pl.col("junction_aa").str.contains(_CANON))
        .group_by(["junction_aa", "v_call", "j_call"])
        .agg(pl.col("duplicate_count").sum())
    )
    # polars group_by output order is non-deterministic; sort so a seeded .sample() downstream
    # is reproducible.
    if top is not None:
        return out.sort(["duplicate_count", "junction_aa"], descending=[True, False]).head(top)
    return out.sort(["junction_aa", "v_call", "j_call"])
