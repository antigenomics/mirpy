"""Parsing helpers for 10x VDJ v1 single-cell paired-chain workflows."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

from mir.common.clonotype import Clonotype

LOCUS_PAIR_TO_LOCI: dict[str, tuple[str, str]] = {
    "TRA_TRB": ("TRA", "TRB"),
    "TRG_TRD": ("TRG", "TRD"),
    "IGH_IGK": ("IGH", "IGK"),
    "IGH_IGL": ("IGH", "IGL"),
}

_VALID_LOCI = {locus for pair in LOCUS_PAIR_TO_LOCI.values() for locus in pair}


def _to_int(v, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _is_truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _detect_first_existing(header: set[str], candidates: list[str]) -> str:
    for col in candidates:
        if col in header:
            return col
    return ""


def _read_consensus_minimal(path: Path) -> pl.DataFrame:
    """Read consensus fields needed for matched single-cell clonotype rows."""
    header_df = pl.read_csv(
        path,
        separator=",",
        n_rows=0,
        infer_schema_length=0,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
    )
    header = set(header_df.columns)

    cols = ["clonotype_id", "consensus_id", "chain", "cdr3_nt", "reads", "umis"]

    aa_col = _detect_first_existing(header, ["cdr3_aa", "cdr3"])
    if aa_col:
        cols.append(aa_col)

    for optional in ["v_gene", "d_gene", "j_gene", "c_gene", "v_call", "d_call", "j_call", "c_call"]:
        if optional in header:
            cols.append(optional)

    return pl.read_csv(
        path,
        separator=",",
        infer_schema_length=0,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
        columns=cols,
    )


def _read_all_contig_minimal(path: Path, *, check_is_cell: bool = True) -> pl.DataFrame:
    """Read all-contig linkage fields required for cell-to-consensus matching."""
    header_df = pl.read_csv(
        path,
        separator=",",
        n_rows=0,
        infer_schema_length=0,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
    )
    header = set(header_df.columns)

    cols = ["barcode", "raw_clonotype_id", "raw_consensus_id"]
    if check_is_cell and "is_cell" in header:
        cols.append("is_cell")
    if "cdr3_nt" in header:
        cols.append("cdr3_nt")

    return pl.read_csv(
        path,
        separator=",",
        infer_schema_length=0,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
        columns=cols,
    )


def _build_consensus_lookup(consensus_df: pl.DataFrame) -> dict[tuple[str, str], Clonotype]:
    lookup: dict[tuple[str, str], Clonotype] = {}

    for row in consensus_df.iter_rows(named=True):
        pair_id = str(row.get("clonotype_id") or "").strip()
        sequence_id = str(row.get("consensus_id") or "").strip()
        locus = str(row.get("chain") or "").strip().upper()
        if not pair_id or not sequence_id or locus not in _VALID_LOCI:
            continue

        aa = str(row.get("cdr3_aa") or row.get("cdr3") or "").strip()
        lookup[(pair_id, sequence_id)] = Clonotype(
            _validate=False,
            sequence_id=sequence_id,
            duplicate_count=_to_int(row.get("reads"), 0),
            umi_count=_to_int(row.get("umis"), 0),
            locus=locus,
            junction=str(row.get("cdr3_nt") or "").strip(),
            junction_aa=aa,
            v_gene=str(row.get("v_gene") or row.get("v_call") or "").strip(),
            d_gene=str(row.get("d_gene") or row.get("d_call") or "").strip(),
            j_gene=str(row.get("j_gene") or row.get("j_call") or "").strip(),
            c_gene=str(row.get("c_gene") or row.get("c_call") or "").strip(),
        )
    return lookup


def _warn_if_donor_mismatch(
    consensus_lookup: dict[tuple[str, str], Clonotype],
    all_contig_rows: list[dict[str, str]],
    *,
    mismatch_threshold: float = 0.10,
) -> None:
    """Warn when cdr3_nt disagrees for matched consensus and all-contig keys."""
    checked = 0
    mismatched = 0
    mismatch_examples: list[str] = []

    for row in all_contig_rows:
        ac_junc = str(row.get("cdr3_nt") or "").strip()
        if not ac_junc:
            continue

        pair_id = row["raw_clonotype_id"]
        sequence_id = row["raw_consensus_id"]
        clonotype = consensus_lookup.get((pair_id, sequence_id))
        if clonotype is None:
            continue

        cons_junc = clonotype.junction
        if not cons_junc:
            continue

        checked += 1
        if ac_junc != cons_junc:
            mismatched += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append(
                    f"  consensus_id={sequence_id!r}: "
                    f"consensus={cons_junc!r} vs all_contig={ac_junc!r}"
                )

    if checked > 0 and mismatched / checked > mismatch_threshold:
        example_str = "\n".join(mismatch_examples)
        warnings.warn(
            f"Possible donor mismatch: {mismatched}/{checked} checked contigs have "
            f"different cdr3_nt in consensus_annotations vs all_contig_annotations "
            f"({100 * mismatched / checked:.1f}% > threshold "
            f"{100 * mismatch_threshold:.0f}%). "
            f"Verify that both files belong to the same donor.\n"
            f"Example mismatches:\n{example_str}",
            UserWarning,
            stacklevel=3,
        )


def load_10x_vdj_v1_cell_clonotypes(
    consensus_annotations_path: str | Path,
    all_contig_annotations_path: str | Path,
    *,
    donor_id: str = "",
    check_is_cell: bool = True,
) -> pl.DataFrame:
    """Load matched per-cell clonotype rows from 10x VDJ v1 files.

    Returns a table with one row per matched (barcode, raw_pair_id, sequence_id)
    after optional is_cell filtering.
    """
    consensus_path = Path(consensus_annotations_path)
    all_contig_path = Path(all_contig_annotations_path)
    if not donor_id:
        donor_id = consensus_path.name

    consensus_df = _read_consensus_minimal(consensus_path)
    all_contig_df = _read_all_contig_minimal(all_contig_path, check_is_cell=check_is_cell)
    consensus_lookup = _build_consensus_lookup(consensus_df)

    filtered_all_contig_rows: list[dict[str, str]] = []
    has_is_cell = "is_cell" in all_contig_df.columns
    for row in all_contig_df.iter_rows(named=True):
        if check_is_cell and has_is_cell and not _is_truthy(row.get("is_cell")):
            continue

        barcode = str(row.get("barcode") or "").strip()
        pair_id = str(row.get("raw_clonotype_id") or "").strip()
        sequence_id = str(row.get("raw_consensus_id") or "").strip()
        if not barcode or not pair_id or not sequence_id:
            continue

        filtered_all_contig_rows.append(
            {
                "barcode": barcode,
                "raw_clonotype_id": pair_id,
                "raw_consensus_id": sequence_id,
                "cdr3_nt": str(row.get("cdr3_nt") or "").strip(),
            }
        )

    _warn_if_donor_mismatch(consensus_lookup, filtered_all_contig_rows)

    out_rows: list[dict[str, str | int]] = []
    for row in filtered_all_contig_rows:
        pair_id = row["raw_clonotype_id"]
        sequence_id = row["raw_consensus_id"]
        clonotype = consensus_lookup.get((pair_id, sequence_id))
        if clonotype is None:
            continue

        out_rows.append(
            {
                "donor_id": donor_id,
                "barcode": row["barcode"],
                "raw_pair_id": pair_id,
                "sequence_id": clonotype.sequence_id,
                "locus": clonotype.locus,
                "duplicate_count": clonotype.duplicate_count,
                "umi_count": clonotype.umi_count,
                "junction": clonotype.junction,
                "junction_aa": clonotype.junction_aa,
                "v_gene": clonotype.v_gene,
                "d_gene": clonotype.d_gene,
                "j_gene": clonotype.j_gene,
                "c_gene": clonotype.c_gene,
            }
        )

    schema = {
        "donor_id": pl.Utf8,
        "barcode": pl.Utf8,
        "raw_pair_id": pl.Utf8,
        "sequence_id": pl.Utf8,
        "locus": pl.Utf8,
        "duplicate_count": pl.Int64,
        "umi_count": pl.Int64,
        "junction": pl.Utf8,
        "junction_aa": pl.Utf8,
        "v_gene": pl.Utf8,
        "d_gene": pl.Utf8,
        "j_gene": pl.Utf8,
        "c_gene": pl.Utf8,
    }
    if not out_rows:
        return pl.DataFrame(schema=schema)

    return pl.from_dicts(out_rows, schema=schema)
