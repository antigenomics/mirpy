"""Single-cell paired-chain repertoire structures and 10x VDJ v1 loader.

This module provides a lightweight paired-clonotype model for cell-level
analysis while keeping barcode mappings separate for downstream multimodal
integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import polars as pl

from mir.common.clonotype import Clonotype

LOCUS_PAIR_TO_LOCI: dict[str, tuple[str, str]] = {
    "TRA_TRB": ("TRA", "TRB"),
    "TRG_TRD": ("TRG", "TRD"),
    "IGH_IGK": ("IGH", "IGK"),
    "IGH_IGL": ("IGH", "IGL"),
}


@dataclass(frozen=True, slots=True)
class PairedClonotype:
    """A paired clonotype tuple with a stable pair identifier."""

    pair_id: str
    clonotype1: Clonotype
    clonotype2: Clonotype


@dataclass(slots=True)
class PairedLocusRepertoire:
    """Collection of paired clonotypes for one locus-pair family."""

    locus_pair: str
    paired_clonotypes: list[PairedClonotype]

    def __post_init__(self) -> None:
        if self.locus_pair not in LOCUS_PAIR_TO_LOCI:
            raise ValueError(
                f"Unsupported locus_pair {self.locus_pair!r}; "
                f"expected one of {sorted(LOCUS_PAIR_TO_LOCI)}"
            )

    @property
    def clonotype_count(self) -> int:
        return len(self.paired_clonotypes)


@dataclass(slots=True)
class SingleCellRepertoire:
    """Cell barcode to paired-clonotype links for multimodal integration."""

    barcode_pair_ids: list[tuple[str, str]]

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "barcode": [x[0] for x in self.barcode_pair_ids],
                "pair_id": [x[1] for x in self.barcode_pair_ids],
            },
            schema={"barcode": pl.Utf8, "pair_id": pl.Utf8},
        )


@dataclass(slots=True)
class TenXVdjV1DonorData:
    """Assembled 10x VDJ v1 donor-level paired-chain data."""

    donor_id: str
    single_cell_repertoire: SingleCellRepertoire
    paired_locus_repertoires: dict[str, PairedLocusRepertoire]
    chain_multiplicity: pl.DataFrame
    loaded_cell_count: int
    loaded_clonotype_count: int


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


def _build_consensus_lookup(consensus_df: pl.DataFrame) -> dict[tuple[str, str], Clonotype]:
    lookup: dict[tuple[str, str], Clonotype] = {}
    valid_loci = {locus for pair in LOCUS_PAIR_TO_LOCI.values() for locus in pair}

    for row in consensus_df.iter_rows(named=True):
        pair_id = str(row.get("clonotype_id") or "").strip()
        sequence_id = str(row.get("consensus_id") or "").strip()
        locus = str(row.get("chain") or "").strip().upper()
        if not pair_id or not sequence_id or locus not in valid_loci:
            continue
        lookup[(pair_id, sequence_id)] = Clonotype(
            _validate=False,
            sequence_id=sequence_id,
            duplicate_count=_to_int(row.get("reads"), 0),
            umi_count=_to_int(row.get("umis"), 0),
            locus=locus,
            junction=str(row.get("cdr3_nt") or "").strip(),
            junction_aa=str(row.get("cdr3_aa") or row.get("cdr3") or "").strip(),
        )
    return lookup


def _read_consensus_minimal(path: Path) -> pl.DataFrame:
    """Read only consensus fields required for paired-chain assembly."""
    header = pl.read_csv(
        path,
        separator=",",
        n_rows=0,
        infer_schema_length=0,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
    )
    aa_col = "cdr3_aa" if "cdr3_aa" in header.columns else "cdr3"
    return pl.read_csv(
        path,
        separator=",",
        infer_schema_length=0,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
        columns=["clonotype_id", "consensus_id", "chain", "cdr3_nt", aa_col, "reads", "umis"],
    )


def _read_all_contig_minimal(path: Path) -> pl.DataFrame:
    """Read only all-contig fields required for cell-to-chain linkage."""
    return pl.read_csv(
        path,
        separator=",",
        infer_schema_length=0,
        null_values=["", "NA"],
        truncate_ragged_lines=True,
        columns=["is_cell", "barcode", "raw_clonotype_id", "raw_consensus_id"],
    )


def load_10x_vdj_v1_donor(
    consensus_annotations_path: str | Path,
    all_contig_annotations_path: str | Path,
    donor_id: str = "",
) -> TenXVdjV1DonorData:
    """Load one donor from 10x_vdj_v1 files into paired single-cell structures."""
    consensus_path = Path(consensus_annotations_path)
    all_contig_path = Path(all_contig_annotations_path)
    if not donor_id:
        donor_id = consensus_path.name

    consensus_df = _read_consensus_minimal(consensus_path)
    all_contig_df = _read_all_contig_minimal(all_contig_path)

    consensus_lookup = _build_consensus_lookup(consensus_df)

    filtered_all_contig_rows: list[dict[str, str]] = []
    for row in all_contig_df.iter_rows(named=True):
        if not _is_truthy(row.get("is_cell")):
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
            }
        )

    grouped: dict[tuple[str, str], dict[str, dict[str, Clonotype]]] = {}
    matched_clonotype_keys: set[tuple[str, str]] = set()
    for row in filtered_all_contig_rows:
        barcode = row["barcode"]
        pair_id = row["raw_clonotype_id"]
        sequence_id = row["raw_consensus_id"]

        clonotype = consensus_lookup.get((pair_id, sequence_id))
        if clonotype is None:
            continue
        matched_clonotype_keys.add((pair_id, sequence_id))

        key = (barcode, pair_id)
        if key not in grouped:
            grouped[key] = {}
        if clonotype.locus not in grouped[key]:
            grouped[key][clonotype.locus] = {}
        grouped[key][clonotype.locus][clonotype.sequence_id] = clonotype

    barcode_pair_ids: list[tuple[str, str]] = []
    paired_by_family: dict[str, list[PairedClonotype]] = {
        name: [] for name in LOCUS_PAIR_TO_LOCI
    }
    multiplicity_rows: list[dict[str, object]] = []

    for (barcode, raw_pair_id), loci_map in grouped.items():
        for locus_pair, (locus1, locus2) in LOCUS_PAIR_TO_LOCI.items():
            chains1 = sorted(loci_map.get(locus1, {}).values(), key=lambda x: x.sequence_id)
            chains2 = sorted(loci_map.get(locus2, {}).values(), key=lambda x: x.sequence_id)
            n_chain1 = len(chains1)
            n_chain2 = len(chains2)
            multiplicity_rows.append(
                {
                    "donor_id": donor_id,
                    "barcode": barcode,
                    "raw_pair_id": raw_pair_id,
                    "locus_pair": locus_pair,
                    "n_chain1": n_chain1,
                    "m_chain2": n_chain2,
                }
            )
            if n_chain1 == 0 or n_chain2 == 0:
                continue

            pairs = list(product(chains1, chains2))
            use_suffix = len(pairs) > 1
            for idx, (chain1, chain2) in enumerate(pairs, start=1):
                pair_id = f"{raw_pair_id}_{idx}" if use_suffix else raw_pair_id
                paired_by_family[locus_pair].append(
                    PairedClonotype(pair_id=pair_id, clonotype1=chain1, clonotype2=chain2)
                )
                barcode_pair_ids.append((barcode, pair_id))

    multiplicity_df = pl.DataFrame(multiplicity_rows)
    if multiplicity_df.height == 0:
        chain_multiplicity = pl.DataFrame(
            schema={
                "donor_id": pl.Utf8,
                "locus_pair": pl.Utf8,
                "n_chain1": pl.Int64,
                "m_chain2": pl.Int64,
                "cell_count": pl.Int64,
            }
        )
    else:
        chain_multiplicity = (
            multiplicity_df.group_by(["donor_id", "locus_pair", "n_chain1", "m_chain2"])
            .len()
            .rename({"len": "cell_count"})
            .sort(["donor_id", "locus_pair", "n_chain1", "m_chain2"])
        )

    paired_locus_repertoires = {
        locus_pair: PairedLocusRepertoire(locus_pair=locus_pair, paired_clonotypes=paired)
        for locus_pair, paired in paired_by_family.items()
    }

    return TenXVdjV1DonorData(
        donor_id=donor_id,
        single_cell_repertoire=SingleCellRepertoire(barcode_pair_ids=barcode_pair_ids),
        paired_locus_repertoires=paired_locus_repertoires,
        chain_multiplicity=chain_multiplicity,
        loaded_cell_count=len(grouped),
        loaded_clonotype_count=len(matched_clonotype_keys),
    )
