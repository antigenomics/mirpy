"""Single-cell paired-chain repertoire structures and 10x VDJ v1 loader."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import polars as pl

from mir.common.clonotype import Clonotype
from mir.common.single_cell_parser import (
    LOCUS_PAIR_TO_LOCI,
    load_10x_vdj_v1_cell_clonotypes,
)


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


def _cell_row_to_clonotype(row: dict[str, object]) -> Clonotype:
    return Clonotype(
        _validate=False,
        sequence_id=str(row.get("sequence_id") or ""),
        duplicate_count=int(row.get("duplicate_count") or 0),
        umi_count=int(row.get("umi_count") or 0),
        locus=str(row.get("locus") or ""),
        junction=str(row.get("junction") or ""),
        junction_aa=str(row.get("junction_aa") or ""),
        v_gene=str(row.get("v_gene") or ""),
        d_gene=str(row.get("d_gene") or ""),
        j_gene=str(row.get("j_gene") or ""),
        c_gene=str(row.get("c_gene") or ""),
    )


def build_tenx_donor_from_cell_clonotypes(
    cell_clonotypes_df: pl.DataFrame,
    *,
    donor_id: str,
) -> TenXVdjV1DonorData:
    """Assemble donor-level paired structures from matched per-cell clonotype rows."""
    grouped: dict[tuple[str, str], dict[str, dict[str, Clonotype]]] = {}
    matched_clonotype_keys: set[tuple[str, str]] = set()

    for row in cell_clonotypes_df.iter_rows(named=True):
        barcode = str(row.get("barcode") or "").strip()
        raw_pair_id = str(row.get("raw_pair_id") or "").strip()
        sequence_id = str(row.get("sequence_id") or "").strip()
        if not barcode or not raw_pair_id or not sequence_id:
            continue

        clonotype = _cell_row_to_clonotype(row)
        if not clonotype.locus:
            continue

        matched_clonotype_keys.add((raw_pair_id, sequence_id))
        key = (barcode, raw_pair_id)
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


def load_10x_vdj_v1_donor(
    consensus_annotations_path: str | Path,
    all_contig_annotations_path: str | Path,
    donor_id: str = "",
    *,
    check_is_cell: bool = True,
) -> TenXVdjV1DonorData:
    """Load one donor from 10x_vdj_v1 files into paired single-cell structures.

    Args:
        consensus_annotations_path: Path to donor consensus_annotations CSV(.gz).
        all_contig_annotations_path: Path to donor all_contig_annotations CSV(.gz).
        donor_id: Optional donor identifier. Defaults to consensus filename.
        check_is_cell: If True (default), only keep rows where is_cell is truthy.
    """
    consensus_path = Path(consensus_annotations_path)
    if not donor_id:
        donor_id = consensus_path.name
    cell_df = load_10x_vdj_v1_cell_clonotypes(
        consensus_annotations_path=consensus_annotations_path,
        all_contig_annotations_path=all_contig_annotations_path,
        donor_id=donor_id,
        check_is_cell=check_is_cell,
    )
    return build_tenx_donor_from_cell_clonotypes(cell_df, donor_id=donor_id)
