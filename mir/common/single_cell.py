"""Single-cell paired-chain repertoire structures and 10x VDJ v1 loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import polars as pl

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.single_cell_parser import LOCUS_PAIR_TO_LOCI, load_10x_vdj_v1_cell_clonotypes


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
    barcode_metadata: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "barcode": [x[0] for x in self.barcode_pair_ids],
                "pair_id": [x[1] for x in self.barcode_pair_ids],
            },
            schema={"barcode": pl.Utf8, "pair_id": pl.Utf8},
        )

    def metadata_to_polars(self) -> pl.DataFrame:
        """Return per-barcode metadata as a tabular view."""
        if not self.barcode_metadata:
            return pl.DataFrame(schema={"barcode": pl.Utf8})

        keys = sorted({key for meta in self.barcode_metadata.values() for key in meta})
        rows = []
        for barcode, meta in sorted(self.barcode_metadata.items()):
            row = {"barcode": barcode}
            row.update({key: meta.get(key, "") for key in keys})
            rows.append(row)
        return pl.DataFrame(rows)


class _LazyClonotypeIndex:
    """Lazy clonotype lookup keyed by (locus, sequence_id)."""

    def __init__(self, sample_repertoire: SampleRepertoire) -> None:
        self._sample_repertoire = sample_repertoire
        self._index: dict[str, dict[str, Clonotype]] | None = None

    def _materialize(self) -> dict[str, dict[str, Clonotype]]:
        if self._index is not None:
            return self._index
        built: dict[str, dict[str, Clonotype]] = {}
        for locus, loc_rep in self._sample_repertoire.loci.items():
            built[locus] = {c.sequence_id: c for c in loc_rep.clonotypes}
        self._index = built
        return built

    def get(self, locus: str, sequence_id: str) -> Clonotype | None:
        return self._materialize().get(locus, {}).get(sequence_id)


@dataclass(slots=True)
class PairedRepertoire:
    """Paired-chain sample object with optional barcode pair linkage."""

    sample_id: str
    single_cell_repertoire: SingleCellRepertoire
    paired_locus_repertoires: dict[str, PairedLocusRepertoire]
    chain_multiplicity: pl.DataFrame
    loaded_cell_count: int
    loaded_clonotype_count: int
    _clonotype_lookup: dict[str, dict[str, Clonotype]] | None = field(default=None, init=False, repr=False)

    @property
    def clonotype_count(self) -> int:
        return self.loaded_clonotype_count

    def to_sample_repertoire(self) -> SampleRepertoire:
        """Collapse paired representation into plain per-locus repertoires."""
        per_locus: dict[str, dict[str, Clonotype]] = {}
        for locus_pair in self.paired_locus_repertoires.values():
            for pair in locus_pair.paired_clonotypes:
                for clonotype in (pair.clonotype1, pair.clonotype2):
                    per_locus.setdefault(clonotype.locus, {})[clonotype.sequence_id] = clonotype

        loci = {
            locus: LocusRepertoire(clonotypes=list(by_id.values()), locus=locus)
            for locus, by_id in per_locus.items()
        }
        return SampleRepertoire(loci=loci, sample_id=self.sample_id)

    def _materialize_clonotype_lookup(self) -> dict[str, dict[str, Clonotype]]:
        if self._clonotype_lookup is not None:
            return self._clonotype_lookup
        index: dict[str, dict[str, Clonotype]] = {}
        for pair_rep in self.paired_locus_repertoires.values():
            for pair in pair_rep.paired_clonotypes:
                for clonotype in (pair.clonotype1, pair.clonotype2):
                    index.setdefault(clonotype.locus, {})[clonotype.sequence_id] = clonotype
        self._clonotype_lookup = index
        return index

    def get_clonotype(self, locus: str, clonotype_id: str) -> Clonotype | None:
        """Lazily retrieve a clonotype by (locus, sequence_id)."""
        return self._materialize_clonotype_lookup().get(locus, {}).get(clonotype_id)

    @classmethod
    def from_sample_repertoire(
        cls,
        sample_repertoire: SampleRepertoire,
        pairing_rows: list[tuple[str, str, str, str, str]],
        *,
        sample_id: str = "",
        barcode_pair_ids: list[tuple[str, str]] | None = None,
        barcode_metadata: dict[str, dict[str, str]] | None = None,
    ) -> "PairedRepertoire":
        """Build paired repertoire from sample repertoire and explicit pairing rows.

        pairing_rows schema:
            (pair_id, locus_1, locus_2, clonotype_id_1, clonotype_id_2)
        """
        if not sample_id:
            sample_id = sample_repertoire.sample_id

        lookup = _LazyClonotypeIndex(sample_repertoire)
        paired_by_family: dict[str, list[PairedClonotype]] = {k: [] for k in LOCUS_PAIR_TO_LOCI}
        used_clonotypes: set[tuple[str, str]] = set()

        for pair_id, locus_1, locus_2, clonotype_id_1, clonotype_id_2 in pairing_rows:
            locus_1 = str(locus_1).upper()
            locus_2 = str(locus_2).upper()

            locus_pair = ""
            for pair_name, (left, right) in LOCUS_PAIR_TO_LOCI.items():
                if (locus_1, locus_2) == (left, right) or (locus_1, locus_2) == (right, left):
                    locus_pair = pair_name
                    break
            if not locus_pair:
                raise ValueError(f"Unsupported locus pair ({locus_1}, {locus_2})")

            c1 = lookup.get(locus_1, clonotype_id_1)
            c2 = lookup.get(locus_2, clonotype_id_2)
            if c1 is None:
                raise KeyError(f"Unknown clonotype id {clonotype_id_1!r} for locus {locus_1!r}")
            if c2 is None:
                raise KeyError(f"Unknown clonotype id {clonotype_id_2!r} for locus {locus_2!r}")

            paired_by_family[locus_pair].append(PairedClonotype(pair_id=pair_id, clonotype1=c1, clonotype2=c2))
            used_clonotypes.add((c1.locus, c1.sequence_id))
            used_clonotypes.add((c2.locus, c2.sequence_id))

        barcode_pair_ids = list(barcode_pair_ids or [])
        chain_multiplicity = pl.DataFrame(
            schema={
                "sample_id": pl.Utf8,
                "locus_pair": pl.Utf8,
                "n_chain1": pl.Int64,
                "m_chain2": pl.Int64,
                "cell_count": pl.Int64,
            }
        )

        return cls(
            sample_id=sample_id,
            single_cell_repertoire=SingleCellRepertoire(
                barcode_pair_ids=barcode_pair_ids,
                barcode_metadata=dict(barcode_metadata or {}),
            ),
            paired_locus_repertoires={
                name: PairedLocusRepertoire(locus_pair=name, paired_clonotypes=pairs)
                for name, pairs in paired_by_family.items()
            },
            chain_multiplicity=chain_multiplicity,
            loaded_cell_count=len({barcode for barcode, _ in barcode_pair_ids}),
            loaded_clonotype_count=len(used_clonotypes),
        )


def _cell_row_to_clonotype(row: dict[str, object]) -> Clonotype:
    clonotype = Clonotype(
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
    for key in (
        "vdjdb_record_id",
        "mhc.a",
        "mhc.b",
        "mhc.class",
        "antigen.epitope",
        "antigen.gene",
        "antigen.species",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            clonotype.clone_metadata[key] = value
    return clonotype


def build_tenx_sample_from_cell_clonotypes(
    cell_clonotypes_df: pl.DataFrame,
    *,
    sample_id: str,
    barcode_metadata: dict[str, dict[str, str]] | None = None,
) -> PairedRepertoire:
    """Assemble sample-level paired structures from matched per-cell clonotype rows."""
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
    paired_by_family: dict[str, list[PairedClonotype]] = {name: [] for name in LOCUS_PAIR_TO_LOCI}
    multiplicity_rows: list[dict[str, object]] = []

    for (barcode, raw_pair_id), loci_map in grouped.items():
        for locus_pair, (locus1, locus2) in LOCUS_PAIR_TO_LOCI.items():
            chains1 = sorted(loci_map.get(locus1, {}).values(), key=lambda x: x.sequence_id)
            chains2 = sorted(loci_map.get(locus2, {}).values(), key=lambda x: x.sequence_id)
            n_chain1 = len(chains1)
            n_chain2 = len(chains2)
            multiplicity_rows.append(
                {
                    "sample_id": sample_id,
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
                "sample_id": pl.Utf8,
                "locus_pair": pl.Utf8,
                "n_chain1": pl.Int64,
                "m_chain2": pl.Int64,
                "cell_count": pl.Int64,
            }
        )
    else:
        chain_multiplicity = (
            multiplicity_df.group_by(["sample_id", "locus_pair", "n_chain1", "m_chain2"])
            .len()
            .rename({"len": "cell_count"})
            .sort(["sample_id", "locus_pair", "n_chain1", "m_chain2"])
        )

    paired_locus_repertoires = {
        locus_pair: PairedLocusRepertoire(locus_pair=locus_pair, paired_clonotypes=paired)
        for locus_pair, paired in paired_by_family.items()
    }

    return PairedRepertoire(
        sample_id=sample_id,
        single_cell_repertoire=SingleCellRepertoire(
            barcode_pair_ids=barcode_pair_ids,
            barcode_metadata=dict(barcode_metadata or {}),
        ),
        paired_locus_repertoires=paired_locus_repertoires,
        chain_multiplicity=chain_multiplicity,
        loaded_cell_count=len(grouped),
        loaded_clonotype_count=len(matched_clonotype_keys),
    )


def load_10x_vdj_v1_sample(
    consensus_annotations_path: str | Path,
    all_contig_annotations_path: str | Path,
    sample_id: str = "",
    *,
    check_is_cell: bool = True,
) -> PairedRepertoire:
    """Load one sample from 10x_vdj_v1 files into paired single-cell structures."""
    consensus_path = Path(consensus_annotations_path)
    if not sample_id:
        sample_id = consensus_path.name

    cell_df = load_10x_vdj_v1_cell_clonotypes(
        consensus_annotations_path=consensus_annotations_path,
        all_contig_annotations_path=all_contig_annotations_path,
        sample_id=sample_id,
        check_is_cell=check_is_cell,
    )
    return build_tenx_sample_from_cell_clonotypes(cell_df, sample_id=sample_id)


# Backward-compatible function alias for older code paths.
build_tenx_donor_from_cell_clonotypes = build_tenx_sample_from_cell_clonotypes


def load_10x_vdj_v1_donor(
    consensus_annotations_path: str | Path,
    all_contig_annotations_path: str | Path,
    donor_id: str = "",
    *,
    check_is_cell: bool = True,
) -> PairedRepertoire:
    """Compatibility wrapper for donor-named API."""
    return load_10x_vdj_v1_sample(
        consensus_annotations_path=consensus_annotations_path,
        all_contig_annotations_path=all_contig_annotations_path,
        sample_id=donor_id,
        check_is_cell=check_is_cell,
    )
