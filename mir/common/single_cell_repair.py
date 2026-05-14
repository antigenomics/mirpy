"""Repair and cleanup helpers for single-cell 10x clonotype tables."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mir.basic.pgen import OlgaModel

_REQUIRED_COLUMNS = {
    "barcode",
    "raw_pair_id",
    "sequence_id",
    "locus",
    "duplicate_count",
    "umi_count",
    "junction",
    "junction_aa",
    "v_gene",
    "d_gene",
    "j_gene",
    "c_gene",
}


@dataclass(slots=True)
class _SyntheticRecord:
    junction: str
    junction_aa: str
    v_gene: str
    d_gene: str
    j_gene: str


class _SyntheticGenerator:
    """Small OLGA-backed generator with per-locus model cache."""

    def __init__(self, *, species: str, seed: int) -> None:
        self.species = species
        self.seed = int(seed)
        self._counter = 0
        self._models: dict[str, OlgaModel] = {}

    def _get_model(self, locus: str) -> OlgaModel:
        if locus not in self._models:
            self._models[locus] = OlgaModel(species=self.species, locus=locus, seed=self.seed)
        return self._models[locus]

    def next(self, locus: str) -> _SyntheticRecord:
        self._counter += 1
        local_seed = self.seed + self._counter
        try:
            rec = self._get_model(locus).generate_pool(1, n_jobs=1, seed=local_seed)[0]
            return _SyntheticRecord(
                junction=str(rec.get("junction") or ""),
                junction_aa=str(rec.get("junction_aa") or ""),
                v_gene=str(rec.get("v_gene") or ""),
                d_gene=str(rec.get("d_gene") or ""),
                j_gene=str(rec.get("j_gene") or ""),
            )
        except Exception:
            # Keep a deterministic fallback so cleanup can proceed even if OLGA models are unavailable.
            aa = "CASSLGQETQYF" if locus.startswith("TR") else "CARDRSTGYYFDYW"
            return _SyntheticRecord(
                junction="",
                junction_aa=aa,
                v_gene="",
                d_gene="",
                j_gene="",
            )


def _ensure_columns(cell_clonotypes: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(cell_clonotypes.columns))
    if missing:
        raise ValueError(f"cell_clonotypes missing required columns: {missing}")


def _sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda r: (
            -int(r.get("duplicate_count") or 0),
            -int(r.get("umi_count") or 0),
            str(r.get("sequence_id") or ""),
        ),
    )


def _keep_primary_or_secondary(
    rows: list[dict[str, object]],
    *,
    secondary_ratio_threshold: float,
    secondary_min_umi_count: int,
    secondary_min_duplicate_count: int,
) -> list[dict[str, object]]:
    if len(rows) <= 1:
        return rows

    rows = _sort_rows(rows)
    first = rows[0]
    second = rows[1]
    first_dup = max(1, int(first.get("duplicate_count") or 0))
    first_umi = max(1, int(first.get("umi_count") or 0))
    second_dup = int(second.get("duplicate_count") or 0)
    second_umi = int(second.get("umi_count") or 0)

    dup_ratio = second_dup / first_dup
    umi_ratio = second_umi / first_umi
    keep_two = (
        dup_ratio > secondary_ratio_threshold
        and umi_ratio > secondary_ratio_threshold
        and second_umi >= secondary_min_umi_count
        and second_dup >= secondary_min_duplicate_count
    )
    return rows[:2] if keep_two else rows[:1]


def impute_missing_chains(
    cell_clonotypes: pl.DataFrame,
    *,
    species: str = "human",
    seed: int = 42,
    default_b_light_locus: str = "IGK",
) -> pl.DataFrame:
    """Impute missing paired loci with synthetic OLGA-generated clonotypes.

    Missingness is checked per (barcode, raw_pair_id) group.
    Added synthetic rows are emitted with duplicate_count=1 and umi_count=1.
    """
    _ensure_columns(cell_clonotypes)
    default_b_light_locus = default_b_light_locus.upper().strip()
    if default_b_light_locus not in {"IGK", "IGL"}:
        raise ValueError("default_b_light_locus must be 'IGK' or 'IGL'")

    synth = _SyntheticGenerator(species=species, seed=seed)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in cell_clonotypes.iter_rows(named=True):
        key = (str(row["barcode"]), str(row["raw_pair_id"]))
        grouped.setdefault(key, []).append(dict(row))

    next_id = cell_clonotypes.height
    augmented: list[dict[str, object]] = []

    def _add_synth_row(base: dict[str, object], locus: str) -> None:
        nonlocal next_id
        next_id += 1
        rec = synth.next(locus)
        augmented.append(
            {
                **base,
                "sequence_id": f"synthetic_{locus}_{next_id}",
                "locus": locus,
                "duplicate_count": 1,
                "umi_count": 1,
                "junction": rec.junction,
                "junction_aa": rec.junction_aa,
                "v_gene": rec.v_gene,
                "d_gene": rec.d_gene,
                "j_gene": rec.j_gene,
                "c_gene": "",
            }
        )

    for rows in grouped.values():
        augmented.extend(rows)
        loci = {str(r.get("locus") or "").upper() for r in rows}
        base = rows[0]

        if "TRA" in loci and "TRB" not in loci:
            _add_synth_row(base, "TRB")
        if "TRB" in loci and "TRA" not in loci:
            _add_synth_row(base, "TRA")

        if "TRG" in loci and "TRD" not in loci:
            _add_synth_row(base, "TRD")
        if "TRD" in loci and "TRG" not in loci:
            _add_synth_row(base, "TRG")

        has_igh = "IGH" in loci
        has_light = "IGK" in loci or "IGL" in loci
        if has_igh and not has_light:
            _add_synth_row(base, default_b_light_locus)
        if has_light and not has_igh:
            _add_synth_row(base, "IGH")

    return pl.from_dicts(augmented, schema=cell_clonotypes.schema)


def cleanup_cell_clonotypes(
    cell_clonotypes: pl.DataFrame,
    *,
    secondary_ratio_threshold: float = 0.1,
    secondary_min_umi_count: int = 2,
    secondary_min_duplicate_count: int = 5,
) -> pl.DataFrame:
    """Reduce over-expanded per-cell chains using locus-specific rules.

    Rules:
    - Keep top one for TRB/TRD/IGH.
    - For TRA/TRG keep one or two by ratio and minimum support thresholds.
    - For IGK/IGL jointly keep one or two by the same secondary criteria.
    """
    _ensure_columns(cell_clonotypes)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in cell_clonotypes.iter_rows(named=True):
        key = (str(row["barcode"]), str(row["raw_pair_id"]))
        grouped.setdefault(key, []).append(dict(row))

    kept: list[dict[str, object]] = []

    for rows in grouped.values():
        by_locus: dict[str, list[dict[str, object]]] = {}
        for r in rows:
            locus = str(r.get("locus") or "").upper()
            by_locus.setdefault(locus, []).append(r)

        for heavy in ("TRB", "TRD", "IGH"):
            heavy_rows = by_locus.get(heavy, [])
            if heavy_rows:
                kept.extend(_sort_rows(heavy_rows)[:1])

        for light in ("TRA", "TRG"):
            light_rows = by_locus.get(light, [])
            if light_rows:
                kept.extend(
                    _keep_primary_or_secondary(
                        light_rows,
                        secondary_ratio_threshold=secondary_ratio_threshold,
                        secondary_min_umi_count=secondary_min_umi_count,
                        secondary_min_duplicate_count=secondary_min_duplicate_count,
                    )
                )

        b_light_rows = by_locus.get("IGK", []) + by_locus.get("IGL", [])
        if b_light_rows:
            kept.extend(
                _keep_primary_or_secondary(
                    b_light_rows,
                    secondary_ratio_threshold=secondary_ratio_threshold,
                    secondary_min_umi_count=secondary_min_umi_count,
                    secondary_min_duplicate_count=secondary_min_duplicate_count,
                )
            )

    return pl.from_dicts(kept, schema=cell_clonotypes.schema)
