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

_MASTER_TO_SLAVE_LOCI: dict[str, tuple[str, ...]] = {
    "TRB": ("TRA",),
    "TRD": ("TRG",),
    "IGH": ("IGK", "IGL"),
}

_HEAVY_LOCI: tuple[str, ...] = ("TRB", "TRD", "IGH")
_LIGHT_LOCI: tuple[str, ...] = ("TRA", "TRG")
_B_LIGHT_LOCI: tuple[str, ...] = ("IGK", "IGL")


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


def _group_cell_rows(cell_clonotypes: pl.DataFrame) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in cell_clonotypes.iter_rows(named=True):
        key = (str(row["barcode"]), str(row["raw_pair_id"]))
        grouped.setdefault(key, []).append(dict(row))
    return grouped


def _rows_by_locus(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_locus: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_locus.setdefault(str(row.get("locus") or "").upper(), []).append(row)
    return by_locus


def _primary_sequence_id(by_locus: dict[str, list[dict[str, object]]], locus: str) -> str | None:
    rows = by_locus.get(locus, [])
    if not rows:
        return None
    return str(_sort_rows(rows)[0].get("sequence_id") or "")


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
    reuse_slave_per_master: bool = False,
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
    per_master_slave: dict[tuple[str, str, str], dict[str, object]] = {}
    grouped = _group_cell_rows(cell_clonotypes)

    next_id = cell_clonotypes.height
    augmented: list[dict[str, object]] = []

    def _add_synth_row(
        base: dict[str, object],
        locus: str,
        *,
        master_locus: str | None = None,
        master_sequence_id: str | None = None,
    ) -> None:
        nonlocal next_id
        reused: dict[str, object] | None = None
        if reuse_slave_per_master and master_locus and master_sequence_id:
            reused = per_master_slave.get((master_locus, master_sequence_id, locus))

        if reused is None:
            next_id += 1
            rec = synth.next(locus)
            reused = {
                "sequence_id": f"synthetic_{locus}_{next_id}",
                "junction": rec.junction,
                "junction_aa": rec.junction_aa,
                "v_gene": rec.v_gene,
                "d_gene": rec.d_gene,
                "j_gene": rec.j_gene,
                "c_gene": "",
            }
            if reuse_slave_per_master and master_locus and master_sequence_id:
                per_master_slave[(master_locus, master_sequence_id, locus)] = reused

        augmented.append(
            {
                **base,
                "sequence_id": reused["sequence_id"],
                "locus": locus,
                "duplicate_count": 1,
                "umi_count": 1,
                "junction": reused["junction"],
                "junction_aa": reused["junction_aa"],
                "v_gene": reused["v_gene"],
                "d_gene": reused["d_gene"],
                "j_gene": reused["j_gene"],
                "c_gene": reused["c_gene"],
            }
        )

    for rows in grouped.values():
        augmented.extend(rows)
        by_locus = _rows_by_locus(rows)
        loci = set(by_locus)
        base = rows[0]

        if "TRA" in loci and "TRB" not in loci:
            _add_synth_row(base, "TRB")
        if "TRB" in loci and "TRA" not in loci:
            trb_master_seq = _primary_sequence_id(by_locus, "TRB")
            _add_synth_row(
                base,
                "TRA",
                master_locus="TRB" if trb_master_seq else None,
                master_sequence_id=trb_master_seq,
            )

        if "TRG" in loci and "TRD" not in loci:
            _add_synth_row(base, "TRD")
        if "TRD" in loci and "TRG" not in loci:
            trd_master_seq = _primary_sequence_id(by_locus, "TRD")
            _add_synth_row(
                base,
                "TRG",
                master_locus="TRD" if trd_master_seq else None,
                master_sequence_id=trd_master_seq,
            )

        has_igh = "IGH" in loci
        has_light = "IGK" in loci or "IGL" in loci
        if has_igh and not has_light:
            igh_master_seq = _primary_sequence_id(by_locus, "IGH")
            _add_synth_row(
                base,
                default_b_light_locus,
                master_locus="IGH" if igh_master_seq else None,
                master_sequence_id=igh_master_seq,
            )
        if has_light and not has_igh:
            _add_synth_row(base, "IGH")

    return pl.from_dicts(augmented, schema=cell_clonotypes.schema)


def cleanup_cell_clonotypes(
    cell_clonotypes: pl.DataFrame,
    *,
    secondary_ratio_threshold: float = 0.1,
    secondary_min_umi_count: int = 2,
    secondary_min_duplicate_count: int = 5,
    enforce_consistent_slave_per_master: bool = False,
    consistency_only_on_synthetic_slave: bool = True,
    max_slave_edges_per_master: int | None = None,
) -> pl.DataFrame:
    """Reduce over-expanded per-cell chains using locus-specific rules.

    Rules:
    - Keep top one for TRB/TRD/IGH.
    - For TRA/TRG keep one or two by ratio and minimum support thresholds.
    - For IGK/IGL jointly keep one or two by the same secondary criteria.
    """
    _ensure_columns(cell_clonotypes)
    if max_slave_edges_per_master is not None and int(max_slave_edges_per_master) <= 0:
        raise ValueError("max_slave_edges_per_master must be positive when provided")

    def _master_edges(
        grouped_rows: dict[tuple[str, str], list[dict[str, object]]],
    ) -> tuple[
        dict[tuple[str, str], dict[tuple[str, str], int]],
        dict[tuple[str, str], dict[tuple[str, str], int]],
    ]:
        master_to_slave_counts: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
        master_to_slave_support: dict[tuple[str, str], dict[tuple[str, str], int]] = {}

        for rows_local in grouped_rows.values():
            by_locus = _rows_by_locus(rows_local)

            for master_locus, slave_loci in _MASTER_TO_SLAVE_LOCI.items():
                masters = by_locus.get(master_locus, [])
                slaves = [s for slave_locus in slave_loci for s in by_locus.get(slave_locus, [])]
                if not masters or not slaves:
                    continue

                for master_row in masters:
                    mkey = (master_locus, str(master_row.get("sequence_id") or ""))
                    for slave_row in slaves:
                        seq_id = str(slave_row.get("sequence_id") or "")
                        if consistency_only_on_synthetic_slave and not seq_id.startswith("synthetic_"):
                            continue
                        skey = (str(slave_row.get("locus") or "").upper(), seq_id)
                        master_to_slave_counts.setdefault(mkey, {})[skey] = master_to_slave_counts.setdefault(mkey, {}).get(skey, 0) + 1
                        support = int(slave_row.get("duplicate_count") or 0) + int(slave_row.get("umi_count") or 0)
                        master_to_slave_support.setdefault(mkey, {})[skey] = master_to_slave_support.setdefault(mkey, {}).get(skey, 0) + support

        return master_to_slave_counts, master_to_slave_support

    def _enforce_consistent_slaves(
        grouped_rows: dict[tuple[str, str], list[dict[str, object]]],
    ) -> dict[tuple[str, str], list[dict[str, object]]]:
        edge_counts, edge_support = _master_edges(grouped_rows)
        canonical_slave: dict[tuple[str, str], tuple[str, str]] = {}
        for master_key, slaves in edge_counts.items():
            canonical_slave[master_key] = max(
                slaves,
                key=lambda slave_key: (
                    slaves[slave_key],
                    edge_support.get(master_key, {}).get(slave_key, 0),
                    slave_key[0],
                    slave_key[1],
                ),
            )

        updated: dict[tuple[str, str], list[dict[str, object]]] = {}
        for group_key, rows_local in grouped_rows.items():
            by_locus = _rows_by_locus(rows_local)
            keep_rows: list[dict[str, object]] = list(rows_local)

            for master_locus, slave_loci in _MASTER_TO_SLAVE_LOCI.items():
                masters = by_locus.get(master_locus, [])
                if not masters:
                    continue
                master = _sort_rows(masters)[0]
                mkey = (master_locus, str(master.get("sequence_id") or ""))
                target = canonical_slave.get(mkey)
                if target is None:
                    continue
                target_locus, target_id = target

                filtered_rows: list[dict[str, object]] = []
                for row in keep_rows:
                    locus = str(row.get("locus") or "").upper()
                    seq_id = str(row.get("sequence_id") or "")
                    if locus not in slave_loci:
                        filtered_rows.append(row)
                        continue
                    if consistency_only_on_synthetic_slave and not seq_id.startswith("synthetic_"):
                        filtered_rows.append(row)
                        continue
                    if locus == target_locus and seq_id == target_id:
                        filtered_rows.append(row)
                keep_rows = filtered_rows

            updated[group_key] = keep_rows
        return updated

    def _prune_high_degree_masters(
        grouped_rows: dict[tuple[str, str], list[dict[str, object]]],
    ) -> dict[tuple[str, str], list[dict[str, object]]]:
        edge_counts, _ = _master_edges(grouped_rows)
        flagged = {
            master_key
            for master_key, slaves in edge_counts.items()
            if len(slaves) > int(max_slave_edges_per_master)
        }
        if not flagged:
            return grouped_rows

        pruned: dict[tuple[str, str], list[dict[str, object]]] = {}
        for group_key, rows_local in grouped_rows.items():
            rows_out = list(rows_local)
            by_locus = _rows_by_locus(rows_out)
            for master_locus, slave_loci in _MASTER_TO_SLAVE_LOCI.items():
                masters = by_locus.get(master_locus, [])
                if not masters:
                    continue
                master = _sort_rows(masters)[0]
                mkey = (master_locus, str(master.get("sequence_id") or ""))
                if mkey in flagged:
                    blocked_loci = set((master_locus, *slave_loci))
                    rows_out = [row for row in rows_out if str(row.get("locus") or "").upper() not in blocked_loci]
                    by_locus = _rows_by_locus(rows_out)
            pruned[group_key] = rows_out
        return pruned

    grouped = _group_cell_rows(cell_clonotypes)

    kept_by_group: dict[tuple[str, str], list[dict[str, object]]] = {}

    for key, rows in grouped.items():
        by_locus = _rows_by_locus(rows)
        kept_rows: list[dict[str, object]] = []

        for heavy in _HEAVY_LOCI:
            heavy_rows = by_locus.get(heavy, [])
            if heavy_rows:
                kept_rows.extend(_sort_rows(heavy_rows)[:1])

        for light in _LIGHT_LOCI:
            light_rows = by_locus.get(light, [])
            if light_rows:
                kept_rows.extend(
                    _keep_primary_or_secondary(
                        light_rows,
                        secondary_ratio_threshold=secondary_ratio_threshold,
                        secondary_min_umi_count=secondary_min_umi_count,
                        secondary_min_duplicate_count=secondary_min_duplicate_count,
                    )
                )

        b_light_rows = []
        for locus in _B_LIGHT_LOCI:
            b_light_rows.extend(by_locus.get(locus, []))
        if b_light_rows:
            kept_rows.extend(
                _keep_primary_or_secondary(
                    b_light_rows,
                    secondary_ratio_threshold=secondary_ratio_threshold,
                    secondary_min_umi_count=secondary_min_umi_count,
                    secondary_min_duplicate_count=secondary_min_duplicate_count,
                )
            )

        kept_by_group[key] = kept_rows

    if enforce_consistent_slave_per_master:
        kept_by_group = _enforce_consistent_slaves(kept_by_group)

    if max_slave_edges_per_master is not None:
        kept_by_group = _prune_high_degree_masters(kept_by_group)

    kept = [r for rows_local in kept_by_group.values() for r in rows_local]
    return pl.from_dicts(kept, schema=cell_clonotypes.schema)
