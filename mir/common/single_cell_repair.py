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

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in cell_clonotypes.iter_rows(named=True):
        key = (str(row["barcode"]), str(row["raw_pair_id"]))
        grouped.setdefault(key, []).append(dict(row))

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
        loci = {str(r.get("locus") or "").upper() for r in rows}
        base = rows[0]

        trb_rows = [r for r in rows if str(r.get("locus") or "").upper() == "TRB"]
        tra_rows = [r for r in rows if str(r.get("locus") or "").upper() == "TRA"]
        trg_rows = [r for r in rows if str(r.get("locus") or "").upper() == "TRG"]
        trd_rows = [r for r in rows if str(r.get("locus") or "").upper() == "TRD"]
        igh_rows = [r for r in rows if str(r.get("locus") or "").upper() == "IGH"]

        if "TRA" in loci and "TRB" not in loci:
            _add_synth_row(base, "TRB")
        if "TRB" in loci and "TRA" not in loci:
            trb_master = _sort_rows(trb_rows)[0] if trb_rows else None
            _add_synth_row(
                base,
                "TRA",
                master_locus="TRB" if trb_master is not None else None,
                master_sequence_id=str(trb_master.get("sequence_id") or "") if trb_master is not None else None,
            )

        if "TRG" in loci and "TRD" not in loci:
            _add_synth_row(base, "TRD")
        if "TRD" in loci and "TRG" not in loci:
            trd_master = _sort_rows(trd_rows)[0] if trd_rows else None
            _add_synth_row(
                base,
                "TRG",
                master_locus="TRD" if trd_master is not None else None,
                master_sequence_id=str(trd_master.get("sequence_id") or "") if trd_master is not None else None,
            )

        has_igh = "IGH" in loci
        has_light = "IGK" in loci or "IGL" in loci
        if has_igh and not has_light:
            igh_master = _sort_rows(igh_rows)[0] if igh_rows else None
            _add_synth_row(
                base,
                default_b_light_locus,
                master_locus="IGH" if igh_master is not None else None,
                master_sequence_id=str(igh_master.get("sequence_id") or "") if igh_master is not None else None,
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

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in cell_clonotypes.iter_rows(named=True):
        key = (str(row["barcode"]), str(row["raw_pair_id"]))
        grouped.setdefault(key, []).append(dict(row))

    kept_by_group: dict[tuple[str, str], list[dict[str, object]]] = {}

    for rows in grouped.values():
        by_locus: dict[str, list[dict[str, object]]] = {}
        for r in rows:
            locus = str(r.get("locus") or "").upper()
            by_locus.setdefault(locus, []).append(r)

        for heavy in ("TRB", "TRD", "IGH"):
            heavy_rows = by_locus.get(heavy, [])
            if heavy_rows:
                kept_by_group.setdefault((str(rows[0]["barcode"]), str(rows[0]["raw_pair_id"])), []).extend(
                    _sort_rows(heavy_rows)[:1]
                )

        for light in ("TRA", "TRG"):
            light_rows = by_locus.get(light, [])
            if light_rows:
                kept_by_group.setdefault((str(rows[0]["barcode"]), str(rows[0]["raw_pair_id"])), []).extend(
                    _keep_primary_or_secondary(
                        light_rows,
                        secondary_ratio_threshold=secondary_ratio_threshold,
                        secondary_min_umi_count=secondary_min_umi_count,
                        secondary_min_duplicate_count=secondary_min_duplicate_count,
                    )
                )

        b_light_rows = by_locus.get("IGK", []) + by_locus.get("IGL", [])
        if b_light_rows:
            kept_by_group.setdefault((str(rows[0]["barcode"]), str(rows[0]["raw_pair_id"])), []).extend(
                _keep_primary_or_secondary(
                    b_light_rows,
                    secondary_ratio_threshold=secondary_ratio_threshold,
                    secondary_min_umi_count=secondary_min_umi_count,
                    secondary_min_duplicate_count=secondary_min_duplicate_count,
                )
            )

    def _master_edges(
        grouped_rows: dict[tuple[str, str], list[dict[str, object]]],
    ) -> tuple[
        dict[tuple[str, str], dict[tuple[str, str], int]],
        dict[tuple[str, str], dict[tuple[str, str], int]],
    ]:
        master_to_slave_counts: dict[tuple[str, str], dict[tuple[str, str], int]] = {}
        master_to_slave_support: dict[tuple[str, str], dict[tuple[str, str], int]] = {}

        for rows_local in grouped_rows.values():
            by_locus: dict[str, list[dict[str, object]]] = {}
            for r in rows_local:
                by_locus.setdefault(str(r.get("locus") or "").upper(), []).append(r)

            for master_locus, slave_loci in _MASTER_TO_SLAVE_LOCI.items():
                masters = by_locus.get(master_locus, [])
                slaves = [s for slave_locus in slave_loci for s in by_locus.get(slave_locus, [])]
                if not masters or not slaves:
                    continue

                for m in masters:
                    mkey = (master_locus, str(m.get("sequence_id") or ""))
                    for s in slaves:
                        if consistency_only_on_synthetic_slave and not str(s.get("sequence_id") or "").startswith("synthetic_"):
                            continue
                        skey = (str(s.get("locus") or "").upper(), str(s.get("sequence_id") or ""))
                        master_to_slave_counts.setdefault(mkey, {})[skey] = master_to_slave_counts.setdefault(mkey, {}).get(skey, 0) + 1
                        support = int(s.get("duplicate_count") or 0) + int(s.get("umi_count") or 0)
                        master_to_slave_support.setdefault(mkey, {})[skey] = master_to_slave_support.setdefault(mkey, {}).get(skey, 0) + support

        return master_to_slave_counts, master_to_slave_support

    if enforce_consistent_slave_per_master:
        edge_counts, edge_support = _master_edges(kept_by_group)
        canonical_slave: dict[tuple[str, str], tuple[str, str]] = {}
        for master_key, slaves in edge_counts.items():
            best = max(
                slaves,
                key=lambda sk: (
                    slaves[sk],
                    edge_support.get(master_key, {}).get(sk, 0),
                    sk[0],
                    sk[1],
                ),
            )
            canonical_slave[master_key] = best

        updated: dict[tuple[str, str], list[dict[str, object]]] = {}
        for key, rows_local in kept_by_group.items():
            by_locus: dict[str, list[dict[str, object]]] = {}
            for r in rows_local:
                by_locus.setdefault(str(r.get("locus") or "").upper(), []).append(r)

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
                for r in keep_rows:
                    locus = str(r.get("locus") or "").upper()
                    seq_id = str(r.get("sequence_id") or "")
                    if locus not in slave_loci:
                        filtered_rows.append(r)
                        continue
                    if consistency_only_on_synthetic_slave and not seq_id.startswith("synthetic_"):
                        filtered_rows.append(r)
                        continue
                    if locus == target_locus and seq_id == target_id:
                        filtered_rows.append(r)
                keep_rows = filtered_rows

            updated[key] = keep_rows
        kept_by_group = updated

    if max_slave_edges_per_master is not None:
        edge_counts, _ = _master_edges(kept_by_group)
        flagged = {
            master_key
            for master_key, slaves in edge_counts.items()
            if len(slaves) > int(max_slave_edges_per_master)
        }
        if flagged:
            pruned: dict[tuple[str, str], list[dict[str, object]]] = {}
            for key, rows_local in kept_by_group.items():
                rows_out = list(rows_local)
                for master_locus, slave_loci in _MASTER_TO_SLAVE_LOCI.items():
                    masters = [r for r in rows_out if str(r.get("locus") or "").upper() == master_locus]
                    if not masters:
                        continue
                    master = _sort_rows(masters)[0]
                    mkey = (master_locus, str(master.get("sequence_id") or ""))
                    if mkey in flagged:
                        rows_out = [
                            r
                            for r in rows_out
                            if str(r.get("locus") or "").upper() not in (master_locus, *slave_loci)
                        ]
                pruned[key] = rows_out
            kept_by_group = pruned

    kept = [r for rows_local in kept_by_group.values() for r in rows_local]
    return pl.from_dicts(kept, schema=cell_clonotypes.schema)
