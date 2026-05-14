"""Utilities for single-cell paired-chain donor objects."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mir.common.single_cell import TenXVdjV1DonorData


@dataclass(slots=True)
class PairingGraph:
    """Pairing graph represented as node and edge tables."""

    nodes: pl.DataFrame
    edges: pl.DataFrame


def _node_key(row: dict[str, object]) -> str:
    locus = str(row.get("locus") or "")
    seq = str(row.get("sequence_id") or "")
    return f"{locus}:{seq}"


def build_pairing_graph(
    donor: TenXVdjV1DonorData,
    *,
    min_shared_cells: int = 1,
) -> PairingGraph:
    """Build clonotype pairing graph from paired-clonotype donor output.

    Nodes are clonotypes. Edges connect paired clonotypes and are weighted by
    the number of unique barcodes carrying that pair.
    """
    if min_shared_cells < 1:
        raise ValueError("min_shared_cells must be >= 1")

    barcode_pairs = set(donor.single_cell_repertoire.barcode_pair_ids)
    pair_cell_counts: dict[str, int] = {}
    for barcode, pair_id in barcode_pairs:
        del barcode
        pair_cell_counts[pair_id] = pair_cell_counts.get(pair_id, 0) + 1

    pair_to_records: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for rep in donor.paired_locus_repertoires.values():
        for paired in rep.paired_clonotypes:
            left = paired.clonotype1.serialize()
            right = paired.clonotype2.serialize()
            pair_to_records.setdefault(paired.pair_id, []).append((left, right))

    node_attrs: dict[str, dict[str, object]] = {}
    edge_weights: dict[tuple[str, str], int] = {}

    for pair_id, records in pair_to_records.items():
        weight = pair_cell_counts.get(pair_id, 0)
        if weight <= 0:
            continue

        unique_record_keys: set[tuple[str, str]] = set()
        for left, right in records:
            left_id = _node_key(left)
            right_id = _node_key(right)
            if not left_id or not right_id or left_id == right_id:
                continue

            record_key = tuple(sorted((left_id, right_id)))
            if record_key in unique_record_keys:
                continue
            unique_record_keys.add(record_key)

            node_attrs[left_id] = {
                "node_id": left_id,
                "locus": str(left.get("locus") or ""),
                "sequence_id": str(left.get("sequence_id") or ""),
                "junction_aa": str(left.get("junction_aa") or ""),
            }
            node_attrs[right_id] = {
                "node_id": right_id,
                "locus": str(right.get("locus") or ""),
                "sequence_id": str(right.get("sequence_id") or ""),
                "junction_aa": str(right.get("junction_aa") or ""),
            }

            edge = record_key
            edge_weights[edge] = edge_weights.get(edge, 0) + weight

    node_schema = {
        "node_id": pl.Utf8,
        "locus": pl.Utf8,
        "sequence_id": pl.Utf8,
        "junction_aa": pl.Utf8,
    }
    edge_schema = {
        "source": pl.Utf8,
        "target": pl.Utf8,
        "cell_count": pl.Int64,
    }

    nodes_df = (
        pl.from_dicts(list(node_attrs.values()), schema=node_schema)
        if node_attrs
        else pl.DataFrame(schema=node_schema)
    )
    edges_rows = [
        {"source": src, "target": dst, "cell_count": w}
        for (src, dst), w in edge_weights.items()
        if w >= min_shared_cells
    ]
    edges_df = (
        pl.from_dicts(edges_rows, schema=edge_schema).sort(["cell_count", "source"], descending=[True, False])
        if edges_rows
        else pl.DataFrame(schema=edge_schema)
    )

    return PairingGraph(nodes=nodes_df, edges=edges_df)
