"""Benchmark for TCRNET-like enrichment with GIL-like motif spikes.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_tcrnet_benchmark.py -s
"""

from __future__ import annotations

import random
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mir.biomarkers.tcrnet import compute_tcrnet
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser, VDJdbSlimParser
from mir.common.repertoire import LocusRepertoire
from tests.benchmark_helpers import (
    benchmark_log_line,
    load_gilg_target_repertoire,
    real_control_repertoire,
    synthetic_control_repertoire,
    synthetic_control_size,
)
from tests.conftest import benchmark_max_seconds, skip_benchmarks


_AA = "ACDEFGHIKLMNPQRSTVWY"
_B35_FILE = Path(__file__).parent / "assets" / "B35+.txt.gz"
_VDJDB_FILE = Path(__file__).parent / "assets" / "vdjdb.slim.txt.gz"


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.Series(p_values, dtype=float)
    if values.empty:
        return values

    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(out, index=values.index)


def _load_b35_target_repertoire() -> LocusRepertoire:
    if not _B35_FILE.exists():
        pytest.skip("B35+ benchmark asset missing: tests/assets/B35+.txt.gz")

    parser = ClonotypeTableParser()
    rep = LocusRepertoire(parser.parse(str(_B35_FILE)), locus="TRB", repertoire_id="B35+")
    return filter_functional(rep)


def _load_vdjdb_b35_epl_reference() -> LocusRepertoire:
    if not _VDJDB_FILE.exists():
        pytest.skip("VDJdb asset missing: tests/assets/vdjdb.slim.txt.gz")

    sample = VDJdbSlimParser().parse_file(_VDJDB_FILE, species="HomoSapiens")
    filtered = [
        c
        for c in sample["TRB"].clonotypes
        if c.clone_metadata.get("antigen.epitope") == "EPLPQGQLTAY"
        and "B*35" in c.clone_metadata.get("mhc.a", "")
    ]

    # Keep exactly one record per junction_aa so benchmark cardinality checks are stable.
    unique_by_cdr3: dict[str, Clonotype] = {}
    for c in filtered:
        unique_by_cdr3.setdefault(c.junction_aa, c)

    return LocusRepertoire(
        clonotypes=list(unique_by_cdr3.values()),
        locus="TRB",
        repertoire_id="vdjdb-b35-epl",
    )


def _sequence_table(rep: LocusRepertoire) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    total = max(1, rep.duplicate_count)
    for c in rep.clonotypes:
        rows.append(
            {
                "cdr3aa": c.junction_aa,
                "count": int(c.duplicate_count),
                "v_gene": c.v_gene,
                "j_gene": c.j_gene,
            }
        )
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["cdr3aa", "v_gene", "j_gene"], as_index=False)
        .agg(count=("count", "sum"))
        .sort_values("count", ascending=False)
    )
    grouped["freq"] = grouped["count"] / total
    return grouped


def _tcrnet_table_with_counts(rep: LocusRepertoire, result_table: pd.DataFrame) -> pd.DataFrame:
    counts = _sequence_table(rep)[["cdr3aa", "count", "freq"]].rename(columns={"cdr3aa": "junction_aa"})
    table = result_table.merge(counts, on="junction_aa", how="left")
    table = table.rename(columns={"junction_aa": "cdr3aa", "fold_enrichment": "fold"})
    table = table[table["count"].fillna(0).gt(1)].copy()
    table["p.adj"] = _bh_adjust(table["p_value"])
    return table.sort_values(["p.adj", "fold"], ascending=[True, False]).reset_index(drop=True)


def _collect_neighbor_sequences(rep: LocusRepertoire, query_sequences: list[str]) -> set[str]:
    found: set[str] = set()
    for seq in query_sequences:
        hits = rep.trie.SearchIndices(seq, maxSubstitution=1, maxInsertion=0, maxDeletion=0, maxEdits=1)
        for idx, _ in hits:
            found.add(rep.clonotypes[int(idx)].junction_aa)
    return found


def _sequences_to_clonotypes(df: pd.DataFrame) -> list[Clonotype]:
    clonotypes: list[Clonotype] = []
    for idx, row in enumerate(df.drop_duplicates("cdr3aa").itertuples(index=False)):
        clonotypes.append(
            Clonotype(
                sequence_id=str(idx),
                locus="TRB",
                junction_aa=str(row.cdr3aa),
                duplicate_count=int(row.count),
                v_gene=str(getattr(row, "v_gene", "") or ""),
                j_gene=str(getattr(row, "j_gene", "") or ""),
                _validate=False,
            )
        )
    return clonotypes


def _rand_cdr3(rng: random.Random, n: int = 13) -> str:
    return "C" + "".join(rng.choice(_AA) for _ in range(n - 2)) + "F"


def _clone(sid: str, aa: str, *, v: str = "TRBV5-1*01", j: str = "TRBJ2-7*01") -> Clonotype:
    return Clonotype(
        sequence_id=sid,
        locus="TRB",
        junction_aa=aa,
        v_gene=v,
        j_gene=j,
        duplicate_count=1,
        _validate=False,
    )


@skip_benchmarks
def test_tcrnet_benchmark_gil_like_motif_enrichment(capsys, tmp_path: Path) -> None:
    rng = random.Random(42)

    # Base control
    control = [_clone(f"c{i}", _rand_cdr3(rng), v="TRBV7-9*01", j="TRBJ2-1*01") for i in range(500)]
    # Sparse GIL-like motifs in control
    control.extend(
        [
            _clone("cg0", "CASSGILGNTQYF"),
            _clone("cg1", "CASSGILANTQYF"),
            _clone("cg2", "CASSGILGDTQYF"),
        ]
    )

    # Target has stronger GIL-like motif neighborhood
    target = [_clone(f"t{i}", _rand_cdr3(rng), v="TRBV7-9*01", j="TRBJ2-1*01") for i in range(180)]
    target.extend([_clone(f"tg{i}", "CASSGILGNTQYF") for i in range(20)])
    target.extend(
        [
            _clone("tg20", "CASSGILANTQYF"),
            _clone("tg21", "CASSGILGDTQYF"),
            _clone("tg22", "CASSGILGNTQFF"),
            _clone("tg23", "CASSGILGNTQHF"),
        ]
    )

    target_rep = LocusRepertoire(target, locus="TRB")
    control_rep = LocusRepertoire(control, locus="TRB")

    t0 = time.perf_counter()
    serial = compute_tcrnet(
        target_rep,
        control=control_rep,
        metric="hamming",
        threshold=1,
        match_mode="vj",
        pvalue_mode="beta-binomial",
        n_jobs=1,
    )
    elapsed_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = compute_tcrnet(
        target_rep,
        control=control_rep,
        metric="hamming",
        threshold=1,
        match_mode="vj",
        pvalue_mode="beta-binomial",
        n_jobs=4,
    )
    elapsed = time.perf_counter() - t0

    assert result.table.equals(serial.table)

    df = result.table
    hits = df[
        (df["junction_aa"].str.contains("GILG", na=False))
        & (df["n_neighbors"] >= 10)
        & (df["p_value"] < 0.03)
    ]

    with capsys.disabled():
        print("\n" + "=" * 72)
        print("TCRNET benchmark: motif enrichment")
        print(f"target size: {len(target_rep.clonotypes)}")
        print(f"control size: {len(control_rep.clonotypes)}")
        print(f"runtime serial(1): {elapsed_serial:.3f}s")
        print(f"runtime parallel(4): {elapsed:.3f}s")
        if elapsed > 0:
            print(f"speedup: {elapsed_serial / elapsed:.2f}x")
        print(f"motif-enriched hits: {len(hits)}")
        print("Top rows:")
        print(df.head(10).to_string(index=False))
        print("=" * 72)

    assert len(df) == len(target_rep.clonotypes)
    assert len(hits) >= 1

    max_s = benchmark_max_seconds(default=120.0)
    assert elapsed < max_s


@skip_benchmarks
def test_tcrnet_runtime_gilg_vs_synthetic_1m(capsys) -> None:
    """Benchmark TCRNET runtime for a GIL target vs synthetic control (1e6 default)."""
    n_control = synthetic_control_size(default=1_000_000)
    require_cached = os.getenv("MIRPY_BENCH_REQUIRE_CACHED_CONTROL", "1") != "0"
    manager = ControlManager()

    target = load_gilg_target_repertoire()
    control = synthetic_control_repertoire(
        manager=manager,
        species="human",
        locus="TRB",
        n=n_control,
        require_cached=require_cached,
    )

    t0 = time.perf_counter()
    serial = compute_tcrnet(
        target,
        control=control,
        metric="hamming",
        threshold=1,
        match_mode="none",
        pvalue_mode="binomial",
        n_jobs=1,
    )
    elapsed_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel = compute_tcrnet(
        target,
        control=control,
        metric="hamming",
        threshold=1,
        match_mode="none",
        pvalue_mode="binomial",
        n_jobs=4,
    )
    elapsed_parallel = time.perf_counter() - t0

    assert parallel.table.equals(serial.table)

    with capsys.disabled():
        print("\n" + "=" * 72)
        print("TCRNET benchmark: GIL target vs synthetic control")
        print(f"target size: {len(target.clonotypes)}")
        print(f"control size: {len(control.clonotypes)} (requested n={n_control})")
        print(f"runtime serial(1): {elapsed_serial:.3f}s")
        print(f"runtime parallel(4): {elapsed_parallel:.3f}s")
        if elapsed_parallel > 0:
            print(f"speedup: {elapsed_serial / elapsed_parallel:.2f}x")
        print("=" * 72)

    max_s = benchmark_max_seconds(default=600.0)
    assert elapsed_parallel < max_s


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_tcrnet_benchmark_b35_epl_connected_component_vs_real_control(capsys) -> None:
    """Notebook-derived B35+ benchmark with VDJdb EPL/HLA-B*35 component checks."""
    manager = ControlManager()

    target = _load_b35_target_repertoire()
    vdjdb_ref = _load_vdjdb_b35_epl_reference()
    control = real_control_repertoire(manager=manager, species="human", locus="TRB")
    ref_sequences = {c.junction_aa for c in vdjdb_ref.clonotypes}

    t0 = time.perf_counter()
    result = compute_tcrnet(
        target,
        control=control,
        metric="hamming",
        threshold=1,
        match_mode="vj",
        pvalue_mode="binomial",
        n_jobs=4,
    )
    elapsed = time.perf_counter() - t0

    table = _tcrnet_table_with_counts(target, result.table)
    dedup = table.drop_duplicates(["cdr3aa", "v_gene", "j_gene"]).copy()

    # High-confidence donor-enriched clonotypes versus real control.
    enriched = dedup[(dedup["p.adj"] < 1e-6) & (dedup["fold"] >= 2.0)].copy()
    enriched_vdjdb = enriched[enriched["cdr3aa"].isin(ref_sequences)].copy()

    seq_table = _sequence_table(target)
    seed_sequences = enriched_vdjdb["cdr3aa"].drop_duplicates().tolist()
    neighbor_sequences = _collect_neighbor_sequences(target, seed_sequences)
    component_nodes = seq_table[seq_table["cdr3aa"].isin(neighbor_sequences)].copy()
    component_vdjdb = component_nodes[component_nodes["cdr3aa"].isin(ref_sequences)].copy()

    component_sizes: list[int] = []
    if not component_nodes.empty:
        from mir.graph.edit_distance_graph import build_edit_distance_graph

        graph = build_edit_distance_graph(
            _sequences_to_clonotypes(component_nodes),
            metric="hamming",
            threshold=1,
            nproc=4,
        )
        component_sizes = sorted(graph.components().sizes(), reverse=True)

    largest_component = component_sizes[0] if component_sizes else 0
    n_components_ge5 = sum(size >= 5 for size in component_sizes)
    component_overlap_fraction = (
        len(component_vdjdb) / len(ref_sequences) if ref_sequences else 0.0
    )
    enriched_overlap_fraction = (
        len(enriched_vdjdb) / len(enriched) if len(enriched) else 0.0
    )

    benchmark_log_line(
        "tcrnet_b35_epl_real_control "
        f"elapsed_s={elapsed:.3f} "
        f"target_n={len(target.clonotypes)} control_n={len(control.clonotypes)} "
        f"strict_enriched_n={len(enriched)} enriched_vdjdb_n={len(enriched_vdjdb)} "
        f"neighbor_nodes_n={len(component_nodes)} largest_component={largest_component} "
        f"components_ge5={n_components_ge5} component_vdjdb_n={len(component_vdjdb)} "
        f"vdjdb_ref_n={len(ref_sequences)} "
        f"component_overlap_fraction={component_overlap_fraction:.3f} "
        f"enriched_overlap_fraction={enriched_overlap_fraction:.3f}"
    )

    with capsys.disabled():
        print("\n" + "=" * 72)
        print("TCRNET benchmark: B35+ EPL/HLA-B*35 connected component vs real control")
        print(f"runtime parallel(4): {elapsed:.3f}s")
        print(f"target size: {len(target.clonotypes)}")
        print(f"real control size: {len(control.clonotypes)}")
        print(f"VDJdb B*35 EPL reference sequences: {len(ref_sequences)}")
        print(f"strict enriched clonotypes (padj<1e-6, fold>=2): {len(enriched)}")
        print(f"strict enriched clonotypes overlapping VDJdb EPL ref: {len(enriched_vdjdb)}")
        print(f"neighbor nodes from VDJdb EPL seeds: {len(component_nodes)}")
        print(f"component nodes overlapping VDJdb EPL ref: {len(component_vdjdb)}")
        print(f"component overlap fraction vs VDJdb ref: {component_overlap_fraction:.3f}")
        print(f"enriched overlap fraction vs strict enriched: {enriched_overlap_fraction:.3f}")
        print(f"largest connected component size: {largest_component}")
        print(f"connected components with size>=5: {n_components_ge5}")
        print("=" * 72)

    # Local VDJdb slim asset currently contains 39 unique human TRB sequences
    # for EPLPQGQLTAY with HLA-B*35.
    assert len(ref_sequences) == 39
    assert len(enriched) >= 20
    assert len(enriched_vdjdb) >= 3
    assert len(component_nodes) >= 20
    assert len(component_vdjdb) >= 5
    assert component_overlap_fraction >= 0.12
    assert enriched_overlap_fraction >= 0.08
    assert largest_component >= 20
    assert n_components_ge5 >= 1

    max_s = benchmark_max_seconds(default=900.0)
    assert elapsed < max_s
