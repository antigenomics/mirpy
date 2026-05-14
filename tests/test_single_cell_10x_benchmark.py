"""Benchmark and concordance tests for 10x paired-chain loading on AIRR benchmark data."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import polars as pl
import psutil
import pytest

from mir.common.single_cell import load_10x_vdj_v1_donor
from tests.prepare_airr_benchmark_data import ensure_test_data

RUN_BENCHMARK = os.getenv("RUN_BENCHMARK") == "1"
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_BENCHMARK, reason="set RUN_BENCHMARK=1 to run benchmark tests"),
]


def _discover_donors() -> list[tuple[str, Path, Path]]:
    dcode_root = Path(__file__).resolve().parents[1] / "airr_benchmark" / "dcode"
    if not dcode_root.exists():
        return []

    all_contig_files = sorted(dcode_root.glob("*_all_contig_annotations.csv.gz"))
    consensus_files = sorted(dcode_root.glob("*_consensus_annotations.csv.gz"))
    by_key = {
        p.name.replace("_consensus_annotations.csv.gz", ""): p
        for p in consensus_files
    }
    donors: list[tuple[str, Path, Path]] = []
    for all_contig in all_contig_files:
        key = all_contig.name.replace("_all_contig_annotations.csv.gz", "")
        consensus = by_key.get(key)
        if consensus is not None:
            donors.append((key, all_contig, consensus))
    return donors


def _measure_load(fn):
    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    payload = fn()
    dt = time.perf_counter() - t0
    gc.collect()
    rss_after = proc.memory_info().rss
    rss_delta = max(0, rss_after - rss_before)
    return payload, dt, rss_delta


def _quadrants_from_mirpy(chain_multiplicity: pl.DataFrame) -> dict[tuple[str, str], int]:
    out = (
        chain_multiplicity
        .filter(pl.col("locus_pair") == "TRA_TRB")
        .with_columns(
            pl.when(pl.col("n_chain1") > 0).then(pl.lit("TRA+")).otherwise(pl.lit("TRA-")).alias("tra"),
            pl.when(pl.col("m_chain2") > 0).then(pl.lit("TRB+")).otherwise(pl.lit("TRB-")).alias("trb"),
        )
        .group_by(["tra", "trb"])
        .agg(pl.sum("cell_count").alias("cells"))
    )
    return {(r["tra"], r["trb"]): int(r["cells"]) for r in out.to_dicts()}


def _quadrants_from_scirpy(all_contig: Path) -> dict[tuple[str, str], int]:
    scirpy = pytest.importorskip("scirpy")
    _ = pytest.importorskip("scanpy")
    _ = pytest.importorskip("awkward")
    import awkward as ak

    adata = scirpy.io.read_10x_vdj(all_contig, filtered=False)
    airr = adata.obsm["airr"]
    n_tra = ak.to_numpy(ak.sum(airr.locus == "TRA", axis=1))
    n_trb = ak.to_numpy(ak.sum(airr.locus == "TRB", axis=1))

    out = (
        pl.DataFrame({"n_tra": n_tra, "n_trb": n_trb})
        .with_columns(
            pl.when(pl.col("n_tra") > 0).then(pl.lit("TRA+")).otherwise(pl.lit("TRA-")).alias("tra"),
            pl.when(pl.col("n_trb") > 0).then(pl.lit("TRB+")).otherwise(pl.lit("TRB-")).alias("trb"),
        )
        .group_by(["tra", "trb"])
        .len()
        .rename({"len": "cells"})
    )
    return {(r["tra"], r["trb"]): int(r["cells"]) for r in out.to_dicts()}


def test_10x_loader_speed_memory_and_counts() -> None:
    ensure_test_data(force=False, verbose=False)
    donors = _discover_donors()
    if not donors:
        pytest.skip("No dcode donor assets found under airr_benchmark/dcode")

    timings: list[float] = []
    rss_deltas: list[int] = []

    for donor_id, all_contig, consensus in donors:
        donor, dt, rss_delta = _measure_load(
            lambda: load_10x_vdj_v1_donor(
                consensus_annotations_path=consensus,
                all_contig_annotations_path=all_contig,
                donor_id=donor_id,
            )
        )
        timings.append(dt)
        rss_deltas.append(rss_delta)

        # Core integrity asserts on loaded objects.
        assert donor.loaded_cell_count > 0
        assert donor.loaded_clonotype_count > 0
        assert donor.chain_multiplicity.height > 0
        assert donor.paired_locus_repertoires["TRA_TRB"].clonotype_count > 0

        # Performance guardrails (generous for CI variability).
        assert dt < 30.0
        assert rss_delta < 700 * 1024 * 1024

    assert max(timings) < 30.0
    assert max(rss_deltas) < 700 * 1024 * 1024


def test_10x_loader_concordance_and_speed_vs_scirpy() -> None:
    ensure_test_data(force=False, verbose=False)
    donors = _discover_donors()
    if not donors:
        pytest.skip("No dcode donor assets found under airr_benchmark/dcode")

    donor_id, all_contig, consensus = donors[0]

    donor, mir_time, mir_rss = _measure_load(
        lambda: load_10x_vdj_v1_donor(
            consensus_annotations_path=consensus,
            all_contig_annotations_path=all_contig,
            donor_id=donor_id,
        )
    )

    _, sc_time, sc_rss = _measure_load(lambda: _quadrants_from_scirpy(all_contig))
    mir_quads = _quadrants_from_mirpy(donor.chain_multiplicity)
    sc_quads = _quadrants_from_scirpy(all_contig)

    # Speed/memory comparison assertions.
    assert mir_time < sc_time
    assert mir_rss <= int(sc_rss * 1.5 + 200 * 1024 * 1024)

    # Concordance assertions: same dominant quadrant and qualitative pattern.
    mir_dominant = max(mir_quads.items(), key=lambda kv: kv[1])[0]
    sc_dominant = max(sc_quads.items(), key=lambda kv: kv[1])[0]
    assert mir_dominant == sc_dominant == ("TRA+", "TRB+")

    for quadrant in (("TRA+", "TRB+"), ("TRA+", "TRB-"), ("TRA-", "TRB+")):
        assert mir_quads.get(quadrant, 0) > 0
        assert sc_quads.get(quadrant, 0) > 0

    rel_gap = abs(mir_quads[("TRA+", "TRB+")] - sc_quads[("TRA+", "TRB+")]) / max(
        mir_quads[("TRA+", "TRB+")],
        sc_quads[("TRA+", "TRB+")],
    )
    assert rel_gap < 0.3
