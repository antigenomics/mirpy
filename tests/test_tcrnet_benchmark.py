"""Benchmark for TCRNET-like enrichment with GIL-like motif spikes.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_tcrnet_benchmark.py -s
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from mir.biomarkers.tcrnet import compute_tcrnet
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire
from tests.conftest import benchmark_max_seconds, skip_benchmarks


_AA = "ACDEFGHIKLMNPQRSTVWY"


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
