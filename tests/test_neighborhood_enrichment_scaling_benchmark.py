"""Scaling benchmarks for self-neighborhood enrichment.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_neighborhood_enrichment_scaling_benchmark.py -s

The standard benchmark covers repertoire sizes 1e2, 1e3, 1e4, and 1e5 on
1/4/8 workers. The 1e6 case is isolated in a very-slow benchmark and enabled
with RUN_FULL_BENCHMARK=1 or MIRPY_BENCH_INCLUDE_1M=1.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_repertoire_workers, skip_benchmarks

_AA = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype="U1")


def _generate_clustered_repertoire(n: int, *, seed: int = 42) -> LocusRepertoire:
    rng = np.random.default_rng(seed)
    motif_count = max(32, min(2048, max(1, n // 64)))
    motif_len = 14
    motifs = rng.choice(_AA, size=(motif_count, motif_len))
    motifs[:, 0] = "C"
    motifs[:, -1] = "F"

    clones: list[Clonotype] = []
    for i in range(int(n)):
        seq = motifs[i % motif_count].copy()
        if i % 3 != 0:
            pos = int(rng.integers(1, motif_len - 1))
            seq[pos] = str(rng.choice(_AA))
        if i % 11 == 0:
            pos = int(rng.integers(1, motif_len - 1))
            seq[pos] = str(rng.choice(_AA))
        junction_aa = "".join(seq.tolist())
        clones.append(
            Clonotype(
                sequence_id=str(i),
                locus="TRB",
                junction_aa=junction_aa,
                v_gene="TRBV5-1*01",
                j_gene="TRBJ2-7*01",
                duplicate_count=1,
                _validate=False,
            )
        )
    return LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=f"synthetic-self-{n}")


@skip_benchmarks
@pytest.mark.benchmark
def test_neighborhood_self_scaling_up_to_1e5(capsys) -> None:
    sizes = [100, 1_000, 10_000, 100_000]
    workers = benchmark_repertoire_workers(default="1,4,8")
    runtimes: list[dict[str, float | int]] = []

    with capsys.disabled():
        print("\n" + "=" * 84)
        print("Neighborhood self-scaling benchmark up to 1e5 clonotypes")

    for n in sizes:
        rep = _generate_clustered_repertoire(n)
        baseline = None
        for w in workers:
            t0 = time.perf_counter()
            stats = compute_neighborhood_stats(
                rep,
                metric="hamming",
                threshold=1,
                match_v_gene=False,
                match_j_gene=False,
                n_jobs=w,
            )
            elapsed = time.perf_counter() - t0
            runtimes.append({"size": n, "workers": w, "elapsed_s": elapsed})
            if baseline is None and n <= 10_000:
                baseline = stats
            elif baseline is not None and n <= 10_000:
                assert stats == baseline

        benchmark_log_line(
            "neighborhood_self_scaling "
            f"size={n} " + " ".join(
                f"w{int(row['workers'])}={float(row['elapsed_s']):.3f}s"
                for row in runtimes
                if int(row["size"]) == n
            )
        )

    df = np.array([(int(r["size"]), int(r["workers"]), float(r["elapsed_s"])) for r in runtimes], dtype=float)
    with capsys.disabled():
        for n in sizes:
            rows = [r for r in runtimes if int(r["size"]) == n]
            print(
                f"size={n}: " + ", ".join(
                    f"n_jobs={int(r['workers'])} -> {float(r['elapsed_s']):.3f}s" for r in rows
                )
            )
        print("=" * 84)

    assert len(df) == len(sizes) * len(workers)


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.very_slow_benchmark
def test_neighborhood_self_scaling_1e6(capsys) -> None:
    if not (
        os.getenv("RUN_FULL_BENCHMARK") == "1"
        or os.getenv("RUN_FULL_BENCHMARKS") == "1"
        or os.getenv("MIRPY_BENCH_INCLUDE_1M") == "1"
    ):
        pytest.skip("Enable RUN_FULL_BENCHMARK=1 or MIRPY_BENCH_INCLUDE_1M=1 for the 1e6 self-scaling benchmark")

    workers = benchmark_repertoire_workers(default="1,4,8")
    rep = _generate_clustered_repertoire(1_000_000)
    runtimes: list[tuple[int, float]] = []
    with capsys.disabled():
        print("\n" + "=" * 84)
        print("Neighborhood self-scaling benchmark at 1e6 clonotypes")
    for w in workers:
        t0 = time.perf_counter()
        stats = compute_neighborhood_stats(
            rep,
            metric="hamming",
            threshold=1,
            match_v_gene=False,
            match_j_gene=False,
            n_jobs=w,
        )
        elapsed = time.perf_counter() - t0
        runtimes.append((w, elapsed))
        assert len(stats) == 1_000_000
    benchmark_log_line(
        "neighborhood_self_scaling size=1000000 " + " ".join(f"w{w}={elapsed:.3f}s" for w, elapsed in runtimes)
    )
    with capsys.disabled():
        for w, elapsed in runtimes:
            print(f"size=1000000 n_jobs={w} -> {elapsed:.3f}s")
        print("=" * 84)
