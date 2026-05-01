"""Benchmark: parallel loading of real repertoires from real_repertoires/.

Run with:  RUN_BENCHMARK=1 pytest tests/test_repertoire_benchmark.py -s

Reports throughput and memory for increasing worker counts so the
bottleneck (I/O vs CPU-bound GIL) is immediately visible.
Target: 1000 samples x 100k clonotypes loaded in < 10 minutes.
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path

import pytest

from mir.common.parser import VDJtoolsParser
from mir.common.repertoire_dataset import RepertoireDataset
from tests.conftest import (
    benchmark_max_seconds,
    benchmark_repertoire_workers,
    benchmark_track_memory,
    skip_benchmarks,
)

REAL_REPS = Path(__file__).parent / "real_repertoires"
METADATA  = REAL_REPS / "metadata_aging.txt"


def _load(n_workers: int) -> dict:
    gc.collect()
    track_mem = benchmark_track_memory(default=False)
    if track_mem:
        tracemalloc.start()
    t0 = time.perf_counter()

    ds = RepertoireDataset.from_folder_polars(
        REAL_REPS,
        parser=VDJtoolsParser(),
        metadata_file="metadata_aging.txt",
        file_name_column="file_name",
        sample_id_column="sample_id",
        metadata_sep="\t",
        skip_missing_files=True,
        n_workers=n_workers,
        progress=False,
    )

    elapsed = time.perf_counter() - t0
    if track_mem:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        peak_bytes = 0

    n_samples     = len(ds.samples)
    total_clones  = sum(len(lr.clonotypes)
                        for srep in ds.samples.values()
                        for lr in srep.loci.values())
    peak_mb       = peak_bytes / (1024 ** 2)

    del ds
    gc.collect()

    return dict(
        n_workers=n_workers,
        n_samples=n_samples,
        total_clones=total_clones,
        elapsed_s=elapsed,
        clones_per_s=total_clones / elapsed if elapsed > 0 else 0,
        samples_per_s=n_samples / elapsed if elapsed > 0 else 0,
        peak_mb=peak_mb,
        mb_per_1k=peak_mb / (total_clones / 1000) if total_clones > 0 else 0,
        # Extrapolated time for 1000 samples x 100k clonotypes = 100M clonotypes
        eta_1000x100k_min=(100_000_000 / (total_clones / elapsed)) / 60
                           if total_clones > 0 and elapsed > 0 else float("inf"),
    )


@skip_benchmarks
def test_benchmark_parallel_load_aging(capsys):
    """Load aging cohort at 1 / 4 / 8 workers; print throughput table."""
    workers = benchmark_repertoire_workers(default="1,4")
    per_case_upper_s = benchmark_max_seconds(default=30.0)

    t_all = time.perf_counter()
    results = []
    for w in workers:
        r = _load(w)
        assert r["elapsed_s"] <= per_case_upper_s, (
            f"Benchmark case exceeded upper bound: worker={w}, "
            f"elapsed={r['elapsed_s']:.2f}s > {per_case_upper_s:.2f}s"
        )
        results.append(r)
    elapsed_all = time.perf_counter() - t_all
    total_upper_s = per_case_upper_s * len(workers) + 5.0

    with capsys.disabled():
        print("\n" + "=" * 76)
        print(f"  RepertoireDataset I/O benchmark  ({METADATA.name})")
        print(f"  Runtime upper bound: <= {total_upper_s:.1f}s "
              f"({per_case_upper_s:.1f}s x {len(workers)} cases + 5s overhead)")
        print("=" * 76)
        hdr = (f"  {'workers':>7}  {'samples':>7}  {'clonotypes':>11}  "
               f"{'elapsed_s':>9}  {'samples/s':>9}  {'clones/s':>10}  "
               f"{'MB':>7}  {'MB/1kCl':>8}  {'ETA@1kx100k':>12}")
        print(hdr)
        print("  " + "-" * 74)
        for r in results:
            eta = f"{r['eta_1000x100k_min']:.1f} min"
            print(
                f"  {r['n_workers']:>7}  {r['n_samples']:>7}  "
                f"{r['total_clones']:>11,}  {r['elapsed_s']:>9.2f}  "
                f"{r['samples_per_s']:>9.1f}  {r['clones_per_s']:>10,.0f}  "
                f"{r['peak_mb']:>7.1f}  {r['mb_per_1k']:>8.3f}  {eta:>12}"
            )
        print("=" * 76)
        best = max(results, key=lambda r: r["clones_per_s"])
        print(f"  Best: {best['n_workers']} workers  "
              f"{best['clones_per_s']:,.0f} clonotypes/s  "
              f"ETA for 100M clonotypes: {best['eta_1000x100k_min']:.1f} min")
        print(f"  Total benchmark runtime: {elapsed_all:.2f}s "
              f"(upper bound {total_upper_s:.2f}s)")
        print("=" * 76)

    best = max(results, key=lambda r: r["clones_per_s"])
    assert best["n_samples"] > 0, "No samples loaded"
    assert best["total_clones"] > 0, "No clonotypes loaded"
    assert elapsed_all <= total_upper_s, (
        f"Benchmark suite exceeded upper bound: "
        f"{elapsed_all:.2f}s > {total_upper_s:.2f}s"
    )
    # Fail loudly if still way off target (>60 min extrapolated)
    assert best["eta_1000x100k_min"] < 60, (
        f"Too slow: {best['eta_1000x100k_min']:.1f} min extrapolated for 100M clonotypes "
        f"(target <10 min, hard fail at 60 min)"
    )
