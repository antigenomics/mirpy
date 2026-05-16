"""Execution-oriented overlap benchmark for large aging cohorts.

This benchmark measures serial runtimes on a small subset of aging repertoires
and extrapolates expected full-cohort runtime for notebook planning.

Run with:
    RUN_BENCHMARK=1 pytest -s tests/test_overlap_execution_benchmark.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mir.common.parser import VDJtoolsParser
from mir.common.repertoire import LocusRepertoire
from tests.benchmark_helpers import many_vs_many_sample_overlap, many_vs_pool_sample_overlap
from tests.conftest import skip_benchmarks


_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "notebooks" / "assets" / "large" / "airr_benchmark" / "vdjtools"
_META = _DATA / "metadata_aging.txt"


def _load_subset(max_samples: int = 8) -> tuple[list[LocusRepertoire], list[str], list[int]]:
    import polars as pl

    if not _META.exists():
        pytest.skip(f"Missing benchmark metadata at {_META}")

    meta = pl.read_csv(_META, separator="\t")
    rename_map = {c: c.lstrip("#") for c in meta.columns if c.lstrip("#") != c}
    if rename_map:
        meta = meta.rename(rename_map)

    parser = VDJtoolsParser(sep="\t")
    reps: list[LocusRepertoire] = []
    sample_ids: list[str] = []
    ages: list[int] = []

    for row in meta.iter_rows(named=True):
        if len(reps) >= max_samples:
            break
        file_name = str(row["file_name"])
        p = _DATA / file_name
        if not p.exists():
            p = _DATA / f"{file_name}.gz"
        if not p.exists():
            p = _DATA / file_name.replace(".txt", ".txt.gz")
        if not p.exists():
            continue

        clones = parser.parse(str(p))
        reps.append(LocusRepertoire(clonotypes=clones, locus="TRB"))
        sample_ids.append(str(row["sample_id"]))
        ages.append(int(row["age"]))

    if len(reps) < 4:
        pytest.skip("Need at least 4 aging repertoires for benchmark")

    return reps, sample_ids, ages


@skip_benchmarks
@pytest.mark.benchmark
class TestOverlapExecutionBenchmark:
    def test_serial_pair_and_pool_extrapolation(self) -> None:
        reps, sample_ids, ages = _load_subset(max_samples=8)
        pool = LocusRepertoire(clonotypes=[c for r in reps for c in r.clonotypes], locus="TRB")

        n = len(reps)
        subset_pairs = n * (n - 1) // 2
        full_n = 79
        full_pairs = full_n * (full_n - 1) // 2

        # Exact many-vs-many serial
        t0 = time.perf_counter()
        df_pairs = many_vs_many_sample_overlap(
            reps,
            sample_ids=sample_ids,
            metric="exact",
            threshold=0,
            overlap_space="aavj",
            n_jobs=1,
        )
        t_pairs = time.perf_counter() - t0
        per_pair_s = t_pairs / max(1, subset_pairs)
        extrapolated_full_pairs_s = per_pair_s * full_pairs

        # Exact many-vs-pool serial
        t0 = time.perf_counter()
        df_pool = many_vs_pool_sample_overlap(
            reps,
            pool,
            sample_ids=sample_ids,
            ages=ages,
            metric="exact",
            threshold=0,
            overlap_space="aavj",
            n_jobs=1,
        )
        t_pool = time.perf_counter() - t0
        per_sample_pool_s = t_pool / max(1, n)
        extrapolated_full_pool_s = per_sample_pool_s * full_n

        print("\n=== Overlap execution benchmark (serial, subset) ===")
        print(f"subset samples: {n}, subset pairs: {subset_pairs}")
        print(f"many-vs-many exact: {t_pairs:.2f}s total, {per_pair_s:.4f}s per pair")
        print(f"extrapolated full many-vs-many (79 donors): {extrapolated_full_pairs_s/60:.2f} min")
        print(f"many-vs-pool exact: {t_pool:.2f}s total, {per_sample_pool_s:.4f}s per sample")
        print(f"extrapolated full many-vs-pool (79 donors): {extrapolated_full_pool_s:.2f}s")

        assert len(df_pairs) == subset_pairs
        assert len(df_pool) == n

    def test_thread_parallel_subset(self) -> None:
        reps, sample_ids, _ = _load_subset(max_samples=8)

        t0 = time.perf_counter()
        df = many_vs_many_sample_overlap(
            reps,
            sample_ids=sample_ids,
            metric="exact",
            threshold=0,
            overlap_space="aavj",
            n_jobs=4,
        )
        elapsed = time.perf_counter() - t0
        print("\nthread-parallel many-vs-many subset")
        print(f"rows={len(df)} elapsed={elapsed:.2f}s effective_jobs={int(df['n_jobs_effective'].iloc[0])}")

        assert len(df) == len(reps) * (len(reps) - 1) // 2
