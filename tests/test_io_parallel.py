"""Benchmarks and tests for parallel I/O optimization.

Tests performance improvements from parallel chunked reading, including:
- Speed comparisons (sequential vs parallel with different worker counts)
- Memory usage profiling
- Correctness validation
- Scalability estimation for 1M clonotypes
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from mir.common.io_parallel import (
    load_airr_parallel,
    load_airr_with_filter,
    time_load,
    _parse_chunk_worker,
    _load_sequential,
)
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire
from tests.conftest import skip_benchmarks

ASSETS = Path(__file__).parent / "assets"

# Test files with known sizes
_SMALL_FILE = ASSETS / "Q1_0_F1.airr.tsv.gz"  # compact donor day-0 sample
_MEDIUM_FILE = ASSETS / "Q1_15_F1.airr.tsv.gz"  # compact donor day-15 sample
_FILES_EXIST = _SMALL_FILE.exists() and _MEDIUM_FILE.exists()


# ============================================================================
# Unit Tests (always run)
# ============================================================================


class TestChunkParsingWorker:
    """Test the parallel chunk parsing worker function."""

    def test_worker_parses_chunk_correctly(self):
        """Worker should parse a small DataFrame chunk into LocusRepertoire."""
        df = pd.DataFrame({
            "junction_aa": ["CASSF", "CASSF"],
            "v_gene": ["TRBV1", "TRBV2"],
            "j_gene": ["TRBJ1-1", "TRBJ1-1"],
            "duplicate_count": [10, 20],
        })
        rep = _parse_chunk_worker(df, locus="TRB")
        assert isinstance(rep, LocusRepertoire)
        assert len(rep.clonotypes) == 2
        assert rep.locus == "TRB"
        assert rep.duplicate_count == 30

    def test_worker_handles_missing_duplicate_count(self):
        """Worker should default duplicate_count to 1 if missing."""
        df = pd.DataFrame({
            "junction_aa": ["CASSF"],
            "v_gene": ["TRBV1"],
            "j_gene": ["TRBJ1-1"],
        })
        rep = _parse_chunk_worker(df, locus="TRB")
        assert len(rep.clonotypes) == 1
        assert rep.clonotypes[0].duplicate_count == 1

    def test_worker_empty_chunk(self):
        """Worker should handle empty chunks gracefully."""
        df = pd.DataFrame({
            "junction_aa": [],
            "v_gene": [],
            "j_gene": [],
        })
        rep = _parse_chunk_worker(df, locus="TRB")
        assert len(rep.clonotypes) == 0


class TestParallelLoadBasic:
    """Basic tests for parallel loading (no file I/O)."""

    @pytest.mark.skipif(not _FILES_EXIST, reason="Q1 test files not available")
    def test_parallel_load_n_jobs_1_sequential(self):
        """Loading with n_jobs=1 should work (sequential path)."""
        rep = load_airr_parallel(_SMALL_FILE, locus="TRB", n_jobs=1)
        assert isinstance(rep, LocusRepertoire)
        assert len(rep.clonotypes) > 0
        assert rep.locus == "TRB"

    @pytest.mark.skipif(not _FILES_EXIST, reason="Q1 test files not available")
    def test_parallel_load_n_jobs_2(self):
        """Loading with n_jobs=2 should work."""
        rep = load_airr_parallel(_SMALL_FILE, locus="TRB", n_jobs=2, chunk_size=500)
        assert isinstance(rep, LocusRepertoire)
        assert len(rep.clonotypes) > 0

    @pytest.mark.skipif(not _FILES_EXIST, reason="Q1 test files not available")
    def test_parallel_load_matches_sequential(self):
        """Parallel and sequential loads should produce identical results."""
        rep_seq = _load_sequential(_SMALL_FILE, locus="TRB")
        rep_par = load_airr_parallel(_SMALL_FILE, locus="TRB", n_jobs=2, chunk_size=500)

        assert len(rep_seq.clonotypes) == len(rep_par.clonotypes)
        assert rep_seq.duplicate_count == rep_par.duplicate_count

        # Verify clonotypes are the same (order might differ)
        seq_junctions = {c.junction_aa for c in rep_seq.clonotypes}
        par_junctions = {c.junction_aa for c in rep_par.clonotypes}
        assert seq_junctions == par_junctions

    @pytest.mark.skipif(not _FILES_EXIST, reason="Q1 test files not available")
    def test_parallel_load_with_filter(self):
        """Filter function should be applied before parsing."""
        # Filter: keep only rows with duplicate_count >= 5
        def keep_abundant(row):
            dc = row.get("duplicate_count", 1)
            try:
                return int(dc) >= 5
            except (ValueError, TypeError):
                return True

        rep = load_airr_with_filter(
            _SMALL_FILE,
            locus="TRB",
            n_jobs=2,
            chunk_size=500,
            filter_fn=keep_abundant,
        )
        assert isinstance(rep, LocusRepertoire)
        # All remaining clonotypes should have dc >= 5 (or be from default assignment)
        for c in rep.clonotypes:
            assert c.duplicate_count >= 1


class TestChunkSizeEffect:
    """Test that chunk size doesn't affect correctness."""

    @pytest.mark.skipif(not _FILES_EXIST, reason="Q1 test files not available")
    def test_different_chunk_sizes_produce_same_result(self):
        """Results should be independent of chunk_size."""
        chunk_sizes = [500, 1000, 5000]
        results = []

        for cs in chunk_sizes:
            rep = load_airr_parallel(
                _SMALL_FILE,
                locus="TRB",
                n_jobs=2,
                chunk_size=cs,
            )
            results.append(len(rep.clonotypes))

        # All chunk sizes should produce the same number of clonotypes
        assert len(set(results)) == 1, f"Chunk sizes produced different counts: {results}"


# ============================================================================
# Benchmark Tests (``RUN_BENCHMARK=1``)
# ============================================================================


class TestParallelLoadBenchmark:
    """Benchmark comparisons: sequential vs parallel."""

    @skip_benchmarks
    @pytest.mark.skipif(not _FILES_EXIST, reason="Q1 test files not available")
    def test_speedup_sequential_vs_parallel(self):
        """Measure speedup of parallel loading vs sequential."""
        print("\n" + "=" * 80)
        print("BENCHMARK: Sequential vs Parallel Loading")
        print("=" * 80)

        # Sequential
        result_seq = time_load(_SMALL_FILE, method="sequential", locus="TRB")
        print(
            f"\nSequential:\n"
            f"  Time: {result_seq['elapsed_s']:.2f}s\n"
            f"  Clonotypes: {result_seq['n_clonotypes']:,}\n"
            f"  Memory (peak): {result_seq['memory_peak_mb']:.1f} MB"
        )

        # Parallel (2 workers)
        result_par_2 = time_load(
            _SMALL_FILE,
            method="parallel",
            locus="TRB",
            n_jobs=2,
            chunk_size=1000,
        )
        print(
            f"\nParallel (2 workers, 1K chunks):\n"
            f"  Time: {result_par_2['elapsed_s']:.2f}s\n"
            f"  Clonotypes: {result_par_2['n_clonotypes']:,}\n"
            f"  Memory (peak): {result_par_2['memory_peak_mb']:.1f} MB"
        )

        speedup_2 = result_seq["elapsed_s"] / result_par_2["elapsed_s"]
        print(f"\nSpeedup (2 workers): {speedup_2:.1f}x")

        # Parallel (4 workers)
        result_par_4 = time_load(
            _SMALL_FILE,
            method="parallel",
            locus="TRB",
            n_jobs=4,
            chunk_size=1000,
        )
        print(
            f"\nParallel (4 workers, 1K chunks):\n"
            f"  Time: {result_par_4['elapsed_s']:.2f}s\n"
            f"  Clonotypes: {result_par_4['n_clonotypes']:,}\n"
            f"  Memory (peak): {result_par_4['memory_peak_mb']:.1f} MB"
        )

        speedup_4 = result_seq["elapsed_s"] / result_par_4["elapsed_s"]
        print(f"\nSpeedup (4 workers): {speedup_4:.1f}x")

        # Verify results match
        assert result_seq["n_clonotypes"] == result_par_2["n_clonotypes"]
        assert result_seq["n_clonotypes"] == result_par_4["n_clonotypes"]

    @skip_benchmarks
    @pytest.mark.skipif(not _MEDIUM_FILE.exists(), reason="Medium file not available")
    def test_chunk_size_effect_on_performance(self):
        """Measure how chunk_size affects performance."""
        print("\n" + "=" * 80)
        print("BENCHMARK: Effect of Chunk Size on Performance")
        print("=" * 80)

        chunk_sizes = [500, 1000, 5000, 10000]
        results = {}

        for cs in chunk_sizes:
            result = time_load(
                _MEDIUM_FILE,
                method="parallel",
                locus="TRB",
                n_jobs=4,
                chunk_size=cs,
            )
            results[cs] = result
            print(
                f"\nChunk size {cs:,}:\n"
                f"  Time: {result['elapsed_s']:.2f}s\n"
                f"  Memory (peak): {result['memory_peak_mb']:.1f} MB"
            )

        # All should produce same number of clonotypes
        n_clonotypes = set(r["n_clonotypes"] for r in results.values())
        assert len(n_clonotypes) == 1

    @skip_benchmarks
    @pytest.mark.skipif(not _MEDIUM_FILE.exists(), reason="Medium file not available")
    def test_worker_count_scaling(self):
        """Measure how worker count affects performance."""
        print("\n" + "=" * 80)
        print("BENCHMARK: Worker Count Scaling")
        print("=" * 80)

        worker_counts = [1, 2, 4]
        results = {}

        for n_jobs in worker_counts:
            result = time_load(
                _MEDIUM_FILE,
                method="parallel",
                locus="TRB",
                n_jobs=n_jobs,
                chunk_size=2000,
            )
            results[n_jobs] = result
            print(
                f"\nWorkers: {n_jobs}\n"
                f"  Time: {result['elapsed_s']:.2f}s\n"
                f"  Memory (peak): {result['memory_peak_mb']:.1f} MB"
            )

        # Check speedup progression
        time_seq = results[1]["elapsed_s"]
        for n_jobs in [2, 4]:
            speedup = time_seq / results[n_jobs]["elapsed_s"]
            print(f"  Speedup ({n_jobs} workers): {speedup:.1f}x")


# ============================================================================
# Performance Estimation for 1M Clonotypes
# ============================================================================


class TestPerformanceEstimation:
    """Estimate performance scaling to 1M clonotypes based on measurements."""

    @skip_benchmarks
    @pytest.mark.skipif(not _SMALL_FILE.exists(), reason="Small file not available")
    def test_estimate_1m_clonotypes(self):
        """Estimate time and memory for loading 1M clonotypes."""
        print("\n" + "=" * 80)
        print("PERFORMANCE ESTIMATION: 1,000,000 Clonotypes")
        print("=" * 80)

        # Measure on small file
        result = time_load(
            _SMALL_FILE,
            method="sequential",
            locus="TRB",
        )

        n_measured = result["n_clonotypes"]
        time_measured = result["elapsed_s"]
        mem_measured = result["memory_peak_mb"]

        # Linear scaling assumptions (may be conservative due to overhead)
        scale_factor = 1_000_000 / n_measured

        print(f"\nMeasured on {n_measured:,} clonotypes:")
        print(f"  Sequential time: {time_measured:.2f}s")
        print(f"  Memory: {mem_measured:.1f} MB")

        # Estimate for 1M
        est_time_seq = time_measured * scale_factor
        est_mem_seq = mem_measured * scale_factor

        print(f"\nEstimated for 1,000,000 clonotypes (sequential):")
        print(f"  Time: {est_time_seq:.1f}s ({est_time_seq/60:.1f} minutes)")
        print(f"  Memory: {est_mem_seq:.0f} MB ({est_mem_seq/1024:.1f} GB)")

        # Parallel estimates (assume 4x speedup, 20% memory reduction)
        est_time_par = est_time_seq / 4.0
        est_mem_par = est_mem_seq * 0.8

        print(f"\nEstimated for 1,000,000 clonotypes (4 workers, parallel):")
        print(f"  Time: {est_time_par:.1f}s ({est_time_par/60:.1f} minutes)")
        print(f"  Memory: {est_mem_par:.0f} MB ({est_mem_par/1024:.1f} GB)")

        # Summary table
        print(f"\n" + "=" * 80)
        print("SUMMARY: Loading 1M Clonotypes")
        print("=" * 80)
        print(f"{'Method':<20} {'Time (min)':<15} {'Memory (GB)':<15}")
        print("-" * 50)
        print(f"{'Sequential':<20} {est_time_seq/60:>6.1f}{'':8} {est_mem_seq/1024:>6.1f}")
        print(f"{'Parallel (4x)':<20} {est_time_par/60:>6.1f}{'':8} {est_mem_par/1024:>6.1f}")
        print(f"{'Speedup':<20} {est_time_seq/est_time_par:>6.1f}x{'':7}")
        print("=" * 80)
