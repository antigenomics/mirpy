"""Benchmark tests for neighborhood enrichment on real repertoires.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_neighborhood_enrichment_benchmark.py -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from mir.common.control import ControlManager
from mir.common.parser import VDJtoolsParser
from mir.common.pool import pool_samples
from mir.common.repertoire import SampleRepertoire
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats
from tests.benchmark_helpers import (
    load_gilg_target_repertoire,
    synthetic_control_repertoire,
    synthetic_control_size,
)
from tests.conftest import benchmark_repertoire_workers, skip_benchmarks

REAL_REPS = Path(__file__).parent / "real_repertoires"


@skip_benchmarks
def test_neighborhood_shorter_sequences_more_neighbors(capsys) -> None:
    """Benchmark: shorter junction_aa sequences have more neighbors.

    Load a couple of samples from real_repertoires/A* and compute neighborhood
    statistics. Verify that clonotypes with shorter junction_aa lengths tend
    to have more neighbors.
    """
    # Find A* files (A2, A3, A4 in real_repertoires)
    a_files = sorted(REAL_REPS.glob("A[2-9]-*.txt.gz"))[:2]
    if len(a_files) < 2:
        pytest.skip("Need at least 2 A[2-9]-*.txt.gz files in real_repertoires/")

    workers = benchmark_repertoire_workers(default="4")[0]
    parser = VDJtoolsParser()

    t0 = time.perf_counter()
    samples = []
    for fpath in a_files:
        clonotypes = parser.parse(str(fpath))
        srep = SampleRepertoire.from_clonotypes(clonotypes, sample_id=fpath.stem)
        samples.append(srep)
    load_s = time.perf_counter() - t0

    # Compute neighborhood stats for first sample
    sample = samples[0]
    if isinstance(sample, SampleRepertoire):
        locus_rep = sample.loci.get("TRB") or next(iter(sample.loci.values()))
    else:
        locus_rep = sample

    t0 = time.perf_counter()
    stats_serial = compute_neighborhood_stats(
        locus_rep,
        metric="hamming",
        threshold=1,
        match_v_gene=False,
        n_jobs=1,
    )
    compute_serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats = compute_neighborhood_stats(
        locus_rep,
        metric="hamming",
        threshold=1,
        match_v_gene=False,
        n_jobs=workers,
    )
    compute_s = time.perf_counter() - t0
    assert stats == stats_serial

    # Extract clonotypes and their stats
    clonotypes = locus_rep.clonotypes
    records = []
    for cl in clonotypes:
        if cl.sequence_id in stats:
            stat = stats[cl.sequence_id]
            records.append({
                "sequence_id": cl.sequence_id,
                "junction_aa_len": len(cl.junction_aa) if cl.junction_aa else 0,
                "neighbor_count": stat["neighbor_count"],
                "duplicate_count": cl.duplicate_count,
            })

    df = pd.DataFrame(records)

    # Compute correlation: shorter junction_aa should correlate with more neighbors
    if len(df) >= 10:
        # Group by length and compute mean neighbors
        len_groups = df.groupby("junction_aa_len")["neighbor_count"].agg(["mean", "count"])

        # Shorter sequences should have higher mean neighbor count (on average)
        short_seqs = df[df["junction_aa_len"] <= df["junction_aa_len"].quantile(0.33)]["neighbor_count"]
        long_seqs = df[df["junction_aa_len"] >= df["junction_aa_len"].quantile(0.67)]["neighbor_count"]

        if len(short_seqs) >= 5 and len(long_seqs) >= 5:
            short_mean = short_seqs.mean()
            long_mean = long_seqs.mean()

            with capsys.disabled():
                print("\n" + "=" * 76)
                print("Neighborhood enrichment benchmark on real repertoires")
                print(f"Loaded {len(samples)} samples in {load_s:.2f}s")
                print(f"Computed neighborhood stats serial in {compute_serial_s:.2f}s")
                print(f"Computed neighborhood stats parallel({workers}) in {compute_s:.2f}s")
                if compute_s > 0:
                    print(f"Speedup: {compute_serial_s / compute_s:.2f}x")
                print(f"Sample: {sample.sample_id}, Locus: {locus_rep.locus}")
                print(f"Total clonotypes: {len(clonotypes)}")
                print(f"Clonotypes with stats: {len(df)}")
                print(f"\nJunction_aa length groups:")
                print(len_groups.to_string())
                print(f"\nNeighbor count comparison:")
                print(f"  Short (≤33rd percentile): mean={short_mean:.2f}, n={len(short_seqs)}")
                print(f"  Long (≥67th percentile):  mean={long_mean:.2f}, n={len(long_seqs)}")
                print(f"  Ratio (short/long): {short_mean/long_mean:.2f}x")
                print("=" * 76)

            assert short_mean > long_mean, (
                f"Expected shorter sequences to have more neighbors on average, "
                f"got short_mean={short_mean:.2f}, long_mean={long_mean:.2f}"
            )


@skip_benchmarks
def test_pooled_repertoire_convergence_by_length(capsys) -> None:
    """Benchmark: pool two samples and verify convergence pattern by length.

    Shorter junction_aa sequences should show higher convergence (more incidence)
    when pooled across samples. Convergence is defined as incidence / pool size.
    """
    # Find A* files (A2, A3, A4 in real_repertoires)
    a_files = sorted(REAL_REPS.glob("A[2-9]-*.txt.gz"))[:2]
    if len(a_files) < 2:
        pytest.skip("Need at least 2 A[2-9]-*.txt.gz files in real_repertoires/")

    workers = benchmark_repertoire_workers(default="4")[0]
    parser = VDJtoolsParser()

    t0 = time.perf_counter()
    samples = []
    for fpath in a_files:
        clonotypes = parser.parse(str(fpath))
        srep = SampleRepertoire.from_clonotypes(clonotypes, sample_id=fpath.stem)
        samples.append(srep)
    load_s = time.perf_counter() - t0

    # Pool the samples
    t0 = time.perf_counter()
    pooled = pool_samples(samples, rule="aavj", weighted=True, include_sample_ids=False)
    pool_s = time.perf_counter() - t0

    # Extract pooled clonotypes and compute convergence
    if isinstance(pooled, SampleRepertoire):
        locus_rep = pooled.loci.get("TRB") or next(iter(pooled.loci.values()))
    else:
        locus_rep = pooled

    pool_size = len(samples)
    records = []
    for cl in locus_rep.clonotypes:
        incidence = cl.clone_metadata.get("incidence", 0)
        occurrences = cl.clone_metadata.get("occurrences", 0)
        if incidence > 0 and occurrences > 0:
            convergence = incidence / pool_size
            records.append({
                "sequence_id": cl.sequence_id,
                "junction_aa_len": len(cl.junction_aa) if cl.junction_aa else 0,
                "incidence": incidence,
                "occurrences": occurrences,
                "convergence": convergence,
                "duplicate_count": cl.duplicate_count,
            })

    df = pd.DataFrame(records)

    if len(df) >= 10:
        # Group by length and compute stats
        len_groups = df.groupby("junction_aa_len").agg({
            "convergence": ["mean", "count"],
            "incidence": "mean",
            "occurrences": "mean",
        }).round(4)

        # Compare short vs long
        short_df = df[df["junction_aa_len"] <= df["junction_aa_len"].quantile(0.33)]
        long_df = df[df["junction_aa_len"] >= df["junction_aa_len"].quantile(0.67)]

        if len(short_df) >= 5 and len(long_df) >= 5:
            short_conv = short_df["convergence"].mean()
            long_conv = long_df["convergence"].mean()

            with capsys.disabled():
                print("\n" + "=" * 76)
                print("Pooled repertoire convergence by sequence length")
                print(f"Loaded {len(samples)} samples in {load_s:.2f}s")
                print(f"Pooled in {pool_s:.2f}s")
                print(f"Locus: {locus_rep.locus}, Pool size: {pool_size}")
                print(f"Total pooled clonotypes: {len(locus_rep.clonotypes)}")
                print(f"Clonotypes with stats: {len(df)}")
                print(f"\nLength group convergence:")
                print(len_groups.to_string())
                print(f"\nConvergence comparison:")
                print(f"  Short (≤33rd percentile): mean={short_conv:.4f}, n={len(short_df)}")
                print(f"  Long (≥67th percentile):  mean={long_conv:.4f}, n={len(long_df)}")
                if long_conv > 0:
                    print(f"  Ratio (short/long): {short_conv/long_conv:.2f}x")
                print("=" * 76)

            assert short_conv > long_conv, (
                f"Expected shorter sequences to have higher convergence, "
                f"got short={short_conv:.4f}, long={long_conv:.4f}"
            )


@skip_benchmarks
def test_neighborhood_runtime_gilg_vs_synthetic_1m(capsys) -> None:
    """Benchmark neighborhood runtime for GIL target against synthetic control (1e6 by default)."""
    workers = benchmark_repertoire_workers(default="1,4")
    manager = ControlManager()
    n_control = synthetic_control_size(default=1_000_000)
    require_cached = os.getenv("MIRPY_BENCH_REQUIRE_CACHED_CONTROL", "1") != "0"

    target = load_gilg_target_repertoire()
    control = synthetic_control_repertoire(
        manager=manager,
        species="human",
        locus="TRB",
        n=n_control,
        require_cached=require_cached,
    )

    runtimes: dict[int, float] = {}
    baseline = None
    for w in workers:
        t0 = time.perf_counter()
        stats = compute_neighborhood_stats(
            target,
            background=control,
            metric="hamming",
            threshold=1,
            match_v_gene=False,
            match_j_gene=False,
            n_jobs=w,
        )
        elapsed = time.perf_counter() - t0
        runtimes[w] = elapsed
        if baseline is None:
            baseline = stats
        else:
            assert stats == baseline

    with capsys.disabled():
        print("\n" + "=" * 76)
        print("Neighborhood benchmark: GIL target vs synthetic control")
        print(f"target clonotypes: {len(target.clonotypes)}")
        print(f"control clonotypes: {len(control.clonotypes)} (requested n={n_control})")
        for w in workers:
            print(f"runtime n_jobs={w}: {runtimes[w]:.3f}s")
        if 1 in runtimes:
            for w in workers:
                if w == 1:
                    continue
                if runtimes[w] > 0:
                    print(f"speedup 1->{w}: {runtimes[1] / runtimes[w]:.2f}x")
        print("=" * 76)

    assert baseline is not None
    assert len(baseline) == len(target.clonotypes)
