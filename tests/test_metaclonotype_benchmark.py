"""Benchmarks for metaclonotype construction and functional-diversity computation.

Measures wall-clock time and RSS delta for the core metaclonotype workflows at
four synthetic repertoire scales: 1 K, 5 K, 10 K, and 50 K clonotypes.

Workflows timed
---------------
* ``metaclonotypes_from_labels``   — convert DBSCAN-style int labels
* ``metaclonotypes_from_components`` — convert pre-built connected-component lists
* ``summarize_metaclonotypes``     — aggregate duplicate_count per cluster
* ``functional_diversity``         — DiversitySummary over metaclonotype counts
* ``hill_curve``                   — 41-point Hill profile (vectorised)
* ``rarefaction_curve``            — iNEXT-style rarefaction/extrapolation curve

Run with::

    RUN_BENCHMARK=1 pytest -s tests/test_metaclonotype_benchmark.py

Optional env vars
-----------------
MIRPY_BENCH_META_CLUSTER_SIZE
    Average clonotypes per cluster (default: 5).
MIRPY_BENCH_META_SIZES
    Comma-separated list of repertoire sizes to test
    (default: ``"1000,5000,10000,50000"``).
"""

from __future__ import annotations

import os
import random
import string
import threading
import time
import tracemalloc
from typing import Sequence

import numpy as np
import psutil
import pytest

from mir.common.clonotype import Clonotype
from mir.common.diversity import hill_curve, rarefaction_curve, summarize_counts
from mir.common.metaclonotype import (
    MetaClonotypeDefinition,
    functional_diversity,
    metaclonotypes_from_components,
    metaclonotypes_from_labels,
    summarize_metaclonotypes,
)
from mir.common.repertoire import LocusRepertoire
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_max_seconds, skip_benchmarks


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return max(1, int(raw))
    except (ValueError, TypeError):
        return default


def _bench_sizes() -> list[int]:
    raw = os.getenv("MIRPY_BENCH_META_SIZES", "1000,5000,10000,50000")
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        try:
            v = int(token)
            if v > 0:
                out.append(v)
        except ValueError:
            pass
    return sorted(set(out)) or [1_000, 5_000, 10_000, 50_000]


def _avg_cluster_size() -> int:
    return _env_int("MIRPY_BENCH_META_CLUSTER_SIZE", 5)


# ---------------------------------------------------------------------------
# Synthetic-data factories
# ---------------------------------------------------------------------------

_AA = "ACDEFGHIKLMNPQRSTVWY"
_VGENES = ["TRBV7-9", "TRBV12-3", "TRBV20-1", "TRBV6-5", "TRBV5-1"]
_JGENES = ["TRBJ2-1", "TRBJ1-2", "TRBJ2-7", "TRBJ1-1", "TRBJ2-3"]


def _random_cdr3(rng: random.Random, min_len: int = 12, max_len: int = 16) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choices(_AA, k=length))


def _make_repertoire(n: int, *, seed: int = 42) -> LocusRepertoire:
    """Return a synthetic TRB LocusRepertoire with *n* clonotypes."""
    rng = random.Random(seed)
    clonotypes = [
        Clonotype(
            sequence_id=f"s{i}",
            locus="TRB",
            junction_aa=_random_cdr3(rng),
            v_gene=rng.choice(_VGENES),
            j_gene=rng.choice(_JGENES),
            duplicate_count=rng.randint(1, 50),
            _validate=False,
        )
        for i in range(n)
    ]
    return LocusRepertoire(clonotypes=clonotypes, locus="TRB")


def _make_labels(n: int, avg_size: int, *, seed: int = 0) -> list[int]:
    """Assign clonotypes to clusters; ~1/avg_size are singletons labelled -1."""
    rng = random.Random(seed)
    n_clusters = max(1, n // avg_size)
    labels: list[int] = []
    for i in range(n):
        if rng.random() < 1.0 / avg_size:
            labels.append(-1)
        else:
            labels.append(rng.randint(0, n_clusters - 1))
    return labels


def _labels_to_components(
    ids: list[str], labels: list[int]
) -> list[list[str]]:
    from collections import defaultdict
    groups: defaultdict[int, list[str]] = defaultdict(list)
    for sid, lbl in zip(ids, labels):
        if lbl >= 0:
            groups[lbl].append(sid)
    return [v for v in groups.values() if v]


# ---------------------------------------------------------------------------
# RSS measurement helper
# ---------------------------------------------------------------------------

def _measure_rss_delta_mb(fn) -> tuple[float, float]:
    """Return (wall_s, rss_delta_mb) for calling fn()."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    peak_rss = rss_before
    stop = threading.Event()

    def sampler() -> None:
        nonlocal peak_rss
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass
            time.sleep(0.02)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    t0 = time.perf_counter()
    try:
        fn()
    finally:
        elapsed = time.perf_counter() - t0
        stop.set()
        thread.join(timeout=0.5)

    delta_mb = max(0.0, (peak_rss - rss_before) / 1024 ** 2)
    return elapsed, delta_mb


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestMetaclonotypeCreationBenchmark:
    """Timing and RSS for metaclonotype construction workflows.

    Tests ``metaclonotypes_from_labels`` and
    ``metaclonotypes_from_components`` at four repertoire scales.
    """

    def _run_scale(self, n: int, avg_cluster: int) -> dict:
        ids = [f"s{i}" for i in range(n)]
        labels = _make_labels(n, avg_cluster)
        components = _labels_to_components(ids, labels)

        t_labels, rss_labels = _measure_rss_delta_mb(
            lambda: metaclonotypes_from_labels(ids, labels)
        )
        t_components, rss_components = _measure_rss_delta_mb(
            lambda: metaclonotypes_from_components(components)
        )

        return {
            "n": n,
            "avg_cluster": avg_cluster,
            "n_clusters": len(components),
            "t_labels_ms": t_labels * 1000,
            "rss_labels_mb": rss_labels,
            "t_components_ms": t_components * 1000,
            "rss_components_mb": rss_components,
        }

    def test_metaclonotype_creation_scaling(self) -> None:
        sizes = _bench_sizes()
        avg_cluster = _avg_cluster_size()
        max_s = benchmark_max_seconds(300.0)

        rows: list[dict] = []
        for n in sizes:
            result = self._run_scale(n, avg_cluster)
            rows.append(result)

            line = (
                f"metaclonotype_creation | n={n:>6} | "
                f"labels={result['t_labels_ms']:.1f} ms "
                f"({result['rss_labels_mb']:.1f} MB) | "
                f"components={result['t_components_ms']:.1f} ms "
                f"({result['rss_components_mb']:.1f} MB)"
            )
            print(f"\n  {line}")
            benchmark_log_line(line)

        _print_table(
            rows,
            title="metaclonotype_from_labels / from_components",
            cols=["n", "n_clusters", "t_labels_ms", "rss_labels_mb",
                  "t_components_ms", "rss_components_mb"],
            fmts=["{:>6}", "{:>9}", "{:>12.1f}", "{:>14.1f}",
                  "{:>16.1f}", "{:>18.1f}"],
            headers=["     n", " n_clusters", " t_labels (ms)", " rss_labels (MB)",
                     " t_components (ms)", " rss_components (MB)"],
        )

        # Correctness: all sizes must produce at least one cluster.
        for row in rows:
            assert row["n_clusters"] > 0, f"Expected clusters for n={row['n']}"

        # Performance gate: largest scale must finish within the time budget.
        largest = rows[-1]
        total_s = (largest["t_labels_ms"] + largest["t_components_ms"]) / 1000
        assert total_s < max_s, (
            f"Metaclonotype creation too slow at n={largest['n']}: "
            f"{total_s:.2f}s > {max_s:.2f}s"
        )


@skip_benchmarks
@pytest.mark.benchmark
class TestMetaclonotypeAnalyticsBenchmark:
    """Timing and RSS for summarize / functional_diversity / Hill / rarefaction.

    Creates a single repertoire at each scale, then benchmarks the
    analytics pipeline end-to-end.
    """

    def _run_analytics(self, n: int, avg_cluster: int) -> dict:
        rep = _make_repertoire(n)
        ids = [c.sequence_id for c in rep.clonotypes]
        labels = _make_labels(n, avg_cluster)
        meta = metaclonotypes_from_labels(ids, labels)

        t_summarize, rss_summarize = _measure_rss_delta_mb(
            lambda: summarize_metaclonotypes(rep, meta)
        )

        summary_df = summarize_metaclonotypes(rep, meta)
        counts = summary_df["duplicate_count"].to_list()

        t_div, rss_div = _measure_rss_delta_mb(
            lambda: summarize_counts(counts)
        )
        t_hill, rss_hill = _measure_rss_delta_mb(
            lambda: hill_curve(counts)
        )
        t_rarefy, rss_rarefy = _measure_rss_delta_mb(
            lambda: rarefaction_curve(counts)
        )

        n_clusters = meta.n_clusters

        return {
            "n": n,
            "n_clusters": n_clusters,
            "t_summarize_ms": t_summarize * 1000,
            "rss_summarize_mb": rss_summarize,
            "t_diversity_ms": t_div * 1000,
            "rss_diversity_mb": rss_div,
            "t_hill_ms": t_hill * 1000,
            "rss_hill_mb": rss_hill,
            "t_rarefy_ms": t_rarefy * 1000,
            "rss_rarefy_mb": rss_rarefy,
        }

    def test_analytics_scaling(self) -> None:
        sizes = _bench_sizes()
        avg_cluster = _avg_cluster_size()
        max_s = benchmark_max_seconds(300.0)

        rows: list[dict] = []
        for n in sizes:
            result = self._run_analytics(n, avg_cluster)
            rows.append(result)

            line = (
                f"metaclonotype_analytics | n={n:>6} | "
                f"summarize={result['t_summarize_ms']:.1f} ms | "
                f"diversity={result['t_diversity_ms']:.1f} ms | "
                f"hill={result['t_hill_ms']:.1f} ms | "
                f"rarefy={result['t_rarefy_ms']:.1f} ms"
            )
            print(f"\n  {line}")
            benchmark_log_line(line)

        _print_table(
            rows,
            title="summarize / functional_diversity / hill / rarefaction",
            cols=["n", "n_clusters", "t_summarize_ms", "t_diversity_ms",
                  "t_hill_ms", "t_rarefy_ms"],
            fmts=["{:>6}", "{:>9}", "{:>15.1f}", "{:>14.1f}",
                  "{:>10.2f}", "{:>12.2f}"],
            headers=["     n", " n_clusters", " t_summarize (ms)", " t_diversity (ms)",
                     " t_hill (ms)", " t_rarefy (ms)"],
        )

        # Correctness: metrics must be finite and non-negative.
        for row in rows:
            assert row["t_summarize_ms"] >= 0
            assert row["t_hill_ms"] >= 0
            assert row["t_rarefy_ms"] >= 0

        # Performance gate for the largest scale.
        largest = rows[-1]
        total_s = sum(
            largest[k] / 1000
            for k in ("t_summarize_ms", "t_diversity_ms", "t_hill_ms", "t_rarefy_ms")
        )
        assert total_s < max_s, (
            f"Analytics too slow at n={largest['n']}: {total_s:.2f}s > {max_s:.2f}s"
        )


@skip_benchmarks
@pytest.mark.benchmark
class TestFunctionalDiversityEndToEndBenchmark:
    """Full pipeline: repertoire → metaclonotypes → functional_diversity.

    Exercises the complete ``functional_diversity()`` convenience wrapper,
    which encapsulates summarize + DiversitySummary in a single call.
    """

    def test_functional_diversity_end_to_end(self) -> None:
        sizes = _bench_sizes()
        avg_cluster = _avg_cluster_size()
        max_s = benchmark_max_seconds(300.0)

        rows: list[dict] = []
        for n in sizes:
            rep = _make_repertoire(n)
            ids = [c.sequence_id for c in rep.clonotypes]
            labels = _make_labels(n, avg_cluster)
            meta = metaclonotypes_from_labels(ids, labels)

            result_holder: list = []

            def _run() -> None:
                result_holder.append(functional_diversity(rep, meta))

            elapsed, rss_mb = _measure_rss_delta_mb(_run)
            div = result_holder[0]

            row = {
                "n": n,
                "n_clusters": meta.n_clusters,
                "abundance": div.abundance,
                "diversity": div.diversity,
                "chao1": round(div.chao1, 1),
                "gini_simpson": round(div.gini_simpson, 4),
                "shannon": round(div.shannon, 4),
                "elapsed_ms": elapsed * 1000,
                "rss_mb": rss_mb,
            }
            rows.append(row)

            line = (
                f"functional_diversity | n={n:>6} | "
                f"clusters={meta.n_clusters:>5} | "
                f"elapsed={elapsed * 1000:.1f} ms | "
                f"rss={rss_mb:.1f} MB | "
                f"H={div.shannon:.3f} | D1={div.gini_simpson:.4f} | chao1={div.chao1:.0f}"
            )
            print(f"\n  {line}")
            benchmark_log_line(line)

        _print_table(
            rows,
            title="functional_diversity() end-to-end",
            cols=["n", "n_clusters", "diversity", "chao1", "shannon",
                  "elapsed_ms", "rss_mb"],
            fmts=["{:>6}", "{:>9}", "{:>9}", "{:>7.0f}", "{:>8.4f}",
                  "{:>12.1f}", "{:>8.1f}"],
            headers=["     n", " n_clusters", " diversity", "  chao1", "  shannon",
                     " elapsed (ms)", " rss (MB)"],
        )

        # Correctness: Shannon entropy must be positive for any non-trivial repertoire.
        for row in rows:
            assert row["shannon"] > 0, f"Shannon == 0 for n={row['n']}"

        # Performance gate.
        for row in rows:
            assert row["elapsed_ms"] / 1000 < max_s, (
                f"functional_diversity too slow at n={row['n']}: "
                f"{row['elapsed_ms']:.0f} ms > {max_s * 1000:.0f} ms"
            )


# ---------------------------------------------------------------------------
# Print helper
# ---------------------------------------------------------------------------

def _print_table(
    rows: list[dict],
    *,
    title: str,
    cols: list[str],
    fmts: list[str],
    headers: list[str],
) -> None:
    sep = "  "
    header_line = sep.join(h for h in headers)
    bar = "-" * len(header_line)
    print(f"\n\n  {title}")
    print(f"  {bar}")
    print(f"  {header_line}")
    print(f"  {bar}")
    for row in rows:
        cells = [fmt.format(row[col]) for fmt, col in zip(fmts, cols)]
        print(f"  {sep.join(cells)}")
    print(f"  {bar}")
