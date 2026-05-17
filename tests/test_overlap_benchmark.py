"""Benchmark for :func:`~mir.comparative.overlap.pairwise_overlap`.

Measures wall-clock time and peak RSS for a pair of A* repertoires from the
``isalgo/airr_benchmark`` HuggingFace dataset (vdjtools_lite/ folder), for:

* exact matching (serial and parallel)
* hamming distance 1 (serial and parallel)
* levenshtein distance 1 (serial and parallel)

Run with::

    RUN_BENCHMARK=1 pytest -s tests/test_overlap_benchmark.py

Optional env vars
-----------------
MIRPY_BENCH_OVERLAP_N_JOBS
    Number of parallel workers (default: all physical cores via ``-1``).
MIRPY_BENCH_OVERLAP_SAMPLE_A
    Path to first A* repertoire file (VDJtools format).
MIRPY_BENCH_OVERLAP_SAMPLE_B
    Path to second A* repertoire file (VDJtools format).
"""

from __future__ import annotations

import os
import threading
import time
import tracemalloc
from pathlib import Path

import pytest
import psutil

from mir.common.parser import VDJtoolsParser
from mir.common.repertoire import LocusRepertoire
from mir.comparative.overlap import (
    pairwise_overlap,
    pairwise_overlap_matrix,
)
from tests.benchmark_helpers import estimate_many_vs_many_runtime, many_vs_many_sample_overlap
from tests.conftest import skip_benchmarks

# ---------------------------------------------------------------------------
# Locate A* benchmark repertoires
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VDJTOOLS_LITE = _REPO_ROOT / "airr_benchmark" / "vdjtools_lite"
_ASSETS_REPS = _REPO_ROOT / "tests" / "assets" / "real_repertoires"

# Prefer samples from the full benchmark dataset; fall back to test assets.
_CANDIDATE_DIRS = [_VDJTOOLS_LITE, _ASSETS_REPS]


def _find_a_star_samples(n: int = 2) -> list[Path]:
    """Return up to *n* largest A*-prefixed VDJtools files found in candidate dirs."""
    candidates: list[tuple[int, Path]] = []
    for d in _CANDIDATE_DIRS:
        if not d.is_dir():
            continue
        for p in d.glob("A*.txt.gz"):
            try:
                candidates.append((p.stat().st_size, p))
            except OSError:
                pass
    candidates.sort(reverse=True)
    return [p for _, p in candidates[:n]]


def _load_rep(path: Path) -> LocusRepertoire:
    parser = VDJtoolsParser(sep="\t")
    clones = parser.parse(str(path))
    return LocusRepertoire(clonotypes=clones, locus="TRB")


def _peak_mb() -> str:
    _, peak = tracemalloc.get_traced_memory()
    return f"{peak / 1024 ** 2:.1f} MB"


def _measure_peak_rss_mb(fn) -> float:
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
            time.sleep(0.05)

    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    try:
        fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)

    return max(0.0, (peak_rss - rss_before) / 1024 ** 2)


# ---------------------------------------------------------------------------
# Benchmark fixture
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestPairwiseOverlapBenchmark:
    """Pairwise overlap timing and memory for two A* TRB repertoires.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_overlap_benchmark.py::TestPairwiseOverlapBenchmark
    """

    N_JOBS = int(os.getenv("MIRPY_BENCH_OVERLAP_N_JOBS", "-1"))

    @pytest.fixture(scope="class")
    def rep_pair(self) -> tuple[LocusRepertoire, LocusRepertoire, str, str]:
        env_a = os.getenv("MIRPY_BENCH_OVERLAP_SAMPLE_A")
        env_b = os.getenv("MIRPY_BENCH_OVERLAP_SAMPLE_B")

        if env_a and env_b:
            paths = [Path(env_a), Path(env_b)]
        else:
            paths = _find_a_star_samples(2)

        if len(paths) < 2:
            pytest.skip("Need ≥ 2 A* repertoire files; set MIRPY_BENCH_OVERLAP_SAMPLE_A/B or "
                        "run `python tests/prepare_airr_benchmark_data.py`.")

        rep_a = _load_rep(paths[0])
        rep_b = _load_rep(paths[1])
        return rep_a, rep_b, paths[0].name, paths[1].name

    def _time_overlap(
        self,
        rep_a: LocusRepertoire,
        rep_b: LocusRepertoire,
        *,
        metric: str,
        threshold: int,
        n_jobs: int,
    ) -> tuple[float, float, object]:
        tracemalloc.start()
        t0 = time.perf_counter()
        result = pairwise_overlap(rep_a, rep_b, metric=metric, threshold=threshold, n_jobs=n_jobs)
        elapsed = time.perf_counter() - t0
        peak_b = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return elapsed, peak_b / 1024 ** 2, result

    def _header(self, name_a: str, name_b: str, rep_a: LocusRepertoire, rep_b: LocusRepertoire) -> None:
        print(f"\n{'='*70}")
        print(f"  Repertoires: {name_a}  ×  {name_b}")
        print(f"  Sizes: {rep_a.clonotype_count:,} × {rep_b.clonotype_count:,} clonotypes")
        print(f"  n_jobs: {self.N_JOBS}")
        print(f"{'='*70}")

    def test_exact_serial(self, rep_pair) -> None:
        rep_a, rep_b, na, nb = rep_pair
        self._header(na, nb, rep_a, rep_b)
        elapsed, peak_mb, r = self._time_overlap(rep_a, rep_b, metric="exact", threshold=0, n_jobs=1)
        rss_delta_mb = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="exact", threshold=0, n_jobs=1)
        )
        print(f"\n[exact  serial ] n1_matched={r.n1_matched:,}  D={r.d_similarity:.4f}  "
              f"F={r.f_similarity:.4f}  J={r.jaccard:.4f}  {elapsed*1e3:.1f} ms  {peak_mb:.0f} MB")
        assert rss_delta_mb < 4_500
        assert r.n1_matched >= 0

    def test_exact_parallel(self, rep_pair) -> None:
        rep_a, rep_b, na, nb = rep_pair
        elapsed_s, _, r_s = self._time_overlap(rep_a, rep_b, metric="exact", threshold=0, n_jobs=1)
        elapsed_p, peak_mb, r_p = self._time_overlap(rep_a, rep_b, metric="exact", threshold=0, n_jobs=self.N_JOBS)
        rss_delta_s = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="exact", threshold=0, n_jobs=1)
        )
        rss_delta_p = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="exact", threshold=0, n_jobs=self.N_JOBS)
        )
        print(f"\n[exact  parallel n_jobs={self.N_JOBS}] serial={elapsed_s*1e3:.1f} ms  "
              f"parallel={elapsed_p*1e3:.1f} ms  {peak_mb:.0f} MB")
        assert rss_delta_p <= rss_delta_s * 1.25 + 250
        # Results must match
        assert r_s.n1_matched == r_p.n1_matched
        assert abs(r_s.d_similarity - r_p.d_similarity) < 1e-9

    def test_hamming1_serial(self, rep_pair) -> None:
        rep_a, rep_b, na, nb = rep_pair
        elapsed, peak_mb, r = self._time_overlap(rep_a, rep_b, metric="hamming", threshold=1, n_jobs=1)
        rss_delta_mb = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="hamming", threshold=1, n_jobs=1)
        )
        r_exact = pairwise_overlap(rep_a, rep_b, metric="exact", threshold=0)
        print(f"\n[ham:1  serial ] n1_matched={r.n1_matched:,} (exact={r_exact.n1_matched:,})  "
              f"D={r.d_similarity:.4f}  F={r.f_similarity:.4f}  {elapsed*1e3:.1f} ms  {peak_mb:.0f} MB")
        assert rss_delta_mb < 4_500
        assert r.n1_matched >= r_exact.n1_matched

    def test_hamming1_parallel(self, rep_pair) -> None:
        rep_a, rep_b, na, nb = rep_pair
        elapsed_s, _, r_s = self._time_overlap(rep_a, rep_b, metric="hamming", threshold=1, n_jobs=1)
        elapsed_p, peak_mb, r_p = self._time_overlap(rep_a, rep_b, metric="hamming", threshold=1, n_jobs=self.N_JOBS)
        rss_delta_s = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="hamming", threshold=1, n_jobs=1)
        )
        rss_delta_p = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="hamming", threshold=1, n_jobs=self.N_JOBS)
        )
        print(f"\n[ham:1  parallel n_jobs={self.N_JOBS}] serial={elapsed_s*1e3:.1f} ms  "
              f"parallel={elapsed_p*1e3:.1f} ms  {peak_mb:.0f} MB")
        assert rss_delta_p <= rss_delta_s * 1.25 + 250
        assert r_s.n1_matched == r_p.n1_matched

    def test_levenshtein1_serial(self, rep_pair) -> None:
        rep_a, rep_b, na, nb = rep_pair
        elapsed, peak_mb, r = self._time_overlap(rep_a, rep_b, metric="levenshtein", threshold=1, n_jobs=1)
        rss_delta_mb = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="levenshtein", threshold=1, n_jobs=1)
        )
        print(f"\n[lev:1  serial ] n1_matched={r.n1_matched:,}  "
              f"D={r.d_similarity:.4f}  F={r.f_similarity:.4f}  {elapsed*1e3:.1f} ms  {peak_mb:.0f} MB")
        assert rss_delta_mb < 4_500
        assert r.n1_matched >= 0

    def test_levenshtein1_parallel(self, rep_pair) -> None:
        rep_a, rep_b, na, nb = rep_pair
        elapsed_s, _, r_s = self._time_overlap(rep_a, rep_b, metric="levenshtein", threshold=1, n_jobs=1)
        elapsed_p, peak_mb, r_p = self._time_overlap(rep_a, rep_b, metric="levenshtein", threshold=1, n_jobs=self.N_JOBS)
        rss_delta_s = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="levenshtein", threshold=1, n_jobs=1)
        )
        rss_delta_p = _measure_peak_rss_mb(
            lambda: pairwise_overlap(rep_a, rep_b, metric="levenshtein", threshold=1, n_jobs=self.N_JOBS)
        )
        print(f"\n[lev:1  parallel n_jobs={self.N_JOBS}] serial={elapsed_s*1e3:.1f} ms  "
              f"parallel={elapsed_p*1e3:.1f} ms  {peak_mb:.0f} MB")
        assert rss_delta_p <= rss_delta_s * 1.25 + 250
        assert r_s.n1_matched == r_p.n1_matched

    def test_summary_table(self, rep_pair) -> None:
        """Print a compact summary table of all modes × serial/parallel."""
        rep_a, rep_b, na, nb = rep_pair
        modes = [
            ("exact", 0),
            ("hamming", 1),
            ("levenshtein", 1),
        ]
        print(f"\n{'Mode':<20} {'Jobs':>5} {'n1_matched':>12} {'D':>8} {'F':>8} {'time (ms)':>12} {'peak MB':>10}")
        print("-" * 80)
        for metric, threshold in modes:
            for n_jobs in (1, self.N_JOBS):
                if n_jobs == self.N_JOBS and n_jobs == 1:
                    continue  # skip duplicate if N_JOBS == 1
                label = f"{metric}:{threshold}"
                elapsed, peak_mb, r = self._time_overlap(rep_a, rep_b, metric=metric, threshold=threshold, n_jobs=n_jobs)
                print(f"{label:<20} {n_jobs:>5} {r.n1_matched:>12,} {r.d_similarity:>8.4f} "
                      f"{r.f_similarity:>8.4f} {elapsed*1e3:>12.1f} {peak_mb:>10.0f}")
        assert True  # formatting test, always passes


@skip_benchmarks
@pytest.mark.benchmark
class TestPairwiseMatrixBenchmark:
    """Matrix-level overlap benchmark for all aging samples.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_overlap_benchmark.py::TestPairwiseMatrixBenchmark
    """

    N_JOBS = int(os.getenv("MIRPY_BENCH_OVERLAP_N_JOBS", "-1"))

    @pytest.fixture(scope="class")
    def aging_reps(self) -> tuple[list[LocusRepertoire], list[str]]:
        """Load all A* samples from vdjtools_lite that appear in metadata_aging.txt."""
        import csv

        meta_path = _VDJTOOLS_LITE / "metadata_aging.txt"
        if not meta_path.exists():
            meta_path = _ASSETS_REPS / "metadata_aging.txt"
        if not meta_path.exists():
            pytest.skip("metadata_aging.txt not found.")

        parser = VDJtoolsParser(sep="\t")
        reps, ids = [], []
        with open(meta_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                for base_dir in (_VDJTOOLS_LITE, _ASSETS_REPS):
                    p = base_dir / row["file_name"]
                    if p.exists():
                        clones = parser.parse(str(p))
                        reps.append(LocusRepertoire(clonotypes=clones, locus="TRB"))
                        ids.append(row["sample_id"])
                        break

        if len(reps) < 2:
            pytest.skip(f"Loaded only {len(reps)} aging repertoires; need ≥ 2.")

        print(f"\nLoaded {len(reps)} aging repertoires "
              f"(mean size: {sum(r.clonotype_count for r in reps)//len(reps):,} clonotypes)")
        return reps, ids

    def test_matrix_exact_parallel(self, aging_reps) -> None:
        reps, ids = aging_reps
        tracemalloc.start()
        t0 = time.perf_counter()
        df = pairwise_overlap_matrix(reps, sample_ids=ids, metric="exact", n_jobs=self.N_JOBS)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_delta_mb = _measure_peak_rss_mb(
            lambda: pairwise_overlap_matrix(reps, sample_ids=ids, metric="exact", n_jobs=self.N_JOBS)
        )
        n_pairs = len(reps) * (len(reps) - 1) // 2
        print(f"\n[exact  matrix n={len(reps)} n_jobs={self.N_JOBS}] "
              f"{n_pairs} pairs  {elapsed:.2f}s  {peak/1024**2:.0f} MB")
        assert rss_delta_mb < 5_000
        assert len(df) == n_pairs

    def test_matrix_hamming1_parallel(self, aging_reps) -> None:
        reps, ids = aging_reps
        tracemalloc.start()
        t0 = time.perf_counter()
        df = pairwise_overlap_matrix(reps, sample_ids=ids, metric="hamming", threshold=1, n_jobs=self.N_JOBS)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_delta_mb = _measure_peak_rss_mb(
            lambda: pairwise_overlap_matrix(reps, sample_ids=ids, metric="hamming", threshold=1, n_jobs=self.N_JOBS)
        )
        n_pairs = len(reps) * (len(reps) - 1) // 2
        print(f"\n[ham:1  matrix n={len(reps)} n_jobs={self.N_JOBS}] "
              f"{n_pairs} pairs  {elapsed:.2f}s  {peak/1024**2:.0f} MB")
        assert rss_delta_mb < 5_000
        assert len(df) == n_pairs

    def test_matrix_levenshtein1_parallel(self, aging_reps) -> None:
        reps, ids = aging_reps
        tracemalloc.start()
        t0 = time.perf_counter()
        df = pairwise_overlap_matrix(reps, sample_ids=ids, metric="levenshtein", threshold=1, n_jobs=self.N_JOBS)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_delta_mb = _measure_peak_rss_mb(
            lambda: pairwise_overlap_matrix(reps, sample_ids=ids, metric="levenshtein", threshold=1, n_jobs=self.N_JOBS)
        )
        n_pairs = len(reps) * (len(reps) - 1) // 2
        print(f"\n[lev:1  matrix n={len(reps)} n_jobs={self.N_JOBS}] "
              f"{n_pairs} pairs  {elapsed:.2f}s  {peak/1024**2:.0f} MB")
        assert rss_delta_mb < 5_000
        assert len(df) == n_pairs


@skip_benchmarks
@pytest.mark.benchmark
class TestManyVsManyPilotBenchmark:
    """Pilot-based runtime benchmark for many-vs-many helper.

    Uses a subset of samples to estimate full matrix runtime and validates that
    a small exact run completes in meaningful time.
    """

    N_JOBS = int(os.getenv("MIRPY_BENCH_OVERLAP_N_JOBS", "4"))

    @pytest.fixture(scope="class")
    def aging_reps_small(self) -> tuple[list[LocusRepertoire], list[str]]:
        import csv

        meta_path = _VDJTOOLS_LITE / "metadata_aging.txt"
        if not meta_path.exists():
            meta_path = _ASSETS_REPS / "metadata_aging.txt"
        if not meta_path.exists():
            pytest.skip("metadata_aging.txt not found.")

        parser = VDJtoolsParser(sep="\t")
        reps, ids = [], []
        with open(meta_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                fname = row.get("file_name") or row.get("#file_name")
                if not fname:
                    continue
                candidates = [fname, f"{fname}.gz", fname.replace(".txt", ".txt.gz")]
                found = None
                for base_dir in (_VDJTOOLS_LITE, _ASSETS_REPS):
                    for cand in candidates:
                        p = base_dir / cand
                        if p.exists():
                            found = p
                            break
                    if found is not None:
                        break
                if found is None:
                    continue
                clones = parser.parse(str(found))
                reps.append(LocusRepertoire(clonotypes=clones, locus="TRB"))
                ids.append(row.get("sample_id", found.stem))
                if len(reps) >= 16:
                    break

        if len(reps) < 8:
            pytest.skip(f"Loaded only {len(reps)} repertoires; need >= 8 for pilot benchmark.")

        print(f"\nLoaded {len(reps)} repertoires for pilot benchmark")
        return reps, ids

    def test_pilot_estimate_and_exact_subset_runtime(self, aging_reps_small) -> None:
        reps, ids = aging_reps_small

        est = estimate_many_vs_many_runtime(
            reps,
            metric="exact",
            threshold=0,
            overlap_space="aavj",
            n_jobs=self.N_JOBS,
            pilot_sample_count=8,
        )

        print(
            "\n[pilot estimate exact] "
            f"pilot_pairs={est['pilot_pairs']} pilot_s={est['pilot_seconds']:.2f} "
            f"s_per_pair={est['seconds_per_pair']:.4f} est_full_s={est['estimated_total_seconds']:.2f}"
        )

        t0 = time.perf_counter()
        df = many_vs_many_sample_overlap(
            reps,
            sample_ids=ids,
            metric="exact",
            threshold=0,
            overlap_space="aavj",
            n_jobs=self.N_JOBS,
        )
        elapsed = time.perf_counter() - t0
        expected_pairs = len(reps) * (len(reps) - 1) // 2

        print(
            f"[many-vs-many exact subset] pairs={expected_pairs} elapsed={elapsed:.2f}s "
            f"n_jobs_effective={int(df['n_jobs_effective'].iloc[0])}"
        )

        assert len(df) == expected_pairs
        assert elapsed < 600, f"subset many-vs-many exact took too long: {elapsed:.2f}s"
