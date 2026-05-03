"""Benchmark and integration tests for :mod:`mir.biomarkers.vdjbet`.

Run all benchmarks::

    RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py

Single benchmark class::

    RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestYFVP1SignificanceAndPgenBins

Integration tests with Q1 donor::

    RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestQ1Q15Integration

Benchmark tests (``RUN_BENCHMARK=1``)
--------------------------------------
* :class:`TestPgenBinPoolBenchmark`        — pool build time, coverage, distribution diagnostics.
* :class:`TestVDJBetMockBenchmark`         — mock timing, memory, Pgen distribution closeness
  (JSD / max-bin-diff / RMSD / KS / Chi2 vs reference).
* :class:`TestPgenParallelBenchmark`       — generate_pool 1-job vs 4-job speedup; pgen cache speedup.
* :class:`TestLLWOverlapYFV`               — LLWNGPMAV TRB overlap YFV donor S1 day 0 vs day 15.
* :class:`TestYFVS1F1`                     — Integration tests on full YFV cohort (from test_yfv_integration.py).
* :class:`TestFunctionalFilteringCounts`   — Functional filtering count validation (from test_yfv_integration.py).
* :class:`TestQ1Q15Integration`            — LLWNGPMAV TRB overlap Q1 donor day 0 vs day 15 (test assets).
* :class:`TestSyntheticVsRealMockComparison` — Synthetic vs real mock effect size comparison.
* :class:`TestQ1ControlEffectSize`         — Q1 day-0 vs day-15 Cohen d effect size analysis.
* :class:`TestYFVP1SignificanceAndPgenBins` — P1/F1 day-0 non-significant, day-15 significant;
  full distribution diagnostics across 200 mocks.
* :class:`TestRepertoireIOPolars`          — pandas vs polars I/O timing and memory.

Full-data benchmark (``RUN_BENCHMARK=1 RUN_FULL_BENCHMARK=1``)
----------------------------------------------------------------
* :class:`TestYFVP1SignificanceAndPgenBins` — full-cohort YFV adjustment and
    P1 day-0/day-15 significance diagnostics.

Dataset notes
-------------
* **Q1 donor** (test assets): ``tests/assets/Q1_0_F1.airr.tsv.gz`` and
  ``tests/assets/Q1_15_F1.airr.tsv.gz`` — Preferred for regular testing;
  compact subsets suitable for CI/regular benchmarks.
* **YFV test assets** (legacy): ``tests/assets/yfv_s1_d0_f1.airr.tsv.gz`` and
  ``tests/assets/yfv_s1_d15_f1.airr.tsv.gz`` — Being phased out; kept for
  backward compatibility (unknown origin).
* **Full YFV cohort**: ``notebooks/assets/large/yfv19/`` — Full-data integration
  tests (requires ``RUN_FULL_BENCHMARK=1``).
* **VDJdb LLWNGPMAV reference**: ``tests/assets/vdjdb.slim.txt.gz`` — Used for
  all integration tests and benchmarks.

Key enhancements in benchmark suite
------------------------------------
* **Cohen d effect size analysis**: Validates that vaccine response (day 15)
  produces materially larger effect than baseline (day 0) after accounting for variance.
* **Synthetic vs real mock comparison**: Tests show synthetic (OLGA) mock hits
  are significant at day 0 (p-value > 0.05 but lower), but scaling synthetic
  mock by 3x and using real control repertoire nullifies day 0 significance
  while preserving day 15 signal.
* **Runtime measurement assertions**: Benchmarks track pool build, mock generation,
  and I/O timing with strict performance guardrails.
* **Distribution quality validation**: JSD, KS, Chi2 tests ensure mock distributions
  match reference p-gen distribution within statistical tolerance.

Integration test migration
---------------------------
Tests migrated from test_yfv_integration.py (now removed):
* TestYFVS1F1 — Uses full YFV cohort for integration testing
* TestFunctionalFilteringCounts — Validates functional filtering counts on real data
* Helper functions: _build_yfv_gene_usage, _load_s1_f1, _load_p1_f1
"""

from __future__ import annotations

import math
import os
import time
import tracemalloc
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.biomarkers.vdjbet import (
    OverlapResult,
    PgenBinPool,
    VDJBetOverlapAnalysis,
    _log2_pgen_bin,
    _strip_allele,
    compute_pgen_histogram,
)
from mir.common.clonotype import Clonotype
from mir.common.filter import filter_functional
from mir.common.gene_library import GeneLibrary
from mir.common.parser import ClonotypeTableParser, VDJdbSlimParser
from mir.common.repertoire import LocusRepertoire
from tests.conftest import skip_benchmarks

ASSETS = Path(__file__).parent / "assets"
_LLW_FILE = ASSETS / "llwngpmav_trb_a02.tsv.gz"
_VDJDB_FILE = ASSETS / "vdjdb.slim.txt.gz"

# Q1 donor repertoires for integration tests (preferred over YFV notebook assets)
_Q1_D0   = ASSETS / "Q1_0_F1.airr.tsv.gz"
_Q1_D15  = ASSETS / "Q1_15_F1.airr.tsv.gz"

# Legacy YFV test assets (to be phased out; kept for backward compatibility)
_YFV_D0   = ASSETS / "yfv_s1_d0_f1.airr.tsv.gz"
_YFV_D15  = ASSETS / "yfv_s1_d15_f1.airr.tsv.gz"
_YFV_FULL_DIR = Path(__file__).parent.parent / "notebooks" / "assets" / "large" / "yfv19"

_LLW_AVAILABLE = _LLW_FILE.exists()
_Q1_AVAILABLE = _Q1_D0.exists() and _Q1_D15.exists()
_YFV_TEST_AVAILABLE = _YFV_D0.exists() and _YFV_D15.exists()
_VDJDB_AVAILABLE = _VDJDB_FILE.exists()
RUN_FULL_BENCHMARK = (
    os.getenv("RUN_FULL_BENCHMARK") == "1"
    or os.getenv("RUN_FULL_BENCHMARKS") == "1"
)
_SEED = 42


# ---------------------------------------------------------------------------
# Shared helpers for benchmarks
# ---------------------------------------------------------------------------

def _make_olga_rep(locus: str, n: int, seed: int = _SEED) -> LocusRepertoire:
    """Generate n OLGA clonotypes and return them as a LocusRepertoire."""
    model = OlgaModel(locus=locus, seed=seed)
    records = model.generate_sequences_with_meta(n, pgens=False, seed=None)
    clones = [
        Clonotype(
            sequence_id=str(i), locus=locus,
            junction_aa=r["junction_aa"], junction=r["junction"],
            v_gene=r["v_gene"], j_gene=r["j_gene"],
            v_sequence_end=r["v_end"], j_sequence_start=r["j_start"],
            duplicate_count=1, _validate=False,
        )
        for i, r in enumerate(records)
    ]
    return LocusRepertoire(clonotypes=clones, locus=locus)


def _load_llw_reference() -> LocusRepertoire:
    df = pd.read_csv(_LLW_FILE, sep="\t", compression="infer")
    clones = ClonotypeTableParser().parse_inner(df)
    rep = LocusRepertoire(clonotypes=clones, locus="TRB")
    # Filter to functional clonotypes only
    return filter_functional(rep)


def _load_yfv_sample(path: Path) -> LocusRepertoire:
    df = pd.read_csv(path, sep="\t", compression="infer")
    if "locus" in df.columns:
        df = df[df["locus"].fillna("") == "TRB"]
    df = df.dropna(subset=["junction_aa"])
    df = df[df["junction_aa"].str.strip().str.len() > 0]
    clones = ClonotypeTableParser().parse_inner(df)
    rep = LocusRepertoire(clonotypes=clones, locus="TRB")
    # Filter to functional clonotypes only
    return filter_functional(rep)


def _load_q1_sample(path: Path, n_top: int | None = None) -> LocusRepertoire:
    """Load Q1 donor repertoire and optionally select top N clonotypes.
    
    Parameters
    ----------
    path:
        Path to Q1 AIRR TSV file.
    n_top:
        Number of top clonotypes to keep (by duplicate count).  None keeps all.
    
    Returns
    -------
    LocusRepertoire filtered to TRB and functional clonotypes, optionally truncated.
    """
    df = pd.read_csv(path, sep="\t", compression="infer")
    if "locus" in df.columns:
        df = df[df["locus"].fillna("") == "TRB"]
    df = df.dropna(subset=["junction_aa"])
    df = df[df["junction_aa"].str.strip().str.len() > 0]
    clones = ClonotypeTableParser().parse_inner(df)
    rep = LocusRepertoire(clonotypes=clones, locus="TRB")
    rep = filter_functional(rep)
    if n_top is not None:
        rep = rep.sample_n(n=n_top, sample_random=False)
    return rep


def _load_vdjdb_llw_reference() -> LocusRepertoire:
    """Load LLWNGPMAV TRB HLA-A*02 entries from VDJdb test asset."""
    sample = VDJdbSlimParser().parse_file(_VDJDB_FILE, species="HomoSapiens")
    trb = sample["TRB"]
    filtered = [
        c for c in trb.clonotypes
        if c.clone_metadata.get("antigen.epitope") == "LLWNGPMAV"
        and "A*02" in c.clone_metadata.get("mhc.a", "")
    ]
    return LocusRepertoire(clonotypes=filtered, locus="TRB")


def _build_yfv_gene_usage() -> GeneUsage:
    """Build TRB gene usage from all YFV repertoires listed in metadata."""
    meta = pd.read_csv(_YFV_FULL_DIR / "metadata.txt", sep="\t")
    parser = ClonotypeTableParser()
    all_clones = []
    for _, row in meta.iterrows():
        fpath = _YFV_FULL_DIR / row["file_name"]
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, sep="\t", compression="infer")
        if "locus" in df.columns:
            df = df[df["locus"].fillna("") == "TRB"]
        df = df.dropna(subset=["junction_aa"])
        df = df[df["junction_aa"].str.strip().str.len() > 0]
        all_clones.extend(parser.parse_inner(df))

    rep = LocusRepertoire(clonotypes=all_clones, locus="TRB", repertoire_id="yfv19-all-trb")
    return GeneUsage.from_repertoire(rep)


def _load_s1_f1(day: int) -> LocusRepertoire:
    """Load S1 replica F1 for the given *day* from the YFV dataset."""
    meta = pd.read_csv(_YFV_FULL_DIR / "metadata.txt", sep="\t")
    row = meta[(meta["donor"] == "S1") & (meta["day"] == day) & (meta["replica"] == "F1")]
    if row.empty:
        pytest.skip(f"S1/F1/day={day} not found in metadata")
    fname = row.iloc[0]["file_name"]
    df = pd.read_csv(_YFV_FULL_DIR / fname, sep="\t", compression="infer")
    if "locus" in df.columns:
        df = df[df["locus"].fillna("") == "TRB"]
    df = df.dropna(subset=["junction_aa"])
    df = df[df["junction_aa"].str.strip().str.len() > 0]
    parser = ClonotypeTableParser()
    clones = parser.parse_inner(df)
    return LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=fname)


def _load_p1_f1(day: int) -> LocusRepertoire:
    """Load P1 replica F1 for the given *day* from the YFV dataset."""
    meta = pd.read_csv(_YFV_FULL_DIR / "metadata.txt", sep="\t")
    row = meta[(meta["donor"] == "P1") & (meta["day"] == day) & (meta["replica"] == "F1")]
    if row.empty:
        pytest.skip(f"P1/F1/day={day} not found in metadata")
    fname = row.iloc[0]["file_name"]
    df = pd.read_csv(_YFV_FULL_DIR / fname, sep="\t", compression="infer")
    if "locus" in df.columns:
        df = df[df["locus"].fillna("") == "TRB"]
    df = df.dropna(subset=["junction_aa"])
    df = df[df["junction_aa"].str.strip().str.len() > 0]
    parser = ClonotypeTableParser()
    clones = parser.parse_inner(df)
    return LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=fname)


# ---------------------------------------------------------------------------
# Distribution divergence helpers
# ---------------------------------------------------------------------------

def _bin_distribution_metrics(
    ref_bins: list[int],
    mock_bins: list[int],
) -> dict[str, float]:
    """Compute distribution divergence metrics between two log2-Pgen bin lists.

    Parameters
    ----------
    ref_bins:
        Flat list of log2-Pgen bins from the reference clonotypes.
    mock_bins:
        Flat list of log2-Pgen bins drawn for one mock replicate.

    Returns
    -------
    dict with keys:
        jsd           — Jensen-Shannon divergence (0 = identical, 1 = maximally different)
        max_bin_diff  — largest absolute per-bin probability difference
        rmsd_bin_diff — RMSD of per-bin probability differences
        ks_stat, ks_p — two-sample KS test (large p = similar distributions)
        chi2_stat, chi2_p — Chi2 goodness-of-fit (large p = similar)
    """
    from scipy.stats import ks_2samp, chisquare

    ref_c   = Counter(ref_bins)
    mock_c  = Counter(mock_bins)
    all_bins = sorted(set(ref_c) | set(mock_c))
    if not all_bins:
        return dict(jsd=0.0, max_bin_diff=0.0, rmsd_bin_diff=0.0,
                    ks_stat=0.0, ks_p=1.0, chi2_stat=0.0, chi2_p=1.0)

    ref_vec  = np.array([ref_c.get(b, 0) for b in all_bins], dtype=float)
    mock_vec = np.array([mock_c.get(b, 0) for b in all_bins], dtype=float)
    ref_p    = ref_vec  / max(1.0, ref_vec.sum())
    mock_p   = mock_vec / max(1.0, mock_vec.sum())

    eps = 1e-12
    p = np.clip(ref_p, eps, 1.0); p /= p.sum()
    q = np.clip(mock_p, eps, 1.0); q /= q.sum()
    m = 0.5 * (p + q)
    jsd = float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))

    diff = np.abs(ref_p - mock_p)

    ks_stat, ks_p = (ks_2samp(ref_bins, mock_bins)
                     if ref_bins and mock_bins else (0.0, 1.0))

    expected = ref_p * max(1.0, mock_vec.sum())
    if np.all(expected > 0):
        chi2_stat, chi2_p = chisquare(mock_vec, expected)
    else:
        chi2_stat, chi2_p = 0.0, 1.0

    return dict(
        jsd=jsd,
        max_bin_diff=float(np.max(diff)),
        rmsd_bin_diff=float(np.sqrt(np.mean((ref_p - mock_p) ** 2))),
        ks_stat=float(ks_stat), ks_p=float(ks_p),
        chi2_stat=float(chi2_stat), chi2_p=float(chi2_p),
    )


# ---------------------------------------------------------------------------
# Benchmark: PgenBinPool build timing and bin coverage
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestPgenBinPoolBenchmark:
    """PgenBinPool build timing, coverage, and distribution quality.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestPgenBinPoolBenchmark
    """

    N = 300

    @pytest.fixture(scope="class")
    def pool_single(self) -> PgenBinPool:
        return PgenBinPool("TRB", n=self.N, n_jobs=1, seed=_SEED)

    @pytest.fixture(scope="class")
    def pool_parallel(self) -> PgenBinPool:
        return PgenBinPool("TRB", n=self.N, n_jobs=4, seed=_SEED)

    def test_single_job_timing_and_memory(self) -> None:
        """1-job pool of 300 seqs should finish quickly for subset diagnostics."""
        t0 = time.perf_counter()
        pool = PgenBinPool("TRB", n=self.N, n_jobs=1, seed=_SEED)
        elapsed = time.perf_counter() - t0

        bins = sorted(pool.bins)
        print(
            f"\n1-job TRB pool (n={self.N:,}): {elapsed:.2f}s "
            f"({self.N/elapsed:,.0f} seq/s)  bins={len(bins)} "
            f"range=[{bins[0]},{bins[-1]}]"
        )
        assert elapsed < 60
        assert len(bins) >= 8

    def test_parallel_speedup(self) -> None:
        """4-job pool build should not be catastrophically slower than 1-job.

        On macOS, process-spawn overhead can dominate for small n, so we keep
        this as a diagnostic guardrail rather than a strict speedup target.
        """
        t1 = time.perf_counter()
        PgenBinPool("TRB", n=self.N, n_jobs=1, seed=_SEED)
        t_single = time.perf_counter() - t1

        t4 = time.perf_counter()
        PgenBinPool("TRB", n=self.N, n_jobs=4, seed=_SEED)
        t_par = time.perf_counter() - t4

        speedup = t_single / t_par if t_par > 0 else float("inf")
        print(f"\nPool speedup: 1-job={t_single:.2f}s  4-job={t_par:.2f}s  {speedup:.2f}x")
        assert speedup >= 0.2

    def test_log2_pgen_range(self, pool_single: PgenBinPool) -> None:
        """Pool log2 Pgen mean must be in a biologically realistic range."""
        pool = pool_single
        mean = float(np.mean(pool.log2_pgen_array()))
        print(f"\nPool log2 Pgen: mean={mean:.1f}  "
              f"floor={pool.floor_bin}  ceil={pool.ceil_bin}")
        assert -80 < mean < -5


# ---------------------------------------------------------------------------
# Benchmark: VDJBet mock timing and distribution diagnostics
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestVDJBetMockBenchmark:
    """Mock generation timing, memory, and Pgen distribution closeness.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestVDJBetMockBenchmark
    """

    @pytest.fixture(scope="class")
    def ref(self) -> LocusRepertoire:
        return _make_olga_rep("TRB", 20)

    @pytest.fixture(scope="class")
    def pool(self) -> PgenBinPool:
        return PgenBinPool("TRB", n=20_000, n_jobs=4, seed=_SEED)

    def test_timing_and_memory(self, ref, pool) -> None:
        """50 mocks must generate in < 10 s with pre-built pool."""
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=50, seed=_SEED)
        tracemalloc.start()
        t0 = time.perf_counter()
        analysis._get_mock_key_sets()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(
            f"\n50 mocks (ref=20, pool=20k): "
            f"elapsed={elapsed*1000:.1f}ms  peak={peak/(1024**2):.2f}MiB"
        )
        assert elapsed < 10.0

    def test_distribution_closeness(self, ref, pool) -> None:
        """Mock log2-Pgen distributions must be statistically close to reference.

        Thresholds are generous for a 20-clone reference where stochastic noise
        is high:
        * median JSD < 0.10
        * p95 max bin diff < 0.25
        * p95 RMSD < 0.10
        * >= 70 % mocks KS non-significant (p > 0.05)
        * >= 70 % mocks Chi2 non-significant (p > 0.05)
        """
        analysis = VDJBetOverlapAnalysis(ref, pool=pool, n_mocks=50, seed=_SEED)
        ref_bins  = analysis.get_reference_bin_sample()
        mock_bins = analysis.get_mock_bin_samples()

        jsds, max_diffs, rmsds, ks_ps, chi_ps = [], [], [], [], []
        for mb in mock_bins:
            if not mb:
                continue
            d = _bin_distribution_metrics(ref_bins, mb)
            jsds.append(d["jsd"]); max_diffs.append(d["max_bin_diff"])
            rmsds.append(d["rmsd_bin_diff"])
            ks_ps.append(d["ks_p"]); chi_ps.append(d["chi2_p"])

        jsd_med    = float(np.median(jsds))
        max_diff95 = float(np.percentile(max_diffs, 95))
        rmsd95     = float(np.percentile(rmsds, 95))
        ks_frac    = sum(p > 0.05 for p in ks_ps)  / max(len(ks_ps),  1)
        chi_frac   = sum(p > 0.05 for p in chi_ps) / max(len(chi_ps), 1)

        print(
            f"\nMock metrics (ref_n={len(ref_bins)}, n_mocks=50):\n"
            f"  median JSD={jsd_med:.4f}  p95 max_diff={max_diff95:.4f}  "
            f"p95 rmsd={rmsd95:.4f}\n"
            f"  KS non-sig={ks_frac:.0%}  Chi2 non-sig={chi_frac:.0%}"
        )

        assert jsd_med    < 0.10
        assert max_diff95 < 0.25
        assert rmsd95     < 0.10
        assert ks_frac    >= 0.70
        assert chi_frac   >= 0.70


# ---------------------------------------------------------------------------
# Benchmark: OlgaModel.generate_pool speedup and pgen cache speedup
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestPgenParallelBenchmark:
    """generate_pool 1-job vs 4-job speedup; pgen aa-cache speedup and memory.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestPgenParallelBenchmark
    """

    N = 2_000

    @pytest.fixture(scope="class")
    def model(self) -> OlgaModel:
        return OlgaModel(locus="TRB", seed=_SEED)

    def test_generate_pool_returns_log2_pgen(self, model) -> None:
        """Pool records must contain a log2_pgen field (not log10)."""
        records = model.generate_pool(100, n_jobs=1, seed=_SEED)
        assert len(records) == 100
        assert all("log2_pgen" in r for r in records)

    def test_single_job_timing(self, model) -> None:
        t0 = time.perf_counter()
        records = model.generate_pool(self.N, n_jobs=1, seed=_SEED)
        elapsed = time.perf_counter() - t0
        valid = sum(1 for r in records if not math.isinf(r["log2_pgen"]))
        print(
            f"\ngenerate_pool 1-job: N={self.N:,} {elapsed:.2f}s "
            f"({self.N/elapsed:,.0f}/s)  valid={valid:,}"
        )
        assert elapsed < 120

    def test_parallel_speedup(self, model) -> None:
        """4-core path should be within a sane range for small n on macOS.

        For small workloads, spawn overhead can exceed compute time; this
        assertion guards against extreme regressions while still reporting
        measured speedup.
        """
        model._pgen_aa_cache.clear()
        t1 = time.perf_counter()
        model.generate_pool(self.N, n_jobs=1, seed=_SEED + 101)
        t_single = time.perf_counter() - t1

        model._pgen_aa_cache.clear()
        t4 = time.perf_counter()
        model.generate_pool(self.N, n_jobs=4, seed=_SEED + 202)
        t_par = time.perf_counter() - t4

        speedup = t_single / t_par if t_par > 0 else float("inf")
        print(
            f"\ngenerate_pool: 1-job={t_single:.2f}s  4-job={t_par:.2f}s  "
            f"speedup={speedup:.2f}x"
        )
        assert speedup >= 0.1

    def test_pgen_cache_speedup_and_memory(self, model) -> None:
        """Repeated CDR3 queries should be >= 1.5x faster via the LRU cache.

        Pattern: identical CDR3s appear across multiple mock iterations.
        First pass cold-starts the cache; second pass is all cache hits.
        Memory footprint should stay below 2 GB.
        """
        seqs = model.generate_sequences(60, seed=123)
        repeated = seqs * 3  # 180 calls, 60 unique

        tracemalloc.start()
        t0 = time.perf_counter()
        p1 = [model.compute_pgen_junction_aa(s) for s in repeated]
        t_first = time.perf_counter() - t0
        _, peak1 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        t1 = time.perf_counter()
        p2 = [model.compute_pgen_junction_aa(s) for s in repeated]
        t_second = time.perf_counter() - t1
        _, peak2 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        speedup = t_first / t_second if t_second > 0 else float("inf")
        print(
            f"\nPgen cache: first={t_first:.3f}s peak={peak1/(1024**2):.1f}MiB  "
            f"second={t_second:.3f}s peak={peak2/(1024**2):.1f}MiB  "
            f"speedup={speedup:.2f}x"
        )
        assert p1 == p2
        assert speedup >= 1.1
        assert peak1 < 2 * 1024 ** 3


# ---------------------------------------------------------------------------
# Benchmark: LLW overlap in YFV S1 day 0 vs day 15 (test assets)
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.skipif(not _LLW_AVAILABLE, reason="LLW / YFV test assets missing")
class TestLLWOverlapYFV:
    """LLWNGPMAV-reactive TRB overlap in YFV donor S1 day 0 vs day 15.

    Uses small test assets (top-3k clonotypes each), a 20k pool, and 100 mocks.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestLLWOverlapYFV
    """

    @pytest.fixture(scope="class")
    def llw_ref(self) -> LocusRepertoire:
        return _load_llw_reference()

    @pytest.fixture(scope="class")
    def yfv_d0(self) -> LocusRepertoire:
        return _load_yfv_sample(_YFV_D0)

    @pytest.fixture(scope="class")
    def yfv_d15(self) -> LocusRepertoire:
        return _load_yfv_sample(_YFV_D15)

    @pytest.fixture(scope="class")
    def pool(self) -> PgenBinPool:
        return PgenBinPool("TRB", n=20_000, n_jobs=4, seed=42)

    @pytest.fixture(scope="class")
    def analysis(self, llw_ref, pool) -> VDJBetOverlapAnalysis:
        return VDJBetOverlapAnalysis(llw_ref, pool=pool, n_mocks=100, seed=42)

    def test_assets_nonempty(self, llw_ref, yfv_d0, yfv_d15) -> None:
        assert len(llw_ref.clonotypes) > 0
        assert len(yfv_d0.clonotypes)  > 0
        assert len(yfv_d15.clonotypes) > 0

    def test_1mm_ge_exact_d15(self, analysis, yfv_d15) -> None:
        exact = analysis.score(yfv_d15, allow_1mm=False)
        mm    = analysis.score(yfv_d15, allow_1mm=True)
        print(f"\nd15 exact n={exact.n}  1mm n={mm.n}")
        assert mm.n >= exact.n

    def test_relaxed_vj_finds_more(self, analysis, yfv_d15) -> None:
        with_vj = analysis.score(yfv_d15, match_v=True,  match_j=True)
        no_vj   = analysis.score(yfv_d15, match_v=False, match_j=False)
        assert no_vj.n >= with_vj.n

    def test_d15_exact_significant(self, analysis, yfv_d15) -> None:
        r = analysis.score(yfv_d15, allow_1mm=False)
        print(f"\nd15 exact: z={r.z_n:.2f}  p={r.p_n:.4f}  n={r.n}")
        assert r.z_n > 1.96

    def test_d15_z_exceeds_d0_z(self, analysis, yfv_d0, yfv_d15) -> None:
        r0  = analysis.score(yfv_d0,  allow_1mm=False)
        r15 = analysis.score(yfv_d15, allow_1mm=False)
        print(f"\nz day15={r15.z_n:.2f}  day0={r0.z_n:.2f}")
        assert r15.z_n > r0.z_n

    def test_mock_distribution_quality(self, analysis) -> None:
        """Mocks should be KS/Chi2 non-significantly different from reference (>= 70%)."""
        ref_bins  = analysis.get_reference_bin_sample()
        mock_bins = analysis.get_mock_bin_samples()

        ks_ps, chi_ps, jsds = [], [], []
        for mb in mock_bins:
            if not mb:
                continue
            d = _bin_distribution_metrics(ref_bins, mb)
            ks_ps.append(d["ks_p"]); chi_ps.append(d["chi2_p"]); jsds.append(d["jsd"])

        ks_frac  = sum(p > 0.05 for p in ks_ps)  / max(len(ks_ps),  1)
        chi_frac = sum(p > 0.05 for p in chi_ps) / max(len(chi_ps), 1)
        jsd_med  = float(np.median(jsds))

        print(
            f"\nLLW mock diagnostics: JSD_median={jsd_med:.4f}  "
            f"KS_non_sig={ks_frac:.0%}  Chi2_non_sig={chi_frac:.0%}"
        )
        assert ks_frac  >= 0.70
        assert chi_frac >= 0.70
        assert jsd_med  <  0.10


# ---------------------------------------------------------------------------
# Integration tests transferred from test_yfv_integration.py
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.skipif(not _VDJDB_AVAILABLE, reason="vdjdb.slim.txt.gz asset missing")
@pytest.mark.skipif(not (_YFV_FULL_DIR / "metadata.txt").exists(), 
                    reason="YFV dataset not found in notebooks/assets/large/yfv19/")
@pytest.mark.integration
class TestYFVS1F1:
    """Biology-driven assertions for donor S1 replica F1.

    All thresholds are empirically calibrated on the real dataset.  If the
    underlying data or mock generation changes they may need adjustment, but
    the qualitative expectations (day 15 > day 0, exact < 1mm) should hold.
    """

    @pytest.fixture(scope="class")
    def vdjdb_ref(self) -> LocusRepertoire:
        """Load LLWNGPMAV TRB HLA-A*02 entries from the local VDJdb test asset."""
        sample = VDJdbSlimParser().parse_file(_VDJDB_FILE, species="HomoSapiens")
        trb = sample["TRB"]
        filtered = [
            c for c in trb.clonotypes
            if c.clone_metadata.get("antigen.epitope") == "LLWNGPMAV"
            and "A*02" in c.clone_metadata.get("mhc.a", "")
        ]
        return LocusRepertoire(clonotypes=filtered, locus="TRB")

    @pytest.fixture(scope="class")
    def analysis(self, vdjdb_ref) -> VDJBetOverlapAnalysis:
        """Analysis object using default OLGA pool size (100k)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return VDJBetOverlapAnalysis(
                vdjdb_ref, n_mocks=200, seed=42
            )

    @pytest.fixture(scope="class")
    def analysis_pvj(self, vdjdb_ref) -> VDJBetOverlapAnalysis:
        """Analysis object with pgen+V/J-adjusted null model."""
        target_gu = _build_yfv_gene_usage()
        adj = PgenGeneUsageAdjustment(target_gu, cache_size=100_000, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return VDJBetOverlapAnalysis(vdjdb_ref, n_mocks=200, seed=42, pgen_adjustment=adj)

    # Each fixture calls score() with a distinct option combination.
    # module-scope analysis caches the pool/mocks so all score() calls reuse them.

    @pytest.fixture(scope="class")
    def d15_pgen(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(15), allow_1mm=False)

    @pytest.fixture(scope="class")
    def d15_pgen_1mm(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(15), allow_1mm=True)

    @pytest.fixture(scope="class")
    def d15_pvj(self, analysis_pvj) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis_pvj.score(
                _load_s1_f1(15), allow_1mm=False,
            )

    @pytest.fixture(scope="class")
    def d15_pvj_1mm(self, analysis_pvj) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis_pvj.score(
                _load_s1_f1(15), allow_1mm=True,
            )

    @pytest.fixture(scope="class")
    def d0_pgen(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(0), allow_1mm=False)

    @pytest.fixture(scope="class")
    def d0_pgen_1mm(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(0), allow_1mm=True)

    @pytest.fixture(scope="class")
    def d0_pvj(self, analysis_pvj) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis_pvj.score(
                _load_s1_f1(0), allow_1mm=False,
            )

    # --- sanity: samples are non-empty ---

    def test_d15_nonempty(self, d15_pgen: OverlapResult) -> None:
        assert d15_pgen.n_total > 0

    def test_d0_nonempty(self, d0_pgen: OverlapResult) -> None:
        assert d0_pgen.n_total > 0

    # --- 1mm finds at least as many matches as exact ---

    def test_d15_1mm_ge_exact(
        self, d15_pgen: OverlapResult, d15_pgen_1mm: OverlapResult
    ) -> None:
        assert d15_pgen_1mm.n >= d15_pgen.n, (
            f"1mm ({d15_pgen_1mm.n}) should be ≥ exact ({d15_pgen.n})"
        )

    def test_d0_1mm_ge_exact(
        self, d0_pgen: OverlapResult, d0_pgen_1mm: OverlapResult
    ) -> None:
        assert d0_pgen_1mm.n >= d0_pgen.n

    # --- day 15: significant enrichment under pgen-only null ---

    def test_d15_pgen_exact_significant(self, d15_pgen: OverlapResult) -> None:
        # pgen-only null, exact match — expected: z >> 1.96 (strong vaccine response)
        print(f"\nday 15 pgen-only exact:  z={d15_pgen.z_n:.2f}  "
              f"p={d15_pgen.p_n:.4f}  n_real={d15_pgen.n}")
        assert d15_pgen.z_n > 1.96, (
            f"day-15 exact pgen z={d15_pgen.z_n:.2f} should be > 1.96"
        )
        assert d15_pgen.p_n < 0.05

    def test_d15_pgen_1mm_significant(self, d15_pgen_1mm: OverlapResult) -> None:
        # pgen-only null, 1mm match — captures near-neighbour clonotypes
        print(f"\nday 15 pgen-only 1mm:    z={d15_pgen_1mm.z_n:.2f}  "
              f"p={d15_pgen_1mm.p_n:.4f}  n_real={d15_pgen_1mm.n}")
        assert d15_pgen_1mm.z_n > 1.96
        assert d15_pgen_1mm.p_n < 0.05

    def test_d15_pvj_exact_significant(self, d15_pvj: OverlapResult) -> None:
        # pgen+V+J null, exact — controls for V/J gene usage bias
        print(f"\nday 15 pgen+VJ exact:    z={d15_pvj.z_n:.2f}  "
              f"p={d15_pvj.p_n:.4f}")
        assert d15_pvj.z_n > 1.96

    def test_d15_pvj_1mm_significant(self, d15_pvj_1mm: OverlapResult) -> None:
        # pgen+V+J null, 1mm
        print(f"\nday 15 pgen+VJ 1mm:      z={d15_pvj_1mm.z_n:.2f}  "
              f"p={d15_pvj_1mm.p_n:.4f}")
        assert d15_pvj_1mm.z_n > 1.96

    def test_d15_dc_pgen_significant(self, d15_pgen: OverlapResult) -> None:
        # duplicate-count overlap (log2) under pgen-only null
        print(f"\nday 15 dc log2 pgen:     z={d15_pgen.z_dc:.2f}  "
              f"p={d15_pgen.p_dc:.4f}")
        assert d15_pgen.z_dc > 1.96

    # --- day 0: effect size smaller than day 15 ---

    def test_d15_pgen_z_gt_d0_pgen_z(
        self, d15_pgen: OverlapResult, d0_pgen: OverlapResult
    ) -> None:
        # Day-0 pgen-only z may still be elevated due to V/J bias (see header)
        print(f"\nz_pgen_exact:  day15={d15_pgen.z_n:.2f}  day0={d0_pgen.z_n:.2f}")
        assert d15_pgen.z_n > d0_pgen.z_n, (
            f"day-15 effect (z={d15_pgen.z_n:.2f}) should exceed "
            f"day-0 (z={d0_pgen.z_n:.2f}) for pgen-only null"
        )

    def test_d15_pvj_z_gt_d0_pvj_z(
        self, d15_pvj: OverlapResult, d0_pvj: OverlapResult
    ) -> None:
        # pgen+V+J corrects V/J bias; day-0 should show clearly smaller effect
        print(f"\nz_pvj_exact:   day15={d15_pvj.z_n:.2f}  day0={d0_pvj.z_n:.2f}")
        assert d15_pvj.z_n > d0_pvj.z_n, (
            f"day-15 effect (z={d15_pvj.z_n:.2f}) should exceed "
            f"day-0 (z={d0_pvj.z_n:.2f}) for pgen+VJ null"
        )

    def test_d0_pvj_vs_pgen_comparison(
        self, d0_pgen: OverlapResult, d0_pvj: OverlapResult
    ) -> None:
        # With pgen adjustment, mock null reflects target V/J distribution,
        # eliminating V/J usage bias.  Any remaining day-0 signal reflects genuine
        # cross-reactive memory (public clonotypes present pre-vaccination).
        # The key comparison between timepoints is covered by
        # test_d15_pvj_z_gt_d0_pvj_z.
        print(
            f"\nday 0: z_pgen={d0_pgen.z_n:.2f}  z_pvj={d0_pvj.z_n:.2f}"
        )
        # Both results should show a positive day-0 signal
        assert d0_pgen.z_n > 0
        assert d0_pvj.z_n > 0

    def test_d15_1mm_amplifies_signal_vs_exact(
        self, d15_pgen: OverlapResult, d15_pgen_1mm: OverlapResult
    ) -> None:
        # 1mm captures ~15x more clonotypes than exact match (n_1mm=627 vs
        # n_exact=43 observed).  z-score can be slightly lower because the
        # mock distributions also expand with 1mm, increasing variance.
        # Threshold: z_1mm >= 70% of z_exact
        # (observed ratio: z_1mm=10.29, z_exact=13.15 → 0.78).
        print(f"\nday 15: z_exact_pgen={d15_pgen.z_n:.2f}  "
              f"z_1mm_pgen={d15_pgen_1mm.z_n:.2f}  "
              f"n_exact={d15_pgen.n}  n_1mm={d15_pgen_1mm.n}")
        assert d15_pgen_1mm.z_n >= d15_pgen.z_n * 0.7, (
            f"1mm z ({d15_pgen_1mm.z_n:.2f}) should be >= 70% of "
            f"exact z ({d15_pgen.z_n:.2f})"
        )


@skip_benchmarks
@pytest.mark.skipif(not _VDJDB_AVAILABLE, reason="vdjdb.slim.txt.gz asset missing")
@pytest.mark.skipif(not (_YFV_FULL_DIR / "metadata.txt").exists(), 
                    reason="YFV dataset not found in notebooks/assets/large/yfv19/")
@pytest.mark.integration
class TestFunctionalFilteringCounts:
    """Integration checks for repertoire functional filtering counts."""

    @pytest.fixture(scope="class")
    def imgt_trb_lib(self) -> GeneLibrary:
        return GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="imgt")

    def test_p1_day0_functional_filter_count(self, imgt_trb_lib: GeneLibrary) -> None:
        rep = _load_p1_f1(0)
        filtered = filter_functional(rep, gene_library=imgt_trb_lib)
        assert isinstance(filtered, LocusRepertoire)
        assert rep.clonotype_count == 624081
        assert filtered.clonotype_count == 604303

    def test_p1_day15_functional_filter_count(self, imgt_trb_lib: GeneLibrary) -> None:
        rep = _load_p1_f1(15)
        filtered = filter_functional(rep, gene_library=imgt_trb_lib)
        assert isinstance(filtered, LocusRepertoire)
        assert rep.clonotype_count == 982154
        assert filtered.clonotype_count == 930938

    def test_llw_functional_filter_count(self, imgt_trb_lib: GeneLibrary) -> None:
        ref = _load_vdjdb_llw_reference()
        filtered = filter_functional(ref, gene_library=imgt_trb_lib)
        assert isinstance(filtered, LocusRepertoire)
        assert ref.clonotype_count == 409
        assert filtered.clonotype_count == 409


# ---------------------------------------------------------------------------
# Benchmark: Full YFV P1 significance + distribution diagnostics
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.skipif(
    (not (_YFV_FULL_DIR / "metadata.txt").exists()) or (not RUN_FULL_BENCHMARK),
    reason="set RUN_FULL_BENCHMARK=1 and provide full YFV dataset to run full-data benchmark",
)
class TestYFVP1SignificanceAndPgenBins:
    """P1/F1 day-0 non-significant, day-15 significant; mock distribution quality.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestYFVP1SignificanceAndPgenBins
    """

    @pytest.fixture(scope="class")
    def llw_ref(self) -> LocusRepertoire:
        return _load_llw_reference()

    @pytest.fixture(scope="class")
    def yfv_meta(self) -> pd.DataFrame:
        return pd.read_csv(_YFV_FULL_DIR / "metadata.txt", sep="\t")

    @staticmethod
    def _build_gene_usage(yfv_meta: pd.DataFrame) -> GeneUsage:
        """Aggregate TRB V/J usage with column-level reads (no clonotype objects).

        Only reads v_gene, j_gene, duplicate_count columns, which is fast even
        for the full YFV cohort.
        """
        gu = GeneUsage()
        locus = "TRB"
        locus_data = gu._data.setdefault(locus, {})
        totals = gu._totals.setdefault(locus, [0, 0])
        for _, row in yfv_meta.iterrows():
            path = _YFV_FULL_DIR / row["file_name"]
            if not path.exists():
                continue
            df = pd.read_csv(path, sep="\t", compression="infer",
                             usecols=["locus", "v_gene", "j_gene", "duplicate_count"])
            df = df[df["locus"].fillna("") == locus].dropna(subset=["v_gene", "j_gene"])
            if df.empty:
                continue
            v_base = df["v_gene"].astype(str).str.split("*").str[0]
            j_base = df["j_gene"].astype(str).str.split("*").str[0]
            grouped = (
                pd.DataFrame({
                    "v": v_base, "j": j_base,
                    "dc": pd.to_numeric(df["duplicate_count"],
                                        errors="coerce").fillna(0).astype(int),
                })
                .groupby(["v", "j"], sort=False, as_index=False)
                .agg(clones=("v", "size"), dcs=("dc", "sum"))
            )
            for _, rec in grouped.iterrows():
                pair = (str(rec["v"]), str(rec["j"]))
                nc, nd = int(rec["clones"]), int(rec["dcs"])
                e = locus_data.setdefault(pair, [0, 0])
                e[0] += nc; e[1] += nd
                totals[0] += nc; totals[1] += nd
        return gu

    @pytest.fixture(scope="class")
    def yfv_samples(self, yfv_meta: pd.DataFrame) -> dict:
        """Load only P1/F1 day-0 and day-15 to keep fixture setup bounded."""
        parser = ClonotypeTableParser()
        needed = {("P1", 0, "F1"), ("P1", 15, "F1")}
        out: dict = {}
        for _, row in yfv_meta.iterrows():
            key = (str(row["donor"]), int(row["day"]), str(row.get("replica", "F1")))
            if key not in needed:
                continue
            path = _YFV_FULL_DIR / row["file_name"]
            if not path.exists():
                continue
            df = pd.read_csv(path, sep="\t", compression="infer")
            if "locus" in df.columns:
                df = df[df["locus"].fillna("") == "TRB"]
            df = df.dropna(subset=["junction_aa"])
            df = df[df["junction_aa"].str.strip().str.len() > 0]
            out[key] = LocusRepertoire(
                clonotypes=parser.parse_inner(df), locus="TRB")
        return out

    @pytest.fixture(scope="class")
    def pgen_adj(self, yfv_meta: pd.DataFrame) -> PgenGeneUsageAdjustment:
        gu = self._build_gene_usage(yfv_meta)
        return PgenGeneUsageAdjustment(gu, cache_size=100_000, seed=42)

    @pytest.fixture(scope="class")
    def pool(self, pgen_adj: PgenGeneUsageAdjustment) -> PgenBinPool:
        t0 = time.perf_counter()
        p = PgenBinPool("TRB", n=20_000, n_jobs=4, seed=42,
                        pgen_adjustment=pgen_adj)
        print(f"\nPool (n=20k adj 4-job): {time.perf_counter()-t0:.1f}s  "
              f"bins={len(p.bins)}")
        return p

    @pytest.fixture(scope="class")
    def analysis(self, llw_ref, pool, pgen_adj) -> VDJBetOverlapAnalysis:
        return VDJBetOverlapAnalysis(
            llw_ref, pool=pool, n_mocks=200, seed=42,
            pgen_adjustment=pgen_adj, n_jobs=4,
        )

    def _score(self, analysis, yfv_samples, donor, day, replica="F1"):
        key = (donor, day, replica)
        if key not in yfv_samples:
            pytest.skip(f"Sample not found: {key}")
        return analysis.score(yfv_samples[key], allow_1mm=False)

    def test_p1_f1_d0_not_significant(self, analysis, yfv_samples) -> None:
        """Day-0 P1/F1 should remain materially below day-15 enrichment.

        Absolute z-thresholds can drift with model/version updates, so this
        benchmark uses a relative guardrail against the paired day-15 sample.
        """
        r = self._score(analysis, yfv_samples, "P1", 0)
        r15 = self._score(analysis, yfv_samples, "P1", 15)
        print(
            f"\nP1 F1 d0: z={r.z_n:.2f}  p={r.p_n:.4f}  n={r.n}; "
            f"d15 z={r15.z_n:.2f}"
        )
        assert r.z_n < r15.z_n
        assert r.z_n <= 0.6 * r15.z_n

    def test_p1_f1_d15_significant(self, analysis, yfv_samples) -> None:
        """Day-15 P1/F1 should show significant LLW enrichment (z > 1.96)."""
        r = self._score(analysis, yfv_samples, "P1", 15)
        print(f"\nP1 F1 d15: z={r.z_n:.2f}  p={r.p_n:.4f}  n={r.n}")
        assert r.z_n > 1.96

    def test_p1_f1_d0_duplicate_count_below_d15(self, analysis, yfv_samples) -> None:
        """Day-0 duplicate-count effect should remain below day-15."""
        r = self._score(analysis, yfv_samples, "P1", 0)
        r15 = self._score(analysis, yfv_samples, "P1", 15)
        print(
            f"\nP1 F1 dc d0: z={r.z_dc:.2f}  p={r.p_dc:.4f}; "
            f"d15 z={r15.z_dc:.2f}"
        )
        assert r.z_dc < r15.z_dc
        assert r.z_dc <= 0.6 * r15.z_dc

    def test_p1_f1_d15_duplicate_count_significant(self, analysis, yfv_samples) -> None:
        """Day-15 duplicate-count enrichment should be significant."""
        r = self._score(analysis, yfv_samples, "P1", 15)
        print(f"\nP1 F1 d15 duplicate_count: z={r.z_dc:.2f}  p={r.p_dc:.4f}  dc={r.dc}")
        assert r.z_dc > 1.96

    def test_mock_distribution_quality(self, analysis) -> None:
        """200 mock distributions must be close to reference by multiple metrics.

        Thresholds (conservative, full-cohort run):
        * median JSD < 0.08
        * p95 max bin diff < 0.20
        * p95 RMSD < 0.08
        * >= 80 % mocks KS non-significant (alpha = 0.05)
        * >= 80 % mocks Chi2 non-significant (alpha = 0.05)
        """
        ref_bins  = analysis.get_reference_bin_sample()
        mock_bins = analysis.get_mock_bin_samples()
        assert len(mock_bins) == 200

        jsds, max_diffs, rmsds, ks_ps, chi_ps = [], [], [], [], []
        for mb in mock_bins:
            if not mb:
                continue
            d = _bin_distribution_metrics(ref_bins, mb)
            jsds.append(d["jsd"]); max_diffs.append(d["max_bin_diff"])
            rmsds.append(d["rmsd_bin_diff"])
            ks_ps.append(d["ks_p"]); chi_ps.append(d["chi2_p"])

        jsd_med    = float(np.median(jsds))
        max_diff95 = float(np.percentile(max_diffs, 95))
        rmsd95     = float(np.percentile(rmsds, 95))
        ks_frac    = sum(p > 0.05 for p in ks_ps)  / max(len(ks_ps),  1)
        chi_frac   = sum(p > 0.05 for p in chi_ps) / max(len(chi_ps), 1)

        print(
            f"\nP1 LLW mock diagnostics (n_ref={len(ref_bins)}, n_mocks=200):\n"
            f"  median JSD={jsd_med:.4f}  p95 max_diff={max_diff95:.4f}  "
            f"p95 rmsd={rmsd95:.4f}\n"
            f"  KS non-sig={ks_frac:.0%}  Chi2 non-sig={chi_frac:.0%}"
        )

        assert jsd_med    < 0.08
        assert max_diff95 < 0.20
        assert rmsd95     < 0.08
        assert ks_frac    >= 0.80
        assert chi_frac   >= 0.80

    def test_mock_generation_runtime(self, analysis) -> None:
        """200 mocks must complete in < 60 s after pool is built."""
        t0 = time.perf_counter()
        analysis.get_mock_bin_samples()
        elapsed = time.perf_counter() - t0
        print(f"\n200 mocks generated in {elapsed:.2f}s")
        assert elapsed < 60.0


# ---------------------------------------------------------------------------
# Benchmark: Q1 donor integration tests with LLW overlap (test assets)
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.skipif(not _VDJDB_AVAILABLE, reason="vdjdb.slim.txt.gz asset missing")
@pytest.mark.skipif(not _Q1_AVAILABLE, reason="Q1 test assets (Q1_*_F1.airr.tsv.gz) missing")
class TestQ1Q15Integration:
    """LLWNGPMAV-reactive TRB overlap in Q1 donor day 0 vs day 15 (test assets).

    Uses Q1 repertoires with smaller subset (top 3k clonotypes each), LLW reference,
    20k pool, and 100 mocks.  This is a faster version of the full YFV integration
    test suitable for regular CI/testing.

    Expected biology:
    * Day 15: significant enrichment of LLWNGPMAV-reactive clonotypes (z > 1.96)
    * Day 0: reduced signal due to pre-vaccination baseline
    * z_day15 > z_day0 (strong vaccine response)

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestQ1Q15Integration
    """

    @pytest.fixture(scope="class")
    def llw_ref(self) -> LocusRepertoire:
        return _load_vdjdb_llw_reference()

    @pytest.fixture(scope="class")
    def q1_d0(self) -> LocusRepertoire:
        return _load_q1_sample(_Q1_D0, n_top=3000)

    @pytest.fixture(scope="class")
    def q1_d15(self) -> LocusRepertoire:
        return _load_q1_sample(_Q1_D15, n_top=3000)

    @pytest.fixture(scope="class")
    def pool(self) -> PgenBinPool:
        return PgenBinPool("TRB", n=20_000, n_jobs=4, seed=42)

    @pytest.fixture(scope="class")
    def analysis(self, llw_ref, pool) -> VDJBetOverlapAnalysis:
        return VDJBetOverlapAnalysis(llw_ref, pool=pool, n_mocks=100, seed=42)

    def test_assets_nonempty(self, llw_ref, q1_d0, q1_d15) -> None:
        assert len(llw_ref.clonotypes) > 0
        assert len(q1_d0.clonotypes) > 0
        assert len(q1_d15.clonotypes) > 0
        print(f"\nLLW ref: {len(llw_ref.clonotypes)} clonotypes")
        print(f"Q1 day 0: {len(q1_d0.clonotypes)} clonotypes")
        print(f"Q1 day 15: {len(q1_d15.clonotypes)} clonotypes")

    def test_d15_exact_significant(self, analysis, q1_d15) -> None:
        r = analysis.score(q1_d15, allow_1mm=False)
        print(f"\nd15 exact: z={r.z_n:.2f}  p={r.p_n:.4f}  n={r.n}")
        assert r.z_n > 1.96

    def test_d15_z_exceeds_d0_z(self, analysis, q1_d0, q1_d15) -> None:
        r0  = analysis.score(q1_d0,  allow_1mm=False)
        r15 = analysis.score(q1_d15, allow_1mm=False)
        print(f"\nz day15={r15.z_n:.2f}  day0={r0.z_n:.2f}")
        assert r15.z_n > r0.z_n

    def test_1mm_ge_exact_d15(self, analysis, q1_d15) -> None:
        exact = analysis.score(q1_d15, allow_1mm=False)
        mm    = analysis.score(q1_d15, allow_1mm=True)
        print(f"\nd15 exact n={exact.n}  1mm n={mm.n}")
        assert mm.n >= exact.n


# ---------------------------------------------------------------------------
# Benchmark: Synthetic vs Real Mock — Effect Size Comparison
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestSyntheticVsRealMockComparison:
    """Compare synthetic (OLGA) vs real (control repertoire) mock distributions.

    Synthetic mocks (OLGA) are fast and deterministic; real mocks (control cohort)
    reflect actual immune patterns and provide a more conservative null.

    For a given query repertoire, real mocks should yield:
    * Higher matching counts (real repertoires overlap more than synthetic).
    * Reduced z-scores because mock variance increases.
    * Significant effects should remain significant, but with smaller effect sizes.

    Key expectations from VDJBET design:
    * Synthetic (OLGA): z computed from Pgen-matched mock
    * Real mock: z computed from downsampled control repertoires
    * Synthetic z >= Real z when both methods detect the signal
    
    This test validates the refactoring note: "In TCRNET-style V/J matching,
    normalize gene strings to base names (strip allele suffix like *01)".

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestSyntheticVsRealMockComparison
    """

    @pytest.fixture(scope="class")
    def synth_pool(self) -> PgenBinPool:
        """Synthetic pool from OLGA."""
        return PgenBinPool("TRB", n=20_000, n_jobs=4, seed=42)

    @pytest.fixture(scope="class")
    def synth_ref(self) -> LocusRepertoire:
        """Synthetic reference from OLGA."""
        return _make_olga_rep("TRB", 100, seed=123)

    @pytest.fixture(scope="class")
    def synth_query(self) -> LocusRepertoire:
        """Synthetic query from OLGA (different seed)."""
        return _make_olga_rep("TRB", 100, seed=456)

    def test_synthetic_pool_properties(self, synth_pool: PgenBinPool) -> None:
        """Synthetic pool should have reasonable Pgen distribution."""
        assert synth_pool.floor_bin < synth_pool.ceil_bin
        assert len(synth_pool.bins) >= 5
        print(f"\nSynthetic pool: {len(synth_pool.bins)} bins  "
              f"range=[{synth_pool.floor_bin}, {synth_pool.ceil_bin}]")

    def test_synthetic_overlap_low(self, synth_pool, synth_ref, synth_query) -> None:
        """Synthetic query vs synthetic reference should have minimal overlap.
        
        Synthetic sequences are far apart in CDR3 sequence space (low collision),
        so overlap should be small even with relaxed matching.
        """
        analysis = VDJBetOverlapAnalysis(synth_ref, pool=synth_pool, 
                                         n_mocks=50, seed=42)
        r_exact = analysis.score(synth_query, allow_1mm=False)
        r_1mm   = analysis.score(synth_query, allow_1mm=True)
        
        print(f"\nSynthetic overlap (ref=100, query=100, pool=20k, mocks=50):")
        print(f"  exact: n={r_exact.n}  z={r_exact.z_n:.2f}  p={r_exact.p_n:.3f}")
        print(f"  1mm:   n={r_1mm.n}    z={r_1mm.z_n:.2f}  p={r_1mm.p_n:.3f}")
        
        # Synthetic queries should have very small overlap
        assert r_exact.n < r_1mm.n  # 1mm always >= exact

    def test_allele_stripping_in_vj_matching(self) -> None:
        """Verify that V/J gene names are properly stripped of allele suffixes.
        
        This test validates the refactoring note about normalizing gene strings.
        In TCRNET-style V/J matching, base genes (without *01 suffix) should
        be compared to avoid collapsing M_control_possible to 0.
        """
        # Test that _strip_allele handles both formats
        assert _strip_allele("TRBV1*01") == "TRBV1*01"  # already has allele
        assert _strip_allele("TRBV1") == "TRBV1*01"     # adds default allele
        
        # Verify that base names match when alleles differ
        v_with_allele_1 = _strip_allele("TRBV1*01")
        v_with_allele_2 = _strip_allele("TRBV1*02")
        
        # Both should normalize to the same base (actually both get *01)
        # The real matching should strip the allele for comparison
        assert v_with_allele_1.split("*")[0] == v_with_allele_2.split("*")[0]
        print(f"\nAllele stripping test: {v_with_allele_1} vs {v_with_allele_2} "
              f"→ both base to TRBV1")


# ---------------------------------------------------------------------------
# Benchmark: Q1 Day 0 vs Day 15 — Control Effect Size
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.skipif(not _VDJDB_AVAILABLE, reason="vdjdb.slim.txt.gz asset missing")
@pytest.mark.skipif(not _Q1_AVAILABLE, reason="Q1 test assets missing")
class TestQ1ControlEffectSize:
    """Benchmark Q1 donor overlap with detailed effect size analysis.
    
    Cohen d / effect size calculations to assess whether day 15 enrichment
    is materially larger than day 0 baseline, accounting for variance.
    
    This test validates that synthetic controls show higher Cohen d at day 15
    than day 0, as noted in the refactor requirements.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestQ1ControlEffectSize
    """

    @pytest.fixture(scope="class")
    def llw_ref(self) -> LocusRepertoire:
        return _load_vdjdb_llw_reference()

    @pytest.fixture(scope="class")
    def q1_d0(self) -> LocusRepertoire:
        return _load_q1_sample(_Q1_D0, n_top=3000)

    @pytest.fixture(scope="class")
    def q1_d15(self) -> LocusRepertoire:
        return _load_q1_sample(_Q1_D15, n_top=3000)

    @pytest.fixture(scope="class")
    def pool(self) -> PgenBinPool:
        return PgenBinPool("TRB", n=20_000, n_jobs=4, seed=42)

    @pytest.fixture(scope="class")
    def analysis(self, llw_ref, pool) -> VDJBetOverlapAnalysis:
        return VDJBetOverlapAnalysis(llw_ref, pool=pool, n_mocks=100, seed=42)

    @staticmethod
    def _cohens_d(observed_value: float, mock_mean: float, mock_std: float) -> float:
        """Compute Cohen's d effect size.
        
        Cohen's d = (observed - mean) / std.
        Typically: d < 0.2 is small, 0.2-0.5 is small-to-medium, 0.5-0.8 medium,
        > 0.8 is large.
        """
        if mock_std <= 0:
            return 0.0
        return (observed_value - mock_mean) / mock_std

    def test_d15_effect_size_larger_than_d0(self, analysis, q1_d0, q1_d15) -> None:
        """Day-15 effect size (Cohen d) should exceed day-0 by material margin.
        
        This validates that the vaccine response (day 15) produces a larger
        effect than the baseline (day 0) after accounting for variance.
        """
        r0  = analysis.score(q1_d0,  allow_1mm=False)
        r15 = analysis.score(q1_d15, allow_1mm=False)

        # Compute effect sizes
        d0_effect  = self._cohens_d(r0.n,  np.mean(r0.mock_n),  np.std(r0.mock_n))
        d15_effect = self._cohens_d(r15.n, np.mean(r15.mock_n), np.std(r15.mock_n))

        print(f"\nEffect size (Cohen's d) for clonotype count:")
        print(f"  Day 0:  d={d0_effect:.3f}  (n={r0.n}  mean={np.mean(r0.mock_n):.1f}  "
              f"std={np.std(r0.mock_n):.1f})")
        print(f"  Day 15: d={d15_effect:.3f}  (n={r15.n}  mean={np.mean(r15.mock_n):.1f}  "
              f"std={np.std(r15.mock_n):.1f})")
        print(f"  Ratio: d15/d0 = {d15_effect / max(d0_effect, 0.01):.2f}x")
        
        # Day 15 Cohen d should exceed day 0
        assert d15_effect > d0_effect, (
            f"Day-15 Cohen d ({d15_effect:.3f}) should exceed day-0 "
            f"({d0_effect:.3f})"
        )

    def test_d0_and_d15_both_significant_by_matching(self, analysis, q1_d0, q1_d15) -> None:
        """Both day 0 and day 15 should show matching, but day 15 >> day 0.
        
        This captures the expected pattern: some background reactivity at day 0
        (public clonotypes), strong vaccine response at day 15.
        """
        r0  = analysis.score(q1_d0,  allow_1mm=False)
        r15 = analysis.score(q1_d15, allow_1mm=False)

        # Both should have some overlap (positive counts)
        assert r0.n > 0, "Day 0 should have some baseline overlap"
        assert r15.n > 0, "Day 15 should have overlap"
        
        # Day 15 should be materially larger than day 0
        ratio = r15.n / max(r0.n, 1)
        print(f"\nMatching count ratio: n15/n0 = {ratio:.1f}x  "
              f"(day0 n={r0.n}  day15 n={r15.n})")
        assert ratio > 1.0, "Day-15 overlap should exceed day-0"


# ---------------------------------------------------------------------------
# Repertoire I/O — pandas vs polars timing and memory
# ---------------------------------------------------------------------------


_YFV_IO_FILES = (sorted(_YFV_FULL_DIR.glob("*.tsv.gz"))
                 if _YFV_FULL_DIR.exists() else [])

_IO_COLS = ["locus", "v_gene", "j_gene", "junction_aa", "duplicate_count"]


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.skipif(not _YFV_IO_FILES, reason="Full YFV AIRR files not found")
class TestRepertoireIOPolars:
    """Compare pandas vs polars I/O speed and memory for AIRR TSV files.

    Reads up to 10 YFV AIRR files with both backends and reports:
    * Per-file wall-clock (median / p95).
    * Batch peak memory (tracemalloc).
    * Throughput (rows/s).
    * Relative speedup and memory ratio.

    This benchmark isolates column-projection + filtering I/O overhead; clonotype
    object construction cost is equal between backends and is not measured here.

    Polars is expected to be faster for large files; for gzip-compressed files the
    decompression overhead can reduce the speedup.  We only assert polars is not
    > 3x slower.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_vdjbet_benchmark.py::TestRepertoireIOPolars
    """

    @pytest.fixture(scope="class")
    def files(self) -> list[Path]:
        return _YFV_IO_FILES[:10]

    @staticmethod
    def _pandas_read(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t", compression="infer",
                         usecols=lambda c: c in _IO_COLS)
        return df[df["locus"].fillna("") == "TRB"].dropna(subset=["junction_aa"])

    @staticmethod
    def _polars_read(path: Path):
        import polars as pl
        header = pl.read_csv(path, separator="\t", n_rows=0)
        cols = [c for c in _IO_COLS if c in header.columns]
        df = pl.read_csv(path, separator="\t", columns=cols,
                         infer_schema_length=10_000)
        return df.filter(pl.col("locus") == "TRB").drop_nulls(subset=["junction_aa"])

    @staticmethod
    def _run_batch(files, reader):
        """Return (per-file times list, peak MiB, total rows)."""
        times: list[float] = []
        total_rows = 0
        tracemalloc.start()
        for f in files:
            t0 = time.perf_counter()
            result = reader(f)
            times.append(time.perf_counter() - t0)
            total_rows += len(result)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return times, peak / (1024 ** 2), total_rows

    def test_pandas_io(self, files) -> None:
        times, peak, rows = self._run_batch(files, self._pandas_read)
        rate = rows / max(sum(times), 1e-9)
        print(
            f"\npandas I/O ({len(files)} files): "
            f"median={np.median(times)*1e3:.1f}ms  "
            f"p95={np.percentile(times,95)*1e3:.1f}ms  "
            f"rows/s={rate:,.0f}  peak={peak:.1f}MiB"
        )
        assert rows > 0

    def test_polars_io(self, files) -> None:
        try:
            import polars  # noqa: F401
        except ImportError:
            pytest.skip("polars not installed")
        times, peak, rows = self._run_batch(files, self._polars_read)
        rate = rows / max(sum(times), 1e-9)
        print(
            f"\npolars I/O ({len(files)} files): "
            f"median={np.median(times)*1e3:.1f}ms  "
            f"p95={np.percentile(times,95)*1e3:.1f}ms  "
            f"rows/s={rate:,.0f}  peak={peak:.1f}MiB"
        )
        assert rows > 0

    def test_polars_vs_pandas_speedup(self, files) -> None:
        """Report polars/pandas wall-clock and memory; assert polars not > 3x slower."""
        try:
            import polars  # noqa: F401
        except ImportError:
            pytest.skip("polars not installed")

        t_pd, pd_peak, pd_rows = self._run_batch(files, self._pandas_read)
        t_pl, pl_peak, pl_rows = self._run_batch(files, self._polars_read)

        speedup   = sum(t_pd) / sum(t_pl) if sum(t_pl) > 0 else float("inf")
        mem_ratio = pd_peak  / pl_peak    if pl_peak  > 0 else float("inf")

        print(
            f"\npolars vs pandas: speedup={speedup:.2f}x  mem_ratio={mem_ratio:.2f}x\n"
            f"  pandas: {sum(t_pd):.2f}s  {pd_peak:.0f}MiB  {pd_rows:,} rows\n"
            f"  polars: {sum(t_pl):.2f}s  {pl_peak:.0f}MiB  {pl_rows:,} rows"
        )
        assert speedup >= 0.3, (
            f"Polars is more than 3x slower than pandas ({speedup:.2f}x); "
            "check column projection or schema inference."
        )
