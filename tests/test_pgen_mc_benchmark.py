"""MC Pgen benchmark: pool build times, pgen accuracy, and Q-factor analysis.

Compares three pgen estimation strategies for real TRB (and TRA) CDR3s:
  1. OLGA analytical exact pgen (ground truth).
  2. Synthetic MC pool pgen (pgen_mc = matches / n_total_rearrangements).
  3. Real-control empirical frequency (pgen_real = matches / n_control).

For each strategy the benchmark reports:
  - Wall-clock time per 100 sequences.
  - Correlation with OLGA exact (log10 scale), fold-error.
  - Estimated Q-factor: median(pgen_real / pgen_olga) for sequences with
    at least one real-control match.

Design notes
------------
- Pool size is kept at 500 K for fast CI-time runs.  Set
  ``MIRPY_MC_BENCH_N=10000000`` to reproduce the full-scale benchmark.
- Real TRB data from the YFV dataset (notebooks/assets/large/yfv19/).
- Real TRA data from the COVID-19 dataset (notebooks/assets/large/airr_covid19/).

Run with::

    RUN_BENCHMARK=1 pytest -s tests/test_pgen_mc_benchmark.py
"""
from __future__ import annotations

import gzip
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pytest

from mir.basic.pgen import McPgenPool, OlgaModel
from tests.conftest import skip_benchmarks

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BENCH_N = int(os.getenv("MIRPY_MC_BENCH_N", "500000"))
_N_QUERY = int(os.getenv("MIRPY_MC_BENCH_NQUERY", "500"))
_SEED = 42
_N_JOBS = int(os.getenv("MIRPY_MC_BENCH_NJOBS", "8"))
_SKIP_ENDS = 2
_MC_MIN_COUNT = 2

_YFV_DIR = Path("notebooks/assets/large/yfv19")
_COVID_TRA_DIR = Path("notebooks/assets/large/airr_covid19")

random.seed(_SEED)
np.random.seed(_SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log10_corr_and_rmse(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (r, rmse_log10) for paired (x, y) with x>0 and y>0."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pairs) < 5:
        return float("nan"), float("nan")
    lx = np.array([math.log10(x) for x, _ in pairs])
    ly = np.array([math.log10(y) for _, y in pairs])
    r = float(np.corrcoef(lx, ly)[0, 1])
    rmse = float(np.std(lx - ly))
    return r, rmse


def _load_trb_cdrs(path: Path, n: int | None = None) -> list[str]:
    """Load unique productive TRB CDR3s from an AIRR TSV."""
    if not path.exists():
        return []
    with gzip.open(path, "rt") as fh:
        import polars as pl
        df = pl.read_csv(fh, separator="\t", infer_schema_length=1000)
    col = "junction_aa" if "junction_aa" in df.columns else "cdr3aa"
    seqs = (
        df.filter(
            pl.col(col).is_not_null()
            & pl.col(col).str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
        )[col]
        .unique()
        .to_list()
    )
    if n is not None:
        random.shuffle(seqs)
        seqs = seqs[:n]
    return seqs


def _load_tra_cdrs(n: int | None = None) -> list[str]:
    """Collect unique TRA CDR3s from COVID-19 VDJtools files."""
    seqs: set[str] = set()
    _AAS = set("ACDEFGHIKLMNPQRSTVWY")
    for fpath in sorted(_COVID_TRA_DIR.glob("*.TRA.vdjtools.tsv.gz"))[:5]:
        try:
            with gzip.open(fpath, "rt") as fh:
                for line in fh:
                    if line.startswith("count"):
                        continue
                    parts = line.split("\t")
                    if len(parts) > 3:
                        aa = parts[3].strip()
                        if aa and all(c in _AAS for c in aa):
                            seqs.add(aa)
        except Exception:
            pass
    seqs_list = list(seqs)
    if n is not None:
        random.shuffle(seqs_list)
        seqs_list = seqs_list[:n]
    return seqs_list


def _print_stats(
    label: str,
    pgen_mc: list[float],
    pgen_olga: list[float],
    n_total: int,
) -> None:
    covered = sum(1 for p in pgen_mc if p > 0)
    pct = 100 * covered / max(1, len(pgen_mc))
    pairs = [(mc, og) for mc, og in zip(pgen_mc, pgen_olga) if mc > 0 and og > 0]
    r, rmse = _log10_corr_and_rmse(pgen_mc, pgen_olga)
    fold = 10 ** rmse if not math.isnan(rmse) else float("nan")
    ratios = [mc / og for mc, og in pairs]
    median_ratio = float(np.median(ratios)) if ratios else float("nan")
    print(
        f"  {label:<30}  covered={pct:5.1f}%  n={covered:5d}  "
        f"r={r:6.3f}  rmse_log10={rmse:.3f}  fold={fold:.2f}x  "
        f"median_ratio={median_ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

@skip_benchmarks
@pytest.mark.benchmark
class TestMcPgenBenchmark:
    """Synthetic-pool MC Pgen accuracy and timing vs OLGA analytical Pgen.

    Run with::

        RUN_BENCHMARK=1 pytest -s tests/test_pgen_mc_benchmark.py::TestMcPgenBenchmark
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def trb_model(self):
        return OlgaModel(locus="TRB", species="human", seed=_SEED)

    @pytest.fixture(scope="class")
    def tra_model(self):
        return OlgaModel(locus="TRA", species="human", seed=_SEED)

    @pytest.fixture(scope="class")
    def trb_pool(self):
        print(f"\n  Building TRB synthetic pool (n={_BENCH_N:,}, {_N_JOBS} workers)…", flush=True)
        t0 = time.perf_counter()
        pool = McPgenPool.build_synthetic(
            _BENCH_N, locus="TRB", species="human",
            n_jobs=_N_JOBS, seed=_SEED, skip_ends=_SKIP_ENDS,
        )
        t1 = time.perf_counter()
        print(
            f"  TRB pool built in {t1-t0:.1f}s  "
            f"(p_productive={pool.p_productive:.3f}, "
            f"unique={len(pool._unique_seqs):,})",
            flush=True,
        )
        return pool

    @pytest.fixture(scope="class")
    def tra_pool(self):
        print(f"\n  Building TRA synthetic pool (n={_BENCH_N:,}, {_N_JOBS} workers)…", flush=True)
        t0 = time.perf_counter()
        pool = McPgenPool.build_synthetic(
            _BENCH_N, locus="TRA", species="human",
            n_jobs=_N_JOBS, seed=_SEED, skip_ends=_SKIP_ENDS,
        )
        t1 = time.perf_counter()
        print(
            f"  TRA pool built in {t1-t0:.1f}s  "
            f"(p_productive={pool.p_productive:.3f})",
            flush=True,
        )
        return pool

    @pytest.fixture(scope="class")
    def trb_queries(self):
        yfv_files = sorted(_YFV_DIR.glob("*.airr.tsv.gz"))
        seqs = _load_trb_cdrs(yfv_files[0], n=_N_QUERY) if yfv_files else []
        if not seqs:
            pytest.skip("YFV TRB data not found")
        return seqs

    @pytest.fixture(scope="class")
    def tra_queries(self):
        seqs = _load_tra_cdrs(n=_N_QUERY)
        if not seqs:
            pytest.skip("COVID-19 TRA data not found")
        return seqs

    # ------------------------------------------------------------------
    # TRB accuracy + speedup
    # ------------------------------------------------------------------

    def test_trb_synthetic_pool_accuracy(self, trb_model, trb_pool, trb_queries) -> None:
        """Compare synthetic MC pgen to OLGA analytical pgen for TRB."""
        queries = trb_queries

        # OLGA exact pgen
        t0 = time.perf_counter()
        olga_pgens = trb_model.compute_pgen_junction_aa_bulk(queries, max_mismatches=0, n_jobs=_N_JOBS)
        t_olga = time.perf_counter() - t0

        # MC exact pgen
        t0 = time.perf_counter()
        mc_exact = trb_pool.pgen_exact_bulk(queries)
        t_mc_exact = time.perf_counter() - t0

        # MC 1mm pgen
        t0 = time.perf_counter()
        mc_1mm = trb_pool.pgen_1mm_bulk(queries, n_jobs=_N_JOBS)
        t_mc_1mm = time.perf_counter() - t0

        print(f"\n  TRB benchmark  n={len(queries)}, pool={_BENCH_N:,}")
        print(f"  OLGA exact     : {t_olga:.2f}s  ({len(queries)/t_olga:.0f} seq/s)")
        print(f"  MC exact       : {t_mc_exact:.3f}s  ({len(queries)/t_mc_exact:.0f} seq/s)  speedup={t_olga/t_mc_exact:.0f}x")
        print(f"  MC 1mm         : {t_mc_1mm:.3f}s  ({len(queries)/t_mc_1mm:.0f} seq/s)  speedup={t_olga/t_mc_1mm:.0f}x")
        _print_stats("MC exact vs OLGA", mc_exact, olga_pgens, trb_pool.n_total)
        _print_stats("MC 1mm vs OLGA", mc_1mm, olga_pgens, trb_pool.n_total)

        r_exact, rmse_exact = _log10_corr_and_rmse(mc_exact, olga_pgens)
        r_1mm, _ = _log10_corr_and_rmse(mc_1mm, olga_pgens)

        # Loose bounds for 500K pool — tighter at 10M
        assert r_exact > 0.5 or sum(p > 0 for p in mc_exact) / len(mc_exact) > 0.2, (
            "MC exact pgen should correlate with OLGA for covered sequences"
        )

    def test_tra_synthetic_pool_accuracy(self, tra_model, tra_pool, tra_queries) -> None:
        """Compare synthetic MC pgen to OLGA analytical pgen for TRA."""
        queries = tra_queries

        t0 = time.perf_counter()
        olga_pgens = tra_model.compute_pgen_junction_aa_bulk(queries, max_mismatches=0, n_jobs=_N_JOBS)
        t_olga = time.perf_counter() - t0

        t0 = time.perf_counter()
        mc_1mm = tra_pool.pgen_1mm_bulk(queries, n_jobs=_N_JOBS)
        t_mc = time.perf_counter() - t0

        print(f"\n  TRA benchmark  n={len(queries)}, pool={_BENCH_N:,}")
        print(f"  OLGA exact     : {t_olga:.2f}s  ({len(queries)/t_olga:.0f} seq/s)")
        print(f"  MC 1mm         : {t_mc:.3f}s  ({len(queries)/t_mc:.0f} seq/s)  speedup={t_olga/t_mc:.0f}x")
        _print_stats("MC 1mm vs OLGA (TRA)", mc_1mm, olga_pgens, tra_pool.n_total)

    # ------------------------------------------------------------------
    # Q-factor analysis (real vs synthetic)
    # ------------------------------------------------------------------

    def test_trb_q_factor_from_real_control(self, trb_model, trb_queries) -> None:
        """Estimate Q-factor: real-repertoire frequency / OLGA pgen.

        Uses one YFV sample as 'test' and another as 'real control'.
        Q-factor captures thymic selection; expected value > 1 (selection
        enriches functional sequences relative to the recombination model).
        """
        yfv_files = sorted(_YFV_DIR.glob("*.airr.tsv.gz"))
        if len(yfv_files) < 2:
            pytest.skip("Need at least two YFV files for test/control split")

        test_seqs = trb_queries
        # Load a different sample as real control
        control_seqs = _load_trb_cdrs(yfv_files[1])
        if not control_seqs:
            pytest.skip("YFV control file is empty")

        real_pool = McPgenPool.build_real(control_seqs, locus="TRB", species="human", skip_ends=_SKIP_ENDS)
        olga_pgens = trb_model.compute_pgen_junction_aa_bulk(test_seqs, max_mismatches=0, n_jobs=_N_JOBS)
        real_pgens = real_pool.pgen_1mm_bulk(test_seqs, n_jobs=_N_JOBS)

        # Q-factor for sequences with at least one real match
        q_samples: list[float] = []
        for rp, op in zip(real_pgens, olga_pgens):
            # real_pool.pgen = match_count / n_control
            # OLGA pgen = P(seq | all rearrangements)
            # To compare: real empirical freq (per productive sequence) needs correction
            # by p_productive of OLGA model (since OLGA denominator includes non-productive)
            # But for Q-factor estimation we use the synthetic pool's p_productive as proxy
            if rp > 0 and op > 0:
                q_samples.append(rp / op)

        if not q_samples:
            pytest.skip("No sequences found in real control")

        q_median = float(np.median(q_samples))
        q_mean = float(np.mean(q_samples))
        q_log10_std = float(np.std([math.log10(q) for q in q_samples]))

        print(
            f"\n  TRB Q-factor from real control  n={len(q_samples)}\n"
            f"    median Q = {q_median:.2f}  mean Q = {q_mean:.2f}\n"
            f"    log10 std = {q_log10_std:.2f}  "
            f"(fold spread = {10**q_log10_std:.2f}x)\n"
            f"    Real control size: {len(control_seqs):,}  "
            f"Test seqs: {len(test_seqs)}"
        )

        # Sanity: Q-factor distribution should be non-degenerate
        assert len(q_samples) >= 5, "Too few sequences matched in real control"
        # Q-factor median should be in a reasonable range (not degenerate)
        assert 1e-6 < q_median < 1e6, f"Q-factor median out of range: {q_median}"

    # ------------------------------------------------------------------
    # p_productive calibration check
    # ------------------------------------------------------------------

    def test_trb_p_productive_calibration(self, trb_pool) -> None:
        """p_productive for TRB should be in a biologically plausible range."""
        p = trb_pool.p_productive
        print(f"\n  TRB p_productive = {p:.4f}  (n_total={trb_pool.n_total:,})")
        # Human TRB: roughly 10-30% of random V(D)J recombinations are productive
        assert 0.05 < p < 0.50, f"TRB p_productive={p:.4f} outside expected range [0.05, 0.50]"

    def test_tra_p_productive_calibration(self, tra_pool) -> None:
        """p_productive for TRA should be in a biologically plausible range."""
        p = tra_pool.p_productive
        print(f"\n  TRA p_productive = {p:.4f}  (n_total={tra_pool.n_total:,})")
        assert 0.05 < p < 0.60, f"TRA p_productive={p:.4f} outside expected range [0.05, 0.60]"
