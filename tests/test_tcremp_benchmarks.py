"""Benchmarks for TCREmp: throughput, multiprocessing scaling, and embedding quality.

Run with:
    env RUN_BENCHMARK=1 pytest tests/test_tcremp_benchmarks.py -s -v

Three benchmark suites:
  BenchmarkTCREmpDistanceCorrelation
      Embed 1 000 prototypes (as input clonotypes) against 1 000 prototypes.
      Compute pairwise sequence-space and latent-space distances, then report
      Pearson R², Spearman rho, and significance for the correlation.

  BenchmarkTCREmpThroughput
      Wall time + peak memory for 10 k / 50 k / 100 k clonotypes × 1 000 / 3 000
      prototypes at n_jobs=1.

  BenchmarkTCREmpMultiprocessing
      Compare n_jobs=1, 2, 4, 8 for 1 000 / 10 000 / 50 000 clonotypes × 1 000
      prototypes; report speedup ratio and note where spawn overhead dominates.
"""

from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np
import pytest
from scipy import stats

from mir.common.clonotype import Clonotype
from mir.embedding.tcremp import TCREmp

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_BENCHMARK"),
    reason="set RUN_BENCHMARK=1 to run",
)

_N_CPUS = os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clonotype(v: str, j: str, cdr3: str) -> Clonotype:
    return Clonotype(v_gene=v, j_gene=j, junction_aa=cdr3)


def _measure(fn):
    """Run fn(), return (result, elapsed_s, peak_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 ** 2)


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Return (N, N) float64 pairwise L2 distance matrix for rows of X."""
    X64 = X.astype(np.float64)
    sq = np.einsum("ij,ij->i", X64, X64)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X64 @ X64.T)
    np.clip(D2, 0.0, None, out=D2)
    return np.sqrt(D2)


def _print_separator():
    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# Correlation benchmark: sequence-space vs latent-space distances
# ---------------------------------------------------------------------------

class TestBenchmarkDistanceCorrelation:
    """Embed 1 000 prototypes against 1 000 prototypes; measure R² and rho.

    Since the input clonotypes ARE the prototypes, the embedding matrix X has a
    direct interpretation:
        X[i, 3*k]     = d_V  (prototype_i, prototype_k)
        X[i, 3*k + 1] = d_J  (prototype_i, prototype_k)
        X[i, 3*k + 2] = d_CDR3(prototype_i, prototype_k)

    So the "sequence-space distance" between prototypes i and j is:
        d_seq(i,j) = X[i,3*j] + X[i,3*j+1] + X[i,3*j+2]

    And the "latent-space distance" is the Euclidean L2 norm:
        d_emb(i,j) = ||X[i] - X[j]||_2

    Pearson R² and Spearman rho quantify how well the embedding preserves the
    original pairwise distance structure.
    """

    N_PROTO = 1000

    @pytest.fixture(scope="class")
    def model_and_X(self):
        _print_separator()
        print(f"[CORR] Building TCREmp: human TRB, {self.N_PROTO} prototypes")
        model, t_build, mb_build = _measure(
            lambda: TCREmp.from_defaults("human", "TRB", n_prototypes=self.N_PROTO)
        )
        print(f"       model build: {t_build:.2f}s  peak={mb_build:.0f} MB")

        clonotypes = [
            _clonotype(r["v_gene"], r["j_gene"], r["junction_aa"])
            for r in model.prototypes.iter_rows(named=True)
        ]
        print(f"[CORR] Embedding {self.N_PROTO} clonotypes × {self.N_PROTO} prototypes (n_jobs=1)")
        X, t_embed, mb_embed = _measure(lambda: model.embed(clonotypes, n_jobs=1))
        print(f"       embed: {t_embed:.2f}s  peak={mb_embed:.0f} MB  shape={X.shape}")
        return model, X

    def test_embed_shape(self, model_and_X):
        model, X = model_and_X
        assert X.shape == (self.N_PROTO, 3 * self.N_PROTO)
        assert X.dtype == np.float32

    def test_distance_correlation(self, model_and_X):
        """Pearson R² and Spearman rho between sequence-space and latent-space distances."""
        model, X = model_and_X
        N = self.N_PROTO

        # Sequence-space pairwise distances (direct from embedding matrix)
        # d_seq[i,j] = X[i,3*j] + X[i,3*j+1] + X[i,3*j+2]
        t0 = time.perf_counter()
        d_seq = (X[:, 0::3] + X[:, 1::3] + X[:, 2::3]).astype(np.float64)  # (N, N)

        # Latent-space pairwise Euclidean distances
        d_emb = _pairwise_euclidean(X)  # (N, N)
        t_dist = time.perf_counter() - t0

        # Upper triangle (exclude diagonal)
        idx = np.triu_indices(N, k=1)
        seq_flat = d_seq[idx]
        emb_flat = d_emb[idx]
        n_pairs = len(seq_flat)

        pearson_r, pearson_p = stats.pearsonr(seq_flat, emb_flat)
        spearman_r, spearman_p = stats.spearmanr(seq_flat, emb_flat)
        r2 = pearson_r ** 2

        _print_separator()
        print(
            f"\n[CORR] Distance correlation  (N={N}, pairs={n_pairs:,})  "
            f"dist_compute={t_dist:.2f}s"
        )
        print(f"       Pearson  r={pearson_r:.4f}  R²={r2:.4f}  p={pearson_p:.2e}")
        print(f"       Spearman ρ={spearman_r:.4f}          p={spearman_p:.2e}")
        _print_separator()

        # Sanity assertions — not hard acceptance thresholds
        assert r2 > 0.0, "R² must be positive"
        assert pearson_p < 0.05, "Correlation must be significant"
        assert spearman_r > 0.0, "Spearman rho must be positive"

    def test_per_component_correlation(self, model_and_X):
        """Pearson R² for each component (V, J, CDR3) vs total sequence distance."""
        model, X = model_and_X
        N = self.N_PROTO
        idx = np.triu_indices(N, k=1)

        d_total = (X[:, 0::3] + X[:, 1::3] + X[:, 2::3]).astype(np.float64)[idx]
        d_emb = _pairwise_euclidean(X)[idx]

        print(f"\n[CORR] Per-component Pearson R² vs total seq distance (N={N})")
        for comp, label in [(0, "V"), (1, "J"), (2, "CDR3")]:
            d_comp = X[:, comp::3].astype(np.float64)[idx]
            r, _ = stats.pearsonr(d_total, d_comp)
            print(f"       {label:5s}: R²={r**2:.4f}")

        r_emb, _ = stats.pearsonr(d_total, d_emb)
        print(f"       L2-emb: R²={r_emb**2:.4f}")


# ---------------------------------------------------------------------------
# Throughput benchmark: wall time and peak memory at scale
# ---------------------------------------------------------------------------

class TestBenchmarkThroughput:
    """Wall time and peak memory for single-process embedding at various scales."""

    @pytest.fixture(scope="class")
    def models(self):
        print("\n[THROUGHPUT] Building models...")
        return {
            n: TCREmp.from_defaults("human", "TRB", n_prototypes=n)
            for n in (1000, 3000)
        }

    @pytest.fixture(scope="class")
    def clonotype_sets(self):
        base = TCREmp.from_defaults("human", "TRB", n_prototypes=100)
        rows = base.prototypes.to_dicts()
        return {
            n: [
                _clonotype(rows[i % 100]["v_gene"], rows[i % 100]["j_gene"],
                           rows[i % 100]["junction_aa"])
                for i in range(n)
            ]
            for n in (10_000, 100_000, 500_000, 1_000_000)
        }

    @pytest.mark.parametrize("n_clono,n_proto", [
        (10_000,   1000),
        (100_000,  1000),
        (100_000,  3000),
        (500_000,  1000),
        (1_000_000, 1000),
    ])
    def test_single_process(self, models, clonotype_sets, n_clono, n_proto):
        X, elapsed, peak_mb = _measure(
            lambda: models[n_proto].embed(clonotype_sets[n_clono], n_jobs=1)
        )
        throughput = n_clono / elapsed
        print(
            f"\n[THROUGHPUT n_jobs=1] n_clono={n_clono:>7d} n_proto={n_proto:>4d} | "
            f"{elapsed:6.2f}s | {throughput:,.0f} clono/s | peak={peak_mb:.0f} MB"
        )
        assert X.shape == (n_clono, 3 * n_proto)
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# Multiprocessing scaling benchmark
# ---------------------------------------------------------------------------

class TestBenchmarkMultiprocessing:
    """Measure speedup for n_jobs=1,2,4,8 and find the useful parallelism threshold."""

    @pytest.fixture(scope="class")
    def model_1k(self):
        return TCREmp.from_defaults("human", "TRB", n_prototypes=1000)

    @pytest.fixture(scope="class")
    def clonotype_sets(self):
        base = TCREmp.from_defaults("human", "TRB", n_prototypes=100)
        rows = base.prototypes.to_dicts()
        return {
            n: [
                _clonotype(rows[i % 100]["v_gene"], rows[i % 100]["j_gene"],
                           rows[i % 100]["junction_aa"])
                for i in range(n)
            ]
            for n in (10_000, 100_000, 500_000)
        }

    @pytest.mark.parametrize("n_clono", [10_000, 100_000, 500_000])
    def test_scaling(self, model_1k, clonotype_sets, n_clono):
        """Compare n_jobs=1 vs multi-process.  After score_batch_max optimization,
        n_jobs=1 is typically fastest on macOS due to spawn overhead."""
        clonos = clonotype_sets[n_clono]
        results: dict[int, tuple[float, float]] = {}

        _, t1, _ = _measure(lambda: model_1k.embed(clonos, n_jobs=1))
        results[1] = (t1, 1.0)

        n_jobs_list = [j for j in (2, 4, 8) if j <= _N_CPUS]
        for nj in n_jobs_list:
            _, tnj, _ = _measure(lambda: model_1k.embed(clonos, n_jobs=nj))
            results[nj] = (tnj, t1 / tnj)

        _print_separator()
        print(f"\n[MP SCALING] n_clono={n_clono:>7d}  n_proto=1000")
        print(f"  {'n_jobs':>8}  {'time(s)':>8}  {'speedup':>8}")
        for nj, (t, sp) in sorted(results.items()):
            flag = "  <- baseline" if nj == 1 else (
                "  <- faster" if sp > 1.05 else "  <- overhead dominates"
            )
            print(f"  {nj:>8}  {t:>8.2f}  {sp:>8.2f}x{flag}")
        _print_separator()

        # Sanity: result shape must be correct for all n_jobs values
        for nj in [1] + n_jobs_list:
            X = model_1k.embed(clonos, n_jobs=nj)
            assert X.shape == (n_clono, 3000), f"Wrong shape for n_jobs={nj}"
