"""Benchmarks for exact and 1-mismatch Pgen calculation.

Run with:
    RUN_BENCHMARK=1 pytest -s tests/test_pgen_benchmark.py
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from mir.basic.pgen import OlgaModel
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_max_seconds, benchmark_repertoire_workers, skip_benchmarks


def _unique_generated_sequences(model: OlgaModel, n: int, *, seed: int = 42) -> list[str]:
    seqs: list[str] = []
    seen: set[str] = set()
    batch_seed = seed
    while len(seqs) < n:
        for seq in model.generate_sequences(max(2 * n, 200), seed=batch_seed):
            if seq in seen:
                continue
            seen.add(seq)
            seqs.append(seq)
            if len(seqs) >= n:
                break
        batch_seed += 1
    return seqs[:n]


def _profile_bulk_pgen(seqs: list[str], *, max_mismatches: int, workers: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    baseline: list[float] | None = None
    for n_jobs in workers:
        model = OlgaModel(locus="TRB", species="human", seed=42)
        t0 = time.perf_counter()
        values = model.compute_pgen_junction_aa_bulk(
            seqs,
            max_mismatches=max_mismatches,
            n_jobs=n_jobs,
        )
        elapsed = time.perf_counter() - t0
        if baseline is None:
            baseline = values
        else:
            assert values == baseline
        rows.append(
            {
                "max_mismatches": max_mismatches,
                "workers": n_jobs,
                "n_sequences": len(seqs),
                "elapsed_s": elapsed,
                "seqs_per_s": len(seqs) / elapsed if elapsed > 0 else float("inf"),
            }
        )
        benchmark_log_line(
            "PGEN_BULK "
            f"max_mismatches={max_mismatches} workers={n_jobs} n_sequences={len(seqs)} "
            f"elapsed_s={elapsed:.3f} seqs_per_s={len(seqs) / elapsed if elapsed > 0 else float('inf'):.1f}"
        )
    return pd.DataFrame(rows)


@skip_benchmarks
def test_pgen_bulk_parallel_exact_and_1mm(capsys) -> None:
    workers = benchmark_repertoire_workers(default="1,8")
    model = OlgaModel(locus="TRB", species="human", seed=42)
    seqs_exact = _unique_generated_sequences(model, 1000, seed=42)
    seqs_1mm = seqs_exact[:200]

    df_exact = _profile_bulk_pgen(seqs_exact, max_mismatches=0, workers=workers)
    df_1mm = _profile_bulk_pgen(seqs_1mm, max_mismatches=1, workers=workers)
    df = pd.concat([df_exact, df_1mm], ignore_index=True)

    with capsys.disabled():
        print("\n" + "=" * 92)
        print("Pgen benchmark: exact vs 1-mismatch")
        print(df.to_string(index=False))
        print("=" * 92)

    assert not df.empty
    cap_s = benchmark_max_seconds(default=900.0)
    assert (df["elapsed_s"] < cap_s).all()
