"""Benchmark tests for control generation/download workflows.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_control_benchmark.py -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from mir.common.control import ControlManager
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_max_seconds, skip_benchmarks


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except ValueError:
        return default
    return max(1.0, val)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return max(1, val)


def _is_full_benchmark_enabled() -> bool:
    return (
        os.getenv("RUN_FULL_BENCHMARK") == "1"
        or os.getenv("RUN_FULL_BENCHMARKS") == "1"
        or os.getenv("MIRPY_BENCH_INCLUDE_1M") == "1"
    )


def _diagnose_synth_record(rec_path: Path, df_rows: int) -> tuple[int, int]:
    return df_rows, rec_path.stat().st_size


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_synthetic_control_generation_small_matrix(capsys, tmp_path: Path) -> None:
    """Generate 1e4 and 1e5 synthetic controls in separate timed steps.

    This benchmark is intentionally split from 1e6 to keep routine runs under
    5-10 minutes while still validating synthetic build and cache-hit behavior.
    """
    control_dir = tmp_path / "controls_benchmark_synth_small"
    mgr = ControlManager(control_dir=control_dir)
    n_jobs = _env_int("MIRPY_BENCH_SYNTHETIC_N_JOBS", 4)
    sizes = [10_000, 100_000]
    rows: list[dict[str, float | int]] = []

    for n in sizes:
        t0 = time.perf_counter()
        rec_build = mgr.ensure_synthetic_control(
            "human",
            "TRB",
            n=n,
            overwrite=True,
            seed=42,
            chunk_size=100_000,
            progress=False,
            n_jobs=n_jobs,
        )
        build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = mgr.ensure_synthetic_control(
            "human",
            "TRB",
            n=n,
            overwrite=False,
            seed=42,
            chunk_size=100_000,
            progress=False,
            n_jobs=n_jobs,
        )
        cache_hit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        df = mgr.load_control_df("synthetic", "human", "TRB", n=n)
        load_s = time.perf_counter() - t0

        row_count, byte_size = _diagnose_synth_record(Path(rec_build.path), len(df))
        rows.append(
            {
                "n": n,
                "build_s": build_s,
                "cache_hit_s": cache_hit_s,
                "load_s": load_s,
                "rows": row_count,
                "bytes": byte_size,
            }
        )
        benchmark_log_line(
            "control_synthetic_small "
            f"n={n} build_s={build_s:.3f} cache_hit_s={cache_hit_s:.3f} load_s={load_s:.3f}"
        )

    with capsys.disabled():
        print("\n" + "=" * 76)
        print("Synthetic-control benchmark (1e4 + 1e5)")
        print(f"Control root folder: {mgr.control_dir}")
        for row in rows:
            print(
                "n={n:>8,d}: build={build_s:>8.2f}s cache_hit={cache_hit_s:>7.3f}s "
                "load={load_s:>7.3f}s rows={rows_count:>8,d} size={bytes_size:>12,d} bytes".format(
                    n=int(row["n"]),
                    build_s=float(row["build_s"]),
                    cache_hit_s=float(row["cache_hit_s"]),
                    load_s=float(row["load_s"]),
                    rows_count=int(row["rows"]),
                    bytes_size=int(row["bytes"]),
                )
            )
        print("=" * 76)

    per_test_cap = _env_float("MIRPY_BENCH_SYNTH_SMALL_CAP_S", 600.0)
    for row in rows:
        assert int(row["rows"]) == int(row["n"])
        assert float(row["cache_hit_s"]) <= float(row["build_s"])
        assert float(row["cache_hit_s"]) <= max(2.0, 2.0 * float(row["load_s"]))
        assert float(row["build_s"]) < per_test_cap, (
            f"synthetic n={int(row['n'])} build too slow: {float(row['build_s']):.2f}s >= {per_test_cap:.2f}s"
        )


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_real_control_build_and_cache_hit(capsys, tmp_path: Path) -> None:
    """Benchmark real-control build/load and cache-hit behavior in one focused test."""
    control_dir = tmp_path / "controls_benchmark_real"
    mgr = ControlManager(control_dir=control_dir)

    t0 = time.perf_counter()
    rec_human = mgr.ensure_real_control("human", "TRB", overwrite=True)
    build_human_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = mgr.ensure_real_control("human", "TRB", overwrite=False)
    cache_human_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_human = mgr.load_control_df("real", "human", "TRB")
    load_human_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rec_mouse = mgr.ensure_real_control("mouse", "TRA", overwrite=True)
    build_mouse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = mgr.ensure_real_control("mouse", "TRA", overwrite=False)
    cache_mouse_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_mouse = mgr.load_control_df("real", "mouse", "TRA")
    load_mouse_s = time.perf_counter() - t0

    benchmark_log_line(
        "control_real_build_cache "
        f"human_build_s={build_human_s:.3f} human_cache_s={cache_human_s:.3f} human_load_s={load_human_s:.3f} "
        f"mouse_build_s={build_mouse_s:.3f} mouse_cache_s={cache_mouse_s:.3f} mouse_load_s={load_mouse_s:.3f}"
    )

    with capsys.disabled():
        print("\n" + "=" * 76)
        print("Real-control benchmark (build + cache-hit)")
        print(f"Control root folder: {mgr.control_dir}")
        print(
            f"human/TRB build={build_human_s:.2f}s cache_hit={cache_human_s:.2f}s load={load_human_s:.2f}s "
            f"rows={len(df_human)} size={Path(rec_human.path).stat().st_size} bytes"
        )
        print(
            f"mouse/TRA build={build_mouse_s:.2f}s cache_hit={cache_mouse_s:.2f}s load={load_mouse_s:.2f}s "
            f"rows={len(df_mouse)} size={Path(rec_mouse.path).stat().st_size} bytes"
        )
        print("=" * 76)

    cap_s = _env_float("MIRPY_BENCH_REAL_CONTROL_CAP_S", 600.0)
    assert len(df_human) > 0
    assert len(df_mouse) > 0
    assert cache_human_s <= build_human_s
    assert cache_mouse_s <= build_mouse_s
    assert build_human_s < cap_s
    assert build_mouse_s < cap_s


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_real_control_repeated_cache_loads_no_extra_overhead(capsys, tmp_path: Path) -> None:
    """Benchmark repeated real-control cache loads and ensure low wrapper overhead."""
    control_dir = tmp_path / "controls_real_cache_benchmark"
    mgr = ControlManager(control_dir=control_dir)
    repeats = max(5, int(os.getenv("MIRPY_BENCH_REAL_CACHE_REPEATS", "25")))

    t0 = time.perf_counter()
    rec = mgr.ensure_real_control("human", "TRB", overwrite=True)
    build_s = time.perf_counter() - t0
    assert Path(rec.path).exists()

    load_times: list[float] = []
    ensure_times: list[float] = []
    rows_seen: set[int] = set()
    for _ in range(repeats):
        t0 = time.perf_counter()
        df = mgr.load_control_df("real", "human", "TRB")
        load_times.append(time.perf_counter() - t0)
        rows_seen.add(len(df))

        t0 = time.perf_counter()
        _ = mgr.ensure_real_control("human", "TRB", overwrite=False)
        ensure_times.append(time.perf_counter() - t0)

    load_med = float(np.median(np.array(load_times, dtype=float)))
    ensure_med = float(np.median(np.array(ensure_times, dtype=float)))
    load_mean = float(np.mean(np.array(load_times, dtype=float)))
    ensure_mean = float(np.mean(np.array(ensure_times, dtype=float)))
    overhead_ratio = ensure_med / load_med if load_med > 0 else 0.0

    benchmark_log_line(
        "control_real_cache_repeat "
        f"build_s={build_s:.3f} repeats={repeats} "
        f"load_med_s={load_med:.4f} ensure_med_s={ensure_med:.4f} ratio={overhead_ratio:.3f}"
    )

    with capsys.disabled():
        print("\n" + "=" * 76)
        print("Real-control cache-repeat benchmark")
        print(f"Control root folder: {mgr.control_dir}")
        print(f"Build time (overwrite=True):           {build_s:.2f}s")
        print(f"Repeated load_control_df x{repeats}:   mean={load_mean:.4f}s median={load_med:.4f}s")
        print(f"Repeated ensure_real_control x{repeats}: mean={ensure_mean:.4f}s median={ensure_med:.4f}s")
        print(f"Wrapper overhead ratio (ensure/load):  {overhead_ratio:.3f}x")
        print("Expected behavior: cache-hit ensure should be close to pure load time.")
        print("=" * 76)

    assert len(rows_seen) == 1 and next(iter(rows_seen)) > 0
    assert ensure_med <= build_s, "cache-hit ensure should be much faster than a full rebuild"
    assert overhead_ratio <= 1.5, f"unexpected cache wrapper overhead: {overhead_ratio:.3f}x"
    assert build_s < _env_float("MIRPY_BENCH_REAL_CACHE_BUILD_CAP_S", 600.0)


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.very_slow_benchmark
def test_synthetic_control_1e6_cache_hit_and_optional_cold_build(capsys, tmp_path: Path) -> None:
    """Benchmark 1e6 synthetic control in cache-first mode.

    Optimization rationale:
    - default path validates that 1e6 cache-hit and load are efficient,
    - expensive cold build is opt-in via MIRPY_BENCH_1M_COLD_BUILD=1.
    """
    if not _is_full_benchmark_enabled():
        pytest.skip("Enable RUN_FULL_BENCHMARK=1 or MIRPY_BENCH_INCLUDE_1M=1 for 1e6 synthetic-control benchmark")

    cold_build = os.getenv("MIRPY_BENCH_1M_COLD_BUILD") == "1"
    shared_dir = os.getenv("MIRPY_BENCH_1M_CONTROL_DIR")
    if shared_dir:
        control_dir = Path(shared_dir)
    else:
        # Use default cache dir to allow reuse across benchmark runs.
        control_dir = tmp_path / "controls_synth_1m_benchmark"

    mgr = ControlManager(control_dir=control_dir)
    n_jobs = _env_int("MIRPY_BENCH_SYNTHETIC_N_JOBS", 4)
    n = 1_000_000
    control_path = mgr.synthetic_control_path("human", "TRB", n)

    if not control_path.exists() and not cold_build:
        pytest.skip(
            "1e6 synthetic cache missing. Set MIRPY_BENCH_1M_COLD_BUILD=1 to build once, "
            "or point MIRPY_BENCH_1M_CONTROL_DIR to a prebuilt cache."
        )

    build_s = 0.0
    if cold_build:
        t0 = time.perf_counter()
        _ = mgr.ensure_synthetic_control(
            "human",
            "TRB",
            n=n,
            overwrite=not control_path.exists(),
            seed=42,
            chunk_size=100_000,
            progress=False,
            n_jobs=n_jobs,
        )
        build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rec = mgr.ensure_synthetic_control(
        "human",
        "TRB",
        n=n,
        overwrite=False,
        seed=42,
        chunk_size=100_000,
        progress=False,
        n_jobs=n_jobs,
    )
    cache_hit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = mgr.load_control_df("synthetic", "human", "TRB", n=n)
    load_s = time.perf_counter() - t0

    rows_count, byte_size = _diagnose_synth_record(Path(rec.path), len(df))

    benchmark_log_line(
        "control_synthetic_1m "
        f"cold_build={int(cold_build)} build_s={build_s:.3f} cache_hit_s={cache_hit_s:.3f} load_s={load_s:.3f} rows={rows_count}"
    )

    with capsys.disabled():
        print("\n" + "=" * 76)
        print("Synthetic-control benchmark (1e6 cache-first)")
        print(f"Control root folder: {mgr.control_dir}")
        print(
            f"n=1,000,000 build={build_s:.2f}s cache_hit={cache_hit_s:.3f}s "
            f"load={load_s:.3f}s rows={rows_count:,d} size={byte_size:,d} bytes cold_build={cold_build}"
        )
        print("=" * 76)

    assert rows_count == n
    assert cache_hit_s <= max(2.0, 2.0 * load_s)

    # Keep routine benchmark runs within ~5-10 minutes for this subtest.
    cache_cap_s = _env_float("MIRPY_BENCH_1M_CACHE_CAP_S", 600.0)
    assert cache_hit_s < cache_cap_s, (
        f"1e6 synthetic cache-hit too slow: {cache_hit_s:.2f}s >= {cache_cap_s:.2f}s"
    )

    if cold_build:
        # Cold build is optional and typically much slower; keep it explicit.
        cold_cap_s = benchmark_max_seconds(default=3600.0)
        assert build_s < cold_cap_s, (
            f"1e6 synthetic cold-build too slow: {build_s:.2f}s >= {cold_cap_s:.2f}s"
        )
