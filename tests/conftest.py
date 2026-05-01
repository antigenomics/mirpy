import os

import pytest


RUN_BENCHMARKS = (
    os.getenv("RUN_BENCHMARK") == "1"
    or os.getenv("RUN_BENCHMARKS") == "1"
)
RUN_INTEGRATION = os.getenv("RUN_INTEGRATION") == "1"

skip_benchmarks = pytest.mark.skipif(
    not RUN_BENCHMARKS,
    reason="set RUN_BENCHMARK=1 to run benchmark tests",
)

skip_integration = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set RUN_INTEGRATION=1 to run integration tests",
)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value


def benchmark_scale(default: float = 0.5) -> float:
    """Global benchmark workload multiplier.

    Set MIRPY_BENCHMARK_SCALE to reduce or increase benchmark loop counts.
    """
    scale = _env_float("MIRPY_BENCHMARK_SCALE", default)
    return max(0.05, scale)


def benchmark_max_seconds(default: float = 120.0) -> float:
    """Maximum allowed wall-clock time for one benchmark test."""
    return max(1.0, _env_float("MIRPY_BENCHMARK_MAX_SECONDS", default))


def benchmark_repertoire_workers(default: str = "1,4") -> list[int]:
    """Worker counts for repertoire benchmark matrix.

    Set MIRPY_BENCH_WORKERS to a comma-separated list, e.g. "1,4,8".
    """
    raw = os.getenv("MIRPY_BENCH_WORKERS", default)
    workers: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            w = int(tok)
        except ValueError:
            continue
        if w > 0:
            workers.append(w)
    if not workers:
        workers = [1, 4]
    # stable unique
    seen = set()
    uniq: list[int] = []
    for w in workers:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def benchmark_repetition_count(base_n: int, *, minimum: int = 100) -> int:
    """Scaled integer repetition count for micro-benchmarks."""
    n = int(round(base_n * benchmark_scale()))
    return max(minimum, n)


def benchmark_track_memory(default: bool = False) -> bool:
    """Whether to enable tracemalloc in benchmark tests."""
    raw = os.getenv("MIRPY_BENCH_TRACK_MEMORY")
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "TRUE", "yes", "YES"}
