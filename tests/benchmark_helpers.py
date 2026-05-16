"""Shared helpers for benchmark tests."""

from __future__ import annotations

import gzip
import os
import time
from pathlib import Path
import pandas as pd
import pytest

from mir.common.alleles import allele_to_major
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire
from mir.comparative.overlap import pairwise_overlap

ASSETS = Path(__file__).parent / "assets"
GILG_FILE = ASSETS / "gilgfvftl_trb_cdr3.txt.gz"
BENCH_LOG = Path(__file__).parent / "benchmarks.log"


def benchmark_log_line(message: str) -> None:
    BENCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with BENCH_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {message}\n")


def synthetic_control_size(default: int = 1_000_000) -> int:
    raw = os.getenv("MIRPY_BENCH_SYNTHETIC_N")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1_000, value)


def _mk_clonotype(
    sid: str,
    aa: str,
    *,
    v_gene: str = "TRBV7-9*01",
    j_gene: str = "TRBJ2-1*01",
    duplicate_count: int = 1,
) -> Clonotype:
    return Clonotype(
        sequence_id=sid,
        locus="TRB",
        junction_aa=aa,
        v_gene=allele_to_major(v_gene),
        j_gene=allele_to_major(j_gene),
        duplicate_count=max(1, int(duplicate_count)),
        _validate=False,
    )


def load_gilg_target_repertoire(*, max_sequences: int | None = None) -> LocusRepertoire:
    if not GILG_FILE.exists():
        pytest.skip("GIL target asset missing: tests/assets/gilgfvftl_trb_cdr3.txt.gz")

    with gzip.open(GILG_FILE, "rt", encoding="utf-8") as fh:
        seqs = [line.strip() for line in fh if line.strip()]
    if max_sequences is not None:
        seqs = seqs[:max_sequences]
    clonotypes = [_mk_clonotype(f"g{i}", seq) for i, seq in enumerate(seqs)]
    return LocusRepertoire(clonotypes=clonotypes, locus="TRB")


def synthetic_control_repertoire(
    *,
    manager: ControlManager,
    species: str = "human",
    locus: str = "TRB",
    n: int = 1_000_000,
    require_cached: bool = True,
) -> LocusRepertoire:
    if require_cached:
        control_path = manager.synthetic_control_path(species, locus, n)
        if not control_path.exists():
            pytest.skip(
                f"Synthetic control cache not found at {control_path}. "
                "Build it first with mirpy-control-setup or set MIRPY_BENCH_REQUIRE_CACHED_CONTROL=0."
            )

    df = manager.ensure_and_load_control_df(
        "synthetic",
        species,
        locus,
        n=n,
        seed=42,
        chunk_size=100_000,
        progress=True,
    ).to_pandas()

    clonotypes = [
        _mk_clonotype(
            f"c{i}",
            str(rec.get("junction_aa", "")),
            v_gene=str(rec.get("v_gene", "TRBV7-9*01")),
            j_gene=str(rec.get("j_gene", "TRBJ2-1*01")),
            duplicate_count=int(rec.get("duplicate_count", 1) or 1),
        )
        for i, rec in enumerate(df.to_dict(orient="records"))
    ]
    return LocusRepertoire(clonotypes=clonotypes, locus=locus)


def real_control_limit(default: int = 2_000_000) -> int:
    """Max rows to load from the real control (env: MIRPY_BENCH_REAL_CONTROL_N)."""
    raw = os.getenv("MIRPY_BENCH_REAL_CONTROL_N")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1_000, value)


def real_control_repertoire(
    *,
    manager: ControlManager,
    species: str = "human",
    locus: str = "TRB",
    limit: int | None = None,
) -> LocusRepertoire:
    df = manager.ensure_and_load_control_df("real", species, locus).to_pandas()
    cap = limit if limit is not None else real_control_limit()
    if cap < len(df):
        df = df.sample(n=cap, random_state=42).reset_index(drop=True)
    clonotypes = [
        _mk_clonotype(
            f"r{i}",
            str(rec.get("junction_aa", "")),
            v_gene=str(rec.get("v_gene", "TRBV7-9*01")),
            j_gene=str(rec.get("j_gene", "TRBJ2-1*01")),
            duplicate_count=int(rec.get("duplicate_count", 1) or 1),
        )
        for i, rec in enumerate(df.to_dict(orient="records"))
    ]
    return LocusRepertoire(clonotypes=clonotypes, locus=locus)


def _resolve_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        try:
            import psutil
            n = psutil.cpu_count(logical=False)
            if n:
                return n
        except Exception:
            pass
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def many_vs_many_sample_overlap(
    repertoires: list[LocusRepertoire],
    sample_ids: list[str] | None = None,
    *,
    metric: str = "exact",
    threshold: int = 0,
    overlap_space: str | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Test-only helper: all pair overlaps with progress-friendly semantics."""
    n = len(repertoires)
    if n < 2:
        raise ValueError("Need at least 2 repertoires for many-vs-many overlap.")

    ids = sample_ids if sample_ids is not None else [f"s{i}" for i in range(n)]
    if len(ids) != n:
        raise ValueError("sample_ids length must match repertoires length.")

    jobs = _resolve_jobs(n_jobs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    rows = []
    for i, j in pairs:
        r = pairwise_overlap(
            repertoires[i],
            repertoires[j],
            metric=metric,
            threshold=threshold,
            overlap_space=overlap_space,
            n_jobs=jobs,
        )
        rows.append(
            {
                "sample_id_1": ids[i],
                "sample_id_2": ids[j],
                "n_jobs_effective": jobs,
                "qi_estimated_gb": float("nan"),
                **r.as_dict(),
            }
        )
    return pd.DataFrame(rows)


def many_vs_pool_sample_overlap(
    repertoires: list[LocusRepertoire],
    pool_repertoire: LocusRepertoire,
    *,
    sample_ids: list[str] | None = None,
    ages: list[int] | None = None,
    metric: str = "exact",
    threshold: int = 0,
    overlap_space: str | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Test-only helper: overlap of each sample against a pooled repertoire."""
    n = len(repertoires)
    ids = sample_ids if sample_ids is not None else [f"s{i}" for i in range(n)]
    if len(ids) != n:
        raise ValueError("sample_ids length must match repertoires length.")
    if ages is not None and len(ages) != n:
        raise ValueError("ages length must match repertoires length.")

    rows: list[dict] = []
    for idx, rep in enumerate(repertoires):
        r = pairwise_overlap(
            rep,
            pool_repertoire,
            metric=metric,
            threshold=threshold,
            overlap_space=overlap_space,
            n_jobs=n_jobs,
        )
        row = {"sample_id": ids[idx], **r.as_dict()}
        if ages is not None:
            row["age"] = int(ages[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_many_vs_many_runtime(
    repertoires: list[LocusRepertoire],
    *,
    metric: str = "exact",
    threshold: int = 0,
    overlap_space: str | None = None,
    n_jobs: int = -1,
    pilot_sample_count: int = 12,
) -> dict[str, float | int | str]:
    """Test-only pilot extrapolation helper for many-vs-many overlap runtime."""
    if len(repertoires) < 2:
        raise ValueError("Need at least 2 repertoires to estimate runtime.")

    pilot_n = max(2, min(len(repertoires), int(pilot_sample_count)))
    pilot_reps = repertoires[:pilot_n]
    pilot_ids = [f"pilot_{i}" for i in range(pilot_n)]

    pilot_pairs = pilot_n * (pilot_n - 1) // 2
    full_pairs = len(repertoires) * (len(repertoires) - 1) // 2

    t0 = time.perf_counter()
    _ = many_vs_many_sample_overlap(
        pilot_reps,
        sample_ids=pilot_ids,
        metric=metric,
        threshold=threshold,
        overlap_space=overlap_space,
        n_jobs=n_jobs,
    )
    pilot_seconds = time.perf_counter() - t0

    seconds_per_pair = pilot_seconds / max(1, pilot_pairs)
    return {
        "metric": metric,
        "threshold": int(threshold),
        "pilot_samples": pilot_n,
        "pilot_pairs": pilot_pairs,
        "pilot_seconds": pilot_seconds,
        "seconds_per_pair": seconds_per_pair,
        "full_pairs": full_pairs,
        "estimated_total_seconds": seconds_per_pair * full_pairs,
        "n_jobs_requested": int(n_jobs),
    }
