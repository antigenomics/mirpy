"""Notebook-style ALICE benchmark on YF repertoires.

Run with:
    RUN_BENCHMARK=1 pytest -s tests/test_alice_benchmark.py

Default behavior isolates the ALICE Pgen path by using ``match_mode='none'``.
Set ``MIRPY_ALICE_BENCH_MATCH_MODE=vj`` to reproduce the notebook's exact
conditioning mode.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

import mir.biomarkers.alice as alice_mod
from mir.biomarkers.alice import compute_alice
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire
from mir.utils.stats import bh_fdr
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_repertoire_workers, benchmark_track_memory, skip_benchmarks

REPO_ROOT = Path(__file__).resolve().parents[1]
YF_DIR = REPO_ROOT / "airr_benchmark" / "alice" / "yf"
YF_FILES = ("Q1_d0.tsv.gz", "Q1_d15.tsv.gz")


def _env_int_list(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


def _env_str_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    values = [token.strip() for token in raw.split(",") if token.strip()]
    return values or list(default)


def _load_yf_repertoire(path: Path) -> LocusRepertoire:
    parser = ClonotypeTableParser()
    clonotypes = parser.parse(str(path))
    rep = LocusRepertoire(
        clonotypes=[c for c in clonotypes if c.locus == "TRB"],
        locus="TRB",
        repertoire_id=path.stem,
    )
    rep = filter_functional(rep)
    return LocusRepertoire(clonotypes=list(rep.clonotypes), locus="TRB", repertoire_id=path.stem)


def _top_clonotype_subset(rep: LocusRepertoire, limit: int | None) -> LocusRepertoire:
    if limit is None or limit >= rep.clonotype_count:
        return LocusRepertoire(clonotypes=list(rep.clonotypes), locus=rep.locus, repertoire_id=rep.repertoire_id)
    clones = sorted(rep.clonotypes, key=lambda c: int(c.duplicate_count or 0), reverse=True)[:limit]
    return LocusRepertoire(clonotypes=list(clones), locus=rep.locus, repertoire_id=rep.repertoire_id)


def _profile_alice_run(
    rep: LocusRepertoire,
    *,
    n_jobs: int,
    match_mode: str,
    pgen_mode: str,
    track_memory: bool,
) -> dict[str, float | int | str]:
    timings: dict[str, float] = defaultdict(float)

    orig_stats = alice_mod.compute_neighborhood_stats_by_locus
    orig_pgen = alice_mod._compute_pgen_raw_by_junction_aa
    orig_metrics = alice_mod._compute_alice_metrics_batch

    def _wrap(label: str, fn):
        def _inner(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            timings[label] += time.perf_counter() - t0
            return out

        return _inner

    alice_mod.compute_neighborhood_stats_by_locus = _wrap("neighborhood_s", orig_stats)
    alice_mod._compute_pgen_raw_by_junction_aa = _wrap("pgen_s", orig_pgen)
    alice_mod._compute_alice_metrics_batch = _wrap("metrics_s", orig_metrics)

    peak_bytes = None
    if track_memory:
        tracemalloc.start()

    t0 = time.perf_counter()
    try:
        result = compute_alice(
            rep,
            species="human",
            threshold=1,
            match_mode=match_mode,
            pgen_mode=pgen_mode,
            as_table=True,
            n_jobs=n_jobs,
            random_seed=42,
        )
    finally:
        alice_mod.compute_neighborhood_stats_by_locus = orig_stats
        alice_mod._compute_pgen_raw_by_junction_aa = orig_pgen
        alice_mod._compute_alice_metrics_batch = orig_metrics

    total_s = time.perf_counter() - t0
    if track_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    t_bh = time.perf_counter()
    table = result.table.copy()
    table["q_value"] = bh_fdr(table["p_value"].to_numpy())
    hits = int((table["q_value"] < 0.05).sum())
    bh_s = time.perf_counter() - t_bh

    unique_aa = len({c.junction_aa for c in rep.clonotypes})
    return {
        "sample_id": rep.repertoire_id,
        "clonotypes": rep.clonotype_count,
        "unique_aa": unique_aa,
        "workers": n_jobs,
        "match_mode": match_mode,
        "pgen_mode": pgen_mode,
        "alice_total_s": total_s,
        "neighborhood_s": timings.get("neighborhood_s", 0.0),
        "pgen_s": timings.get("pgen_s", 0.0),
        "metrics_s": timings.get("metrics_s", 0.0),
        "bh_s": bh_s,
        "rows": len(table),
        "hits": hits,
        "peak_mem_mib": float(peak_bytes / (1024 ** 2)) if peak_bytes is not None else float("nan"),
        "legacy_extra_index_entries": int(rep.clonotype_count * 2),
    }


@skip_benchmarks
@pytest.mark.slow_benchmark
def test_alice_yf_notebook_cell6_scaling(capsys) -> None:
    yf_files = tuple(_env_str_list("MIRPY_ALICE_BENCH_FILES", YF_FILES))
    missing = [name for name in yf_files if not (YF_DIR / name).exists()]
    if missing:
        pytest.skip(f"Missing YF benchmark files: {missing}")

    workers = benchmark_repertoire_workers(default="1,8")
    subsamples = _env_int_list("MIRPY_ALICE_BENCH_SUBSAMPLES", "5000,10000,25000")
    match_mode = os.getenv("MIRPY_ALICE_BENCH_MATCH_MODE", "none").strip() or "none"
    include_full = os.getenv("MIRPY_ALICE_BENCH_INCLUDE_FULL", "0") in {"1", "true", "TRUE", "yes", "YES"}
    track_memory = benchmark_track_memory(default=False)

    loaded = {name: _load_yf_repertoire(YF_DIR / name) for name in yf_files}
    rows: list[dict[str, float | int | str]] = []

    for name, full_rep in loaded.items():
        limits = [limit for limit in subsamples if limit < full_rep.clonotype_count]
        if include_full or not limits:
            limits.append(full_rep.clonotype_count)
        for limit in limits:
            rep = _top_clonotype_subset(full_rep, limit)
            for n_jobs in workers:
                prof = _profile_alice_run(
                    rep,
                    n_jobs=n_jobs,
                    match_mode=match_mode,
                    pgen_mode="exact",
                    track_memory=track_memory,
                )
                rows.append(prof)
                print(
                    "ALICE_BENCH "
                    f"sample={prof['sample_id']} clonotypes={prof['clonotypes']} workers={prof['workers']} "
                    f"total={prof['alice_total_s']:.3f}s neighborhood={prof['neighborhood_s']:.3f}s "
                    f"pgen={prof['pgen_s']:.3f}s metrics={prof['metrics_s']:.3f}s bh={prof['bh_s']:.3f}s"
                )
                benchmark_log_line(
                    "ALICE_NOTEBOOK_YF "
                    f"sample={prof['sample_id']} clonotypes={prof['clonotypes']} unique_aa={prof['unique_aa']} "
                    f"workers={prof['workers']} match_mode={prof['match_mode']} "
                    f"alice_total_s={prof['alice_total_s']:.3f} neighborhood_s={prof['neighborhood_s']:.3f} "
                    f"pgen_s={prof['pgen_s']:.3f} metrics_s={prof['metrics_s']:.3f} bh_s={prof['bh_s']:.3f} "
                    f"rows={prof['rows']} hits={prof['hits']} peak_mem_mib={prof['peak_mem_mib']:.2f} "
                    f"legacy_extra_index_entries={prof['legacy_extra_index_entries']}"
                )

    df = pd.DataFrame(rows).sort_values(["sample_id", "clonotypes", "workers"]).reset_index(drop=True)
    diagnosis = (
        "Historical stall reason: ALICE already carries O(n) neighborhood statistics. "
        "When an additional per-sequence Pgen dict and a second per-sequence metrics dict are materialized on "
        "400k+ clonotype repertoires, memory pressure spikes and macOS can spend most wall time swapping."
    )

    with capsys.disabled():
        print("\n" + "=" * 92)
        print("ALICE notebook-style YF benchmark")
        print(f"YF fixtures: {', '.join(yf_files)}")
        print(f"match_mode={match_mode!r} (set MIRPY_ALICE_BENCH_MATCH_MODE=vj for notebook-exact conditioning)")
        print(diagnosis)
        print(df.to_string(index=False))
        print("=" * 92)

    assert not df.empty
    assert (df["rows"] == df["clonotypes"]).all()
