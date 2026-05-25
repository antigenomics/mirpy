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
from mir.basic.pgen import OlgaModel
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats_by_locus
from mir.utils.stats import bh_fdr
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_max_seconds, benchmark_repertoire_workers, benchmark_track_memory, skip_benchmarks

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
    table = result.table.to_pandas()
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
@pytest.mark.benchmark
@pytest.mark.very_slow_benchmark
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
    assert float(df["alice_total_s"].max()) < benchmark_max_seconds(default=900.0)
    assert float(df["alice_total_s"].sum()) < benchmark_max_seconds(default=1800.0)


@skip_benchmarks
@pytest.mark.benchmark
def test_alice_pgen_10k_single_vs_parallel(capsys) -> None:
    yf_file = YF_DIR / os.getenv("MIRPY_ALICE_PGEN_BENCH_FILE", "Q1_d0.tsv.gz")
    if not yf_file.exists():
        pytest.skip(f"Missing YF benchmark file: {yf_file.name}")

    n_sequences = _env_int_list("MIRPY_ALICE_PGEN_BENCH_N", "10000")[0]
    workers = benchmark_repertoire_workers(default="8")
    n_jobs_parallel = max(2, workers[-1])
    pgen_mode = os.getenv("MIRPY_ALICE_PGEN_BENCH_MODE", "exact").strip() or "exact"

    rep = _load_yf_repertoire(yf_file)
    unique_aas = list(dict.fromkeys(c.junction_aa for c in rep.clonotypes if c.junction_aa))
    if len(unique_aas) < n_sequences:
        pytest.skip(
            f"Need at least {n_sequences} unique CDR3aa for pgen benchmark; found {len(unique_aas)} in {yf_file.name}"
        )

    bench_aas = unique_aas[:n_sequences]
    bench_rep = LocusRepertoire(
        clonotypes=[
            c
            for c in rep.clonotypes
            if c.junction_aa in set(bench_aas)
        ],
        locus="TRB",
        repertoire_id=f"{rep.repertoire_id}_pgen10k",
    )

    # Warm model + pool initialization outside timed block.
    # Step 1: load the OlgaModel (serial path, 1 sequence).
    alice_mod._compute_pgen_raw_by_junction_aa(
        bench_rep.clonotypes[:1],
        locus="TRB",
        species="human",
        random_seed=None,
        pgen_mode=pgen_mode,
        n_jobs=1,
    )
    # Step 2: spawn worker pool and load the OLGA model in each worker.
    # The persistent pool is the design: pay spawn+model-load once, amortise
    # across all ALICE samples.  The timed passes below measure sustained
    # throughput, not cold-start latency.
    seen_warmup: set[str] = set()
    warmup_clones: list = []
    for c in bench_rep.clonotypes:
        if c.junction_aa not in seen_warmup:
            seen_warmup.add(c.junction_aa)
            warmup_clones.append(c)
        if len(warmup_clones) >= 300:
            break
    alice_mod._compute_pgen_raw_by_junction_aa(
        warmup_clones,
        locus="TRB",
        species="human",
        random_seed=None,
        pgen_mode=pgen_mode,
        n_jobs=n_jobs_parallel,
    )

    t0 = time.perf_counter()
    p1 = alice_mod._compute_pgen_raw_by_junction_aa(
        bench_rep.clonotypes,
        locus="TRB",
        species="human",
        random_seed=None,
        pgen_mode=pgen_mode,
        n_jobs=1,
    )
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    pn = alice_mod._compute_pgen_raw_by_junction_aa(
        bench_rep.clonotypes,
        locus="TRB",
        species="human",
        random_seed=None,
        pgen_mode=pgen_mode,
        n_jobs=n_jobs_parallel,
    )
    tn = time.perf_counter() - t0

    speedup = t1 / tn if tn > 0 else float("inf")
    # 4x is the conservative floor after pool warm-up on macOS with spawn mode.
    # True sustained speedup on a healthy 8-core M3 is typically 6-8x; the gap
    # is IPC serialisation overhead from pool.map chunk dispatch.
    min_speedup = float(os.getenv("MIRPY_ALICE_PGEN_BENCH_MIN_SPEEDUP", "4.0"))
    benchmark_log_line(
        "ALICE_PGEN_10K "
        f"file={yf_file.name} n_sequences={n_sequences} pgen_mode={pgen_mode} "
        f"single_s={t1:.3f} parallel_s={tn:.3f} workers={n_jobs_parallel} speedup={speedup:.2f}"
    )

    with capsys.disabled():
        print("\n" + "=" * 92)
        print("ALICE pgen benchmark (10k sequences)")
        print(f"source_file={yf_file.name} n_sequences={n_sequences} pgen_mode={pgen_mode}")
        print(f"single_thread_s={t1:.3f}")
        print(f"parallel_s={tn:.3f} workers={n_jobs_parallel}")
        print(f"speedup={speedup:.2f}x")
        print("=" * 92)

    assert len(p1) == n_sequences
    assert len(pn) == n_sequences
    assert set(p1.keys()) == set(pn.keys())
    assert speedup >= min_speedup, (
        f"Expected ALICE pgen speedup >= {min_speedup:.2f}x for 10k benchmark, got {speedup:.2f}x "
        f"(single={t1:.3f}s parallel={tn:.3f}s workers={n_jobs_parallel})"
    )


@skip_benchmarks
@pytest.mark.benchmark
def test_alice_1mm_pgen_rate_after_neighbor_filter(capsys) -> None:
    """Benchmark 1mm pgen throughput on sequences passing the min_neighbors=2 filter.

    This test isolates the actual ALICE bottleneck for large TRB repertoires:
    pgen is only computed for sequences with n_neighbors >= min_neighbors (default 2),
    which represent the only candidates for ALICE cluster membership.

    Run with:
        RUN_BENCHMARK=1 pytest -s tests/test_alice_benchmark.py::test_alice_1mm_pgen_rate_after_neighbor_filter
        MIRPY_ALICE_PGEN_BENCH_FILE=P2_d15.tsv.gz RUN_BENCHMARK=1 pytest -s ...  # larger file
    """
    yf_file = YF_DIR / os.getenv("MIRPY_ALICE_PGEN_BENCH_FILE", "Q1_d0.tsv.gz")
    if not yf_file.exists():
        pytest.skip(f"Missing YF benchmark file: {yf_file.name}")

    min_neighbors = int(os.getenv("MIRPY_ALICE_BENCH_MIN_NEIGHBORS", "2"))
    workers = benchmark_repertoire_workers(default="8")
    n_jobs = max(1, workers[-1])
    bench_n = int(os.getenv("MIRPY_ALICE_PGEN_BENCH_N", "300"))

    rep = _load_yf_repertoire(yf_file)
    total_n = rep.clonotype_count

    # Phase 1: trie neighbor density
    t0 = time.perf_counter()
    stats = compute_neighborhood_stats_by_locus(
        rep,
        background=None,
        metric="hamming",
        threshold=1,
        match_v_gene=True,
        match_j_gene=True,
        add_background_pseudocount=False,
        n_jobs=n_jobs,
    )
    trie_s = time.perf_counter() - t0
    locus_stats = stats.get("TRB", {})

    n_ge1 = sum(1 for v in locus_stats.values() if v.get("neighbor_count", 0) >= 1)
    n_ge2 = sum(1 for v in locus_stats.values() if v.get("neighbor_count", 0) >= min_neighbors)
    pct_ge1 = 100.0 * n_ge1 / max(1, total_n)
    pct_ge2 = 100.0 * n_ge2 / max(1, total_n)

    # Phase 2: collect filtered CDR3s for pgen benchmark
    filtered_aas = list(dict.fromkeys(
        c.junction_aa
        for c in rep.clonotypes
        if locus_stats.get(c.sequence_id, {}).get("neighbor_count", 0) >= min_neighbors
    ))
    if not filtered_aas:
        pytest.skip(f"No sequences with n_neighbors >= {min_neighbors} in {yf_file.name}")

    bench_seqs = filtered_aas[:min(bench_n, len(filtered_aas))]

    # Phase 3: warm model + pool outside timed window
    model = OlgaModel(locus="TRB", species="human")
    model.compute_pgen_junction_aa_bulk(bench_seqs[:1], max_mismatches=1, n_jobs=1)
    model.compute_pgen_junction_aa_bulk(bench_seqs[:min(50, len(bench_seqs))], max_mismatches=1, n_jobs=n_jobs)

    # Phase 4: timed 1mm pgen on filtered sequences
    t0 = time.perf_counter()
    model.compute_pgen_junction_aa_bulk(bench_seqs, max_mismatches=1, n_jobs=n_jobs)
    pgen_s = time.perf_counter() - t0
    rate = len(bench_seqs) / max(pgen_s, 1e-9)

    model.close()

    benchmark_log_line(
        "ALICE_1MM_PGEN_FILTER "
        f"file={yf_file.name} total={total_n} "
        f"n_ge1={n_ge1} pct_ge1={pct_ge1:.1f} "
        f"n_ge{min_neighbors}={n_ge2} pct_ge{min_neighbors}={pct_ge2:.1f} "
        f"trie_s={trie_s:.3f} bench_n={len(bench_seqs)} "
        f"pgen_s={pgen_s:.3f} rate_junction_per_s={rate:.1f} workers={n_jobs}"
    )

    with capsys.disabled():
        print("\n" + "=" * 80)
        print("ALICE 1mm pgen rate after min_neighbors filter")
        print(f"  file          : {yf_file.name}  ({total_n:,} functional TRB)")
        print(f"  trie search   : {trie_s:.1f}s")
        print(f"  n_neighbors>=1: {n_ge1:,} ({pct_ge1:.1f}%)")
        print(f"  n_neighbors>={min_neighbors}: {n_ge2:,} ({pct_ge2:.1f}%)")
        print(f"  unique filtered CDR3s: {len(filtered_aas):,}")
        print(f"  bench sample  : {len(bench_seqs)} CDR3s, n_jobs={n_jobs}")
        print(f"  1mm pgen time : {pgen_s:.2f}s  ({rate:.1f} CDR3s/s)")
        if filtered_aas:
            eta_s = len(filtered_aas) / max(rate, 0.001)
            print(f"  ETA this file : {eta_s:.0f}s ({eta_s / 60:.1f} min)")
        print("=" * 80)

    assert len(filtered_aas) > 0
    assert rate > 0.0
