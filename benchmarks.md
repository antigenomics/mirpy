# mirpy Benchmark Results

Reference run recorded **2025-05-10** on an Apple M3 Max (arm64), 48 GB RAM, 16 cores, Python 3.12.12, macOS 15.

All benchmarks run from the repo root with the activated venv:

```fish
env RUN_BENCHMARK=1 python -m pytest tests/test_*_benchmark.py -s -v
```

---

## Notebook execution — full suite

All 11 notebooks run cell-by-cell via `tmp/run_notebook.py` (per-cell timeout 900 s, CWD `notebooks/`, headless matplotlib via `MPLBACKEND=Agg`).
All data loaded from HuggingFace via `mir/utils/notebook_assets.py` utilities.

| Notebook | Cells | Wall time | Peak RSS | Notes |
|---|---|---|---|---|
| parsing_example.ipynb | 6 | 5.8 s | ~80 MB | AIRR parse + basic stats |
| gene_similarity.ipynb | 5 | 2.6 s | ~80 MB | V-gene cosine similarity |
| sample_repertoire_overview.ipynb | 7 | 3.3 s | ~80 MB | Repertoire QC plots |
| token_graph.ipynb | 8 | 12.0 s | ~80 MB | GLIPH token graph |
| vdjdb_cdr3_graph.ipynb | 9 | 16.5 s | ~80 MB | VDJdb CDR3 Hamming graph |
| edit_distance_graph.ipynb | 6 | 5.3 s | ~80 MB | Levenshtein edit-distance graph |
| gene_usage_correction.ipynb | 8 | 205 s | ~80 MB | OLGA V/J usage correction |
| tcrnet_analysis.ipynb | 10 | 93.8 s | ~85 MB | TCRnet motif enrichment on YF |
| gliph_analysis.ipynb | 11 | 261.1 s | ~90 MB | GLIPH2 on YF + B35 studies |
| alice_analysis.ipynb | 12 | 652.5 s | 76 MB | ALICE on YF/AS/MLR (3 cohorts) |
| vdjbet_yf.ipynb | 14 | 309.5 s | 82 MB | VDJbet YF enrichment + real control |

**Total: 11/11 notebooks OK, 0 errors, 0 timeouts. Combined wall time: ~27 min.**

### alice_analysis.ipynb — per-cell breakdown

`match_mode=none`, `pgen_mode=exact`, 10 k YF downsample, 250 hits/donor AS graph cap.

| Cell | Time | Notes |
|---|---|---|
| 1 setup | 2.1 s | env + dataset download |
| 2 parse YF | 24.4 s | 12 samples, ~9 k clonotypes each after 10 k downsample |
| 3 ALICE YF | 166.6 s | 12 samples × 8 workers; 100% hit rate at 9–10 k scale |
| 4 bar plot | 0.1 s | |
| 5 VDJdb LLW | 37.3 s | VDJdb 2025-12-29 download + filter |
| 6 parse AS | 0.3 s | 4 donors, 9–38 k clonotypes |
| 7 ALICE AS | 130.9 s | 4 donors × 8 workers; 104 k total hits |
| 8 build graph | 0.0 s | Capped at 250 hits/donor → 1 000 nodes, 1 563 edges |
| 9 render graph | 0.9 s | FR layout niter=500 |
| 10 parse MLR | 61.6 s | 24 Adaptive files → top-5 000 by duplicate_count each |
| 11 ALICE MLR | 227.6 s | 24 samples × 8 workers; 100% hit rate at 5 k scale |
| 12 fresh/prolif boxplot | 0.0 s | Mann-Whitney p=1.0 (all groups 100% hits) |

> **Note on 100% hit rate**: ALICE with `match_mode=none` at 5–10 k sample sizes yields ~100% FDR hits because the neighborhood lambda is very small when every CDR3 has ≥ 1 Hamming-1 neighbor. This is expected algorithm behavior at these sample sizes and is consistent with the original notebook design.

### vdjbet_yf.ipynb — per-cell breakdown

42 YF TRB repertoires; 100 k real-control pool; OLGA usage cache 1 M sequences.

| Cell | Time | Notes |
|---|---|---|
| 1 setup | 1.8 s | |
| 2 VDJdb LLW | 44.3 s | VDJdb 2025-12-29 + 409 LLWNGPMAV clonotypes |
| 3 parse 42 YF samples | 95.5 s | 29.9 M clonotypes, 56.7 M duplicates total |
| 4 OLGA usage | 0.8 s | Cache hit (1 M entries) |
| 5 real control pool | 119.4 s | 99 991 sequences, 47 bins, 117.9 s build |
| 6 score 42 samples | 22.7 s | 42 samples, ~0.5 s each |
| 7–10 plots + summary | < 1 s | |
| 11 reload module | 0.0 s | |
| 12 synthetic comparison | 23.0 s | Synthetic overlap 8.7 vs real 25.4; scale factor 2.91 |
| 13–14 diagnostics | < 0.1 s | |

Unless stated otherwise, worker counts use `MIRPY_BENCH_WORKERS=8` (production default).
Real-control tests cap at 2 M rows via `MIRPY_BENCH_REAL_CONTROL_N=2000000`.

---

## Quick reference — total wall-clock by test file

| Test file | Tests | Wall time | Notes |
|---|---|---|---|
| test_mirseq_benchmark.py | 6/6 | ~2 s | C-extension micro-benchmarks |
| test_pgen_benchmark.py | 1/1 | ~47 s | OLGA exact + 1-mm pgen |
| test_pool_benchmark.py | 1/1 | ~2 s | Clonotype pooling |
| test_repertoire_benchmark.py | 1/1 | ~3 s | Parallel I/O |
| test_neighborhood_enrichment_benchmark.py | 3/3 | ~27 s | Hamming neighbour search |
| test_neighborhood_enrichment_scaling_benchmark.py | 1/1 | ~27 s | up to 1e5 (default) |
| test_gliph_benchmark.py | 1/1 | ~304 s | GLIPH2 on two real studies |
| test_gliph_control_benchmark.py | 2/2 | ~200 s | Tokenisation + rare-token coverage |
| test_tcrnet_benchmark.py | 3/3 | ~63 s | TCRnet motif enrichment |
| test_control_benchmark.py | 3/3 + 1 skip | ~760 s | Synthetic + real control build |
| test_bag_of_kmers_benchmark.py | 1/1 | ~169 s | k-mer profile build from 2 M rows |
| test_alice_tcrnet_benchmark.py | 4/4 | ~480 s | ALICE vs TCRnet concordance |
| test_alice_benchmark.py | 2/2 | ~620 s | YF notebook scaling + pgen parallel |
| test_vdjbet_benchmark.py | 29/29 + 6 skip | ~260 s | VDJbet mock + Q1/Q15 integration |

Tests marked **skip** in the table above require additional environment flags; see the
[Skipped / opt-in tests](#skipped--opt-in-tests) section below.

---

## Detailed results

### test_mirseq_benchmark.py

C-extension speedups over pure-Python equivalents on ~10 k sequences.

| Operation | Python | C | Speedup |
|---|---|---|---|
| translate_linear | 18.3 ms | 0.9 ms | 21 × |
| hamming | 18.3 ms | 1.1 ms | 17 × |
| levenshtein | 506 ms | 4.8 ms | 105 × |
| tokenize_bytes | 13.2 ms | 5.0 ms | 2.6 × |
| tokenize_gapped_bytes | 41.8 ms | 4.8 ms | 8.7 × |
| aa_to_reduced | 1.0 ms | 1.7 ms | 0.6 × (Python bytes.translate is fast here) |

---

### test_pgen_benchmark.py

OLGA pgen throughput (exact and 1-mismatch) at 1- and 8-worker parallelism.

| max_mismatches | workers | n_seqs | elapsed | seqs/s |
|---|---|---|---|---|
| 0 (exact) | 1 | 1 000 | 7.41 s | 135 |
| 0 (exact) | 8 | 1 000 | 2.00 s | 499 |
| 1 | 1 | 200 | 31.45 s | 6.4 |
| 1 | 8 | 200 | 5.87 s | 34 |

---

### test_repertoire_benchmark.py

Parallel AIRR dataset load from `metadata_aging.txt` (41 samples, 281 k clonotypes).

| workers | elapsed | clones/s |
|---|---|---|
| 1 | 0.59 s | 477 k |
| 4 | 0.67 s | 420 k |

---

### test_neighborhood_enrichment_benchmark.py

Three correctness + performance tests; see test file for assertion details.

- **test_neighborhood_runtime_gilg_vs_synthetic_1m** — 5 236-clone target vs 1 M synthetic control: serial 0.88 s, 4-job 4.28 s (overhead-dominated at this size).

---

### test_neighborhood_enrichment_scaling_benchmark.py

Self-neighbour search scaling (n_jobs = 1 / 4 / 8).

| Size | 1 job | 4 jobs | 8 jobs |
|---|---|---|---|
| 100 | < 0.001 s | 0.73 s | 1.05 s |
| 1 000 | 0.012 s | 0.70 s | 1.05 s |
| 10 000 | 0.19 s | 1.01 s | 1.45 s |
| 100 000 | 3.31 s | 2.79 s | 2.84 s |

**1e6 scaling** (opt-in, `MIRPY_BENCH_INCLUDE_1M=1`): 1-job 218.7 s / 4-job 66.9 s / 8-job 46.0 s (**4.7× speedup**). Total wall: 340 s.

---

### test_gliph_benchmark.py

GLIPH2 enrichment + graph clustering on two real studies with a 2 M-row real-control background.

| Study | n_enriched | leiden clusters | AMI (leiden) | Coverage |
|---|---|---|---|---|
| Glanville2017 | 275 | 22 | 0.37 | 0.08 |
| Huang2020 | 1 302 | 200 | 0.55 | 0.22 |

Wall time: **~5 min** (capped real control 2 M rows).

---

### test_gliph_control_benchmark.py

GLIPH tokenisation performance: batch-control mode vs summed single-family mode.

| Control size | Batch elapsed | Single-family total | Speedup |
|---|---|---|---|
| 10 000 | 5.9 s | 7.2 s | 1.22× |
| 100 000 | 59.7 s | 74.9 s | 1.25× |

Rare-token coverage analysis: at 100 k control size, ≥90 % of tokens with frequency ≥ 3 are covered. Power-law fit (R² ≈ 1.0) gives predicted control sizes for 90/95/99 % coverage.

Wall time: **~3.3 min**.

---

### test_tcrnet_benchmark.py

| Test | Result |
|---|---|
| test_tcrnet_benchmark_gil_like_motif_enrichment | 2 enriched motif hits, serial 0.04 s |
| test_tcrnet_runtime_gilg_vs_synthetic_1m | 5 236 target vs 1 M synthetic: serial 1.18 s, 4-job 7.88 s (overhead-dominated) |
| test_tcrnet_benchmark_b35_epl_connected_component_vs_real_control | 20 enriched hits, 4 VDJdb overlaps, largest component 21 nodes (0.26 fraction), 17.9 s @ 4 workers |

---

### test_control_benchmark.py

| Test | Result |
|---|---|
| Synthetic 10 k | Build 22.0 s, cache-hit 0.003 s, load 0.002 s |
| Synthetic 100 k | Build 187.3 s, cache-hit 0.017 s, load 0.014 s |
| Real human TRB (28.3 M rows) | Build 63.6 s, cache-hit 4.58 s, load 4.00 s, file 2.75 GB |
| Real mouse TRA (839 k rows) | Build 2.12 s, cache-hit 0.14 s, load 0.12 s |
| Cache-repeat (real, ×25) | Mean load 7.39 s, ensure overhead 0.976× (< 2.5 %) |

---

### test_bag_of_kmers_benchmark.py

k = 3, human TRB, 2 M rows (sampled from 28.3 M real control).

| Path | Time | Details |
|---|---|---|
| In-memory build | 136.6 s | 75.9 M total k-mers, dominant token "CAS" (position 0) |
| With cache write | 31.6 s | Cache warm-up amortised across subsequent calls |

Peak RSS during build: ~8 GB (Python allocator retains freed arenas; current RSS drops after GC).

---

### test_alice_tcrnet_benchmark.py

ALICE (Pgen-based) vs TCRnet (neighbourhood enrichment) concordance on B35+ and CMV+ samples, 300 clonotypes, 100 k synthetic control.

| Test | Specs | Wall time |
|---|---|---|
| test_alice_tcrnet_synthetic_hamming_concordance | 16 alice + 16 tcrnet (hamming, 4 match-modes) | ~169 s |
| test_tcrnet_synthetic_levenshtein_matrix | 16 tcrnet (levenshtein, 4 match-modes) | ~63 s |
| test_tcrnet_real_hamming_matrix | 16 tcrnet (hamming, real control 50 k) | ~55 s |
| test_tcrnet_real_levenshtein_matrix | 16 tcrnet (levenshtein, real control 50 k) | ~55 s |

ALICE 1mm runs take ~10–13 s each vs ~3 s for exact; 1mm speedup with 4 workers ≈ 2–3×.
TCRnet with real control: ≈ 3–4 s per run regardless of metric.

---

### test_alice_benchmark.py

#### test_alice_yf_notebook_cell6_scaling

Q1 donor repertoires (Q1_d0: 390 k clonotypes; Q1_d15: 477 k clonotypes).
Subsamples: 5 k / 10 k / 25 k. Workers: 1 and 8.

| Sample | Clonotypes | Workers | Total | pgen | Neighborhood |
|---|---|---|---|---|---|
| Q1_d0 | 5 000 | 1 | 33 s | 32 s | 0.02 s |
| Q1_d0 | 5 000 | 8 | 8.0 s | 5.0 s | 1.2 s |
| Q1_d0 | 10 000 | 1 | 64 s | 64 s | 0.03 s |
| Q1_d0 | 10 000 | 8 | 12 s | 9 s | 1.3 s |
| Q1_d0 | 25 000 | 1 | 162 s | 162 s | 0.07 s |
| Q1_d0 | 25 000 | 8 | 25 s | 21 s | 1.3 s |

Wall time for all 12 runs: **613 s (~10 min)**.
Bottleneck is OLGA pgen computation (>99 % of wall time at 1 worker); neighbourhood search is negligible.
8-worker speedup: 4–6× depending on subsample size.

#### test_alice_pgen_10k_single_vs_parallel

10 k sequences, Q1_d0, exact pgen: single-thread 66.9 s → 8-thread 8.8 s → **7.6× speedup**.

---

### test_vdjbet_benchmark.py

#### Fast classes (≤ 55 s total)

| Class | Tests | Key result |
|---|---|---|
| TestPgenBinPoolBenchmark | 3 | 27 k seq/s @ 1 job; range [-70, -19] log₂ Pgen |
| TestVDJBetMockBenchmark | 2 | 4.5 s for 50 mocks; JSD ≈ 0 |
| TestPgenParallelBenchmark | 4 | **4.0× speedup** @ 4 workers; pool reuse 2.7× |
| TestLLWOverlapYFV | 3 | LLW reference mock JSD ≈ 0 |

#### Slow classes (~97 s)

| Class | Tests | Key result |
|---|---|---|
| TestYFVQ1F1 | 4 | Day-15 LLWNGPMAV enrichment significant (p < 0.01); day-0 not; total 79 s |
| TestQ1Q15Integration | 4 | z day-15 = ∞ (n=3 exact hits); z day-0 = 4.36; d15 1mm n = 13 |

#### Other classes (≤ 40 s)

| Class | Tests | Key result |
|---|---|---|
| TestSyntheticVsRealMockComparison | 3 | Synthetic overlap ≈ 0; allele stripping verified |
| TestQ1ControlEffectSize | 2 | d15/d0 effect-size ratio ≥ 1; matching count ratio 3× |
| TestRepertoireIOPolars | 3 | Polars **9.5×** faster than pandas; **71 000×** lower peak memory |

---

## Skipped / opt-in tests

The tests below are excluded from the standard `RUN_BENCHMARK=1` run. Each entry gives the flag
needed to enable it, estimated wall time, and peak RSS on the reference hardware.

---

### `test_control_benchmark.py::test_synthetic_control_1e6_cache_hit_and_optional_cold_build`

**Enable with:** `RUN_FULL_BENCHMARK=1` (or `MIRPY_BENCH_INCLUDE_1M=1`)

Two sub-paths:

| Sub-path | Flag | Est. time | Est. peak RSS |
|---|---|---|---|
| Cache-hit only (prebuilt 1 M cache required) | — | ~30 s | ~1 GB |
| Cold build | `MIRPY_BENCH_1M_COLD_BUILD=1` | **~25–35 min** | ~4–6 GB |

**Time extrapolation:** from the small-matrix benchmark (n=10 k → 22 s; n=100 k → 187 s, ratio ≈ 8.5×/decade), a 1 M synthetic control follows a super-linear curve. Rough estimate: 187 s × 9 ≈ 28 min (the pgen generation step dominates and scales roughly O(n log n) due to sorting).

**Memory:** the pgen pool for 1 M sequences fits in ~2–4 GB; the resulting pickle is ~100 MB on disk.

**Tip:** build once into a shared directory and point subsequent runs at it:
```fish
env RUN_BENCHMARK=1 MIRPY_BENCH_INCLUDE_1M=1 MIRPY_BENCH_1M_COLD_BUILD=1 \
    MIRPY_BENCH_1M_CONTROL_DIR=/path/to/cache \
    python -m pytest tests/test_control_benchmark.py::test_synthetic_control_1e6_cache_hit_and_optional_cold_build -s
```

---

### `test_neighborhood_enrichment_scaling_benchmark.py::test_neighborhood_self_scaling_1e6`

**Enable with:** `MIRPY_BENCH_INCLUDE_1M=1` (or `RUN_FULL_BENCHMARK=1`)

| metric | n_jobs=1 | n_jobs=4 | n_jobs=8 |
|---|---|---|---|
| Wall time | 218.7 s | 66.9 s | 46.0 s |
| Speedup vs serial | — | 3.3× | 4.7× |

**Peak RSS:** ~1.5 GB (synthetic repertoire of 1 M random clonotypes; no real control loaded).
**Total wall time:** ~340 s for all three worker counts.

```fish
env RUN_BENCHMARK=1 MIRPY_BENCH_INCLUDE_1M=1 \
    python -m pytest tests/test_neighborhood_enrichment_scaling_benchmark.py::test_neighborhood_self_scaling_1e6 -s
```

---

### `test_vdjbet_benchmark.py::TestYFVP1SignificanceAndPgenBins` (6 tests)

**Enable with:** `RUN_FULL_BENCHMARK=1` **and** full YFV dataset present at
`notebooks/assets/large/yfv19/` (metadata.txt + per-sample AIRR files).

This class tests P1/F1 day-0 vs day-15 enrichment against the LLWNGPMAV reference on the full
donor-1 repertoires (not the top-3 k subset used in TestQ1Q15Integration).

| Test | What it checks |
|---|---|
| test_p1_f1_d0_not_significant | Day-0 z < 1.96 |
| test_p1_f1_d15_significant | Day-15 z > 1.96 |
| test_p1_f1_d0_duplicate_count_below_d15 | Day-0 total DC < day-15 |
| test_p1_f1_d15_duplicate_count_significant | Day-15 DC-weighted z > 1.96 |
| test_mock_distribution_quality | JSD / KS / Chi² diagnostics |
| test_mock_generation_runtime | Pool generation within budget |

**Est. time:** 30–60 min (full P1/F1 repertoires have ~400 k clonotypes each; pool generation dominates).
**Est. peak RSS:** 4–8 GB (pool of 20 k synthetic sequences + 400 k-clone repertoires in-process).

```fish
env RUN_BENCHMARK=1 RUN_FULL_BENCHMARK=1 \
    python -m pytest tests/test_vdjbet_benchmark.py::TestYFVP1SignificanceAndPgenBins -s
```

---

## Memory guard configuration

The conftest enforces RSS limits via psutil (current RSS, not the macOS `ru_maxrss` high-water mark).

| Benchmark tier | Marker | Default RSS cap | Override env var |
|---|---|---|---|
| Standard | `@benchmark` | 8 GB | `MIRPY_BENCH_MEMORY_LIMIT_GB` |
| Very slow | `@very_slow_benchmark` | 24 GB | `MIRPY_BENCH_MEMORY_LIMIT_VERY_SLOW_GB` |

Tests that load the 28.3 M-row real control pickle peak at ~13–15 GB in-process (Python allocator
retains freed arenas). These tests carry the `very_slow_benchmark` marker specifically to remain
within the 24 GB cap.

---

## Timeout configuration

| Tier | Marker | Default cap | Override env var |
|---|---|---|---|
| Standard | `@slow_benchmark` | 600 s | `MIRPY_BENCH_SLOW_TIMEOUT_S` |
| Very slow | `@very_slow_benchmark` | 1 800 s | `MIRPY_BENCH_VERY_SLOW_TIMEOUT_S` |

The timeout hook fires after the test function returns, so it measures total wall time including
fixture setup. Tests with large real-control loads should carry `very_slow_benchmark`.

---

## Useful environment variables

| Variable | Default | Effect |
|---|---|---|
| `RUN_BENCHMARK` | 0 | Master switch; set to `1` to run benchmark tests |
| `RUN_FULL_BENCHMARK` | 0 | Enables 1 M-scale and full-dataset opt-in tests |
| `MIRPY_BENCH_INCLUDE_1M` | 0 | Enables only the 1 M neighbourhood scaling test |
| `MIRPY_BENCH_REAL_CONTROL_N` | 2 000 000 | Cap rows sampled from the 28.3 M real control |
| `MIRPY_BENCH_WORKERS` | 8 | Comma-separated worker counts for repertoire benchmarks |
| `MIRPY_BENCHMARK_SCALE` | 0.5 | Multiplier for micro-benchmark loop counts (0.05–1.0) |
| `MIRPY_BENCHMARK_MAX_SECONDS` | 120.0 | Per-test wall-clock cap for standard benchmarks |
| `MIRPY_BENCH_SLOW_TIMEOUT_S` | 600 | Timeout for `@slow_benchmark` tests |
| `MIRPY_BENCH_VERY_SLOW_TIMEOUT_S` | 1800 | Timeout for `@very_slow_benchmark` tests |
| `MIRPY_BENCH_MEMORY_LIMIT_GB` | 8 | RSS cap for `@benchmark` tests (GB) |
| `MIRPY_BENCH_MEMORY_LIMIT_VERY_SLOW_GB` | 24 | RSS cap for `@very_slow_benchmark` tests (GB) |
| `MIRPY_BENCH_BAG_OF_KMERS_MAX_ROWS` | 2 000 000 | Row cap for bag-of-k-mers control profile |
| `MIRPY_BENCH_1M_COLD_BUILD` | 0 | Build 1 M synthetic control from scratch |
| `MIRPY_BENCH_1M_CONTROL_DIR` | (tmp) | Shared directory for prebuilt 1 M synthetic cache |
| `MIRPY_BENCH_RESET_LOG` | 1 | Truncate `tests/benchmarks.log` at session start |
| `MIRPY_BENCHMARK_LOG` | tests/benchmarks.log | Path for structured benchmark log |

---

## TODO — skipped tests not yet recorded

The entries below were not executed in the reference run. Estimates are extrapolated from
smaller-scale measurements; actual numbers should replace these once a run is performed.

- [ ] **`test_control_benchmark.py::test_synthetic_control_1e6_cache_hit_and_optional_cold_build`**
  (`RUN_FULL_BENCHMARK=1` or `MIRPY_BENCH_INCLUDE_1M=1`)
  - Cold-build path (`MIRPY_BENCH_1M_COLD_BUILD=1`): est. **25–35 min**, est. **4–6 GB** peak RSS.
    Build time extrapolated from n=10 k (22 s) → n=100 k (187 s), ratio 8.5×/decade.
  - Cache-hit path (prebuilt pickle present): est. **~30 s**, est. **~1 GB** peak RSS.
  - Output to record: build time, cache-hit time, load time, rows, file size on disk.

- [ ] **`test_vdjbet_benchmark.py::TestYFVP1SignificanceAndPgenBins`** (6 tests)
  (`RUN_FULL_BENCHMARK=1` + full YFV dataset at `notebooks/assets/large/yfv19/`)
  - Covers: P1/F1 day-0 not-significant, day-15 significant, d0/d15 duplicate-count ordering,
    mock distribution quality (JSD / KS / Chi²), mock generation runtime.
  - Est. **30–60 min** total (full P1/F1 repertoires ≈ 400 k clonotypes each; pgen dominates).
  - Est. **4–8 GB** peak RSS (pool 20 k seqs + two 400 k-clone repertoires in-process).
  - Output to record: per-test pass/fail, z-scores, p-values, mock diagnostics, runtime.

- [ ] **`test_vdjbet_benchmark.py::TestFunctionalFilteringCounts`** (3 tests)
  (`RUN_INTEGRATION=1` + full YFV dataset at `notebooks/assets/large/yfv19/`)
  - Covers: P1/F1 day-0 and day-15 functional-filter clonotype counts, LLW reference count.
  - Est. **< 5 min** (counting only, no pgen), est. **< 2 GB** peak RSS.
  - Output to record: filtered clonotype counts for each repertoire.
