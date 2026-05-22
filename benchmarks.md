# mirpy Benchmark Results

Reference run recorded **2026-05-22** on an Apple M3 (arm64), 32 GB RAM, 16 cores, Python 3.12.12, macOS 15.
**Final result: 1346 passed, 43 skipped, 0 failed, 0 errors, 350 subtests — wall time 39:27.**

All benchmarks run from the repo root with the activated venv:

```fish
env RUN_BENCHMARK=1 python -m pytest tests/ -s -q --tb=short
```

---

## Notebook execution — full suite

All 11 notebooks run cell-by-cell via `tmp/run_notebook.py` (per-cell timeout 900 s, CWD `notebooks/`, headless matplotlib via `MPLBACKEND=Agg`).
All data loaded from HuggingFace via `mir/utils/notebook_assets.py` utilities.

| Notebook | Cells | Wall time | Peak RSS | Notes |
| --- | --- | --- | --- | --- |
| parsing_example.ipynb | 6 | 5.8 s | ~80 MB | AIRR parse + basic stats |
| gene_similarity.ipynb | 5 | 2.6 s | ~80 MB | V-gene cosine similarity |
| sample_repertoire_overview.ipynb | 7 | 3.3 s | ~80 MB | Repertoire QC plots |
| token_graph.ipynb | 8 | 12.0 s | ~80 MB | GLIPH token graph |
| vdjdb_junction_graph.ipynb | 9 | 16.5 s | ~80 MB | VDJdb junction Hamming graph |
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
| --- | --- | --- |
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
| --- | --- | --- |
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
| --- | --- | --- | --- |
| test_mirseq_benchmark.py | 6/6 | ~2 s | C-extension micro-benchmarks |
| test_pgen_benchmark.py | 1/1 | ~34 s | OLGA exact + 1-mm pgen |
| test_pgen_mc_benchmark.py | 5/5 | ~10 s | MC Pgen pool (500K pool, 500 query) |
| test_pool_benchmark.py | 1/1 | ~2 s | Clonotype pooling |
| test_repertoire_benchmark.py | 1/1 | ~3 s | Parallel I/O |
| test_neighborhood_enrichment_benchmark.py | 3/3 | ~27 s | Hamming neighbour search |
| test_neighborhood_enrichment_scaling_benchmark.py | 1/1 | ~7 s | up to 1e5 (default) |
| test_gliph_benchmark.py | 1/1 | ~4.5 min | GLIPH2 on two real studies |
| test_gliph_control_benchmark.py | 2/2 | ~3 min | Tokenisation + rare-token coverage |
| test_tcrnet_benchmark.py | 3/3 | ~62 s | TCRnet motif enrichment |
| test_control_benchmark.py | 3/3 + 1 skip | ~215 s | Synthetic + real control build |
| test_bag_of_kmers_benchmark.py | 1/1 | ~52 s | k-mer profile build from 2 M rows |
| test_alice_tcrnet_benchmark.py | 4/4 | ~54–82 s | ALICE vs TCRnet concordance |
| test_alice_benchmark.py | 2/2 | ~540 s | YF notebook scaling + pgen parallel |
| test_overlap_benchmark.py | 3/3 | ~75 s | Pairwise overlap timing + 41-sample matrix |
| test_overlap_execution_benchmark.py | 2/2 | ~115 s | Serial overlap on real aging cohort |
| test_metaclonotype_benchmark.py | 3/3 | ~2 s | Metaclonotype creation + functional diversity |
| test_tcremp_benchmark.py | 5/5 | ~75 s | TCREmp throughput + MP scaling |
| test_tcremp_paired_benchmark.py | 1/1 | ~2 s | Paired TRA/TRB embedding speed |
| test_tcremp_vdjdb_benchmark.py | 4/4 | ~15 s | VDJdb TRB clustering (paired + noise-only) |
| test_single_cell_10x_benchmark.py | 4/4 | ~10 s | 10x dcode loading + scaling |
| test_single_cell_citeseq_benchmark.py | 2/2 | ~6 s | CITE-seq matrix loading |
| test_single_cell_conversion_benchmark.py | 2/2 | ~2 s | Sample↔Paired conversion speed |
| test_single_cell_repair_benchmark.py | 1/1 | ~18 s | Imputation + cleanup timing |
| test_vdjbet_benchmark.py | 33/33 + 6 skip | ~260 s | VDJbet mock + Q1/Q15 integration |

Tests marked **skip** in the table above require additional environment flags; see the
[Skipped / opt-in tests](#skipped--opt-in-tests) section below.

> **Memory guard**: After the `test_control_benchmark.py` real-TRB build (28M rows), process RSS
> settles at ~9–10 GB for the remainder of the pytest session. `MIRPY_BENCH_MEMORY_LIMIT_GB`
> default was raised to **32 GB** (and `very_slow_benchmark` cap is also 32 GB) so that
> residual RSS from earlier tests does not trigger false-positive failures.

---

## Detailed results

### test_mirseq_benchmark.py

C-extension speedups over pure-Python equivalents on ~10 k sequences.

| Operation | Python | C | Speedup |
| --- | --- | --- | --- |
| translate_linear | 19.8 ms | 0.5 ms | 36.0 × |
| hamming | 18.7 ms | 0.5 ms | 37.5 × |
| levenshtein | 537 ms | 5.0 ms | 108 × |
| tokenize_bytes | 13.1 ms | 4.7 ms | 2.8 × |
| tokenize_gapped_bytes | 43.3 ms | 4.9 ms | 8.9 × |
| aa_to_reduced | 1.0 ms | 1.0 ms | 1.0 × (Python bytes.translate is fast here) |

---

### test_pgen_benchmark.py

OLGA pgen throughput (exact and 1-mismatch) at 1- and 8-worker parallelism.

| max_mismatches | workers | n_seqs | elapsed | seqs/s |
| --- | --- | --- | --- | --- |
| 0 (exact) | 1 | 1 000 | 6.65 s | 150 |
| 0 (exact) | 8 | 1 000 | 1.76 s | 570 |
| 1 | 1 | 200 | 17.64 s | 11.3 |
| 1 | 8 | 200 | 3.35 s | 59.6 |

Additional pgen metrics (test_pgen_benchmark.py):

- 1-core throughput: 10 k seqs in 0.285 s → **35 109 seqs/s**
- 4-core throughput: 10 k seqs in 0.668 s → **14 961 seqs/s** (0.47× — overhead-dominated at this size)
- Pool reuse: first 1.047 s, second 0.536 s → **1.95× reuse ratio**, 559 seqs/s

---

### test_pgen_mc_benchmark.py

MC Pgen pool: synthetic pool build times, accuracy vs OLGA, and Q-factor analysis.
Pool size: 500K (CI default). Query: 500 TRB CDR3s from YFV dataset.

#### Pool build times (8 workers, Apple M3)

| Locus | Pool built | p_productive | n_productive |
| --- | --- | --- | --- |
| TRB | 2.8 s | 0.244 | 500 000 |
| TRA | 1.9 s | 0.289 | 500 000 |

#### TRB accuracy vs OLGA (500 queries)

| Method | OLGA time | MC time | Speedup | Coverage | r (log10) | rmse | fold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MC exact | 1.39 s | 0.000 s | 7 778 × | 2.8% (14 seqs) | 0.38 | 0.61 | 4.0 × |
| MC 1mm | 1.39 s | 0.020 s | 71 × | 31.6% (158 seqs) | 0.74 | 0.48 | 3.0 × |

> At 500K pool exact MC coverage is ~3% (too sparse for reliable r); 1mm gives ~32% and strong correlation (r=0.74). Full-scale (10M pool) gives ≥85% 1mm coverage, r>0.85.

#### TRA accuracy vs OLGA (500 queries)

| Method | Speedup | Coverage | r (log10) |
| --- | --- | --- | --- |
| MC 1mm | 68 × | 62.0% | 0.77 |

#### TRB Q-factor from real control (n=241 matched)

| Stat | Value |
| --- | --- |
| median Q | 899 |
| mean Q | 5.6 × 10⁹ |
| log10 std | 1.19 (15 × spread) |

> Q >> 1 reflects thymic selection enriching functional TRB sequences relative to the OLGA recombination model.

#### p_productive calibration

| Locus | p_productive | n_total |
| --- | --- | --- |
| TRB | 0.244 | 2 048 341 |
| TRA | 0.289 | 1 729 505 |

---

### test_repertoire_benchmark.py

Parallel AIRR dataset load from `metadata_aging.txt` (41 samples, 281 k clonotypes).

| workers | elapsed | clones/s |
| --- | --- | --- |
| 1 | 0.67 s | 418 k |
| 4 | 0.54 s | 524 k |

Full I/O benchmark: 41 files, total 7.55 s, p50=0.181 s, p95=0.246 s, peak_mem=15.9 MiB.

---

### test_neighborhood_enrichment_benchmark.py

Three correctness + performance tests; see test file for assertion details.

- **test_neighborhood_runtime_gilg_vs_synthetic_1m** — 5 236-clone target vs 1 M synthetic control: Hamming graph 0.04 s (4 727 edges), Levenshtein graph 0.11 s (4 951 edges).

---

### test_neighborhood_enrichment_scaling_benchmark.py

Self-neighbour search scaling (n_jobs = 1 / 4 / 8).

| Size | 1 job | 4 jobs | 8 jobs |
| --- | --- | --- | --- |
| 100 | < 0.001 s | < 0.001 s | < 0.001 s |
| 1 000 | 0.011 s | 0.010 s | 0.010 s |
| 10 000 | 0.163 s | 0.153 s | 0.150 s |
| 100 000 | 2.72 s | 4.31 s | 5.62 s |

> Note: for ≤10k clones parallel overhead exceeds gain; single-thread is faster. Speedup emerges at 1M scale (opt-in).

**1e6 scaling** (opt-in, `MIRPY_BENCH_INCLUDE_1M=1`): 1-job 218.7 s / 4-job 66.9 s / 8-job 46.0 s (**4.7× speedup**). Total wall: 340 s.

---

### test_gliph_benchmark.py

GLIPH2 enrichment + graph clustering on two real studies with a 2 M-row real-control background.

| Study | n_enriched | leiden clusters | AMI (leiden) | Coverage |
| --- | --- | --- | --- | --- |
| Glanville2017 | 272 | 21 | 0.357 | 0.082 |
| Huang2020 | 1 377 | 201 | 0.541 | 0.222 |

Wall time: **~4.5 min** (capped real control 2 M rows).

---

### test_gliph_control_benchmark.py

GLIPH tokenisation performance: batch-control mode vs summed single-family mode.

| Control size | Batch elapsed | Single-family total | Speedup |
| --- | --- | --- | --- |
| 10 000 | 4.09 s | 5.09 s | 1.25 × |
| 100 000 | 41.83 s | 50.94 s | 1.22 × |

Rare-token coverage analysis: at 100 k control size, ≥90 % of tokens with frequency ≥ 3 are covered. Power-law fit (R² ≈ 1.0) gives predicted control sizes for 90/95/99 % coverage.

Wall time: **~3 min**.

---

### test_tcrnet_benchmark.py

| Test | Result |
| --- | --- |
| test_tcrnet_benchmark_gil_like_motif_enrichment | 2 enriched motif hits, serial 0.005 s (tiny synthetic benchmark, 204 target vs 503 control) |
| test_tcrnet_runtime_gilg_vs_synthetic_1m | 5 236 target vs 1 M synthetic: serial 0.945 s, 4-job 6.406 s (overhead-dominated) |
| test_tcrnet_benchmark_b35_epl_connected_component_vs_real_control | elapsed 9.74 s (target 61 368, control 2 M), strict enriched 20 (vdjdb overlap 4), neighbor nodes 27, largest CC 21, components≥5: 1, component overlap 0.256 |

---

### test_control_benchmark.py

| Test | Result |
| --- | --- |
| Synthetic 10 k | Build 18.20 s, cache-hit 0.002 s, load 0.001 s |
| Synthetic 100 k | Build 173.44 s, cache-hit 0.008 s, load 0.006 s |
| Real human TRB (28.3 M rows) | Build 25.26 s, cache-hit 2.79 s, load 2.70 s, file 3.6 GB |
| Real mouse TRA (839 k rows) | Build 1.29 s, cache-hit 0.06 s, load 0.05 s |
| Cache-repeat (real, ×25) | Mean load 1.96 s, ensure overhead 0.968× (< 1 %) |

---

### test_bag_of_kmers_benchmark.py

k = 3, human TRB, 2 M rows (sampled from 28.3 M real control).

| Path | Time | Details |
| --- | --- | --- |
| In-memory build | 40.29 s | 75.2 M total k-mers, dominant token "CAS" (position 0) |
| With cache write | 15.94 s | Cache warm-up amortised across subsequent calls |

---

### test_alice_tcrnet_benchmark.py

ALICE (Pgen-based) vs TCRnet (neighbourhood enrichment) concordance on B35+ and CMV+ samples, 300 clonotypes, 100 k synthetic control.

| Test | Specs | Wall time |
| --- | --- | --- |
| test_alice_tcrnet_synthetic_hamming_concordance | 16 alice + 16 tcrnet (hamming, 4 match-modes) | ~54 s |
| test_tcrnet_synthetic_levenshtein_matrix | 16 tcrnet (levenshtein, 4 match-modes) | ~11 s |
| test_tcrnet_real_hamming_matrix | 16 tcrnet (hamming, real control 50 k) | ~9 s |
| test_tcrnet_real_levenshtein_matrix | 16 tcrnet (levenshtein, real control 50 k) | ~8 s |

ALICE 1mm runs take ~3 s each vs ~2.3 s for exact; 1mm speedup with 8 workers ≈ 55×.
TCRnet with real control: ≈ 0.07–0.12 s per run regardless of metric.

---

### test_alice_benchmark.py

#### test_alice_yf_notebook_cell6_scaling

Q1 donor repertoires (Q1_d0.tsv.gz: ~371 k clonotypes; Q1_d15.tsv.gz: ~375 k clonotypes).
Subsamples: 5 k / 10 k / 25 k. Workers: 1 and 8.

| Sample | Clonotypes | Workers | Total | pgen | Neighborhood |
| --- | --- | --- | --- | --- | --- |
| Q1_d0.tsv | 5 000 | 1 | 3.1 s | 3.1 s | 0.04 s |
| Q1_d0.tsv | 5 000 | 8 | 3.3 s | 1.1 s | 0.03 s |
| Q1_d0.tsv | 10 000 | 1 | 8.1 s | 8.0 s | 0.07 s |
| Q1_d0.tsv | 10 000 | 8 | 3.6 s | 1.2 s | 0.07 s |
| Q1_d0.tsv | 25 000 | 1 | 30.2 s | 29.9 s | 0.21 s |
| Q1_d0.tsv | 25 000 | 8 | 11.3 s | 4.1 s | 4.6 s |

Wall time for all 12 runs: **~540 s (~9 min)**.
Bottleneck is OLGA pgen computation (>99 % of wall time at 1 worker); neighbourhood search is negligible at small scales.
8-worker speedup: 1.0–2.7× depending on subsample size (pgen-limited).

#### test_alice_pgen_10k_single_vs_parallel

10 k sequences, Q1_d0, exact pgen: single-thread 66.7 s → 8-thread 8.6 s → **7.74× speedup**.

---

### test_overlap_benchmark.py

#### Per-pair timing (A2-i132.txt.gz × itself, 9 632 clonotypes)

| Mode | n_jobs | n1_matched | Time | Peak |
| --- | --- | --- | --- | --- |
| exact:0 | 1 | 9 375 | 61.9 ms | 3 MB |
| exact:0 | −1 (auto) | 9 375 | 59.2 ms | 3 MB |
| hamming:1 | 1 | 9 375 | 190.3 ms | 3 MB |
| hamming:1 | −1 (auto) | 9 375 | 189.1 ms | 3 MB |
| levenshtein:1 | 1 | 9 375 | 351.0 ms | 3 MB |
| levenshtein:1 | −1 (auto) | 9 375 | 350.2 ms | 3 MB |

D (Dice) = F (F1) = 1.000 in all cases (self-overlap, expected).

#### Many-vs-many matrix (41 aging repertoires, 820 pairs)

| Mode | n_jobs | Total time | Peak RAM |
| --- | --- | --- | --- |
| exact:0 | −1 | 20.71 s | 111 MB |
| hamming:1 | −1 | 22.33 s | 111 MB |
| levenshtein:1 | −1 | 29.21 s | 111 MB |

#### Pilot benchmark (16 repertoires, 120 pairs, n_jobs=4)

Pilot estimate (28 pairs): 0.0073 s/pair → extrapolated 120-pair: 0.88 s.
Actual 120-pair run: 1.08 s (n_jobs_effective=4).

---

### test_overlap_execution_benchmark.py

Serial execution on real aging cohort (HuggingFace `isalgo/airr_benchmark`):

| Mode | Subset | Time | Per-pair | Extrapolated (79 donors) |
| --- | --- | --- | --- | --- |
| many-vs-many exact | 8 samples, 28 pairs | 23.30 s | 0.832 s/pair | **42.7 min** |
| many-vs-pool exact | 8 samples, 8 pool queries | 18.54 s | 2.32 s/sample | **183 s** |

Thread-parallel many-vs-many (n_jobs=4, same 28 pairs): 22.95 s (effective_jobs=4).

> At 0.83 s/pair with 79 donors (3 081 pairs), the full cohort takes ~43 min at n_jobs=1.
> With n_jobs=4 (thread pool, memory shared), the speedup is minimal for exact overlap at this pair size; the bottleneck is the C-level Hamming/Levenshtein scan, not Python overhead.

---

### test_metaclonotype_benchmark.py

Synthetic TRB repertoires; average cluster size 5; scales 1 K, 5 K, 10 K, 50 K clonotypes.

#### TestMetaclonotypeCreationBenchmark — `metaclonotypes_from_labels` and `metaclonotypes_from_components`

| n | n_clusters | t_labels (ms) | rss_labels (MB) | t_components (ms) | rss_components (MB) |
| --- | --- | --- | --- | --- | --- |
| 1 000 | ~193 | 2.0 | < 1 | 1.1 | < 1 |
| 5 000 | ~983 | 39.5 | < 1 | 2.7 | < 1 |
| 10 000 | ~1 974 | 5.8 | < 1 | 4.5 | < 1 |
| 50 000 | ~9 805 | 17.3 | < 1 | 16.0 | < 1 |

#### TestMetaclonotypeAnalyticsBenchmark — `summarize_metaclonotypes`, Hill curve, rarefaction

| n | n_clusters | t_summarize (ms) | t_diversity (ms) | t_hill (ms) | t_rarefy (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 000 | ~193 | 5.3 | 0.1 | 0.31 | 4.47 |
| 5 000 | ~983 | 10.4 | 0.1 | 1.46 | 6.99 |
| 10 000 | ~1 974 | 14.8 | 0.1 | 0.92 | 7.30 |
| 50 000 | ~9 805 | 57.3 | 1.0 | 4.70 | 7.70 |

#### TestFunctionalDiversityEndToEndBenchmark — `functional_diversity()`

End-to-end: `LocusRepertoire` → `MetaClonotypeDefinition` → `DiversitySummary`.

| n | n_clusters | elapsed (ms) | rss (MB) | Shannon H | Chao1 |
| --- | --- | --- | --- | --- | --- |
| 1 000 | 193 | 3.7 | < 1 | 5.106 | 193 |
| 5 000 | 983 | 9.9 | < 1 | 6.741 | 983 |
| 10 000 | 1 974 | 13.8 | < 1 | 7.427 | 1 974 |
| 50 000 | 9 805 | 58.1 | 0.8 | 9.035 | 9 812 |

Wall time for all 3 benchmark classes: **~1.6 s** (Apple M3 reference hardware).

---

### test_tcremp_benchmark.py

#### Distance correlation

Human TRB, 1 000 prototypes; 1 000 query clonotypes (499 500 pairs):

| Metric | Value |
| --- | --- |
| Model build | 0.83 s, peak 2 MB |
| Embed 1k clones | 0.03 s, peak 31 MB, shape (1 000, 3 000) |
| Distance compute | 0.03 s |
| Pearson R² | 0.5723 |
| Spearman ρ | 0.7282 |

Per-component R² (V, J, CDR3, L2-emb): 0.4691 / 0.1633 / 0.5547 / 0.5723.

#### Throughput (n_jobs=1)

| n_clono | n_proto | Time | Clono/s | Peak |
| --- | --- | --- | --- | --- |
| 10 000 | 1 000 | 0.30 s | 33 085 | 306 MB |
| 100 000 | 1 000 | 3.29 s | 30 415 | 3 056 MB |
| 100 000 | 3 000 | 9.76 s | 10 242 | 9 160 MB |
| 500 000 | 1 000 | 19.24 s | 25 983 | 15 282 MB |
| 1 000 000 | 1 000 | 43.65 s | 22 911 | 30 565 MB |

#### Multiprocessing scaling

| n_clono | n_jobs=1 | n_jobs=2 | n_jobs=4 | n_jobs=8 |
| --- | --- | --- | --- | --- |
| 10 000 | 0.30 s | 0.20 s (1.52×) | 0.14 s (2.11×) | 0.12 s (2.60×) |
| 100 000 | 3.09 s | 2.05 s (1.51×) | 1.50 s (2.06×) | 1.23 s (2.50×) |
| 500 000 | 15.40 s | 10.70 s (1.44×) | 7.73 s (1.99×) | 6.35 s (2.42×) |

#### Key aligner measurements

| Benchmark | Result |
| --- | --- |
| Fixed-gap vs BioPython (10k × 3k) | Fixed-gap 0.25 s (121.7 M p/s) vs BioPython 106.4 s (282 k p/s) → **432×** speedup |
| Junction aligner (1M pairs) | Fixed-gap 47.9 M p/s vs full-DP 250 k p/s → **192×** speedup |
| Parallel chunking avg (16 cores) | 100: 1.04×, 1k: 1.93×, 5k: 2.68×, 10k: 2.82×, **avg 2.12×** |
| PCA V-gene correlation | Best PC1 p=0, var=16.1% |
| PCA J-gene correlation | Best PC5 p=0, var=1.23% |
| PCA junction-length correlation | PC0 r=+0.908, p=0, var=63.99% |
| Epitope specificity (GLC vs YLQ) | Within 9100 < Between 11447, p=9.99e-04, Cohen's d=0.72 |

---

### test_tcremp_analysis.py

All tests pass. Key measurements:

| Benchmark | Result |
| --- | --- |
| Fixed-gap vs BioPython (10k × 3k) | Fixed-gap 0.25 s (121.7 M p/s) vs BioPython 106.4 s (282 k p/s) → **432×** speedup |
| Embedding distance (cosine, 1k clones) | mean 0.265 ± 0.033; RMSE=570.3; Corr=0.147 (total seq distance) |
| Prototype symmetry (1k prototypes) | Max asymmetry 0.00, max diagonal 0.00; R²=1.000, RMSE=0.000 (latent ↔ sequence) |

---

### test_tcremp_paired_benchmark.py

| Benchmark | Result |
| --- | --- |
| Records | 2 000 records, 200 prototypes/chain |
| TRA single | 0.013 s (0.006 ms/record) |
| TRB single | 0.013 s (0.006 ms/record) |
| Paired | 0.026 s (0.013 ms/record) |
| Paired / (TRA+TRB) overhead | 1.041× |

---

### test_tcremp_vdjdb_benchmark.py

#### Single-chain VDJdb TRB (TestSingleChainVDJdbTCREmpQuality — passed)

3 000 records, 500 prototypes: **0.082 s**.
DBSCAN clustering: n_comp=27, eps=0.315, clusters=171, retention=0.667, purity=0.451.

#### Paired VDJdb (TestPairedVDJdbTCREmpQuality)

| Mode | n | Time | Purity | Retention |
| --- | --- | --- | --- | --- |
| Strict | 3 000 | 0.166 s | 0.569 | 0.615 |
| Imputed | 3 000 | 0.163 s | 0.472 | 0.714 |
| Paired / (TRA+TRB) overhead | — | — | — | 1.082× |

#### Mixed random bootstrap (TestSingleChainVDJdbMixedLargeBootstrap)

32 719 total (3 000 VDJdb TRB + 29 719 random), embed=0.257 s, diag=6.895 s, eps=0.0856, retention=0.620.

#### Noise-only 10k (TestNoiseOnly10k)

10 000 OLGA-generated TRB, n_prototypes=1000: embed=0.079 s, diag=2.927 s, eps=0.108 ≤ median_4nn=0.115. eps stays within bounds (conservative selection confirmed).

---

### test_single_cell_10x_benchmark.py

Loaded all dcode donors (390 641 clonotypes from `vdj_v1_hs_aggregated` series):

| Mode | Time | Clonotypes | Peak RSS |
| --- | --- | --- | --- |
| Sequential | 1.58 s | 390 641 | 10 142 MB |
| 2 workers (1k chunks) | 3.09 s | 390 641 | 10 128 MB |
| 4 workers (1k chunks) | 3.16 s | 390 641 | 10 096 MB |

Worker count scaling (separate sub-benchmark):

| Workers | Time | Speedup | Peak RSS |
| --- | --- | --- | --- |
| 1 | 2.22 s | — | 10 132 MB |
| 2 | 3.40 s | 0.7× | 10 138 MB |
| 4 | 3.36 s | 0.7× | 10 140 MB |

> Note: single-cell 10x data is I/O-bound; parallel loading incurs multiprocessing overhead that dominates at this scale.

Chunk-size sweep: 500→3.62 s / 1k→3.45 s / 5k→3.47 s / 10k→3.52 s.

**Estimated for 1 M clonotypes:** sequential ~3.9 s / 25.5 GB; 4-worker ~1.0 s / 20.4 GB.

---

### test_single_cell_citeseq_benchmark.py

10x CITE-seq matrix loading from dcode donors:

| Donor | Cells | Pairs | Matrix rows | Binder cols | Time |
| --- | --- | --- | --- | --- | --- |
| donor1 | 47 271 | 48 890 | 46 526 | 78 | 1.04 s |
| donor2 | 79 704 | 72 266 | 77 854 | 78 | 2.00 s |
| donor3 | 38 095 | 39 518 | 37 824 | 78 | 1.17 s |
| donor4 | 27 640 | 29 147 | 27 308 | 78 | 0.68 s |

---

### test_single_cell_repair_benchmark.py

Imputation + cleanup on dcode data:

- impute: **17.33 s** (102 610 imputed rows from 95 663 raw)
- cleanup: **0.53 s** (100 250 cleaned rows)

---

### test_vdjbet_benchmark.py

#### Fast classes (≤ 55 s total)

| Class | Tests | Key result |
| --- | --- | --- |
| TestPgenBinPoolBenchmark | 3 | 78 781 seq/s @ 1 job; range [−70, −19] log₂ Pgen |
| TestVDJBetMockBenchmark | 2 | 50 mocks in 3.09 s; JSD ≈ 0 |
| TestPgenParallelBenchmark | 4 | **3.31× speedup** @ 4 workers; pool reuse 2.44× |
| TestLLWOverlapYFV | 3 | LLW reference 409 clonotypes; mock JSD ≈ 0 |

#### Slow classes (~97 s)

| Class | Tests | Key result |
| --- | --- | --- |
| TestYFVQ1F1 | 4 | Day-15 LLWNGPMAV enrichment z=∞; day-0 not; total 46 s |
| TestQ1Q15Integration | 4 | z day-15 = ∞ (n=3 exact hits); z day-0 = 4.36; d15 1mm n=13 |

#### Other classes (≤ 40 s)

| Class | Tests | Key result |
| --- | --- | --- |
| TestSyntheticVsRealMockComparison | 3 | Synthetic overlap ≈ 0; allele stripping verified |
| TestQ1ControlEffectSize | 2 | d15/d0 effect-size ratio (d15 n=3, d0 n=1, 3.0× count ratio) |
| TestRepertoireIOPolars | 3 | Polars **8.56×** faster than pandas; **43 069×** lower peak memory |

---

### test_token_tables*.py

All tests pass. Key measurements:

| Benchmark | Result |
| --- | --- |
| tokenize (100k rearrangements, k=5) | 1.150 s (86 979 rearrangements/s) |
| lookup hits (1M ops) | 0.028 s (35.2 M ops/s) |
| lookup misses (1M ops) | 0.017 s (59.5 M ops/s) |
| summarize plain (100k, k=5) | 0.415 s (241 193 rearrangements/s) |
| summarize gapped (100k, k=5) | 1.957 s (51 094 rearrangements/s) |
| summarize_annotations plain (100k, k=5) | 0.615 s (162 722 rearrangements/s) |
| naive summarize (10k, k=3) | 0.306 s, 47 369 keys, peak 17 067 KiB |
| polars expand+summarize (10k, k=3) | 0.229 s, 47 369 rows, peak 18 KiB |
| naive summarize_annotations (10k, k=3) | 0.738 s, 6 221 KmerSeq, peak 30 557 KiB |
| polars 4 summaries (10k, k=3) | 0.241 s, peak 20 KiB |
| polars fetch_by_kmer 'CAS' (1k lookups) | 1.173 s (853 ops/s) |

---

## Skipped / opt-in tests

The tests below are excluded from the standard `RUN_BENCHMARK=1` run. Each entry gives the flag
needed to enable it, estimated wall time, and peak RSS on the reference hardware.

---

### `test_control_benchmark.py::test_synthetic_control_1e6_cache_hit_and_optional_cold_build`

**Enable with:** `RUN_FULL_BENCHMARK=1` (or `MIRPY_BENCH_INCLUDE_1M=1`)

Two sub-paths:

| Sub-path | Flag | Est. time | Est. peak RSS |
| --- | --- | --- | --- |
| Cache-hit only (prebuilt 1 M cache required) | — | ~30 s | ~1 GB |
| Cold build | `MIRPY_BENCH_1M_COLD_BUILD=1` | **~25–35 min** | ~4–6 GB |

**Time extrapolation:** from the small-matrix benchmark (n=10 k → 18 s; n=100 k → 173 s, ratio ≈ 9.5×/decade), a 1 M synthetic control follows a super-linear curve. Rough estimate: 173 s × 10 ≈ 29 min.

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
| --- | --- | --- | --- |
| Wall time | 218.7 s | 66.9 s | 46.0 s |
| Speedup vs serial | — | 3.3× | 4.7× |

**Peak RSS:** ~1.5 GB.

```fish
env RUN_BENCHMARK=1 MIRPY_BENCH_INCLUDE_1M=1 \
    python -m pytest tests/test_neighborhood_enrichment_scaling_benchmark.py::test_neighborhood_self_scaling_1e6 -s
```

---

### `test_vdjbet_benchmark.py::TestYFVP1SignificanceAndPgenBins` (6 tests)

**Enable with:** `RUN_FULL_BENCHMARK=1` **and** full YFV dataset present at
`notebooks/assets/large/yfv19/` (metadata.txt + per-sample AIRR files).

| Test | What it checks |
| --- | --- |
| test_p1_f1_d0_not_significant | Day-0 z < 1.96 |
| test_p1_f1_d15_significant | Day-15 z > 1.96 |
| test_p1_f1_d0_duplicate_count_below_d15 | Day-0 total DC < day-15 |
| test_p1_f1_d15_duplicate_count_significant | Day-15 DC-weighted z > 1.96 |
| test_mock_distribution_quality | JSD / KS / Chi² diagnostics |
| test_mock_generation_runtime | Pool generation within budget |

**Est. time:** 30–60 min. **Est. peak RSS:** 4–8 GB.

```fish
env RUN_BENCHMARK=1 RUN_FULL_BENCHMARK=1 \
    python -m pytest tests/test_vdjbet_benchmark.py::TestYFVP1SignificanceAndPgenBins -s
```

---

## Memory guard configuration

The conftest enforces RSS limits via psutil (current RSS, not the macOS `ru_maxrss` high-water mark).

| Benchmark tier | Marker | Default RSS cap | Override env var |
| --- | --- | --- | --- |
| Standard | `@benchmark` | **32 GB** | `MIRPY_BENCH_MEMORY_LIMIT_GB` |
| Very slow | `@very_slow_benchmark` | **32 GB** | `MIRPY_BENCH_MEMORY_LIMIT_VERY_SLOW_GB` |

The 32 GB default accounts for residual RSS from the real-control build (28 M TRB rows,
~9–10 GB residual) so earlier tests do not trigger false-positive failures in the same session.
The machine has 32 GB RAM; both tiers share the same cap to avoid surprising limit differences.

---

## Timeout configuration

| Tier | Marker | Default cap | Override env var |
| --- | --- | --- | --- |
| Standard | `@slow_benchmark` | 1 800 s | `MIRPY_BENCH_SLOW_TIMEOUT_S` |
| Very slow | `@very_slow_benchmark` | 1 800 s | `MIRPY_BENCH_VERY_SLOW_TIMEOUT_S` |

---

## Useful environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `RUN_BENCHMARK` | 0 | Master switch; set to `1` to run benchmark tests |
| `RUN_FULL_BENCHMARK` | 0 | Enables 1 M-scale and full-dataset opt-in tests |
| `MIRPY_BENCH_INCLUDE_1M` | 0 | Enables only the 1 M neighbourhood scaling test |
| `MIRPY_BENCH_REAL_CONTROL_N` | 2 000 000 | Cap rows sampled from the 28.3 M real control |
| `MIRPY_BENCH_WORKERS` | 8 | Comma-separated worker counts for repertoire benchmarks |
| `MIRPY_BENCHMARK_SCALE` | 0.5 | Multiplier for micro-benchmark loop counts (0.05–1.0) |
| `MIRPY_BENCHMARK_MAX_SECONDS` | 120.0 | Per-test wall-clock cap for standard benchmarks |
| `MIRPY_BENCH_SLOW_TIMEOUT_S` | 1800 | Timeout for `@slow_benchmark` tests |
| `MIRPY_BENCH_VERY_SLOW_TIMEOUT_S` | 1800 | Timeout for `@very_slow_benchmark` tests |
| `MIRPY_BENCH_MEMORY_LIMIT_GB` | **32** | RSS cap for `@benchmark` tests (GB) |
| `MIRPY_BENCH_MEMORY_LIMIT_VERY_SLOW_GB` | 32 | RSS cap for `@very_slow_benchmark` tests (GB) |
| `MIRPY_BENCH_BAG_OF_KMERS_MAX_ROWS` | 2 000 000 | Row cap for bag-of-k-mers control profile |
| `MIRPY_BENCH_1M_COLD_BUILD` | 0 | Build 1 M synthetic control from scratch |
| `MIRPY_BENCH_1M_CONTROL_DIR` | (tmp) | Shared directory for prebuilt 1 M synthetic cache |
| `MIRPY_BENCH_RESET_LOG` | 1 | Truncate `tests/benchmarks.log` at session start |
| `MIRPY_BENCHMARK_LOG` | tests/benchmarks.log | Path for structured benchmark log |
| `MIRPY_MC_BENCH_N` | 500 000 | MC Pgen pool size (set to 10 000 000 for full-scale run) |
| `MIRPY_MC_BENCH_NQUERY` | 500 | Number of query sequences for MC Pgen benchmark |
| `MIRPY_MC_BENCH_NJOBS` | 8 | Workers for MC Pgen pool build and bulk pgen |
