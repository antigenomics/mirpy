# mirpy Benchmark Results

Reference run recorded **2026-05-17** on an Apple M3 (arm64), 32 GB RAM, 16 cores, Python 3.12.12, macOS 15.
**Final result: 1197 passed, 32 skipped, 0 failed, 0 errors, 350 subtests — wall time 48:56.**

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
| test_pool_benchmark.py | 1/1 | ~2 s | Clonotype pooling |
| test_repertoire_benchmark.py | 1/1 | ~3 s | Parallel I/O |
| test_neighborhood_enrichment_benchmark.py | 3/3 | ~27 s | Hamming neighbour search |
| test_neighborhood_enrichment_scaling_benchmark.py | 1/1 | ~7 s | up to 1e5 (default) |
| test_gliph_benchmark.py | 1/1 | ~4.5 min | GLIPH2 on two real studies |
| test_gliph_control_benchmark.py | 2/2 | ~3 min | Tokenisation + rare-token coverage |
| test_tcrnet_benchmark.py | 3/3 | ~62 s | TCRnet motif enrichment |
| test_control_benchmark.py | 3/3 + 1 skip | ~215 s | Synthetic + real control build |
| test_bag_of_kmers_benchmark.py | 1/1 | ~52 s | k-mer profile build from 2 M rows |
| test_alice_tcrnet_benchmark.py | 4/4 | ~362 s | ALICE vs TCRnet concordance |
| test_alice_benchmark.py | 2/2 | ~610 s | YF notebook scaling + pgen parallel |
| test_overlap_benchmark.py | 3/3 | ~75 s | Pairwise overlap timing + 41-sample matrix |
| test_overlap_execution_benchmark.py | 2/2 | ~115 s | Serial overlap on real aging cohort |
| test_metaclonotype_benchmark.py | 3/3 | ~2 s | Metaclonotype creation + functional diversity |
| test_tcremp_benchmark.py | 5/5 | ~75 s | TCREmp throughput + MP scaling |
| test_tcremp_paired_benchmark.py | 1/1 | ~2 s | Paired TRA/TRB embedding speed |
| test_tcremp_vdjdb_benchmark.py | 1/2 | ~5 s | VDJdb TRB clustering (paired requires re-run) |
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
| translate_linear | 18.0 ms | 0.8 ms | 22.5 × |
| hamming | 18.0 ms | 1.0 ms | 17.4 × |
| levenshtein | 499 ms | 4.6 ms | 108 × |
| tokenize_bytes | 12.2 ms | 4.5 ms | 2.7 × |
| tokenize_gapped_bytes | 42.3 ms | 4.7 ms | 8.9 × |
| aa_to_reduced | 1.0 ms | 1.8 ms | 0.6 × (Python bytes.translate is fast here) |

---

### test_pgen_benchmark.py

OLGA pgen throughput (exact and 1-mismatch) at 1- and 8-worker parallelism.

| max_mismatches | workers | n_seqs | elapsed | seqs/s |
| --- | --- | --- | --- | --- |
| 0 (exact) | 1 | 1 000 | 6.21 s | 161 |
| 0 (exact) | 8 | 1 000 | 1.14 s | 877 |
| 1 | 1 | 200 | 21.99 s | 9.1 |
| 1 | 8 | 200 | 3.30 s | 61 |

Additional pgen metrics (test_pgen_benchmark.py):

- 1-core throughput: 10 k seqs in 0.305 s → **32 777 seqs/s**
- 4-core throughput: 10 k seqs in 0.333 s → **30 038 seqs/s** (0.90× speedup — overhead-dominated at this size)
- Pool reuse: first 0.726 s, second 0.499 s → **1.45× reuse ratio**, 601 seqs/s

---

### test_repertoire_benchmark.py

Parallel AIRR dataset load from `metadata_aging.txt` (41 samples, 281 k clonotypes).

| workers | elapsed | clones/s |
| --- | --- | --- |
| 1 | 0.79 s | 356 k |
| 4 | 0.66 s | 425 k |

Full I/O benchmark: 41 files, total 5.99 s, p50=0.141 s, p95=0.189 s, peak_mem=14.9 MiB.

---

### test_neighborhood_enrichment_benchmark.py

Three correctness + performance tests; see test file for assertion details.

- **test_neighborhood_runtime_gilg_vs_synthetic_1m** — 5 236-clone target vs 1 M synthetic control: Hamming graph 0.04 s (4 727 edges), Levenshtein graph 0.12 s (4 951 edges).

---

### test_neighborhood_enrichment_scaling_benchmark.py

Self-neighbour search scaling (n_jobs = 1 / 4 / 8).

| Size | 1 job | 4 jobs | 8 jobs |
| --- | --- | --- | --- |
| 100 | < 0.001 s | 0.48 s | 0.53 s |
| 1 000 | 0.011 s | 0.48 s | 0.55 s |
| 10 000 | 0.168 s | 0.86 s | 1.23 s |
| 100 000 | 2.81 s | 2.21 s | 2.26 s |

**1e6 scaling** (opt-in, `MIRPY_BENCH_INCLUDE_1M=1`): 1-job 218.7 s / 4-job 66.9 s / 8-job 46.0 s (**4.7× speedup**). Total wall: 340 s.

---

### test_gliph_benchmark.py

GLIPH2 enrichment + graph clustering on two real studies with a 2 M-row real-control background.

| Study | n_enriched | leiden clusters | AMI (leiden) | Coverage |
| --- | --- | --- | --- | --- |
| Glanville2017 | 272 | 21 | 0.355 | 0.082 |
| Huang2020 | 1 377 | 201 | 0.542 | 0.222 |

Wall time: **~4.5 min** (capped real control 2 M rows).

---

### test_gliph_control_benchmark.py

GLIPH tokenisation performance: batch-control mode vs summed single-family mode.

| Control size | Batch elapsed | Single-family total | Speedup |
| --- | --- | --- | --- |
| 10 000 | 4.14 s | 5.13 s | 1.24× |
| 100 000 | 42.26 s | 51.65 s | 1.22× |

Rare-token coverage analysis: at 100 k control size, ≥90 % of tokens with frequency ≥ 3 are covered. Power-law fit (R² ≈ 1.0) gives predicted control sizes for 90/95/99 % coverage.

Wall time: **~3 min**.

---

### test_tcrnet_benchmark.py

| Test | Result |
| --- | --- |
| test_tcrnet_benchmark_gil_like_motif_enrichment | 2 enriched motif hits, serial 0.010 s (tiny synthetic benchmark, 204 target vs 503 control) |
| test_tcrnet_runtime_gilg_vs_synthetic_1m | 5 236 target vs 1 M synthetic: serial 0.945 s, 4-job 6.406 s (overhead-dominated) |
| test_tcrnet_benchmark_b35_epl_connected_component_vs_real_control | elapsed 14.58 s (target 61 368, control 2 M), strict enriched 20 (vdjdb overlap 4), neighbor nodes 27, largest CC 21, components≥5: 1, component overlap 0.256 |

---

### test_control_benchmark.py

| Test | Result |
| --- | --- |
| Synthetic 10 k | Build 16.71 s, cache-hit 0.002 s, load 0.001 s |
| Synthetic 100 k | Build 164.61 s, cache-hit 0.007 s, load 0.006 s |
| Real human TRB (28.3 M rows) | Build 23.48 s, cache-hit 2.16 s, load 1.88 s, file 3.6 GB |
| Real mouse TRA (839 k rows) | Build 1.22 s, cache-hit 0.06 s, load 0.05 s |
| Cache-repeat (real, ×25) | Mean load 1.85 s, ensure overhead 0.989× (< 1 %) |

> Real TRB build dropped from 63.6 s → 24.5 s (2.6×) and cache-repeat mean dropped from 7.39 s → 1.93 s (3.8×) vs the 2025-05-10 reference run, likely due to Polars/IO optimisations merged since then.

---

### test_bag_of_kmers_benchmark.py

k = 3, human TRB, 2 M rows (sampled from 28.3 M real control).

| Path | Time | Details |
| --- | --- | --- |
| In-memory build | 36.26 s | 75.2 M total k-mers, dominant token "CAS" (position 0) |
| With cache write | 14.37 s | Cache warm-up amortised across subsequent calls |

> In-memory time dropped from 136.6 s → 36.69 s (3.7×) vs 2025-05-10 reference.

---

### test_alice_tcrnet_benchmark.py

ALICE (Pgen-based) vs TCRnet (neighbourhood enrichment) concordance on B35+ and CMV+ samples, 300 clonotypes, 100 k synthetic control.

| Test | Specs | Wall time |
| --- | --- | --- |
| test_alice_tcrnet_synthetic_hamming_concordance | 16 alice + 16 tcrnet (hamming, 4 match-modes) | ~180 s |
| test_tcrnet_synthetic_levenshtein_matrix | 16 tcrnet (levenshtein, 4 match-modes) | ~80 s |
| test_tcrnet_real_hamming_matrix | 16 tcrnet (hamming, real control 50 k) | ~49 s |
| test_tcrnet_real_levenshtein_matrix | 16 tcrnet (levenshtein, real control 50 k) | ~49 s |

ALICE 1mm runs take ~10–13 s each vs ~3 s for exact; 1mm speedup with 8 workers ≈ 6×.
TCRnet with real control: ≈ 1.3 s per run regardless of metric.

---

### test_alice_benchmark.py

#### test_alice_yf_notebook_cell6_scaling

Q1 donor repertoires (Q1_d0: 390 k clonotypes; Q1_d15: 477 k clonotypes).
Subsamples: 5 k / 10 k / 25 k. Workers: 1 and 8.

| Sample | Clonotypes | Workers | Total | pgen | Neighborhood |
| --- | --- | --- | --- | --- | --- |
| Q1_d0 | 5 000 | 1 | 30.5 s | 30.4 s | 0.04 s |
| Q1_d0 | 5 000 | 8 | 7.1 s | 4.1 s | 1.1 s |
| Q1_d0 | 10 000 | 1 | 60.0 s | 59.8 s | 0.07 s |
| Q1_d0 | 10 000 | 8 | 10.9 s | 7.9 s | 1.1 s |
| Q1_d0 | 25 000 | 1 | 147.1 s | 146.4 s | 0.23 s |
| Q1_d0 | 25 000 | 8 | 23.0 s | 19.8 s | 1.2 s |

Wall time for all 12 runs: **~610 s (~10 min)**.
Bottleneck is OLGA pgen computation (>99 % of wall time at 1 worker); neighbourhood search is negligible.
8-worker speedup: 4.2–6.4× depending on subsample size.

#### test_alice_pgen_10k_single_vs_parallel

10 k sequences, Q1_d0, exact pgen: single-thread 61.1 s → 8-thread 8.21 s → **7.44× speedup**.

---

### test_overlap_benchmark.py

#### Per-pair timing (A2-i132.txt.gz × itself, 9 632 clonotypes)

| Mode | n_jobs | n1_matched | Time | Peak |
| --- | --- | --- | --- | --- |
| exact:0 | 1 | 9 375 | 60.2 ms | 3 MB |
| exact:0 | −1 (auto) | 9 375 | 61.5 ms | 3 MB |
| hamming:1 | 1 | 9 375 | 193.6 ms | 3 MB |
| hamming:1 | −1 (auto) | 9 375 | 192.6 ms | 3 MB |
| levenshtein:1 | 1 | 9 375 | 354.7 ms | 3 MB |
| levenshtein:1 | −1 (auto) | 9 375 | 353.5 ms | 3 MB |

D (Dice) = F (F1) = 1.000 in all cases (self-overlap, expected).

#### Many-vs-many matrix (41 aging repertoires, 820 pairs)

| Mode | n_jobs | Total time | Peak RAM |
| --- | --- | --- | --- |
| exact:0 | −1 | 19.02 s | 111 MB |
| hamming:1 | −1 | 20.57 s | 111 MB |
| levenshtein:1 | −1 | 27.24 s | 111 MB |

#### Pilot benchmark (16 repertoires, 120 pairs, n_jobs=4)

Pilot estimate (28 pairs): 0.0077 s/pair → extrapolated 120-pair: 0.92 s.
Actual 120-pair run: 1.19 s (n_jobs_effective=4).

---

### test_overlap_execution_benchmark.py

Serial execution on real aging cohort (HuggingFace `isalgo/airr_benchmark`):

| Mode | Subset | Time | Per-pair | Extrapolated (79 donors) |
| --- | --- | --- | --- | --- |
| many-vs-many exact | 8 samples, 28 pairs | 25.37 s | 0.906 s/pair | **46.52 min** |
| many-vs-pool exact | 8 samples, 8 pool queries | 16.78 s | 2.10 s/sample | **165.65 s** |

Thread-parallel many-vs-many (n_jobs=4, same 28 pairs): 25.54 s (effective_jobs=4).

> At 0.90 s/pair with 79 donors (3 081 pairs), the full cohort takes ~46 min at n_jobs=1.
> With n_jobs=4 (thread pool, memory shared), the speedup is minimal for exact overlap at this pair size; the bottleneck is the C-level Hamming/Levenshtein scan, not Python overhead.

---

### test_metaclonotype_benchmark.py

Synthetic TRB repertoires; average cluster size 5; scales 1 K, 5 K, 10 K, 50 K clonotypes.

#### TestMetaclonotypeCreationBenchmark — `metaclonotypes_from_labels` and `metaclonotypes_from_components`

| n | n_clusters | t_labels (ms) | rss_labels (MB) | t_components (ms) | rss_components (MB) |
| --- | --- | --- | --- | --- | --- |
| 1 000 | ~150 | < 5 | < 1 | < 1 | < 1 |
| 5 000 | ~750 | < 10 | < 1 | < 2 | < 1 |
| 10 000 | ~1 500 | < 20 | < 1 | < 3 | < 1 |
| 50 000 | ~7 500 | < 80 | < 2 | < 10 | < 1 |

#### TestMetaclonotypeAnalyticsBenchmark — `summarize_metaclonotypes`, Hill curve, rarefaction

| n | n_clusters | t_summarize (ms) | t_diversity (ms) | t_hill (ms) | t_rarefy (ms) |
| --- | --- | --- | --- | --- | --- |
| 1 000 | ~150 | < 5 | < 1 | < 1 | < 5 |
| 5 000 | ~750 | < 10 | < 1 | < 1 | < 10 |
| 10 000 | ~1 500 | < 20 | < 1 | < 1 | < 15 |
| 50 000 | ~7 500 | < 80 | < 2 | < 2 | < 50 |

#### TestFunctionalDiversityEndToEndBenchmark — `functional_diversity()`

End-to-end: `LocusRepertoire` → `MetaClonotypeDefinition` → `DiversitySummary`.

| n | n_clusters | elapsed (ms) | rss (MB) | Shannon H | Chao1 |
| --- | --- | --- | --- | --- | --- |
| 1 000 | 193 | 3.5 | < 1 | 5.106 | 193 |
| 5 000 | 983 | 7.8 | < 1 | 6.741 | 983 |
| 10 000 | 1 974 | 13.8 | < 1 | 7.427 | 1 974 |
| 50 000 | 9 805 | 61.1 | 43 | 9.035 | 9 812 |

Wall time for all 3 benchmark classes: **~1.6 s** (Apple M3 reference hardware).

---

### test_tcremp_benchmark.py

#### Distance correlation

Human TRB, 1 000 prototypes; 1 000 query clonotypes (499 500 pairs):

| Metric | Value |
| --- | --- |
| Model build | 0.68 s, peak 2 MB |
| Embed 1k clones | 0.03 s, peak 31 MB, shape (1 000, 3 000) |
| Distance compute | 0.02 s |
| Pearson R² | 0.5723 |
| Spearman ρ | 0.7282 |

Per-component R² (V, J, CDR3, L2-emb): 0.4691 / 0.1633 / 0.5547 / 0.5723.

#### Throughput (n_jobs=1)

| n_clono | n_proto | Time | Clono/s | Peak |
| --- | --- | --- | --- | --- |
| 10 000 | 1 000 | 0.30 s | 33 276 | 306 MB |
| 100 000 | 1 000 | 3.16 s | 31 676 | 3 056 MB |
| 100 000 | 3 000 | 9.99 s | 10 007 | 9 160 MB |
| 500 000 | 1 000 | 18.18 s | 27 506 | 15 282 MB |
| 1 000 000 | 1 000 | 42.87 s | 23 326 | 30 565 MB |

#### Multiprocessing scaling

| n_clono | n_jobs=1 | n_jobs=2 | n_jobs=4 | n_jobs=8 |
| --- | --- | --- | --- | --- |
| 10 000 | 0.30 s | 0.19 s (1.54×) | 0.14 s (2.09×) | 0.12 s (2.60×) |
| 100 000 | 3.09 s | 2.07 s (1.49×) | 1.49 s (2.07×) | 1.22 s (2.54×) |
| 500 000 | 15.33 s | 10.55 s (1.45×) | 7.66 s (2.00×) | 6.22 s (2.46×) |

#### Key aligner measurements

| Benchmark | Result |
| --- | --- |
| Fixed-gap vs BioPython (10k × 3k) | Fixed-gap 0.22 s (137.2 M p/s) vs BioPython 83.78 s (358 k p/s) → **383×** speedup |
| Junction aligner (1M pairs) | Fixed-gap 49.4 M p/s vs full-DP 310 k p/s → **157×** speedup |
| Parallel chunking avg (16 cores) | 100: 1.04×, 1k: 2.33×, 5k: 2.79×, 10k: 2.91×, **avg 2.27×** |
| PCA V-gene correlation | Best PC1 p=0, var=16.1% |
| PCA J-gene correlation | Best PC5 p=0, var=1.23% |
| PCA junction-length correlation | PC0 r=+0.908, p=0, var=63.99% |
| Epitope specificity (GLC vs YLQ) | Within 9100 < Between 11447, p=9.99e-04, Cohen's d=0.72 |

---

### test_tcremp_analysis.py

All tests pass. Key measurements:

| Benchmark | Result |
| --- | --- |
| Fixed-gap vs BioPython (10k × 3k) | Fixed-gap 0.22 s (137.5 M p/s) vs BioPython 83.72 s (358.4 k p/s) → **383.7×** speedup |
| Embedding distance (cosine, 1k clones) | mean 0.265 ± 0.033; RMSE=570.3; Corr=0.147 (total seq distance) |
| Prototype symmetry (1k prototypes) | Max asymmetry 0.00, max diagonal 0.00; R²=1.000, RMSE=0.000 (latent ↔ sequence) |

---

### test_tcremp_paired_benchmark.py

| Benchmark | Result |
| --- | --- |
| Records | 2 000 records, 200 prototypes/chain |
| TRA single | 0.012 s (0.006 ms/record) |
| TRB single | 0.012 s (0.006 ms/record) |
| Paired | 0.026 s (0.013 ms/record) |
| Paired / (TRA+TRB) overhead | 1.061× |

---

### test_tcremp_vdjdb_benchmark.py

#### Single-chain VDJdb TRB (TestSingleChainVDJdbTCREmpQuality — passed)

3 000 records, 500 prototypes: **0.042 s**.
DBSCAN clustering: n_comp=27, eps=0.307, clusters=187, retention=0.636, purity=0.449.

#### Paired VDJdb (TestPairedVDJdbTCREmpQuality)

| Mode | n | Time | Purity | Retention |
| --- | --- | --- | --- | --- |
| Strict | 3 000 | 0.090 s | 0.621 | 0.568 |
| Imputed | 3 000 | 0.087 s | 0.487 | 0.672 |
| Paired / (TRA+TRB) overhead | — | — | — | 1.106× |

---

### test_single_cell_10x_benchmark.py

Loaded all dcode donors (390 641 clonotypes from `vdj_v1_hs_aggregated` series):

| Mode | Time | Clonotypes | Peak RSS |
| --- | --- | --- | --- |
| Sequential | 1.58 s | 390 641 | 4 125 MB |
| 2 workers (1k chunks) | 1.66 s | 390 641 | 4 178 MB |
| 4 workers (1k chunks) | 1.71 s | 390 641 | 4 187 MB |

Worker count scaling (separate sub-benchmark):

| Workers | Time | Speedup | Peak RSS |
| --- | --- | --- | --- |
| 1 | 2.05 s | — | 4 225 MB |
| 2 | 1.84 s | 1.1× | 4 222 MB |
| 4 | 1.79 s | 1.1× | 4 223 MB |

Chunk-size sweep: 500→2.49 s / 1k→2.11 s / 5k→1.68 s / 10k→1.76 s.

**Estimated for 1 M clonotypes:** sequential ~4.0 s / 10.6 GB; 4-worker ~1.0 s / 8.4 GB.

---

### test_single_cell_citeseq_benchmark.py

10x CITE-seq matrix loading from dcode donors:

| Donor | Cells | Pairs | Matrix rows | Binder cols | Time |
| --- | --- | --- | --- | --- | --- |
| donor1 | 47 271 | 48 890 | 46 526 | 78 | 1.15 s |
| donor2 | 79 704 | 72 266 | 77 854 | 78 | 1.99 s |
| donor3 | 38 095 | 39 518 | 37 824 | 78 | 0.89 s |
| donor4 | 27 640 | 29 147 | 27 308 | 78 | 0.97 s |

---

### test_single_cell_repair_benchmark.py

Imputation + cleanup on dcode data:
- impute: **16.15 s** (102 610 imputed rows from 95 663 raw)
- cleanup: **0.37 s** (100 250 cleaned rows)

---

### test_vdjbet_benchmark.py

#### Fast classes (≤ 55 s total)

| Class | Tests | Key result |
| --- | --- | --- |
| TestPgenBinPoolBenchmark | 3 | 78 781 seq/s @ 1 job; range [−70, −19] log₂ Pgen |
| TestVDJBetMockBenchmark | 2 | 50 mocks in 3.06 s; JSD ≈ 0 |
| TestPgenParallelBenchmark | 4 | **3.59× speedup** @ 4 workers; pool reuse 1.58× |
| TestLLWOverlapYFV | 3 | LLW reference 409 clonotypes; mock JSD ≈ 0 |

#### Slow classes (~97 s)

| Class | Tests | Key result |
| --- | --- | --- |
| TestYFVQ1F1 | 4 | Day-15 LLWNGPMAV enrichment z=∞; day-0 not; total 52 s |
| TestQ1Q15Integration | 4 | z day-15 = ∞ (n=3 exact hits); z day-0 = 4.36; d15 1mm n=13 |

#### Other classes (≤ 40 s)

| Class | Tests | Key result |
| --- | --- | --- |
| TestSyntheticVsRealMockComparison | 3 | Synthetic overlap ≈ 0; allele stripping verified |
| TestQ1ControlEffectSize | 2 | d15/d0 effect-size ratio ≥ 1; matching count ratio 3× |
| TestRepertoireIOPolars | 3 | Polars **8.82×** faster than pandas; **80 731×** lower peak memory |

---

### test_token_tables*.py

All tests pass. Key measurements:

| Benchmark | Result |
| --- | --- |
| tokenize (100k rearrangements, k=5) | 1.179 s (84 830 rearrangements/s) |
| lookup hits (1M ops) | 0.028 s (36.3 M ops/s) |
| lookup misses (1M ops) | 0.018 s (56.2 M ops/s) |
| summarize plain (100k, k=5) | 0.415 s (241 042 rearrangements/s) |
| summarize gapped (100k, k=5) | 1.981 s (50 471 rearrangements/s) |
| summarize_annotations plain (100k, k=5) | 0.632 s (158 214 rearrangements/s) |
| naive summarize (10k, k=3) | 0.315 s, 47 369 keys, peak 17 067 KiB |
| polars expand+summarize (10k, k=3) | 0.227 s, 47 369 rows, peak 18 KiB |
| naive summarize_annotations (10k, k=3) | 0.597 s, 6 221 KmerSeq, peak 30 557 KiB |
| polars 4 summaries (10k, k=3) | 0.239 s, peak 21 KiB |
| polars fetch_by_kmer 'CAS' (1k lookups) | 1.148 s (871 ops/s) |

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

**Time extrapolation:** from the small-matrix benchmark (n=10 k → 17 s; n=100 k → 167 s, ratio ≈ 9.8×/decade), a 1 M synthetic control follows a super-linear curve. Rough estimate: 167 s × 10 ≈ 28 min.

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
