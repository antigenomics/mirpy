# Benchmark results — regression baseline

Recorded results for the sample-level (repertoire) embedding benchmarks (`experiments/benchmark_repertoire_*.py`),
so a re-run can be checked for regression. All numbers are **repeated 50-fold CV mean±std** unless noted, on
arda-native coordinates. Theory/interpretation: `THEORY.md` T7. Data provenance: `SOURCES.md`.

- **Env**: conda `mirpy` (Python 3.12), Apple M3 (32 GB). Seeds fixed (`seed=0`). `RUN_BENCHMARK` not required
  for these (they are experiment scripts, not the guarded pytest tier).
- **Recorded**: 2026-07-14. Re-run and compare; a drop beyond ~1 std on the headline metric is a regression.
- **Reproducibility**: all load **local data first, HF fallback** — `aging`/`airr_hip` via `_hf.fetch`
  (`hf_hub_download`), `airr_covid19` via `_covid.covid_path` (local `~/hf/airr_covid19` else `isalgo/airr_covid19`).

## Sample-level embedding — established regime (airr_benchmark aging, airr_hip)

| Benchmark | Command | Headline metric | Value |
|---|---|---|---|
| Depth-robustness (`prop:kme`) | `benchmark_repertoire_depth.py` | log–log slope `‖ΔΦ₁‖` vs `n_eff` (theory −0.5) | **−0.55** |
| Age regression (n=79) | `benchmark_repertoire_aging.py` | \|Spearman(age,pred)\|: diversity / kernel-mean / k-mer | **0.76** / 0.58 / 0.28 |
| CMV serostatus (n=240) | `benchmark_repertoire_cmvhla.py` | AUC: diversity / Φ blocks / learned | **0.83±0.05** / 0.59–0.63 / 0.49 |
| CMV age-matched control | `benchmark_repertoire_cmvhla.py` | AUC: diversity / age-only (decade-matched) | **0.83** / 0.45 |
| HLA-A\*02 (n=500, 25k) | `benchmark_repertoire_hla.py 250 25000` | AUC: second-moment / diversity | **0.623±0.048** / 0.460±0.048 |
| HLA panel (airr_hip, n=300) | `benchmark_repertoire_hla*panel` | second-moment AUC: B\*08 / B\*07 / B\*44 | **0.82** / 0.76 / 0.66 |
| Spike-in recovery | `benchmark_repertoire_spikein.py` | recall @N≤3k / FPR (clean P_gen null) | **~35–50%** / ~1.2% |

## Aging divergence — `benchmark_repertoire_agediverge.py 500000 30` (58 donors @500k)

Divergence rises with age at depth but is **diversity-coupled, not an independent axis**.

| Metric | Value | Note |
|---|---|---|
| ρ(¹D, divergence) biased KME | −0.153 | biased V-stat `1/n_eff` artifact |
| ρ(¹D, divergence) **unbiased** KME | **−0.052** | `mmd_matrix(unbiased=True)`; artifact removed |
| ρ(¹D, divergence) overlap −logF | −0.680 | clonal-expansion driven |
| ρ(age, divergence overlap F) | 0.704 | (0.24 at 250k — depth-dependent) |
| **partial ρ(age, divergence \| ¹D)** | **0.067** (p=0.64) | n.s. — no independent axis (same at 40k/250k) |

## COVID cohort (airr_covid19, 300 donors, ≤20k reads)

### Batch cookbook — `benchmark_repertoire_covidbatch.py 300 20000` → **PASS**

| signal | naive AUC | batch-residualized |
|---|---|---|
| batch identity (OvR) | 0.777 | **0.032** |
| COVID status (⟂̸ batch) | **0.660** | 0.409 |
| HLA-A\*02 (⟂ batch) | 0.596 | **0.609** |

within-mixed-batch COVID (honest) **0.539**; MMD offset:biology ratio **1.05**.

### HLA imprint class I+II — `benchmark_repertoire_covidhla.py 300 20000` (TRB)

second-moment > diversity in direction **15/17** alleles; class-II present **8/9**.

| allele | class | 2nd-moment AUC | Δ vs diversity |
|---|---|---|---|
| DRB1\*07:01 | II | **0.758** | +0.204 |
| B\*07:02 | I | 0.702 | +0.154 |
| DRB1\*15:01 | II | 0.702 | +0.110 |
| DQB1\*05:01 | II | 0.619 | +0.152 |
| A\*02:01 | I | 0.628 | +0.035 |

### Paired α+β — `benchmark_repertoire_covidpaired.py 300 20000`

**TRA carries the stronger HLA imprint**; paired concat sits between (noisier β dilutes).

| allele | β (TRB) | α (TRA) | α+β paired |
|---|---|---|---|
| A\*02 | 0.538 | **0.636** | 0.612 |
| B\*07 | 0.646 | **0.701** | 0.664 |
| DRB1\*15 | 0.749 | **0.810** | 0.796 |

### COVID biomarker — honest **negative** (`covidstatus.py` / `covidpaired.py`)

| test | β | α | paired |
|---|---|---|---|
| COVID status, naive | 0.673 | 0.685 | **0.702** |
| COVID status, batch-resid | 0.488 | 0.512 | **0.522** (chance) |
| witness rediscovery of paper's clones | 0.373 | **0.450** | — |

Convalescent (long-past) COVID leaves **no batch-robust bulk clonotype-identity signal** at RNA-seq depth; the
naive 0.66–0.72 was batch confound. Ground truth is 87% α (4393 α vs 567 β).
