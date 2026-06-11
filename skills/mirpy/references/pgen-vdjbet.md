# Pgen and VDJBet Reference

This reference covers the mirpy Pgen estimation and VDJBet overlap analysis workflows, including the OlgaModel API, Monte Carlo Pgen pool, V/J/VJ usage from model marginals, and VDJBet scoring. Key publications:

- Pre-immune TCR landscape: Pogorelyy et al. (2018) Genome Medicine PMID:30144804 doi:10.1186/s13073-018-0577-7
- Neighbourhood enrichment: Lupyr et al. (2025) Brief. Bioinform. PMID:40996146 doi:10.1093/bib/bbaf495

## OlgaModel API

```python
from mir.basic.pgen import OlgaModel

model = OlgaModel(locus="TRB", seed=42)

# Bulk exact Pgen (~1000 seqs/s)
pgens = model.compute_pgen_junction_aa_bulk(seqs, max_mismatches=0, n_jobs=8)

# 1mm Pgen (~90 seqs/s) — uses OLGA's vectorized 1-mismatch path
pgen_1mm = model.compute_pgen_junction_aa_1mm("CASSIRSSYEQYF")

# Generation (~270K seqs/s) — returns (seqs, n_total_rearrangements)
seqs, n_total = model.generate_sequences_counted(10_000_000, n_jobs=8, seed=42)

# Always close the model to release the persistent worker pool
with OlgaModel(locus="TRB", species="human") as m:
    pgens = m.compute_pgen_junction_aa_bulk(seqs, n_jobs=8)
```

Performance notes:
- A **persistent `multiprocessing.Pool`** loads the OLGA model once per worker; reused across all `compute_pgen_junction_aa_bulk` calls on the same instance (zero spawn overhead for repeated calls on 12+ samples).
- `generate_sequences_counted(n, n_jobs, seed)` returns `(seqs, n_total_rearrangements)` for MC Pgen denominator calibration.
- `OlgaModel.gen_model` exposes the underlying `GenerativeModelVDJ/VJ` for direct model marginals.

## McPgenPool API

`McPgenPool` estimates Pgen by counting exact and 1mm matches in a large synthetic (or real) control pool.

**`p_productive` calibration constants** (stored in `_P_PRODUCTIVE_TABLE`):

| Locus | human | mouse |
|-------|-------|-------|
| TRA | 0.2891 | 0.3147 |
| TRB | 0.2441 | 0.2704 |
| TRG | 0.2709 | — |
| TRD | 0.2572 | — |
| IGH | 0.1281 | — |
| IGK | 0.2798 | — |
| IGL | 0.2917 | — |
| other | 0.20 (generic fallback) | |

```python
from mir.basic.pgen import (
    McPgenPool, get_or_build_mc_pool, clear_mc_pool_cache, get_p_productive,
)

# Look up calibrated p_productive for a locus/species
p = get_p_productive("TRB", "human")   # → 0.2441

# Build a synthetic pool
pool = McPgenPool.build_synthetic(
    10_000_000, locus="TRB", species="human", n_jobs=8, seed=42, skip_ends=2,
)
# pool.n_productive   = M (productive sequences)
# pool.n_total        = M + K (all rearrangement attempts)
# pool.p_productive   = M / n_total

# Bulk Pgen estimation
pgens_exact = pool.pgen_exact_bulk(cdr3_list)
pgens_1mm   = pool.pgen_1mm_bulk(cdr3_list, n_jobs=8)

# Build from real repertoire (for Q-factor analysis)
real_pool  = McPgenPool.build_real(real_cdr3_list, locus="TRB", species="human")

# Session-level cache (same pool reused across ALICE samples)
pool = get_or_build_mc_pool(locus="TRB", species="human", n=10_000_000, seed=42)
clear_mc_pool_cache()  # release memory
```

**MC Pgen normalisation:**
OLGA analytical Pgen is defined over ALL rearrangements (productive + non-productive).
`pgen_mc = n_matches / (M + K)` uses the tracked total to match the OLGA denominator.

## V/J/VJ Usage From OLGA Model (Analytical, Instant)

```python
from mir.basic.pgen import OlgaModel
from mir.basic.gene_usage import get_gene_usage_from_olga_model

m = OlgaModel(locus="TRB", species="human")
probs = get_gene_usage_from_olga_model(m)
# probs["v"]  — {gene_name: P(V)}
# probs["j"]  — {gene_name: P(J)}
# probs["vj"] — {(v_call, j_call): P(V,J)}
```

Reads IGoR model marginals directly. Probabilities are aggregated under the major-allele key
(e.g., all `TRBV5-1*02` mass folds into `TRBV5-1*01`).

## VDJBet Workflow

```python
from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.comparative.vdjbet import PgenBinPool, VDJBetOverlapAnalysis

model      = OlgaModel(locus="TRB", seed=42)
target_gu  = GeneUsage.from_repertoire(rep)
adjustment = PgenGeneUsageAdjustment(target_gu, seed=42)
pool       = PgenBinPool("TRB", n=100_000, n_jobs=-1, seed=42, pgen_adjustment=adjustment)
analysis   = VDJBetOverlapAnalysis(reference_rep, pool=pool, n_mocks=200, seed=42)
result     = analysis.score(query_rep, match_v=True, match_j=True)
```

## VDJBet High-Level Helpers

```python
from mir.comparative.vdjbet_workflow import (
    build_real_control_analysis,
    build_synthetic_comparison,
    compute_bin_alignment_diagnostics,
    compute_olga_usage_adjustment,
    load_yfv_trb_samples,
    score_samples_dataframe,
)

samples, yfv_gu = load_yfv_trb_samples(yfv_dir)
usage = compute_olga_usage_adjustment(yfv_gu, seed=42, olga_usage_n=1_000_000, n_jobs=8,
                                       count_mode="count_rearrangement", pseudocount=1.0)

real = build_real_control_analysis(
    reference_rep, yfv_gu, seed=42, count_mode="count_rearrangement",
    pseudocount=1.0, pool_size=100_000, n_mocks=100, n_jobs=8,
)

diag   = compute_bin_alignment_diagnostics(real.analysis)
df_res = score_samples_dataframe(real.analysis, samples)

pool_s, analysis_s, df_synth, x_scale, df_synth_scaled = build_synthetic_comparison(
    reference_rep, samples, pgen_adj_olga=usage.pgen_adj_olga,
    pool_size=100_000, n_mocks=100, n_jobs=8, seed=42, df_res_real=df_res,
)
```

Recommended defaults for reproducible runs:
- `seed=42`
- `pool_size=100_000` for notebook iteration; `1_000_000` for final analyses
- `n_mocks=100` for exploratory runs; `200+` for stable tail p-values
