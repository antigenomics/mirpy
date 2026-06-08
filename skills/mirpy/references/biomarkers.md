# Biomarker Analysis Reference (ALICE, TCRNET, GLIPH)

**Contents:** ALICE enrichment · TCRNET neighbourhood enrichment · GLIPH k-mer
enrichment · clonotype metadata associations · COVID-19 case studies (SVM,
HLA-stratified, TRA×TRB co-occurrence).

This reference covers mirpy's three primary biomarker enrichment modules: ALICE, TCRNET, and GLIPH-style k-mer analysis. These methods identify antigen-enriched T-cell receptor sequences by comparing neighborhood density in a study repertoire against a background model or control repertoire.

It also covers cohort-level clonotype association scans in
`mir.biomarkers.associations`, including sample/rearrangement count modes,
Fisher/chi-square tests, and depth-aware GLM mode (`test="depth_glm"`).

Relevant publications:

- **ALICE**: Pogorelyy et al. (2019) *PLoS Biology* — PMID:31194732, doi:[10.1371/journal.pbio.3000314](https://doi.org/10.1371/journal.pbio.3000314)
- **TCRNET / Neighbourhood enrichment**: Lupyr et al. (2025) *Brief. Bioinform.* — PMID:40996146, doi:[10.1093/bib/bbaf495](https://doi.org/10.1093/bib/bbaf495)
- **Antigen annotation framework**: Pogorelyy et al. (2019) *Front. Immunol.* — PMID:31616409, doi:[10.3389/fimmu.2019.02159](https://doi.org/10.3389/fimmu.2019.02159)
- **Pre-immune landscape**: Pogorelyy et al. (2018) *Genome Medicine* — PMID:30144804, doi:[10.1186/s13073-018-0577-7](https://doi.org/10.1186/s13073-018-0577-7)
- **VDJdb**: Shugay et al. (2018) *Nucleic Acids Res.* — PMID:28977646, doi:[10.1093/nar/gkx760](https://doi.org/10.1093/nar/gkx760)
- **Regulatory Treg repertoire**: Feng et al. (2015) *Nature* — PMID:26605529, doi:[10.1038/nature16141](https://doi.org/10.1038/nature16141)

---

## Clonotype Metadata Association (Whole-Cohort)

```python
from mir.biomarkers.associations import (
    AssociationParams,
    associate_clonotype_metadata,
    build_public_clonotype_panel,
)

targets = build_public_clonotype_panel(samples, locus="TRB", min_sample_fraction=0.03)

result = associate_clonotype_metadata(
    samples,
    targets,
    metadata_field="COVID_status",
    metadata_value=["COVID", "healthy"],
    params=AssociationParams(
        match_mode="none",
        count_mode="sample",         # or "rearrangement"
        test="auto",                 # "auto" | "fisher" | "chi2" | "depth_glm"
        correction_method="fdr_bh",
    ),
)
```

### Key behavior notes

- `count_mode="sample"`: per-sample presence/absence counts.
- `count_mode="rearrangement"`: depth-weighted counts by clonotype multiplicity.
- `test="auto"`: Fisher for binary labels, chi-square for multiclass labels.
- `test="depth_glm"`: logistic/binomial GLM with `log1p(depth)` covariate for
  binary labels; automatically falls back to table tests when unavailable.
- Output includes `detected_counts`, `background_counts`, `p_value`, `q_value`,
  `odds_ratio`, and `log2_odds_ratio`.

### Recommended COVID workflow

1. Keep only functional/productive clonotypes before counting.
2. Apply first-pass batch correction (see `mir.basic.gene_usage`).
3. Re-normalize per-sample totals after correction.
4. Run Fisher scan (`count_mode="sample"`) for a baseline ranking.
5. Run depth-aware scan (`count_mode="rearrangement", test="depth_glm"`) to
   quantify depth-related rank shifts.

---

## ALICE Enrichment

ALICE (Antigen-specific Lymphocyte Identification by Clustering and Enrichment) identifies receptor sequences that are more densely neighboured in CDR3 space than expected under the null recombination model (Pogorelyy et al. 2019, PLoS Biol., PMID:31194732). mirpy's implementation uses OLGA for generation probability estimates and supports multiple Pgen computation strategies including a Monte Carlo pool mode.

Use `compute_alice` / `add_alice_metadata` from `mir.biomarkers.alice`.

```python
from mir.biomarkers.alice import compute_alice, AliceParams, AliceResult

# Returns AliceResult(table=pd.DataFrame, params=AliceParams) when as_table=True
result = compute_alice(
    rep,
    species="human",
    match_mode="vj",      # "none" | "v" | "j" | "vj"
    pgen_mode="exact",    # "exact" | "1mm" | "mc"  — see notes below
    pvalue_mode="poisson",         # "poisson" | "negative-binomial"
    pseudocount=0.0,               # added to n and N before p-value computation
    min_neighbors=2,               # sequences with fewer neighbors get p_value=1.0
    q_factor=1.0,                  # thymic-selection correction multiplier (λ = N × pgen × Q)
    # MC mode options (only used when pgen_mode="mc"):
    mc_n_pool=10_000_000,          # synthetic pool size (built once, cached)
    mc_seed=42,
    mc_min_count=2,                # min pool matches to use MC pgen (else OLGA 1mm fallback)
    n_jobs=8,
)

# result.table columns:
#   sequence_id, locus, junction_aa, v_gene, j_gene,
#   n_neighbors, N_possible, pgen_raw, pgen,
#   expected_neighbors, fold_enrichment, p_value, q_value

# Filter at FDR < 0.05 (q_value is BH-corrected over all locus clonotypes)
hits = result.table.filter(pl.col("q_value") < 0.05)
```

### `pgen_mode` options

| Mode    | Speed           | Accuracy                          | Notes |
|---------|-----------------|-----------------------------------|-------|
| `exact` | Fast (7 ms/seq) | OLGA analytical exact Pgen        | Default; underestimates λ for ALICE — use `"1mm"` or `"mc"` |
| `1mm`   | Slow (70 ms/seq)| Sums Pgen over 1mm neighbors      | Best sensitivity; use skip_ends=2 (env `MIRPY_PGEN_1MM_SKIP_ENDS=2`) |
| `mc`    | Very fast after pool build | MC match counting + OLGA fallback | Pool built once, cached; 100–1000× faster than OLGA per sample |

### ALICE runtime scaling (TRB, `match_mode="vj"`, `min_neighbors=2`, `n_jobs=8`)

| Dataset size    | `"exact"` wall time | `"1mm"` wall time | `"mc"` wall time (after pool) |
|-----------------|---------------------|-------------------|-------------------------------|
| 1 K clonotypes  | < 1 s               | < 5 s             | < 1 s                         |
| 10 K clonotypes | 1–3 s               | 5–30 s            | < 1 s                         |
| 100 K clonotypes| 3–15 s              | 30–150 s          | 1–5 s                         |
| 1 M clonotypes  | 15–90 s             | 2–15 min          | 5–30 s                        |

First `"mc"` call builds the 10 M pool (37 s, 8 workers, human TRB); all subsequent samples reuse it from cache.

### Differences from the original paper

- Paper uses 100 M sequences; this implementation uses 10 M and falls back to OLGA analytical 1mm Pgen for sequences with < 2 pool matches. 100 M requires ~17 GB and ~16 min — use `mc_n_pool=100_000_000` only when that budget is available.
- Default is now `match_mode="vj"` (matching the paper). TCRNET default is `match_mode="none"`.
- **Gene-usage conditioning** (new): when `match_mode != "none"`, N and pgen are scaled by P_OLGA(V,J). Same logic for TCRNET: M replaced by P_ctrl(VJ) × M_total.
- Paper's exact pre-screen has been removed — it filtered 0 sequences in practice.

### Cluster analysis — `alice_hit_clusters`

```python
from mir.biomarkers.alice import alice_hit_clusters

# Default: cluster ALICE hits only (V-gene-restricted 1mm CDR3 edges)
hits_clustered = alice_hit_clusters(hits_df)

# Expand clusters with 1mm non-enriched neighbors from the full repertoire table
hits_expanded = alice_hit_clusters(hits_df, full_df=full_table, non_enriched_neighbors=True)
```

- **V-gene restriction**: edges only between sequences sharing the same V-gene family. Without this, transitive chains across different V families create one giant mixed component.
- `non_enriched_neighbors=True`: adds non-hit sequences from `full_df` that are 1mm (same V-gene) neighbors of any hit.
- Returns `hits_df` with `cluster_id` (int) and `is_hit` (bool) columns.

**Important**: Always build connected components per (V, J, length) group for motif discovery. Building on all sequences creates one giant mixed component that dilutes motif signal.

### Key behavior notes

- `min_neighbors=2` requires a sequence with at least 1 Hamming-1 neighbour. Isolated sequences get `p_value=1.0` without OLGA computation.
- `q_factor` multiplies λ = N × pgen × Q. Calibrate from real data: Q ≈ median(pgen_real / pgen_olga). See Pogorelyy et al. (2018) *Genome Medicine* (PMID:30144804) for pre-immune landscape context relevant to calibration.
- `q_value` is BH-corrected over all clonotypes in the locus.
- For multi-sample workflows, reuse one OlgaModel instance across all samples.

---

## TCRNET Enrichment

TCRNET is a **purely MC-control** enrichment algorithm — no OLGA Pgen is computed at any stage. It compares neighbourhood density in the study repertoire against an explicit control repertoire (real donors or a synthetic pool). The method and its statistical framework are described in Lupyr et al. (2025) *Brief. Bioinform.* (PMID:40996146).

```python
from mir.biomarkers.tcrnet import compute_tcrnet, TcrnetParams, TcrnetResult

result = compute_tcrnet(
    rep,
    control=control_rep,               # explicit LocusRepertoire / SampleRepertoire
    # control_type="real",             # alternative: load managed control
    species="human",
    metric="hamming",                  # "hamming" | "levenshtein"
    threshold=1,                       # 0 (exact) or 1
    match_mode="vj",                   # "none" | "v" | "j" | "vj"
    pvalue_mode="binomial",            # "binomial" | "beta-binomial"
    pseudocount=1.0,                   # added to control m and M (Laplace smoothing)
    q_factor=1.0,                      # selection correction for synthetic controls
    normalize_control_vj_usage=False,  # resample control to match sample V/J usage
    n_jobs=-1,
)

# result.table columns:
#   sequence_id, locus, junction_aa, v_gene, j_gene,
#   n_neighbors, N_possible,
#   m_control_neighbors (raw), M_control_possible,
#   sample_density, control_density (q-adjusted),
#   fold_enrichment, p_value, q_value

# Filter at FDR < 0.001 (paper-correct threshold)
hits = result.table[result.table["q_value"] < 0.001]
```

### Key behavior notes

- p-value: P(X >= n) where X ~ Binomial(N, q_factor × (m+pc)/(M+pc)). `beta-binomial` uses BetaBinom(N, alpha=(m+pc)×q_factor, beta=(M-m)+pc) — q_factor scales the alpha (effective successes) the same way it scales the binomial numerator. If alpha ≥ M (degenerate: effective density ≥ 1), p_value = 1.0.
- Raw m and M (including pseudocount) are stored in metadata; q_factor is applied only when computing p_value and fold_enrichment.
- q_value is BH-corrected over all clonotypes in the locus.

### `q_factor` — selection correction for synthetic controls

OLGA synthetic sequences are drawn from the recombination model (pre-thymic selection). Their neighborhood density is systematically lower than a post-selection real repertoire by a factor Q ≈ 3–5 for human TRB. The pre-immune repertoire landscape underlying this correction is characterized in Pogorelyy et al. (2018) *Genome Medicine* (PMID:30144804).

Estimate Q from a real control sample:

```python
from mir.basic.pgen import McPgenPool, OlgaModel
import numpy as np

model = OlgaModel(locus="TRB", species="human")
olga_pgens = model.compute_pgen_junction_aa_bulk(test_seqs, max_mismatches=0, n_jobs=8)
real_pool  = McPgenPool.build_real(control_seqs, locus="TRB")
real_pgens = real_pool.pgen_1mm_bulk(test_seqs, n_jobs=8)
q_samples  = [rp / op for rp, op in zip(real_pgens, olga_pgens) if rp > 0 and op > 0]
Q = float(np.median(q_samples))   # typical value: 3–5 for human TRB
```

Leave `q_factor=1.0` (default) for real controls.

### TCRNET as original ALICE (V+J+1mm, 100 M pool)

```python
from mir.basic.pgen import McPgenPool
pool = McPgenPool.build_synthetic(100_000_000, locus="TRB", n_jobs=8)
result = compute_tcrnet(
    rep,
    control=LocusRepertoire([Clonotype(...) for s in pool._unique_seqs], locus="TRB"),
    match_mode="vj",
    pvalue_mode="binomial",
    q_factor=Q,
)
```

Statistically equivalent to `compute_alice(rep, pgen_mode="mc", mc_n_pool=100_000_000, match_mode="vj", q_factor=Q)`.

### TCRNET vs ALICE summary

|                     | TCRNET                                  | ALICE                                      |
|---------------------|-----------------------------------------|--------------------------------------------|
| Background          | Any MC control (real or synthetic)      | OLGA Pgen (MC pool or analytical)          |
| V/J bias            | Captured via real control               | Via `match_mode` parameter                 |
| Pgen calls          | None                                    | OLGA 1mm Pgen (or 10 M MC approximation)  |
| Statistics          | Binomial / Beta-Binomial                | Poisson                                    |
| Selection correction| `q_factor` (explicit)                   | `q_factor` (explicit)                      |
| Default control     | Must be provided explicitly             | Synthetic OLGA pool                        |

---

## GLIPH-Style K-mer Enrichment

GLIPH (Grouping of Lymphocyte Interactions by Paratope Hotspots) identifies short CDR3 motifs that are statistically over-represented in a study repertoire relative to a control. The antigen annotation framework that motivates motif-based grouping is described in Pogorelyy et al. (2019) *Front. Immunol.* (PMID:31616409); VDJdb (Shugay et al. (2018) *Nucleic Acids Res.* PMID:28977646) provides the curated reference database against which enriched motifs can be annotated.

The GLIPH algorithm is described in Glanville *et al.* (2017) *Nature* 547:94–98 (PMID:[28636589](https://pubmed.ncbi.nlm.nih.gov/28636589/), doi:[10.1038/nature22976](https://doi.org/10.1038/nature22976)); the improved GLIPH2 algorithm scaling to millions of TCRs is in Huang *et al.* (2020) *Nat. Biotechnol.* 38:1194–1202 (PMID:[32341563](https://pubmed.ncbi.nlm.nih.gov/32341563/), doi:[10.1038/s41587-020-0505-4](https://doi.org/10.1038/s41587-020-0505-4)).

```python
from mir.biomarkers.gliph import (
    compare_gliph_token_incidence,
    extract_gliph_artifacts_batch_from_repertoire,
)

families = ["v3", "pos3", "u3", "u4", "g4", "g5"]

# Compute control artifacts once (counts only, memory-safe chunking)
ctrl_artifacts = extract_gliph_artifacts_batch_from_repertoire(
    control_repertoire,
    families,
    count_mode="clonotype",
    build_mappings=False,
    trim_first=3,
    trim_last=4,
    chunk_size=200_000,
)

# Reuse for each study with identical trim settings
study_artifacts = extract_gliph_artifacts_batch_from_repertoire(
    study_repertoire, families,
    count_mode="clonotype", build_mappings=False,
    trim_first=3, trim_last=4, chunk_size=200_000,
)

comp = compare_gliph_token_incidence(
    study_artifacts["u3"], ctrl_artifacts["u3"],
    test="binom", p_adj_method="fdr_bh", pseudocount=1,
)
sig = (comp["p_val_adj"] < 0.05) & (comp["freq_fc"] > 1.0)
```

### Motif family descriptions

| Family | Description |
|--------|-------------|
| `v3`   | V-gene anchor 3-mers (first 3 positions) |
| `pos3` | Position-specific 3-mers |
| `u3`   | Unanchored (variable-position) 3-mers — most informative for motif discovery |
| `u4`   | Unanchored 4-mers |
| `g4`   | Global 4-mers (all reading frames) |
| `g5`   | Global 5-mers |

### Interpretation notes

- In `test="binom"`, `p_background` = count_2 / total_control_clonotypes.
- p-value is one-sided enrichment: P[X >= count_1] under Binomial(total_sample_clonotypes, p_background).
- For large real controls, use `build_mappings=False` plus `chunk_size` for streaming.
- Keep `trim_first`/`trim_last` the same for sample and control; GLIPH defaults are `trim_first=3`, `trim_last=4`.

---

## COVID-19 case studies (notebook companions)

Detailed result tables for the covid19 biomarker notebooks. The SKILL.md keeps
only the API entry points; the specifics live here.

### SVM classifier (Vlasova 2026 replication) — `covid19_biomarkers.ipynb`

Log-frequency features + RBF-SVM over 1137 paired donors, AUC≈0.70 target.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# X: (n_donors, n_biomarkers) log-frequency matrix; y: binary (1=COVID, 0=healthy)
X_log = np.log(X + 1e-7)
clf = SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(clf, X_log, y, cv=cv, method="predict_proba")[:, 1]
auc = roc_auc_score(y, y_prob)  # target >= 0.70
```

See `benchmarks/covid19_svm_benchmark.py` and
`tests/test_associations_covid19_benchmark.py::test_covid19_svm_classifier_auc`.

> Vlasova *et al.* (2026) *Genome Med.* DOI:10.1186/s13073-025-01589-4

### HLA-stratified analysis — `covid19_hla_biomarkers.ipynb`

HLA class II association for 1137 paired donors (761 COVID / 376 healthy). Public
biomarkers from the global Fisher scan are re-tested within HLA-stratified
sub-cohorts (DRB1\*16: n=76, DQB1\*05: n=352). A focused pre-specified correction
(TRBV12-3/CASS set, 1297 candidates) replicates the Vlasova 2026 finding that
global multiple-testing is too conservative for rare allele strata.

- Top DRB1\*16 hit: `CASSRTGTGSSYNSPLHF` (TRBV12-3), 26 COVID / 0 healthy,
  log₂FE = 4.38, FDR = 0.035 (focused BH within TRBV12-3/CASS set).
- Global HLA × CDR3 scan (83 alleles × 43 CDR3s): one significant pair —
  `CAGQLYGGSQGNLIF` depleted in HLA-DPB1\*02:01 carriers (log₂FE = −1.51, q = 0.003).

```python
# Per-donor CDR3 presence scan pattern
donor_cdr3_presence = {}
for donor_id, row in metadata.iterrows():
    df = pd.read_csv(data_root / row['file_name'], sep='\t', usecols=['cdr3aa', 'v'])
    donor_cdr3_presence[donor_id] = set(zip(df['cdr3aa'], df['v'].str.split('*').str[0]))
```

### TRA × TRB co-occurrence — `covid19_pairing_biomarkers.ipynb`

Paired-chain co-occurrence Fisher test (156 TRA×TRB pairs across 3 strata,
`scipy.stats.fisher_exact` + BH FDR per stratum) with VDJdb cross-validation.

| Stratum | Significant pairs | Direction |
|---------|-------------------|-----------|
| All (n=1137) | 1 | Negative: CALSEETSGSRLTF × CASSLGGGDTQYF (q=0.027) |
| COVID (n=761) | 0 | — |
| Healthy (n=376) | 2 | CAGQNYGGSQGNLIF co-occurs with CASSLGETQYF (q=0.001), CASSPSTDTQYF (q=0.013) |

VDJdb overlap (hamming ≤ 1, V-gene fixed): 3/4 TRA CDR3s and 15/39 TRB CDR3s
validated against SARS-CoV-2 records; 0 pairs co-matched in one record (sparse
paired coverage).

```python
def hamming(a, b):
    if len(a) != len(b):
        return len(a) + len(b)  # length mismatch -> disqualify
    return sum(x != y for x, y in zip(a, b))
```
