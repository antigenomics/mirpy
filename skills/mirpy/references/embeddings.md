# TCREmp Embedding Reference

TCREmp is a bioinformatic pipeline for efficient embedding of T-cell receptor sequences as
distance vectors to a fixed set of prototypes. The method is described in:

> Kremlyakova et al. (2025) "TCREMP: A Bioinformatic Pipeline for Efficient Embedding of T-cell
> Receptor Sequences." *J. Mol. Biol.* PMID:40368275 doi:10.1016/j.jmb.2025.169205

Paired TRA/TRB workflow examples use VDJdb data from:

> Shugay et al. (2018) VDJdb. PMID:28977646

---

## Feature modes: `vjcdr3` vs `cdr123`

`TCREmp.from_defaults(..., mode=...)` (also `from_file` and `PairedTCREmp.from_defaults`)
selects the per-prototype feature triple:

- `"vjcdr3"` (default) — `[V-gene, J-gene, CDR3/junction]` distances.
- `"cdr123"` — `[CDR1, CDR2, CDR3/junction]`. CDR1/CDR2 are germline
  V-gene-determined, precomputed from the bundled region annotations via
  `GermlineAligner.from_library_region`; both are looked up by the clonotype's
  `v_call`. Requires `region_annotations.txt` (raises otherwise).

Output is `(N, 3*K)` float32 in both modes; only the first two components of each
triple change. Prototype genes absent from the aligner fall back to the max
region distance, never NaN. Benchmark: `notebooks/tcremp_features_compare.ipynb`.
See `references/region-annotation.md` for how the region annotations are built.

---

## Prototype-Based Embeddings With TCREMP

Use `TCREmp` from `mir.embedding.tcremp` to embed clonotypes as distance vectors
to a fixed set of prototypes.

```python
from mir.embedding.tcremp import TCREmp
from mir.common.clonotype import Clonotype

# Build from defaults: fast fixed-gap junction alignment (C-accelerated, ~25M pairs/s)
model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000, junction_method="fixed_gap")

# Embed clonotypes
clonotypes = [
    Clonotype(v_call="TRBV10-3*01", j_call="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF"),
    Clonotype(v_call="TRBV20-1*01", j_call="TRBJ1-1*01", junction_aa="CSARDSSYEQYF"),
]
X = model.embed(clonotypes, n_jobs=4)  # shape: (2, 3000), dtype: float32

# For full DP semantics (slower, ~270k pairs/s):
model_bio = TCREmp.from_defaults("human", "TRB", n_prototypes=100, junction_method="biopython")

# For custom prototypes:
model_custom = TCREmp.from_file("my_prototypes.tsv", species="human", locus="TRB")
```

**Embedding structure**: Each clonotype is embedded as `[v_1, j_1, junc_1, v_2, j_2, junc_2, ..., v_K, j_K, junc_K]`
where K is the number of prototypes. Distance formula: `d(a, b) = s(a,a) + s(b,b) - 2 * s(a,b)`

Useful properties:
- `model.n_prototypes` — number of prototypes (K)
- `model.embedding_dim` — total vector length (3·K)
- `model.locus`, `model.species` — canonical identifiers
- `model.prototypes` — Polars DataFrame with columns `v_call`, `j_call`, `junction_aa`

**n_jobs auto-selection:**
- `n_jobs=None` (default): auto-switch based on `len(clonotypes) * n_prototypes` between serial (1) and `os.cpu_count()`
- `n_jobs=1`: force serial processing
- `n_jobs>1`: force explicit worker count
- In auto mode, BioPython backend stays serial (thread overhead usually dominates)
- Use `n_jobs=1` (default) on macOS/ARM — multiprocessing with spawn is slower at all practical batch sizes

**Performance** (Apple M3, human TRB, K=1000 prototypes):

| Config             | Throughput          | Notes |
|--------------------|---------------------|-------|
| `n_jobs=1` (C)     | ~25 000 clono/s     | default, optimal on macOS |
| `n_jobs=8` (spawn) | ~10 000 clono/s     | spawn overhead dominates |

**Embedding quality** (R² between sequence-space and latent-space distances, 1000×1000):

| Correlation metric | Value |
|--------------------|-------|
| Pearson R²         | 0.57  |
| Spearman ρ         | 0.73  |

Per-component R²: V=0.47, J=0.16, CDR3=0.55. CDR3 variability is the strongest predictor.

**Backend choice guidance:**
- Use `junction_method="fixed_gap"` for production (stable behavior, ~90× faster than biopython).
- Use `junction_method="biopython"` when full DP alignment semantics are needed.
- Current benchmarks do not show consistent downstream quality improvement from BioPython.

**Backward compatibility:**
- `cdr3_aligner` property and `_proto_cdr3` attribute remain as aliases to `junction_aligner` and `_proto_junction`.
- Existing pickled models with `CDRAligner` unpickle without modification.

---

## Paired TRA/TRB Embeddings From VDJdb Full

```python
from mir.common.parser import VDJdbFullPairedParser
from mir.common.single_cell import build_tenx_sample_from_cell_clonotypes
from mir.common.single_cell_repair import impute_missing_chains
from mir.embedding.tcremp import PairedTCREmp

parser = VDJdbFullPairedParser()

# Strict paired mode.
strict_df, strict_meta = parser.parse_cell_clonotypes_file(
  "tests/assets/vdjdb_full.txt.gz",
  species="HomoSapiens",
  include_incomplete=False,
)
strict_sample = build_tenx_sample_from_cell_clonotypes(
  strict_df, sample_id="vdjdb_full_human_strict", barcode_metadata=strict_meta,
)

# Imputation mode for single-chain rows.
impute_df, impute_meta = parser.parse_cell_clonotypes_file(
  "tests/assets/vdjdb_full.txt.gz",
  species="HomoSapiens",
  include_incomplete=True,
)
imputed_df = impute_missing_chains(impute_df)
imputed_sample = build_tenx_sample_from_cell_clonotypes(
  imputed_df, sample_id="vdjdb_full_human_imputed", barcode_metadata=impute_meta,
)

paired_model = PairedTCREmp.from_defaults(
  species="human", locus_pair="TRA_TRB", n_prototypes=500,
)
paired_clonotypes = imputed_sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
X_pair = paired_model.embed(paired_clonotypes)
```

Operational notes:
- The paired embedding dimension is the sum of the two chain embedding dimensions.
- `parse_cell_clonotypes_file(..., include_incomplete=True)` returns synthetic single-cell style rows so you can run `impute_missing_chains` before building paired repertoires.
- Each synthetic barcode stores `vdjdb_record_id`, `mhc.a`, `mhc.b`, `mhc.class`, `antigen.epitope`, `antigen.gene`, and `antigen.species` in `SingleCellRepertoire.barcode_metadata`.
- For tabular metadata workflows, use `SingleCellRepertoire.metadata_to_polars()`.

---

## Embedding Diagnostics — DBSCAN Clustering

```python
from mir.utils.embedding_diagnostics import (
    analyze_embedding_dbscan,
    majority_vote_cluster_predictions,
    classification_scores_by_label,
)

result = analyze_embedding_dbscan(X_raw, labels, seed=42)

# Key output fields:
#   n_comp      — PCA components retained for 90% variance
#   eps         — selected DBSCAN epsilon
#   n_clusters  — number of clusters found (excl. noise)
#   retention   — fraction of non-noise points
#   purity      — mean per-cluster purity
#   consistency — fraction of clusters with >70% single label
#   median_4nn  — median 4-NN distance (scale reference)
#   kth         — full sorted k-NN distance curve (numpy array)
#   knee_idx    — index in kth corresponding to selected eps
#   X_pca       — L2-normalised PCA embedding used for DBSCAN
#   clusters    — DBSCAN cluster labels (−1 = noise)

predicted = majority_vote_cluster_predictions(labels, result["clusters"])
scores = classification_scores_by_label(labels, predicted)
# scores keys: accuracy, macro_f1, weighted_f1, per_label (list of dicts)
```

**Eps selection — `select_eps_kneedle_stable` algorithm:**

1. Compute the sorted 4-NN distance curve `kth` of the L2-normalised PCA embedding.
2. Set `eps = kth[q_floor]` (default `q_floor=0.40`) as the safe minimum.
3. Run `KneeLocator` on the narrow window `[q_floor, q_floor + knee_fraction*(q_cap−q_floor)]` (default: ≈[0.40, 0.45]). Accept the knee only if it falls strictly within that window.
4. For flat k-NN curves (typical in TCREmp embeddings), no knee is found; the selection stays at the floor quantile.

The floor `q_floor=0.40` was cross-validated on five balanced VDJdb TRB epitope subsets (n≈3 000 each): it gives the highest minimum quality margin and keeps retention ≈ 0.62, purity ≈ 0.48.

**Key parameters for `analyze_embedding_dbscan`:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `pca_variance_threshold` | `0.90` | Cumulative variance retained |
| `q_floor` | `0.40` | Lower quantile bound for eps (empirically validated) |
| `q_cap` | `0.65` | Upper quantile cap |
| `knee_fraction` | `0.20` | Fraction of `[q_floor, q_cap]` to search for a knee |
| `min_samples` | `3` | DBSCAN min_samples |
| `eps_selection_mode` | `"stable_kneedle"` | Use `"kneedle"` for legacy full-range mode |

**Dimensionality scaling**: The quantile approach is inherently dimensionality-adaptive — absolute eps grows with n_comp, but the quantile boundary stays constant. No n_comp correction is needed.
