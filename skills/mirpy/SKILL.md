# mirpy Agentic Skills

This guide summarizes stable, reusable workflows in mirpy.
Examples below use the public APIs that match the current source tree, tests,
and documentation.

## Core Objects

- `Clonotype`: one rearrangement with junction sequence, V/D/J calls, and counts.
- `LocusRepertoire`: one locus worth of clonotypes for a single sample.
- `SampleRepertoire`: a sample that may contain multiple loci.
- `RepertoireDataset`: a collection of `SampleRepertoire` objects plus metadata.

## 1. Parse Repertoire Files

Use `VDJtoolsParser`, `AIRRParser`, `AdaptiveParser`, `OldMiXCRParser`,
`VDJdbSlimParser`, and `OlgaParser` from `mir.common.parser`.

```python
from mir.common.parser import VDJtoolsParser
from mir.common.repertoire import LocusRepertoire

parser = VDJtoolsParser(sep="\t")
clonotypes = parser.parse("sample.tsv.gz")
rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
```

For Adaptive immunoSEQ / MLR tables, use `AdaptiveParser` and the returned
`LocusRepertoire` directly:

```python
from mir.common.parser import AdaptiveParser

parser = AdaptiveParser(locus="TRB")
rep = parser.parse_file("alice/mlr/MLR7_TCR1_Fresh_1.adap.txt.results.tsv")
```

For multi-locus formats, prefer parser helpers that already return a
`SampleRepertoire`:

```python
from mir.common.parser import VDJdbSlimParser, OlgaParser

vdjdb_sample = VDJdbSlimParser().parse_file("assets/vdjdb.slim.txt.gz", species="HomoSapiens")
olga_sample = OlgaParser().parse_file("assets/olga_humanTRB.txt", locus="TRB")
```

## 2. Build Datasets

Use `RepertoireDataset.from_folder_polars(...)` for batch loading from a folder
and metadata table.

```python
from mir.common.parser import VDJtoolsParser
from mir.common.repertoire_dataset import RepertoireDataset

dataset = RepertoireDataset.from_folder_polars(
    "data",
    parser=VDJtoolsParser(),
    metadata_file="metadata.tsv",
    file_name_column="file_name",
    sample_id_column="sample_id",
    metadata_sep="\t",
    n_workers=4,
)
```

Notes:

- Metadata rows are grouped by `sample_id` before loading, so split TRA/TRB files can be merged into one sample.
- For pandas-to-polars conversion paths in parsers, NaN values are normalized to `None` to avoid mixed-type schema issues.

## 3. Filter And Curate Repertoires

Use `filter_functional()` and `filter_canonical()` from `mir.common.filter`.

```python
from mir.common.filter import filter_functional
from mir.common.gene_library import GeneLibrary

imgt_lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="imgt")
functional_rep = filter_functional(rep, gene_library=imgt_lib)
```

## 4. Pool Samples And Repertoires

Use `pool_samples()` to combine clonotypes across samples with explicit identity
rules.

```python
from mir.common.pool import pool_samples

pooled = pool_samples(dataset, rule="aavj", include_sample_ids=True)
```

Supported keys:

- `ntvj`
- `nt`
- `aavj`
- `aa`

## 5. Gene Usage And Batch Correction

Use `GeneUsage` for raw usage summaries and
`compute_batch_corrected_gene_usage()` for batch-aware correction.

```python
from mir.basic.gene_usage import (
    GeneUsage,
    compute_batch_corrected_gene_usage,
    marginalize_batch_corrected_gene_usage,
)

gu = GeneUsage.from_repertoire(rep)
v_usage = gu.v_fraction("TRB", count="duplicates", pseudocount=1.0)

corr_vj = compute_batch_corrected_gene_usage(
    dataset,
    batch_field="batch_id",
    scope="vj",
    weighted=True,
)
v_marginal = marginalize_batch_corrected_gene_usage(corr_vj, scope="v")
```

Important behavior:

- Public count aliases such as `clonotypes`, `rearrangements`, and `duplicates` are normalized internally.
- Corrected probabilities should be consumed from `pfinal`, which is already renormalized per sample and locus.

## 6. Downsampling And Resampling

Use `downsample()`, `select_top()`, and `resample_to_gene_usage()` from
`mir.common.sampling`.

```python
from mir.common.sampling import downsample, resample_to_gene_usage

rep_small = downsample(rep, 1_000, random_seed=42)
rep_matched = resample_to_gene_usage(rep, target_usage, scope="v", weighted=True, random_seed=42)
```

## 7. Graph And Neighborhood Analysis

Use the graph utilities in `mir.graph` for edit-distance neighborhoods and
token (k-mer) graphs.

```python
from mir.graph import compute_neighborhood_stats
from mir.graph.edit_distance_graph import build_edit_distance_graph
from mir.basic.token_tables import filter_token_table, tokenize_clonotypes
from mir.graph.token_graph import build_token_graph

# Edit-distance graph (Hamming or Levenshtein on junction_aa)
stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, n_jobs=-1)
graph = build_edit_distance_graph(rep.clonotypes, metric="levenshtein", threshold=1, n_jobs=-1)

# Token graph filtered to RS-bearing 3-mers
table    = tokenize_clonotypes(rep.clonotypes, k=3)
rs_table = filter_token_table(table, kmer_pattern="RS")
g_rs     = build_token_graph(rep.clonotypes, rs_table)
```

Notes:

- `tokenize_clonotypes` accepts a `list[Clonotype]`; `tokenize_rearrangements` remains as a deprecated alias for compatibility.
- Use `Graph.are_adjacent()` instead of the deprecated `Graph.are_connected()` when querying igraph graphs directly.
- Trie-backed search is used for edit-distance graphs when available.
- For long amino-acid queries, exact brute-force fallback is used to avoid false negatives from bit-parallel limits.
- `compute_neighborhood_stats` and `build_edit_distance_graph` use multiprocess workers when `n_jobs > 1` for true multi-core execution.
- `build_edit_distance_graph` returns an igraph Graph with vertex attributes: `name` (junction_aa), `r_id`, `v_gene`, `j_gene`, `c_gene`. Use `g.vs["j_gene"]` directly; strip alleles with `.split("*")[0]` when comparing against gene family names.
- **V/J-restricted search is faster than unrestricted search on natural repertoires.**  When `match_v_gene=True` or `match_j_gene=True`, `compute_neighborhood_stats` builds one small trie per gene group (grouped-trie strategy) instead of filtering one large trie in Python.  Each per-group trie is ~N_total / N_groups sequences; all hits are gene-correct by construction so no Python post-filtering loop is needed.  Benchmark on 300 K clustered TRB sequences (n_jobs=8): unrestricted 9.9 s → V+J restricted 5.5 s (1.8× faster).  On natural 1 M+ repertoires the gain is larger because average group size stays roughly constant while the full-trie Python validation loop would grow with N.

For donor-vs-pool overlap workflows, use `many_vs_pool_overlap()` when scoring a sequence of repertoires against one pooled reference. It keeps the pooled worker state shared across batches and avoids repeating per-sample setup in hotspot notebooks such as `aging_analysis.ipynb`.

## 8. Control Repertoires

Use `ControlManager` to build or load synthetic and real controls.

```python
from mir.common.control import ControlManager

mgr = ControlManager()
mgr.ensure_synthetic_control("human", "TRB", n=1_000_000)
df_control = mgr.ensure_and_load_control_df("real", "human", "TRB")
```

Operational notes:

- Default cache root is `~/.cache/mirpy/controls`.
- Override with `MIRPY_CONTROL_DIR` for shared or scratch storage.
- Synthetic caches are keyed by species, locus, and `n`.

To get V/J/VJ usage probabilities analytically from an already-loaded OLGA
model (instant, no sampling), use `get_gene_usage_from_olga_model`:

```python
from mir.basic.pgen import OlgaModel
from mir.basic.gene_usage import get_gene_usage_from_olga_model

m = OlgaModel(locus="TRB", species="human")
probs = get_gene_usage_from_olga_model(m)
# probs["v"]  — {gene_name: P(V)}  (aggregated at gene level, alleles as *01)
# probs["j"]  — {gene_name: P(J)}
# probs["vj"] — {(v_gene, j_gene): P(V,J)}
```

Reads IGoR model marginals directly: `PV`/`PDJ` for VDJ models (TRB/TRD/IGH),
`PVJ` for VJ models (TRA/TRG/IGK/IGL).  Probabilities are aggregated under
the major-allele key (e.g., all `TRBV5-1*02` mass folds into `TRBV5-1*01`).

To precompute and cache OLGA-derived V/J/VJ probabilities via synthetic
sampling (larger but sampling-based), use
`precompute_olga_gene_usage_probabilities`:

```python
from mir.basic.gene_usage import precompute_olga_gene_usage_probabilities

probs = precompute_olga_gene_usage_probabilities(
  species="human",
  locus="TRB",
  synthetic_n=5_000_000,
  n_jobs=8,
  progress=True,
)
```

This stores/reuses the synthetic control artifact via `ControlManager` and
returns a dict with `v`, `j`, and `vj` probability maps.

## 9. Prototype-Based Embeddings With TCREMP

Use `TCREmp` from `mir.embedding.tcremp` to embed clonotypes as distance vectors
to a fixed set of prototypes, enabling rapid downstream analysis and ML.

```python
from mir.embedding.tcremp import TCREmp
from mir.common.clonotype import Clonotype

# Build from defaults: fast fixed-gap junction alignment (C-accelerated, ~25M pairs/s)
model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000, junction_method="fixed_gap")

# Embed clonotypes
clonotypes = [
    Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF"),
    Clonotype(v_gene="TRBV20-1*01", j_gene="TRBJ1-1*01", junction_aa="CSARDSSYEQYF"),
]
X = model.embed(clonotypes, n_jobs=4)  # shape: (2, 3000), dtype: float32

# For full DP semantics (slower, ~270k pairs/s):
model_bio = TCREmp.from_defaults("human", "TRB", n_prototypes=100, junction_method="biopython")

# For custom prototypes (with incomparability warning):
model_custom = TCREmp.from_file("my_prototypes.tsv", species="human", locus="TRB")
```

**Embedding structure**: Each clonotype is embedded as `[v_1, j_1, junc_1, v_2, j_2, junc_2, ..., v_K, j_K, junc_K]`
where `K` is the number of prototypes. All distances use the formula:

$$d(a, b) = s(a,a) + s(b,b) - 2 \cdot s(a,b)$$

This ensures metric properties: $d(a,a) = 0$, $d(a,b) = d(b,a)$, and non-negativity.

## 10. Metaclonotypes (Functional Clusters)

Use metaclonotypes as a lightweight cluster layer over existing repertoires.
This avoids rebuilding `LocusRepertoire` objects while enabling cluster-level
count aggregation, functional diversity, and overlap workflows.

```python
from mir.common.metaclonotype import (
    metaclonotypes_from_components,
    summarize_metaclonotypes,
    functional_diversity,
)

# components: list[list[sequence_id]] from any clustering backend
meta = metaclonotypes_from_components(components, cluster_prefix="cc")
rep.set_metaclonotypes(meta)

summary = summarize_metaclonotypes(rep, meta)
func_div = functional_diversity(rep, meta, count_field="duplicate_count")
```

Common builders:

- `metaclonotypes_from_labels` for DBSCAN/Leiden/Louvain-style label vectors.
- `metaclonotypes_from_igraph` for connected components or graph memberships.
- `metaclonotypes_from_seed_neighbors` for seed + first-neighborhood clusters
  (useful for ALICE/TCRNET-derived representative clonotypes).
- `mir.utils.metaclonotype_clustering.metaclonotypes_from_cluster_labels` as
  a backend-agnostic helper for DBSCAN/OPTICS/VDBSCAN-style labels.
- `mir.utils.metaclonotype_clustering.metaclonotypes_from_graph_communities`
  for components/Leiden/Louvain in a shared implementation.
- `mir.utils.metaclonotype_clustering.metaclonotypes_from_search_scope` for
  representative-centered search scopes (tcrtrie or score-radius style).

TCREmp convenience wrappers:

- `mir.embedding.metaclonotypes_from_tcremp_labels`
- `mir.embedding.paired_metaclonotypes_from_tcremp_labels`

Token-graph/GLIPH convenience wrappers:

- `mir.graph.metaclonotypes_from_token_clonotype_graph`
- `mir.graph.build_gliph_metaclonotypes`

Biomarker integration:

- `mir.biomarkers.metaclonotypes_from_alice`
- `mir.biomarkers.metaclonotypes_from_tcrnet`

Functional overlap:

- `functional_overlap_1` computes overlap where two metaclonotypes match if
  they share at least one clonotype identity.

Mutual-information-style summary:

- Use pooled and separate metaclonotype count vectors with
  `pooled_entropy_difference` to compute
  `H(A pooled with B) - H(A) - H(B)`.

## 11. Diversity Metrics, Hill Curves, And Rarefaction

Prefer function-first diversity APIs from `mir.common.diversity`.
Repertoire object methods are convenience delegates to the same functions.

```python
from mir.common.diversity import (
    summarize_clonotypes,
    summarize_loci_clonotypes,
    summarize_count_groups,
    hill_curve_clonotypes,
    rarefaction_curve_clonotypes,
)

summary = summarize_clonotypes(sample["TRB"].clonotypes)
per_locus = summarize_loci_clonotypes({locus: rep.clonotypes for locus, rep in sample.loci.items()})
pair_level = summarize_count_groups({"TRA_TRB": [2, 1, 1], "TRG_TRD": [1]})

hill = hill_curve_clonotypes(sample["TRB"].clonotypes)
rare = rarefaction_curve_clonotypes(sample["TRB"].clonotypes, m_steps=[10, 25, 50, 100], include_exact=True)
```

Available summary fields:

- `abundance`
- `diversity`
- `singletons`
- `doubletons`
- `expanded` (>0.1%)
- `hyperexpanded` (>1%)
- `chao1`
- `gini_simpson`
- `shannon`

Count modes:

- `duplicate_count` (default for locus/sample repertoires)
- `umi_count` (optional)
- `barcode_count` (default for paired/single-cell repertoire diversity methods)

Object-level usage (delegates to function-first APIs):

```python
# Locus/sample
trb_summary = sample["TRB"].diversity()
sample_per_locus = sample.diversity(per_locus=True)

# Paired and single-cell
pair_summary = paired_repertoire.diversity()                 # default barcode_count
chain_summary = paired_repertoire.diversity_by_locus()       # TRA/TRB/... per-locus
sc_summary = single_cell_sample.diversity(per_locus=True)    # delegates to paired repertoire
```

Notebook: `notebooks/diversity_analysis.ipynb` — donor summary tables, rarefaction, coverage, Hill
curves, and Healthy vs MS cohort comparisons.

## 12. Pairwise Overlap Workflows

Use `pairwise_overlap` and `pairwise_overlap_matrix` from
`mir.comparative.overlap`.

```python
from mir.comparative.overlap import pairwise_overlap, pairwise_overlap_matrix

# One pair
r = pairwise_overlap(
  rep_a,
  rep_b,
  overlap_space="aavj",   # one of: ntvj, nt, aavj, aa
  metric="hamming",       # exact | hamming | levenshtein
  threshold=1,
)

# All pairs
df = pairwise_overlap_matrix(
  reps,
  sample_ids=sample_ids,
  overlap_space="aavj",
  metric="exact",
  threshold=0,
  n_jobs=-1,
)
```

Operational notes:

- Approximate matching (`threshold > 0`) is supported only for `aa` and `aavj` overlap spaces.
- In amino-acid overlap spaces, non-coding clonotypes are excluded from overlap matching.
- Similarity outputs are primary (`f_similarity`, `d_similarity`, `f2_similarity`).
- Use metric transforms only when distance-like inputs are required:
  - `f_metric = 1 - f_similarity`
  - `d_metric = 1 / d_similarity` (for `d_similarity > 0`)
- For repeated sample-vs-fixed-target calls, `pairwise_overlap` reuses target-side prepared data internally, which avoids repeated trie setup.
- For a single pair, forcing many workers can be slower due to process startup; use `n_jobs=1` unless the query side is very large.

Notebook analysis guidance:

- Keep heavy transformations in Polars; convert to pandas only for table display.
- For cohort embedding, use distance-like matrices (for example from `f_metric`/`d_metric`) as precomputed dissimilarities in UMAP/MDS.
- For diversity-vs-age comparisons in aging notebooks, prefer a fixed-depth subsample that most donors satisfy (for example 250k reads).

**Parallelization**: TCREmp supports workload-aware `n_jobs` auto selection:
- `n_jobs=None` (default): auto-switch based on `len(clonotypes) * n_prototypes`
  between serial (`1`) and `os.cpu_count()`
- `n_jobs=1`: force serial processing
- `n_jobs>1`: force explicit worker count
- In auto mode, the BioPython backend stays serial by default (thread overhead usually dominates)

Why workload uses both clonotypes and prototypes:
- Work is split on the clonotype/query side for threading.
- Each query still scores against all prototypes, so per-query work scales with
  prototype count.
- Practical complexity is therefore proportional to `N_queries * N_prototypes`.

Backend choice guidance:
- Use `junction_method="fixed_gap"` for production embedding speed and stable behavior.
- Use `junction_method="biopython"` when full DP alignment semantics are needed.
- Current repository benchmarks do not demonstrate a consistent downstream
  quality improvement from BioPython that would justify changing the default.

**Performance**:
- Fixed-gap: ~25 M pairs/s (C-accelerated via seqdist C extension)
- BioPython: ~270 k pairs/s (full DP)
- Speedup: ~90× for fixed-gap

**Example: epitope analysis**

```python
# Embed clonotypes from two epitopes
epi1_clonos = [...]  # ~200 clonotypes with epitope 1
epi2_clonos = [...]  # ~200 clonotypes with epitope 2

X_epi1 = model.embed(epi1_clonos)
X_epi2 = model.embed(epi2_clonos)

# Compute within-epitope vs between-epitope distances
from scipy.spatial.distance import cdist
within_dist = cdist(X_epi1, X_epi1)  # within epitope 1
between_dist = cdist(X_epi1, X_epi2)  # across epitopes

print(f"Mean within-dist: {within_dist[~np.eye(len(epi1_clonos), dtype=bool)].mean():.3f}")
print(f"Mean between-dist: {between_dist.mean():.3f}")
```

**Backward compatibility**:

- `cdr3_aligner` property and `_proto_cdr3` attribute remain available (aliases to `junction_aligner` and `_proto_junction`).
- Existing pickled models with `CDRAligner` unpickle without modification.

### Paired TRA/TRB embeddings from VDJdb full

Use `VDJdbFullPairedParser` when the source file is `vdjdb_full.txt.gz` and you
want paired TRA/TRB records instead of independent slim rows.

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
  strict_df,
  sample_id="vdjdb_full_human_strict",
  barcode_metadata=strict_meta,
)

# Imputation mode for single-chain rows.
impute_df, impute_meta = parser.parse_cell_clonotypes_file(
  "tests/assets/vdjdb_full.txt.gz",
  species="HomoSapiens",
  include_incomplete=True,
)
imputed_df = impute_missing_chains(impute_df)
imputed_sample = build_tenx_sample_from_cell_clonotypes(
  imputed_df,
  sample_id="vdjdb_full_human_imputed",
  barcode_metadata=impute_meta,
)

paired_model = PairedTCREmp.from_defaults(
  species="human",
  locus_pair="TRA_TRB",
  n_prototypes=500,
)
paired_clonotypes = imputed_sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
X_pair = paired_model.embed(paired_clonotypes)
```

Operational notes:

- The paired embedding dimension is the sum of the two chain embedding dimensions.
- `parse_cell_clonotypes_file(..., include_incomplete=True)` returns synthetic single-cell style rows so you can run `impute_missing_chains` before building paired repertoires.
- Each synthetic barcode stores `vdjdb_record_id`, `mhc.a`, `mhc.b`, `mhc.class`, `antigen.epitope`, `antigen.gene`, and `antigen.species` in `SingleCellRepertoire.barcode_metadata`.
- For tabular metadata workflows, use `SingleCellRepertoire.metadata_to_polars()` and keep downstream analysis polars-native.
- Notebook asset downloads use `notebooks/assets/large/airr_benchmark`; test bootstrap mirrors `vdjdb_full.txt.gz` into `tests/assets/vdjdb_full.txt.gz`.
- `notebooks/tcremp_vdjdb_analysis_paired.ipynb` demonstrates strict vs imputed paired analysis with cumulative PCA variance, floor-quantile kneedle eps selection (`select_eps_kneedle_stable`), DBSCAN purity/retention/consistency summaries, and SLL epitope outlier diagnosis against paired/TRA-only/TRB-only embeddings.

## 12.5. ALICE Enrichment

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

Metadata-first variant (writes results into clonotype metadata in-place):

```python
from mir.biomarkers.alice import add_alice_metadata

rep = add_alice_metadata(
    rep,
    species="human",
    match_mode="vj",
    pgen_mode="exact",
    pvalue_mode="poisson",
    pseudocount=0.0,
    n_jobs=8,
)
# Metadata keys: alice_n, alice_N, alice_pgen_raw, alice_pgen,
#                alice_expected, alice_fold, alice_p_value, alice_q_value
```

**`pgen_mode` options:**

| Mode    | Speed         | Accuracy                   | Notes |
|---------|---------------|----------------------------|-------|
| `exact` | Fast (7ms/seq) | OLGA analytical exact Pgen | Default; underestimates λ for ALICE — use `"1mm"` or `"mc"` |
| `1mm`   | Slow (70ms/seq) | Sums Pgen over 1mm neighbors | Best sensitivity; use skip_ends=2 (env `MIRPY_PGEN_1MM_SKIP_ENDS=2`) |
| `mc`    | Very fast after pool build | MC match counting + OLGA fallback | Pool built once, cached; 100–1000× faster than OLGA per sample |

**ALICE runtime scaling (TRB, `match_mode="vj"`, `min_neighbors=2`, `n_jobs=8`):**

Pgen is computed only for sequences that pass the `min_neighbors` filter (~1–5% of clonotypes).
V+J-restricted neighbor search uses the grouped-trie strategy and is 1.5–2× faster than
unrestricted search on natural repertoires.

| Dataset size | `"exact"` wall time | `"1mm"` wall time | `"mc"` wall time (after pool) |
|---|---|---|---|
| 1 K clonotypes | < 1 s | < 5 s | < 1 s |
| 10 K clonotypes | 1–3 s | 5–30 s | < 1 s |
| 100 K clonotypes | 3–15 s | 30–150 s | 1–5 s |
| 1 M clonotypes | 15–90 s | 2–15 min | 5–30 s |

First `"mc"` call builds the 10 M pool (37 s, 8 workers, human TRB); all subsequent samples reuse it from cache.  Use `"mc"` for all production runs (paper-correct 1mm pgen approximation, ~100× faster than `"1mm"` after pool build).  `"exact"` underestimates λ and inflates hit counts — not recommended for ALICE.

**`pgen_mode="mc"` details:**
- Generates `mc_n_pool` productive sequences on first call (37s for 10M TRB, 8 workers).
- Pool cached in `mir.basic.pgen._MC_POOL_CACHE` keyed by `(locus, species, n_pool, seed, skip_ends)`.
- Pgen estimate: `pgen_mc = n_1mm_matches / n_total_rearrangements` (includes non-productive rejection count).
- Sequences with `< mc_min_count` pool matches fall back to OLGA analytical 1mm Pgen (same λ scale).
- Fold-error vs OLGA at count≥2: ~1.5× (1M pool) / ~1.45× (10M pool).
- Call `mir.basic.pgen.clear_mc_pool_cache()` to release pool memory.

**Differences from the original paper:**
- Paper uses **100 M** sequences; this implementation uses **10 M** (`mc_n_pool`) and falls back to
  OLGA analytical 1mm Pgen for sequences with < 2 pool matches.  100 M requires ~17 GB and ~16 min — use
  `mc_n_pool=100_000_000` only when that budget is available.
- Default is now `match_mode="vj"` (matching the paper).  TCRNET default is `match_mode="none"`.
- **Gene-usage conditioning** (new): when `match_mode != "none"`, `N` and `pgen` are scaled by
  `P_OLGA(V,J)` (from `get_gene_usage_from_olga_model`): `N_adj = P(VJ) × N_total`,
  `pgen_adj = pgen / P(VJ)`.  These cancel in λ = N_total × pgen, but the observed `k` counts only
  V/J-matching neighbours — making the test more specific without inflating λ.
  Same logic for TCRNET: `M` replaced by `P_ctrl(VJ) × M_total` (using `GeneUsage.from_repertoire`).
- Paper's exact pre-screen (`n ≤ N × pgen_exact → skip 1mm`) has been removed — it filtered 0 sequences
  in practice on large repertoires (371 K clonotypes: 100% pass rate, 32 s wasted).

**ALICE / TCRNET relationship:**
ALICE (Poisson, λ = N × pgen_1mm) and TCRNET (Binomial, p = m/M) converge in the limit of a large pool:
- TCRNET is purely MC-control — no OLGA Pgen calls. Works with real or synthetic controls. When used with
  a real control it naturally captures V/J bias without conditioning.
- ALICE uses OLGA Pgen with optional MC estimation. Falls back to analytical Pgen for sparse sequences.
- To reproduce the original ALICE paper using TCRNET: `match_mode="vj"`, 100 M synthetic pool, `q_factor=Q`.
- See section 16 (TCRNET) for full ALICE-as-TCRNET recipe.

Key behavior notes:

- `min_neighbors=2` requires a sequence + at least 1 Hamming-1 neighbour (default). Isolated sequences get `p_value=1.0` without OLGA computation.
- `q_factor` multiplies λ = N × pgen × Q. Calibrate from real data: `Q ≈ median(pgen_real / pgen_olga)` for functional sequences.
- P-value batch execution defaults to process workers (`MIRPY_ALICE_PVALUE_EXECUTOR=process`).
- `pvalue_mode="negative-binomial"` uses `NB(mu=N*pgen, dispersion=1)` — more conservative than Poisson.
- `q_value` in the output table is BH-corrected over all clonotypes in the locus (before frequency filtering).
- For multi-sample workflows, reuse a single `OlgaModel` instance across all samples; the internal persistent `multiprocessing.Pool` is created once per model instance and reused on each `compute_pgen_junction_aa_bulk` call (zero spawn overhead after the first call).

**Cluster analysis — `alice_hit_clusters`:**

```python
from mir.biomarkers.alice import alice_hit_clusters

# Default: cluster ALICE hits only (V-gene-restricted 1mm CDR3 edges)
hits_clustered = alice_hit_clusters(hits_df)

# Expand clusters with 1mm non-enriched neighbors from the full repertoire table
hits_expanded = alice_hit_clusters(hits_df, full_df=full_table, non_enriched_neighbors=True)
```

- **V-gene restriction**: edges only between sequences sharing the same V-gene family (`TRBV9*01` → `TRBV9`). Without this, transitive chains across different V families merge unrelated clusters into one giant component.
- `non_enriched_neighbors=True`: any non-hit sequence from `full_df` that is 1mm (same V-gene) from any hit is added to that hit's cluster. Useful for visualising the halo of non-enriched neighbours.
- Returns `hits_df` (plus non-enriched neighbors when applicable) with `cluster_id` (int) and `is_hit` (bool) columns added.
- **Joint B27+ analysis pattern** (AS dataset from Pogorelyy 2019): pool deduplicated hits from all B27+ donors, cluster, verify CASSVGL[YF]STDTQYF (TRBV9 TRBJ2-3) is rank-1 cluster (size 58 in published data); B27- donor has 0 motif sequences.

```python
# Pool B27+ hits, deduplicate per donor to avoid nucleotide-variant inflation
b27_hits = pd.concat([
    as_alice_hits[d].drop_duplicates(subset=["junction_aa", "v_gene"]).assign(donor_id=d)
    for d in b27_pos_ids
], ignore_index=True)
clustered = alice_hit_clusters(b27_hits)
sizes = clustered.groupby("cluster_id").size().sort_values(ascending=False)
# sizes.index[0] is the motif cluster for AS B27+ data
```

## 13. Single-Cell 10x Paired Chains

Use `load_10x_vdj_v1_sample` to assemble paired-chain objects from 10x v1 sample
files where consensus annotations define clonotypes and all-contig annotations
define cell barcode linkage.

```python
from mir.common.single_cell import load_10x_vdj_v1_sample

sample = load_10x_vdj_v1_sample(
    consensus_annotations_path="airr_benchmark/dcode/vdj_v1_hs_aggregated_donor1_consensus_annotations.csv.gz",
    all_contig_annotations_path="airr_benchmark/dcode/vdj_v1_hs_aggregated_donor1_all_contig_annotations.csv.gz",
  sample_id="vdj_v1_hs_aggregated_donor1",
)

print(sample.loaded_cell_count)
print(sample.loaded_clonotype_count)
print(sample.paired_locus_repertoires["TRA_TRB"].clonotype_count)
print(sample.chain_multiplicity)
```

Key behavior:

- Supported locus-pair families: `TRA_TRB`, `TRG_TRD`, `IGH_IGK`, `IGH_IGL`.
- Multi-chain cells are expanded deterministically by cartesian product per
  locus pair (e.g., `2x1` yields two paired clonotypes).
- `SingleCellRepertoire` keeps barcode -> pair_id links separate for future
  multimodal integration.

## 13.1 Single-Cell 10x + CITE-seq Integration

Use `load_10x_vdj_v1_citeseq_sample` when a donor has both 10x VDJ files and
an accompanying `*_binarized_matrix.csv.gz` CITE-seq matrix.

```python
from mir.common.single_cell import (
    load_10x_vdj_v1_citeseq_sample,
    validate_citeseq_binders_against_vdjdb_10x,
)

sample = load_10x_vdj_v1_citeseq_sample(
    consensus_annotations_path="airr_benchmark/dcode/vdj_v1_hs_aggregated_donor1_consensus_annotations.csv.gz",
    all_contig_annotations_path="airr_benchmark/dcode/vdj_v1_hs_aggregated_donor1_all_contig_annotations.csv.gz",
    binarized_matrix_path="airr_benchmark/dcode/vdj_v1_hs_aggregated_donor1_binarized_matrix.csv.gz",
    sample_id="vdj_v1_hs_aggregated_donor1",
)

print(sample.paired_repertoire.loaded_cell_count)
print(sample.cite_seq_matrix.height)
print(sample.cite_seq_binder_columns.height)

missing = validate_citeseq_binders_against_vdjdb_10x(
    sample.cite_seq_binder_columns,
    "airr_benchmark/vdjdb/vdjdb-2025-12-29/vdjdb_full.txt.gz",
)
print(missing)
```

Notes:

- `SingleCellSample` packages the paired repertoire with CITE-seq matrix and
  parsed binder metadata (`column`, `hla`, `antigen.epitope`, `is_binder`).
- The VDJdb 10x sanity check normalizes HLA formatting and compares
  `(HLA, epitope)` targets against rows whose `reference.id` contains `10x`.
- Current dcode donors have two consistent residual unmatched targets
  (`CLGGLLTMV`, `LLMGTLGIVC`) that are tracked in tests.
- `notebooks/tcremp_10xdcode_analysis.ipynb` provides a full diagnostics run with
  TRA/TRB/TRA_TRB embeddings, cumulative PCA variance plots, sorted 4-NN
  kneedle/eps plots, UMAP projections colored by epitope, and per-epitope
  precision/recall/F1 support tables.

## 14. 10x Benchmark And scirpy Concordance

Use the dedicated benchmark test module for speed, memory, and parity checks on
AIRR benchmark donors:

```bash
env RUN_BENCHMARK=1 python -m pytest tests/test_single_cell_10x_benchmark.py -s -x
env RUN_BENCHMARK=1 python -m pytest tests/test_single_cell_repair_benchmark.py -s -x
env RUN_BENCHMARK=1 python -m pytest tests/test_single_cell_citeseq_benchmark.py -s -x
env RUN_BENCHMARK=1 python -m pytest tests/test_tcremp_vdjdb_benchmark.py -s -x
```

This suite validates:

- bounded runtime and RSS deltas for mirpy 10x loading,
- non-empty loaded cells/clonotypes/pairing summaries,
- mirpy vs scirpy TRA/TRB quadrant concordance on dominant patterns,
- speed/memory competitiveness relative to scirpy on the same donor.

## 15. Single-Cell Parsing, Repair, And Pairing Graphs

Use the parser-first API when you need to apply cleanup or imputation before
assembling sample paired-clonotype objects.

```python
from mir.common.single_cell import build_tenx_sample_from_cell_clonotypes
from mir.common.single_cell_parser import load_10x_vdj_v1_cell_clonotypes
from mir.common.single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains
from mir.graph.single_cell_pairing import build_pairing_graph

raw = load_10x_vdj_v1_cell_clonotypes(
    "..._consensus_annotations.csv.gz",
    "..._all_contig_annotations.csv.gz",
    sample_id="sample1",
    check_is_cell=True,  # default
)

imputed = impute_missing_chains(raw, species="human", seed=42, reuse_slave_per_master=True)
cleaned = cleanup_cell_clonotypes(
    imputed,
    secondary_ratio_threshold=0.1,
    secondary_min_umi_count=2,
    secondary_min_duplicate_count=5,
    enforce_consistent_slave_per_master=True,
    consistency_only_on_synthetic_slave=True,
    max_slave_edges_per_master=10,
)

sample = build_tenx_sample_from_cell_clonotypes(cleaned, sample_id="sample1")
pairing_graph = build_pairing_graph(sample, min_shared_cells=1)

print(pairing_graph.nodes)
print(pairing_graph.edges)
```

Repair behavior summary:

- Missing chain families are imputed per `(barcode, raw_pair_id)` group.
- Optional `reuse_slave_per_master=True` reuses one synthetic slave clonotype
  per master clonotype during imputation.
- Synthetic rows are OLGA-based where possible and always use
  `duplicate_count=1`, `umi_count=1`.
- Cleanup keeps top-1 for `TRB`, `TRD`, `IGH` and conditionally keeps top-2
  for `TRA`/`TRG` and `IGK`/`IGL` using ratio and minimum support thresholds.
- Cleanup can enforce one synthetic slave chain per master clonotype and can
  prune master/slave families where one master is connected to too many slave
  clonotypes (`max_slave_edges_per_master`).
- `consistency_only_on_synthetic_slave=True` limits consistency enforcement to
  synthetic slaves; set `False` to enforce consistency on all slave chains.

## 16. TCRNET Enrichment

Use `compute_tcrnet` / `add_tcrnet_metadata` from `mir.biomarkers.tcrnet`.

TCRNET is a **purely MC-control** enrichment algorithm — no OLGA Pgen is computed
at any stage.  It counts Hamming/Levenshtein-1 neighbors of each query CDR3 in a
provided control repertoire (real or synthetic) and tests whether the sample
neighborhood density exceeds the control background using a binomial (or
beta-binomial) model.

```python
from mir.biomarkers.tcrnet import compute_tcrnet, TcrnetParams, TcrnetResult

# Pass an explicit control repertoire or request a managed one via control_type.
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
    q_factor=1.0,                      # selection correction for synthetic controls (see below)
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

Metadata-first variant:

```python
from mir.biomarkers.tcrnet import add_tcrnet_metadata

rep = add_tcrnet_metadata(
    rep,
    control=control_rep,
    metric="hamming",
    threshold=1,
    match_mode="vj",
    pvalue_mode="binomial",
    pseudocount=1.0,
    q_factor=1.0,
    n_jobs=-1,
)
# Metadata keys: tcrnet_n, tcrnet_N, tcrnet_m (raw), tcrnet_M,
#                tcrnet_sample_density, tcrnet_control_density (q-adjusted),
#                tcrnet_fold, tcrnet_p_value, tcrnet_q_value
```

**Key behavior notes:**

- p-value: `P(X >= n) where X ~ Binomial(N, q_factor × (m+pc)/(M+pc))`. `beta-binomial` uses `BetaBinom(N, alpha=m+pc, beta=(M-m)+pc)`.
- Raw `m` and `M` (including pseudocount) are stored in metadata; `q_factor` is applied only when computing `p_value`, `fold_enrichment`, and `control_density`.
- Control pseudocount (`pseudocount`, default 1.0) adds one virtual neighbor match to avoid zero-division.
- `q_value` is BH-corrected over all clonotypes in the locus.
- `n_jobs=-1` uses all physical cores.

**`q_factor` — selection correction for synthetic controls:**

OLGA synthetic sequences are drawn from the *recombination* model (pre-thymic
selection).  Their neighborhood density is systematically lower than a
post-selection real repertoire by a factor Q ≈ 3–5 for human TRB.  Without
correction the enrichment test is too liberal (inflates hits).

Estimate Q from a real control sample:
```python
from mir.basic.pgen import McPgenPool, OlgaModel
import numpy as np, math

model = OlgaModel(locus="TRB", species="human")
olga_pgens = model.compute_pgen_junction_aa_bulk(test_seqs, max_mismatches=0, n_jobs=8)
real_pool  = McPgenPool.build_real(control_seqs, locus="TRB")
real_pgens = real_pool.pgen_1mm_bulk(test_seqs, n_jobs=8)

q_samples = [rp / op for rp, op in zip(real_pgens, olga_pgens) if rp > 0 and op > 0]
Q = float(np.median(q_samples))   # typical value: 3–5 for human TRB
```

Then pass `q_factor=Q` to `compute_tcrnet` when using a synthetic control.
Leave `q_factor=1.0` (default) for real controls.

**TCRNET as original ALICE (V+J+1mm, 100M pool):**
```python
from mir.basic.pgen import McPgenPool
pool = McPgenPool.build_synthetic(100_000_000, locus="TRB", n_jobs=8)
# Convert pool to LocusRepertoire — use the pool's unique sequences as control
result = compute_tcrnet(
    rep,
    control=LocusRepertoire([Clonotype(...) for s in pool._unique_seqs], locus="TRB"),
    match_mode="vj",       # V+J gene restriction (as in original paper)
    pvalue_mode="binomial",
    q_factor=Q,            # thymic-selection correction
)
```
This is statistically equivalent to `compute_alice(rep, pgen_mode="mc", mc_n_pool=100_000_000, match_mode="vj", q_factor=Q)`.

**TCRNET vs ALICE summary:**

| | TCRNET | ALICE |
|---|---|---|
| Background | Any MC control (real or synthetic) | OLGA Pgen (via MC pool or analytical) |
| V/J bias | Captured via real control or `normalize_control_vj_usage` | Via `match_mode` parameter |
| Pgen calls | None | OLGA 1mm Pgen (or 10M MC approximation) |
| Statistics | Binomial / Beta-Binomial | Poisson |
| Selection correction | `q_factor` (explicit) | `q_factor` (explicit) |
| Default control | Must be provided explicitly | Synthetic OLGA pool |

## 17. GLIPH-Style K-mer Enrichment (binomial)

For GLIPH-like motif workflows, prefer repertoire-first extraction and reuse
one shared unnormalized control background across studies.

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
    study_repertoire,
    families,
    count_mode="clonotype",
    build_mappings=False,
    trim_first=3,
    trim_last=4,
    chunk_size=200_000,
)

comp = compare_gliph_token_incidence(
    study_artifacts["u3"],
    ctrl_artifacts["u3"],
    test="binom",
    p_adj_method="fdr_bh",
    pseudocount=1,
)

sig = (
  (comp["p_val_adj"] < 0.05)
  & (comp["freq_fc"] > 1.0)
)
```

Interpretation notes:

- In `test="binom"`, `p_background` is computed as `count_2 / total_control_clonotypes`.
- The p-value is one-sided enrichment (`P[X >= count_1]`) under `Binomial(total_sample_clonotypes, p_background)`.
- For large real controls, use `build_mappings=False` plus `chunk_size` to stream control extraction without materializing large clonotype-token maps.
- Keep `trim_first`/`trim_last` the same for sample and control; GLIPH defaults are `trim_first=3`, `trim_last=4`.
- For interactive notebooks, start with `chunk_size=100_000` to `200_000`; increase only after runtime and memory are stable.

## 17.5. Pairwise Sample Overlap Metrics (Detailed Reference)

Use `pairwise_overlap` for a single pair and `pairwise_overlap_matrix` for all
N×(N−1)/2 pairs across a cohort.  Both functions live in
`mir.comparative.overlap` and are re-exported from `mir.comparative`.

### Single pair

```python
from mir.comparative.overlap import pairwise_overlap

# Exact matching
r = pairwise_overlap(rep1, rep2)

# Approximate matching — Hamming distance 1 (1 substitution)
r_h1 = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)

# Approximate matching — Levenshtein distance 1 (any single edit)
r_l1 = pairwise_overlap(rep1, rep2, metric="levenshtein", threshold=1)

print(r.jaccard, r.d_similarity, r.f_similarity, r.morisita_horn)
print(r.f2_similarity, r.correlation)  # nan for approximate matching
```

`PairwiseOverlapResult` fields:

| Field | Description |
|---|---|
| `n1`, `n2` | unique clonotypes in each sample |
| `n1_matched`, `n2_matched` | clones with ≥1 match in the other sample |
| `f1_overlap`, `f2_overlap` | total frequency of matched clones |
| `jaccard` | n12 / (n1 + n2 − n12) |
| `szymkiewicz_simpson` | min(n1_matched, n2_matched) / min(n1, n2) |
| `d_similarity` | n12 / sqrt(n1 × n2) |
| `f_similarity` | sqrt(f1_overlap × f2_overlap) |
| `morisita_horn` | 2 Σ(p_i q_i) / (D1 + D2) |
| `correlation` | Pearson r of overlap frequencies (NaN for approximate) |
| `f2_similarity` | Σ sqrt(p_i × q_i) over matched pairs (NaN for approximate) |
| `mode` | "exact", "hamming:N", "levenshtein:N" |
| `is_approximate` | True when threshold > 0 |

Use `result.as_dict()` to convert all fields to a plain `dict` for DataFrame construction.

For approximate matching (threshold > 0), `correlation` and `f2_similarity` are
`nan`; Jaccard and D-metric use the geometric mean of `n1_matched` and
`n2_matched` for symmetry.

### Pairwise matrix (cohort)

```python
from mir.comparative.overlap import pairwise_overlap_matrix

# Returns a long-format DataFrame with one row per ordered pair (i < j)
df = pairwise_overlap_matrix(
    reps,
    sample_ids=ids,        # optional list of string IDs
    metric="exact",        # or "hamming" / "levenshtein"
    threshold=0,
    n_jobs=-1,             # -1 = all physical cores
)

# Pivot to symmetric NxN matrix of a single metric
pivot = df.pivot(index="sample_id_1", columns="sample_id_2", values="f_similarity")
```

### Dissimilarity for UMAP / clustering

```python
import numpy as np
from umap import UMAP

f_vals = df.pivot(index="sample_id_1", columns="sample_id_2", values="f_similarity")
f_mat = f_vals.to_numpy()
# Symmetrize and fill self-distance
n = len(reps)
dissim = np.zeros((n, n))
dissim[np.triu_indices(n, 1)] = 1.0 - f_mat[np.triu_indices(n, 1)]
dissim += dissim.T  # symmetric

embedding = UMAP(n_components=2, metric="precomputed", random_state=42).fit_transform(dissim)
```

Dissimilarity conventions:
- **D-metric**: `max(D) − D`
- **F-metric**: `1 − F`

### Parallel workers (`n_jobs`)

- `n_jobs=-1` (default): all physical cores (uses `psutil.cpu_count(logical=False)`)
- `n_jobs=1`: serial — useful for deterministic profiling
- In `pairwise_overlap`: parallelises trie search within a single pair (chunk workers)
- In `pairwise_overlap_matrix`: parallelises across pairs (matrix workers)

### VDJBet harmonisation

The existing `count_overlap` / `compute_overlaps` / `make_reference_keys` /
`make_query_index` API used by `VDJBetOverlapAnalysis` is unchanged.
`pairwise_overlap` builds on top of the same `make_query_index` primitive.

## 18. Pgen And VDJBet Workflows

Use `OlgaModel` for sequence generation and pgen computation, and combine it
with `PgenGeneUsageAdjustment` and `VDJBetOverlapAnalysis` for overlap tests.

```python
from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.comparative.vdjbet import PgenBinPool, VDJBetOverlapAnalysis

model = OlgaModel(locus="TRB", seed=42)
target_gu = GeneUsage.from_repertoire(rep)
adjustment = PgenGeneUsageAdjustment(target_gu, seed=42)
pool = PgenBinPool("TRB", n=100_000, n_jobs=-1, seed=42, pgen_adjustment=adjustment)
analysis = VDJBetOverlapAnalysis(reference_rep, pool=pool, n_mocks=200, seed=42)
result = analysis.score(query_rep, match_v=True, match_j=True)
```

### 17.2 Monte-Carlo Pgen Pool (`McPgenPool`)

`McPgenPool` estimates Pgen by counting exact and inner-1mm matches in a large synthetic (or real)
control pool, using tcrtrie for fast Hamming-1 neighbor search.  It is the backbone of
`pgen_mode="mc"` in ALICE.

**`p_productive` calibration constants** (calibrated at N=30,000; stored in `_P_PRODUCTIVE_TABLE`):

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
p = get_p_productive("TRB", "human")   # → 0.2441 (generic fallback 0.20 for unknowns)

# Build a synthetic pool (uses p_productive table; ~2x faster than counting rejections)
pool = McPgenPool.build_synthetic(
    10_000_000, locus="TRB", species="human", n_jobs=8, seed=42, skip_ends=2,
    use_p_productive_table=True,   # default; set False to count actual rejections
)
# pool.n_productive   = M (productive sequences)
# pool.n_total        = M + K (all rearrangement attempts, including non-productive)
# pool.p_productive   = M / n_total (fraction of productive events)

# Bulk Pgen estimation
pgens_exact = pool.pgen_exact_bulk(cdr3_list)          # O(1) per seq via Counter
pgens_1mm   = pool.pgen_1mm_bulk(cdr3_list, n_jobs=8)  # tcrtrie Hamming-1 + inner-pos filter

# Build from real repertoire (for Q-factor analysis)
real_pool = McPgenPool.build_real(real_cdr3_list, locus="TRB", species="human")
# pgen_real = matches / n_control (no productive-fraction correction)
# Q-factor  = pgen_real / pgen_olga  (thymic-selection enrichment)

# Session-level cache (same pool reused across ALICE samples)
pool = get_or_build_mc_pool(locus="TRB", species="human", n=10_000_000, seed=42)
clear_mc_pool_cache()  # release memory
```

**MC Pgen normalisation:**  
OLGA analytical Pgen is defined over ALL rearrangements (productive + non-productive).
When generating M productive sequences, OLGA internally rejects K non-productive events.
`pgen_mc = n_matches / (M + K)` uses the tracked total to match the OLGA denominator.  
Formula: `pgen_mc(seq) = match_count / n_total_rearrangements`.

**Generating sequences with rejection counting:**

```python
model = OlgaModel(locus="TRB", species="human")
seqs, n_total = model.generate_sequences_counted(10_000_000, n_jobs=8, seed=42)
# n_total = M + K; seqs = productive sequences only
pool = McPgenPool(seqs, n_total, skip_ends=2, locus="TRB", species="human")
```

`OlgaModel` performance notes:

- No per-sequence Pgen caching. Performance comes from a **persistent `multiprocessing.Pool`** that loads the OLGA model once per worker; the pool is reused across all `compute_pgen_junction_aa_bulk` calls on the same `OlgaModel` instance.
- `compute_pgen_junction_aa_bulk(seqs, max_mismatches=0, n_jobs=N)` spawns `N` workers at first call; subsequent calls reuse the same pool (zero spawn overhead for repeated calls on 12+ ALICE samples).
- `generate_sequences_counted(n, n_jobs, seed)` returns `(seqs, n_total_rearrangements)` for MC Pgen denominator calibration.
- `compute_pgen_junction_aa_1mm(seq)` uses OLGA's vectorized 1-mismatch path; ~18× slower than exact.
- `OlgaModel.gen_model` exposes the underlying `GenerativeModelVDJ/VJ` for direct model marginals.
- Typical throughput at `n_jobs=8`: ~1000 seqs/s exact Pgen, ~90 seqs/s 1mm Pgen, ~270K seqs/s generation.
- **Lifecycle**: call `model.close()` or use `with OlgaModel(...) as model:` to guarantee pool teardown.

## 18.1 VDJBet YF Shortcuts (reusable workflow API)

Use the high-level helpers in ``mir.comparative.vdjbet_workflow`` to avoid
copying large notebook blocks:

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
usage = compute_olga_usage_adjustment(
    yfv_gu,
    seed=42,
    olga_usage_n=1_000_000,
    n_jobs=8,
    count_mode="count_rearrangement",
    pseudocount=1.0,
)

real = build_real_control_analysis(
    reference_rep,
    yfv_gu,
    seed=42,
    count_mode="count_rearrangement",
    pseudocount=1.0,
    pool_size=100_000,
    n_mocks=100,
    n_jobs=8,
)

diag = compute_bin_alignment_diagnostics(real.analysis)
df_res = score_samples_dataframe(real.analysis, samples)

pool_s, analysis_s, df_synth, x_scale, df_synth_scaled = build_synthetic_comparison(
  reference_rep,
  samples,
  pgen_adj_olga=usage.pgen_adj_olga,
  pool_size=100_000,
  n_mocks=100,
  n_jobs=8,
  seed=42,
  df_res_real=df_res,
)
```

Recommended defaults for reproducible runs:

- ``seed=42``
- ``pool_size=100_000`` for notebook iteration speed, ``1_000_000`` for final analyses
- ``n_mocks=100`` for exploratory runs, ``200+`` for stable tail p-values
- ``n_jobs`` set to available cores but avoid oversubscription in shared environments

## 19. Plotting Standards (publication-ready)

Use these defaults for all notebook and report figures.

Typography and export:

- Prefer vector output (SVG/PDF) for manuscripts and slides.
- Use consistent font families in all panels (e.g. Source Sans Pro, Helvetica, Arial fallback).
- Keep body text in the 8-10 pt range for multi-panel figures.
- Match axis/title/legend sizes across panels.

Theme and panel style:

- Use light backgrounds with subtle panel gridlines (alpha ~0.15-0.25).
- Keep frame lines thin and consistent.
- Minimize chart junk; favor clean panel labels and short titles.

Color strategy (Nature/Science style):

- Use restrained palettes with high contrast and colorblind safety.
- Keep one semantic mapping per color across all panels.
- Avoid red-green pairings for binary contrasts; prefer blue/orange or blue/magenta.

Bar/stacked/overlay plots:

- Keep identical bin width across compared groups.
- For grouped bars, use a position_dodge-style layout (side-by-side, same width).
- For overlays, show outlines or transparency so both groups remain visible.

Scatter/volcano labeling:

- Ensure labels are readable at final figure size.
- Use repel or position_dodge-type label placement to reduce overlap.
- Label only key points (top hits/outliers), not all points.

Distribution plots:

- Pair boxplots/violin plots with individual points (ggbeeswarm-like jitter).
- Keep jitter width small and deterministic where possible (seeded randomness).

Graph/network layouts:

- Avoid overclumped full-graph rendering with all edges visible.
- For large graphs: show a node-only overview + zoom-in inset with edges.
- Prefer spring/charge or MDS layouts depending on structure and interpretability.
- If needed, sample a representative subgraph by component or degree quantile.

Multi-panel composition:

- Use a compact grid with shared axes where meaningful.
- Align panel extents, labels, and legends.
- Reserve white space for panel letters and concise captions (Nature/Science style).

VDJBet notebook-specific plotting tips:

- Keep day-aligned x-axes consistent across all donor/replica panels.
- When comparing real vs mock nulls, keep mock boxplot widths/offsets fixed in every panel.
- Use the same y-axis transform (raw or log2) for directly compared metrics.

## 20. SampleRepertoire Construction

`SampleRepertoire` organises multiple loci for one donor/timepoint. Build it
from a flat clonotype list rather than pre-built locus repertoires wherever
possible:

```python
from mir.common.clonotype import Clonotype
from mir.common.repertoire import SampleRepertoire

clonotypes = [
    Clonotype(junction_aa=cdr3, locus="TRB", v_gene=v, duplicate_count=cnt)
    for cdr3, v, cnt in rows
]
sample = SampleRepertoire.from_clonotypes(clonotypes, sample_id="donor1")
```

Notes:

- `SampleRepertoire.__init__` takes `loci: dict[str, LocusRepertoire]`; do not
  pass `segments=` or `runs=`.
- When merging multiple sequencing runs into one sample, concatenate all
  clonotype lists from each run and pass the merged list to `from_clonotypes`.
- AIRR TSV files from SRA do not always contain a `locus` column; infer it from
  the first four characters of `v_call` (e.g. `"TRBV…"` → `"TRB"`).

## 21. TCREMP Embeddings

Use `TCREmp` from `mir.embedding.tcremp` to embed clonotypes as distance vectors
against a fixed set of prototype clonotypes.  Each clonotype is represented as
`[v_1, j_1, junc_1, ..., v_K, j_K, junc_K]` where triplets correspond to the K
prototypes and distances use BLOSUM62: `d(a,b) = s(a,a) + s(b,b) − 2·s(a,b)`.

```python
from mir.embedding.tcremp import TCREmp
from mir.common.clonotype import Clonotype

# Build from library defaults (computes all pairwise germline distances once)
model = TCREmp.from_defaults(
    species="human",
    locus="TRB",
    n_prototypes=1000,          # first 1 000 of 10 000 bundled prototypes
    junction_method="fixed_gap", # or "biopython" for full DP alignment
)

# Embed a list of Clonotype objects
clonotypes = [
    Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF"),
    Clonotype(v_gene="TRBV20-1*01", j_gene="TRBJ1-1*01", junction_aa="CSARDSSYEQYF"),
]
X = model.embed(clonotypes, n_jobs=4)  # shape: (2, 3000), dtype: float32
```

Useful properties:

- `model.n_prototypes` — number of prototypes (K)
- `model.embedding_dim` — total vector length (3·K)
- `model.locus`, `model.species` — canonical identifiers
- `model.prototypes` — Polars DataFrame with columns `v_gene`, `j_gene`, `junction_aa`

Germline-only aligner (multi-locus, used internally by TCREmp):

```python
from mir.common.gene_library import GeneLibrary
from mir.distances.aligner import GermlineAligner

lib = GeneLibrary.load_default(loci={"TRB", "TRA"}, species={"human"})
ga = GermlineAligner.from_library(lib, loci=["TRB", "TRA"])

# O(1) lookup: pre-computed distance (0 for identical genes)
d = ga.gene_dist("TRB", "TRBV10-1*01", "TRBV10-2*01")
```

All supported species/loci:

| Species | Loci |
|---------|------|
| human   | TRA, TRB, TRG, TRD, IGH, IGK, IGL |
| mouse   | TRA, TRB |

Performance (Apple M3, human TRB, K=1000 prototypes):

| Config             | Throughput          | Notes |
|--------------------|---------------------|-------|
| `n_jobs=1` (C)     | ~25 000 clono/s     | default, optimal on macOS |
| `n_jobs=8` (spawn) | ~10 000 clono/s     | spawn overhead dominates |

Use `n_jobs=1` (the default) on macOS/ARM. Multiprocessing with `spawn` is slower
at all practical batch sizes due to process startup cost (~3 s per batch).
On Linux with `fork`, multi-process may help for very large batches.

Key implementation notes:

- Germline distances are pre-computed at construction (O(n²) BioAlignerWrapper calls);
  embed time uses numpy matrix row lookups (O(1) per gene).
- Proto CDR3 self-scores are cached at construction; CDR3 distances per clonotype are
  computed by `CDRAligner.score_batch` — a single C call that loops over all K refs
  in C (~25× faster than K separate Python→C calls).
- Genes absent from the library (pseudogenes, missing alleles) silently receive the
  maximum observed distance for that locus/gene-type via `gene_dist` fallback.
- Output is `float32` and TensorFlow/Keras-compatible.
- Prototype files live in `mir/resources/prototypes/`; regenerate with
  `python mir/resources/prototypes/generate_prototypes.py`.

Embedding quality (R² between sequence-space and latent-space distances, 1000×1000):

| Correlation metric | Value |
|--------------------|-------|
| Pearson R²         | 0.57  |
| Spearman ρ         | 0.73  |

Per-component R² vs total sequence distance: V=0.47, J=0.16, CDR3=0.55.  CDR3
variability is the strongest single predictor of embedding distance.

## 21.1 Embedding Diagnostics — DBSCAN Clustering

Use `analyze_embedding_dbscan` from `mir.utils.embedding_diagnostics` to
standardize, PCA-reduce, and DBSCAN-cluster a raw embedding in one call.

```python
from mir.utils.embedding_diagnostics import (
    analyze_embedding_dbscan,
    majority_vote_cluster_predictions,
    classification_scores_by_label,
)

# X_raw: raw TCREmp embedding, shape (n_samples, n_features)
# labels: ground-truth epitope labels per row
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
#   eps_selector_meta — diagnostic dict with knee_found, eps_floor, eps_cap, etc.

# Derive per-point predicted labels and compute classification scores:
predicted = majority_vote_cluster_predictions(labels, result["clusters"])
scores = classification_scores_by_label(labels, predicted)
# scores keys: accuracy, macro_f1, weighted_f1, per_label (list of dicts)
```

**Eps selection — `select_eps_kneedle_stable` algorithm:**

1. Compute the sorted 4-NN distance curve `kth` of the L2-normalised PCA embedding.
2. Set `eps = kth[q_floor]` (default `q_floor=0.40`) as the safe minimum.
3. Run `KneeLocator` on the narrow window `[q_floor, q_floor + knee_fraction*(q_cap−q_floor)]`
   (default: ≈`[0.40, 0.45]`).  Accept the knee only if it falls strictly within that window.
4. For flat k-NN curves (typical in TCREmp embeddings) no knee is found; the selection
   stays at the floor quantile.

The floor `q_floor=0.40` was cross-validated on five balanced VDJdb TRB epitope subsets
(n≈3 000 each): it gives the highest minimum quality margin across all subsets and keeps
retention ≈ 0.62, purity ≈ 0.48.

Key parameters for `analyze_embedding_dbscan`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `pca_variance_threshold` | `0.90` | Cumulative variance retained |
| `q_floor` | `0.40` | Lower quantile bound for eps (empirically validated) |
| `q_cap` | `0.65` | Upper quantile cap |
| `knee_fraction` | `0.20` | Fraction of `[q_floor, q_cap]` to search for a knee |
| `min_samples` | `3` | DBSCAN min_samples |
| `eps_selection_mode` | `"stable_kneedle"` | Use `"kneedle"` for legacy full-range mode |

**Dimensionality scaling note:** The quantile approach is inherently dimensionality-adaptive
— absolute eps grows with n_comp (0.12 at n_comp=10, 0.49 at n_comp=100 on TRB/VDJdb),
but the quantile boundary stays constant.  No n_comp or variance-explained correction is
needed.

## 22. Practical Defaults

- Use `RepertoireDataset.from_folder_polars(...)` for real multi-sample loads.
- Strip alleles for most comparative analyses unless allele-specific behavior is the point of the analysis.
- Reuse prebuilt controls outside inner benchmark loops.
- When you need only V or J views after a VJ correction run, derive them via `marginalize_batch_corrected_gene_usage(...)` instead of ad hoc notebook code.

## 23. CDR3 Sequence Logos and Motif PWMs

`mir.biomarkers.motif_logo` — Shannon IC logos and OLGA-background-normalised selection logos.

### Scientific purpose

V-gene and J-gene templates encode conserved residues at CDR3 ends (N-terminal Cys,
J-gene STDTQYF stretch).  A plain IC logo shows these germline residues as the tallest
columns, hiding the antigen-specific motif.  Subtracting an OLGA background for the
**same V/J/length** removes the germline signal: `h_sel ≈ 0` at germline positions,
`h_sel >> 0` at antigen-driven positions (Pogorelyy et al. 2019, PLoS Biol.).

### CDR3 omega loop geometry

V-gene encodes the first ~5 residues; J-gene encodes the last ~4; the centre
(D-gene + N-additions) varies in length.  CDR3s of **different lengths are NOT
linearly aligned** — they share the terminals but insert/delete at the centre.
Aggregate profiles must therefore use **fractional position** `p / (L−1)`:
- 0 → conserved N-terminal Cys
- 1 → conserved C-terminal Phe/Trp
- 0.5 → approximate hypervariable centre

### Key formulas

| Logo type | Formula | Notes |
|---|---|---|
| IC logo | `h_IC[p,a] = f[p,a] · (log₂20 + Σₐ f·log₂f)` | Always ≥ 0 (bits) |
| Selection logo | `h_sel[p,a] = f[p,a] · log₂(f[p,a] / f_bg[p,a])` | Negative = depleted |
| motif_pwms height.I | IC / log₂(20) | [0,1] scale — **not bits**; multiply by log₂20 to convert |
| motif_pwms height.I.norm | −Σₐ f·ln(f_bg) / ln(20) / 2 | Cross-entropy, always ≥ 0 |

### Build a PWM from raw sequences

```python
from mir.biomarkers.motif_logo import compute_pwm, compute_logo, get_vj_background

pwm = compute_pwm(sequences, pseudocount=0.5)   # → pos, aa, count, frequency
logo = compute_logo(pwm)                         # adds ic_height (bits)
bg   = get_vj_background(pwms, v_gene="TRBV19*01", j_gene="TRBJ2-7*01",
                          length=13, species="HomoSapiens", gene="TRB")
logo_bg = compute_logo(pwm, background=bg)       # adds ic_height + bg_height
```

`compute_pwm` filters to the modal CDR3 length; set `length=` to override.
Always pass `species=` and `gene=` to avoid mixing TRA/TRB or mouse/human OLGA pools.

### Two background regimes

| Regime | Function | Removes |
|---|---|---|
| Per-VJ-len | `get_vj_background(v, j, len, species, gene)` | V-gene **and** J-gene germline |
| All-VJ aggregate | `aggregate_vj_background(pwms, length=L, species=S, gene=G)` | Length-composition bias only |

```python
from mir.biomarkers.motif_logo import aggregate_vj_background

agg_bg = aggregate_vj_background(pwms, length=13, species="HomoSapiens", gene="TRB")
# Returns pl.DataFrame[pos, aa, frequency] — weighted average over all VJ clusters
# of the given length.  Returns None if no matching clusters.
```

`get_vj_background` picks the cluster with the largest `total.bg` for the matching
V/J/length; prefix matching (strip allele suffix) is tried if exact match fails.

### Automated per-VJ-len logos from ALICE / TCRNET hits

`build_motif_logos_vj` is the recommended entry point for ALICE / TCRNET output.
It groups by (V, J, length), builds a per-VJ-len logo with matched OLGA background
for each group, and adds one all-VJ aggregate logo keyed `(None, None, len)`.

```python
from mir.biomarkers.motif_logo import build_motif_logos_vj
import polars as pl

# hits_df must have columns: junction_aa, v_gene, j_gene
logos = build_motif_logos_vj(
    hits_df,
    pwms,
    species="HomoSapiens",
    gene="TRB",
    min_seqs=5,         # skip groups with fewer sequences
    pseudocount=0.5,
)
# Returns {(v, j, len): logo_df, ..., (None, None, len): logo_df, ...}
# Each logo_df has columns: pos, aa, count, frequency, ic_height, bg_height

# Typical usage
vj_logo  = logos.get(("TRBV9", "TRBJ2-3", 15))
agg_logo = logos.get((None, None, 15))
```

### Load pre-computed cluster logos from motif_pwms.txt.gz

```python
from mir.biomarkers.motif_logo import load_motif_pwms, pwm_from_motif_pwms

pwms = load_motif_pwms(path)                          # full cluster table
logo = pwm_from_motif_pwms(pwms, "H.B.GILGFVFTL.1")  # pos/aa/ic_height/bg_height
```

`motif_pwms.txt.gz` is in `isalgo/airr_benchmark` on HuggingFace (`vdjdb/**`).
Key columns: `cid`, `csz`, `species`, `gene`, `antigen.epitope`, `v.segm.repr`,
`j.segm.repr`, `len`, `pos`, `aa`, `freq`, `freq.bg`, `height.I`, `height.I.norm`.

**Sparse storage**: `freq.bg` stores only non-zero residues per position (implicit
zero for missing AAs). `height.I` is in [0,1]-normalised scale, not bits.

### Plotting

```python
from mir.biomarkers.motif_logo import plot_motif_logos, plot_logo, BIOCHEMISTRY_COLORS

fig, axes = plot_motif_logos(
    logo_with_bg,
    v_gene="TRBV19*01",
    j_gene="TRBJ2-7*01",
    n_seqs=896,
    title="GILGFVFTL (Influenza A, HLA-A*02:01)",
)
# axes[0] = IC logo (always ≥ 0); axes[1] = selection logo (can be negative)
# V/J gene label appears ONLY in the figure suptitle, not around the axes.
# Letters sorted ascending so the tallest letter is drawn on top (WebLogo convention).
```

`BIOCHEMISTRY_COLORS` maps all 20 amino acids to 5 colour categories matching
Pogorelyy et al. 2019 Fig 2e:
- Aromatic: W, F, Y, H (purple)
- Nonpolar aliphatic: A, V, I, L, M, G, P (green)
- Polar: S, T, N, Q, C (yellow)
- Negatively charged: D, E (blue)
- Positively charged: K, R (red)

### Aggregate cluster IC/entropy profiles

```python
from mir.biomarkers.motif_logo import compute_cluster_profiles

# Per-position IC, H (entropy), I_norm for all csz>=30 clusters, TRB only
profiles = compute_cluster_profiles(pwms, min_csz=30, gene="TRB")
# Columns: cid, species, gene, antigen.epitope, v/j.segm.repr, len, csz, pos,
#          IC (bits), H (bits), I_norm (VDJdb-motifs cross-entropy)

# Fractional-position profile (p/(L-1) maps 0→Cys, 1→Phe/Trp)
profiles_frac = profiles.with_columns(
    (pl.col("pos") / (pl.col("len") - 1)).alias("frac_pos")
)
```

Formula: `IC = Σₐ height.I · log₂(20)` (position-level); `H = log₂(20) − IC`.
`I_norm = Σₐ height.I.norm` (pre-stored cross-entropy, always ≥ 0).

### Important cluster IDs for reference

| Cluster | Epitope | V | J | len | csz |
|---------|---------|---|---|-----|-----|
| H.B.GILGFVFTL.1 | GILGFVFTL (InfluenzaA, HLA-A*02) | TRBV19*01 | TRBJ2-7*01 | 13 | 896 |
| H.B.GILGFVFTL.4 | GILGFVFTL | TRBV19*01 | TRBJ1-5*01 | 13 | 129 |

The B27 AS CASSVGL[YF]STDTQYF motif is NOT in motif_pwms — use VDJdb
TRBV9/TRBJ2-3/len=15 sequences with `get_vj_background(..., v_gene="TRBV9*01",
j_gene="TRBJ2-3*01", length=15, species="HomoSapiens", gene="TRB")`.

**B27 AS analysis (Fig 2e reproduction)** — use pre-computed ALICE results, not VDJdb proxy:
```python
import pickle
from pathlib import Path
from mir.biomarkers.alice import alice_hit_clusters
from mir.biomarkers.motif_logo import get_vj_background, compute_pwm, compute_logo

# Load pre-computed ALICE cache (tuple: raw_results, annotated_hits)
_cache = pickle.load(open("tmp/_as_alice_cache.pkl", "rb"))
as_hits_dict = _cache[1]   # {donor_id: pd.DataFrame}
AS_DONOR_META = {1: "B27_pos", 2: "B27_pos", 3: "B27_neg", 4: "B27_pos"}
b27_pos_donors = [k for k, v in AS_DONOR_META.items() if v == "B27_pos"]
as_b27pos = pd.concat([as_hits_dict[d] for d in b27_pos_donors], ignore_index=True).drop_duplicates("junction_aa")

# Filter to TRBV9/TRBJ2-3/len=15 ALICE hits
alice_15 = as_b27pos[(as_b27pos.v_gene.str.startswith("TRBV9")) &
                     (as_b27pos.j_gene.str.startswith("TRBJ2-3")) &
                     (as_b27pos.junction_aa.str.len() == 15)]
as_bg = get_vj_background(pwms, v_gene="TRBV9*01", j_gene="TRBJ2-3*01",
                           length=15, species="HomoSapiens", gene="TRB")
# Selection logo: CASS (pos 1-4) and STDTQYF (pos 9-15) collapse to ≈0;
# VGL[YF] (pos 5-8) shows the antigen-driven enrichment.
logo_alice = compute_logo(compute_pwm(alice_15.junction_aa.tolist()), background=as_bg)
```

### Background from real or synthetic control (without motif_pwms.txt.gz)

`get_vj_background_from_control` builds a VJ/length PWM background directly from a
ControlManager DataFrame — useful when `motif_pwms.txt.gz` has no entry for a
specific VJ/length, or to validate against an empirically measured control.

```python
from mir.biomarkers.motif_logo import get_vj_background_from_control
from mir.common.control import ControlManager

cm = ControlManager()
ctrl_real  = cm.load_control_df("real",      "human", "TRB")  # ~28M rows
ctrl_synth = cm.load_control_df("synthetic", "human", "TRB", n=100_000)

bg_real = get_vj_background_from_control(
    ctrl_real, v_gene="TRBV9", j_gene="TRBJ2-3", length=15,
    min_seqs=100,   # returns None if fewer sequences match
)
bg_synth = get_vj_background_from_control(
    ctrl_synth, v_gene="TRBV9", j_gene="TRBJ2-3", length=15,
    min_seqs=20,    # lower threshold for synthetic controls
)
# Both return pl.DataFrame[pos, aa, frequency] or None
logo = compute_logo(pwm, background=bg_real)
```

Background correlations across 158 matched VJ/len combos (43,580 frequency pairs):
- `motif_pwms` vs real control: R = 0.97
- `motif_pwms` vs synthetic 100K: R = 0.96
- Real vs synthetic: R = 0.98

Use `min_seqs=20` for synthetic controls (small pool); `min_seqs=100` for real repertoires.

### Mixed-length terminal-anchored logos

`build_terminal_anchored_logo` combines CDR3s of different lengths into one display by
anchoring the first `n_term` and last `c_term` positions. Background subtraction happens
in the original linear CDR3 coordinate space (per CDR3 length), THEN positions are
mapped to the terminal display — architecturally correct and required for valid h_sel.

```python
from mir.biomarkers.motif_logo import build_terminal_anchored_logo
import polars as pl

# sequences_df must have columns: junction_aa, v_gene, j_gene
seqs_pl = pl.from_pandas(hits_df[["junction_aa", "v_gene", "j_gene"]])
logo_anchored = build_terminal_anchored_logo(
    seqs_pl,
    pwms,               # motif_pwms DataFrame; or pass motif_pwms=None for IC-only
    n_term=8,           # anchor first 8 positions (V-gene end)
    c_term=8,           # anchor last 8 positions (J-gene end)
    species="HomoSapiens",
    gene="TRB",
)
# Returns pl.DataFrame[pos_label, aa, count, frequency, ic_height, bg_height]
# pos_label: "1","2",…,"n_term","gap","-c_term",…,"-1"
# Supports both motif_pwms and get_vj_background_from_control() backgrounds.
```

### De-novo motif discovery: per-VJ-len connected components

When running `alice_hit_clusters` for motif discovery, **always build CCs per VJ/len group**
separately. Building on all sequences at once creates one giant CC that mixes J-genes and
dilutes the motif signal.

```python
from mir.biomarkers.alice import alice_hit_clusters

# CORRECT: build CCs per (v_gene, j_gene, length) group
for v, j, L in top_vj_len_groups:
    sub = hits_df[(hits_df.v_gene.str.startswith(v)) &
                  (hits_df.j_gene.str.startswith(j)) &
                  (hits_df.junction_aa.str.len() == L)]
    clustered = alice_hit_clusters(sub)  # adds cluster_id column
    cc_sizes  = clustered.groupby("cluster_id").size().sort_values(ascending=False)
    top_seqs  = clustered[clustered.cluster_id == cc_sizes.index[0]]
    # Top CC for TRBV19/TRBJ2-7/len=13 shows 93% R at RS positions (pos 5-6)

# WRONG: calling on all sequences creates one giant mixed CC
# alice_hit_clusters(all_hits_df)  ← never do this for motif discovery
```

### Background pool size

≥ 1,000 OLGA sequences per VJ/length gives stable background frequencies (MAD < 0.002);
`motif_pwms.txt.gz` uses ~23,000 per combination (well above threshold for all cases).
