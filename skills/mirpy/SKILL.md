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
stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, n_jobs=4)
graph = build_edit_distance_graph(rep.clonotypes, metric="levenshtein", threshold=1, n_jobs=4)

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
- `notebooks/tcremp_vdjdb_analysis_paired.ipynb` demonstrates strict vs imputed paired analysis with cumulative PCA variance, bounded-kneedle eps selection, DBSCAN purity/retention/consistency summaries, and SLL epitope outlier diagnosis against paired/TRA-only/TRB-only embeddings.

## 10. ALICE Enrichment

Use `compute_alice` / `add_alice_metadata` from `mir.biomarkers.alice`.

```python
from mir.biomarkers.alice import compute_alice, AliceParams, AliceResult

# Returns AliceResult(table=pd.DataFrame, params=AliceParams) when as_table=True
result = compute_alice(
    rep,
    species="human",
    match_mode="vj",      # "none" | "v" | "j" | "vj"
    pgen_mode="1mm",      # "exact" (Hamming-0) | "1mm" (Hamming-1)
    pvalue_mode="poisson",         # "poisson" | "negative-binomial"
    pseudocount=0.0,               # added to n and N before p-value computation
    n_jobs=8,
)

# result.table columns:
#   sequence_id, locus, junction_aa, v_gene, j_gene,
#   n_neighbors, N_possible, pgen_raw, pgen,
#   expected_neighbors, fold_enrichment, p_value, q_value

# Filter at FDR < 0.05 (q_value is BH-corrected over all locus clonotypes)
hits = result.table[result.table["q_value"] < 0.05]
```

Metadata-first variant (writes results into clonotype metadata in-place):

```python
from mir.biomarkers.alice import add_alice_metadata

rep = add_alice_metadata(
    rep,
    species="human",
    match_mode="vj",
    pgen_mode="1mm",
    pvalue_mode="poisson",
    pseudocount=0.0,
    n_jobs=8,
)
# Metadata keys: alice_n, alice_N, alice_pgen_raw, alice_pgen,
#                alice_expected, alice_fold, alice_p_value, alice_q_value
```

Key behavior notes:

- ALICE computes neighborhood stats first, then OLGA Pgen values, then BH FDR; heavy parallel sections use multiprocess workers by default for true multi-core scaling.
- Raw OLGA Pgen is used directly as the generation probability — no V/J gene-usage conditioning. `match_mode` restricts which sequences count as neighbors but does not modify Pgen.
- P-value batch execution defaults to process workers (`MIRPY_ALICE_PVALUE_EXECUTOR=process`) with optional thread mode override via env var.
- `pvalue_mode="negative-binomial"` uses `NB(mu=N*pgen, dispersion=1)` — more conservative than Poisson for overdispersed data.
- `q_value` in the output table is BH-corrected over all clonotypes in the locus (before any frequency filtering).

## 11. Single-Cell 10x Paired Chains

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

## 11.1 Single-Cell 10x + CITE-seq Integration

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

## 12. 10x Benchmark And scirpy Concordance

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

## 13. Single-Cell Parsing, Repair, And Pairing Graphs

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

## 9.1 TCRNET Enrichment

Use `compute_tcrnet` / `add_tcrnet_metadata` from `mir.biomarkers.tcrnet`.

TCRNET compares sample neighborhoods to a control (real or synthetic) using
binomial or beta-binomial statistics.

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
    normalize_control_vj_usage=False,  # resample control to match sample V/J usage
    n_jobs=4,
)

# result.table columns:
#   sequence_id, locus, junction_aa, v_gene, j_gene,
#   n_neighbors, N_possible,
#   m_control_neighbors, M_control_possible,
#   sample_density, control_density,
#   fold_enrichment, p_value, q_value

# Filter at FDR < 0.05
hits = result.table[result.table["q_value"] < 0.05]
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
    n_jobs=4,
)
# Metadata keys: tcrnet_n, tcrnet_N, tcrnet_m, tcrnet_M,
#                tcrnet_sample_density, tcrnet_control_density,
#                tcrnet_fold, tcrnet_p_value, tcrnet_q_value
```

Key behavior notes:

- p-value: `P(X >= n) where X ~ Binomial(N, (m+pc)/(M+pc))` (binomial mode). `beta-binomial` is overdispersed alternative using `BetaBinom(N, alpha=m+pc, beta=(M-m)+pc)`.
- Control pseudocount (`pseudocount`, default 1.0) is added to both `m` and `M` — equivalent to inserting one virtual match in the control.
- `q_value` in the output table is BH-corrected over all clonotypes in the locus.
- Use `normalize_control_vj_usage=True` to match control V/J gene usage distribution to the sample via resampling.

## 9.2 GLIPH-Style K-mer Enrichment (binomial)

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

## 10. Pgen And VDJBet Workflows

Use `OlgaModel` for sequence generation and pgen computation, and combine it
with `PgenGeneUsageAdjustment` and `VDJBetOverlapAnalysis` for overlap tests.

```python
from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.comparative.vdjbet import PgenBinPool, VDJBetOverlapAnalysis

model = OlgaModel(locus="TRB", seed=42)
target_gu = GeneUsage.from_repertoire(rep)
adjustment = PgenGeneUsageAdjustment(target_gu, seed=42)
pool = PgenBinPool("TRB", n=100_000, n_jobs=4, seed=42, pgen_adjustment=adjustment)
analysis = VDJBetOverlapAnalysis(reference_rep, pool=pool, n_mocks=200, seed=42)
result = analysis.score(query_rep, match_v=True, match_j=True)
```

`OlgaModel` performance notes:

- No per-sequence Pgen caching. Performance comes from a **persistent `multiprocessing.Pool`** that loads the OLGA model once per worker; the pool is reused across all `compute_pgen_junction_aa_bulk` calls on the same `OlgaModel` instance.
- `compute_pgen_junction_aa_bulk(seqs, max_mismatches=0, n_jobs=N)` spawns `N` workers at first call; subsequent calls reuse the same pool (zero spawn overhead for repeated calls on 12+ ALICE samples).
- `compute_pgen_junction_aa_1mm(seq)` uses OLGA's vectorized 1-mismatch path (`compute_hamming_dist_1_pgen`); 1mm pgen is ~18× slower per sequence than exact — use downsampling for large repertoires.
- `OlgaModel.gen_model` exposes the underlying `GenerativeModelVDJ/VJ` for direct access to model marginals (`PV`, `PDJ`, `PVJ`).
- For repeated ALICE runs on the same locus, the model-level cache in `_OLGA_MODEL_CACHE` (keyed by `(locus, species, seed, class)`) avoids model re-initialization.
- Typical throughput (single-core exact): ~135 seqs/s for TRB; 1mm: ~8 seqs/s. True scaling with `n_jobs=8`: ~900 seqs/s exact.

## 10.1 VDJBet YF Shortcuts (new reusable workflow API)

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

## 11. Plotting Standards (publication-ready)

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

## 12. SampleRepertoire Construction

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

## 13. TCREMP Embeddings

Use `TCREmp` from `mir.embedding.tcremp` to embed clonotypes as distance vectors
against a fixed set of prototype clonotypes.  Each clonotype is represented as
`[v_1, j_1, cdr3_1, ..., v_K, j_K, cdr3_K]` where triplets correspond to the K
prototypes and distances use BLOSUM62: `d(a,b) = s(a,a) + s(b,b) − 2·s(a,b)`.

```python
from mir.embedding.tcremp import TCREmp
from mir.common.clonotype import Clonotype

# Build from library defaults (computes all pairwise germline distances once)
model = TCREmp.from_defaults(
    species="human",
    locus="TRB",
    n_prototypes=1000,       # first 1 000 of 10 000 bundled prototypes
    cdr3_method="fixed_gap", # or "biopython" for full DP alignment
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

## 13. Practical Defaults

- Use `RepertoireDataset.from_folder_polars(...)` for real multi-sample loads.
- Strip alleles for most comparative analyses unless allele-specific behavior is the point of the analysis.
- Reuse prebuilt controls outside inner benchmark loops.
- When you need only V or J views after a VJ correction run, derive them via `marginalize_batch_corrected_gene_usage(...)` instead of ad hoc notebook code.
