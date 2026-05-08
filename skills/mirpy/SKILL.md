# mirpy Agentic Skills

This guide summarizes the current, reusable workflows in mirpy.
Examples below use the public APIs that match the current source tree, tests,
and documentation.

## Core Objects

- `Clonotype`: one rearrangement with junction sequence, V/D/J calls, and counts.
- `LocusRepertoire`: one locus worth of clonotypes for a single sample.
- `SampleRepertoire`: a sample that may contain multiple loci.
- `RepertoireDataset`: a collection of `SampleRepertoire` objects plus metadata.

## 1. Parse Repertoire Files

Use `VDJtoolsParser`, `AIRRParser`, `OldMiXCRParser`, `VDJdbSlimParser`, and
`OlgaParser` from `mir.common.parser`.

```python
from mir.common.parser import VDJtoolsParser
from mir.common.repertoire import LocusRepertoire

parser = VDJtoolsParser(sep="\t")
clonotypes = parser.parse("sample.tsv.gz")
rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
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
from mir.basic.token_tables import filter_token_table, tokenize_rearrangements
from mir.graph.token_graph import build_token_graph

# Edit-distance graph (Hamming or Levenshtein on junction_aa)
stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, n_jobs=4)
graph = build_edit_distance_graph(rep.clonotypes, metric="levenshtein", threshold=1, n_jobs=4)

# K-mer (token) graph filtered to RS-bearing 3-mers
table    = tokenize_rearrangements(rep.clonotypes, k=3)
rs_table = filter_token_table(table, kmer_pattern="RS")
g_rs     = build_token_graph(rep.clonotypes, rs_table)
```

Notes:

- `tokenize_rearrangements` accepts a `list[Clonotype]`; the former `Rearrangement` wrapper class has been removed.
- Use `Graph.are_adjacent()` instead of the deprecated `Graph.are_connected()` when querying igraph graphs directly.
- Trie-backed search is used for edit-distance graphs when available.
- For long amino-acid queries, exact brute-force fallback is used to avoid false negatives from bit-parallel limits.

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

## 8.1 GLIPH-Style K-mer Enrichment (binomial)

For GLIPH-like motif workflows, use clonotype-level token counts and a
binomial enrichment test against control background frequency.

```python
from mir.biomarkers.gliph import extract_v3mer_artifacts, normalize_control_v
from mir.biomarkers.kmer_stats import compare_kmer_counts

study_art = extract_v3mer_artifacts(study_df, threads=4, count_mode="clonotype")
ctrl_df = normalize_control_v(study_df, ctrl_pool_df, n=1_000_000, seed=42)
ctrl_art = extract_v3mer_artifacts(ctrl_df, threads=4, count_mode="clonotype")

comp = compare_kmer_counts(
  study_art.counts,
  ctrl_art.counts,
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
- If controls are only V-matched, inspect residual VJ drift; strong VJ imbalance can inflate enriched-kmer calls.

## 9. Pgen And VDJBet Workflows

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

### 9.1 VDJBet YF Shortcuts (new reusable workflow API)

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

## 12. Plotting Standards (publication-ready)

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

## 10. SampleRepertoire Construction

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

## 11. Practical Defaults

- Use `RepertoireDataset.from_folder_polars(...)` for real multi-sample loads.
- Strip alleles for most comparative analyses unless allele-specific behavior is the point of the analysis.
- Reuse prebuilt controls outside inner benchmark loops.
- When you need only V or J views after a VJ correction run, derive them via `marginalize_batch_corrected_gene_usage(...)` instead of ad hoc notebook code.
