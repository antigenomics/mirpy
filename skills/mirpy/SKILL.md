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
token graphs.

```python
from mir.graph import compute_neighborhood_stats
from mir.graph.edit_distance_graph import build_edit_distance_graph

stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, n_jobs=4)
graph = build_edit_distance_graph(rep.clonotypes, metric="levenshtein", threshold=1, n_jobs=4)
```

Notes:

- Trie-backed search is used when available.
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

## 10. Practical Defaults

- Use `RepertoireDataset.from_folder_polars(...)` for real multi-sample loads.
- Strip alleles for most comparative analyses unless allele-specific behavior is the point of the analysis.
- Reuse prebuilt controls outside inner benchmark loops.
- When you need only V or J views after a VJ correction run, derive them via `marginalize_batch_corrected_gene_usage(...)` instead of ad hoc notebook code.
