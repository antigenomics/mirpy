# Single-Cell 10x Reference

This reference covers the mirpy single-cell 10x VDJ workflow, including sample loading, CITE-seq integration, parser-first repair pipelines, and benchmark tests. Key publications:

- Clonal expansion analysis: Pavlova et al. (2024) Front. Immunol. PMID:38633256 doi:10.3389/fimmu.2024.1321603
- Antigen annotation: Pogorelyy et al. (2019) Front. Immunol. PMID:31616409 doi:10.3389/fimmu.2019.02159

## 10x VDJ v1 Sample Loading

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
- Multi-chain cells are expanded deterministically by cartesian product per locus pair (e.g., `2x1` yields two paired clonotypes).
- `SingleCellRepertoire` keeps barcode -> pair_id links separate for future multimodal integration.

## 10x + CITE-seq Integration

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
- `SingleCellSample` packages the paired repertoire with CITE-seq matrix and parsed binder metadata (`column`, `hla`, `antigen.epitope`, `is_binder`).
- The VDJdb 10x sanity check normalizes HLA formatting and compares `(HLA, epitope)` targets against rows whose `reference.id` contains `10x`.
- Current dcode donors have two consistent residual unmatched targets (`CLGGLLTMV`, `LLMGTLGIVC`) that are tracked in tests.

## Parser-First Workflow With Repair

```python
from mir.common.single_cell import build_tenx_sample_from_cell_clonotypes
from mir.common.single_cell_parser import load_10x_vdj_v1_cell_clonotypes
from mir.common.single_cell_repair import cleanup_cell_clonotypes, impute_missing_chains
from mir.graph.single_cell_pairing import build_pairing_graph

raw = load_10x_vdj_v1_cell_clonotypes(
    "..._consensus_annotations.csv.gz",
    "..._all_contig_annotations.csv.gz",
    sample_id="sample1",
    check_is_cell=True,
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
- Optional `reuse_slave_per_master=True` reuses one synthetic slave clonotype per master clonotype.
- Synthetic rows use OLGA-based sequences with `duplicate_count=1`, `umi_count=1`.
- Cleanup keeps top-1 for `TRB`, `TRD`, `IGH` and conditionally top-2 for `TRA`/`TRG` and `IGK`/`IGL`.
- `consistency_only_on_synthetic_slave=True` limits consistency enforcement to synthetic slaves.

## Benchmark Tests

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

## Key Notebooks

- `notebooks/tcremp_10xdcode_analysis.ipynb` — TRA/TRB/paired TCREmp diagnostics with CITE-seq.
- `notebooks/tcremp_vdjdb_analysis_paired.ipynb` — strict vs imputed paired analysis, DBSCAN purity/retention/consistency.
