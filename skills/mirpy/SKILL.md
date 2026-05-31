---
name: mirpy
description: >
  Python library for AIRR-seq and immune repertoire analysis of T-cell (TCR)
  and B-cell (BCR) receptor sequences. Use when the user asks to: parse AIRR,
  VDJtools, Adaptive, or 10x repertoire files; compute diversity metrics
  (Shannon, Chao1, Hill curves, rarefaction); detect antigen-specific TCR
  clusters (ALICE Poisson enrichment, TCRNET MC-control, GLIPH k-mer motifs);
  compute TCREmp prototype embeddings or TCRdist distances; analyse pairwise
  sample overlap (Jaccard, D-metric, F-metric, Morisita-Horn); build CDR3
  motif logos (IC or selection logos against OLGA background); load or analyse
  single-cell 10x VDJ ± CITE-seq data; compute Pgen via OLGA or Monte-Carlo
  pools; or run VDJBet Pgen-matched overlap analysis. Load references/ files
  when detailed parameter tables, performance notes, or edge-case gotchas are
  needed.
license: GPL-3.0
compatibility: >
  Python 3.11+; mirpy-lib installed (pip install mirpy-lib or ./setup.sh from
  the repo root). A C/C++ build toolchain is required for compiled extensions.
  Shell is fish — use fish syntax in all terminal commands.
metadata:
  author: Immunosequencing Algorithms Laboratory (ISALGO lab)
  version: "1.1.1"
  docs: https://antigenomics.github.io/mirpy
  pypi: https://pypi.org/project/mirpy-lib/
  repo: https://github.com/antigenomics/mirpy
---

# mirpy Skills Guide

## Core Objects

- `Clonotype`: one rearrangement with junction sequence, V/D/J calls, and counts.
- `LocusRepertoire`: clonotypes for one locus in one sample.
- `SampleRepertoire`: multi-locus container for one donor / timepoint.
- `RepertoireDataset`: collection of `SampleRepertoire` objects plus metadata.

## 1. Parse Repertoire Files

```python
from mir.common.parser import (
    VDJtoolsParser, AIRRParser, AdaptiveParser,
    VDJdbSlimParser, OlgaParser, VDJdbFullPairedParser,
)
from mir.common.repertoire import LocusRepertoire

rep    = LocusRepertoire(clonotypes=VDJtoolsParser(sep="\t").parse("sample.tsv.gz"), locus="TRB")
rep    = AdaptiveParser(locus="TRB").parse_file("sample.adap.txt")
vdjdb  = VDJdbSlimParser().parse_file("assets/vdjdb.slim.txt.gz", species="HomoSapiens")
olga   = OlgaParser().parse_file("assets/olga_humanTRB.txt", locus="TRB")
```

> VDJdb: Shugay *et al.* (2018) *Nucleic Acids Res.* PMID:[28977646](https://pubmed.ncbi.nlm.nih.gov/28977646/);
> updated with SARS-CoV-2 data: Goncharov *et al.* (2022) *Nat. Methods* PMID:[35970936](https://pubmed.ncbi.nlm.nih.gov/35970936/)

## 2. Build Datasets

```python
from mir.common.repertoire_dataset import RepertoireDataset

dataset = RepertoireDataset.from_folder_polars(
    "data", parser=VDJtoolsParser(), metadata_file="metadata.tsv",
    file_name_column="file_name", sample_id_column="sample_id",
    metadata_sep="\t", n_workers=4,
)
```

Metadata rows are grouped by `sample_id` — split TRA/TRB files merge into one sample automatically.

## 3. Filter And Curate

```python
from mir.common.filter import filter_functional, filter_canonical
from mir.common.gene_library import GeneLibrary

imgt_lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="imgt")
functional_rep = filter_functional(rep, gene_library=imgt_lib)
```

## 4. Pool Samples

```python
from mir.common.pool import pool_samples

pooled = pool_samples(dataset, rule="aavj", include_sample_ids=True)
```

Supported rules: `ntvj` | `nt` | `aavj` | `aa`.

## 5. Gene Usage And Batch Correction

```python
from mir.basic.gene_usage import (
    GeneUsage, compute_batch_corrected_gene_usage,
    marginalize_batch_corrected_gene_usage, get_gene_usage_from_olga_model,
)

gu        = GeneUsage.from_repertoire(rep)
v_usage   = gu.v_fraction("TRB", count="duplicates", pseudocount=1.0)
corr_vj   = compute_batch_corrected_gene_usage(dataset, batch_field="batch_id", scope="vj")
v_marginal = marginalize_batch_corrected_gene_usage(corr_vj, scope="v")
```

Always consume corrected probabilities from `pfinal` (renormalised per sample/locus).

## 6. Downsampling

```python
from mir.common.sampling import downsample, resample_to_gene_usage

rep_small   = downsample(rep, 1_000, random_seed=42)
rep_matched = resample_to_gene_usage(rep, target_usage, scope="v", weighted=True, random_seed=42)
```

## 7. Graph And Neighborhood Analysis

```python
from mir.graph import compute_neighborhood_stats
from mir.graph.edit_distance_graph import build_edit_distance_graph
from mir.basic.token_tables import filter_token_table, tokenize_clonotypes
from mir.graph.token_graph import build_token_graph

stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, n_jobs=-1)
graph = build_edit_distance_graph(rep.clonotypes, metric="levenshtein", threshold=1, n_jobs=-1)

# Token (k-mer) graph — GLIPH-style RS-bearing 3-mers
table    = tokenize_clonotypes(rep.clonotypes, k=3)
rs_table = filter_token_table(table, kmer_pattern="RS")
g_rs     = build_token_graph(rep.clonotypes, rs_table)
```

V/J-restricted neighbour search (`match_v_gene=True`, `match_j_gene=True`) is 1.5–2× faster
than unrestricted on natural repertoires via the grouped-trie strategy.

`build_edit_distance_graph` returns an igraph Graph; vertices carry `name` (junction_aa),
`r_id`, `v_gene`, `j_gene`, `c_gene`. Use `g.vs["j_gene"]` directly.

## 8. Control Repertoires

```python
from mir.common.control import ControlManager

mgr = ControlManager()
mgr.ensure_synthetic_control("human", "TRB", n=1_000_000)
df_control = mgr.ensure_and_load_control_df("real", "human", "TRB")
```

Default cache: `~/.cache/mirpy/controls`. Override with `MIRPY_CONTROL_DIR`.

## 9. Prototype-Based Embeddings (TCREmp)

> Kremlyakova *et al.* (2025) *J. Mol. Biol.* PMID:[40368275](https://pubmed.ncbi.nlm.nih.gov/40368275/)

```python
from mir.embedding.tcremp import TCREmp, PairedTCREmp

model  = TCREmp.from_defaults("human", "TRB", n_prototypes=1000, junction_method="fixed_gap")
X      = model.embed(clonotypes, n_jobs=None)   # shape: (N, 3000), float32

paired = PairedTCREmp.from_defaults("human", "TRA_TRB", n_prototypes=500)
X_pair = paired.embed(paired_clonotypes)
```

Each clonotype → `[v_1, j_1, junc_1, …, v_K, j_K, junc_K]`.
Distance formula: `d(a,b) = s(a,a) + s(b,b) − 2·s(a,b)`.
Use `fixed_gap` (~25 M pairs/s) for production; `biopython` for full DP semantics (~270 K pairs/s).
`n_jobs=1` is the best default on macOS/ARM (spawn overhead dominates at all sizes).

Read [references/embeddings.md](references/embeddings.md) when you need:
`n_jobs` auto-selection logic, DBSCAN diagnostics (`analyze_embedding_dbscan`), eps selection
algorithm, VDJdb-full paired workflow, performance tables, or `select_eps_kneedle_stable` details.

## 10. Metaclonotypes

```python
from mir.biomarkers.metaclonotype_cluster import MetaclonotypeClusterConfig, cluster_metaclonotypes

cfg  = MetaclonotypeClusterConfig(method="edit_distance", graph_algo="leiden", min_cluster_size=2)
meta = cluster_metaclonotypes(rep, cfg)

cfg_dist  = MetaclonotypeClusterConfig(method="tcrdist", locus="TRB", max_distance=24.5)
meta_dist = cluster_metaclonotypes(rep, cfg_dist)
```

Methods: `"alice"` | `"tcrnet"` | `"tcrdist"` | `"edit_distance"` | `"tcremp"` | `"gliph"`.
Graph algorithms: `"components"` | `"leiden"` | `"louvain"`.
Embedding algorithms: `"dbscan"` | `"optics"`.

```python
from mir.common.metaclonotype import functional_diversity, functional_overlap_1

func_div     = functional_diversity(rep, meta)
func_overlap = functional_overlap_1(meta_a, meta_b, rep_a, rep_b)
```

## 11. Diversity Metrics, Hill Curves, Rarefaction

> VDJtools: Shugay *et al.* (2015) *PLoS Comput. Biol.* PMID:[26606115](https://pubmed.ncbi.nlm.nih.gov/26606115/)  
> Aging cohort (79 donors): Britanova *et al.* (2016) *J. Immunol.* PMID:[27183615](https://pubmed.ncbi.nlm.nih.gov/27183615/)

```python
from mir.common.diversity import summarize_clonotypes, hill_curve_clonotypes, rarefaction_curve_clonotypes

summary = summarize_clonotypes(rep.clonotypes)
hill    = hill_curve_clonotypes(rep.clonotypes)
rare    = rarefaction_curve_clonotypes(rep.clonotypes, m_steps=[10, 25, 50, 100], include_exact=True)
```

Summary fields: `abundance`, `diversity`, `singletons`, `doubletons`, `expanded` (>0.1%),
`hyperexpanded` (>1%), `chao1`, `gini_simpson`, `shannon`.

Notebook: `notebooks/aging_analysis.ipynb` — 79-donor aging cohort, rarefaction, F overlap.

## 12. Pairwise Overlap

```python
from mir.comparative.overlap import pairwise_overlap, pairwise_overlap_matrix

r  = pairwise_overlap(rep_a, rep_b, overlap_space="aavj", metric="hamming", threshold=1)
df = pairwise_overlap_matrix(reps, sample_ids=ids, metric="exact", n_jobs=-1)
```

Key result fields: `jaccard`, `d_similarity`, `f_similarity`, `morisita_horn`, `szymkiewicz_simpson`.
Dissimilarity for UMAP: `f_metric = 1 − f_similarity`, `d_metric = 1/d_similarity`.

Read [references/overlap.md](references/overlap.md) for the full field table, approximate
matching constraints, parallel worker semantics, and UMAP/MDS embedding patterns.

## 12.5. ALICE Enrichment

> Pogorelyy *et al.* (2019) *PLoS Biol.* PMID:[31194732](https://pubmed.ncbi.nlm.nih.gov/31194732/)

```python
from mir.biomarkers.alice import compute_alice, add_alice_metadata

result = compute_alice(
    rep, species="human",
    match_mode="vj",       # "none" | "v" | "j" | "vj"
    pgen_mode="mc",        # "exact" | "1mm" | "mc"  ← use "mc" for production
    mc_n_pool=10_000_000,
    n_jobs=8,
)
hits = result.table.filter(pl.col("q_value") < 0.05)
rep  = add_alice_metadata(rep, species="human", match_mode="vj", pgen_mode="mc")
```

Use `pgen_mode="mc"` (100–1000× faster than `"1mm"` after first pool build; paper-correct λ).
`pgen_mode="exact"` underestimates λ — **not recommended**.
V+J-restricted search is 1.5–2× faster than unrestricted (grouped-trie strategy).

Read [references/biomarkers.md](references/biomarkers.md) when you need:
full `AliceParams`, pgen_mode comparison table, MC pool internals, `alice_hit_clusters`,
TCRNET–ALICE relationship, or the GLIPH k-mer reference.

## 12.6. Clonotype Metadata Associations

```python
from mir.biomarkers.associations import (
    AssociationParams,
    associate_clonotype_metadata,
    build_public_clonotype_panel,
)

targets = build_public_clonotype_panel(samples, locus="TRB", min_sample_fraction=0.03)

res = associate_clonotype_metadata(
    samples,
    targets,
    metadata_field="COVID_status",
    metadata_value=["COVID", "healthy"],
    params=AssociationParams(count_mode="sample", test="auto"),
)

res_depth = associate_clonotype_metadata(
    samples,
    targets,
    metadata_field="COVID_status",
    metadata_value=["COVID", "healthy"],
    params=AssociationParams(count_mode="rearrangement", test="depth_glm"),
)
```

Use cases:

- Binary or multiclass association tests with BH/FDR correction.
- Paired-chain association via `associate_paired_clonotype_metadata`.
- Depth-aware mode (`test="depth_glm"`) for uneven sequencing depth.
- Co-occurrence screens with `associate_clonotype_cooccurrence`.

For the full COVID workflow (functional filtering, first batch correction,
sample re-normalization, Fisher + depth-aware scans, and reference concordance)
see `notebooks/covid19_biomarkers.ipynb`.

For the Vlasova 2026 SVM replication (log-frequency features, RBF-SVM, 1137 paired
donors, AUC=0.70 target) see `benchmarks/covid19_svm_benchmark.py` and
`tests/test_associations_covid19_benchmark.py::test_covid19_svm_classifier_auc`.

```python
# Minimal SVM pipeline (Vlasova 2026 approach)
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# X: (n_donors, n_biomarkers) log-frequency matrix
# y: (n_donors,) binary labels (1=COVID, 0=healthy)
X_log = np.log(X + 1e-7)
clf = SVC(kernel="rbf", probability=True, class_weight="balanced", C=1.0, gamma="scale")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob = cross_val_predict(clf, X_log, y, cv=cv, method="predict_proba")[:, 1]
auc = roc_auc_score(y, y_prob)  # target ≥ 0.70
```

> Vlasova *et al.* (2026) *Genome Med.* [DOI:10.1186/s13073-025-01589-4](https://doi.org/10.1186/s13073-025-01589-4)

## 13. Single-Cell 10x

> Clonal expansion analysis: Pavlova *et al.* (2024) *Front. Immunol.* PMID:[38633256](https://pubmed.ncbi.nlm.nih.gov/38633256/)

```python
from mir.common.single_cell import load_10x_vdj_v1_sample

sample = load_10x_vdj_v1_sample(
    consensus_annotations_path="…_consensus_annotations.csv.gz",
    all_contig_annotations_path="…_all_contig_annotations.csv.gz",
    sample_id="donor1",
)
print(sample.loaded_cell_count, sample.loaded_clonotype_count)
```

Supported locus pairs: `TRA_TRB`, `TRG_TRD`, `IGH_IGK`, `IGH_IGL`.

Read [references/single-cell.md](references/single-cell.md) when you need:
CITE-seq loading (`load_10x_vdj_v1_citeseq_sample`), parser-first workflow,
single-cell repair (`impute_missing_chains`, `cleanup_cell_clonotypes`), pairing graphs,
or VDJdb full paired parser.

## 14. TCRNET Enrichment

> Lupyr *et al.* (2025) *Brief. Bioinform.* PMID:[40996146](https://pubmed.ncbi.nlm.nih.gov/40996146/)

```python
from mir.biomarkers.tcrnet import compute_tcrnet

result = compute_tcrnet(
    rep, control=control_rep, species="human",
    metric="hamming", threshold=1,
    match_mode="none", pvalue_mode="binomial",  # default; use "vj" for V+J-restricted search
    pseudocount=1.0, q_factor=1.0, n_jobs=-1,
)
hits = result.table[result.table["q_value"] < 0.001]
```

TCRNET is purely MC-control — no OLGA Pgen calls.
Pass `q_factor ≈ 3–5` for synthetic OLGA controls; leave `1.0` for real controls.

Read [references/biomarkers.md](references/biomarkers.md) for full params, beta-binomial mode,
Q-factor estimation, TCRNET-as-ALICE recipe (100 M pool), and comparison table.

## 15. GLIPH K-mer Enrichment

```python
from mir.biomarkers.gliph import compare_gliph_token_incidence, extract_gliph_artifacts_batch_from_repertoire

families = ["v3", "pos3", "u3", "u4", "g4", "g5"]
ctrl_art = extract_gliph_artifacts_batch_from_repertoire(
    control_rep, families, trim_first=3, trim_last=4, build_mappings=False)
study_art = extract_gliph_artifacts_batch_from_repertoire(
    study_rep, families, trim_first=3, trim_last=4, build_mappings=False)
comp = compare_gliph_token_incidence(study_art["u3"], ctrl_art["u3"], test="binom")
```

Keep `trim_first=3`, `trim_last=4` (GLIPH defaults) identical for sample and control.
Read [references/biomarkers.md](references/biomarkers.md) for motif-family descriptions.

## 16. Pgen Computation (OLGA / McPgenPool)

> Pre-immune T cell landscape: Pogorelyy *et al.* (2018) *Genome Medicine* PMID:[30144804](https://pubmed.ncbi.nlm.nih.gov/30144804/)

```python
from mir.basic.pgen import OlgaModel, McPgenPool, get_or_build_mc_pool, clear_mc_pool_cache

with OlgaModel(locus="TRB", species="human") as model:
    pgens = model.compute_pgen_junction_aa_bulk(seqs, max_mismatches=0, n_jobs=8)

pool  = get_or_build_mc_pool(locus="TRB", species="human", n=10_000_000, seed=42)
pgens = pool.pgen_1mm_bulk(cdr3_list, n_jobs=8)
clear_mc_pool_cache()  # release memory
```

Always use `with OlgaModel(...) as m:` or `m.close()` — leaves a persistent worker pool alive.

Read [references/pgen-vdjbet.md](references/pgen-vdjbet.md) for `McPgenPool` internals,
`generate_sequences_counted`, `p_productive` calibration table, and VDJBet workflow.

## 17. CDR3 Motif Logos

> Selection logos vs OLGA background: Pogorelyy *et al.* (2019) *PLoS Biol.* PMID:[31194732](https://pubmed.ncbi.nlm.nih.gov/31194732/)

```python
from mir.biomarkers.motif_logo import (
    compute_pwm, compute_logo, get_vj_background,
    build_motif_logos_vj, load_motif_pwms,
)

pwms  = load_motif_pwms("motif_pwms.txt.gz")
bg    = get_vj_background(pwms, v_gene="TRBV19*01", j_gene="TRBJ2-7*01",
                           length=13, species="HomoSapiens", gene="TRB")
logo  = compute_logo(compute_pwm(sequences), background=bg)

# Batch per-(V, J, length) group from ALICE/TCRNET hits
logos = build_motif_logos_vj(hits_df, pwms, species="HomoSapiens", gene="TRB", min_seqs=5)
```

`height.I` in `motif_pwms` is [0,1]-normalised — **not bits**; multiply by log₂(20) to convert.

Read [references/motif-logos.md](references/motif-logos.md) for formula table, aggregate
backgrounds, terminal-anchored logos, de-novo motif discovery gotchas, and `BIOCHEMISTRY_COLORS`.

## 18. TCRdist

> TCR structural modeling: Shcherbinin *et al.* (2023) *Front. Immunol.* PMID:[37649481](https://pubmed.ncbi.nlm.nih.gov/37649481/)  
> Thymic selection signatures: Luppov *et al.* (2025) *Front. Immunol.* PMID:[41050667](https://pubmed.ncbi.nlm.nih.gov/41050667/)

```python
from mir.distances.tcrdist import TcrDist
import numpy as np

td   = TcrDist.from_defaults("TRB", "human", w_v=1.0, w_j=0.0, w_cdr3=3.0,
                              fixed_gaps=(3, 4, -4, -3))   # ~28 M pairs/s
d    = td.dist(cln1, cln2)
mat  = td.dist_matrix(queries, refs, n_jobs=4)
radii = td.compute_radius(hits, bg_clns, percentile=50, n_jobs=4)
meta  = td.find_metaclonotypes(rep, max_distance=float(np.percentile(radii, 25)), n_jobs=4)
```

Read [references/tcrdist.md](references/tcrdist.md) for gap modes (`"Mid"`, `None`),
V-gene/CDR3 weight rationale, parallel performance table, and metaclonotype API details.

## 19. Plotting Standards (publication-ready)

- Vector output (SVG/PDF) for manuscripts; 8–10 pt body text, consistent font.
- Light backgrounds, subtle gridlines (alpha ≈ 0.15–0.25), minimal chart junk.
- Colorblind-safe palettes (blue/orange preferred); avoid red-green binary contrasts.
- Use `BIOCHEMISTRY_COLORS` from `mir.biomarkers.motif_logo` for amino-acid plots.
- Scatter/volcano: repel-style label placement, key outliers only.
- Distribution plots: pair boxplots with jittered individual points.
- Multi-panel: shared axes where meaningful, compact grid, panel letters (Nature/Science style).

## 20. SampleRepertoire Construction

```python
from mir.common.repertoire import SampleRepertoire

clonotypes = [Clonotype(junction_aa=cdr3, locus="TRB", v_gene=v, duplicate_count=cnt) for ...]
sample = SampleRepertoire.from_clonotypes(clonotypes, sample_id="donor1")
```

Merge sequencing runs by concatenating clonotype lists before `from_clonotypes`.
AIRR SRA files often lack a `locus` column — infer from the first four chars of `v_call`.

## 21. Practical Defaults

- `RepertoireDataset.from_folder_polars()` for all multi-sample loads.
- Strip alleles (`split("*")[0]`) for most comparative analyses.
- Reuse prebuilt controls outside inner loops.
- V/J marginals: use `marginalize_batch_corrected_gene_usage()`, not ad-hoc groupby.
- ALICE + TCRNET: `match_mode="vj"` for both specificity and speed.
- Pgen: one `OlgaModel` instance per session (persistent pool, zero respawn overhead).
- For many-vs-pool overlap notebooks: use `many_vs_pool_overlap()` to avoid repeated trie setup.

## Gotchas

- `tokenize_clonotypes` takes `list[Clonotype]`; `tokenize_rearrangements` is a deprecated alias.
- Use `Graph.are_adjacent()` — `Graph.are_connected()` is deprecated.
- `motif_pwms` `height.I` is [0,1]-normalised, **not bits**.
- For motif discovery via `alice_hit_clusters`: build CCs **per (V, J, length) group** separately.
  Building on all sequences at once creates one giant mixed connected component.
- OLGA synthetic sequences are pre-thymic; pass `q_factor ≈ 3–5` for TCRNET with synthetic controls.
- `pgen_mode="exact"` underestimates λ in ALICE; use `"mc"` for all production runs.
- `n_jobs=1` is the best TCREmp default on macOS/ARM; spawn overhead dominates multiprocessing.
- `SampleRepertoire.__init__` accepts `loci: dict[str, LocusRepertoire]`; never pass `segments=`.
- Always activate the venv first: `source .venv/bin/activate.fish` (fish syntax — not `source .venv/bin/activate`).
- `VDJdbFullPairedParser(..., include_incomplete=True)` → run `impute_missing_chains` before pairing.
- Calling `add_alice_metadata` or `add_tcrnet_metadata` with `mc_n_pool > 0` builds the pool once; subsequent samples reuse cache automatically.

## Key References

| Topic | Citation |
|-------|----------|
| VDJtools / diversity indices | Shugay *et al.* (2015) *PLoS Comput. Biol.* PMID:[26606115](https://pubmed.ncbi.nlm.nih.gov/26606115/) |
| ALICE enrichment | Pogorelyy *et al.* (2019) *PLoS Biol.* PMID:[31194732](https://pubmed.ncbi.nlm.nih.gov/31194732/) |
| TCRNET / neighbourhood enrichment | Lupyr KR *et al.* (2025) *Brief. Bioinform.* PMID:[40996146](https://pubmed.ncbi.nlm.nih.gov/40996146/) |
| TCREmp prototype embeddings | Kremlyakova Y *et al.* (2025) *J. Mol. Biol.* PMID:[40368275](https://pubmed.ncbi.nlm.nih.gov/40368275/) |
| TCRdist distance metric | Dash P *et al.* (2017) *Nature* PMID:[28636592](https://pubmed.ncbi.nlm.nih.gov/28636592/) |
| GLIPH CDR3 motif clustering | Glanville *et al.* (2017) *Nature* PMID:[28636589](https://pubmed.ncbi.nlm.nih.gov/28636589/) |
| GLIPH2 (large-scale) | Huang *et al.* (2020) *Nat. Biotechnol.* PMID:[32341563](https://pubmed.ncbi.nlm.nih.gov/32341563/) |
| VDJdb database | Shugay M *et al.* (2018) *Nucleic Acids Res.* PMID:[28977646](https://pubmed.ncbi.nlm.nih.gov/28977646/) |
| VDJdb 2019 update | Bagaev *et al.* (2020) *Nucleic Acids Res.* PMID:[31588507](https://pubmed.ncbi.nlm.nih.gov/31588507/) |
| VDJdb SARS-CoV-2 update | Goncharov M *et al.* (2022) *Nat. Methods* PMID:[35970936](https://pubmed.ncbi.nlm.nih.gov/35970936/) |
| Antigen-specificity annotation | Pogorelyy MV & Shugay M (2019) *Front. Immunol.* PMID:[31616409](https://pubmed.ncbi.nlm.nih.gov/31616409/) |
| TCR aging dynamics | Britanova OV *et al.* (2016) *J. Immunol.* PMID:[27183615](https://pubmed.ncbi.nlm.nih.gov/27183615/) |
| Pre-immune TCR landscape | Pogorelyy MV *et al.* (2018) *Genome Med.* PMID:[30144804](https://pubmed.ncbi.nlm.nih.gov/30144804/) |
| COVID-19 TCR biomarker classifier | Vlasova EK *et al.* (2026) *Genome Med.* [DOI:10.1186/s13073-025-01589-4](https://doi.org/10.1186/s13073-025-01589-4) |
| TCR structural modeling | Shcherbinin *et al.* (2023) *Front. Immunol.* PMID:[37649481](https://pubmed.ncbi.nlm.nih.gov/37649481/) |
| TCRen structural prediction | Karnaukhov *et al.* (2024) *Nat. Comput. Sci.* PMID:[38987378](https://pubmed.ncbi.nlm.nih.gov/38987378/) |
| Thymic selection repertoire | Luppov *et al.* (2025) *Front. Immunol.* PMID:[41050667](https://pubmed.ncbi.nlm.nih.gov/41050667/) |
| Regulatory T cell repertoire | Feng *et al.* (2015) *Nature* PMID:[26605529](https://pubmed.ncbi.nlm.nih.gov/26605529/) |
| Clonal T cell tracking (cancer) | Shagina *et al.* (2026) *Cancer Immunol. Res.* PMID:[41843768](https://pubmed.ncbi.nlm.nih.gov/41843768/) |
| T cell clonal expansion (single-cell) | Pavlova *et al.* (2024) *Front. Immunol.* PMID:[38633256](https://pubmed.ncbi.nlm.nih.gov/38633256/) |
| Bystander Tfh/Tfr activation | Ritvo *et al.* (2018) *Proc. Natl. Acad. Sci. USA* PMID:[30158170](https://pubmed.ncbi.nlm.nih.gov/30158170/) |
