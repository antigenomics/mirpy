# mirpy

[![PyPI](https://img.shields.io/pypi/v/mirpy-lib)](https://pypi.org/project/mirpy-lib/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/badge/docs-antigenomics.github.io-informational)](https://antigenomics.github.io/mirpy)

![mirpy logo](assets/mirpy_logo.png)

`mirpy` is a Python library for AIRR-seq and immune repertoire analysis.
It provides composable building blocks for parsing, filtering, comparing, and
characterising T-cell and B-cell receptor repertoires.

## Contents

- [Installation](#installation)
- [Module overview](#module-overview)
- [Quick start](#quick-start)
  - [Load a segment library](#load-a-segment-library)
  - [Parse a clonotype table](#parse-a-clonotype-table)
  - [Work with repertoires](#work-with-repertoires)
  - [Pool repertoires across samples](#pool-repertoires-across-samples)
- [Diversity metrics](#diversity-metrics)
- [Metaclonotypes](#metaclonotypes)
- [TCRdist](#tcrdist)
- [CDR3 Motif Logos](#cdr3-motif-logos)
- [ALICE and TCRNET](#alice-and-tcrnet)
- [Prototype-based embeddings (TCREmp)](#prototype-based-embeddings-with-tcremp)
- [Copilot Agent Workflow](#copilot-agent-workflow)
- [Resources](#resources)
- [Project status](#project-status)

---

## Installation

Requirements:

- Python 3.11+
- a C/C++ build toolchain for compiled extensions

Install from PyPI:

```bash
pip install mirpy-lib
```

Install from source (one-shot):

```bash
git clone https://github.com/antigenomics/mirpy.git
cd mirpy
pip install .
```

Install from source (editable development mode):

```bash
git clone https://github.com/antigenomics/mirpy.git
cd mirpy
./setup.sh
```

`setup.sh` already installs mirpy in editable mode.

Prefer `pip install mirpy-lib` for project usage.
Use the cloned repo setup when developing or running docs/notebooks locally.

---

## Module overview

| Package | Responsibilities |
| --- | --- |
| `mir.common` | Clonotypes, repertoires, parsers, segment libraries |
| `mir.distances` | Aligners, Hamming/Levenshtein search, graph utilities, TCRdist |
| `mir.basic` | Sampling, segment usage, alphabet helpers, Pgen utilities |
| `mir.graph` | Edit-distance graphs, neighbourhood enrichment, token graphs, single-cell pairing |
| `mir.embedding` | Prototype embeddings: TCREmp, PairedTCREmp |
| `mir.comparative` | Pairwise overlap metrics (Jaccard, D, F, Morisita-Horn), trie-accelerated approximate matching, VDJBet Pgen-matched null distributions |
| `mir.biomarkers` | ALICE enrichment, TCRNET, CDR3 sequence logos |
| `mir.utils` | Embedding diagnostics, shared memory, notebook asset helpers |

---

## Quick start

### Load a segment library

```python
from mir.common.gene_library import GeneLibrary

lib = GeneLibrary.load_default(
    loci={"TRA", "TRB"},
    species={"human"},
    source="imgt",
)
```

If a requested organism/locus pair is absent from the default local segment
file, mirpy downloads the missing V and J segments from IMGT and appends them
automatically.

### Parse a clonotype table

```python
from mir.common.parser import VDJtoolsParser

parser = VDJtoolsParser(sep="\t")
clonotypes = parser.parse("example.tsv")
```

Supported parsers: `VDJtoolsParser`, `AIRRParser`, `AdaptiveParser`,
`VDJdbFullPairedParser`, and others in `mir.common.parser`.

### Work with repertoires

```python
from mir.common.repertoire import LocusRepertoire

repertoire = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
print(repertoire.duplicate_count)   # total read count
print(repertoire.clonotype_count)   # unique clonotypes

# Functional / canonical filtering using IMGT annotations
from mir.common.filter import filter_functional, filter_canonical
from mir.common.gene_library import GeneLibrary

imgt_lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="imgt")
functional_rep = filter_functional(repertoire, gene_library=imgt_lib)
canonical_rep  = filter_canonical(repertoire, gene_library=imgt_lib)
```

### Pool repertoires across samples

```python
from mir.common.pool import pool_samples

# Pool by amino-acid CDR3 + V/J; retain contributing sample IDs
pooled = pool_samples(dataset, rule="aavj", include_sample_ids=True)
```

Supported pooling rules: `ntvj`, `nt`, `aavj`, `aa`.
For each rule the representative clonotype is selected by frequency
(`duplicate_count` when `weighted=True`); `duplicate_count` is reassigned to
the total sum, and `incidence` / `occurrences` metadata are added.

---

## Diversity metrics

`mir.common.diversity` implements VDJtools-style summary indices
(Shugay *et al.* 2015, PMID:26606115) and iNEXT-style Hill diversity profiles
and rarefaction/extrapolation curves (Hsieh *et al.* 2016).

### Summary statistics

```python
from mir.common.diversity import summarize_counts

counts = [c.duplicate_count for c in repertoire.clonotypes]
div = summarize_counts(counts)

print(div.abundance)      # total read count
print(div.diversity)      # observed richness
print(div.chao1)          # bias-corrected Chao1 species richness estimator
print(div.shannon)        # Shannon entropy H′
print(div.gini_simpson)   # Gini-Simpson index (1 − Σp²)
print(div.singletons)     # clones seen exactly once
print(div.doubletons)     # clones seen exactly twice
```

### Hill diversity profile

```python
from mir.common.diversity import hill_curve

# Returns a Polars DataFrame with columns q, D_q
profile = hill_curve(counts)
# q=0 → species richness; q=1 → exp(Shannon); q=2 → inverse Simpson
```

### Rarefaction / extrapolation curve

```python
from mir.common.diversity import rarefaction_curve

curve = rarefaction_curve(counts)
# Polars DataFrame with m, s_obs, s_est, s_lwr, s_upr, sample coverage C
```

See `notebooks/diversity_analysis.ipynb` for a full donor-level workflow
including rarefaction curves, Hill profiles, and Healthy vs MS cohort
comparisons.

---

## Metaclonotypes

A *metaclonotype* is a lightweight cluster layer over an existing
`LocusRepertoire`. It stores cluster membership as a Polars DataFrame
(mapping `cluster_id` → `clonotype_id`) without rebuilding repertoire objects.
This supports any clustering backend: DBSCAN, ALICE/TCRNET enriched clusters,
TCRdist radius clusters, or pre-computed connected components.

### Unified clustering interface

`MetaclonotypeClusterConfig` + `cluster_metaclonotypes` dispatch to any
supported backend (ALICE, TCRNET, TCRdist, edit-distance graph, TCREmp, GLIPH):

```python
from mir.biomarkers.metaclonotype_cluster import (
    MetaclonotypeClusterConfig,
    cluster_metaclonotypes,
    cluster_paired_metaclonotypes,
)

# Edit-distance graph, Leiden communities
cfg = MetaclonotypeClusterConfig(method="edit_distance", graph_algo="leiden")
meta = cluster_metaclonotypes(rep, cfg)

# TCRdist radius clusters
cfg_dist = MetaclonotypeClusterConfig(method="tcrdist", locus="TRB", max_distance=24.5)
meta_dist = cluster_metaclonotypes(rep, cfg_dist)
```

Paired-chain metaclonotypes via single-chain-combine (works for **all** methods):

```python
# Computes per-chain edit-distance clusters, combines IDs as "TRA_cluster.TRB_cluster"
cfg = MetaclonotypeClusterConfig(method="edit_distance", min_cluster_size=1)
meta_paired = cluster_paired_metaclonotypes(paired_locus_rep, cfg)
```

For TCREmp, `cluster_paired_metaclonotypes` uses the built-in `PairedTCREmp`
joint embedding by default. See `notebooks/metaclonotype_method_compare.ipynb`
for a comparison of methods including concordance analysis.

### Build metaclonotypes from cluster labels

```python
from mir.common.metaclonotype import metaclonotypes_from_labels

# labels is a list of ints; -1 denotes noise/singleton (excluded by default)
meta = metaclonotypes_from_labels(clonotype_ids, labels)

print(meta.n_clusters)        # number of clusters
print(meta.cluster_ids[:5])   # sorted cluster IDs
```

### Build from pre-computed connected components

```python
from mir.common.metaclonotype import metaclonotypes_from_components

# components: list of lists of clonotype IDs
meta = metaclonotypes_from_components(components)
```

### Summarise cluster abundance

```python
from mir.common.metaclonotype import summarize_metaclonotypes

# Returns a Polars DataFrame with cluster_id and aggregated duplicate_count
summary = summarize_metaclonotypes(repertoire, meta)
```

### Functional diversity of the metaclonotype layer

```python
from mir.common.metaclonotype import functional_diversity

# One-call wrapper: summarize → DiversitySummary
div = functional_diversity(repertoire, meta)

print(div.shannon)       # Shannon entropy at the cluster level
print(div.chao1)         # Chao1 estimator for cluster richness
print(div.gini_simpson)  # Gini-Simpson index
```

### Cross-repertoire functional overlap

```python
from mir.common.metaclonotype import functional_overlap_1

# Fraction of metaclonotypes in rep_a that share a CDR3 identity with rep_b
overlap = functional_overlap_1(meta_a, meta_b, repertoire_a, repertoire_b)
```

---

## TCRdist

`TcrDist` (`mir.distances.tcrdist`) computes the weighted V-gene + CDR3
alignment distance between TCR clonotypes, following the TCRdist3 metric
(Dash *et al.* 2017).  All V-gene pairwise distances are pre-computed once
from full germline sequences; CDR3 alignment uses BLOSUM62 with a fixed-gap
C extension that releases the GIL for thread parallelism.

```python
from mir.distances.tcrdist import TcrDist
from mir.common.clonotype import Clonotype

# Build once — loads OLGA library and pre-computes V-gene distances (~3–10 s)
td = TcrDist.from_defaults(
    "TRB", "human",
    w_v=1.0, w_j=0.0, w_cdr3=3.0,
    fixed_gaps=(3, 4, -4, -3),   # C-accelerated (default)
    # fixed_gaps="Mid"  → midpoint gap per pair (Python, ~330× slower)
    # fixed_gaps=None   → full BioPython DP  (~780× slower)
)

cln1 = Clonotype(v_gene="TRBV19*01", j_gene="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF")
cln2 = Clonotype(v_gene="TRBV19*01", j_gene="TRBJ2-7*01", junction_aa="CASSIRASYEQYF")

d    = td.dist(cln1, cln2)                         # single pair
row  = td.dist_one_to_many(cln1, refs)             # (K,) array
mat  = td.dist_matrix(queries, refs, n_jobs=4)     # (N, K) matrix
```

### Radius and metaclonotype discovery

```python
from mir.basic.pgen import OlgaModel

model = OlgaModel(locus="TRB", species="human")
bg_seqs, _ = model.generate_sequences_counted(10_000, n_jobs=4, seed=42)
bg_clns = [Clonotype(junction_aa=s, locus="TRB") for s in bg_seqs]

# Median background distance for each query clonotype
radii = td.compute_radius(hits, bg_clns, percentile=50, n_jobs=4)

# Cluster around seeds whose radius falls in the bottom quartile
import numpy as np
threshold = float(np.percentile(radii, 25))
meta = td.find_metaclonotypes(rep, max_distance=threshold, n_jobs=4)
```

**Performance** (Apple M3, TRB, `fixed_gaps=(3,4,-4,-3)`, n_jobs=1):
28 M pairs/s at 1K–5K scale; ~76 M pairs/s with n_jobs=8.
See `notebooks/tcrdist_analysis.ipynb` for an influenza GILGFVFTL worked example.

---

## CDR3 Motif Logos

`mir.biomarkers.motif_logo` builds **IC** and **selection** sequence logos for
CDR3 motifs, following Pogorelyy *et al.* 2019 (PMID:31194732).  The key
idea is to subtract an OLGA-derived background for the *same* V-gene / J-gene /
CDR3-length bin, which collapses the germline signal and reveals only the
antigen-driven component.

```python
from mir.biomarkers.motif_logo import (
    compute_pwm, compute_logo, get_vj_background,
    build_terminal_anchored_pwm, load_motif_pwms, plot_logo,
)

motif_pwms = load_motif_pwms("motif_pwms.txt.gz")   # OLGA backgrounds

seqs = ["CASSGRSYEQYF", "CASSGRTNEQYF", ...]        # CDR3 sequences

bg  = get_vj_background(
    motif_pwms, v_gene="TRBV19*01", j_gene="TRBJ2-7*01",
    length=13, species="HomoSapiens", gene="TRB",
)
pwm  = compute_pwm(seqs)
logo = compute_logo(pwm, background=bg)   # adds ic_height + bg_height columns

fig, ax = plt.subplots()
plot_logo(logo, ax, height_col="bg_height")   # selection logo
```

For CDR3s of **mixed lengths** (different J-genes), use the
**terminal-anchored logo** which anchors V-side and J-side blocks independently:

```python
ta_pwm  = build_terminal_anchored_pwm(seqs, n_term=8, c_term=7)
ta_logo = compute_logo(ta_pwm, background=bg)
```

For automated per-VJ-len logos from ALICE/TCRNET hit DataFrames use
`build_motif_logos_vj`.  Background data (`motif_pwms.txt.gz`) is fetched
automatically by the notebook bootstrap helpers in `mir.utils.notebook_assets`.

See `notebooks/motif_logos.ipynb` for GILGFVFTL (Influenza A) and HLA-B27 AS
worked examples.

---

## ALICE and TCRNET

Both modules detect antigen-driven CDR3 clusters, but differ in how they model
the background:

| Feature | ALICE | TCRNET |
| --- | --- | --- |
| Background model | OLGA Pgen (analytical or MC pool) | Any MC control — real or synthetic |
| Pgen calls | OLGA 1mm Pgen (10M pool + fallback) | **None** |
| V/J restriction | `match_mode="vj"` (default) | `match_mode="none"` (default) |
| Statistics | Poisson | Binomial / Beta-Binomial |
| Selection correction | `q_factor` | `q_factor` (needed for synthetic controls) |

**ALICE** (`mir.biomarkers.alice`) implements the Pogorelyy *et al.* 2019
(PMID:31194732) Poisson enrichment test.  Default is `match_mode="vj"` with
OLGA gene-usage conditioning: `N` and `pgen` are scaled by `P_OLGA(V,J)` so
that `λ = N_total × pgen` regardless of gene restriction, while the observed
count `k` is V/J-filtered.  Uses a 10 M-sequence MC pool by default (the
paper uses 100 M, which requires ~17 GB; set `mc_n_pool=100_000_000` if memory
allows) with fallback to OLGA analytical 1mm Pgen for rare sequences.

V/J-restricted neighbour counting uses a **grouped-trie** strategy: one small
trie per (V, J) gene group.  This makes `match_mode="vj"` 1.5–2× *faster*
than `match_mode="none"` on natural repertoires (benchmark: 300 K sequences,
8 workers — unrestricted 9.9 s, V+J restricted 5.5 s).

**TCRNET** (`mir.biomarkers.tcrnet`) is a purely MC-control algorithm.
When used with a real control it captures V/J bias automatically.  Pass
`q_factor ≈ 3–5` when using a synthetic OLGA pool to correct for the
pre-thymic selection deficit.  TCRNET with a 100 M synthetic pool,
`match_mode="vj"`, and `q_factor=Q` is statistically equivalent to the
original ALICE paper.

---

## Prototype-based embeddings with TCREmp

`TCREmp` embeds immune receptor clonotypes as distance vectors to a fixed set
of prototype clonotypes, enabling rapid downstream analysis, dimensionality
reduction, and machine learning.

```python
from mir.embedding.tcremp import TCREmp
from mir.common.clonotype import Clonotype

model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000, junction_method="fixed_gap")

clonotypes = [
    Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF"),
    Clonotype(v_gene="TRBV20-1*01", j_gene="TRBJ1-1*01", junction_aa="CSARDSSYEQYF"),
]
X = model.embed(clonotypes)   # shape (2, 3000) — float32 array
```

Column layout: `[v_1, j_1, junc_1, v_2, j_2, junc_2, …, v_K, j_K, junc_K]`
where each distance uses `d(a, b) = s(a,a) + s(b,b) − 2·s(a,b)`.

For full DP alignment use `junction_method="biopython"` (~383× slower).
For custom prototypes use `TCREmp.from_file("prototypes.tsv", ...)`.

Paired-chain embedding concatenates TRA and TRB embeddings per `PairedClonotype`:

```python
from mir.embedding.tcremp import PairedTCREmp

paired_model = PairedTCREmp.from_defaults("human", "TRA_TRB", n_prototypes=500)
X_pair = paired_model.embed(paired_clonotypes)
```

`n_jobs` behaviour:

- `n_jobs=None` (default): auto-select based on `len(clonotypes) × n_prototypes`.
- `n_jobs=1`: force serial.
- `n_jobs>1`: force explicit worker count.

---

## Mask and match sequences

```python
from mir.basic.alphabets import (
    aa_to_reduced, mask, matches, matches_aa_reduced, NT_MASK, AA_MASK,
)

nt_masked = mask("ATCGAT", (2, 5), NT_MASK)
assert nt_masked == b"ATNNNT"

aa      = "CASTIV"
reduced = aa_to_reduced(aa)

# Matching ignores mask symbols: N (nucleotide) or X (amino acid)
assert matches(mask(aa, 0, AA_MASK), aa, AA_MASK)
assert matches_aa_reduced(aa, mask(reduced, 3, AA_MASK))
```

---

## Copilot Agent Workflow

This repository ships a dedicated Copilot custom agent and companion prompt:

- Agent: `.github/agents/mirpy-analysis.agent.md`
- Companion prompt: `.github/prompts/mirpy-analysis.prompt.md`

Use `/mirpy-analysis` from chat to supply input data paths, optional metadata
schema, and workflow definition.  The agent creates dedicated notebooks,
installs/validates dependencies, executes cells sequentially, and reports
outcomes.  For large datasets it benchmarks small chunks first and asks before
any run expected to exceed ~10–15 min on 4–8 cores or ~12–16 GB RAM.

---

## Resources

- Example notebooks: [notebooks/](https://github.com/antigenomics/mirpy/tree/main/notebooks)
- API reference: [https://antigenomics.github.io/mirpy/modules.html](https://antigenomics.github.io/mirpy/modules.html)
- Notebook gallery: [https://antigenomics.github.io/mirpy/examples.html](https://antigenomics.github.io/mirpy/examples.html)
- Docs source: [docs/](docs/)
- Agent skill guide (Claude, GitHub Copilot): [skills/mirpy/SKILL.md](skills/mirpy/SKILL.md)
- Benchmark baselines: [benchmarks.md](benchmarks.md)

---

## Project status

The library is actively evolving. Some modules are more mature than others,
and parts of the public API may still change.
