# mirpy

[![PyPI](https://img.shields.io/pypi/v/mirpy-lib)](https://pypi.org/project/mirpy-lib/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs](https://img.shields.io/badge/docs-antigenomics.github.io-informational)](https://antigenomics.github.io/mirpy)

![mirpy logo](assets/mirpy_logo.png)

`mirpy` is a Python library for working with AIRR-seq and immune repertoire data.
It provides building blocks for:

- parsing clonotype and repertoire tables;
- loading and updating V/J segment libraries;
- repertoire-level statistics and diversity analysis;
- distance calculation and sequence matching;
- prototype-based embeddings and comparative analysis.

The package is designed as a reusable toolkit rather than a single pipeline.

For a full diversity workflow, see `notebooks/diversity_analysis.ipynb`, which
includes donor-level tables, rarefaction and coverage curves, Hill profiles,
and Healthy vs MS cohort comparisons.

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
pip install -e .
```

If you only need the package for project usage, prefer `pip install mirpy-lib`.
If you plan to develop or run docs/notebooks locally, use the cloned repo setup.

## Copilot Agent Workflow

This repository ships a dedicated Copilot custom agent and a companion prompt for notebook-first mirpy analysis:

- Agent: `.github/agents/mirpy-analysis.agent.md`
- Companion prompt: `.github/prompts/mirpy-analysis.prompt.md`

Use the companion prompt from chat (`/mirpy-analysis`) to provide:

- input data paths (AIRR/VDJtools/Adaptive or other parser-supported formats),
- optional metadata schema/path,
- workflow definition and/or hypotheses to test.

The agent creates dedicated notebook(s), installs/validates dependencies, executes cells sequentially, and reports outcomes.
For large datasets, it benchmarks small chunks first, estimates full runtime/memory, and asks before full runs expected to exceed ~10-15 min on 4-8 cores or ~12-16 GB RAM.

## Main modules

- `mir.common`: clonotypes, repertoires, parsers, segment libraries
- `mir.distances`: aligners, search, graph-based utilities
- `mir.basic`: diversity, sampling, segment usage, pgen helpers
- `mir.embedding`: repertoire and prototype embeddings
- `mir.comparative`: pairwise sample overlap metrics (Jaccard, D, F, Morisita-Horn), trie-accelerated approximate matching (Hamming / Levenshtein), VDJBet Pgen-matched null distributions
- `mir.biomarkers`: enrichment and biomarker detection utilities — ALICE, TCRNET, GLIPH
- `notebooks/diversity_analysis.ipynb`: diversity summary tables, rarefaction,
  Hill curves, and cohort comparisons for Healthy vs MS donors

### ALICE and TCRNET

Both modules detect antigen-driven CDR3 clusters, but differ in how they model the background:

| Feature | ALICE | TCRNET |
| --- | --- | --- |
| Background model | OLGA Pgen (analytical or MC pool) | Any MC control — real or synthetic |
| Pgen calls | OLGA 1mm Pgen (10M pool + fallback) | **None** |
| V/J restriction | `match_mode="vj"` (default) with gene-usage conditioning | `match_mode="none"` (default); optional `"vj"` or via real control |
| Statistics | Poisson | Binomial / Beta-Binomial |
| Selection correction | `q_factor` | `q_factor` (needed for synthetic controls) |

**ALICE** ([`mir.biomarkers.alice`](mir/biomarkers/alice.py)) implements the Pogorelyy et al. *PLoS Biol.* 2019
Poisson enrichment test.  Default is `match_mode="vj"` (V+J gene restriction) with OLGA gene-usage conditioning:
`N` and `pgen` are scaled by `P_OLGA(V,J)` so that `λ = N_total × pgen` regardless of gene restriction, while
the observed count `k` is V/J-filtered.  Uses a 10M-sequence MC pool by default (the paper uses 100M, which
requires ~17 GB and ~16 min; use `mc_n_pool=100_000_000` if memory allows) and falls back to OLGA analytical
1mm Pgen for rare sequences.  Use `pgen_mode="mc"` for all production runs.

**TCRNET** ([`mir.biomarkers.tcrnet`](mir/biomarkers/tcrnet.py)) is a purely MC-control algorithm — no Pgen
calls.  When used with a real control it captures V/J bias automatically.  Pass `q_factor ≈ 3–5` when using
a synthetic OLGA pool to correct for the pre-thymic selection deficit.  TCRNET with a 100M synthetic pool,
`match_mode="vj"`, and `q_factor=Q` is statistically equivalent to the original ALICE paper.

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

If a requested organism/locus pair is missing in the default local segment file,
`mirpy` will download the missing V and J segments from IMGT and append them to
the default resource file automatically.

### Parse a table with clonotypes

```python
from mir.common.parser import VDJtoolsParser

parser = VDJtoolsParser(sep="\t")
clonotypes = parser.parse("example.tsv")
```

### Work with repertoires

```python
from mir.common.repertoire import LocusRepertoire

repertoire = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
print(repertoire.duplicate_count)
print(repertoire.clonotype_count)

# Functional/canonical filtering using IMGT functionality annotations.
from mir.common.filter import filter_functional, filter_canonical
from mir.common.gene_library import GeneLibrary

imgt_lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="imgt")
functional_rep = filter_functional(repertoire, gene_library=imgt_lib)
canonical_rep = filter_canonical(repertoire, gene_library=imgt_lib)
```

You can also load a repertoire directly from a file:

```python
from mir.common.parser import VDJtoolsParser
from mir.common.repertoire import LocusRepertoire

repertoire = LocusRepertoire(
    clonotypes=VDJtoolsParser(sep="\t").parse("example.tsv"),
    locus="TRB",
)
```

### Pool repertoires across samples

`mirpy` provides pooled repertoires with configurable identity rules.

```python
from mir.common.pool import pool_samples

# Pool by nucleotide CDR3 + V/J calls.
pooled_ntvj = pool_samples(
    [sample_rep_1, sample_rep_2],
    rule="ntvj",
    weighted=True,
)

# Pool an entire dataset by amino-acid CDR3 + V/J and keep contributing sample ids.
pooled_aavj = pool_samples(
    dataset,
    rule="aavj",
    include_sample_ids=True,
)
```

Supported pooling rules:

- `ntvj`: key is `(junction, v_gene, j_gene)`
- `nt`: key is `(junction,)`
- `aavj`: key is `(junction_aa, v_gene, j_gene)`
- `aa`: key is `(junction_aa,)`

For each pooled key, the representative clonotype is selected by frequency
(`duplicate_count` when `weighted=True`, otherwise row occurrences), while
`duplicate_count` is reassigned to the total sum over grouped rows. The pooled
clonotype metadata contains `incidence` (unique samples) and `occurrences`
(independent rearrangement rows).

### Prototype-based embeddings with TCREMP

`TCREmp` (T-Cell Receptor EMbedding with Prototypes) embeds immune receptor clonotypes
as distance vectors to a fixed set of prototype clonotypes, enabling rapid downstream
analysis, dimensionality reduction, and machine learning.

```python
from mir.embedding.tcremp import TCREmp
from mir.common.clonotype import Clonotype

# Build a TCREMP model with default prototypes (fast, fixed-gap junction alignment)
model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000, junction_method="fixed_gap")

# Embed clonotypes as distance vectors to all prototypes
clonotypes = [
    Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF"),
    Clonotype(v_gene="TRBV20-1*01", j_gene="TRBJ1-1*01", junction_aa="CSARDSSYEQYF"),
]
X = model.embed(clonotypes)  # shape: (2, 3000) — float32 array

# Embeddings combine three distance types per prototype: V-germline, J-germline, and junction
# Column layout: [v_1, j_1, junc_1, v_2, j_2, junc_2, ..., v_K, j_K, junc_K]
print(X.shape)  # (2, 3000)
```

For full DP alignment semantics, use `junction_method="biopython"` (slower, ~90x):

```python
model_bio = TCREmp.from_defaults("human", "TRB", n_prototypes=100, junction_method="biopython")
```

For custom prototypes:

```python
model_custom = TCREmp.from_file("prototypes.tsv", species="human", locus="TRB")
```

Paired-chain embedding uses the same idea, but concatenates the TRA and TRB
embeddings for each `PairedClonotype`:

```python
from mir.common.parser import VDJdbFullPairedParser
from mir.common.single_cell import build_tenx_sample_from_cell_clonotypes
from mir.common.single_cell_repair import impute_missing_chains
from mir.embedding.tcremp import PairedTCREmp

parser = VDJdbFullPairedParser()

# Keep only complete TRA/TRB rows.
strict_df, strict_meta = parser.parse_cell_clonotypes_file(
    "vdjdb_full.txt.gz",
    species="HomoSapiens",
    include_incomplete=False,
)
strict_sample = build_tenx_sample_from_cell_clonotypes(
    strict_df,
    sample_id="vdjdb_full_human_strict",
    barcode_metadata=strict_meta,
)

# Or keep incomplete rows and impute the missing chain before pairing.
impute_df, impute_meta = parser.parse_cell_clonotypes_file(
    "vdjdb_full.txt.gz",
    species="HomoSapiens",
    include_incomplete=True,
)
imputed_df = impute_missing_chains(impute_df)
imputed_sample = build_tenx_sample_from_cell_clonotypes(
    imputed_df,
    sample_id="vdjdb_full_human_imputed",
    barcode_metadata=impute_meta,
)

paired_model = PairedTCREmp.from_defaults("human", "TRA_TRB", n_prototypes=500)
paired_clonotypes = imputed_sample.paired_locus_repertoires["TRA_TRB"].paired_clonotypes
X_pair = paired_model.embed(paired_clonotypes)
```

The VDJdb full parser uses one synthetic barcode per source row and stores the
row id plus antigen/MHC annotations in
`sample.single_cell_repertoire.barcode_metadata`. A paired analysis notebook is
available at `notebooks/tcremp_vdjdb_analysis_paired.ipynb`.

Key features:

- **Distance formula**: `d(a, b) = s(a,a) + s(b,b) − 2·s(a,b)` ensures metric properties.
- **Smart `n_jobs` auto-switch**: with `n_jobs=None` (default), TCREmp chooses `1` or
    `os.cpu_count()` based on workload `len(clonotypes) * n_prototypes`.
- **Pre-computed germline distances**: V/J distances are cached for O(1) lookup.
- **Biologically interpretable**: Each embedding dimension corresponds to distance to a specific prototype.

`n_jobs` behavior:

- `n_jobs=None` (default): auto policy based on workload threshold.
- `n_jobs=1`: force serial execution.
- `n_jobs>1`: force explicit worker count.

In auto mode, BioPython junction alignment stays serial because thread overhead
usually dominates for that backend.

Why threshold is not based on clonotypes alone:

- Parallel splitting is done on the input clonotype list (query axis).
- However, each query is scored against all prototypes in the C backend.
- So total work is still proportional to `n_clonotypes * n_prototypes`, and both
    terms matter for the auto-switch decision.

Default method guidance:

- `fixed_gap` remains the default because it is substantially faster and is the
    intended production embedding backend.
- `biopython` can be used when full dynamic-programming semantics are required,
    but no repository benchmark currently shows a consistent downstream quality
    gain that justifies making it the default.
- End-to-end embedding pipelines may show smaller speedups (for example 3-4x)
    than raw pairwise core-kernel comparisons due to non-alignment overhead.

### Repertoire internals and lazy tabular backend

`LocusRepertoire` supports three internal representations:

- eager clonotype list (`clonotypes`),
- lazy column bundles (`_pending_cols`) for fast parser paths,
- Polars table (`_polars_table`) for tabular I/O and grouped operations.

When a repertoire is built from a Python clonotype list, no Polars table is
created up front. The table is generated lazily on first `to_polars()` call and
cached for reuse. Count properties (`clonotype_count`, `duplicate_count`) stay
fast and avoid full materialization when lazy columns or Polars data are
available.

### Mask and match sequences

```python
from mir.basic.alphabets import (
    aa_to_reduced,
    mask,
    matches,
    matches_aa_reduced,
    NT_MASK,
    AA_MASK,
)

nt_masked = mask("ATCGAT", (2, 5), NT_MASK)
assert nt_masked == b"ATNNNT"

aa = "CASTIV"
reduced = aa_to_reduced(aa)

# Matching ignores mask symbols: N for nucleotides, X for amino-acid alphabets.
assert matches(mask(aa, 0, AA_MASK), aa, AA_MASK)
assert matches_aa_reduced(aa, mask(reduced, 3, AA_MASK))
```

## Resources

- Example notebooks are available in [notebooks/](https://github.com/antigenomics/mirpy/tree/main/notebooks).
- API and module/function documentation: [https://antigenomics.github.io/mirpy/modules.html](https://antigenomics.github.io/mirpy/modules.html)
- Notebook gallery in docs: [https://antigenomics.github.io/mirpy/examples.html](https://antigenomics.github.io/mirpy/examples.html)
- Docs source tree: [docs/](docs/)
- Agent skill guide for LLM-assisted workflows (Claude, GitHub Copilot, similar agents): [skills/mirpy/SKILL.md](skills/mirpy/SKILL.md)

## Project status

The library is actively evolving. Some modules are more mature than others, and
the public API may still change in places.
