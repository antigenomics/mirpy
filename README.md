# mirpy

![mirpy logo](assets/mirpy_logo.png)

`mirpy` is a Python library for working with AIRR-seq and immune repertoire data.
It provides building blocks for:

- parsing clonotype and repertoire tables;
- loading and updating V/J segment libraries;
- repertoire-level statistics and diversity analysis;
- distance calculation and sequence matching;
- prototype-based embeddings and comparative analysis.

The package is designed as a reusable toolkit rather than a single pipeline.

## Installation

Requirements:

- Python 3.11+
- a C/C++ build toolchain for compiled extensions

Install from PyPI:

```bash
pip install mirpy-lib
```

Install from the repository root:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Main modules

- `mir.common`: clonotypes, repertoires, parsers, segment libraries
- `mir.distances`: aligners, search, graph-based utilities
- `mir.basic`: diversity, sampling, segment usage, pgen helpers
- `mir.embedding`: repertoire and prototype embeddings
- `mir.comparative`: overlap, matching, TCRnet-style comparisons
- `mir.biomarkers`: enrichment and biomarker detection utilities

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
- Source files for the API documentation are stored in [docs/](docs/).
- After GitHub Pages is enabled for the repository, the documentation site will be rebuilt automatically on each push to `main`.

## Project status

The library is actively evolving. Some modules are more mature than others, and
the public API may still change in places.
