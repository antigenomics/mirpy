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
from mir.common.segments import SegmentLibrary

lib = SegmentLibrary.load_default(
    genes={"TRA", "TRB"},
    organisms={"HomoSapiens"},
)
```

If a requested organism/locus pair is missing in the default local segment file,
`mirpy` will download the missing V and J segments from IMGT and append them to
the default resource file automatically.

### Parse a table with clonotypes

```python
from mir.common.parser import VDJtoolsParser

parser = VDJtoolsParser(lib=lib, sep="\t")
clonotypes = parser.parse("example.tsv")
```

### Work with repertoires

```python
from mir.common.repertoire import Repertoire

repertoire = Repertoire(clonotypes=clonotypes, gene="TRB")
print(repertoire.total)
print(repertoire.number_of_clones)
```

You can also load a repertoire directly from a file:

```python
from mir.common.repertoire import Repertoire
from mir.common.parser import VDJtoolsParser

repertoire = Repertoire.load(
    parser=VDJtoolsParser(lib=lib, sep="\t"),
    path="example.tsv",
    gene="TRB",
)
```

## Resources

- Example notebooks are available in [notebooks/](https://github.com/antigenomics/mirpy/tree/main/notebooks).
- Source files for the API documentation are stored in [docs/](docs/).
- After GitHub Pages is enabled for the repository, the documentation site will be rebuilt automatically on each push to `main`.

## Project status

The library is actively evolving. Some modules are more mature than others, and
the public API may still change in places.
