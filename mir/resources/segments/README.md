# mir/resources/segments

Gene segment library files and generation scripts for the `mir` package.

## Files

| File | Description |
|------|-------------|
| `olga_gene_library.txt` | V/D/J alleles from OLGA default models — generated, do not edit by hand |
| `imgt_gene_library.txt` | V/D/J alleles fetched from IMGT V-QUEST — generated, requires network |
| `build_gene_library.log` | Append-only build log with timestamps, commit hashes, allele counts, and consistency reports |
| `olga/` | Local OLGA model copies (full set, includes human TRG/TRD not in the package) |
| `build_gene_library.py` | Script that generates the two libraries and the consistency log |

## Column schema

Both generated files are tab-separated with a header row:

```
species   locus   gene   allele   sequence
```

| Column | Values | Example |
| ------ | ------ | ------- |
| `species` | `human`, `mouse` | `human` |
| `locus` | `TRB`, `TRA`, `TRG`, `TRD`, `IGH`, `IGK`, `IGL` | `TRB` |
| `gene` | `V`, `D`, `J` | `V` |
| `allele` | Full IMGT allele name | `TRBV3-1*02` |
| `sequence` | Nucleotide sequence (uppercase, no gaps) | `GACACAG…` |

Using OLGA as the primary source ensures that allele names and sequences are
**consistent between Pgen probability generation and the segment library** used
in `mir`. The IMGT library provides a broader, network-fresh reference.

## Script functions

| Function | Description |
| -------- | ----------- |
| `build_olga_library()` | Parse OLGA `model_params.txt` files; returns rows |
| `build_imgt_library()` | Fetch alleles from IMGT V-QUEST; returns rows |
| `write_library(rows, path)` | Write rows to a TSV file with header |
| `compute_stats(rows)` | Return per-(species, locus, gene) allele counts |
| `check_library_consistency(olga, imgt)` | Compare the two libraries; return formatted report |
| `append_log(path, section)` | Append a dated section to the build log |

## Generating the libraries

```bash
# Both libraries + consistency check (IMGT requires network)
python mir/resources/segments/build_gene_library.py

# OLGA only (no network needed)
python mir/resources/segments/build_gene_library.py --olga

# IMGT only
python mir/resources/segments/build_gene_library.py --imgt

# Consistency check only (reads existing files from disk)
python mir/resources/segments/build_gene_library.py --check
```

The OLGA builder searches two paths in priority order:

1. `mir/resources/segments/olga/default_models/` — local model copies
2. The installed `olga` package's `default_models/` directory

Note: mouse B-cell (IGH/IGK/IGL) OLGA models use synthetic non-IMGT allele names
and are excluded from the OLGA library; use the IMGT library for those.

The IMGT builder fetches from the IMGT V-QUEST reference directory.
Species `human` → `Homo_sapiens`, `mouse` → `Mus_musculus`.
Loci with D segments (TRB, TRD, IGH) are fetched for V, D, and J;
all others for V and J only.

## Running the tests

```bash
pytest tests/test_gene_library.py -v
```
