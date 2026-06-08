# Germline region annotations (arda integration)

How mirpy attaches germline FR/CDR amino-acid subsequences to gene-library
alleles, and how that powers region-resolved similarity and the TCREmp
`cdr123` embedding mode.

## Data model

`GeneEntry.region_aa: dict[str, str]` holds the germline-encoded subsequences:

- V genes: `fwr1`, `cdr1`, `fwr2`, `cdr2`, `fwr3`
- J genes: `jcdr3` (the J contribution to CDR3 — residues 5' of FR4), `fwr4`

They are loaded from the companion resource
`mir/resources/gene_library/region_annotations.txt` (schema:
`species locus gene allele fwr1_aa cdr1_aa fwr2_aa cdr2_aa fwr3_aa jcdr3_aa fwr4_aa`)
by `GeneLibrary.load_default(..., with_regions=True)` (default `True`). Missing
file or `with_regions=False` → every `region_aa` is `{}`.

```python
lib = GeneLibrary.load_default(loci={"TRB"}, species={"human"})
lib.entries["TRBV9*01"].region_aa["cdr1"]            # 'SGDLS'
lib.get_region_sequences_aa("TRB", "V", "cdr1")      # [(allele, aa), ...]
```

## Consumers

- **Region similarity** — `notebooks/gene_similarity.ipynb` builds
  `GermlineAligner.from_seqs(region_seqs)` per region (full V / CDR1+CDR2 /
  FR1-3; J CDR3-part / FR4) for human and mouse.
- **TCREmp `cdr123`** — `GermlineAligner.from_library_region(lib, loci, region)`
  builds per-V-gene CDR1 and CDR2 distance matrices; `TCREmp(mode="cdr123")`
  emits `[CDR1, CDR2, CDR3]` triples. Prototype V genes lacking annotation fall
  back to the max region distance (never NaN — see `_build_gene_matrix`).

## Generating annotations (build-time only)

`mir.common.region_annotation.annotate_gene_library(lib, organism)` runs
[arda](https://github.com/antigenomics/arda) over each germline V and J
nucleotide sequence and extracts the regions. Key facts:

- arda has **no coverage filter**, so a bare germline V (or J) maps to its
  scaffold and only the covered regions are returned — V-side for V, J-side for J.
  No synthetic V–J rearrangements are needed.
- Call shape: `annotate_records(records, organism, seqtype="nt", strand="forward", map_d=False)`.
- `jcdr3` is computed from the J nt before `fwr4_start` in the J reading frame.
- TRA/TRD are treated as equivalent loci (the dual `TRAV.../DV` genes).
- Requires the optional `arda` extra and the `mmseqs2` binary; arda is imported
  lazily, so importing this module without arda is fine.

Rebuild the shipped TSV (human + mouse, all 7 loci, OLGA ∪ IMGT alleles):

```bash
pip install -e ".[arda]"
conda install -c bioconda mmseqs2
python mir/resources/gene_library/build_region_annotations.py
```

~2,600 rows in ~10 s. A handful of non-mapping alleles are left unannotated
(they fall back gracefully at embedding time).
