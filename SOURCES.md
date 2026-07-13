# SOURCES — data provenance

Every dataset used by mirpy v3, its origin, and how to regenerate it.

## Bundled resources (shipped in the wheel)

| Dataset | Path | What / provenance | Regenerate |
|---|---|---|---|
| Prototype clonotypes | `mir/resources/prototypes/{species}_{locus}.tsv` | 10 000 `(v_call, j_call, junction_aa)` per locus; `human_TRB` sampled from a **real** repertoire control (Britanova et al., *J Immunol* 2016), the rest **synthetic** OLGA-model draws (see `manifest.json` `source`), dedup on the triple, seed 42 | `python mir/resources/prototypes/generate_prototypes.py` (should use `vdjtools.model.generate`; **computed**, not experimental) |
| Germline gene libraries | `mir/resources/gene_library/{olga,imgt}_gene_library.txt` | Germline V/D/J segments (nt + functionality), per species/locus | `build_gene_library.py` |
| Region annotations | `mir/resources/gene_library/region_annotations.txt` | Per-allele FR/CDR amino-acid subsequences (`fwr1..fwr3, cdr1, cdr2, jcdr3, fwr4`), computed by **arda** from the gene library (build-time only) | `build_region_annotations.py` (needs `[build]`: arda + mmseqs2) |
| Baked germline distances | `mir/resources/germline_dist/{species}_{locus}.npz` | Pairwise V/J/CDR1/CDR2 distance matrices `d=s_ii+s_jj−2s_ij` (global BLOSUM62), **derived** from `region_annotations.txt`; runtime is pure numpy | `python mir/resources/germline_dist/build_germline_dist.py` (needs `[build]`: BioPython) |

## Benchmark / experiment data (not shipped; gitignored local fixtures)

| Dataset | Path | Origin | Notes |
|---|---|---|---|
| VDJdb slim dump | `tests/assets/vdjdb.slim.txt.gz` | VDJdb (Goncharov et al., *Nat Methods* 2022) export, dotted-column slim format | Antigen labels for the Table S1 benchmark; a specific release — F1 differs from the paper's 2023 summer release |
| Epitope-specific TRB sets | `tests/assets/{gilgfvftl,llwngpmav}_*.gz` | VDJdb subsets | Single-antigen fixtures |
| OLGA TRB sample | `tests/assets/olga_humanTRB_1000.txt.gz` | OLGA-generated human TRB | Synthetic repertoire sample |

Bundled model / Pgen data used at runtime lives in **vdjtools** (`vdjtools.model` bundled
parquet marginals for 7 loci × {olga, learned}) — see the vdjtools `SOURCES.md`.
