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
| OLGA TRB sample | `tests/assets/olga_humanTRB_1000.txt.gz` | **Computed**, not experimental: 1 000 human TRB rearrangements drawn from the OLGA generative model (Sethna et al., *Bioinformatics* 2019), seed 42. Cols (no header): `junction_nt, junction_aa, v_call, j_call`; kept in **generation order — do not sort** (see note below) | `conda run -n mirpy --no-capture-output olga-generate_sequences --humanTRB -n 1000 -o /tmp/olga.tsv --seed 42` then `awk -F'\t' -v OFS='\t' '{print $1,$2,$3"*01",$4"*01"}' /tmp/olga.tsv \| gzip -n > tests/assets/olga_humanTRB_1000.txt.gz` (the `*01` suffix is appended because OLGA emits gene-level calls but `mir` expects allele-level, matching the bundled prototypes) |

> **Provenance correction (2026-07-17).** Until this date `olga_humanTRB_1000.txt.gz` was **not**
> OLGA output despite its name and this row's claim. It was a `head`-slice of the *alphabetically
> sorted* VDJdb TRB dump: all 969 unique junctions matched `vdjdb.slim.txt.gz` TRB `cdr3` exactly
> (100%, vs **0.96%** for genuine OLGA), the `junction_nt` column was a placeholder run of `N`, and
> the first row (`ATSIRFTDTQYF`) lacked the Cys anchor OLGA always emits. Consequences of the old
> file, for anyone re-reading results predating this fix:
> 1. **Not an antigen-naive null.** 12% of its rows had a Hamming-1 neighbour within the file vs
>    **0.2%** for real OLGA (166 vs 4 pairs per 1 000) — VDJdb is *selected for* the antigen-driven
>    convergence such a background is supposed to lack. Any chance-rate / null calibration against it
>    compared VDJdb to itself and is meaningless.
> 2. **Sorted ⇒ slices are not exchangeable.** Because it was alphabetically sorted, the
>    `_load_olga(n)` / `_load_olga(n, offset=400)` obs-vs-background slice pattern in
>    `tests/test_density.py` drew two systematically different regions of sequence space (the
>    background block was 58% `CAIS`-prefixed and shared only 20 of 54 V genes with obs). Genuine
>    OLGA output is in generation order, which is why the regenerate command above must **not** sort.
>
> The `tests/` suite was **not** invalidated: all 58 tests pass identically on both files (they use
> the asset only as a generic pool of TRB junctions, never as a null), and the Hamming-1 gate in
> `test_continuous_matches_discrete_hamming1` scores rho 0.727 on the old file vs 0.809 on real OLGA.
> The damage was confined to external calibrations that treated the file as a synthetic negative
> control.

Bundled model / Pgen data used at runtime lives in **vdjtools** (`vdjtools.model` bundled
parquet marginals for 7 loci × {olga, learned}) — see the vdjtools `SOURCES.md`.

## Sample-level (repertoire) embedding cohorts (HF `isalgo/*`, not shipped)

Phenotype-labelled repertoire cohorts for the §T.7 sample-level embedding benchmarks
(`experiments/benchmark_repertoire_*.py`; see `REPERTOIRE_EMBEDDING.md`). All **experimental**.

| Dataset | Origin | Metadata / labels | Fetch |
|---|---|---|---|
| `aging` (full depth) | HF `isalgo/airr_benchmark`, Britanova et al. *J Immunol* 2016/2014 | manifest `vdjtools/metadata_aging.txt`: `#file_name, sample_id, sex, age, label`; **79 samples**, ages ~0–103 (cord blood → centenarian); TCRβ. Sample files at `vdjtools/<file_name>.gz` (e.g. `vdjtools/A3-i101.txt.gz`); batch prefix `A2/A3/…` in `sample_id` | `hf_hub_download(repo_id="isalgo/airr_benchmark", filename="vdjtools/metadata_aging.txt", repo_type="dataset")` then per-sample `vdjtools/<name>.gz` — ⚠ use the **full** `vdjtools/`, NOT `vdjtools_lite/` (the *downsampled* set, kept only as a depth-robustness control). Benchmarks downsample per-sample to the RNA-seq regime |
| `airr_hip` | HF `isalgo/airr_hip` = **Emerson et al. 2017** (*Nat Genet*, PMID 28369038, doi:10.1038/ng.3822) | manifest `metadata.txt`: `file_name, sample_id, age, race, sex, cmv, hla`; **786 subjects**; TCRβ + **CMV serostatus** (`+`/`-`/`NA` = 340/421/25) + **HLA-A/B typing** (comma-sep `HLA-A*02,…`; A*02 most common, n=294). Sample files are the `file_name` path (`corr/HIP*.txt.gz`), full-depth (~10⁵ clonotypes) | `hf_hub_download(repo_id="isalgo/airr_hip", filename="metadata.txt", repo_type="dataset")` then per-sample `file_name`. Downsample per-sample; **age-match CMV±** (CMV rises with age, a diversity confound) |
| SRA shallow RNA-seq | HF `isalgo/airr_benchmark/sra/` (`meta.tsv`, `samples.tar.gz`) | `PMID Run BioProject Sample`; 2993 rows; all PMID 30830871, BioProject PRJNA511467 | low-coverage bulk-RNA-seq stress set (the `10²–10⁴` clonotype/chain regime); accession manifest only |
| `airr_covid19` | HF `isalgo/airr_covid19` = **Vlasova, Nekrasova, Komkov et al. 2026** (*Genome Med* 18, 20; DNA-multiplex FMBA cohort — cite per the dataset card, ⚠ verify before appendix use) | `metadata.tsv`: `file_name, reads, batch_id, sample_id, COVID_status` (COVID/healthy/precovid/unknown), `COVID_IgG/IgM/PCR`, full **4-digit HLA class I+II both alleles** (`HLA-{A,B,C}_{1,2}`, `HLA-{DPB1,DQB1,DRB1}_{1,2}`, `HLA-DRB{3,4,5}_1`), `donor_id`; **1258 donors**, **paired TRA+TRB**. **9 real sequencing batches** (`2020/09…2021/01_FMBA_NovaSeq*`, 103–185 donors each), partly status-confounded (NovaSeq4≈all-healthy, NovaSeq9=all-precovid; NovaSeq5/6/7 mixed). Ships `covid_associated_clonotypes.csv` (114 CDR3 clusters, `has_covid_association` T/F — COVID ground-truth motifs). **No `age`** (age+sex live in the sibling `isalgo/airr_covid19_vacc`). Files: **local git-LFS checkout** at `~/hf/airr_covid19/<file_name>` (vdjtools cols `count/freq/cdr3aa/v/j`), *not* the HF hub cache — load by local path, not `hf_hub_download` | local `~/hf/airr_covid19/`; `git clone git@hf.co:datasets/isalgo/airr_covid19` (LFS). Downsample per-sample. **Batch is a strong nuisance ⇒ prop:batch test bed**; HLA ⟂ batch (donor genetics) vs COVID-status ⟂̸ batch (confounded) |
| `airr_tcga` | HF `isalgo/airr_tcga` = **TCGA** tumour bulk RNA-seq → AIRR (RNA-seq AIRR extraction method Bolotin et al. *Nat Biotechnol* 2017, PMID 29020005) | `metadata.tsv`: `sample_id` (`TCGA-XX-XXXX.N`), `subject_id`, `cancer_type`/`disease`/`study_id` (**33 cancer types**), `sex/race/age`, **stage** (`cancer_stage` S1–S4 + `tumor_stage` i–iv; no grade column), `therapy/response`, `OS`+`OS_event` (days + 0/1 death, ~9510 usable; **`PFS`/`PFS_event` are present-but-empty — unusable**), **`total_reads`** (raw FASTQ total, GDC realigned-BAM `Total_Reads`, verified ≠ aligned; 99.6% present) + `aligned_reads`; **9591 samples**, **TCR+BCR but IG-dominant** (~97% IG: IGK/IGL/IGH; ~3% TR), median **634** clonotypes/sample (RNA-seq regime). Caveats: 487 lack clinical, 42 lack `total_reads` | **local git-LFS checkout** at `~/hf/airr_tcga/` (`git clone git@hf.co:datasets/isalgo/airr_tcga`) or `hf_hub_download(repo_id="isalgo/airr_tcga", filename="samples.tar.gz"/"metadata.tsv", repo_type="dataset")` + `load.py`. Per-sample AIRR in `samples.tar.gz` (`samples/<sample_id>.tsv`, AIRR cols). Separate **`metadata.hla.tsv`** = donor-level **HLA class-I** keyed by `subject_id` (`HLA-{A,B,C}_{1,2}`, 4-digit, + `hla_source`), union of TCGA **PanImmune** (Thorsson et al. *Immunity* 2018, PMID 29628290; primary, 7649) + **OptiType** fill (Szolek et al. *Bioinformatics* 2014, PMID 25143287; 1474); **9123/9450 donors = 96.5%**, → 9263 samples. **Tissue depth↔infiltration test bed** (`sec:samp-norm`, `prop:infiltration`): `total_reads` is the technical denominator for the receptor read-fraction infiltration proxy; HLA enables the `prop:hla` stratification |

## Theory appendix (`appendix/`)

| Dataset / asset | Path | What / provenance | Regenerate |
|---|---|---|---|
| Theory figure data | `appendix/data/*.tsv`, `theory_stats.txt` | S1–S3 dissimilarity/distance samples, distribution fits, correlations; **computed** (not experimental) by reusing `mir.bench.theory` on the bundled `human_TRB` prototypes | `python appendix/gen_theory_data.py` (conda `mirpy` env, `[bench]` extra) |
| gnuplot-palettes | *not vendored* (clone to `../gnuplot-palettes` or set `$GNUPLOT_PALETTES`) | ColorBrewer palettes from github.com/Gnuplotting/gnuplot-palettes; used by the `appendix/*.gp` scripts. **External tool, not a submodule** | `git clone https://github.com/Gnuplotting/gnuplot-palettes.git ../gnuplot-palettes` |
