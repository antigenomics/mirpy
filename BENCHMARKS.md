# Benchmark results — regression baseline

Recorded results for the sample-level (repertoire) embedding benchmarks (`experiments/benchmark_repertoire_*.py`),
so a re-run can be checked for regression. All numbers are **repeated 50-fold CV mean±std** unless noted, on
arda-native coordinates. Theory/interpretation: `THEORY.md` T7. Data provenance: `SOURCES.md`.

- **Env**: conda `mirpy` (Python 3.12), Apple M3 (32 GB). Seeds fixed (`seed=0`). `RUN_BENCHMARK` not required
  for these (they are experiment scripts, not the guarded pytest tier).
- **Recorded**: 2026-07-14. Re-run and compare; a drop beyond ~1 std on the headline metric is a regression.
- **Reproducibility**: all load **local data first, HF fallback** — `aging`/`airr_hip` via `_hf.fetch`
  (`hf_hub_download`), `airr_covid19` via `_covid.covid_path` (local `~/hf/airr_covid19` else `isalgo/airr_covid19`).

## Sample-level embedding — established regime (airr_benchmark aging, airr_hip)

| Benchmark | Command | Headline metric | Value |
|---|---|---|---|
| Depth-robustness (`prop:kme`) | `benchmark_repertoire_depth.py` | log–log slope `‖ΔΦ₁‖` vs `n_eff` (theory −0.5) | **−0.55** |
| Age regression (n=79) | `benchmark_repertoire_aging.py` | \|Spearman(age,pred)\|: diversity / kernel-mean / k-mer | **0.76** / 0.58 / 0.28 |
| CMV serostatus (n=240) | `benchmark_repertoire_cmvhla.py` | AUC: diversity / Φ blocks / learned | **0.83±0.05** / 0.59–0.63 / 0.49 |
| CMV age-matched control | `benchmark_repertoire_cmvhla.py` | AUC: diversity / age-only (decade-matched) | **0.83** / 0.45 |
| HLA-A\*02 (n=500, 25k) | `benchmark_repertoire_hla.py 250 25000` | AUC: second-moment / diversity | **0.623±0.048** / 0.460±0.048 |
| HLA panel (airr_hip, n=300) | `benchmark_repertoire_hla*panel` | second-moment AUC: B\*08 / B\*07 / B\*44 | **0.82** / 0.76 / 0.66 |
| Spike-in recovery | `benchmark_repertoire_spikein.py` | recall @N≤3k / FPR (clean P_gen null) | **~35–50%** / ~1.2% |

## Aging divergence — `benchmark_repertoire_agediverge.py 500000 30` (58 donors @500k)

Divergence rises with age at depth but is **diversity-coupled, not an independent axis**.

| Metric | Value | Note |
|---|---|---|
| ρ(¹D, divergence) biased KME | −0.153 | biased V-stat `1/n_eff` artifact |
| ρ(¹D, divergence) **unbiased** KME | **−0.052** | `mmd_matrix(unbiased=True)`; artifact removed |
| ρ(¹D, divergence) overlap −logF | −0.680 | clonal-expansion driven |
| ρ(age, divergence overlap F) | 0.704 | (0.24 at 250k — depth-dependent) |
| **partial ρ(age, divergence \| ¹D)** | **0.067** (p=0.64) | n.s. — no independent axis (same at 40k/250k) |

## COVID cohort (airr_covid19, 300 donors, ≤20k reads)

### Batch cookbook — `benchmark_repertoire_covidbatch.py 300 20000` → **PASS**

| signal | naive AUC | batch-residualized |
|---|---|---|
| batch identity (OvR) | 0.777 | **0.032** |
| COVID status (⟂̸ batch) | **0.660** | 0.409 |
| HLA-A\*02 (⟂ batch) | 0.596 | **0.609** |

within-mixed-batch COVID (honest) **0.539**; MMD offset:biology ratio **1.05**.

### HLA imprint class I+II — `benchmark_repertoire_covidhla.py 300 20000` (TRB)

second-moment > diversity in direction **15/17** alleles; class-II present **8/9**.

| allele | class | 2nd-moment AUC | Δ vs diversity |
|---|---|---|---|
| DRB1\*07:01 | II | **0.758** | +0.204 |
| B\*07:02 | I | 0.702 | +0.154 |
| DRB1\*15:01 | II | 0.702 | +0.110 |
| DQB1\*05:01 | II | 0.619 | +0.152 |
| A\*02:01 | I | 0.628 | +0.035 |

### Paired α+β — `benchmark_repertoire_covidpaired.py 300 20000`

**TRA carries the stronger HLA imprint**; paired concat sits between (noisier β dilutes).

| allele | β (TRB) | α (TRA) | α+β paired |
|---|---|---|---|
| A\*02 | 0.538 | **0.636** | 0.612 |
| B\*07 | 0.646 | **0.701** | 0.664 |
| DRB1\*15 | 0.749 | **0.810** | 0.796 |

### COVID biomarker — honest **negative** (`covidstatus.py` / `covidpaired.py`)

| test | β | α | paired |
|---|---|---|---|
| COVID status, naive | 0.673 | 0.685 | **0.702** |
| COVID status, batch-resid | 0.488 | 0.512 | **0.522** (chance) |
| witness rediscovery of paper's clones | 0.373 | **0.450** | — |

Convalescent (long-past) COVID leaves **no batch-robust bulk clonotype-identity signal** at RNA-seq depth; the
naive 0.66–0.72 was batch confound. Ground truth is 87% α (4393 α vs 567 β).

---

## 2026-07-14 (pm) — spectral block, COVID witness, TCGA survival

### Spectral interaction block — `benchmark_repertoire_spectral.py 300 20000`

The opt-in top-`r` **eigenvalue** compaction of the second-moment block (new `n_eigs=` on `fit_repertoire_space`)
is **lossy for the HLA imprint**: HLA-A\*02 carriage lives in *which* public clones co-occur (directional), and
a rotation-invariant eigenvalue spectrum discards it. Default (`n_eigs=None`) keeps the full upper triangle,
unchanged.

| representation | block dim | HLA-A\*02 AUC |
|---|---|---|
| diversity (baseline) | 4 | 0.560±0.066 |
| D₂=512 top-16 eigvals | 16 | 0.545±0.093 |
| D₂=512 top-128 eigvals | 128 | 0.536±0.092 |
| D₂=256 full upper-tri | 32896 | 0.562±0.078 |
| **D₂=512 full upper-tri** | 131328 | **0.593±0.070** |

### COVID witness — `benchmark_repertoire_covidwitness.py 300 20000`

Why per-clonotype Fisher-significant clones "vanish" in the bulk embedding: **cohort breadth**, not read depth.

| level | finding |
|---|---|
| Fisher genome-wide BH (150–300 donors) | **0** clones pass at any depth (20k or 120k reads) |
| Fisher full cohort (~1137 donors, user tmp scan) | **39 β / 4 α** clones pass BH (min q≈5e-4) |
| witness β: whole-cohort → **mixed-batch** | 0.510 → **0.752** (batch control is the lever) |
| witness β: HLA-strat median / best | 0.532 / 0.712 (best = max-selection; median ≈ whole-cohort) |
| witness α: whole / mixed / strat-best | 0.463 / 0.424 / 0.473 (no recovery — α witness fails) |

Batch control (mixed-batch contrast, where COVID+healthy share runs) recovers the paper's β clones; per-allele
HLA stratification adds more noise than signal at these donor counts. HLA+α+β is **not** the missing key — breadth is.

### COVID motif recovery at full breadth — `benchmark_repertoire_covidmotif.py 0 60000`

Using the tool that *works* at breadth (Emerson incidence Fisher) on the **full 1137-donor cohort** (761 COVID /
376 healthy, 60 k reads), both chains, to actually *find COVID motifs*:

| chain | genome-wide clones q<0.05 | GT-true recovered | + HLA-restricted |
|---|---|---|---|
| **α** | 7 | **4** | +0 |
| β | 0 | 0 | +0 |

The 4 recovered α clones are one coherent public family — **CAG·NYGGSQGNLIF** (single-residue variants), all
members of the paper's COVID-associated α **cluster 31** (27 GT clones, all `has_covid_association`). **HLA
restriction adds nothing** because these α clones are already *public* (present across HLA backgrounds) —
per-allele restriction only loses power. β recovers **nothing** even at full breadth/60 k depth: the β COVID
signal is rarer and HLA-restricted (the user's DRB1\*16-focused β clone needs full native depth + that specific
allele). **HLA+α+β decomposes cleanly:** the recoverable signal is the **public α compartment** (breadth-powered,
no HLA needed — a real motif family found), and β is the HLA-restricted, depth-limited part. This is the honest
"find motifs" route; the breadth-starved bulk MMD witness recovered none of it.

### TCGA — `benchmark_repertoire_tcga.py <chains> BRCA,LUAD,KIRC 50000`

Tumor-type separation is **depth-dependent** (deep IG light chains ≫ shallow TR); **survival adds nothing** over
clinical covariates. Base Cox C-index (age+sex+stage+log reads): BRCA 0.72, LUAD 0.66, KIRC 0.73.

| chain | tumor-type macro-OvR AUC | survival ΔC (BRCA / LUAD / KIRC) |
|---|---|---|
| TRB (median ~24 clonotypes) | 0.523 | +0.003 / −0.003 / +0.002 |
| IGH | 0.500 | +0.002 / +0.001 / −0.004 |
| **IGK (deepest chain)** | **0.666** | −0.001 / −0.004 / −0.001 |
| concat TRB+IGH | 0.538 | −0.002 / +0.001 / +0.001 |

IGK's 0.67 shows the method works when clonotype depth suffices (the deep IG compartment carries tissue signal);
the flat ΔC across every chain says the tumour-infiltrating repertoire adds **no prognostic value beyond clinical
covariates** at TCGA RNA-seq depth.

### TCGA survival — biology-grounded features — `benchmark_repertoire_tcga_survival.py`

The clonotype *embedding* adds nothing, but interpretable **biology axes** (isotype from `c_call`, infiltration /
hot-vs-cold, atypicality = gene-usage divergence from the tumour-type centroid, clonal expansion — modelled on an
internal AIRR-tissue EDA) do, cancer-specifically. Base Cox = age+sex+stage+log reads; **bold** = best gain per row.

C-index gain over clinical (5-fold CV):

| cancer | C base | isotype | infiltration | atypicality | clonality | all-AIRR |
|---|---|---|---|---|---|---|
| SKCM (melanoma) | 0.609 | +0.008 | +0.034 | −0.001 | +0.031 | **+0.036** |
| KIRP (renal pap.) | 0.717 | −0.008 | **+0.030** | −0.003 | +0.014 | +0.015 |
| LGG (glioma) | 0.764 | −0.006 | **+0.019** | +0.008 | +0.016 | +0.011 |
| KIRC (renal cc.) | 0.721 | +0.005 | **+0.009** | +0.002 | +0.002 | +0.006 |

(LUAD / STAD / OV / BLCA: AIRR gain ≈ 0 or negative.)

KM median-split log-rank p (the EDA's stratification test — surfaces threshold effects the linear C-index misses;
**bold** = p<0.05):

| cancer | infiltration | IgA fraction | atypicality |
|---|---|---|---|
| KIRC | **0.002** | 0.089 | 0.880 |
| SKCM | **0.000** | **0.026** | 0.716 |
| KIRP | **0.020** | 0.750 | 0.555 |
| OV | **0.044** | 0.581 | 0.691 |
| LGG | **0.022** | 0.076 | **0.001** |
| BLCA | 0.676 | **0.010** | 0.243 |

**Read:** infiltration (hot/cold) stratifies survival in **5/8** cancers (melanoma, both renal, ovarian, glioma)
and adds C-index in melanoma/KIRP; **IgA** (isotype / mucosal) stratifies **bladder** + melanoma; **atypicality**
stratifies **glioma** — each matching known immune biology. Prognosis lives in infiltration magnitude + isotype +
typicality, **not** clonotype identity, and is cancer-specific — the reason the identity embedding's ΔC was flat.

## Repertoire embedding for TME & survival — pan-cancer (33 TCGA types)

The reframing: the prognostic axes above are **channels of one TME-aware, multi-chain repertoire embedding**
Φ(S) (`_tcga_embedding.py`: per-chain identity ‖ diversity ‖ coverage/infiltration, + isotype + composition
+ atypicality). `benchmark_repertoire_tcga_pancancer.py ALL` fits Φ once over 9 425 OS-annotated samples
(78-dim, 5 embeddable chains + all 7 in the composition channel) and, per cancer, reports the CV **ΔC-index**
of clinical+Φ over clinical (age+sex+stage+log reads) and a **likelihood-ratio p** for the Φ block.

Cancers where Φ is robustly prognostic (**both** LR p<0.05 **and** CV ΔC>0; bold):

| cancer | n | events | C base | C+Φ | ΔC | LR p | top channel |
|---|---|---|---|---|---|---|---|
| **SKCM** (melanoma) | 460 | 219 | 0.609 | 0.647 | **+0.039** | **0.000** | coverage |
| **BLCA** (bladder) | 411 | 182 | 0.650 | 0.675 | **+0.025** | **0.002** | identity |
| **HNSC** (head & neck) | 500 | 218 | 0.617 | 0.639 | **+0.022** | **0.012** | coverage |
| **LGG** (glioma) | 423 | 112 | 0.764 | 0.779 | **+0.016** | **0.000** | coverage |

Strong effect-size positives (ΔC>0, LR n.s. — power-limited): SARC +0.038 (isotype), KIRP +0.034 (atypicality),
LUAD +0.009, KIRC +0.008, BRCA +0.007. Immune-cold / small cohorts are flat-to-negative (overfit): LIHC −0.029,
STAD −0.026, PAAD −0.029, UCEC −0.072 (n=181/35ev).

**Pan-cancer (20 evaluable):** mean ΔC ≈ 0 (median +0.003), Φ significant in 5/20 — most-informative channel
tally **coverage 5 · atypicality 5 · composition 4 · isotype 3 · identity 2 · diversity 1**. The signal is
cancer-specific and lives in the **TME channels (infiltration, atypicality, composition), not clonotype
identity**; the pan-cancer mean is ≈0 because immune-cold cohorts overfit and cancel the immune-hot wins —
so the honest claim is *Φ adds significant prognostic value in immunologically active cancers* (melanoma,
glioma, head&neck, bladder), not universally.

### TME states (unsupervised) — `benchmark_repertoire_tcga_tme.py ALL 6`

KMeans (k=6) on Φ's TME channels (9 425 samples) recovers coherent, interpretable microenvironment states
(z-scored channel means; **bold** = defining extreme). The UMAP (`experiments/figures/umap_tcga_tme`) shows a
clean **infiltration gradient** (hot → cold) organising the pan-cancer cohort.

| state | n | infiltration | T-vs-B | switch | diversity | death | HR (vs ref) | enriched cancers |
|---|---|---|---|---|---|---|---|---|
| cold-humoral | 1565 | −0.53 | −0.36 B | +0.39 | −0.40 | 0.33 | **1.14** (p=0.011) | BRCA, BLCA, PRAD |
| hot-diverse | 2090 | **+1.03** | −0.70 B | +0.35 | **+0.92** | 0.31 | 0.92 (p=0.072) | BRCA, LUAD, STAD |
| T-balanced | 1277 | −0.10 | **+0.86 T** | +0.15 | +0.06 | 0.29 | 1.09 | KIRC, PRAD, BRCA |
| cold-T unsw. | 883 | **−1.58** | **+1.72 T** | **−2.20** | −1.66 | 0.30 | 1.05 | LGG, LIHC, SARC |
| cold | 871 | −1.27 | +0.74 T | −0.70 | −1.13 | 0.28 | 1.08 | THCA, LIHC, LGG |
| warm-B (ref) | 2739 | +0.48 | −0.45 B | +0.37 | +0.40 | 0.31 | — | BRCA, LUAD, HNSC |

Stratified multivariate log-rank across states (blocking tumour type) **p=0.0038** — the states are prognostic
*beyond* cancer type. The **cold-humoral** low-infiltration state carries the **worst** outcome (HR 1.14, p=0.011);
the **hot-diverse** state trends protective (HR 0.92). Glioma (LGG) falls in the cold, strongly T-skewed states,
matching its known low-but-T-biased infiltrate. The repertoire embedding organises the TME unsupervised.

### In-silico evolution — perturb infiltration, decode the coupled response — `benchmark_repertoire_tcga_insilico.py ALL`

`mir.repertoire.sample_descriptor` makes every metric a **smooth derivable coordinate** (infiltration = log-mass,
diversity = log n_eff, clonality = Σw², identity = kernel mean), so the cohort's joint distribution is a
generative **manifold** and moving a sample along the infiltration axis (hot↔cold) while staying on it predicts
how the other metrics respond. Gaussian-manifold conditional slopes d(metric)/d(infiltration), pan-cancer mean:

| response as tumour gets hotter | slope | reading |
|---|---|---|
| diversity | **+0.84** | hot ⇒ more diverse (universal, all cancers) |
| class-switch | +0.52 | hot ⇒ more class-switched B cells |
| IgG | +0.49 | hot ⇒ more IgG (mature humoral) |
| T-vs-B | −0.63 | hot ⇒ more B-skewed (TLS-like) |
| atypicality | −0.23 | hot ⇒ less atypical / more convergent |

**In-silico "make this tumour hotter"** (move cold→hot on-manifold, read the CoxPH Δlog-HR): predicted
**protective in 12/20** cancers (SKCM −0.49, HNSC −0.63, LUAD −0.44, LAML −0.96 …) — TIL-is-good — but correctly
**adverse in glioma** (LGG **+1.19**, GBM +0.47) and renal (KIRC/KIRP +0.3), matching the known "immune-hot glioma
is worse" biology. The embedding *simulates* the hot↔cold survival axis; the couplings are learned, not imposed.
