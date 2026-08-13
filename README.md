<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mirpy_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/mirpy_light.svg">
    <!-- Absolute PNG fallback: PyPI strips <picture>/<source> and cannot render a relative or
         raw-served SVG, so the logo must be an absolute-URL raster here. GitHub uses the SVG sources. -->
    <img alt="mirpy" src="https://raw.githubusercontent.com/antigenomics/mirpy/master/assets/mirpy_light.png" width="360">
  </picture>
</p>

<h1 align="center">mirpy — ML embeddings for immune repertoires</h1>

[![PyPI](https://img.shields.io/pypi/v/mirpy-lib.svg)](https://pypi.org/project/mirpy-lib/)
[![Python](https://img.shields.io/pypi/pyversions/mirpy-lib.svg)](https://pypi.org/project/mirpy-lib/)
[![License](https://img.shields.io/badge/license-GPLv3-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-docs.isalgo.dev-blue)](https://docs.isalgo.dev/mirpy/)

**mirpy v3** turns T-/B-cell receptor sequences into fixed-length numeric vectors you can
cluster, visualize, and feed to ML models. It implements **TCREMP** — embedding each receptor
by its alignment distances to a fixed set of *prototype* sequences — so that Euclidean distance
in embedding space approximates pairwise alignment distance (Theory T1).

> v3 is a slim, embedding-focused rewrite. The classical repertoire toolkit (parsing, overlap,
> diversity, TCRnet, GLIPH, …) lives on the **`legacy-v2`** branch (`mirpy-lib` 2.x) and in the
> sibling tools [`vdjtools`](https://github.com/antigenomics/vdjtools) /
> [`vdjmatch`](https://github.com/antigenomics/vdjmatch).

## Install

```bash
pip install mirpy-lib            # core: numpy, polars, scipy, scikit-learn, seqtree, vdjtools
pip install "mirpy-lib[bench]"   # + benchmark / theory experiments
```

Pure-Python wheel; the heavy lifting (alignment, Pgen, sampling) is reused from
[`seqtree`](https://github.com/antigenomics/seqtree) and `vdjtools`.

## Quick start

```python
import polars as pl
from mir.embedding.tcremp import TCREmp

model = TCREmp.from_defaults("human", "TRB", n_prototypes=3000)   # mode="vjcdr3" | "cdr123"
df = pl.DataFrame({
    "v_call":      ["TRBV10-3*01", "TRBV20-1*01"],
    "j_call":      ["TRBJ2-7*01",  "TRBJ1-2*01"],
    "junction_aa": ["CASSIRSSYEQYF", "CSARVSGYYGYTF"],
})
X = model.embed(df)          # (2, 9000) float32 — 3 distances × 3000 prototypes
```

Downstream (cluster antigen-specific TCRs — needs `[bench]` for the kneedle `eps`, and a real
set of clonotypes rather than the two above):

```python
from mir.embedding.pca import pca_denoise
from mir.bench.metrics import cluster, cluster_metrics
labels = cluster(pca_denoise(model.embed(vdjdb_df), n_components=50))
```

Paired chains concatenate per-chain embeddings via `PairedTCREmp`. Input/output are AIRR polars
frames keyed by `vdjtools.io.schema` column names.

## Command line

`pip install mirpy-lib` also installs a `mir` command for the two embedding scales — no Python
needed. Inputs are any format `vdjtools.io` reads (AIRR TSV, vdjtools, MiXCR, immunoSEQ, parquet).

```bash
# one repertoire  ->  per-clonotype embedding table (e0…), the input to clustering / ML
mir embed clonotypes sample.tsv --pca 50 -o clonotypes.parquet

# a dataset of repertoires  ->  one fingerprint Φ(S) per sample, per chain (phi0…), on one
# shared basis so the rows are mutually comparable; --mmd also writes the pairwise MMD matrix
mir embed repertoires cohort/*.tsv.gz -o phi.tsv --mmd mmd.tsv

# the portable signature  ->  one fixed, named, standardised feature vector per sample
# BOTH halves: vdjtools statistics + mirpy embedding geometry, in one vector.
mir presets                                       # the named feature sets, ranked
mir presets transfer                              # what one preset is, and when to use it
mir signature cohort/*.tsv.gz --preset classify --threads 0 -o sig.parquet   # 0 = every core
mir signature cohort/*.tsv.gz -o sig.parquet --tier standard
mir signature --describe --preset compact         # the column dictionary; reads no input
```

`mir embed clonotypes -h` / `mir embed repertoires -h` list every flag (species, locus,
prototype count, weight, Φ blocks, …). Sample id defaults to the filename stem; the locus is
inferred per file (or restrict with `--locus`, which takes aliases — `beta`, `T-alpha` — and errors
on anything it can't resolve). An MMD matrix is per chain, so across several loci `--mmd mmd.tsv`
writes `mmd.TRB.tsv`, `mmd.TRA.tsv`, …; with one locus the name is used as given.

Both commands drop non-coding clonotypes (stop codon / legacy out-of-frame markers in
`junction_aa`) before embedding by default — pass `--no-filter-functional` to skip this. Without
it, a stop codon silently produces a numerically meaningless embedding and an out-of-frame `_`
marker crashes the run outright (neither is a valid amino acid).

## Recommended presets

`TCREmp.from_defaults(species, locus)` uses the per-chain preset when `n_prototypes` is
omitted. Values are data-driven from the bundled prototypes (prototype geometry saturates by
these counts; PC columns are the PCA dims retaining ~95% / ~99% variance):

| chain | n_prototypes | PCs (95%, clustering) | PCs (99%, reconstruction) |
|---|--:|--:|--:|
| human TRA | 2000 | 65 | 220 |
| human TRB | 2000 | 65 | 260 |
| human TRG | 1000 | 25 | 100 |
| human TRD | 2000 | 65 | 280 |
| human IGH | 2000 | 65 | 300 |
| human IGK | 1000 | 20 | 65 |
| human IGL | 1000 | 20 | 65 |
| mouse TRA | 2000 | 50 | 150 |
| mouse TRB | 2000 | 55 | 225 |

Use **95%** PCs for clustering/visualization (the paper's regime); use **99%** PCs when
*reconstructing* sequences with the neural inverse codec (diverse chains like IGH/TRD/TRA lose
too much sequence detail at 95%). Programmatically: `from mir.embedding import get_preset`.

```python
from mir.embedding import get_preset
from mir.embedding.pca import pca_denoise
p = get_preset("human", "IGH")
Xc = pca_denoise(X, n_components=p.n_components)          # clustering
Xr = pca_denoise(X, n_components=p.n_components_recon)    # codec reconstruction
```

## Prototypes — which receptors, and how much do they matter?

Every embedding is *distances to prototypes*, so the prototype set **is** the coordinate system.
mirpy ships one per chain, and you get them without downloading anything:

| | |
|---|---|
| What | **10 000 real receptors** per chain — a uniform random sample (fixed `seed=42`) of unique, productive, germline-resolvable clonotypes from arda-annotated real repertoires |
| Why real | Model-generated junctions have degenerate lengths and embed measurably worse (negative self-prototype distance correlation); real repertoires give a tight, well-behaved manifold |
| The default | `replicate=0` — the first `n` rows. This is *the* set: every preset, bundled codec, and published number uses it. Don't change it unless you're deliberately testing sensitivity |
| Chains | human TRA/TRB/TRG/TRD/IGH/IGK/IGL, mouse TRA/TRB (`list_available_prototypes()`) |

**Is my result an artefact of which prototypes I drew?** Take a replicate. The file order is itself a
uniform shuffle, so each disjoint block of `n` rows is an independent draw from the same pool —
`n_replicates()` of them, **10 at `n=1000`**, 5 at `n=2000`:

```python
from mir.embedding.prototypes import n_replicates
from mir.embedding.tcremp import TCREmp

scores = [my_metric(TCREmp.from_defaults("human", "TRB", 1000, replicate=r).embed(df))
          for r in range(n_replicates("human", "TRB", 1000))]   # 10 draws; spread = sensitivity
```

Same from the shell: `mir embed clonotypes sample.tsv --n-prototypes 1000 --replicate 3`.

**How much *does* it matter?** Usually very little, and you can check for yourself —
`bench.theory.prototype_source_correlation(queries, protos_a, protos_b)` correlates the pairwise
junction-distance geometry under two prototype sets. Two independent draws, 400 held-out human-TRB
queries:

| prototypes `n` | 100 | 250 | 500 | 1000 (default) | 2000 |
|---|--:|--:|--:|--:|--:|
| R between two draws | 0.922 | 0.971 | 0.990 | 0.993 | **0.997** |

So the geometry is essentially draw-independent from `n≈500` up: at the default counts, *which*
prototypes you drew is not what your result rests on. Below `n≈250` it starts to be.

> Each replicate is a **different coordinate system**. Distances *within* one are comparable;
> distances *across* two are not. The prototype hash covers the replicate index, so codecs,
> `RepertoireSpace` and `DonorCohort` all refuse to mix them — compare summary statistics across
> replicates (AUC, F1, cluster counts), never raw embeddings. Sweeping `n_prototypes` instead is a
> *nested* comparison (draw `r=0` at `n=500` is a prefix of `n=1000`), which answers "how many do I
> need", not "does it matter which".

Provenance and the regenerate command are in [`SOURCES.md`](SOURCES.md).

## What's inside

| Module | Purpose |
|---|---|
| `mir.embedding.tcremp` | `TCREmp` / `PairedTCREmp` — the prototype embedding |
| `mir.embedding.pca` | PCA denoising of embeddings |
| `mir.distances` | junction distance (`seqtree.gapblock`; `metric`/`matrix`/`alignment` options) + baked germline distances |
| `mir.bench` | VDJdb loader, clustering (`cluster(method=…)`: DBSCAN/HDBSCAN/OPTICS) + F1/retention, theory experiments (incl. `codec_losslessness`), cohort scorers (`bench.eval`: `cv_auc`/`cv_cindex`/`km_logrank`) + `recovery_report` (are the basic statistics carried inside the embedding?) |
| `mir.density` | continuous-density TCRNET/ALICE — enrichment (+ clonal-abundance channel, `backend=` exact/kdtree/ann) + noise-filtering (Theory T6) |
| `mir.repertoire` | sample-level (repertoire) embedding — RFF kernel mean ‖ Hill diversity ‖ second moment; MMD / HLA-stratified distance; motif witness; `centroid_atypicality`, multi-locus `fit_repertoire_spaces`; **sub-probability** `missing_mass`/`naive_reference`/`contrast_embedding` (Theory §T.7) |
| `mir.explain` | named-channel fusion (`ChannelBuilder`, incl. `preserve_magnitude` global scaling) + scorer-agnostic ablation (`channel_report`/`channel_drivers`) — which part of Φ carries the signal (§T.7) |
| `mir.cohort` | the **digital donor** — multi-chain `fit_donor_embeddings`/`DonorCohort` (+ `transform`/`save`/`load`) + `residualize` / `cluster_samples` / `incidence_biomarkers` (§T.7) |
| `mir.track` | **exposure trajectory** — PhenoPath-style covariate-disentangled latent progression axis (`fit_exposure_trajectory`) over any channel matrix; repertoire-level exposure detection |
| `mir.generate` | the **generative loop** (mechanical half) — `DescriptorDensity`: sample new synthetic donor states / `evolve` one along a coordinate, over `RepertoireDescriptor` |
| `mir.twin` | the **digital twin** — `DonorTwin`/`make_twins`: perturb or resample one donor's state through a `mir.generate`/`mir.ml.diffusion` generator |
| `mir.ml` | neural codecs (forward/inverse/Pgen/unified) + learned repertoire `set_encoder` (Set-Transformer/DeepRC) + conditional **diffusion generator** (`mir.ml.diffusion`, the generative loop's research half) — Part 2, experimental; `[ml]` extra |

## Background subtraction & clustering (`mir.density`)

TCRNET/ALICE find antigen-driven convergent clusters by *neighbour enrichment*. `mir.density`
does the same test with neighbour-counting in the **embedding space** instead of on a sequence
graph (Theory T6): the enrichment `E(z) = f_obs(z)/f_gen(z)` is estimated by an adaptive-bandwidth
**balloon** estimator with a per-clonotype Poisson/binomial significance test and BH q-values —
no graph, and it scales to whole repertoires.

```python
from mir.density import fit_density_space, neighbor_enrichment, enriched_mask, denoise_and_cluster
from mir.embedding.tcremp import TCREmp

model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
# background = a control repertoire (TCRNET) or generate_background(...) (ALICE, P_gen)
space, obs_emb, bg_emb = fit_density_space(model, obs_df, control_df, n_components=20, space="full")
res  = neighbor_enrichment(obs_emb, bg_emb, test="binomial")   # balloon + water-level calibration
hits = obs_df.filter(enriched_mask(res, alpha=0.05))            # background-subtracted clones
labels, mask = denoise_and_cluster(obs_emb, res)               # noise-filter + DBSCAN the hits
```

Use a **biological control** as the background when you have one (e.g. pre- vs post-vaccination,
patient vs healthy) — differential enrichment cancels generic public convergence and isolates the
antigen-specific response. No control of your own? Pooled healthy-donor repertoires are one fetch
away (HF `isalgo/airr_control`, read with `vdjtools.io.read` — see [`SOURCES.md`](SOURCES.md)).
Failing that, `generate_background(locus, n)` samples the vdjtools P_gen model (the ALICE regime);
the "water level" of a naive repertoire is handled by the empirical-null calibration. Pass
`source="arda"` there when your data is arda-annotated (same allele namespace as the prototypes),
and `species="mouse"` for mouse — both need a vdjtools shipping the bundled `arda` model set. The density benchmarks (YFV, ankylosing-spondylitis B27, TCRNET)
live in the companion [`2026-mirpy-analysis`](https://github.com/antigenomics) repo.

The default backend is `"kdtree"` (exact scipy cKDTree, all cores). At whole-repertoire scale pass
`backend="ann"` (pynndescent, ~30× faster past ~10⁵ clones; `pip install "mirpy-lib[ann]"`). Only
the **observed** side is approximate there — recall < 1 undercounts the observed ball, which biases
enrichment *down* (conservative). The **background** occupancy is always exact, because
undercounting it would shrink the expected count and inflate fold and significance, which is the one
direction an enrichment test must never err in.

## Sample-level (repertoire) embedding (`mir.repertoire`)

One fixed vector `Φ(S)` per **repertoire** — an order-invariant multiset of clonotypes with clone
sizes — depth-robust into the low-coverage bulk-RNA-seq regime (Theory §T.7). `Φ(S)` sketches the
empirical measure `ρ_S = Σ_σ w_σ δ_{φ(σ)}` in three blocks: an RFF **kernel mean** (depth-robust,
codebook-free — no `K`, no clustering), a coverage-standardized **Hill diversity** profile, and a
**second-moment** Fisher vector carrying clonotype co-occurrence (HLA-linked public structure).
Repertoire distance is the **MMD** `‖Φ₁(S) − Φ₁(S')‖`.

The per-clonotype weights `w_σ = g(a_σ)/Σ_τ g(a_τ)` come from a clone-size transform `g` (`weight=`
on `sample_embedding`/`fit_repertoire_space`/`mir embed repertoires --weight`): `"log2p1"` —
`g=log2(1+a)` — is the **default**, concave so one hyperexpanded clone can't dominate;
`"duplicate_count"` weights linearly by clone size (`g=a`); `"distinct"` ignores size entirely
(`g≡1`, presence only). `"log1p"` (natural log) and `"anscombe"` remain available.

```python
from mir.repertoire import fit_repertoire_space, sample_embedding, mmd_matrix, class_witness
from mir.embedding.tcremp import TCREmp
import polars as pl

model  = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
space  = fit_repertoire_space(model, pl.concat(samples))   # ONE basis for the whole cohort
embs   = [sample_embedding(space, s) for s in samples]     # Φ(S): mean ‖ diversity ‖ second moment
D      = mmd_matrix(embs, unbiased=True)                    # pairwise repertoire distance (unbiased MMD²)
motifs = class_witness(space, pos_samples, neg_samples, candidates)   # public clones separating two groups
```

**Comparability invariant** (as with the codecs / density): every sample in a cohort must be
embedded through *one* prototype set and *one* PCA+RFF basis, or the measures are incomparable —
`fit_repertoire_space` fits that basis once and `RepertoireSpace` refuses a prototype-hash mismatch.

Use the **unbiased** MMD (`unbiased=True`) whenever samples differ in depth/diversity — the biased
V-statistic's `1/n_eff` self-term otherwise inflates low-diversity samples and fakes a signal. When a
nuisance batch is present, compare *within-batch* contrasts (residualize `Φ` on the batch indicator):
a batch offset is first-order and cancels, while a batch-orthogonal signal (e.g. HLA) survives. The
empirical rule of thumb — **diversity for how-even, the embedding for which-clones**: clone-size
phenotypes (age, CMV) are a diversity summary's turf, while clonotype identity (HLA — strongest in
TRA and class II) lives in the second moment / witness. A learned co-equal set encoder
(Set-Transformer / DeepRC) is in `mir.ml.set_encoder` (`[ml]` extra). Recorded results and theory
(T7) live in the companion [`2026-mirpy-analysis`](https://github.com/antigenomics) repo
(`benchmarks/{BENCHMARKS,THEORY}.md`) alongside the benchmark scripts.

### Sub-probability embeddings: the deficient measure

`Φ(S)` above is the kernel mean embedding of a **probability** measure — the weights sum to 1, so
every sample asserts one full unit of confidence. At RNA-seq depth that premise fails. Measured: the
**median tissue TRB sample holds 21 unique clonotypes** (blood TRB
254, 1st percentile 1), so `w_σ = a_σ/Σa` is `1/n` for a *technical draw size*, not a clonal
frequency — the true frequencies live at 1e-5…1e-8, and one singleton's weight spans **21,454×**
across blood TRB purely from sample size. Worse, normalising to 1 *forces* a 5-clonotype tumour to
assert full confidence, so it lands somewhere arbitrary on the unit sphere instead of where it
belongs, and callers respond with a minimum-clonotype floor which in tumour deletes the **immune
desert** — the phenotype of interest. (A floor once cut 7,179 labelled donors to 2,129.)

Let the measure be **sub-probability** instead:

```python
from mir.repertoire import contrast_embedding, missing_mass, naive_reference, sample_embedding

emb = sample_embedding(space, sample, missing_mass="chao")   # or "turing"; "none" = old behaviour
emb.mass                                    # retained mass 1 − M₀ ∈ [0, 1]
ref = naive_reference(space)                 # kernel mean of 20k naive V(D)J recombinations (~8 s)
psi = contrast_embedding(emb, ref)           # Ψ = mass·(Φ − naive): signed, magnitude-carrying
```

* **`missing_mass`** estimates the mass `M₀` of the clonotypes that were never drawn — Good–Turing
  (`f₁/N`) or **bias-corrected Chao1** (`S_u/(N+S_u)`, `S_u = f₁(f₁−1)/(2(f₂+1))`; never the
  classical `f₁²/2f₂`, undefined when no clone was seen exactly twice, which is common here).
  `missing_mass=` only sets `.mass`; the blocks are untouched, and the `"none"` default is
  bit-identical to before.
* **Not** a negative measure. `Φ`'s value is that `‖Φ_P − Φ_Q‖` *is* the MMD and that a convex
  combination of two `Φ`'s is the `Φ` of a real pooled repertoire (what makes `mir.twin` and
  trajectory interpolation mean anything); a measure allowed to go negative on a set is not a
  probability measure and loses both. A sub-probability measure costs neither.
* **`naive_reference`** gives the unseen block a principled location — the germline recombination
  model (`vdjtools.model.generate`), not the corpus centroid. This is the load-bearing choice:
  shrinking toward the centroid is James–Stein toward the mean and it measurably **hurt** (it piles
  shallow samples into a dense ball that is itself depth-correlated), while the germline draw dropped
  `R²(PC1, depth)` from 0.259 to **0.001** (blood TRB) and 0.067 to **0.006** (tissue IGH) with kNN
  label entropy unchanged or better.
* **`contrast_embedding`** is where legitimate negativity lives: a signed *difference of two
  probability measures*, negative wherever the sample is depleted relative to unselected
  recombination, still an ordinary RKHS element with `‖Ψ_S‖ = MMD(S, naive)`. Magnitude then reads
  as **confidence × deviation-from-naive**: an immune desert has `M₀ → 1` and lands at the
  **origin**, the right place for "no infiltrate detected", and a shallow blood sample says so by
  its norm instead of being dropped. Give the caller `mass` to weight with; don't add a floor.

> ⚠ **Scale a magnitude-carrying block with one global scalar, never per column.** Per-column
> standardisation forces every coordinate to unit variance across samples, so a matrix where half the
> rows sit at the origin comes out looking exactly like one where none do — it deletes the deficiency
> it was built to preserve. Use `ChannelBuilder.add(..., preserve_magnitude=True)`, which applies one
> pooled RMS per channel and fills holes with `0`. (`stack_embeddings` warns if it is handed
> deficient-mass embeddings, since `Φ.vector` does not carry the mass.)

### Functional diversity, compartments, and depth

Four more consequences of the same measure algebra, all of them cheap once `Φ` is a kernel mean:

```python
from mir.repertoire import (band_embeddings, depth_threshold, mixture_weights,
                            rao_q, rarefy_embedding, sample_statistics)

rao_q(emb)                          # 1 − ‖Φ₁‖² — Rao's quadratic entropy, exactly
depth_threshold(embs).kappa         # the size below which Φ is mostly sampling noise
bands = band_embeddings(space, sample)              # singleton / expanded / top (or IGH isotypes)
mixture_weights(emb, bands)["weights"]              # π per compartment, by NNLS
rarefy_embedding(space, sample, depth=20_000).v_rep # matched-depth Φ + its replicate noise
```

* **`rao_q`** — Rao's quadratic entropy is `1 − ‖Φ₁‖²` *exactly* (verified against an explicit Gram to
  ~1e-16), so the **norm** of the kernel mean is a diversity statistic and no Gram matrix is needed.
  It is the diversity the Hill block cannot express: every Hill number is a functional of the
  clone-size distribution alone, hence invariant to permuting *which* receptor carries which
  abundance, while Rao's Q weights each pair by how different the receptors are. Measured: this one
  scalar recovers R² 0.74–0.85 of classical diversity, and embedding derivatives reach R² 0.974–0.994
  for Shannon. Valid only on the **uncentred** Φ₁ — centring preserves differences (MMD) but not norms.
* **`depth_threshold`** — the damage depth does to a kernel mean is not bias but **variance ∝ 1/n**.
  Regressing `‖Φ_S − Φ̄‖²` on `1/n` splits the spread into between-sample signal `τ²` and sampling
  noise `σ²`, so **κ = σ²/τ²** is the size at which they are equal. Measured κ ≈ 40–70 clonotypes
  across four independent views, with 23–69% of samples below it. Report κ for *your* cohort instead
  of importing a cutoff — and note the library still applies no floor.
* **`band_frames` / `band_embeddings` / `mixture_weights`** — Φ₁ is an *average*, which is the right
  operation for a population mean and the wrong one for a minority signal: writing the repertoire as
  `(1−π)ρ_naive + π ρ_expanded` shows a compartment-confined effect reaching Φ₁ attenuated to `πΔ`
  while the naive compartment supplies the noise. Bands (`singleton` / `expanded` / `top 1%`, or IGH
  isotypes from `c_call`) are embedded through the **same frozen space** — never refit, or band-to-band
  distances stop meaning anything — and bands under 5 clonotypes are recorded absent (`None`), not
  upsampled. Because mixture linearity `Φ(S) = Σ π_c Φ(c)` is exact, a non-negative least
  squares recovers each compartment's share (exact in exact arithmetic; float32 embeddings put the
  realised residual near 1e-5). Measured on IGH isotypes: class-switched **IgG carries
  π = 0.070** of Φ₁(IGH) (IgM 0.230, IgA 0.176, 0.520 uncalled) — the dilution factor measured rather
  than assumed, and a power calculation before you spend compute: a subset with π ≈ 0.001 is not
  detectable by any aggregate distance, so use the per-clonotype witness. Straight about the negative:
  on survival endpoints the bands did **not** beat a diversity reference (0/22 pre-registered cells),
  though `singleton` reproduced a clinical-only score and `expanded` reproduced the whole-repertoire
  score exactly as the mixture argument predicts, and banding did win on kNN entropy in tissue IGH
  (0.3875 vs 0.4686).
* **`rarefy_embedding`** — averaging Φ over multinomial subsamples is itself a kernel mean (of the
  mixture over subsamples), so it is the **only** depth correction that keeps MMD, Rao and mixture
  linearity exactly (~1e-15); an orthogonal projection breaks the norm identity and a per-coordinate
  rescale breaks both. It also hands you a free per-sample noise estimate, because
  `Rao(Φ̄) = mean_r Rao(Φ_r) + v_rep` holds **exactly** — the excess diversity of the average *is* the
  replicate variance. Not a default: rarefying a cohort to its shallowest useful depth throws away the
  deep samples' advantage.

Also `sample_statistics` / `cohort_statistics` (the sampling fingerprint: f₁/f₂/f₃₊, singleton
fraction, missing mass, top-clone fraction, library size — the `stats=` input below, and candidate
biology in their own right) and `mir.cohort.depth_report`, which regresses the leading PCs on that
fingerprint. Read it beside `explained_variance`: with the deficient measure, R²(PC1, depth) fell
0.259 → 0.001 and best-of-PC1–5 0.253 → 0.047 *while PC1's explained variance was unchanged*, which is
what distinguishes "a different direction" from "a collapsed one".

> ⚠ **`mir.cohort.residualize(..., shrink=True)`** for batch offsets fitted from few samples in many
> dimensions. Plain per-group centring can make the batch *easier* to read, measured out-of-sample:
> batch-identity AUC 0.863 raw → **0.985** after centring, 0.978 after ComBat. The mechanism is
> estimation error — `‖µ̂ − µ‖ ≈ √(σ²d/n)` was ≈16 against a true offset of 7–24, so subtracting it
> injects a batch-constant vector as large as the one it removes, invisible in-sample by construction.
> Positive-part James–Stein shrinkage recovered most of it: 0.985 → **0.889**.

The matching evaluation criterion is **recoverability, not competition**:
`mir.bench.recovery_report(X, stats, groups)` runs a grouped-CV ridge from the embedding's PCs back
to each basic repertoire statistic (richness, Shannon, top-clone fraction, singleton fraction, Chao
unseen fraction, library size) and reports R² — high means the statistic is *carried inside* the
embedding and nothing has to be bolted on beside it. Renormalising to mass 1 deletes the magnitude,
so coverage and richness are unrecoverable from `Φ` **by construction**; the deficient measure should
win that question as a design consequence. It sits beside `mir.cohort.missingness_report` as the
other "is this object honest" check.

## The portable signature (`mir.signature`)

`Φ(S)` is a fingerprint, but not one you can hand over: its basis is fitted on *your* cohort, so two
collaborators get incomparable vectors. The **signature** is the hand-off object — a fixed-width,
name-addressed, already-standardised vector that anyone who `pip install mirpy-lib` can compute from
their own AIRR files and drop into PCA, logistic regression, boosting or an MLP with no scaler of
their own.

```python
from mir.signature import signature, signature_cohort, describe

v = signature({"TRB": df})                # 688 named columns (standard tier), ~0.3 s/sample
F = signature_cohort(samples)             # one row per sample, positional
describe("standard")                      # column, sig, block, locus, feature, transform, flags
```

Two halves, concatenated on `sample_id` and namespaced so they never collide: `vsig` (statistics of
the clone-size vector, from [vdjtools](https://github.com/antigenomics/vdjtools)) and `rsig`
(geometry — every column a linear functional, a norm, or a mixture coefficient of `Φ`). Tiers
`core ⊂ standard ⊂ full` are exact **index subsets** of one frozen column order.

Three properties are what make it portable, and each was measured rather than assumed
(numbers in the analysis repo's `benchmarks/SIGNATURE_SCALING.md`):

- **The basis is fit-free.** The rotation reducing `Φ` to coordinates is the PCA of the *bundled
  prototype panel* — no samples, nothing to re-fit, so nobody's coordinates move when a reference
  is refreshed. A basis fitted on a corpus instead is not merely worse but unusable at the sizes
  anyone has: split-half column agreement of a fitted junction basis is **0.23**.
- **Only location and scale come from data**, and they converge. Median and `1.4826·MAD` reach
  0.992 of columns within tolerance at **N = 1,000** reference samples; mean and sd never get there
  (0.841 at N = 2,000), because the columns are heavy-tailed — 0.4–27% of samples sit beyond five
  robust deviations, against the 6·10⁻⁷ a normal gives.
- **Holes are never zeros.** An unsequenced locus, a compartment below its clonotype floor, a
  statistic the sample is too shallow to estimate — each is `nan` plus a `mask:` column, because a
  model that reads "absent" as "zero" reads an unsequenced chain as biology.

What it does *not* fix is batch. Fitted on one cohort and applied to another the spread transfers
(robust sd of z = 1.008) and the offset does not (1.34 robust deviations); `fit_scale(...,
group=...)` records which columns are batch channels so you are handed the list rather than left to
find it. The geometry blocks are the cleanest on that measure and the depth summaries the worst.

## Exposure trajectory, generative loop, digital twin

A repertoire cohort often has a **known covariate** (HLA, batch, vaccine arm) but an **unknown or
noisy progression axis** (days since exposure, response severity). `mir.track.fit_exposure_trajectory`
recovers that latent trajectory `tau` while disentangling it from the covariate — a PhenoPath-style
model (Campbell & Yau 2018, *Nat. Commun.*
[10.1038/s41467-018-04696-6](https://doi.org/10.1038/s41467-018-04696-6), adapted from genes×cells to
repertoire-channels×samples) over *any* per-sample channel matrix (a stacked `Φ`, a `ChannelBuilder`
build, or a raw embedding block):

```python
from mir.explain import stack_embeddings
from mir.track import fit_exposure_trajectory

X, spec = stack_embeddings(embs)                 # channels: mean ‖ diversity ‖ second
fit = fit_exposure_trajectory(X, hla_indicator, channel_names=spec.names)
fit.tau                    # inferred exposure/progression pseudotime, one per sample
fit.top_interactions(5)    # which channels respond to progression differently by covariate
```

Complementing that, `mir.generate.DescriptorDensity` fits a (optionally class-conditional) density
over `RepertoireDescriptor` vectors — `sample` draws brand-new synthetic donor states, `evolve`
perturbs one donor along a coordinate ("what if hotter") and propagates the coupled shift through
every other coordinate via the fitted covariance's conditional mean. `mir.ml.diffusion` (`[ml]`
extra) is the non-linear alternative — a compact conditional DDPM/DDIM generator with
classifier-free guidance, sharing the same `sample(n, condition=…)` call shape so it drops in
unchanged. `mir.twin.DonorTwin` glues a donor's descriptor + trajectory position + covariate into
one object you perturb or resample through, instead of threading the three APIs together by hand:

```python
from mir.generate import fit_descriptor_density
from mir.twin import make_twins

density = fit_descriptor_density(descriptors, labels=tumor_type)
twins = make_twins(descriptors, conditions=tumor_type, donor_ids=sample_ids)
hotter = twins[0].perturb(density, coordinate="infiltration", delta=2.0)
synthetic = twins[0].simulate(density, n=20)     # 20 new synthetic peers of donor 0
```

See `examples/trajectory_and_twin.py` for a runnable end-to-end demo.

## Reproduce the paper

The self-contained theory notebooks run on bundled data:

```bash
pip install "mirpy-lib[examples]"
marimo edit examples/theory.py          # supplementary S1–S3 (distance laws, D↔d, prototype robustness)
marimo edit examples/quickstart.py      # embed + cluster (epitope colours need a local VDJdb dump)
```

The full benchmark suite (VDJdb Table S1, density, repertoire/TCGA) and result docs live in the
companion analysis repo [`2026-mirpy-analysis`](https://github.com/antigenomics) — this repo is the
library + CI tests only.

Method: Kremlyakova *et al.*, *TCREMP: a bioinformatic pipeline for efficient embedding of
T-cell receptor sequences*, **J Mol Biol** 437 (2025) 169205.

## Performance & parallelism

mirpy is CPU-parallel by default and uses the GPU for the neural codecs. Knobs, by hot path:

| Stage | Knob | Default | Notes |
|---|---|---|---|
| Embedding (junction distance) | `TCREmp(..., threads=N)` | `0` = **all cores** | The C++ `seqtree.gapblock` scorer; releases the GIL, ~530 M pairs/s @16 cores. `threads=1` for a serial run. |
| Density kNN / balloon | `neighbor_enrichment(..., backend=…)` | `"kdtree"` (scipy cKDTree, **all cores**) | Exact and multithreaded (`workers=-1`), 5–9× faster than the BallTree baseline. `backend="ann"` = pynndescent, auto all-core, ~30× at ≥1e5 (observed side approximate/conservative, background exact); `backend="exact"` = the 1-core BallTree baseline, for reproducing older runs. |
| Clustering | `cluster(..., n_jobs=-1)` | sklearn default (1) | forwarded to DBSCAN/OPTICS/HDBSCAN via `**kwargs`; parallelizes the neighbour search. |
| BLAS (PCA, RFF, matmul) | `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` env | all cores | numpy/sklearn use the platform BLAS; cap via env if oversubscribed. |
| Neural codecs (`mir.ml`) | `pick_device()` / `device=` / `MIR_DEVICE` env | **CUDA → MPS → CPU**, auto | every `train_*` / codec / bundle takes `device=`; e.g. `MIR_DEVICE=cuda:1` pins the second GPU. Torch-free paths (`density`, `repertoire`) never touch the GPU. |

Rule of thumb: leave `threads=0` (all cores) for embedding; leave density on the default
`backend="kdtree"` (exact, multicore) and switch to `"ann"` only at whole-repertoire scale; the GPU
is used only by `mir.ml`.

## Development

Repo-local `.venv` via [uv](https://docs.astral.sh/uv/) (bash/zsh): `bash setup.sh` — add
`--dev-parents` to editable-install the sibling `seqtree` / `vdjtools` / `vdjmatch` checkouts and
`--tests` to run the fast suite. Tests: `python -m pytest tests/ -q`. See [`CLAUDE.md`](CLAUDE.md)
for the architecture and reuse map.
