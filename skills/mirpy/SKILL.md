---
name: mirpy
description: >
  The public API of mirpy v3 (PyPI `mirpy-lib`, import `mir`) — ML-oriented embeddings for immune
  receptor repertoires. Use when embedding TCR/BCR clonotypes or whole repertoires, computing MMD
  between samples, running density/enrichment (TCRNET/ALICE) analyses, building a multi-chain
  "digital donor" matrix, explaining which channel carries a signal, or training the neural codecs.
  Covers what each module exposes, which function to reach for, and the invariants that make two
  embeddings comparable.
---

# mirpy v3 — API surface

`mirpy` turns receptor sequences into fixed-length vectors at two scales: **per clonotype**
(TCREMP: distances to a fixed prototype panel) and **per repertoire** (`Φ(S)`: a kernel-mean
sketch of the clone-size-weighted measure). Everything downstream — clustering, density,
cohort fusion, generation — is built on those two objects.

Frames are **polars**, keyed by AIRR names (`v_call`, `j_call`, `junction_aa`, `duplicate_count`,
`locus`). There is no `Clonotype` class and no data-model layer: IO, germline reference, Pgen and
sampling all come from `vdjtools`; alignment comes from `seqtree`.

## The one invariant that matters

**Two embeddings are comparable only if the prototype panel *and* the fitted PCA rotation match.**
Everything that can be serialized (`RepertoireSpace.save`, `DonorCohort.save`, `CodecBundle`,
`SetEncoderBundle`, `DiffusionModel`) stores a prototype hash and refuses to load into a
mismatched basis. Never hand-assemble a matrix from two separately-fitted spaces; project
held-out data through the stored basis instead (`DonorCohort.transform`,
`RepertoireSpace.transform_clonotypes`).

Corollary: v3 embeddings are a **new coordinate system**. Anything trained on v2 embeddings
must be retrained.

## Install

```bash
pip install mirpy-lib                     # core: numpy, polars, scikit-learn, scipy, seqtree, vdjtools
pip install "mirpy-lib[bench]"            # benchmark harness (kneed, matplotlib, lifelines, HF)
pip install "mirpy-lib[ann]"              # approximate-NN density backend (pynndescent + numba)
pip install "mirpy-lib[ml]"               # torch: codecs, set encoder, diffusion
pip install "mirpy-lib[build]"            # BioPython + arda: regenerate baked resources, 'sw' alignment
```

## Command line

```bash
mir embed clonotypes  SAMPLE  -o out.parquet      # per-clonotype table (e0…)
mir embed repertoires S1 S2 … -o phi.tsv --mmd mmd.tsv   # one Φ(S) per sample per chain (phi0…)
```

- Reads anything `vdjtools.io.read` sniffs (AIRR TSV, vdjtools, MiXCR, immunoSEQ, parquet).
  Writes TSV, or Parquet when the path ends `.parquet` — prefer Parquet for the wide raw embedding.
- Non-coding clonotypes (stop codon `*`, out-of-frame `_`) are dropped **by default**;
  `--no-filter-functional` keeps them, and then a `_` will legitimately fail in the C++ scorer.
- `--locus` takes aliases (`beta` → `TRB`) and errors on anything unresolvable.
- With several loci, `--mmd mmd.tsv` writes `mmd.TRB.tsv`, `mmd.TRA.tsv`, … (one matrix per chain).

## Modules

### `mir.embedding` — the clonotype embedding

| Symbol | Use |
|---|---|
| `TCREmp` / `PairedTCREmp` | polars frame → `(N, 3K)` float32. `from_defaults(species, locus)` picks the per-chain preset. Row-preserving: output row `i` is input row `i`. |
| `TCREmp.embed(df)` | The workhorse. Rejects null and `_`-containing `junction_aa` with a message naming the count. |
| `list_available_prototypes`, `load_prototypes`, `n_replicates` | The bundled arda-native prototype panels, and independent replicate draws of them. |
| `get_preset` / `ChainPreset` | Recommended `n_prototypes` + PC counts per chain. Compact chains (IGK/IGL/TRG) 1000/~20 PCs; diverse (IGH/TR*) 2000/~65 PCs at 95% var, ~220–300 at 99%. |
| `pca_denoise` | PCA compaction of an embedding matrix (T3). |

Knobs on `TCREmp`, all defaulting to the published space: `metric="squared"` (vs `"sqrt"`),
`matrix=` a custom `seqtree.SubstitutionMatrix`, `alignment="gapblock"` (vs paper-exact `"sw"`,
validation-only), `mode="vjcdr3"` (vs `"cdr123"`), `threads=0` (all cores).

### `mir.distances` — the geometry underneath

`junction_distance_matrix(queries, refs, …)` (seqtree gapblock, GIL-released) and
`load_germline_distances` / `GermlineDistances` (baked V/J/CDR1/CDR2 lookup with allele cascade).
Reach for these directly only when you want the raw distance, not an embedding.

### `mir.repertoire` — the sample-level embedding `Φ(S)`

The core of the cohort tier. `Φ(S)` = RFF **kernel mean** ‖ coverage-standardized **Hill**
diversity ‖ **second-moment** Fisher block.

| Symbol | Use |
|---|---|
| `fit_repertoire_space` / `fit_repertoire_spaces` | Fit one shared basis over a pooled clonotype cloud (the latter: one per locus). Everything else projects through it. |
| `sample_embedding` → `SampleEmbedding` | One sample → `Φ(S)`. `weight=` picks the clone-size transform; `blocks=` picks which blocks to compute. |
| `mmd_distance` / `mmd_matrix` / `hla_stratified_mmd` | Repertoire distance. **Use `unbiased=True`**: the biased V-statistic's `1/n_eff` self-term inflates low-diversity samples and fakes divergence. |
| `class_witness` | Supervised MMD motif finder — which clonotypes drive a group difference. |
| `sample_statistics` / `cohort_statistics` | The sampling fingerprint (12 statistics). Feed to `recovery_report` and `depth_report`. |
| `missing_mass`, `naive_reference`, `contrast_embedding` | The sub-probability tier — see below. |
| `rao_q`, `depth_threshold` | Functional diversity (`1−‖Φ₁‖²`, valid **uncentred** only) and the estimable depth scale `κ`. |
| `band_frames` / `band_embeddings` / `mixture_weights` | Compartment decomposition through one frozen space + NNLS shares. |
| `rarefy_embedding` → `RarefyResult` | The only depth correction preserving MMD / Rao / mixture linearity exactly. Not a default. |
| `sample_descriptor` / `RepertoireDescriptor` / `decode_metrics` | The smooth, **mass-preserving** descriptor — what `mir.generate` and `mir.twin` operate on. |
| `correct_batch`, `centroid_atypicality` | Harmony-like cluster-aware batch correction; per-sample cosine distance to a group centroid. |

**Clone-size weights** (`weight=`): `"log2p1"` = `log2(1+a)` is the **default** (concave, so one
hyperexpanded clone can't dominate); `"duplicate_count"` = `a` (linear); `"distinct"` = `1`
(presence only); `"log1p"`, `"anscombe"` also available.

**Sub-probability**: `Φ` normalised to mass 1 is a lie at RNA-seq depth. `missing_mass(counts,
"turing"|"chao")` estimates the never-drawn mass; `sample_embedding(missing_mass=…)` records
`.mass`; `contrast_embedding = mass·(Φ − naive)` is **signed**, so an immune desert lands at the
origin rather than being deleted by a minimum-clonotype floor. Never add such a floor — it is a
blood rule, not a tissue rule. `missing_mass` requires integer counts and rejects a frequency
column rather than silently returning 0.

`rao_dispersion(U, w)` is the Euclidean companion to `rao_q`, for the fit-free prototype-sum
representation where there is no kernel to lean on. For squared distance the double sum
telescopes to `2(Σw‖u‖² − ‖Φ‖²)`, so functional diversity rides along in the same chunked pass
and never needs an `n × n` Gram. The `n_eff/(n_eff−1)` correction is on by default: the self-pair
bias is of order `1/n_eff`, which varies by orders of magnitude across samples *and* correlates
with phenotype.

### `mir.signature` — the portable signature (geometry half)

A **fixed-width, name-addressed per-sample vector meant to be handed to someone else**, on a
scale their model can consume without fitting a scaler. The column contract lives in
`vdjtools.signature` (which mirpy already depends on, so there is one layout and one transform
registry rather than two that drift); this package supplies the `rsig` blocks, `vdjtools.signature`
supplies the `vsig` ones, and the two concatenate on `sample_id`.

| | `vsig` — statistics | `rsig` — geometry |
|---|---|---|
| blocks | `depth` `div` `clon` `len` `iso` `shm` `pair` `aa` `pchem` `pgen` (+`usage` `pub` pending) | `depth` `div` `band` `contrast` `phiv` `phij` `phic` |
| each column is | a defined statistic of the clone-size vector or the germline vocabulary | a linear functional, a norm, or a mixture coefficient of `Φ = Σ w_σ z_σ` |

`depth` and `div` appear under **both** on purpose — the count-native and embedding-native
readings of the same idea are different objects (`n_eff = 1/Σw²` is a Hill number of the weights
the geometry actually uses), and their head-to-head is a result. The `sig` prefix keeps them apart.

The geometry is **fit-free**: `z_σ` is a distance vector to the bundled prototype panel and the
rotation is the PCA of the *prototype cloud*, fitted to no samples at all, so nothing drifts when
a reference is re-fit. Measured (gate B1a, two cohorts): it retains ≥0.98 of a sample-fitted
rotation at the shipped widths, whereas a *fitted* junction basis has split-half column agreement
of 0.23 — its coordinates do not survive splitting one cohort in half.

`Φ` **must be centred** against the frozen `mu_phi` or the block is nearly blank: every prototype
distance is large and positive, so raw between-donor cosine spans ~0.001 while the shared offset
is 55× the between-donor signal. Centred, the same donors span 1.48. Same reason `contrast`
subtracts a naive reference instead of reporting `Φ`.

Compartment shares are **closed form, not NNLS**: `Φ` is linear in the clone-weight measure and
the bands are a genuine partition, so a compartment's share of `Φ` is exactly its share of the
weight. (`repertoire.band_frames`' default bands overlap — `top ⊂ expanded` — so an NNLS over
them is not a composition at all and its shares can exceed 1.)

**Two artifacts, and the split is the point.** `rsig_v1.npz` (7 loci, 882 KB, **bit-identical on
rebuild**) holds the geometry and is built from bundled resources with no sample read.
`rsig_scale_v1.npz` holds the one thing that *must* come from data — per-column location and
scale — plus the measured `cstar` and `pgen_q05`. Fitting a scale is a different statistical
problem from fitting a basis: a rotation over p=256 is not identified at a thousand samples,
while a median and MAD are identified at any n and converge as `1/sqrt(n)`. That is why the
re-fit story is safe.

```python
from mir.signature import signature, signature_cohort
v = signature({"TRB": df})                 # 716 named columns, standardized, ~0.26 s/sample
F = signature_cohort(samples)              # one row per sample, positional
signature(sample, standardize="none")      # raw values
```

`fit_scale` refuses a column the corpus barely saw (`min_n_obs`) rather than shipping a
confident-looking number from nine samples; statistics come from observed entries only, before
any imputation, since filling first deflates the scale in proportion to sparsity and lets the
least-observed locus dominate. A hole stays `nan` — never centred, never zero-filled.

### `mir.cohort` — the digital donor

`fit_donor_embeddings` → `DonorCohort`: per-chain identity (kernel mean, cross-sample
PCA-reduced) ‖ diversity ‖ coverage, fused across loci through one `ChannelBuilder`, with an
`extra_channels` hook for study-specific blocks. `save`/`load` verify every prototype hash and the
stored identity PCA; **`transform` is the only comparable path for held-out donors** (and keeps the
identity block as holes when the fit cohort had too few donors to reduce it, rather than refitting).

Also: `residualize` (batch offset removal; `shrink=True` applies positive-part James–Stein,
because plain per-group centring made batch *easier* to read out-of-sample), `cluster_samples`,
`incidence_biomarkers`, `align_loci`, `missingness_report`, and `depth_report` — R² of the leading
PCs on the sampling fingerprint. `depth_report` needs meaningfully more samples than statistics:
it saturates at `len(stats)+1` (R² ≡ 1 on any input) and now returns `nan` with a warning below one
residual degree of freedom. Read `residual_dof` before trusting the numbers.

### `mir.density` — continuous TCRNET/ALICE (T6)

Graph-free balloon enrichment `E(z) = f_obs / f_gen` in embedding space. Torch-free.

`fit_density_space` (one shared PCA basis) → `neighbor_enrichment` (adaptive-radius Poisson /
binomial + water-level calibration) → `enriched_mask` / `denoise_and_cluster`.
`generate_background` samples the vdjtools P_gen model.

- `backend=`: `"kdtree"` (**default**, exact scipy cKDTree, multicore) | `"exact"` (1-core
  BallTree baseline) | `"ann"` (pynndescent, ~30× at ≥1e5). Under `"ann"` only the *observed*
  side is approximate (biasing enrichment down, conservative); the background occupancy is exact.
- **Prefer a biological control over a P_gen background.** Real repertoires are pervasively
  convergent, so P_gen flags ~40% of clones; a differential control (day15-vs-day0, B27±,
  CMV-vs-control) gives ~46× the signal-to-noise. Process the **full** repertoire — subsampling
  dilutes the sparse antigen clusters.
- Abundance-aware: pass `abundance=` + `weight=` to swap the distinct in-ball count for a
  variance-stabilised mass, plus a per-clonotype orphan/depth channel Fisher-combined with breadth.

### `mir.explain` — which channel carries the signal

`ChannelSpec` / `ChannelBuilder` / `stack_embeddings` attach the name→column map `Φ.vector` does
not carry. `channel_report(X, spec, scorer, …)` ablates each channel under a **caller-supplied**
scorer — the library never sees `y` and ships no scorers, so a Cox C-index and a CV AUC both plug
in. `mode="in"` (default) is marginal; `"both"` adds the conditional half, and high `delta` with
`delta_out ≈ 0` is the **redundancy** signature. `channel_drivers` hops channel → clonotypes, but
only for a channel declared `attributable` (a kernel mean) — a Hill number has no clonotype
pre-image and it raises.

`add(..., preserve_magnitude=True)` uses one global scalar for a block whose *magnitude* is the
signal (a `contrast_embedding`); per-column z-scoring deletes exactly that.

### `mir.generate`, `mir.twin`, `mir.track` — the generative and trajectory tiers

- `DescriptorDensity` / `fit_descriptor_density` / `evolve`: an optionally class-conditional
  Gaussian over descriptor vectors. `sample` draws synthetic donor states; `evolve` perturbs one
  coordinate and propagates the coupled shift via the fitted covariance's conditional mean. The
  covariance is dense and `p ≫ n` at the default `n_rff` — PCA-reduce the identity block before
  reading couplings off it.
- `DonorTwin` / `make_twins`: descriptor + optional trajectory position + covariate in one object,
  with `.perturb()` and `.simulate()` (either generator drops in — they share a `sample` shape).
- `fit_exposure_trajectory` → `TrajectoryFit`: a PhenoPath-style (Campbell & Yau 2018)
  covariate-disentangled latent factor model. `.top_interactions()` ranks channels.

### `mir.bench` — the evaluation harness (`[bench]`)

`cluster` / `cluster_metrics` / `estimate_dbscan_eps` (DBSCAN default, HDBSCAN ~3× coverage at
lower F1, OPTICS dominated); `load_vdjdb` / `antigen_subset`; the theory checks
`s2_dissimilarity_distance_correlation`, `shm_embedding_drift`, `tcrnet_convergence`,
`codec_losslessness`; and the scorers `channel_report` consumes — `cv_auc`, `held_out_auc`,
`cv_cindex`, `km_logrank`, `recovery_report`, `kmer_matrix`.

`recovery_report` asks **recoverability, not competition**: mass-1 renormalisation makes
coverage/richness unrecoverable from `Φ` by construction, so the deficient measure wins that by
design.

Note `estimate_dbscan_eps(X, k)` counts *other* points (`k`-NN excluding self); the raw kneedle
knee over-merges, so `cluster(eps_factor=0.4)` recovers the paper regime.

### `mir.ml` — neural codecs (`[ml]`, torch)

`train_forward_encoder` (seq → code, reconstruction cosine 0.998), `train_inverse_decoder`
(code → seq; **use 99% variance, not 95%** — geometry survives compaction, reconstruction does
not), `train_pgen_regressor` (~190× faster than the native DP, r 0.967), `train_unified_codec`
(joint, with a geometry-anchor term), `train_set_encoder` (learned repertoire track),
`train_diffusion` / `DiffusionModel` (conditional DDPM/DDIM with classifier-free guidance).

`CodecBundle` / `SetEncoderBundle` / `DiffusionModel.save` serialize the PCA transform + prototype
hash + weights and refuse a mismatched basis. **Ship a bundle, never bare weights.**
`train_val_test_split` refuses a split that leaves no validation or training rows.

Exact-match reconstruction is **training-data-limited, not architecture-limited**: n=20k→100k
drives exact 0.885→0.958. Optimal `(K, PC) = (2000, 300–500)`.

`pick_device()` = CUDA → MPS → CPU; override with `device=` or `MIR_DEVICE`.

## Gotchas

- **`c_call` / isotype is not in the embedding** and is not reconstructable from it — carry it as
  an exact stored column, like `v_call`/`j_call` metadata.
- **`junction_aa` vs `cdr3_aa`**: the junction includes the conserved Cys104/Phe118 anchors, IMGT
  CDR3 excludes them, so `junction_aa` is two residues longer. Confirm which your data uses.
- **Rao's Q is only valid uncentred** — `1 − ‖Φ₁‖²` is Rao's quadratic entropy exactly, and
  centring destroys that.
- **Baked resources are versioned artifacts.** Regenerate `germline_dist/*.npz` whenever the gene
  library changes (`build_germline_dist.py`, needs `[build]`).

## Testing

```bash
python -m pytest tests/ -q -m "not integration and not benchmark"   # fast tier
python -m pytest tests/ -q -m "not benchmark"                       # + torch / ANN / BioPython
```

All tests are self-contained on bundled resources — no network.
