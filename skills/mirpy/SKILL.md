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

## Where the detail lives

This file is the **API surface and the traps** — what to import, what to call, and what produces a
silently wrong answer. It does not repeat the reference documentation.

| You need | Go to |
|---|---|
| Runnable walkthroughs, every module | https://docs.isalgo.dev/mirpy/usage.html |
| Every symbol, autodoc'd | https://docs.isalgo.dev/mirpy/api.html |
| The portable signature, end to end | https://docs.isalgo.dev/mirpy/signature.html |
| The channel vocabulary | https://docs.isalgo.dev/mirpy/channels.html |
| Theory T1–T7 and the maths | https://docs.isalgo.dev/mirpy/math.html |
| Worked notebooks | https://docs.isalgo.dev/mirpy/notebooks.html |
| How to work in this repo, open loops | `CLAUDE.md` |
| Dataset provenance | per-artifact `manifest.json`; HF dataset cards |

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
mir presets                                       # named feature sets, ranked
mir signature S1 S2 …         -o rsig.parquet --tier standard  # geometry ONLY (rsig)
mir signature S1 S2 … --preset classify -o rsig.parquet        # preset's rsig columns only
# the vsig half is `vdjtools signature`; join the two on sample_id for the full vector
mir signature --describe --tier standard          # the column dictionary; reads no input
mir signature --channels                          # the channel vocabulary; reads no input
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

These live in **submodules, not the package root** -- `mir.distances` re-exports only `junction`:

```python
from mir.distances.junction import junction_distance_matrix   # seqtree gapblock, GIL-released
from mir.distances.germline import GermlineDistances, load_germline_distances
```

`GermlineDistances` is the baked V/J/CDR1/CDR2 lookup with an allele cascade. Reach for either
directly only when you want the raw distance rather than an embedding.

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

### `mir.signature` — the portable signature

Full reference: **https://docs.isalgo.dev/mirpy/signature.html** (design, transforms, scale
references, the measurements behind each choice) and
**https://docs.isalgo.dev/mirpy/channels.html** (the channel vocabulary). What follows is the
surface and the traps.

A fixed-width, name-addressed per-sample vector **meant to be handed to someone else**, already
standardised so their model needs no scaler. Two halves on one contract: `vsig` (statistics, from
`vdjtools.signature`) and `rsig` (geometry, here); they concatenate on `sample_id`. Column names
are `<sig>:<channel>:<locus>:<feature>`; tiers `core` (152) ⊂ `standard` (688) ⊂ `full` (1403) are
exact **index subsets** of one frozen order.

```python
from mir.signature import (signature, signature_cohort, channel_spec, channels,
                           columns, describe, load_scale, MODELS)
v = signature({"TRB": df})                 # 688 named columns, standardised, ~0.3 s/sample
F = signature_cohort(samples)              # one row per sample, positional
signature(sample, standardize="none")      # raw block values
```
```bash
mir signature --preset classify --scale blood cohort/*.tsv.gz -o sig.parquet
mir signature --describe --tier standard   # column dictionary; reads no input
mir signature --channels                   # channel vocabulary; reads no input
mir presets                                # the 8 named column subsets, ranked
```

**Presets are the entry point to recommend**, not hand-picked columns: `compact` (86),
`classify` (615), `transfer` (550), `geometry` (514), `statistics` (101), `bcell` (271),
`full` (1403, feature selection only) and `nuisance` (73, ranked *avoid* — a control). They
resolve from the frozen layout alone, so two people choosing one name get identical columns in
identical order. `mir presets NAME` prints any of them in full.

**Channels** are the interpretive layer: the second field of a column name, twenty of them
covering the whole vector, disjoint and exhaustive. `channel_spec(tier, columns=…, per_locus=…)`
wraps the name→index map as a `mir.explain.ChannelSpec`, which is the bridge to `channel_report`
and `channel_drivers` (see `mir.explain` below). Only the four `rsig` geometry channels are
`attributable`; asking a Hill number which clonotypes drive it raises.

**Scale references** — `--scale NAME|PATH`, `load_scale(name)`, `MODELS`. Six ship:
`deep-tcr` (default, targeted TCR, TRA+TRB only, 394/1403 columns scaled), `blood` and
`tissue` (bulk RNA-seq, all 7 loci, 1377 and 1360/1403), their `-unweighted` arms, and
`blood-v3` (superseded, kept loadable so nobody's numbers move).

#### Traps

- **The rotation is fit-free and must stay that way.** It is a PCA of the bundled prototype panel
  — zero samples — so no refit moves a stakeholder's coordinate. Touching
  `src/mir/resources/prototypes/*` or `PC_DIMS` breaks that guarantee.
- **Assay decides the scale, not preference.** `cstar` for TRB is 0.4080 in the amplicon reference
  and 0.1072 in the RNA-seq ones. A `cstar` above what a sample attains forces extrapolation,
  measured to inflate diversity roughly tenfold. Bulk RNA-seq through the default returns `nan`
  for every coverage-standardised diversity column on IGH/IGK/IGL/TRG/TRD.
- **Weighted is the default and it is not close.** One vote per *study* passes 0.843–0.857 of
  columns held-out against 0.533–0.551 per *sample*. Take an unweighted arm only when your cohort
  **is** one large study.
- **Blood is not a substitute for tissue**: median scale ratio 0.721, max location difference 17.2
  robust deviations over the 1,359 columns both establish.
- **Rescaling cannot fix batch.** Fitted on one cohort and applied to another the spread transfers
  (robust sd of z = 1.008) and the offset does not (1.34 robust deviations). `fit_scale(...,
  group=...)` hands you `batch_ratio` per column; the geometry blocks are cleanest, `depth` and
  `pair` worst.
- **A hole stays `nan`** — never centred, never zero-filled. `fit_scale` refuses a column the
  corpus barely saw (`min_n_obs`, `min_n_groups`) rather than shipping a confident number from
  nine samples, and statistics come from observed entries only: imputing first deflates the scale
  in proportion to sparsity.
- **A named model with no installed artifact raises**; an unknown name raises rather than being
  read as a path. Neither ever falls back to a reference fitted on another assay.

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

`mir.ml` re-exports only the tokenizers; everything above imports from its own submodule --
`from mir.ml.train import pick_device, train_forward_encoder`,
`from mir.ml.diffusion import DiffusionModel, train_diffusion`,
`from mir.ml.bundle import CodecBundle`.

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
