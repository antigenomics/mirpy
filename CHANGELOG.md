# Changelog

All notable changes to `mirpy-lib` (import `mir`). This project follows semantic versioning; the v3 line is a
greenfield ML/embedding rewrite (the classical v1.x/v2 toolkit is frozen on branch `legacy-v2`).

## Unreleased

### Fixed

- **A sparsely-observed chain no longer gains weight from its own sparsity.**
  `mir.explain.ChannelBuilder.build` and `mir.cohort.build_donor_cohort` imputed holes *before*
  computing `mean`/`std`, so a column's `sd` was deflated in proportion to how much of it was
  imputed — the filled entries are all one constant and contribute no spread. A chain observed in
  30% of donors therefore had its real values scaled up ~1.8x against a fully-covered one, giving
  the **least**-observed locus the **most** weight in every downstream distance, PCA and penalised
  fit. Both now standardise on observed entries only, and impute at the value the column is centred
  on, so a hole lands at exactly 0 (no information) rather than at a shared offset that every donor
  missing that chain carries. `DonorCohort.transform` reuses the fit cohort's fill.
  Measured before the fix: a 7-locus union embedding clustered by *which loci a sample
  had* at AMI 0.741, with read-depth eta-squared 0.42 across the whole cohort against 0.01 inside
  the fully-observed subset.
  **Behaviour change**: a column containing holes no longer has total variance 1 — its *observed*
  entries do. `test_builder_standardize_and_impute_invariants` was updated accordingly; that
  assertion encoded the defect.

### Added

- **Sub-probability repertoire embeddings — the deficient measure.** `Φ(S)` normalises its weights to
  sum to 1, so it is the kernel mean of a *probability* measure and every sample asserts one full unit
  of confidence. Measured, that premise fails at RNA-seq depth: the
  median tissue TRB sample holds **21 unique clonotypes** (blood TRB 254, 1st percentile 1), so
  `w_σ = a_σ/Σa` is `1/n` for a technical draw size rather than a clonal frequency (true frequencies
  are 1e-5…1e-8, and one singleton's weight spans **21,454×** across blood TRB purely from sample
  size). Normalising also *forces* a 5-clonotype tumour to assert full confidence, so it lands
  arbitrarily on the unit sphere — and the usual response, a minimum-clonotype floor, deletes the
  **immune desert** phenotype in tumour (a floor once cut 7,179 labelled donors to 2,129). Four new
  pieces, composable and off by default:
  - **`mir.repertoire.missing_mass(counts, method)`** — the mass `M₀` of the never-drawn clonotypes:
    `"turing"` (Good–Turing, `f₁/N`) or `"chao"` (`S_u/(N+S_u)`, `S_u = f₁(f₁−1)/(2(f₂+1))`). The
    **bias-corrected** Chao1, never the classical `f₁²/2f₂` — that form is undefined when no clonotype
    was seen exactly twice, which is common at these depths. Measured means: blood TRB 0.552 (Turing)
    vs 0.649 (Chao), tissue IGH 0.189 vs 0.458 (ranks agree, `r` = 0.93 / 0.76).
  - **`SampleEmbedding.mass`** + **`sample_embedding(missing_mass=…)`** — the retained mass `1 − M₀`
    rides along on the embedding. `"none"` is the default and the blocks are untouched by the setting,
    so existing output is **bit-identical**. Deliberately *not* a negative measure: `‖Φ_P − Φ_Q‖`
    being the MMD and a convex combination of two `Φ`'s being the `Φ` of a real pooled repertoire are
    what make `mir.twin` and trajectory interpolation meaningful, and a signed measure loses both. A
    sub-probability measure costs neither.
  - **`mir.repertoire.naive_reference(space)`** — the kernel mean of `n` naive V(D)J recombinations
    (`vdjtools.model.generate`, ~8 s for 20,000), cached per `(n, seed)`, or injectable via
    `sequences=`. This is the load-bearing choice for *where the unseen lives*: shrinking toward the
    corpus centroid is James–Stein toward the mean and it measurably **hurt** (shallow samples pile
    into a dense, itself-depth-correlated ball), while the germline draw took `R²(PC1, depth)` from
    0.259 → **0.001** (blood TRB) and 0.067 → **0.006** (tissue IGH), kNN label entropy unchanged or
    better, PC1's explained variance unchanged (0.309 → 0.334, so a different direction, not a
    collapsed one) and the whole leading block 0.253 → **0.047**.
  - **`mir.repertoire.contrast_embedding(emb, reference)`** — `Ψ_S = mass·(Φ_S − naive)`, where
    legitimate negativity lives: a signed *difference of two probability measures*, negative where the
    sample is depleted relative to unselected recombination, still an RKHS element with
    `‖Ψ_S‖ = MMD(S, naive)`. Magnitude = **confidence × deviation-from-naive**, so an immune desert
    (`M₀ → 1`) lands at the **origin** — the correct place for "no infiltrate detected" — and a shallow
    blood sample says so by its norm instead of being filtered out. No minimum-clonotype floor was
    added to the library, and none should be: it is a blood rule, not a tissue rule.
- **`mir.repertoire.rao_q`** — Rao's quadratic entropy of a repertoire, read straight off the kernel
  mean as `1 − ‖Φ₁‖²`. With the kernel dissimilarity `d = 1 − k` and weights summing to 1, Rao's
  `Q = Σ w_σ w_τ (1 − k)` collapses to exactly that, so the **norm** of Φ₁ is a diversity statistic and
  no Gram matrix is ever formed (verified against an explicit Gram to ~1e-16). It is the diversity the
  Hill block structurally cannot express: every Hill number is a functional of the clone-size
  distribution alone, hence invariant to permuting which receptor carries which abundance, whereas
  Rao's Q weights each pair by receptor dissimilarity — a **functional** diversity, sequence-aware and
  already carried inside Φ. Measured: this one scalar recovers R² 0.74–0.85 of classical diversity,
  while embedding derivatives reach R² 0.974–0.994 for Shannon and 0.985–0.9999 for richness. Valid
  only on the *uncentred* Φ₁ (centring keeps differences, hence MMD, but not norms), and only to the
  RFF error in `k(z,z)=1` — a single-clone sample reads Q ~1e-2, not 0.
- **`mir.repertoire.depth_threshold`** → `κ`, the sample size below which a repertoire's Φ is mostly
  sampling noise. The damage depth does to a kernel mean is not bias but **variance ∝ 1/n** over an `n`
  spanning four orders of magnitude, and in a neighbour graph that heteroscedasticity *is* a depth
  axis. Regressing `‖Φ_S − Φ̄‖²` on `1/n` splits the observed spread into between-sample signal `τ²`
  (intercept) and within-sample sampling noise `σ²` (slope), so **`κ = σ²/τ²`** is where the two are
  equal. Measured κ ≈ 40–70 clonotypes across four independent views, 23–69% of samples below it. The
  estimable replacement for a hand-picked clonotype floor — the library still applies none.
- **`mir.repertoire.sample_statistics` / `cohort_statistics`** — the sampling fingerprint (`f1`, `f2`,
  `f3plus`, singleton fraction, top-clone fraction, Shannon, library size, Turing/Chao missing mass).
  Both the `stats=` input `recovery_report` asks for and candidate biology in their own right:
  abundance classes and top-clone fraction are clonality and expansion, not only depth nuisance.
- **`mir.repertoire.band_frames` / `band_embeddings` / `mixture_weights`** — compartment decomposition.
  Φ₁ is a clone-size-weighted *average*, right for a population mean and wrong for a **minority**
  signal: with `ρ_S = (1−π)ρ_N + π ρ_E`, an effect confined to the expanded compartment reaches Φ₁
  attenuated to `πΔ` while the naive compartment supplies most of the noise. Bands are `singleton`
  (count 1), `expanded` (≥2), `top` (top 1% clipped to [10, 500]) and, for IGH, the isotype cut
  `igm`/`igg`/`iga` from `c_call` — an irreversible molecular event rather than an abundance threshold,
  with null `c_call` rows *excluded* rather than defaulted to IgM (~43% of IGH reads carry no call).
  Every band embeds through the **same frozen space** (refitting would put each band in its own
  geometry and make band-to-band distances meaningless), and a band under `min_clonotypes` is recorded
  **absent** (`None`) rather than embedded — the same hole convention `ChannelBuilder`/`align_loci`
  already understand. `mixture_weights` then recovers each compartment's share by non-negative least
  squares, which is well-posed rather than heuristic because mixture linearity `Φ(S) = Σ π_c Φ(c)` is
  exact in exact arithmetic (the realised residual is set by the float32 clonotype embedding, ~1e-5
  relative). Measured on IGH isotypes: class-switched **IgG carries a median π of 0.070**
  of Φ₁(IGH) (unswitched IgM 0.230, IgA 0.176, 0.520 unaccounted — the uncalled share, in ballpark
  agreement with the ~0.43 counted from reads; different denominators). That number doubles as a power
  calculation: a subset carrying π ≈ 0.001 cannot be detected by any aggregate distance on Φ, so the
  per-clonotype witness is the sensitive route. On survival endpoints the bands did **not** beat a
  diversity reference (0 of 22 pre-registered block × endpoint cells; the isotype cut failed
  identically), but the decomposition confirmed the mixture argument — `singleton` reproduced a
  clinical-covariates-only score (0.600 vs 0.599) while `expanded` alone reproduced the whole-repertoire
  score (0.634 vs 0.634) — and banding won on kNN entropy in tissue IGH (0.3875 vs 0.4686).
- **`mir.repertoire.rarefy_embedding`** — depth-standardised Φ, averaged over multinomial replicate
  draws, plus `v_rep`. Because Φ is linear in the clone-weight measure, the mean over independent
  subsamples is itself a kernel mean — of the mixture distribution over subsamples — so MMD, Rao's Q and
  mixture linearity all survive (~1e-15). It is the **only** depth correction that does: an orthogonal
  projection breaks the norm identity, a per-coordinate location-scale rescale breaks both. The
  replicate dispersion is not a diagnostic bolted on but an exact identity,
  `Rao(Φ̄) = mean_r Rao(Φ_r) + v_rep` — Φ̄ embeds the mixture over subsamples, which is genuinely more
  diverse than any single subsample, so the excess diversity of the average *is* the replicate
  variance, and `v_rep` is a free per-sample estimate of the noise `κ` measures cohort-wide. Not a
  default: rarefying a cohort to its shallowest useful depth discards the deep samples' advantage.
- **`mir.cohort.depth_report`** — R² of the leading PCs against the sampling fingerprint, with an
  optional trusted-subset arm. The companion to `missingness_report`: that asks whether a grouping
  tracks *which* blocks a sample has, this asks whether the dominant axes track *how deeply* it was
  sequenced. Read it beside the returned `explained_variance` — with the deficient measure,
  R²(PC1, depth) fell 0.259 → 0.001 and best-of-PC1–5 0.253 → 0.047 while PC1's explained variance was
  unchanged, which is what distinguishes a different direction from a collapsed one.
- **`mir.cohort.residualize(..., shrink=True)`** — positive-part James–Stein shrinkage of each batch
  offset, `c_b = max(0, 1 − (d−2)(σ̂²/n_b)/‖µ̂_b‖²)`, so a batch whose apparent offset is no larger than
  its own estimation error is left alone. The plain correction the library already shipped can make the
  batch **easier** to read, measured out-of-sample (leave-one-batch-out plus a donor-level split inside
  each batch): batch-identity AUC 0.863 raw → **0.985** after per-group centring, 0.978 after ComBat.
  The mechanism is estimation error, not a bug — with a mean fitted from 8–15 samples in ~1,280
  dimensions, `‖µ̂ − µ‖ ≈ √(σ²d/n)` ≈ 16 against a true offset of norm 7–24, so subtracting it injects a
  batch-constant vector as large as the one it removes, and in-sample this vanishes by construction.
  Shrinkage recovered most of the damage: 0.985 → **0.889**. Default `False`, so existing behaviour is
  byte-identical.
- **Docs: `Mathematical foundations`** (`docs/math.rst`) — every object the library computes with its
  definition, derivation and the call that produces it: the prototype embedding and its Lipschitz
  bound, the measure quotient that removes order *and* length, Bochner/RFF and the empirical
  characteristic function, MMD biased and unbiased, Hill numbers versus Rao's Q, the Good–Turing/Chao
  derivation of the missing mass, the mixture algebra behind bands and rarefaction, the balloon
  density ratio and its two nulls, channel ablation and the MMD witness, the batch-offset estimation
  problem, the trajectory model and the Gaussian conditional used for in-silico evolution — plus a
  **"which transformation preserves what"** table (MMD / Rao / mixture linearity) that is the contract
  for anything applied to Φ. MathJax for the formulas and `sphinx.ext.graphviz` for the schematics, so
  the docs build needs no LaTeX (the CI job installs `graphviz`).

- **`mir.explain.ChannelBuilder.add(..., preserve_magnitude=True)`** — scale a channel by **one global
  scalar** (pooled RMS over observed entries, no centring) and fill its holes with `0`, instead of
  per-column z-scoring. Mandatory for any block whose magnitude carries information (a contrast /
  sub-probability block): per-column standardisation forces every coordinate to unit variance across
  samples, so a matrix where half the rows sit at the origin comes out looking exactly like one where
  none do — it deletes the deficiency it was built to preserve. This already invalidated one
  experiment, whose deficient arm scored 0.6481 against a 0.6179 baseline purely from the rescaling.
  `stack_embeddings` now warns when handed embeddings with `mass < 1`, since `Φ.vector` does not carry
  the mass.
- **`mir.bench.recovery_report(X, stats, groups)`** — grouped-CV ridge R² from an embedding's PCs back
  to each basic repertoire statistic (richness, Shannon, top-clone fraction, singleton fraction, Chao
  unseen fraction, library size). The honest scoring rule is **recoverability, not competition**: is
  the statistic carried *inside* the embedding, so nothing has to be bolted on beside it? Renormalising
  to mass 1 deletes the magnitude, so coverage/richness are unrecoverable from `Φ` by construction and
  the deficient measure should win this question as a design consequence. Sits beside
  `mir.cohort.missingness_report` as the other "is this object honest" check.
- **`mir.cohort.align_loci`** / **`mir.cohort.LocusAlignment`** — align per-locus embedding matrices
  keyed by sample id onto one sample axis, holes where a locus is absent. The step between "one
  matrix per locus, each over its own samples" and "one matrix over one sample set", which the
  library previously left to every caller. `how="union"` is the default because an **inner join
  across seven loci is bound by the thinnest two**: measured on a 62,293-sample tissue cohort,
  intersecting all seven left 19,346 samples (31%) while the union keeps every one — complete-case
  deletion wearing a join's clothes. Absent blocks become `nan`, which is the hole convention
  `ChannelBuilder` already understands, so `LocusAlignment.build()` composes straight through the
  observed-entry standardization fixed above. `how="inner"` is kept so the difference stays
  measurable, not as a default.
  Explicitly **do not zero-fill before aligning**: a literal `0` is a value, and after a naive
  global z-score it becomes `-mean/std`, a large shared constant that every sample missing that
  locus carries. `nan` says "not observed"; `0` says "observed to be zero".
- **`mir.cohort.missingness_report(labels, mask)`** and **`DonorCohort.missingness_report(labels)`**
  — adjusted mutual information between a grouping and the block-presence pattern, plus the
  correlation ratio of the per-sample present-block count. The check for whether a clustering,
  enrichment or trajectory built on holed data is content or is a coverage stratification; near zero
  is what you want, high means re-read the result inside a fully-observed subset. `DonorCohort` now
  records the per-block presence matrix at fit so this needs no bespoke bookkeeping.

## 3.7.0 — 2026-07-30

Four new modules: a PhenoPath-style exposure trajectory, the generative loop's mechanical and
research halves, and the digital twin that glues them to the digital donor — plus a repertoire-level
exposure channel. No public API removed; a minor bump.

### Added

- **`mir.track.fit_exposure_trajectory`** — a covariate-disentangled latent trajectory over any
  per-sample channel matrix (PhenoPath, Campbell & Yau 2018, *Nat. Commun.* 9:2442,
  doi:10.1038/s41467-018-04696-6, adapted from genes×cells to repertoire-channels×samples): infers a
  shared exposure/progression pseudotime `tau` while separating out which channels respond to it
  differently by a known covariate (`TrajectoryFit.top_interactions`). A simplified closed-form
  alternating fit (per-channel ridge + GLS trajectory update + iteratively-reweighted ARD shrinkage
  on the interaction term), not a literal reimplementation of PhenoPath's CAVI engine. Torch-free.
- **`mir.generate`** — the generative loop's mechanical half: `DescriptorDensity` (optionally
  class-conditional Gaussian, Ledoit-Wolf shrinkage) over `RepertoireDescriptor` vectors;
  `sample`/`evolve` (perturb one coordinate, propagate the coupled shift via the fitted covariance's
  conditional mean) promote the ad-hoc `benchmark_repertoire_tcga_insilico.py` `np.cov`-slope pattern
  into a reusable library object. Torch-free.
- **`mir.ml.diffusion`** (needs `[ml]`) — the generative loop's research half: a compact conditional
  DDPM/DDIM generator (classifier-free guidance) over a compact descriptor/code space, sharing
  `DescriptorDensity`'s `sample(n, condition=, seed=)` call shape so either generator drops in
  unchanged. `DiffusionModel.save`/`load` mirrors `CodecBundle`'s shape.
- **`mir.twin.DonorTwin`/`make_twins`** — the digital twin: glue one donor's `RepertoireDescriptor` +
  an optional `mir.track` trajectory position + covariate into one object; `.perturb()` and
  `.simulate()` accept either generator.
- **`mir.density.exposure_score`/`exposure_channel`** — aggregate a per-clonotype
  `neighbor_enrichment` result into repertoire-level exposure scalars (breadth, abundance-weighted
  mass fraction, mean log2 fold), ready for `ChannelBuilder`/`DonorCohort extra_channels` — exposure
  detection promoted from clone-level to a first-class cohort channel.

## 3.6.0 — 2026-07-30

Default-on functional filtering, a new default clone-size weight for repertoire embedding, and a
clearer crash message — a minor bump since the repertoire-embedding default changes the numeric
output for callers who didn't pass `weight=` explicitly.

### Added

- **`mir embed clonotypes` / `mir embed repertoires` drop non-coding clonotypes by default**
  (`--filter-functional`, default on; `--no-filter-functional` to disable) via
  `vdjtools.preprocess.filter_functional` — a stop codon or legacy out-of-frame marker
  (`[*atgc#~_?]`) in `junction_aa` otherwise either crashes (`_`) or silently produces a
  numerically meaningless embedding (`*`). `mir embed repertoires` skips (with a warning) any
  sample/locus left empty after filtering, instead of crashing the whole batch.
- **`"duplicate_count"` (linear, `g(a)=a`) and `"log2p1"` (`g(a)=log2(1+a)`) clone-size weights**
  for repertoire embedding, alongside the existing `"distinct"` (`g≡1`) and `"log1p"`/`"anscombe"`.
  `"log2p1"` is the **new default** — see Changed.

### Fixed

- **`TCREmp.embed` raises a clear error when `junction_aa` contains `'_'`** (the legacy vdjtools
  out-of-frame marker), naming the value count and pointing at `filter_functional`, instead of
  letting seqtree's opaque `"symbol '_' is not in the alphabet"` propagate uncaught. `'*'` (stop
  codon) is unaffected by this check — it doesn't crash, so filter it via `filter_functional` if
  you don't want it silently embedded.

### Changed

- **Default clone-size weight for repertoire embedding is now `"log2p1"`** (`g=log2(1+a)`),
  changed from `"log1p"` (natural log). Affects `mir.repertoire.sample_embedding`,
  `RepertoireSpace.sample_cloud`, `sample_descriptor`, `class_witness`, `mir.explain.channel_drivers`,
  and `mir embed repertoires --weight`. This changes the numeric values of `Φ(S)`'s mean/second
  blocks (and any MMD computed from them) for callers who relied on the implicit default — pass
  `weight="log1p"` to reproduce the old default exactly.

## 3.5.0 — 2026-07-28

A documentation-accuracy pass plus the fixes found by the accompanying code audit. No public API
removed; two new keyword arguments, one runtime dependency dropped, and three paths that used to
fail silently now fail loudly — hence a minor bump rather than a patch.

### Added

- **Prototype replicates — `load_prototypes(..., replicate=r)`, `n_replicates()`, and
  `replicate=` on `TCREmp`/`PairedTCREmp.from_defaults` and both `mir embed` commands.** Each
  bundled chain ships 10 000 real receptors whose row order is already a uniform shuffle, so a
  disjoint block of `n` rows is an independent draw from the same pool: `replicate=r` returns block
  `r`, giving `10000 // n` replicates (**10** at the common `n=1000`) with no new data and no RNG.
  This answers "is my result an artefact of *which* prototypes I drew?" — a question a nested
  `n_prototypes` sweep cannot answer, since those draws are prefixes of one another.

  `replicate=0` is unchanged and remains *the* prototype set behind every preset, bundled codec and
  published number. `prototype_hash` now covers the replicate index, so `CodecBundle`,
  `RepertoireSpace`, `DonorCohort` and `SetEncoderBundle` refuse to mix draws (artifacts written
  before this release read as draw 0). README and the user guide state the default, the provenance
  (real repertoires, `seed=42`), and that cross-replicate embeddings are incomparable.
- **`generate_background(..., species=, source="arda")`** — the P_gen background can now be drawn
  in the **arda IMGT allele namespace** (the same frame as the prototypes and baked germline
  distances, so generated V/J calls resolve exactly instead of taking the allele cascade) and for
  **mouse**. Both need a `vdjtools` shipping the bundled `arda` model set: those 9 models were
  already in the wheel but unreachable, fixed upstream in `vdjtools.model.load_bundled`. The human
  `learned`/`olga` path is unchanged and still works on older vdjtools; asking for arda/mouse
  without it raises a message naming the requirement.
- **`neighbor_enrichment(..., k_max=, seed=)`** — forwarded to the `backend="ann"` engine, which
  already warned "raise `k_max`" for a saturated neighbour ball but gave no way to do it.

### Fixed

- **Germline distances silently maxed out for 42 human and 71 mouse alleles**, including
  **`IGHV3-23*01` and `IGHV1-69*01` — the two most-used human IGHV genes.** arda emits an
  *ambiguity group* (`"IGHV1-69*01,IGHV1-69D*01"`) as one row when the alleles share an identical
  germline region, and the allele index registered only the joined string, so a query naming a
  member matched nothing and took the max-distance fallback — quietly, since a fallback is also the
  legitimate answer for a genuinely unknown gene. Members are now indexed to their group's row (a
  standalone row still wins). Across every bundled locus, 333 of 457 group-member names resolved to
  the fallback before; now none do.
- **`mir embed repertoires --blocks diversity` crashed** with `ValueError: zero-dimensional arrays
  cannot be concatenated`: `SampleEmbedding.vector` assumed the kernel-mean block was always
  present. A mean-less Φ is now a valid vector, and `mmd_distance` / `mmd_matrix` say *why* MMD
  needs the mean block instead of raising a `NoneType` `TypeError`.

### Changed

- **`GermlineDistances.matrix` resolves each *distinct* allele once** instead of once per row, then
  gathers a small `(n_distinct, K)` table and expands it. A repertoire has ~10⁵–10⁶ rows but ~10²
  distinct alleles, so this is the dominant cost of `TCREmp.embed`: measured **2.32 s → 1.04 s** for
  200k clonotypes × 1000 prototypes (the germline block itself 0.765 s → 0.072 s), bit-identical
  output, nulls still take the allele fallback.
- **`RandomFourierFeatures.transform` computes in place** after the matmul — peak memory for a deep
  sample drops ~3× (measured 9.2 → 3.2 GiB at n=200k, D=2048), output identical.
- **`RepertoireSpace.save` / `DonorCohort.save` refuse a model built with a custom `matrix=` or
  `alignment=`.** Neither knob is recorded in `meta`, so `load` rebuilt the default
  gapblock/BLOSUM62 space — a *different* coordinate system that still passed the prototype-hash
  check. Failing at save time keeps the comparability invariant honest.
- **`TCREmp.embed` rejects null `junction_aa`** with a message naming the column and the row count,
  instead of an opaque `TypeError: object of type 'NoneType' has no len()` from inside seqtree.
- **`fit_donor_embeddings` warns when a locus has too few donors to fit the identity PCA** — that
  block was silently emitted as all-NaN, imputed to a constant, information-free channel.
- **`alignment="sw"` raises a directed `ImportError`** ("pip install biopython") instead of a bare
  `ModuleNotFoundError`; `python -m mir.embedding.tcremp` now skips the SW leg when BioPython is
  absent, so the self-check passes in the default `[dev,bench]` environment.
- **Determinism**: `bench.theory._mutate` no longer "mutates" a residue to itself (5.3% of `k=1`
  substitutions were no-ops, and it disagreed with `density._mutate1`); `load_vdjdb` and
  `generate_prototypes` no longer depend on `.unique()`'s arbitrary row order.
- **Single-sourced the version**: `pyproject.toml` takes it from `src/mir/__init__.py`
  (`[tool.hatch.version]`), `docs/conf.py` imports `mir.__version__`, and `publish.yml` validates
  the release tag against the built wheel — one copy instead of three.
- **Corrected the documented gap-block ↔ Smith-Waterman relationship.** `mir.distances.junction`
  claimed the two "agree for equal-length CDR3s and rank-correlate ≥0.99 on gapped pairs". Measured
  over all 44 850 pairs of 300 bundled human-TRB prototypes: close pairs do agree exactly, but SW is
  a *local* alignment, so distant pairs diverge (equal-length outliers 181 vs 77) and the overall
  rank correlation is ρ≈0.78. The docstring now says that, and that the two are different coordinate
  systems rather than interchangeable scales.
- **Tests**: the survival-scorer test is no longer marked `integration` (lifelines ships in
  `[bench]`, which CI installs), lifting `mir.bench.eval` from 23% to 64% coverage in CI;
  `test_set_encoder.py` skips instead of failing on a torch-free install, so the documented
  `pytest tests/` is green again; the VDJdb loader test no longer depends on a gitignored fixture it
  could never find in CI (`mir.bench.vdjdb` 44% → 100%); and `TCREmp`'s public coordinate knobs
  (`metric="sqrt"`, custom `matrix=`, `alignment="sw"`, null rejection) have real tests instead of
  assertions buried in a `__main__` block. Fast-tier coverage 64% → 66%.
- **CI gained an `optional-tiers` job** (`[dev,bench,ml]` + BioPython, `-m "not benchmark"`): all of
  `mir.ml` — ~20% of the library, including both prototype-hash comparability guards — previously
  ran in no CI configuration, because the only job installs `[dev,bench]` and every torch test skips.
- Docs corrections: the density `backend=` default is `"kdtree"` (all-core exact), not `"exact"`;
  the `cv_cindex` snippets in the user guide and in `mir.explain` now match the real
  `(durations, events, *, base, block)` signature (the old form scored every channel `NaN`);
  `CodecBundle.load` is documented as *not* verifying (its `forward_encoder` does); `encoder.code`
  vs `encoder.encode`; the `mir embed repertoires` flag list is complete; `SOURCES.md` records the
  prototypes as `arda-real` (experimental) with `src/`-prefixed regenerate commands; the examples'
  run lines point at `examples/`, not the old `notebooks/`.

### Removed

- **`tqdm`** from the runtime dependencies — nothing in the package imports it.
- **`nbsphinx` / `nbsphinx-link` / `ipython`** from the `[docs]` extra, and the unreferenced
  `docs/requirements.txt`: the docs have contained no notebooks since the examples became marimo
  scripts. The Sphinx build stays zero-warning under `-W`.

## 3.4.0 — 2026-07-18

Minor: a command-line interface, a uv-based dev setup, and a documentation overhaul. No public
Python API removed; one optional-dependency group split out.

### Added

- **`mir` command-line interface** (`[project.scripts]`, also `python -m mir.cli`) — the two
  embedding scales without writing Python:
  - `mir embed clonotypes SAMPLE` → a per-clonotype TCREMP embedding table (`e0…`).
  - `mir embed repertoires SAMPLE…` → one repertoire vector `Φ(S)` per sample **per chain** on one
    shared basis (`phi0…`), with optional `--mmd` pairwise-distance output.

  Inputs are any format `vdjtools.io` reads (AIRR/vdjtools/MiXCR/immunoSEQ/parquet); output is TSV
  or Parquet. See `mir embed <cmd> -h`. Tests in `tests/test_cli.py`.
- **`mir.repertoire.correct_batch`** — Harmony-like cluster-aware batch correction on a stacked
  sample×feature Φ matrix. Removes the batch offset *per soft cluster* (batch-diversity-penalised),
  so a batch confounded with a biological cluster is corrected without erasing that biology; reduces
  exactly to `mir.cohort.residualize` at `n_clusters=1` / `theta=0` (`prop:batch`).
- **`[ann]` optional-dependency group** for the approximate-NN density backend (`pynndescent`).

### Changed

- **Development now uses a repo-local `.venv` via [uv](https://docs.astral.sh/uv/)** instead of
  conda. `setup.sh` is rewritten (bash/zsh portable; `--dev-parents`, `--docs`, `--tests`); the
  conda `environment.yml` is removed. Runtime is unchanged — still a pure-Python `py3-none-any` wheel.
- **`pynndescent` moved from `[bench]` to the new `[ann]` extra.** `[bench]` is now all pure-wheel
  (no numba/llvmlite), so `pip install "mirpy-lib[bench]"` resolves cleanly on any Python. Users of
  `density.neighbor_enrichment(backend="ann")` should install `"mirpy-lib[ann]"`.
- **`vdjtools>=3.0.0`** (was `>=2.3.0`).
- Documentation overhaul: a use-case-driven user guide, the two CLI commands documented, an
  examples/notebooks page, `mir.cohort` and `mir.bench.eval` added to the API reference, a logo, and
  the sample-embedding schematic + real depth-robustness figure. Zero-warning Sphinx build.
- **Repo layout** (no effect on the installed package): adopted the **src-layout** (`mir/` →
  `src/mir/`); renamed `notebooks/` → `examples/`; and moved the working result/plan markdown out of
  the repo root — `THEORY.md` to the manuscript repo, `BENCHMARKS.md` / `REPERTOIRE_{EMBEDDING,LESSONS}.md`
  / `SQRT_D_MIGRATION.md` / `ROADMAP.md` to `2026-mirpy-analysis/benchmarks/`. Root keeps
  README / CHANGELOG / CLAUDE / SOURCES.

### Fixed

- `DEFAULT_GAP_POSITIONS = (3, 4, -4, -3)` had three independent definitions
  (`distances/junction`, `embedding/tcremp`, `ml/bundle`); now defined once in `distances.junction`
  and imported, so the coordinate constant cannot drift.
- `cohort.cluster_samples` docstring described itself; `AntigenMetric` and `mir.bench.eval` gained
  the docstrings/module-map entries they were missing.

## 3.3.0 — 2026-07-17

Minor: one new public parameter, nothing removed or changed.

### Added

- **`fit_density_space(chunk_size=)`** — embed and project in batches so the full raw matrix is never
  resident. Peak memory becomes `max(pca_fit_cap, chunk_size) × n_features` instead of
  `len(df) × n_features`: measured **10.60 GB → 1.81 GB** at 450k pooled clonotypes, and flat in `N`
  (vs linear), at no wall-clock cost. This is what makes whole-cohort density arms runnable on a
  laptop — the 4.2M-clonotype pooled arm is ~51 GB raw and ~102 GB once `scaler.transform` upcasts to
  float64. Chunking is bit-exact at the embedding level (`_embed` of a slice == the slice of
  `_embed`); the projected coordinates agree to float noise (~1e-7 relative), since BLAS summation
  order depends on batch shape.

### Fixed

- `fit_density_space`'s `pca_fit_cap` docstring claimed it "lets whole repertoires be embedded without
  a full-matrix PCA". It caps the **fit**, not the memory — both raw matrices were already
  materialized before the PCA was fitted. Documented, and `chunk_size=` is the actual remedy.
- **`mir.__version__` was stale** — it read `3.1.1` on the published 3.2.0, because the release bump
  moved `pyproject.toml` but not `mir/__init__.py`, and `publish.yml` only validates *pyproject ==
  tag*. Both now read 3.3.0. (`__version__` is still hand-maintained; deriving it from
  `importlib.metadata` would retire this failure mode for good.)
- `tests/assets/olga_humanTRB_1000.txt.gz` was a slice of the alphabetically sorted VDJdb TRB dump,
  not OLGA output as its name and `SOURCES.md` claimed — so it was **not** an antigen-naive null (12%
  of rows had a Hamming-1 neighbour vs 0.2% for real OLGA). Regenerated from `olga-generate_sequences`;
  provenance and a byte-reproducible regenerate command recorded in `SOURCES.md`. No test was
  invalidated (they use it only as a generic TRB junction pool), but external calibrations that treated
  it as a synthetic negative control were comparing VDJdb against itself. Not shipped (tests are
  excluded from the sdist); listed here because it invalidates results, not code.

## 3.2.0 — 2026-07-17

Minor: one new public module, nothing removed or changed.

### Added

- **`mir.explain`** (T7) — explainable readouts over any repertoire feature matrix.
  `ChannelSpec` / `ChannelBuilder` / `stack_embeddings` attach the name→column map that `Φ.vector`
  does not carry (`stack_embeddings` is exact: `X[i] == embs[i].vector`, names only, no transform);
  `channel_report` ablates each named channel under a caller-supplied scorer (leave-one-in by
  default; `mode="both"` adds the conditional half that exposes *redundant* channels — high `delta`,
  `delta_out≈0`; optional row-permutation p-values); `channel_drivers` hops from a winning
  **kernel-mean** channel to the clonotypes driving it via `class_witness`, and refuses channels with
  no clonotype pre-image (a Hill number's "drivers" are a category error, not an open question).
  Scorer-agnostic by design — the library never sees the labels and ships no scorers, so a Cox
  C-index and a CV AUC both plug in. **No existing module changed.**

## 3.1.1 — 2026-07-14

Maintenance re-release of 3.1.0 with no functional changes — 3.1.0 was withdrawn from PyPI, and
this version restores the package under a fresh, clean version. The API, coordinates, and behaviour
are identical to 3.1.0 (see below).

## 3.1.0 — 2026-07-14

The Part-2 feature tier on top of the 3.0 consolidated embedding core: neural codecs, continuous-density
methods, and the sample-level (repertoire) embedding, all on arda-native coordinates.

### Added
- **`mir.density`** (T6) — graph-free continuous-density TCRNET/ALICE: balloon adaptive-radius enrichment
  (Poisson/binomial + BH q, water-level calibration), abundance-aware weighted mass, backends
  `exact`(BallTree) / `kdtree`(multicore) / `ann`(pynndescent).
- **`mir.repertoire`** (T7) — sample-level embedding `Φ(S)` = RFF kernel mean ‖ coverage-standardized Hill ‖
  second-moment Fisher; `mmd_distance`/`mmd_matrix` (now with **`unbiased=True`** diagonal-removed MMD²),
  `hla_stratified_mmd`, `class_witness` motif finder.
- **`mir.ml`** (Part 2, `[ml]` extra) — forward/inverse/pgen/unified neural codecs + `CodecBundle`
  (prototype-hash-verified shipping); learned repertoire track `set_encoder` (Set-Transformer/DeepRC).
- **Bench** — clustering `method=` (dbscan/hdbscan/optics); `bench.theory` T6 `tcrnet_convergence`,
  `codec_losslessness`.
- **Repertoire benchmarks** — `experiments/benchmark_repertoire_{aging,depth,cmvhla,hla,yfv,spikein,
  agediverge}.py` and the COVID cohort suite `{covidbatch,covidhla,covidstatus,covidpaired}.py`
  (`airr_covid19`, local-first + HF fallback). Recorded baselines in `BENCHMARKS.md`.

### Changed
- **Coordinate system re-pinned to arda-native** germline + real-repertoire prototypes (versioned; any model
  trained on the old coords must be retrained).
- **`mir.ml` device selection** now **CUDA → MPS → CPU** (was MPS-only) with a `MIR_DEVICE` env override and
  CUDA seeding — GPU support beyond Apple silicon.
- Documented all parallelism knobs (README "Performance & parallelism"): `TCREmp(threads=0)` all-core,
  density `backend="kdtree"` multicore-exact, `cluster(n_jobs=…)`, BLAS env.

### Fixed
- **Unbiased repertoire MMD** — the biased V-statistic's `1/n_eff` self-term inflated low-diversity samples and
  faked a divergence signal; `unbiased=True` removes it. Aging-divergence re-evaluated at depth: real but
  diversity-coupled, not an independent axis.

### Notes
- New lessons for the theory appendix in `REPERTOIRE_LESSONS.md`; full findings in `THEORY.md` T7.
- Pure-Python `py3-none-any` wheel; native code comes from `seqtree`/`vdjtools` wheels.

## 3.0.0 — greenfield v3 embedding core (unreleased)
- Prototype (TCREMP) embedding on `seqtree.gapblock` + baked arda germline distances; vdjtools reuse (no AIRR
  data-model layer of its own); bench harness (VDJdb Table S1); theory scaffold (`THEORY.md` S1–S3, T1–T5).
