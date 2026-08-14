# Changelog

All notable changes to `mirpy-lib` (import `mir`). This project follows semantic versioning; the v3 line is a
greenfield ML/embedding rewrite (the classical v1.x/v2 toolkit is frozen on branch `legacy-v2`).

## 3.11.0 — 2026-08-14

### Changed — `fit_scale` gives each study one vote, not each sample

`fit_scale(frame, group=...)` used `group` only to record the diagnostic `batch_ratio`; the
location and scale themselves were an unweighted median/MAD over samples. Samples inside a study
share a protocol, a batch and a donor pool, so whichever studies happened to be large were setting
the reference for everyone.

Measured on 23,234 blood samples across 947 SRA study groups, six independent 70/30 **study**
splits, each fit scored against a fit on the held-out studies:

| weighting  | pass rate | sd over splits | scale swing | IQR log-ratio |
|------------|----------:|---------------:|------------:|--------------:|
| per sample |     0.630 |          0.064 |       0.063 | 0.092 ± 0.020 |
| per study  | **0.792** |          0.089 |   **0.044** | **0.079 ± 0.005** |

A column passes when its location lands within 0.10 held-out scales and its scale within ±10%.
"Scale swing" is the sd across splits of the median fitted/held-out scale ratio — the reference is
only identified up to a global multiplicative factor, and which studies it saw moves that factor.
One vote per study shrinks the swing by a third and makes the per-column scatter four times more
reproducible across splits.

`weight_by_group=False` reproduces the old estimator. Without `group=` nothing changes. The
artifact records `weighted_by_group` in its metadata, so a reference cannot be mistaken for the
other kind. **A reference refitted with this release will not equal one fitted before it** — hence
the minor bump.

### Fixed — `batch_ratio` was measured on the largest 40 studies, and counted their noise as batch

`_batch_ratio` skipped groups under `min_per_group=100`. On the 947-study blood reference that
admitted **40 studies** — the largest 40, which are also the most protocol-homogeneous — so the
diagnostic that says "this column separates cohorts, not donors" was measured on 4% of the corpus.

The floor was high because a group centre is itself an estimate: `var(centres)` carries the median's
sampling variance, so small groups inflate the ratio. Subtracting the mean sampling variance — the
usual moment correction — removes that, and a pure-noise draw of 100 groups × 30 samples reads
**0.063** corrected against **0.234** raw, where the truth is 0.

| `min_per_group` | corrected | studies | dominated | median ratio |
|---:|---|---:|---:|---:|
| 100 | no (old) | 40 | 268 | 0.611 |
| 100 | yes | 40 | 264 | 0.591 |
| 25 | no | 238 | 412 | 0.746 |
| **25** | **yes (new)** | **238** | **356** | **0.651** |

The new setting flags **102 columns the old one missed** and clears 15 it flagged wrongly; the two
rank columns at `corr(log ratio) = 0.44`, so this is a different answer, not a refinement. More
studies find *more* batch loading even after the correction, because the 40 largest understate how
much a column moves between cohorts.

`report()` now also carries `batch_groups` — how many groups actually voted. "266 dominated" means
one thing over 40 studies and another over 238, and the report used to give only the numerator.

### Changed — `measure_constants` takes `group=` too, because `cstar` and `pgen_q05` are quantiles

Both constants are quantiles over the corpus, so both had `fit_scale`'s problem, and `pgen_q05` had
it twice: it pools `n_pgen` junctions **per sample**, so a 500-sample study puts a million junctions
into a pool a 5-sample study contributes ten thousand to. `group={sample_id: label}` gives each
label one vote, spread over its samples and over their junctions.

Measured on the 23,234-sample / 947-group blood reference, the shift in `cstar`:

| locus | per sample | per study | Δ |
|---|---:|---:|---:|
| TRA | 0.1477 | 0.1290 | −0.019 |
| TRB | 0.1509 | 0.1256 | −0.025 |
| TRG | 0.2351 | 0.2216 | −0.014 |
| TRD | 0.1818 | 0.1954 | +0.014 |
| IGH | 0.1931 | 0.1706 | −0.023 |
| IGK | 0.3659 | 0.3197 | −0.046 |
| IGL | 0.3517 | 0.3117 | −0.040 |

11–17% relative on TRA/TRB/IGH/IGK/IGL, and downward on six of seven loci because the large studies
are the deep ones. Lower is the safer direction — `cstar` is the depth every Hill number is compared
at, and a value above what a sample attains puts it into extrapolation, the regime measured to
inflate diversity roughly tenfold.

### Added — `min_n_groups`, because `min_n_obs` counts the wrong thing

`min_n_obs=1000` refuses a column the corpus barely saw, counted in samples. Under one vote per
study the effective n is the number of *studies* that contributed an observation, and a column seen
a thousand times across three studies is a claim about three studies. `fit_scale` now also refuses a
column observed in fewer than `min_n_groups=20` groups, and refuses a corpus with fewer than that
many groups outright rather than returning a reference that standardises nothing.

The floor is calibrated to the measurement, not to taste: drawing whole studies from the 947-group
blood reference, 20 studies put 0.16 of columns inside the acceptance gate and 40 put 0.25. It costs
nothing on a broad corpus — every column of that reference that clears `min_n_obs` is supported by
at least 566 study groups (median 926) — and only bites on a narrow one, which is the case that used
to pass silently. Ignored without `group=`; `min_n_groups=0` fits a narrow corpus deliberately.

Note this makes the **shipped `rsig_scale_v2` a narrow fit by its own library's standard**: it was
fitted on seven cohorts, so every one of its 394 scaled columns rests on seven groups.

### Fixed — the convergence claim in `fit_scale`'s docstring is withdrawn

It quoted "median/MAD converges at N = 1,000", from a benchmark that drew samples IID and scored
them against *the same corpus's own* full fit. That measures the estimator's noise around a target
it is guaranteed to reach, not what a new corpus needs. Scored against held-out studies instead,
the unweighted estimator does not converge at any sample count: it plateaus at 0.54 by 2,000
samples and is still at 0.54 with the whole 14,833-sample pool. The unit that binds is the study.
Drawing whole studies, weighted: 80 studies (~2,000 samples) is where every column clears
`min_n_obs`; 160 (~3,800) reaches 0.66, 320 (~7,000) 0.74, 640 (~14,500) 0.85.

## 3.10.1 — 2026-08-14

Audit pass. Requires **vdjtools >= 3.7.1**, which is where the reader change lives.

### Fixed — a `v_identity` column silently moved the geometry into a different coordinate system

`TCREmp.embed` switches the V slot onto SHM-aware distances whenever the frame carries
`v_identity` or `v_mutations`, opting in by column presence. That is the right embedding for a
B-cell study and the wrong one for `rsig`, whose entire claim is that it matches a frozen
reference fitted on germline V coordinates. One IGH repertoire embedded both ways differs by a
third in `rsig:contrast:IGH:norm` (11.7 against 15.4) — with no mask, no warning, and nothing in
the vector to say which coordinate system it is in. The same file read by a pipeline that keeps
the column and one that drops it would produce two incomparable signatures.

`rsig` now drops both columns before embedding, so the geometry is a function of the repertoire
rather than of which fields its file happened to carry. The SHM load is still reported — by the
statistics half, as `vsig:shm:IGH:mean_v_identity`, which is where it belongs. An SHM-aware
`rsig` remains available by calling `TCREmp` directly, and would need its own refitted reference.

### Fixed — `mir signature` could not compute the SHM block at all

`_read` dropped `v_identity` before anything saw it, so `vsig:shm:IGH:mean_v_identity` was `nan`
on every CLI run, including on files that carried it. It is now kept by name (vdjtools 3.7.1
plumbs `keep=` through `io.read`).

### Fixed — the contrast block carried a second copy of its own width

`_contrast_features` hard-coded `{core: 0, standard: 12, full: 32}` beside the layout that
declares the same numbers. Widening the contract would not have raised: `_put` drops what it does
not recognise and leaves what it never received as `nan`, so the block would have shipped a short
vector padded with holes. The width now comes from the registered block, like every other one.

## 3.10.0 — 2026-08-14

### Added — `mir signature`, documented as the command you send someone

The whole point of the signature is that a collaborator can compute it on their own samples without
writing Python, so `--help` is the primary documentation:

```bash
pip install mirpy-lib

mir signature --preset classify cohort/*.tsv.gz -o sig.parquet
mir signature --preset classify --describe    # the columns, reading no input
mir presets                                   # the named feature sets, ranked
```

`mir signature --help` and `mir presets --help` carry worked examples, the three `recommended`
presets and when each applies, the note that every preset resolves here in full (unlike
`vdjtools signature`, which serves the `vsig` half only), and the CDR3-vs-junction trap — the
reader prefers AIRR `junction_aa` (anchors included) over IMGT `cdr3_aa` (anchors excluded), so a
file carrying only `cdr3_aa` is two residues short everywhere, which shifts the length, k-mer and
Pgen features. `docs/signature.rst` opens with the same quickstart, and the README leads with
`--preset classify`.

Requires **vdjtools >= 3.7.0**, which ships the shared column contract and the `vsig` half.

Measured end-to-end on two real AIRR samples: `--preset classify` gives 615 columns
(101 `vsig` + 514 `rsig`), one row per sample.

### Added

- **`mir.signature`** — the geometry half of a portable repertoire signature: a fixed-width,
  name-addressed per-sample vector meant to be handed to a collaborator, on a scale their model
  can consume without fitting a scaler. The column contract and the statistics half live in
  `vdjtools.signature`, which mirpy already depends on, so there is one layout and one transform
  registry rather than two that drift apart.

  Every `rsig` column is a linear functional, a norm, or a mixture coefficient of the prototype-sum
  measure `Φ(S) = Σ w_σ z_σ`, which is **fit-free** — `z_σ` is a distance vector to the bundled
  prototype panel, so no basis is estimated from anybody's cohort. Blocks: `depth` (`n_eff`,
  retained `mass`), `div` (Rao dispersion), `band` (compartment and isotype shares), `contrast`
  (`Ψ = mass·(Φ − naive)`), `phiv`/`phij`/`phic`.

  Compartment shares are closed form rather than NNLS: `Φ` is linear in the clone-weight measure
  and the bands here are a genuine partition, so a share of `Φ` is exactly a share of the weight.
  (`repertoire.band_frames`' default bands overlap by design, and an NNLS over overlapping parts
  is not a composition — its weights need not sum to one.)

- **`mir.repertoire.rao_dispersion`** — Rao's quadratic entropy in a Euclidean embedding, the
  companion to `rao_q` where there is no kernel to lean on. For squared distance the double sum
  telescopes to `2(Σw‖u‖² − ‖Φ‖²)`, so it rides along in the same chunked pass and never needs an
  `n × n` Gram; verified against an explicit Gram to ten decimals. The `n_eff/(n_eff−1)`
  correction defaults on, because the self-pair bias is of order `1/n_eff` and effective size both
  varies by orders of magnitude across samples and correlates with phenotype.

- **`mir.signature.reference` / `mir.signature.assemble`** — loading the frozen basis, and the
  assembled vector itself. `signature()` returns **both halves concatenated** (688 columns at
  standard: 160 `vsig` + 528 `rsig`), positional and in layout order, so column *i* means the
  same thing in every matrix anyone computes. `signature_cohort()` does a whole cohort.
  Measured: 0.26 s/sample on a real cohort.

  `verify()` refuses a basis whose prototype hash does not match the installed panel, because the
  failure it prevents is silent — a mismatched panel still yields a full, plausible vector, in
  coordinates nobody else shares.

  **The identity blocks centre on `naive`, not `mu_phi`**, and this was measured rather than
  assumed. Both are fit-free, but they average different things: `naive` is a sample-level `Φ`
  (an unselected repertoire), while `mu_phi` averages the prototype panel. Between-donor cosine
  spread on two real cohorts — raw 0.0105 / 0.0010, `mu_phi` 0.4008 / 0.0785, `naive` 1.2660 /
  1.6148, oracle (each cohort's own mean) 1.9031 / 1.9088. `naive` recovers 67–85% of the oracle,
  `mu_phi` only 4–21%, because `‖mu_phi − sample mean‖` is 5–7× the between-donor spread.

- **`mir.signature.scale`** — the corpus-fitted half of the reference, and the last thing between
  a set of components and a hand-off object. `signature(standardize="reference")` now returns
  dimensionless columns on a common scale, so a downstream model needs no scaler of its own.

  Robust (median, `1.4826·MAD`), computed from observed entries only *before* any imputation, and
  refusing any column seen fewer than `min_n_obs` times — each because the alternative silently
  corrupts something: moments let a handful of pathological repertoires set everyone's scale;
  imputing first deflates a sparse column's scale until the least-observed locus dominates every
  distance; and a location fitted on nine samples is not a reference. `measure_constants` also
  fixes the per-locus `cstar` and `pgen_q05` as quantiles of what the corpus attains, rather than
  the hand-picked values the blocks previously defaulted to.

- **`mir/resources/signature/rsig_v2.npz`** — the fit-free artifact: per-locus slot rotations,
  cloud location/scale, per-slot eigenvalue gaps, and a naive reference, for all 7 loci in 896 KB.
  Built by `build_rsig.py` from bundled resources with **no sample read**, and **bit-identical on
  rebuild** (verified 1-thread against 8-thread, zero eigenvector sign flips).

  Widths were set by measurement, not taste: the rotation is the PCA of the prototype cloud, and
  gate B1a checks how much *sample-level* variance that retains against a rotation fitted to the
  samples themselves. Two of three first-pass widths passed on one cohort and failed on another,
  which is why the gate runs on two. Shipped: `phiv` 16/24, `phij` 6/12, `phic` 32/48
  (standard/full).

  `Φ` must be centred before it carries information — every prototype distance is large and
  positive, so raw between-donor cosine spans ~0.001 while the shared offset is 55× the
  between-donor signal; centred, the same donors span 1.48. A test pins this so it cannot silently
  regress.

- **`mir signature`** on the CLI — `mir signature cohort/*.tsv.gz -o sig.parquet --tier standard`,
  and `mir signature --describe` for the column dictionary without reading any input. Files sharing
  a sample id are joined into one multi-locus sample, because a donor sequenced on TRA and TRB is
  one signature with both loci filled, not two half-empty ones. `mir.signature` also re-exports
  `columns` / `describe` / `LOCI` / `TIERS` so a caller never has to know the contract is
  implemented in vdjtools.

### Changed — RSIG-v2, and the scaling decisions behind it

Measured on **4,080 real samples** across seven corpora (see the analysis repo's
`benchmarks/SIGNATURE_SCALING.md`). The question was what each of twelve feature families needs
before a learner sees it, and whether that choice survives being handed to someone else.

- **The rotation is now built on the whole bundled prototype panel**, not a 4,000-prototype draw.
  With a draw the rotation is a random variable in `n_cloud`, and the junction slot had not
  converged: matched against the whole-panel rotation by `|cos|`, the V slot is stable in 23/24
  components at n=4,000 and 24/24 at 5,000, and J in 12/12 throughout — but C in only 5/48 and
  14/48. The unstable components are exactly the near-degenerate pairs (relative eigenvalue gap
  < 2% at C15, C35, C38, C41–43, C46), which **swap** rather than drift: the plane is determined,
  the labelling of its two axes is not. Taking the whole panel removes the draw, and costs nothing
  — retention on two real cohorts is unchanged (V 0.9988/0.9983, C 0.9873/0.9898) and the panel is
  already cached.

- **Eigenvalue gaps ship**, with `LocusReference.exchangeable(slot)`. The property outlives the
  fix: a linear model spanning a degenerate plane is unaffected, a per-coordinate feature-importance
  read-out on one of the pair is not interpretable.

- **`naive` recomputed against vdjtools 3.7.0's retrained models**, and `vdjtools_version` is now
  recorded in the artifact metadata. That upgrade moved `naive` by 0.13–1.6% relative per locus
  (TRG bit-identical) and moved **nothing else**: the rotations, `mu_phi` and `sd_phi` are
  bit-identical across it, because the geometry depends only on the prototype panel and the
  recombination models enter only through the naive reference.

- **Median/MAD stays, now for a measured reason.** Against the full-corpus fit, the fraction of
  columns whose location lands within 0.10 scales *and* whose scale lands within ±10%:

  | estimator | N=250 | N=500 | N=1000 | N=2000 |
  |---|---:|---:|---:|---:|
  | mean / sd | 0.362 | 0.386 | 0.576 | 0.841 |
  | median / MAD | 0.669 | 0.908 | **0.992** | 0.999 |
  | Huber M | 0.682 | 0.893 | 0.994 | 1.000 |

  The moment estimator never gets there, and the half that fails is the *scale*. The columns are
  heavy-tailed after their block transform — excess kurtosis −3 to 423, and 0.4–27% of samples
  beyond five robust deviations against the 6·10⁻⁷ a normal gives — so a standard deviation is set
  by a handful of samples. Huber's edge is 0.002, inside the noise of five draws. **N=1,000 is
  where it converges**, which is what the re-fit requirement rests on.

- **No power transform ships.** Yeo-Johnson halves the Anderson–Darling statistic of several blocks
  *in-corpus*, but the artifact would be a λ somebody else applies to their own data. Held out:
  the 875 columns of `phic`/`contrast`/`pchem`/`phij`/`aa` gain between −0.1% and +0.7%; where it
  does help, λ ranges 3.2–6.1 across corpora and those are the batch-dominated blocks, so it is
  fitting sequencing depth rather than shape; and on `clon` a transferred λ makes **57%** of columns
  *less* normal. Recorded rather than silently skipped.

- **`fit_scale(..., group=...)` records a per-column `batch_ratio`**, exposed as
  `ScaleReference.batch_dominated`. Diagnostic only — nothing divides by it — but 136 of 390 usable
  columns have more between-cohort than within-cohort spread, and a collaborator should be handed
  that list rather than left to rediscover it. The ordering is the useful part: the geometry blocks
  are the cleanest (`contrast` 0.65, `phic` 0.71) and the nuisance blocks the worst (`pair` 3.43,
  `depth` 2.19).

- **The `pub` block is out of every shipped tier** until its frozen public-clonotype panel exists.
  Declared in `standard` it contributed 28 permanently-`nan` columns to a 4,080-sample emission —
  indistinguishable, downstream, from 28 columns a sample was too shallow to support. Tier widths
  are now 152 / 688 / 1,403 (core / standard / full). A test now fails if any tier declares a block
  nothing computes.

### Fixed

- **A one-part composition took the whole sample down.** A repertoire whose clone sizes split into
  a few singletons and a few large clones puts every band below its clonotype floor at once, so
  `band_shares` returns only `_residual` and the CLR raised. Found by emitting a real corpus: one
  shallow repertoire aborted 4,000 good ones. It is a hole for that block now.

## 3.9.1 — 2026-08-14

Audit pass. No library behaviour changes.

- **The test suite did not run from the repo root.** A stale `mir/` directory sat next to `src/`,
  holding nothing but `__pycache__`, two `build_gene_library` logs and a `.DS_Store` — but Python
  resolves it as a namespace package, so `import mir` bound to an empty tree ahead of the installed
  `src/mir`. Every `pytest tests/` invocation from the repo root died at collection with
  `ModuleNotFoundError: No module named 'mir.generate'` (20 errors). The directory is gone; the fast
  tier is 246 passed / 1 skipped again. Nothing was wrong with the library.
  (The `.venv` editable install also pointed at a deleted worktree and was re-pointed at `master`.)
- `track._demo` unpacked an unused `p`; `cohort._demo` used a comma import. Unused `os` imports
  dropped from `tests/test_bench.py` and `tests/test_density.py`.
- `[tool.ruff]` added (line length 100, target py311) with `E702`/`E741`/`E402` ignored — the
  paired-short-statement style and the guarded post-skip imports in the torch tests are deliberate.
  `ruff check .` is green.
- Repo cleanup: 1.3 GB of regenerable artifacts removed (a stale `venv/` superseded by `.venv`, the
  local `airr_benchmark/` copy of the HF dataset, `build/`, `docs/_build/`, tool caches, `.DS_Store`,
  and cache/`.npy` dumps under `tmp/` — PDFs, scripts and result tables kept), plus two worktrees
  whose branches were already fully merged into `master`.

Verified at this commit: `pytest -m "not integration and not benchmark"` 246 passed / 1 skipped;
`sphinx-build -W` clean.

## 3.9.0 — 2026-08-03

**A code-review pass, and what it found.** Twenty findings across the library, all fixed, each with
a regression test that fails without the fix. Coverage 87% → 93%. Three of them changed answers
rather than crashing, which is the reason for the detail below.

A **minor** bump rather than a patch, because several fixes change *measured numbers* — the DBSCAN
`eps` curve, `exact_ceiling`, ANN background counts and `rarefy_embedding`'s `n_eff` all move — and
because new public API landed (`train_val_test_split`, `depth_report`'s `residual_dof`, per-locus
`--mmd` paths). Re-derive any recorded `eps` / `eps_factor` baseline rather than comparing across
this release.

*Silently wrong answers*

- `DonorCohort.transform` **re-fitted the identity PCA** whenever the fit cohort had too few donors
  on a chain to build one. The documented "only comparable path for new donors" therefore projected
  held-out donors into a basis the fit cohort had never seen — unscaled scores in a matrix whose
  every other column is unit-variance. It now keeps the holes, as the fit did.
- `cohort.depth_report` had **no degrees-of-freedom guard**. The regression uses an intercept plus
  one column per statistic, so with the 12 `cohort_statistics` it saturates at 13 samples and
  returns R² ≡ 1 for *any* input — a confident "your embedding is entirely depth-driven" on pure
  noise. Below one residual dof it now returns `nan` with a warning, below five it warns that R² is
  an upper bound, and the report carries `residual_dof`.
- `density.neighbor_enrichment(backend="ann")` **undercounted the background** in fixed-radius mode
  (a `k_max`-truncated kNN list), which shrinks the expected count and *inflates* fold and
  significance — the opposite of the documented conservative bias, and the one direction an
  enrichment test must not err in. The background is now an exact scipy radius count, which is both
  correct and cheaper (it drops a whole NNDescent build). Only the observed side stays approximate.
- `bench.theory.codec_losslessness`'s `exact_ceiling` charged for **every** member of a colliding
  group when only `group_size − 1` are unrecoverable, and normalised by `n_unique` while
  `exact_match` uses all `n` — so a measured value could exceed its own ceiling. Both fixed;
  `collision_rate` and `exact_ceiling` are no longer complements.
- `bench.metrics.estimate_dbscan_eps` indexed `[:, k-1]` of a `k`-neighbour query that **includes
  each point's own zero-distance self-match**, yielding the (k−1)-NN curve. At `k=1` it returned a
  curve of zeros. Recorded `eps`/`eps_factor` baselines sit on the old curve — re-derive them.
- `repertoire.sample_statistics` returned `shannon = nan` when any `duplicate_count` was 0, and the
  NaN flowed into `cohort_statistics` → `recovery_report`.
- `repertoire.missing_mass` rounded a frequency column to all-zeros and reported `M₀ = 0.0`,
  silently declaring a shallow sample a complete probability measure. It now rejects non-integer
  input in (0, 1].
- `repertoire.rarefy_embedding` stored the replicate-mean `n_eff` on an embedding whose mean is the
  *mixture* kernel mean, so `mmd_distance(unbiased=True)` over-removed the diagonal self-term. It
  now uses `1/Σw̄²` for the mixture weights.
- `ml.set_encoder` scored regression with `abs(spearman)` while checkpointing on `score > best`, so
  a perfectly **anti-correlated** model scored 1.0 and was saved as best. Now signed.
- `ml.diffusion`'s DDIM `x0` clamp did not re-derive `eps` from the clipped estimate, so at exactly
  the timesteps it exists for the update degenerated to `x ← eps` and the blow-up propagated (a
  measured trajectory excursion to ~493 against a ±6 clamp).

*Crashes and contracts*

- `mir embed repertoires --mmd` split the path on the **first dot anywhere**, so `./mmd.tsv` became
  `.TRB./mmd.tsv` (`FileNotFoundError` after both loci had been embedded and before `-o` was
  written), and an extensionless path silently let one locus overwrite another. Now splits on the
  extension.
- `mir embed clonotypes --locus beta` bypassed `normalize_locus_alias` and matched zero rows.
  Aliases now resolve, and an unresolvable locus is an error.
- `DiffusionModel.load` rebuilt the network at constructor defaults because `hidden` / `time_dim` /
  `class_dim` were never recorded in `meta`; anything trained off-defaults failed with a size
  mismatch. Bundles written before this still load.
- `bench.theory.tcrnet_convergence`, `prototype_source_correlation` and `junction_dissimilarity`
  walked their sequence argument two or three times without materializing it, so a generator over a
  polars column arrived empty and died inside `StandardScaler`.
- `ml.train` / `ml.codec` could floor the validation split to zero rows and return NaN metrics
  instead of erroring. New shared `train_val_test_split` refuses it.
- `pip install "mirpy-lib[ann]"` **could not resolve on Python ≥ 3.10**: unpinned, resolvers picked a
  numba old enough to drag in llvmlite 0.36, which refuses to build. Added a `numba>=0.60` floor.
  `[ann]` is now installed in CI, which previously executed the entire ANN backend in no job at all.

*Memory and speed, no behaviour change*

- `DensitySpace.transform` is chunked, so the `(n, n_features)` raw embedding — 6000 columns wide at
  K=2000 — is never fully resident. Every repertoire-tier caller took the single-shot path, one deep
  sample at a time.
- `Φ₁` and the second-moment block accumulate over row blocks instead of materializing the full
  `(n, D)` random-feature matrix. Exact, and bit-identical below 50k rows.
- `generate.DescriptorDensity.sample` caches a Cholesky factor. `rng.multivariate_normal` redoes an
  `O(dim³)` SVD per call, `dim` is 2051 at the default `n_rff`, and `DonorTwin.simulate` calls it
  once per twin.
- `track.fit_exposure_trajectory` hoists the design Gram out of the per-channel loop (it was rebuilt
  `g` times per iteration for a matrix that does not change).

*Docs*

- New `skills/mirpy/SKILL.md` — the v3 API surface, replacing the v2 one dropped in the 3.0 cleanup.
- README/`usage.rst`: per-locus `--mmd` naming, `--locus` aliases, and the corrected description of
  what `backend="ann"` approximates.

## 3.8.1 — 2026-08-02

*(3.8.0 was withdrawn from PyPI immediately after upload and never reached general availability; a deleted PyPI version can never be reused, so the same content ships as 3.8.1. Nothing in the library differs between the two.)*

**The repertoire measure, repaired.** `Φ(S)` was the kernel mean of a *probability* measure: weights
normalised to 1, so every sample asserted one full unit of confidence about its repertoire. At RNA-seq
depth that premise is false — a weight computed from a median of 21 unique clonotypes is `1/n` for a
technical draw size, not a clonal frequency — and the usual workaround, a minimum-clonotype floor,
deletes the immune-desert phenotype in tissue. This release lets the measure be **sub-probability**
instead (`missing_mass`, `SampleEmbedding.mass`, `naive_reference`, `contrast_embedding`), then follows
the same algebra out to functional diversity (`rao_q`), an estimable depth scale (`depth_threshold`),
compartment decomposition (`band_embeddings`, `mixture_weights`), semantics-preserving depth correction
(`rarefy_embedding`), and the two guards that keep magnitude and batch offsets honest
(`preserve_magnitude`, `residualize(shrink=True)`). Plus `recovery_report`, which scores an embedding on
*recoverability rather than competition*, and a new `Mathematical foundations` docs section deriving all
of it. No public API removed and every new behaviour is opt-in — a minor bump.

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
