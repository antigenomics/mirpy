# Changelog

All notable changes to `mirpy-lib` (import `mir`). This project follows semantic versioning; the v3 line is a
greenfield ML/embedding rewrite (the classical v1.x/v2 toolkit is frozen on branch `legacy-v2`).

## Unreleased

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
