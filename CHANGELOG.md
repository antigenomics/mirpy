# Changelog

All notable changes to `mirpy-lib` (import `mir`). This project follows semantic versioning; the v3 line is a
greenfield ML/embedding rewrite (the classical v1.x/v2 toolkit is frozen on branch `legacy-v2`).

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
