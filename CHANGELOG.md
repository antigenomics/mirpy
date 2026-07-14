# Changelog

All notable changes to `mirpy-lib` (import `mir`). This project follows semantic versioning; the v3 line is a
greenfield ML/embedding rewrite (the classical v1.x/v2 toolkit is frozen on branch `legacy-v2`).

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
