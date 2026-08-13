# CLAUDE.md — mirpy v3

## What this is
`mirpy` (PyPI `mirpy-lib`, import `mir`) v3 is the antigenomics group's **ML / embedding library**
for immune receptors (TCR/BCR): prototype embeddings, density methods, repertoire-level embeddings,
neural codecs. Greenfield slim rewrite — the classical v1.x/v2 toolkit is frozen on branch
**`legacy-v2`** (`mirpy-lib` 2.x); do not develop there.

**API surface lives in [`skills/mirpy/SKILL.md`](skills/mirpy/SKILL.md), not here.** Module-by-module
detail and the completed-work narrative are archived in
`~/vcs/projects/2026-mirpy-analysis/benchmarks/LIBRARY_LOG.md`.

## ⛔ Provenance rule (user, 2026-08-02)
**Quote the numbers, never the source** — no corpus size, no dataset / programme / cohort names
anywhere in this repo. (Pre-dating files still violate this: `README.md`, `CHANGELOG.md`,
`SOURCES.md`, `src/mir/{generate,cohort}.py`.)

## Repo split (2026-07-16) — three homes
- **`~/vcs/code/mirpy`** (here): library only — src-layout `src/mir/`, `tests/`, `examples/`
  (marimo), `docs/`. Only `README`/`CHANGELOG`/`CLAUDE`/`SOURCES` at the root.
- **`~/vcs/projects/2026-mirpy-analysis`**: benchmark scripts (`benchmarks/`, local + aldan3), result
  docs (`BENCHMARKS.md`, `THEORY.md`, `ROADMAP.md`, `REPERTOIRE_*.md`, `LIBRARY_LOG.md`), figures,
  run outputs, dataset catalog. **Run the library from here; numbers of record are refreshed here.**
- **`~/vcs/manuscripts/2026-mirpy-ms`**: theory appendix (`appendix/tcremp_theory.tex`) + paper.

Heavy benchmarks run on **aldan3** (env `mirpy`, `sbatch`); light jobs local.

## ⛔ Worktrees — REQUIRED
Multiple Claude sessions edit mirpy concurrently (the 2026-07-16 split hit real collisions across
parallel sessions). **Never work directly on the main checkout** — call `EnterWorktree` at session
start (worktree under `.claude/worktrees/`, fresh branch off `origin/master`), commit and push from
there, merge via PR/fast-forward. Same for `2026-mirpy-analysis` and `2026-mirpy-ms`.

## Reuse, don't duplicate — the ecosystem
mirpy has **no AIRR data-model layer of its own**. It works on `vdjtools` polars frames and delegates:
- **seqtree** — alignment; junction/CDR3 distance = `seqtree.gapblock.score_matrix` (BLOSUM62 Gram
  penalty, gaps `(3,4,-4,-3)`). Replaced the old C++ scorer entirely.
- **vdjtools** (core dep) — AIRR schema + IO, germline reference, Pgen + synthetic sampling.
- **vdjmatch** (`[annotate]`) — VDJdb annotation / E-values.
- **arda** (`[build]`) — build-time germline region annotation. **arda is the single germline source
  of truth**: prototypes, germline-distance matrices, and all query data share one arda IMGT allele
  namespace. Needs `ARDA_HOME` at build time.

The coordinate system is **arda-native** (2026-07): `resources/germline_dist` is baked from arda
germline; prototypes come from **arda-annotated real repertoires**, giving arda names and a real
junction manifold. NB arda-native generative models (`vdjtools.model.from_arda`) exist and back the
density P_gen null, but their **synthetic junctions embed worse than real repertoires** (degenerate
lengths, negative S2) — so prototypes use real reads, never model generation.

mirpy is normally **read-only to the sibling repos**; the `from_arda` builder and a tandem-D fix were
added to `vdjtools` under the owner's direction.

## Layout (`src/mir/`) — one line each; details in SKILL.md
`aliases`/`alleles` (species/locus + allele normalization) · `distances/` (junction via gapblock,
germline lookup with allele cascade) · `embedding/` (`TCREmp`/`PairedTCREmp`, prototypes, PCA denoise,
per-chain presets) · `density` (continuous TCRNET/ALICE, balloon enrichment, abundance-aware) ·
`repertoire` (`Φ(S)`, MMD, witness, sub-probability + measure-algebra tiers) · `cohort` (the digital
donor, residualize, biomarkers) · `explain` (channel ablation over any feature matrix) · `bench/`
(vdjdb loader, clustering metrics, theory S1–S3/T5/T6, `eval.py` scorers) · `track` (exposure
trajectory) · `generate` + `twin` (generative loop, mechanical half) · `ml/` (torch: codecs,
set encoder, diffusion) · `cli` (the `mir` console script) · `resources/`.

Coordinate knobs default to the published space: `metric="squared"`, `alignment="gapblock"`,
`backend="kdtree"` for density (flipped in Phase 0 — **re-verify any recorded balloon-mode baseline**,
±1 boundary counts).

## Build / test / run
Repo-local **`.venv` via uv**, Python 3.12 (conda retired 2026-07-18). `bash setup.sh`
(`--dev-parents` editable-installs `../seqtree ../vdjtools ../vdjmatch`; `--docs`, `--tests`) or
`uv pip install -e ".[dev,bench]"`. Pure-Python hatchling, no C build for `mir` itself. Extras:
`[bench] [ann] [annotate] [build] [ml] [docs] [dev] [examples]` — `[ann]` (pynndescent) is split out
so `[bench]` stays numba-free.

```sh
python -m pytest tests/ -q -m "not integration and not benchmark"   # fast tier, bundled resources
mir embed clonotypes SAMPLE   |   mir embed repertoires SAMPLE…     # CLI
```

**If collection fails with `No module named 'mir.<anything>'`, check for a stray `mir/` at the repo
root** — Python resolves it as a namespace package ahead of the installed `src/mir`, and every
`pytest` run from the root then dies at import. One appeared from a `build_gene_library.py` run with
the repo root as cwd (v3.9.1). Also confirm the editable install points at *this* checkout and not a
removed worktree: `python -c "import mir; print(mir.__file__)"`.
`ruff check .` is green as of v3.9.1; `ruff format` is NOT the house style (it would rewrite 56
files) — do not run it.

## Conventions
- AIRR polars frames in/out, keyed by `vdjtools.io.schema` names (`v_call, j_call, junction_aa,
  locus`). **No `Clonotype` class.**
- v3 embeddings are a **new versioned coordinate system** (gapblock ≠ the v2 BioPython scorer) — any
  model trained on v2 embeddings must be retrained.
- **Embeddings are comparable only if prototypes + PCA rotation match.** Ship any trained codec as a
  `CodecBundle` (serializes PCA transform + prototype hash), never bare weights; `load` refuses a
  hash mismatch. Same check guards `RepertoireSpace`/`DonorCohort` save/load.
- Baked `germline_dist/*.npz` are versioned artifacts — regenerate whenever the gene library or
  `region_annotations.txt` changes (`build_germline_dist.py`, needs `[build]`).
- **Parallelism**: embedding all-core by default (`TCREmp(threads=0)`, GIL-released C++); density
  `backend="kdtree"` = exact multicore, `"ann"` = auto-all-core; `cluster(n_jobs=-1)`; PCA/RFF ride
  BLAS (cap via `OMP_NUM_THREADS`). GPU only in `mir.ml`: `pick_device()` = **CUDA → MPS → CPU**,
  override via `device=`/`MIR_DEVICE`.

## Interpretation rules that cost real experiments
- **A P_gen background over-flags** — real repertoires are pervasively convergent (~40% of clones).
  Use a *biological control* (differential) for specificity, and process the **full** repertoire;
  subsampling dilutes sparse antigen clusters.
- **Variance retention is chain-adaptive**: 95% preserves geometry, **99% is needed for
  reconstruction** on the compact arda prototypes — for every chain, not just IGH/TRD.
- **Never add a min-clonotype floor** — that's a blood rule, not a tissue rule. Use
  `contrast_embedding`, which sends an immune desert to the origin instead of deleting it.
- **Mass-1 renormalisation makes coverage/richness unrecoverable from Φ by construction** — a
  deficient measure wins those by design, so read `recovery_report` as recoverability, not competition.
- **Per-column z-scoring deletes a magnitude signal** — a sub-probability block needs
  `add(..., preserve_magnitude=True)` (one global scalar). This cost one experiment a bogus result.
- **Plain per-group centring can make batch *easier* to read out-of-sample** (the offset estimate
  injects as much as it removes) — use `residualize(shrink=True)`, and only an out-of-sample eval
  sees the difference.
- Eigenvalue (spectral) summaries are **rotation-invariant and lossy for directional signal** — keep
  the full second-moment upper triangle; `n_eigs=r` stays opt-in.
- Single-split AUCs mislead; **CI-overlapping is not a separation.** Two over-claims were caught
  adversarially this way, and one spike-in result was circular until the select/detect metrics were
  split.

## Open loops / next steps
- **Roadmap** (`2026-mirpy-analysis/benchmarks/ROADMAP.md`): Phases 0, 1, 2 done; Phase 5 partially
  (via `track.fit_exposure_trajectory`). **Next: Phase 3 (embedding inversion/generation), Phase 4
  (multimodal encoders)**; still open — `CodecBundle.from_unified/from_decoder`, `clonotype_flux`.
- **Analysis-repo follow-up**: refactor `_tcga_embedding.build_embedding` onto
  `cohort.fit_donor_embeddings` (+ `extra_channels`) and re-verify the pan-cancer ΔC numbers. Of its
  four jobs, (b)+(d) are promoted into `mir.explain`, (c) is correctly analysis-local; **(a)
  "fit one `RepertoireSpace` per locus over a cohort" still belongs in `mir.repertoire`.**
- **v3.0 remaining**: 10X paired benchmark; docs (Sphinx theory section + notebooks); CI; publish
  `py3-none-any` wheel; regenerate `generate_prototypes.py` via `vdjtools.model.generate`.
- **Codec**: exact-match is **training-data-limited, not architecture-limited** — levers in order are
  more data (free) > PC→99% var > K→~2000 > autoregressive decoder + wider `FIXED_LEN` for the long
  real-IGH tail. Scale on the large HF corpus. **Never embed `c_call`** — isotype is ~3 bits
  independent of V/J/CDR3, so carry it as an exact stored column.
- **Bench tuning**: raw kneedle eps over-merges; `cluster(eps_factor=0.4)` recovers the paper regime.
  Exact Table S1 F1 needs the paper's VDJdb release.
- **Open questions**: epitope/MHC tier; larger cohort/depth for the weak HLA signal; biological-control
  FPR for spike-in; per-cancer `n_pc`; a learned (flow/VAE) manifold for sequence-level simulation.
