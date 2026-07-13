# CLAUDE.md — mirpy v3

## What this is
`mirpy` (PyPI `mirpy-lib`, import `mir`) v3 is the antigenomics group's **ML / embedding
library** for immune receptors (TCR/BCR). A greenfield, slim rewrite: prototype embeddings
(TCREMP) now, neural codecs + density methods next (Part 2). The classical v1.x/v2 toolkit is
frozen on branch **`legacy-v2`** (`mirpy-lib` 2.x) — do not develop there.

## Reuse, don't duplicate — the ecosystem
mirpy v3 has **no AIRR data-model layer of its own**; it works on `vdjtools` polars frames and
delegates:
- **seqtree** — alignment. Junction/CDR3 distance = `seqtree.gapblock.score_matrix` (BLOSUM62
  Gram penalty, gap placements `(3,4,-4,-3)`). Replaces the old C++ scorer entirely.
- **vdjtools** (core dep) — AIRR schema + IO (`vdjtools.io`), germline reference
  (`vdjtools.model.reference`), Pgen + synthetic sampling (`vdjtools.model.{load_bundled,
  native.pgen_aa_batch, generate.generate}`).
- **vdjmatch** (`[annotate]`) — VDJdb annotation / E-values.
- **arda** (`[build]`) — build-time germline region annotation. **arda is the single germline
  source of truth**: prototypes, germline-distance matrices, and all query data share one arda IMGT
  allele namespace. Needs `ARDA_HOME` at build time (`arda-mapper` wheel; no cmake build for
  germline reading, but `arda rnaseq map` fetches mmseqs2).

The reference/prototype coordinate system is **arda-native** (2026-07): `mir/resources/germline_dist`
is baked from arda germline (`build_region_annotations.py` → arda `markup.aa.tsv`); prototypes
(`generate_prototypes.py`) come from **arda-annotated real repertoires** (`isalgo/airr_model_read`
functional reads → `arda rnaseq map`), giving arda names + a real junction manifold. NB: arda-native
generative models via `vdjtools.model.from_arda` exist (EM-learned, used as the density P_gen
background) but their synthetic junctions embed *worse* than real repertoires (degenerate lengths,
negative S2) — so prototypes use real reads, not model generation.

mirpy is normally read-only to the sibling repos; the `from_arda` builder + a tandem-D generation
fix were added to `vdjtools` under the owner's direction (this is that owner's ecosystem).

## Layout (`mir/`)
- `aliases.py`, `alleles.py` — species/locus + allele normalization.
- `distances/junction.py` — `junction_distance_matrix` (seqtree.gapblock). `distances/germline.py`
  — resource-backed V/J/CDR1/CDR2 lookup with allele cascade.
- `embedding/prototypes.py` — bundled prototype loader. `embedding/tcremp.py` — `TCREmp` /
  `PairedTCREmp` (polars frame in → `(N,3K)` float32). `embedding/pca.py` — PCA denoise (T3).
  `embedding/presets.py` — per-chain `n_prototypes` + PC recommendations (`get_preset`);
  `from_defaults(n_prototypes=None)` uses them. Compact chains (IGK/IGL/TRG) 1000 protos / ~20
  PCs; diverse chains (IGH/TR*) 2000 / ~65 PCs (95%), ~220–300 PCs (99%, for codec reconstruction).
- `bench/` — `vdjdb.py` (loader), `metrics.py` (DBSCAN+kneedle, F1/retention), `theory.py`
  (S1–S3 + T5 + T6 `tcrnet_convergence`). Needs `[bench]`.
- `density.py` — continuous-density TCRNET/ALICE (T6): `fit_density_space` (one shared PCA basis),
  `neighbor_enrichment` (balloon adaptive-radius Poisson/binomial + water-level calibration),
  `enriched_mask`, `denoise_and_cluster`, `generate_background` (vdjtools P_gen, lazy). Torch-free
  (scipy/sklearn). Prefer a **biological control** as background (differential) over P_gen.
- `ml/` — Part 2 (torch), neural codecs.
- `resources/` — `prototypes/` (TSVs + manifest), `gene_library/` (region_annotations.txt),
  `germline_dist/` (baked `.npz`, from `build_germline_dist.py`).

## Build / test / run
- Conda env **`mirpy`** (Python 3.12; do NOT use `.venv` here). `pip install -e .`
  (pure-Python hatchling; no C build). Extras: `[bench] [annotate] [build] [ml] [docs] [dev]`.
- Tests: `python -m pytest tests/ -q` (45 pass, ~2s; all self-contained on bundled resources).
- Experiments: `python experiments/reproduce_supplementary.py` (theory S1–S3),
  `python experiments/benchmark_vdjdb.py` (Table S1). See `THEORY.md`.

## Conventions
- AIRR polars frames in/out, keyed by `vdjtools.io.schema` names (`v_call, j_call, junction_aa,
  locus`). No `Clonotype` class.
- v3 embeddings are a **new, versioned coordinate system** (gapblock ≠ the v2 BioPython junction
  scorer) — any model trained on v2 embeddings must be retrained.
- Baked `germline_dist/*.npz` are versioned artifacts; regenerate whenever the gene library /
  `region_annotations.txt` changes (`build_germline_dist.py`, needs `[build]` BioPython).

## Open loops / next steps
- **v3.0 remaining**: 10X paired benchmark; docs (Sphinx theory section + notebooks); CI; publish
  `py3-none-any` wheel; regenerate `generate_prototypes.py` via `vdjtools.model.generate`.
- **Bench tuning**: raw kneedle eps over-merges; `cluster(eps_factor=0.4)` recovers the paper
  regime (Fig 1's dataset-specific factor). Exact Table S1 F1 needs the paper's VDJdb release.
- **Part 2 (v3.1+)** — `mir.ml` (torch, `[ml]` extra), absorbing `irrm-codec`. Codec targets the
  **PCA-compacted junction embedding** (T3: 1000-D junction → ~51 PCs @95% var on arda coords,
  99% for reconstruction), fit train-only.
  V/J stay exact germline lookups (nothing to learn); the codec's job is the junction part.
  Results on M3 MPS (`experiments/train_{forward_encoder,inverse_decoder,pgen_regressor}.py`),
  **re-pinned on the arda-native coords** (2026-07-13):
  - **DONE forward codec** (`tokenize,encoder,train`): seq → compact code, reconstruction
    cosine **0.9984** (n=8k; paper 0.887). DNN inference is K-independent. Geometry is fine at
    95% (51 PCs) — only *reconstruction* is compaction-sensitive (below).
  - **DONE inverse codec** (`decoder`, `train.train_inverse_decoder`): 284-PC code (99% var) →
    seq, exact-match **0.40**, token-acc 0.97 (irrm-codec 0.50 from the *full* embedding). **NB:
    the arda-real prototypes are lower-rank, so TRB 95%-var is only 51 PCs → exact 0.08
    (over-compacted); 99% (284 PCs) restores 0.40.** The T5 chain-adaptive lesson now bites TRB
    too, not just IGH/TRD — the decoder default is 99%.
  - **DONE Pgen regressor** (`train.train_pgen_regressor`): seq → log10 Pgen(1mm), r **0.967**,
    **~190× faster** than the native DP. Breakdown (`experiments/benchmark_pgen_variants.py`):
    r ranks marginalized > J > V > V&J (a CDR3-only regressor best predicts the pure-CDR3
    marginalized Pgen; V&J-conditional depends on unseen genes) and 1mm > exact (smoother ball).
  - **DONE unified codec** (`codec.py`): jointly train encoder+decoder with a geometry-anchor
    term (`lambda_embed`) — code stays ≈ the true embedding (distances preserved) while
    round-tripping seq→code→seq; encoder+decoder co-adapt (roundtrip_exact > decode_true_exact).
  - **DONE smart shipping** (`bundle.py`): **embeddings are only comparable if prototypes + PCA
    rotation match.** `CodecBundle` serializes the PCA transform + a prototype hash + weights;
    `load` refuses a prototype-hash mismatch so incomparable embeddings can't be mixed. Any
    trained codec MUST be shipped as a bundle, never bare weights.
  - Per-chain/species breakdown: `experiments/benchmark_codec_chains.py` (forward cos 0.997–0.999
    universal; inverse chain-dependent — short κ/λ/γ/mouse easy, IGH/TRD hard).
  - **DONE T5 (SHM/IGH)** (`bench.theory.shm_embedding_drift`, `experiments/benchmark_igh_shm.py`):
    SHM embedding drift is ~linear/sublinear in mutation load (bounded; IGH 104/mut < TRB 128/mut
    — IGH lowest slope, robust to SHM). IGH's hard reconstruction is **over-compaction, not the
    frame**: on arda coords 95% code (95 PCs) → exact 0.115, 99% (422 PCs) → 0.356 (> old 0.152;
    real IGH prototypes reconstruct better). ⇒ variance retention should be **chain-adaptive**
    (95% preserves geometry; 99% needed for reconstruction on the compact arda prototypes — TRB
    and IGH/TRD alike); the bundle already ships per-codec.
  - **DONE T6 (continuous-density TCRNET/ALICE)** (`mir/density.py`): graph-free balloon
    enrichment `E(z)=f_obs/f_gen` in embedding space; Poisson (ALICE, P_gen bg) or binomial
    (TCRNET, control bg) + BH q; water-level calibration for the naive regime. Theory
    `tcrnet_convergence` confirms the r→0 graph limit (ρ 0.37→−0.05 as radius grows). Benchmarks
    `experiments/benchmark_density_{yfv,ankspond,tcrnet,vdjdb}.py` on HF
    `isalgo/airr_{yfv19,ankspond,benchmark,control}` + VDJdb slim.
    **Key lesson**: real repertoires are pervasively convergent, so a P_gen background flags ~40% of
    clones — use a *biological control* (differential: day15-vs-day0, B27±, CMV-vs-control) for
    specificity, and process the **full repertoire** (subsampling dilutes the sparse antigen clusters).
    `benchmark_density_vdjdb.py` makes this quantitative: VDJdb TRB ridges bystander-filtered by
    `reference.id` (≥2 PMIDs) + admixed `airr_control` noise; under P_gen the noise over-flags 43%
    vs 1% under a control bg (46× signal:noise lift), and it counts mountains/epitope (GILGFVFTL 32,
    NLVPMVATV 4 — polyclonality tracks precursor freq, Pogorelyy 2018).
  - **TODO**: epitope/MHC; scale codec on HF `airr_benchmark` (10–100M).
- Full plan: `~/.claude/plans/i-want-to-completely-crystalline-lake.md`.
