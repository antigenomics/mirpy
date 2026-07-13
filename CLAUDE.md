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
- **arda** (`[build]`) — build-time germline region annotation.

mirpy is **read-only** from vdjtools' perspective; never edit the sibling repos — surface bugs
to their owners instead.

## Layout (`mir/`)
- `aliases.py`, `alleles.py` — species/locus + allele normalization.
- `distances/junction.py` — `junction_distance_matrix` (seqtree.gapblock). `distances/germline.py`
  — resource-backed V/J/CDR1/CDR2 lookup with allele cascade.
- `embedding/prototypes.py` — bundled prototype loader. `embedding/tcremp.py` — `TCREmp` /
  `PairedTCREmp` (polars frame in → `(N,3K)` float32). `embedding/pca.py` — PCA denoise (T3).
- `bench/` — `vdjdb.py` (loader), `metrics.py` (DBSCAN+kneedle, F1/retention), `theory.py`
  (S1–S3 experiments). Needs `[bench]`.
- `ml/` — Part 2 (torch), empty stub.
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
- **Part 2 (v3.1+)** — `mir.ml` (torch, `[ml]` extra), absorbing `irrm-codec`:
  - **DONE forward codec** — `mir/ml/{tokenize,encoder,train}.py`: fixed-len-40 one-hot →
    CNN → junction-distance embedding, free supervision (gapblock targets). Test mean cosine
    **0.935** on TRB (n=10k, K=1000; paper 0.887). `experiments/train_forward_encoder.py`.
    The junction component is the expensive part; V/J are cheap germline lookups. DNN inference
    is K-independent (wins over gapblock at large K / on GPU).
  - **TODO**: full-embedding variant (feed V/J via `nn.Embedding`); inverse decoder
    (embedding→seq, IGH hard); Pgen-from-embedding regressor; continuous-density TCRNET (T6);
    IGH/SHM (T5); epitope/MHC. Scale training on HF `airr_benchmark` + `vdjtools` sampling.
- Full plan: `~/.claude/plans/i-want-to-completely-crystalline-lake.md`.
