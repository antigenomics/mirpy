---
name: mirpy
description: Use mirpy APIs for clonotype coordinates, sample RFF embeddings, density, corpus rsig features and sample PCA.
---

# mirpy API guide

In a source checkout, read its repository instructions first. Public docs are canonical; this skill routes work rather
than duplicating their examples or historical benchmark records.

| Task | Read | Reuse |
|---|---|---|
| Clonotype coordinates, sample RFF, saved bases and PCA | [Embeddings](https://docs.isalgo.dev/mirpy/embeddings.html) | `TCREmp`, `fit_sample_spaces`, `sample_embeddings`, `RepertoireSpace` |
| Density, MMD, donor embedding, scorers and codecs | [Usage](https://docs.isalgo.dev/mirpy/usage.html), [API](https://docs.isalgo.dev/mirpy/api.html) | Existing batch/native APIs |
| Frozen corpus geometry features | [Signature](https://docs.isalgo.dev/mirpy/signature.html), [Channels](https://docs.isalgo.dev/mirpy/channels.html) | `rsig`, `rsig_cohort`, shared `Corpus` |
| Filtering, input columns and IDs | [Preprocessing](https://docs.isalgo.dev/mirpy/preprocessing.html) | vdjtools AIRR IO/schema |
| Mathematical contracts | [Math](https://docs.isalgo.dev/mirpy/math.html) | Established derivations |

## Contracts

- Sequence-codec training stages one device batch from compact host tokens/targets;
  validation and test calls honor the same batch size. Scaler/PCA fitting still
  needs its selected training matrix. Preserve split/shuffle/batch-normalization
  semantics and frozen resources; see [Usage](https://docs.isalgo.dev/mirpy/usage.html). Diffusion stages bounded
  device batches too, retaining fixed validation noise on the host.

- Neural trainers and trained-model inference use `threads=0` for up to eight allocated CPUs,
  `1` for serial or an explicit positive budget. VDJtools owns phase exclusion;
  Torch intra-op is the sole parallel layer, and caller limits are restored.
  Seeded fits serialize within an interpreter because framework RNGs are shared.

- Sample embedding: clonotype coordinates → shared clonotype scaler/PCA → RFF →
  weighted mean → optional sample PCA. Never substitute RFF of a mean or prototype
  averages for sample RFF. `mir embeddings` aliases `mir signature`, producing
  existing rsig geometry, not RFF. rsig and vsig join on `sample_id`.

- `sample_embeddings` defaults to `n_jobs=0` (allocated cores). Sample mapping
  finishes before optional sample PCA. `mir pca --threads` controls fit and apply;
  `TablePCA.transform(table, threads=0)` uses bounded row blocks, with 0 selecting
  the allocation. Input tables and final sample vectors still occupy memory.
- `TCREmp.embed` interleaves V/J/junction per prototype: `[0::3]`, `[1::3]`,
  `[2::3]`. AIRR `junction_aa` includes conserved Cys and Phe/Trp anchors; IMGT
  `cdr3_aa` excludes them.
- RFF comparison requires the same full basis and abundance weighting. Preserve
  `embedding_space`, sample IDs and channels. Save/reuse each `RepertoireSpace`;
  its pickle files must be trusted. Optional sample PCA fits input RFF, compatible
  control RFF, or loads a saved rotation.
- `vdjtools.signature.corpus.TablePCA` owns table PCA; `fit_pca` is also reused by
  `mir.cohort._identity_matrix`. Search these before implementing any PCA helper.
  `Corpus` owns robust signature transforms. Their artifacts are not interchangeable.
- Missing loci are holes with presence indicators. A shallow locus can have valid
  geometry while dispersion is unmeasurable. Do not erase channels or add a
  min-clonotype floor to the RFF workflow.
- Import `mir` before numerical runtimes. Use repeatable deferred sample readers,
  bounded workers and serial inner kernels. No futures/executors or cohort-wide
  retention of input frames. Count live objects and large matrices before changing
  memory behavior; reuse existing CPU/cgroup memory guards.
- `mir.embedding.reference.build_reference_rotations` fits separate raw-feature/RFF rotations and
  exports their required bases. Each bundle has its own coordinate identity; use
  that basis before applying its RFF rotation. Existing signatures remain distinct.
  Keep raw-source provenance private; no source labels, paths or IDs in bundles.

## Verification

Use the repo-local uv `.venv` and `setup.sh` flags from `AGENTS.md`. For docs,
activate that environment and run `env -C docs make html` with zero warnings.
Check actual CLI help and imports against examples. Use focused regression tests
for changed behavior; do not rerun completed scientific benchmarks without a
producing-code change. Never run `ruff format` across the repository.

## Configured statistical references

Synthetic corpus generation delegates to vdjtools spawned pool workers, followed
by its sample-thread phase; use an importable, guarded entry point.

Reference fitting separates ``jobs`` sample workers from ``threads`` BLAS fitting
after workers join (default 0: up to 16 allocated CPUs; excessive counts raise).

`embedding.reference.build_reference_rotations(..., feature_definition=...)`
accepts vdjtools' frozen basic/joint feature contracts. Reuse vdjtools extraction
and TablePCA; do not duplicate motif counting, vocabulary or clustering here.
Bundles persist the configured definition and vocabulary. Raw statistical reference
bundles export `feature-vocabulary.json`; pass it to `vdjtools features
--feature-vocabulary` (or `raw_features_cohort(vocab=...)`) before applying the
feature rotation, since a newer germline can extend the default vocabulary. This option changes
statistical features only; existing rsig and sample RFF remain separate.
Read [Embeddings](https://docs.isalgo.dev/mirpy/embeddings.html) and vdjtools [Features](https://docs.isalgo.dev/vdjtools/features.html) before applying one.

Configured statistical reference builds delegate to vdjtools
`TablePCA.fit_features`; do not materialize a dense cohort table before fitting.
This preserves raw median-impute/StandardScaler coordinates, distinct from Corpus
composition scaling. The fresh application check reads only small query tables.
`jobs` sample mapping joins before `threads` BLAS; never nest their budgets.
Legacy and RFF reference paths retain their existing behavior.

Frozen references: `mir reference NAME` downloads and verifies the release-pinned
`isalgo/airr_feature_signatures` bundle under `references/NAME`. Choose `rna-blood`,
`rna-tissue`, `synthetic-blood`, `synthetic-tissue`, or `deep-tra-trb`. Pre-stage once
before fan-out; use `--cache-dir PATH --offline` on workers. Keep the matching
`basis/`, RFF PCA, statistical PCA and frozen feature vocabulary together.

Isalgo ``generate_background``/``naive_reference`` calls select one native batch
by default and ``threads=0`` uses the allocation. Explicit OLGA legacy selections
retain reference streams by default; ``engine="reference"`` pins that engine for
any source. Reference cache keys use the resolved engine, not the automatic option.

Model defaults use `isalgo`; `learned`/`arda` are aliases of that fresh family.
`olga-legacy` and `olga` preserve frozen OLGA parameters and reference streams.
Old learned/arda weights require a pinned previous release; saved corpora and
embedding/prototype/germline assets are not regenerated by this integration.
