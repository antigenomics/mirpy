# mirpy ROADMAP — "vdjtools, at the embedding level"

The goal: mirpy is the antigenomics group's **embedding / deep-learning** repertoire library — the
things vdjtools does with counts and graphs (diversity, overlap, biomarkers, generation), re-expressed
**in the language of embeddings**. This file is the durable audit + plan; per-session working notes live
in `~/.claude/plans/`.

## The organizing idea — three verbs

Everything the library does is one of three verbs, and the API is shaped to make that explicit:

- **make** an embedding — `TCREmp.embed` (clonotype) · `density.fit_density_space` (cloud) ·
  `repertoire.sample_embedding` / `sample_descriptor` (repertoire) · `cohort.fit_donor_embeddings`
  (digital donor) · `ml.SetEncoder` (learned).
- **measure across / from** embeddings — `density.neighbor_enrichment` · `repertoire.mmd_distance` /
  `mmd_matrix` · `repertoire.class_witness` · `cohort.incidence_biomarkers` · `explain.channel_report`.
- **generate + decode back** — `ml.decoder.SequenceDecoder` · `repertoire.decode_metrics` ·
  `density.generate_background` · (planned) `generate.*`.

**Division of labour (the house rule):** the library owns **geometry, fusion, serialization under the
prototype-hash comparability contract**; the analysis owns **which clinical/biological channels and which
statistical model**. `explain.py` ships no scorers and never sees `y`; `bench/eval.py` holds the scorers
the analysis hands it.

## Tiers (current)

| Tier | Module(s) | What it makes / measures |
|---|---|---|
| Clonotype | `embedding/`, `distances/`, `ml/` | prototype embedding, junction/germline distance, seq↔emb codec, Pgen regressor |
| Density (cloud) | `density.py` | TCRNET/ALICE neighbour-enrichment in embedding space, spike-in, motif denoise |
| Repertoire | `repertoire.py` | Φ(S) = kernel-mean ‖ Hill ‖ 2nd-moment; MMD; `class_witness`; descriptor |
| Explain | `explain.py` | named-channel fusion + scorer-agnostic ablation |
| **Cohort / digital donor** | `cohort.py` | multi-chain `DonorCohort`, `residualize`, `cluster_samples`, `incidence_biomarkers` |
| Scorers | `bench/eval.py` | `cv_auc` / `cv_cindex` / `km_logrank` / `kmer_matrix` (the `channel_report` closures) |
| Generative | *(planned `generate.py`)* | density over descriptors → evolve/sample → decode to clonotypes |

## vdjtools → embedding parity

- **DONE:** IO/schema (reuse vdjtools) · overlap→**MMD** · diversity→**Hill channel** ·
  TCRNET/ALICE→**density enrichment** · co-occurrence→**2nd-moment block** · Pgen→learned regressor ·
  biomarker-Fisher→**`cohort.incidence_biomarkers`** (delegate) · sample-clustering→**`cohort.cluster_samples`**.
- **PARTIAL:** vdjmatch annotation→nearest-epitope (benchmark only, no `annotate()`) · k-mer/physchem→named
  channels (bus ready) · paired α/β (clonotype concat yes; repertoire-Φ via per-locus `DonorCohort`).
- **MISSING (planned):** clonotype-tracking→**embedding trajectory** · **preprocess** in embedding space
  (downsample/batch-correct beyond `residualize`) · the **repertoire generative loop**.

## Downstream analysis modes

- **digital immunome / digital donor** — `cohort.fit_donor_embeddings` → `DonorCohort` (multi-chain,
  hash-serialized). ✅
- **disease / exposure classification** — `DonorCohort` + `explain.channel_report(scorer=cv_auc)`. ✅
- **cancer prognosis & survival** — `channel_report(scorer=cv_cindex)` + `km_logrank`; the TME-aware
  channels (isotype/composition/atypicality) injected by the analysis via `extra_channels`. ✅ (library);
  full pan-cancer number reproduction is an analysis-repo step (needs `~/hf/airr_tcga`).
- **motif / biomarker detection** — `cohort.incidence_biomarkers` (Fisher, wins at low donor n) beside
  `repertoire.class_witness` (geometry). ✅
- **generative decode to real repertoires/clonotypes** — clonotype codec DONE; repertoire generative loop
  **planned** (below).

## Roadmap (by leverage / risk)

- **Phase 0 — robustness + optimization quick wins. ✅ DONE** (commit `fix(phase0)`): degenerate-input
  guards (unbiased-MMD singleton, empty/zero-count sample, negative-`n` prototypes, zero radius, empty
  density trees); density default `backend="kdtree"`; auto-chunk + frame-sample fits; vectorized
  `hla_stratified_mmd`; precomputed-witness fast-path.
- **Phase 1 — cohort tier / digital donor. ✅ DONE**: `bench/eval.py` scorers;
  `repertoire.{fit_repertoire_spaces, centroid_atypicality}`; `cohort.py`
  (`DonorCohort`/`fit_donor_embeddings`/`transform`/`save`/`load`, `residualize`, `cluster_samples`,
  `incidence_biomarkers`). **Follow-up (analysis repo):** refactor `_tcga_embedding.build_embedding` onto
  `fit_donor_embeddings` (+ `extra_channels` for isotype/composition/atypicality) and re-verify the
  pan-cancer ΔC numbers (SKCM +0.039 / BLCA +0.025 / HNSC +0.022 / LGG +0.016).
- **Phase 2 — generative loop, mechanical half** (priority frontier): `CodecBundle.from_unified/from_decoder`;
  `generate.py` `DescriptorDensity` (conditional Gaussian) + `evolve` + `sample`, promoting the shipped
  in-silico-evolution manifold. Accept: reproduce the coupled in-silico slopes through the library.
- **Phase 3 — generative loop, research half**: RepertoireSpace-PCA ↔ codec-PCA basis bridge →
  `invert_kernel_mean` (herd from real candidates + count model + germline-sampled V/J) + Pgen filter.
  Accept: re-embedding the generated multiset lands near the target μ. *Fund, iterate.*
- **Phase 4 — multimodal encoders**: `modalities.isotype_fractions` / `hla_indicator` /
  `fit_epitope_annotator` (nearest-epitope `annotate`) + paired-Φ recipe. (GEX / pMHC-groove deferred.)
- **Phase 5 — embedding trajectory** (research): `track.repertoire_trajectory` (Φ velocity) +
  `clonotype_flux` (differential enrichment).

## Non-goals / risks

- **Stays in vdjtools** (delegate, never reimplement): AIRR schema/IO, `downsample`, P_gen generation,
  `kmer_profile`, the `fisher_association` engine.
- **Stays analysis-local:** tissue/TME feature engineering (isotype/composition), Cox penalizer / CV scheme.
- **Comparability bites twice** in `DonorCohort`: per-locus `prototype_hash` **and** the stored identity
  PCA — `load` verifies all hashes; a `residualize`d `X` is a different coordinate system.
- **Generation** can go off-manifold — herd from real candidates + Pgen filter bound (don't eliminate) the
  failure rate; junction-only decoding means V/J are germline-sampled, not jointly learned.
