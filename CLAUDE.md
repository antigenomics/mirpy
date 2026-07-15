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
- `distances/junction.py` — `junction_distance_matrix` (seqtree.gapblock). Coordinate knobs (all
  default to the published space, threaded through `TCREmp`): `metric="squared"` (=`d`, default) |
  `"sqrt"` (metric `ρ=√d`, benchmarked a wash — `SQRT_D_MIGRATION.md`); `matrix=` a custom
  `seqtree.SubstitutionMatrix` (pam250/structural/from_similarity); `alignment="gapblock"` (default)
  | `"sw"` (paper-exact Smith-Waterman, lazy BioPython, validation-only). `distances/germline.py`
  — resource-backed V/J/CDR1/CDR2 lookup with allele cascade.
- `embedding/prototypes.py` — bundled prototype loader. `embedding/tcremp.py` — `TCREmp` /
  `PairedTCREmp` (polars frame in → `(N,3K)` float32). `embedding/pca.py` — PCA denoise (T3).
  `embedding/presets.py` — per-chain `n_prototypes` + PC recommendations (`get_preset`);
  `from_defaults(n_prototypes=None)` uses them. Compact chains (IGK/IGL/TRG) 1000 protos / ~20
  PCs; diverse chains (IGH/TR*) 2000 / ~65 PCs (95%), ~220–300 PCs (99%, for codec reconstruction).
- `bench/` — `vdjdb.py` (loader), `metrics.py` (`cluster(method=…)` DBSCAN default | HDBSCAN | OPTICS,
  kneedle eps, F1/retention), `theory.py` (S1–S3 + T5 + T6 `tcrnet_convergence` + `codec_losslessness`).
  Needs `[bench]`. Clustering is a precision/coverage trade-off (`experiments/benchmark_clustering.py`):
  DBSCAN tightest/purest (paper regime), HDBSCAN ~3× coverage at lower F1 (variable-density), OPTICS
  dominated, KMeans no noise-rejection.
- `density.py` — continuous-density TCRNET/ALICE (T6): `fit_density_space` (one shared PCA basis),
  `neighbor_enrichment` (balloon adaptive-radius Poisson/binomial + water-level calibration;
  `backend=` **exact** BallTree default | **kdtree** exact scipy cKDTree 5–9× faster | **ann**
  pynndescent ~30× at ≥1e5, recall<1 conservative — `experiments/benchmark_ann.py`),
  `enriched_mask`, `denoise_and_cluster`, `generate_background` (vdjtools P_gen, lazy). Torch-free
  (scipy/sklearn). Prefer a **biological control** as background (differential) over P_gen.
  **Abundance-aware** (T6 sec:dens-abund): pass `abundance=` (clone sizes) + `weight="log1p"`/`anscombe`
  to swap the distinct in-ball count for the variance-stabilised mass `S=Σg(a_j)` (compound-Poisson
  Gamma tail, dispersion `φ=E[g²]/E[g]`) plus a per-clonotype orphan/depth channel `P(A≥a_j)` Fisher-
  combined with breadth. Default `abundance=None` = distinct count (unchanged, `g≡1`).
- `repertoire.py` — sample-level (repertoire) embedding (T.7): `fit_repertoire_space` (one shared
  clonotype-cloud PCA + RFF basis, prototype-hash-verified `RepertoireSpace`), `sample_embedding`
  (`Φ(S)` = RFF kernel mean ‖ coverage-standardized Hill ‖ second-moment Fisher; `n_eff=(Σw²)⁻¹`),
  `mmd_distance`/`mmd_matrix`/`hla_stratified_mmd`, `class_witness` (supervised MMD motif finder). Torch-free.
  Opt-in `fit_repertoire_space(n_eigs=r)` swaps the second-moment block's full `D₂(D₂+1)/2` upper-triangle
  for its top-`r` eigenvalues (rotation-invariant spectrum; default `None` = upper-tri, unchanged — but
  lossy for the *directional* HLA imprint, so the full triangle stays the recommended block; `benchmark_repertoire_spectral.py`).
- `ml/` — Part 2 (torch), neural codecs + `set_encoder.py` (learned repertoire track: Set-Transformer/DeepRC
  attention pooling + `SetEncoderBundle`).
- `resources/` — `prototypes/` (TSVs + manifest), `gene_library/` (region_annotations.txt),
  `germline_dist/` (baked `.npz`, from `build_germline_dist.py`).

## Build / test / run
- Conda env **`mirpy`** (Python 3.12; do NOT use `.venv` here). `pip install -e .`
  (pure-Python hatchling; no C build). Extras: `[bench] [annotate] [build] [ml] [docs] [dev]`.
- Tests: `python -m pytest tests/ -q` (78 pass; `-m "not integration"` for the ~5s fast tier —
  the pynndescent ANN parity test carries a one-time JIT cost). All self-contained on bundled resources.
- Experiments: `python experiments/reproduce_supplementary.py` (theory S1–S3),
  `python experiments/benchmark_vdjdb.py` (Table S1). Analyses: `analyze_prototype_counts.py`
  (geometry saturates by K≈100 — T.1/S4), `analyze_pc_decomposition.py` (V/J η² ≈0.44/0.49,
  CDR3-length η² 0.13 & R²=0.95; germline low-rank ~13 PCs — T.4). See `THEORY.md`.

## Conventions
- AIRR polars frames in/out, keyed by `vdjtools.io.schema` names (`v_call, j_call, junction_aa,
  locus`). No `Clonotype` class.
- v3 embeddings are a **new, versioned coordinate system** (gapblock ≠ the v2 BioPython junction
  scorer) — any model trained on v2 embeddings must be retrained.
- Baked `germline_dist/*.npz` are versioned artifacts; regenerate whenever the gene library /
  `region_annotations.txt` changes (`build_germline_dist.py`, needs `[build]` BioPython).
- **Parallelism / hardware** (see README "Performance & parallelism"): embedding is all-core by
  default (`TCREmp(threads=0)`, C++ gapblock, GIL-released); density `backend="kdtree"` = exact
  multicore (`workers=-1`), `"ann"` = pynndescent auto-all-core (default `"exact"` BallTree is
  1-core); `cluster(n_jobs=-1)` forwards to sklearn; PCA/RFF ride BLAS (cap via `OMP_NUM_THREADS`).
  GPU only in `mir.ml`: `pick_device()` = **CUDA → MPS → CPU** auto, override `device=`/`MIR_DEVICE`.

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
  - **DONE lossless-recon depth/K/data sweep** (`experiments/benchmark_lossless_{depth,kpc}.py`,
    real held-out TRB from `isalgo/airr_control` vs arda landmarks): exact-match is **training-data-
    limited, not architecture-limited** — the old "~0.40 ceiling" was n≈8k starvation. Same one-shot
    decoder: n=20k→50k→100k drives exact **0.885→0.941→0.958** (token 0.996→0.998) at K=2000/PC=400,
    crossing 95% at n≈100k, peak RSS 11 GB. **Optimal (K,PC)=(2000, 300–500)**: K saturates ~2000–5000
    (K=10000 *regresses* 89.4→88.8 and doubles cost); PC is the stronger lever below ~300 (PC 50→300
    ⇒ exact 0.67→0.89). Cost: K linear (matrix N·K·4 B, ~5 µs/query @K=2000; K=10000 = 32 µs/query,
    0.8 GB @n=20k), PC cheap. Levers to 95%+ exact, in order: **more data (free)** > PC→~99% var >
    K→~2000 > (last %) autoregressive decoder + widen `FIXED_LEN` for the ~0.1% long real-IGH tail.
    Frame is lossless iff `len(junction)≤40` (100% on bundled protos; ~99.9% on real IGH). The
    distance-to-prototypes code is an **expansion** (10 kbit code vs ~63 bit seq) — for archival
    losslessness *store the string*; the codec inverse is for ML/generation. Fine (1-residue) diffs
    survive 99%-var compaction (0/500 collisions, 99.6% variants nearest-to-parent).
  - **C gene / isotype**: absent from the embedding (prototypes are v/j/junction only; germline_dist
    is V/J/CDR1/CDR2). Isotype is a low-cardinality categorical (~9 IGH classes ≈ 3 bits) *independent*
    of V/J/CDR3 ⇒ not reconstructable ⇒ **carry `c_call` as an exact stored column, never embed it**
    (same as v_call/j_call metadata). No codec change.
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
- **Sample-level (repertoire) embedding (v3.x) — DONE** (`mir/repertoire.py` + `mir/ml/set_encoder.py`,
  appendix §T.7 `sec:sample`, `THEORY.md` T7). `Φ(S)` = RFF **kernel mean** (depth-robust `n_eff^{-1/2}`,
  codebook-free) ‖ coverage-standardized **Hill profile** ‖ **second-moment** Fisher; distance = MMD /
  HLA-stratified; `class_witness` = supervised MMD motif finder; learned co-equal Set-Transformer/DeepRC in
  `mir.ml.set_encoder`. Reuse: `TCREmp.embed`, `density.{_WEIGHTS,fit_density_space,calibrate_radius}`,
  `vdjtools.stats.inext`, `preprocess.downsample`, `ml.bundle` hashing. Tests
  `tests/test_{repertoire,set_encoder}.py`; benchmarks `experiments/benchmark_repertoire_{aging,depth,cmvhla,hla}.py`
  (shared `_cohort.py`; `airr_benchmark` aging 79, `airr_hip` Emerson 2017 786). **Key empirical lesson**
  (`THEORY.md` T7; **all adversarially verified — two over-claims caught & corrected**): depth-robustness holds
  (`prop:kme`, slope −0.55) but is a **generic KME/MMD Monte-Carlo rate**, not embedding-specific. **Age & CMV
  are clone-size (diversity) phenomena** — a Hill/coverage summary dominates (CMV AUC **0.83±0.05** 50-fold CV
  n=240 vs Φ blocks 0.59–0.63), real memory-inflation clonality not an age confound (age-matched age-only 0.45);
  Φ₁ discards clone size *by design* so this isn't an embedding defeat. **HLA-A\*02**: diversity 0.45=chance,
  clonotype blocks modestly higher (second-moment **0.535±0.08**) — *direction* favors clonotype-identity but
  **CIs overlap, not decisively separated** (the earlier single-split 0.64 was noise). **Spike-in** (VDJdb
  NLVPMVATV into shallow P_gen): **~50% recall at RNA-seq depth, FPR ~1.2%**, cross-metric (Hamming-select /
  embedding-detect; an earlier same-embedding version was circular, 72%→50%). `class_witness` surfaces coherent
  `CASS…EQYF` motifs; YF real-data witness marginal (0.57). Net: CI-backed value = **depth-robustness + fixed
  modality**; clonotype-identity payoff real-but-weak, in the 2nd moment / witness / density not the first moment.
  Verification workflow: `wf_06465c6e`. Spec: `REPERTOIRE_EMBEDDING.md`. **TODO**: larger HIP cohort/depth to
  establish the weak HLA signal; biological-control FPR for spike-in; HLA-allele panel; epitope/MHC; scale.
- **Sample-level — 2026-07-14 additions** (`airr_covid19` = Vlasova 2026, paired TRB+TRA, 9 real batches,
  4-digit HLA class I+II; `_covid.py` local loader + `benchmark_repertoire_{covidbatch,covidhla,covidstatus,
  covidpaired}.py`; `THEORY.md` T7):
  - **Unbiased MMD** (`mir/repertoire.py` `mmd_distance/mmd_matrix(unbiased=True)`): biased V-stat's `1/n_eff`
    self-term inflates low-diversity samples → fakes divergence. **Age-divergence** re-tested deep (500k, not
    the misleading 40k): real & strong (overlap ρ0.70) but **diversity-coupled, not an independent axis**
    (partial ρ(age,div|¹D)≈0.07 n.s.); overlap-F sign is −0.68 (clonal expansion, not richness).
  - **Batch cookbook (`prop:batch`) — PASS**: batch OvR AUC 0.78 → **0.03** after residualization; HLA (⟂batch)
    survives, COVID status (⟂̸batch) collapses 0.66→0.41 (naive rode the confound). detect→quantify→correct→verify.
  - **HLA imprint**: 15/17 alleles class I+II, class II 8/9 (DRB1\*07:01 0.76 top). **TRA > TRB** for HLA
    (DRB1\*15 α 0.81 vs β 0.75); paired concat dilutes (noisier β) → use α for HLA.
  - **COVID biomarker = honest negative**: chance after batch correction (0.49–0.52); witness doesn't rediscover
    the paper's clones (0.37β/0.45α, GT is 87% α). Long-past exposure has no batch-robust bulk signal at RNA-seq depth.
  - **Parallelism/GPU documented** (README "Performance & parallelism"): `TCREmp(threads=0)` all-core; density
    `backend="kdtree"` multicore exact; `pick_device` **CUDA→MPS→CPU** (was MPS-only) + `MIR_DEVICE`.
- **Sample-level refinements + TCGA — 2026-07-14 (pm)** (`BENCHMARKS.md` "2026-07-14 (pm)"; all numbers recorded):
  - **WS1 spectral block** (`fit_repertoire_space(n_eigs=r)`, `benchmark_repertoire_spectral.py`): opt-in top-`r`
    eigenvalues of the second-moment covariance. **Lossy for HLA** — HLA-A\*02 lives in *which* clones co-occur
    (directional), eigenvalues are rotation-invariant → top-r ≤0.55 while full upper-tri reaches **0.593** (D₂=512).
    Default `None` = upper-tri (unchanged); keep the full triangle. Test `test_spectral_second_block_top_r_eigvals`.
  - **WS2 COVID witness** (`benchmark_repertoire_covidwitness.py`, vdjtools `biomarker.fisher_association`): answers
    "Fisher passes but embedding vanishes — why?" → **breadth, not depth**. Genome-wide Fisher passes 0 clones at
    150–300 donors (any depth 20k/120k); full ~1137-donor cohort passes 39β/4α (user tmp scan). Sample-level lever
    is **batch control** (β witness whole 0.51 → mixed-batch **0.75**); per-allele HLA-stratification adds noise
    (median≈whole). HLA+α+β is not the key — breadth is.
  - **WS2b COVID motif recovery** (`benchmark_repertoire_covidmotif.py`, full 1137-donor incidence Fisher, both
    chains): the *breadth-powered* way to find motifs. **α genome-wide recovers 4 GT-true clones = a coherent
    public family CAG·NYGGSQGNLIF (paper cluster 31)**; β recovers 0 (rarer/HLA-restricted). **HLA restriction
    adds 0** — the recoverable α clones are already public. **WS1 (2026-07-15) closed the α/β loop and REFUTED
    the "β is depth/HLA-limited" guess**: β at full *native* depth genome-wide, and `native_beta` across 12
    class-II/B carrier strata at native depth, both recover **0** GT-β. β's GT clones are short **public
    bystanders** (`CASSx…YEQYF` TRBJ2-7, present in COVID *and* healthy; 253/256 present, median incidence 9),
    so they don't discriminate status by *incidence* — the α/β split is **status-enriched vs public-bystander,
    not depth/breadth/HLA**. (Untested caveat: an abundance/size-based test could still catch β clones enriched
    in *magnitude* not *breadth*.)
  - **WS3 UMAP** (`plot_sample_umap.py {covid,aging,hip}`): faceted UMAP of Φ₁ (=MMD geometry) by age/HLA/batch/
    COVID/CMV → `experiments/figures/`. covid19 visibly clusters by **batch** in raw Φ₁, dissolves after
    `residualize` (the `prop:batch` story made visual).
  - **WS4 TCGA** (`_tcga.py` local-first loader, untars once; `benchmark_repertoire_tcga.py`; `lifelines` in
    `[bench]`): 9591 samples, 7 chains, OS survival. **Tumor-type separation is depth-dependent** — deepest chain
    **IGK 0.67** ≫ shallow TRB 0.52 / IGH 0.50 (IG light chains carry tissue signal). **Survival = honest negative**:
    ΔC-index ≈0 for every chain over a clinical Cox (age+sex+stage+log reads, base C 0.66–0.73) — the tumour-
    infiltrating repertoire adds no prognostic value at RNA-seq depth. No grade column (used stage); PFS empty.
  - **WS4b TCGA survival via biology features** (`_tcga_features.py` isotype/infiltration/atypicality/clonality;
    `benchmark_repertoire_tcga_survival.py`) — the prognosis the *identity embedding* misses lives in interpretable
    axes (modelled on an internal AIRR-tissue EDA). **Infiltration (hot/cold Z-axis) stratifies survival in 5/8
    cancers** (KIRC/SKCM/KIRP/OV/LGG log-rank p<0.05) and adds C-index in melanoma (+0.036) / KIRP (+0.030);
    **IgA isotype** stratifies bladder (mucosal, p=0.010) + melanoma; **atypicality** stratifies glioma (p=0.001)
    — all matching known immunobiology. Isotype/atypicality show via KM stratification, not linear C-index. Net:
    tissue prognosis = infiltration magnitude + isotype + typicality, **not clonotype identity**.
  - **REFRAME → TME-aware repertoire embedding, pan-cancer** (`_tcga_embedding.py` = per-chain identity ‖
    diversity ‖ coverage/infiltration ‖ isotype ‖ composition ‖ atypicality — the biology axes recast as Φ(S)
    channels; `benchmark_repertoire_tcga_{pancancer,tme}.py`; THEORY.md T7 "Repertoire embeddings for the TME",
    BENCHMARKS.md "Repertoire embedding for TME & survival"). One Φ over 9425 OS samples (78-dim, 5 embeddable
    chains + all 7 via composition). **Robustly prognostic** (LR p<0.05 & CV ΔC>0) in **SKCM +0.039 / BLCA
    +0.025 / HNSC +0.022 / LGG +0.016**; effect-size positives SARC (isotype) / KIRP (atypicality). Top channel
    = **coverage/atypicality/composition, ~never identity**; immune-cold cohorts overfit → pan-cancer mean ΔC≈0
    masks the immune-hot wins. **Paradigm lesson (T7 #6): the same Φ(S) that reads infection/HLA in blood via
    its identity channels stratifies the TME + survival in tissue via its non-identity channels.** Unsupervised
    Φ clustering → TME states (`benchmark_repertoire_tcga_tme.py`, UMAP `experiments/figures/umap_tcga_tme`).
  - **Derivable descriptor + in-silico evolution** (`mir.repertoire.sample_descriptor`/`RepertoireDescriptor`/
    `decode_metrics`; `benchmark_repertoire_tcga_insilico.py`; THEORY.md T7 lesson 7): **mass-preserving** smooth
    descriptor — infiltration=log-mass, diversity=log n_eff, clonality=Σw², identity=kernel mean are all smooth
    coordinates (keeps the mass Φ normalises away). The cohort coordinate distribution = a generative manifold;
    perturbing infiltration + conditioning (Gaussian) = **in-silico evolution**: hotter ⇒ diversity +0.84 /
    switch +0.52 / T-vs-B −0.63; CoxPH "make hotter" protective 12/20 cancers, adverse in glioma (LGG +1.19) —
    learned couplings match immunobiology. Test `test_descriptor_metrics_derivable_smooth_and_decodable`.
    **TODO**: per-cancer n_pc; Thorsson immune-subtype validation; a learned (flow/VAE) generative manifold for
    full sequence-level simulation; promote the multi-chain descriptor into `mir.repertoire`.
- Full plan: `~/.claude/plans/i-want-to-completely-crystalline-lake.md`.
