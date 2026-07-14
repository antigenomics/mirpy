# TCREMP embedding — theory & experiments

The prototype embedding rests on one claim: **Euclidean distance in embedding space
approximates pairwise alignment distance between receptors.** This note states the
propositions, points at the in-repo experiments that validate them, and records the
numbers reproduced with the v3 pipeline. The **manuscript is the source of truth for
proofs** (Kremlyakova et al., *JMB* 437 (2025) 169205, and its supplementary revision 2);
this file summarizes and links.

Notation: `s_ij` = alignment similarity of receptors i, j; `d_ij = s_ii + s_jj − 2 s_ij`
the dissimilarity (a valid (semi)metric); `φ(i)_k = d(i, p_k)` the embedding coordinate
against prototype k; `D_ij = ‖φ(i) − φ(j)‖₂` the embedding-space Euclidean distance.

## Propositions

- **T1 — embedding distance ≈ alignment distance** (the core claim). With coordinates
  `φ(i)_k = d(i, p_k)` over prototypes `p_k ~ P_gen`, `‖φ(i)−φ(j)‖ ≤ √K · d(i,j)` (Lipschitz
  upper bound from the triangle inequality); a lower bound holds probabilistically under
  prototype coverage. *Empirically:* supplementary **S2**, `Pearson(D_ij, d_ij)`.
- **T2 — optimal gap placement & substitution matrix.** The best-of-contiguous-gap-block
  junction score (placements `(3,4,-4,-3)`) approximates Smith–Waterman; `d_ij ≥ 0` requires
  the substitution matrix to be conditionally negative definite (seqtree's `blosum62()` is the
  Gram transform `s_aa+s_bb−2s_ab`, clamped ≥ 0). Implemented via `seqtree.gapblock`.
- **T3 — PCA de-redundancy.** The V/J score blocks have rank ≤ #distinct genes ≪ K, so the
  embedding lives near a low-rank subspace; `StandardScaler → PCA` recovers a compact basis
  (`mir.embedding.pca`, `n_components=50`).
- **T4 — distribution laws.** `d_ij ~ Gamma` (right-skew that a Gaussian misses); nearest-in
  embedding-space `D_ij ~` generalized extreme value / Fréchet (min-over-K extreme-value limit).
  *Empirically:* supplementary **S1**.
- **T5 — SHM / IGH** (Part 2). Somatic hypermutation as a perturbation bounds embedding drift by
  mutation load: `D_k = ‖φ(k-mutated) − φ(x)‖` is ~linear/sublinear in `k` (`mir.bench.theory.
  shm_embedding_drift`; linear-R² 0.97–0.99). IGH's longer CDR3 gives the *lowest* per-mutation
  drift (104 vs TRB 128) — the embedding is robust to SHM. IGH's hard *reconstruction* is instead
  over-compaction: on arda coords the 95% code (95 PCs) gives exact-match 0.115, but 99% (422 PCs)
  gives 0.356 (> the old 0.152; real IGH prototypes reconstruct better) — so variance retention
  should be chain-adaptive, not the frame
  (only 0.1% of IGH CDR3s exceed the length-40 frame). `experiments/benchmark_igh_shm.py`.
- **T6 — density space** (Part 2, `mir.density`). Enrichment as observed-density ÷
  `P_gen`-pushforward-density in embedding space, `E(z) = f_obs(z)/f_gen(z)` with `f_gen = φ_# P_gen`;
  graph neighbour-enrichment (TCRNET/ALICE) is the `r→0` limit of this density ratio — the basis for
  continuous, graph-free background subtraction. *Empirically* (`mir.bench.theory.tcrnet_convergence`,
  real TRB repertoire vs vdjtools P_gen background, `n=1000`): the Spearman correlation between the continuous
  radius-`r` neighbour count and the discrete Hamming-1 count **falls monotonically as `r` grows** past
  one substitution — ρ ≈ 0.37 (½·r₁) → 0.34 (r₁) → 0.11 (2·r₁) → −0.05 (3·r₁), where r₁ is the median
  one-substitution embedding drift. So the embedding test reproduces the graph test at the
  one-substitution scale and generalizes it continuously beyond. The same enrichment test with a
  supplied control repertoire (binomial) recovers TCRNET; with a generated `P_gen` background
  (Poisson) it recovers ALICE. Benchmarks (`experiments/benchmark_density_{yfv,ankspond,tcrnet}.py`,
  full repertoires from HF `isalgo/airr_{yfv19,ankspond,benchmark}`): **YFV** — day-15 vs day-0
  LLWNGPMAV(A*02) enriched hits **63 > 35** (vaccine response, ALICE regime); **AS/B27** — the public
  CASSVGL[YF]STDTQYF/TRBV9/TRBJ2-3 motif appears among enriched B27+ synovial CD8 hits (**9**) and is
  **absent** in B27− (0); **TCRNET vs ALICE** — a real control and a generated P_gen background agree
  on the enriched clones at **Jaccard 0.86** (chance ≈ 0.30), i.e. ALICE is a special case of TCRNET.
  *Lesson:* real repertoires are pervasively convergent, so P_gen enrichment flags ~40% of clones — a
  **biological control** (differential) or an antigen-reference match supplies the specificity, and the
  full repertoire must be used (subsampling dilutes the sparse antigen clusters).
  **Full theory — appendix §T.6** (`appendix/tcremp_theory.tex`, 8 subsections T.6.1–T.6.8 with proofs):
  - *Why P_gen over-flags (~40%)*: the observed law is a selection reweighting `π_obs = P_gen·Q/Z_Q`
    (convergent recombination + thymic selection, "rich get richer"), so `E = q/Z_Q` and the **size-biased
    bulk fold `E_obs[E] = 1 + CV²(q) > 1`** — the ~40% is null *miscalibration*, not a 40% non-null fraction.
  - *Three estimators of the one ratio* `E`: **balloon** (local Poisson, exact test), **RuLSIF** (bounded
    relative ratio `r_α ≤ 1/α`, valley-stable), **classifier/flow** (NCE logit `= log E + log(N/M)` — the
    shared offset is *why* ALICE≡TCRNET, Jaccard 0.86); direct estimation beats plug-in at rate
    `n^{-1/(2+2γ)}`.
  - *Exact local test*: `μ₀ = (N−1)(n_bg+1)/(M+1)`, Poisson upper tail (ALICE) or conditional binomial
    `Binom(T, N/(N+M))` (TCRNET); **ALICE is the rare-event limit of TCRNET** (`d_TV ≤ μ₀·p_bg`, Chen–Stein).
  - *Water level = **Efron empirical-null median recentering*** (NOT Storey π₀): `c = max(median(n_obs)/
    median(μ₀), 1)`, robust (breakdown ½) since signal <5% barely moves the median — the DESeq median-of-ratios
    size factor. Then BH under PRDS.
  - *Differential control cancels convergence pointwise*: `R = f_case/f_ctrl = (1+a₁)/(1+a₀) ⊥ q` (no
    water-level offset); control-absent motifs give `R→∞` (grounds B27 9-vs-0, YFV 63>35). Prefer a
    biological control over P_gen whenever one exists.
  - *Depth*: subsampling is Poisson thinning; detection floor `w_e^min = c/D` ⇒ process the **full**
    repertoire. Ridges = density level sets → DBSCAN(ε=r₁) + per-ridge binomial epitope test.
  *Parameter logic (way of action, all derived):* `λ₀∈[1,5]` default 3 (min-detectable `E* = 1 + c(q)/√λ₀`,
  `c(q)=Φ⁻¹(1−q)`); `M ≥ 5N`; PCA per-chain preset (~95% var); radius `= r₁` (median one-substitution drift);
  FDR `q* = 0.05`; Efron median recenter for P_gen, none for a differential control.

- **Codec losslessness / invertibility** (Part 2, `mir.bench.theory.codec_losslessness`; appendix
  §T.8 `sec:losslessness`). Three measurable levels — *geometric* (T1 distance preservation),
  *informational* (`exact_ceiling = 1 − collision_rate`, decoder-independent), *reconstructive*
  (decoder exact-match). On real held-out TRB the code is **injective** (collision_rate 0 ⇒ ceiling
  100% at every K/PC), so every missing exact match is decoder/data-limited, none information-limited.
  Exact-match is a rate–distortion curve that saturates by ~99% var (m≈300 PCs / K≈2000; deeper is a
  wash, K=10000 regresses) and is driven by *training data* — n 20k→50k→100k ⇒ 0.885→0.941→0.958,
  crossing 95% on data alone, same one-shot decoder. The code is a ~10 kbit *expansion* of a ~63-bit
  junction, so it is not a compressor — store the string (+ exact V/J/C) for archival recovery; the
  codec earns its keep for ML/generation. Injectivity is also the linkage hazard of the privacy
  section (same property, opposite sign). `experiments/benchmark_lossless_{depth,kpc,codec_losslessness}.py`.

- **T7 — sample-level (repertoire) embedding** (v3.x, `mir.repertoire` + `mir.ml.set_encoder`; theory
  **appendix §T.7** `sec:sample`). A whole repertoire is the weighted empirical measure
  `ρ_S = Σ_σ w_σ δ_{φ(σ)}` on embedding space (weights `w_σ ∝ g(a_σ)`, the concave VST of T6.9). Its
  fixed-vector embedding `Φ(S)` is a sketch of that measure = "the first two moments of `ψ(φ)` plus a
  coverage-standardized diversity profile", three blocks each owning one requirement:
  - *(a) order-invariance + (b) depth-robustness* — the **RFF kernel mean embedding**
    `Φ₁ = Σ_σ w_σ ψ(φ(σ)) = μ_{ρ_S}`; converges to the population mean map at rate `n_eff^{-1/2}` with
    `n_eff = (Σ w²)⁻¹`, so depth-robustness is set by the sample's spread, not its raw depth. Distance =
    **MMD**. **Codebook-free** — the `K→∞` soft-assignment limit of the global-graph→cluster→histogram
    recipe (VLAD/Fisher are its finite-K truncations), so there is no `K` and no clustering rule to choose.
  - *(c) diversity* — coverage-standardized **Hill profile** `{⁰D,¹D,²D}` at common coverage `Ĉ*` (via
    `vdjtools.stats.inext`). Depth-robustness and diversity are **mathematically antagonistic** (diversity
    lives in the depth-sensitive rare tail); coverage standardization is the ecology result that reconciles
    them, and `n_eff = ^qD` is itself a Hill number — one relation ties (b) to (c). `¹D` tracks the
    age-related decline. Bonus: `‖μ_{ρ_S}‖²` is already an order-2 *similarity-sensitive* diversity
    (Rao/Leinster–Cobbold), so a φ-aware diversity is the squared norm of the same backbone.
  - *(d) HLA-linked interactions* — the compressed **second moment** `Σ_σ w_σ ψψᵀ` (codebook-free Fisher
    vector), co-equal with a learned Set-Transformer/DeepRC attention head (`mir.ml`, torch). CMV clusters
    are HLA-restricted, so CMV⁺ samples are close only within an **HLA-matched stratum** ⇒ use
    HLA-stratified MMD.
  - *(e) decoupling nuisances* — `Φ_obs = Φ_bio + ε_depth + δ_batch`: **depth** is estimation variance
    `O(n_eff^{-1/2})` (handled by frequencies + coverage standardization); **batch** is a shared shift that
    **cancels in a within-batch contrast** (the sample-level image of the T6 differential control) — else
    residualize on batch / stratify MMD / normalize by the P_gen pushforward. Always compare *contrasts*,
    not raw positions. (Variable sample length is a non-issue: the measure `ρ_S` is fixed-dimensional whatever
    `|S|` is; cardinality re-enters only as richness `⁰D`.)
  - *(f) RNA-seq normalization course of action* (appendix §T.7.9 `sec:samp-norm`, Table `tab:norm`) — the
    raw count factorizes `a_σ ≈ g(σ)·R·θ·c·p_σ` (kit gain `g(σ)`, library size `R`, T-cell fraction/**infiltration**
    `θ`, per-cell expression `c`, true frequency `p_σ`). The organizing fact: **σ-independent scalars (`R`, `θ`, `c`)
    cancel under frequency normalization** (`prop:freqquotient`) — depth *and* infiltration vanish for free — while the
    **σ-dependent kit gain `g(σ)` is the only surviving multiplicative nuisance**. So nuisances sort into three fates:
    (1) **depth** — quotiented by frequencies, coverage-standardize diversity, never down-sample to common *depth*;
    (2) **infiltration `θ`** — the tissue confound: `N = R·θ·c` conflates depth and infiltration, but `θ` is
    identifiable from the **TCR read fraction** `N_TCR/N_total ∝ θ` (divides out `R`) — **carry it as an explicit
    scalar channel, never normalize it away** (down-sampling erases the prognostic TIL signal); `Φ` on frequencies is
    orthogonal to `θ` (`prop:infiltration`); irreducible only in the joint low-`R`, low-`θ` coverage-limited regime;
    (3) **kit/read-length** — σ-dependent *shape* distortion (V/J-usage bias + long-CDR3 length-censoring, MNAR,
    `prop:lengthcensor`), corrected as a **subspace projection on an anchor** (technical replicate / reference sample =
    ComBat/Harmony; P_gen-predicted V/J usage = RUV negative-controls) sparing the CDR3-motif directions, or
    restrict-to-common-length / inverse-recovery reweight. One line: *scalars quotiented by frequencies, shape
    distortions projected out on an anchor, infiltration kept as its own channel.* **Statistical machinery**
    (`rem:cmh`, refined from the BostonGene receptor-GNN whitepaper): depth acts through a per-clonotype
    **detection probability** `≈1-(1-f)^{s/s̄}` (a censoring/exponent, the sample-level face of the T6 exact
    point-process test) — so naive inverse-depth weighting `w∝1/s` has *no calibrated null* while the
    Poisson-binomial detection model does; for covariates (age/HLA/CMV) that move marginal frequencies, **stratify
    not regress** — bin by `(⌊log₁₀s⌋ × covariate band)` and use a **Cochran–Mantel–Haenszel** conditional test
    (Mantel & Haenszel 1959), degrading to the exact test in small strata. Regime tiering: **blood** depth ≈
    technical (light adjustment); **tumor/tissue** depth folds in TIL fraction (10⁴-fold range → separate `θ`,
    keep as channel).
  *Way of action (all derived — appendix Table `tab:sample`):* frequencies not counts (scale-free); concave
  `g=log1p`/Anscombe (Zipf-robust); RFF length-scale `= r₁` (one-substitution); coverage- not
  depth-standardized diversity; MMD, HLA-stratified for antigen specificity. *Central use case:* low-coverage
  bulk RNA-seq (all chains, `10²–10⁴` clonotypes/chain, 100–200k clinically-annotated samples). Benchmarks
  (`REPERTOIRE_EMBEDDING.md`): age regression (`aging` full-depth, not `aging_lite`), CMV/HLA-stratified
  (`airr_hip` = Emerson 2017), depth-robustness (`downsample` + `aging_lite`). **Build spec:**
  `REPERTOIRE_EMBEDDING.md`.
  *Reproduced (2026-07-14; `mir.repertoire`, `mir.ml.set_encoder`, `experiments/benchmark_repertoire_*.py`;
  TRB, per-sample downsampled to the RNA-seq regime. Numbers are repeated-50-fold-CV mean±std unless noted.
  **These findings were adversarially verified — two initial over-claims were caught and corrected below.**):*
  - **Depth-robustness (`prop:kme`) — confirmed, but generic.** `‖Φ₁(sub)−Φ₁(full)‖` vs `n_eff` has log–log
    slope **−0.55** (theory −0.5), `err·√n_eff` 1.00→0.80 across `N∈{100…10⁴}`. ⚠ *Honest reading:* this is the
    **Monte-Carlo concentration rate of any weighted mean of bounded features** (`Φ₁=Σwψ`, so `err²≈Σw²·V` and
    the x-axis `n_eff=(Σw²)⁻¹` is built from the same weights) — it validates the KME/MMD estimator, **not** the
    TCREMB coordinate system specifically (a scrambled embedding passes too); the 20% `err·√n_eff` drift is the
    only embedding-dependent content.
  - **Age & CMV are clone-size (diversity) phenomena.** A coverage/Hill diversity summary dominates: age
    |Spearman| **0.76** (diversity, n=79) vs 0.58 (kernel mean) vs 0.28 (k-mer); CMV AUC **0.83±0.05**
    (diversity, n=240) vs 0.59–0.63 (Φ blocks) vs 0.49 (learned). For CMV this is **not an age confound** — under
    decade age-matching, age-only AUC is 0.45 (chance) while diversity is 0.83: **CMV memory inflation reshapes
    the clone-size distribution** (fewer, larger clones), the signal a diversity profile reads and which Φ₁'s
    depth-robust concave weighting down-weights *by design*. So "the embedding adds nothing over diversity here"
    is partly **by construction** (Φ₁ discards clone size) — diversity is the natural sufficient statistic for a
    clone-size phenotype, not a defeat of the embedding.
  - **HLA-A\*02 — clonotype identity beats diversity, established at scale (`prop:interact`).** A pure identity
    signal (public A\*02-restricted clones; HLA leaves diversity unchanged): diversity is at **chance
    (0.46±0.05, as predicted)**, while the **second-moment co-occurrence block separates** at **0.623±0.048**
    (n=500 donors, 25k reads, `n_rff_second=256`, repeated 50-fold CV — the intervals clear each other). ⚠
    *Scale + depth + resolution were required*: an early single 70/30 split reported an inflated 0.64 (noise,
    n_test≈30), the n=240 CV gave a borderline 0.535±0.08 (overlapping), and only at n=500/25k does the
    second-moment CI separate cleanly from chance. The kernel-mean Φ₁ (0.58) and k-mer (0.54) also rise above
    diversity, but the second moment is strongest — the HLA signal lives in clonotype **co-occurrence**, exactly
    where the clone-size distribution is blind. Grounded in DeWitt et al. 2018 (*eLife* 7:e38358: TCR occurrence
    patterns encode HLA on this same Emerson HIP cohort).
  - **Finding motifs (`prop:witness`).** `class_witness` (`w=μ_A−μ_B`, score `s(σ)=⟨w,ψ(φ(σ))⟩`) surfaces
    coherent A\*02-associated `CASS…EQYF` clones (TRBV4/6/7); the injected-motif unit test recovers a planted
    public clone. On real YF data (`benchmark_repertoire_yfv.py`, day-15 vs day-0) it ranks LLWNGPMAV/A\*02 clones
    at mean AUC **0.57** (vs naive fold-change 0.53) — *marginal* (n=1 sample/group per donor, ≤4 donors, no CI);
    the witness is essentially a kernel-smoothed fold-change, so the **density-ratio** recovers convergent
    clusters far better.
  - **Spike-in recovery from RNA-seq depth (corrected)** (`benchmark_repertoire_spikein.py`; VDJdb ground truth):
    plant a real VDJdb epitope's **CDR3-Hamming-selected motif family** (NLVPMVATV/A\*02 CMV, clonally expanded)
    into naive P_gen backgrounds of depth `N` and recover it with `mir.density`. ⚠ *The selection metric (Hamming)
    is independent of the BLOSUM-gapblock detection embedding — a cross-metric test; an earlier version that
    selected the core in the same embedding it detected in was circular (72%→50% once de-circularised).* Honest
    result: **recall ~35–50% at RNA-seq depth (N≤3k), FPR ~1.2%**; breadth dilutes at bulk depth but the
    abundance/clonal-depth channel holds it to ~25% at N=10k. Caveat: FPR is vs a *clean* P_gen null; a real
    repertoire has its own convergent clusters, so a **biological differential control** (T6) is the honest
    false-positive test. Two robust lessons: antigen specificity ≠ sequence convergence (spike a real motif
    family, not a diffuse epitope sample → 0% recall), and shallow depth is *favorable* for a fixed response.
  - **Aging = clonality, not an independent divergence axis (unbiased MMD).** Repertoires *do* grow more
    dissimilar with age at depth (exact-overlap `−logF` divergence vs age ρ **0.70** at 500k, vs 0.24 at 250k —
    the signal lives in deep private expansions, invisible at the old 40k), but it is a **re-expression of the
    clonality/diversity decline**, not a directional axis the embedding adds. ⚠ *The raw pairwise KME-MMD used
    the biased V-statistic, whose `1/n_eff` self-term inflates distances for low-diversity (old) samples — an
    artifact masquerading as signal. `mmd_matrix(unbiased=True)` removes it (diagonal-removed MMD², Gretton
    2012); the diversity↔divergence coupling drops from biased −0.15 to unbiased −0.05.* Metric-family sign
    check: unbiased KME ≈0 (diverse≈naive≈shared P_gen baseline), frequency-weighted overlap **−0.68** (old
    private expansions destroy F-overlap — the "more diverse ⇒ less overlap" intuition holds only for exact
    overlap, and even then abundance-weighting flips its age-sign via clonal expansion). The age-divergence
    *independent of scalar diversity* is n.s. across 40k/250k/500k (partial ρ(age, divergence|¹D) ≈ **0.07**,
    p≈0.6). Net: deeper depth *reveals* the divergence, its content *is* the diversity decline.
    (`benchmark_repertoire_agediverge.py`, `aging` full cohort at native/500k depth.)
  - **Batch effects & correction (`prop:batch`) — clean validation on 9 real batches.** On `airr_covid19`
    (Vlasova 2026; 9 FMBA sequencing runs), Φ strongly encodes batch (one-vs-rest AUC **0.78**) and
    **residualizing Φ on the batch indicator cancels it** (→ **0.03**, chance). Natural experiment: **HLA ⟂
    batch** (donor genetics) so its signal **survives** (A\*02 0.60→0.61), while **COVID status ⟂̸ batch** (some
    runs are ~all-healthy) so its naive AUC **collapses to the honest within-batch value** (0.66→0.41
    residualized, 0.54 within-mixed-batch) — the naive number rode the batch confound. MMD decomposition:
    same-status-cross-batch (offset) ≈ cross-status-within-batch (biology), ratio **1.05**. **Cookbook:**
    *detect* (batch OvR AUC) → *quantify* (MMD offset:biology ratio) → *correct* (residualize / within-batch
    stratify) → *verify* (batch→chance, batch-⟂ signal preserved). (`benchmark_repertoire_covidbatch.py`.)
  - **HLA imprint across loci and both MHC classes; TRA > TRB (`prop:interact`, DeWitt 2018).** On the
    4-digit-typed `airr_covid19`, second-moment beats diversity in direction for **15/17** class-I+II alleles,
    **class II present in 8/9** (a new extension beyond class-I-only airr_hip) — **DRB1\*07:01 0.758** (Δ+0.20)
    strongest. Using the **paired** cohort (α+β, 1258/1258), **TRA carries the stronger HLA imprint** (α vs β:
    A\*02 0.64/0.54, B\*07 0.70/0.65, **DRB1\*15 0.81/0.75**) — TRA's lower junctional diversity makes
    HLA-restricted public α clones more shared; α+β concat sits *between* the chains (noisier β dilutes), so
    **use α, not paired, for HLA**. (`benchmark_repertoire_covidhla.py`, `benchmark_repertoire_covidpaired.py`.)
  - **COVID-exposure biomarker — honest negative at RNA-seq depth.** A convalescent (long-past) SARS-CoV-2
    exposure leaves **no batch-robust bulk clonotype-identity signal**: COVID⁺-vs-healthy classification is naive
    0.67–0.70 but **chance after batch correction** (0.49–0.52 across β/α/paired), and the from-scratch MMD
    witness does **not** rediscover the paper's COVID-associated clones (AUC **0.45 α / 0.37 β**, below chance) —
    the ground truth is **87% α** (4393 vs 567), so a β-only test is doubly wrong. Paired α+β lifts the *naive*
    AUC (0.67→0.70) but not the corrected one. *Reading:* the paper's biomarkers need the supervised clone panel
    as a targeted burden (or deeper/larger cohorts), not an unsupervised bulk contrast — memory-phase antigen
    clones are too rare in 20k-read bulk to surface without a differential control.
    (`benchmark_repertoire_covidstatus.py`, `benchmark_repertoire_covidpaired.py`.)
  *Lesson (verified):* two complementary regimes. **Clone-size phenotypes (age, CMV)** are diversity's turf —
  the embedding discards clone size by design, so a Hill/coverage summary wins (and CMV's clonality is real
  biology, not an age confound). **Clonotype-identity phenotypes (HLA)** are the embedding's turf — diversity is
  at chance while the **second-moment co-occurrence block separates at scale (A\*02: 0.62 vs 0.46, n=500)**, and
  the supervised witness / density recover the underlying public motifs (strongest in **TRA** and **class II** —
  DRB1\*07:01 0.76, DRB1\*15 α 0.81 — a novel extension beyond class-I β work). The first-moment kernel mean's own
  CI-backed value is depth-robustness (a generic KME property) + being a fixed fusion modality. Net: *diversity
  for how-even, the embedding for which-clones.* Two cross-cutting cautions, both from `airr_covid19`: **(1)
  batch is a first-order nuisance** — it can be as large as the biology (MMD ratio 1.05) and *inflates any
  batch-confounded contrast* (naive COVID 0.66→0.41 corrected), so always detect it (OvR AUC) and compare
  *within-batch / residualized* contrasts (`prop:batch`); **(2) honest negatives** — a long-past (convalescent)
  antigen exposure leaves no batch-robust bulk biomarker at RNA-seq depth (COVID status = chance after
  correction; the paper's clones need a supervised targeted burden, not an unsupervised bulk witness). Also:
  pairwise repertoire MMD must use the **unbiased** estimator (`mmd_matrix(unbiased=True)`) when samples differ
  in depth/diversity — the biased V-statistic's `1/n_eff` self-term otherwise fakes a divergence signal.

## Reproduced numbers (v3 pipeline)

Run `python experiments/reproduce_supplementary.py` (S1–S3) and
`python experiments/benchmark_vdjdb.py` (Table S1). `n = 3000` CDR3β from the bundled
`human_TRB` prototypes; SW = the paper's Smith-Waterman/BLOSUM62 metric, gapblock = the v3
pipeline metric.

Prototypes are **arda-native** (2026-07): arda-annotated real repertoires (`isalgo/airr_model_read`
functional reads → `arda rnaseq map`) over arda-baked germline distances — one IMGT allele frame
shared with query data.

| Claim | Paper | mirpy SW | mirpy gapblock (v3) |
|---|---|---|---|
| **S2 / T1** `Pearson(D_ij, d_ij)` | 0.56 | **0.575** | 0.404 |
| **S1 / T4** `d_ij` law | Gamma > Normal | Gamma ≈ Normal (AIC tie) | Gamma ≈ Normal (AIC tie) |
| **S1 / T4** `D_ij` law | GEV/Fréchet ≫ Normal, ξ=+0.11 | **GEV wins** (KS .021 vs .078), ξ≈−0.03 | **GEV wins** (KS .017 vs .082), ξ≈0.00 |
| **S3** real vs model prototypes | 0.96 | — | **0.940** |
| **Table S1** VDJdb TRB antigen clustering | mean F1 91%, retention 18% | — | mean F1 81%, retention 17% |

The core claims reproduce on the arda-native coordinate system: the embedding distance tracks
alignment distance (T1: SW 0.575 ≈ paper 0.56), `D_ij` is extreme-value distributed (T4), and the
prototype *source* barely matters (S3, 0.940 ≈ 0.96). `d_ij` is now a Gamma/Normal near-tie (the
real-repertoire junction-length distribution is tighter than the previous set). The v3 embedding is
a deliberately new, versioned coordinate system; switching prototypes to arda-annotated real
repertoires (from the earlier real/OLGA mix) shifts the exact numbers but preserves every law.
