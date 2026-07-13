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

- **T7 — sample-level (repertoire) embedding** (v3.x, `mir.repertoire` forthcoming; theory
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
