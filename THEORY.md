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
  drift (101 vs TRB 138) — the embedding is robust to SHM. IGH's hard *reconstruction* is instead
  over-compaction: the 95% code (68 PCs) gives exact-match 0.009, but 99% (371 PCs) gives 0.152
  (≈ irrm-codec's 0.16) — so variance retention should be chain-adaptive, not the frame
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
| **Table S1** VDJdb TRB antigen clustering | mean F1 91%, retention 18% | — | *(re-run pending on arda coords)* |

The core claims reproduce on the arda-native coordinate system: the embedding distance tracks
alignment distance (T1: SW 0.575 ≈ paper 0.56), `D_ij` is extreme-value distributed (T4), and the
prototype *source* barely matters (S3, 0.940 ≈ 0.96). `d_ij` is now a Gamma/Normal near-tie (the
real-repertoire junction-length distribution is tighter than the previous set). The v3 embedding is
a deliberately new, versioned coordinate system; switching prototypes to arda-annotated real
repertoires (from the earlier real/OLGA mix) shifts the exact numbers but preserves every law.
