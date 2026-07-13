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
  **Full theory — appendix §T.6** (`appendix/tcremp_theory.tex`): the observed repertoire as a mixture
  `f_obs = (1−π) f_sig + π f_bg` with `f_bg ≈ f_gen` and one "water level" `π` covering both regimes
  (bystander `π≈0.1–0.5`, naive `π≈0.95`); `E(z)` estimated by any of three consistent density-ratio
  estimators — **balloon** (local Poisson, exact significance), **RuLSIF** (bounded relative ratio
  `r_α ≤ 1/α`, valley-stable), **classifier/flow** (NCE logit `= log E + log(N/M)`, scales via the forward
  codec) — with per-point Poisson significance, Storey `π̂₀` as the water level, FDR in place of a hand-set
  threshold, level-set/persistence ridge delineation, and per-ridge binomial epitope enrichment. Ridges are
  **epitope-heterogeneous** (precursor frequency spans ~2 orders across / ~6 within epitopes — Pogorelyy et
  al. *Genome Med* 2018), so no single global cut works and significance is judged locally. *Parameter logic
  (way of action):* neighbourhood scale adaptive to fix expected background occupancy `λ₀∈[1,5]`
  (depth/valley-adaptive; minimum detectable `E* ≳ 1 + c(q)/√λ₀`); background `M ≥ 5N`, P_gen (ALICE) or
  control+`Q` (TCRNET); PCA to the per-chain preset (~95% var); FDR `q*` = 0.05 (0.001 stringent).

## Reproduced numbers (v3 pipeline)

Run `python experiments/reproduce_supplementary.py` (S1–S3) and
`python experiments/benchmark_vdjdb.py` (Table S1). `n = 3000` CDR3β from the bundled
`human_TRB` prototypes; SW = the paper's Smith-Waterman/BLOSUM62 metric, gapblock = the v3
pipeline metric.

| Claim | Paper | mirpy SW | mirpy gapblock (v3) |
|---|---|---|---|
| **S2 / T1** `Pearson(D_ij, d_ij)` | 0.56 | **0.636** | 0.486 |
| **S1 / T4** `d_ij` law | Gamma > Normal | **Gamma wins** (AIC) | **Gamma wins** (AIC) |
| **S1 / T4** `D_ij` law | GEV/Fréchet ≫ Normal, ξ=+0.11 | **GEV wins** (KS .023 vs .080), ξ≈−0.03 | **GEV wins** (KS .018 vs .085), ξ≈+0.01 |
| **S3** real vs model prototypes | 0.96 | — | **0.963** |
| **Table S1** VDJdb TRB antigen clustering | mean F1 91%, retention 18% | — | mean F1 ~80%, **retention 18%** |

The core claims reproduce: the embedding distance tracks alignment distance (T1), `d_ij` is
Gamma and `D_ij` is extreme-value distributed (T4), the prototype *source* barely matters (S3,
0.963 ≈ 0.96), and antigen-cluster retention matches Table S1 exactly. Quantitative gaps
(`D` shape ξ, Table S1 F1) trace to the gapblock-vs-Smith-Waterman metric and a different VDJdb
release — the v3 embedding is deliberately a new, versioned coordinate system.
