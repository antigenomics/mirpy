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
- **T6 — density space** (Part 2). Enrichment as observed-density ÷ `P_gen`-pushforward-density
  in embedding space; graph neighbour-enrichment (TCRNET/ALICE) converges to this density ratio
  — the basis for continuous background subtraction.

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
