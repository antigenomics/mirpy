# Sample-level embedding — lessons for the appendix (§T.7)

Directions for updating `appendix/tcremp_theory.tex` §T.7 (`sec:sample`) with the 2026-07-14 empirical
findings. Each item names the proposition to add/amend, the one-line claim, and the validating script +
recorded number (see `BENCHMARKS.md`, `THEORY.md` T7). Written as directions — the appendix prose is yours.

## 1. Unbiased MMD is mandatory when depth/diversity varies (amend the MMD definition)

- **Claim.** The biased V-statistic `‖μ̂_a−μ̂_b‖²` carries a self-term bias `≈ 1/n_eff` (the `k(z,z)` diagonal).
  For samples of unequal effective size this bias **dominates** and, worse, correlates with the phenotype
  (older = lower `n_eff`), so it *manufactures* a spurious signal. Use the **diagonal-removed MMD²**
  (Gretton et al. 2012): `⟨μ̂,μ̂⟩_unbiased = (‖μ̂‖²−Σw²)/(1−Σw²)` with `Σw² = 1/n_eff`.
- **Where.** Amend the `Φ` / MMD `\begin{definition}` to state both estimators and mandate the unbiased one for
  cross-depth comparison. Add a one-line remark: *biased MMD is admissible only when `n_eff` is matched across
  samples (e.g. a fixed downsample).*
- **Cite.** Gretton, Borgwardt, Rasch, Schölkopf, Smola, *A Kernel Two-Sample Test*, JMLR 13 (2012) 723–773.
- **Evidence.** `benchmark_repertoire_agediverge.py`: diversity↔divergence coupling biased −0.15 → unbiased −0.05.

## 2. Aging divergence is a re-expression of the clonality decline, not an independent axis

- **Claim (negative result worth stating).** Repertoires grow more mutually dissimilar with age at depth
  (exact-overlap `−logF` vs age ρ≈0.70 at 500k, invisible at 40k — a **deep-repertoire** phenomenon in private
  clonal expansions), but the component **independent of scalar diversity** is null: partial
  ρ(age, divergence | ¹D) ≈ 0.07 (n.s.) across 40k/250k/500k.
- **Metric-family remark.** Sign of ρ(diversity, divergence) depends on the metric: unbiased KME ≈0
  (diverse≈naive≈shared P_gen baseline), frequency-weighted overlap **−0.68** (clonal expansion destroys
  F-overlap). The naive "more diverse ⇒ less overlap" holds only for *unweighted richness* overlap; abundance
  weighting flips its age-sign. Good caution against reading a single overlap number.
- **Where.** A short `\begin{remark}` after the depth-robustness proposition. Frame as: the embedding does not
  add a directional aging axis beyond diversity — consistent with the "diversity for how-even" thesis.
- **Evidence.** `benchmark_repertoire_agediverge.py` (`aging` full cohort, native/500k depth).

## 3. Batch is a first-order nuisance that cancels within-batch (new `prop:batch`, promote to a theorem+recipe)

- **Claim.** A sequencing batch shifts a whole cohort-subset by a common offset in `Φ`; it is therefore removed
  by **residualizing `Φ` on the batch indicator** (or comparing within-batch contrasts). Two consequences,
  both validated on 9 real FMBA batches: a batch classifier on `Φ` is strong (OvR AUC 0.78) and collapses to
  chance (0.03) after residualization; a **batch-orthogonal** biological signal (HLA) is preserved (0.60→0.61),
  while a **batch-confounded** one (COVID status, some runs all-healthy) collapses to its honest within-batch
  value (0.66→0.41).
- **Cookbook (add as an algorithm/COA box).** *detect* (batch one-vs-rest AUC) → *quantify* (MMD ratio
  same-status-cross-batch : cross-status-within-batch; ≈1.05 here = batch as large as biology) → *correct*
  (residualize `Φ` on batch, or within-batch/stratified MMD) → *verify* (batch AUC → chance; a known
  batch-⟂ signal is preserved).
- **Where.** New subsection after `prop:hla`. This is the strongest new proposition — it is a clean, general
  method, not COVID-specific.
- **Evidence.** `benchmark_repertoire_covidbatch.py` (airr_covid19).

## 4. HLA imprint spans class I *and* class II, and is stronger in TRA (extend `prop:interact`/`prop:hla`)

- **Claim.** On 4-digit-typed data the second-moment (co-occurrence) block beats diversity for HLA carriage
  across loci and **both MHC classes** (15/17 alleles directional; class II 8/9; DRB1\*07:01 AUC 0.758) — a new
  extension beyond the class-I β-chain literature (DeWitt 2018). On the **paired** cohort, **TRA carries the
  stronger imprint** (DRB1\*15 α 0.81 vs β 0.75): TRA's lower junctional diversity makes HLA-restricted public
  α clones more shared/detectable. α+β concatenation sits *between* the chains (the noisier β dilutes), so the
  recommendation is **α (or α-weighted), not naive concat**, for HLA inference.
- **Where.** Extend the interaction/HLA proposition with a class-II clause and a paired-chain remark. Note the
  chain asymmetry is a testable prediction of the junctional-diversity argument.
- **Evidence.** `benchmark_repertoire_covidhla.py`, `benchmark_repertoire_covidpaired.py`.

## 5. Honest negative — a long-past exposure leaves no batch-robust bulk biomarker (a caution, state it)

- **Claim.** Convalescent (long-resolved) SARS-CoV-2 exposure is **not** detectable from the bulk repertoire at
  RNA-seq depth after batch correction: classification is chance (0.49–0.52 β/α/paired) and a from-scratch MMD
  witness does not rediscover the paper's COVID clones (0.37 β / 0.45 α). The naive 0.66–0.72 was batch
  confound. Detecting rare memory-phase antigen clones needs a **supervised targeted burden** of a known clone
  panel (as the source paper did) or a differential control — not an unsupervised bulk contrast.
- **Where.** A `\begin{remark}` on the limits of the sample-level embedding: it reads *distributional* /
  diversity structure and public-clone co-occurrence, but rare private antigen responses fall below the bulk
  noise floor — deferring to the density/witness machinery (§T.6) with a proper control.
- **Evidence.** `benchmark_repertoire_covidstatus.py`, `benchmark_repertoire_covidpaired.py`.

---

**One-paragraph thesis (unchanged, reinforced):** *diversity for how-even, the embedding for which-clones* —
and, added this round: **compare contrasts, not raw distances** (unbiased MMD + within-batch), the imprint is
richest in **TRA and class II**, and be willing to report **honest negatives** (aging directionality, COVID
convalescent biomarker) where the bulk signal is genuinely diversity or genuinely absent.
