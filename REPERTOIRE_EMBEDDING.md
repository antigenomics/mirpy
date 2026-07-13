# Sample-level (repertoire) embedding — build spec

_Self-contained guideline for a separate build session. Theory is in `appendix/tcremp_theory.tex`
§T.7 (`sec:sample`); this doc is the module + benchmark plan. Written 2026-07-14._

## 0. What & why

Embed a whole **repertoire** — an order-invariant multiset of clonotypes `{(x, a_x)}` with clone
counts — into one fixed vector `Φ(S)`, so repertoires can be compared, regressed against phenotype
(age), and classified (CMV/HLA). Four requirements, each owned by one block of `Φ` (appendix §T.7):

| # | requirement | how | appendix |
|---|---|---|---|
| a | permutation-invariant | pool a per-clonotype feature by a size-weighted sum | Prop. `prop:sampinv` |
| b | robust to sequencing depth | frequency-weighted **kernel mean embedding** (RFF); converges `O(n_eff^{-1/2})` | Prop. `prop:kme` |
| c | preserves diversity | coverage-standardized **Hill profile** `{⁰D,¹D,²D}` | Prop. `prop:antag`,`prop:simdiv` |
| d | HLA-linked interactions | **second moment** `Σ_σ w_σ ψψᵀ` (Fisher) or a learned set-network | Prop. `prop:interact`,`prop:hla` |

**Central constraint: depth-robustness.** The real target is low-coverage bulk **RNA-seq TCR** — all
chains, but only `10²–10⁴` clonotypes/chain — of which there are 100–200k clinically-annotated samples
(default in cancer immunotherapy). `Φ` must be usable at ~100 clonotypes/chain. `~10³` receptors already
fingerprint an individual (Immprint).

**The elegant core (answers "which K, which clustering?"): none.** `Φ₁` is the RFF **kernel mean
embedding** `Σ_σ w_σ ψ(φ(x_σ))` — the codebook-free `K→∞` soft-assignment limit of the
global-graph→cluster→histogram heuristic (VLAD/Fisher are its finite-`K` truncations; Prop. `prop:codebook`).
No `K`, no clustering rule.

## 1. Reuse map (do NOT rebuild — confirmed by exploration)

- **Clonotype `φ`:** `mir.embedding.tcremp.TCREmp.embed(df) -> (N, 3K) float32` (cols
  `v_call,j_call,junction_aa`; counts ignored — read them separately). `PairedTCREmp` for multi-chain.
  `mir.embedding.pca.pca_denoise`, `mir.embedding.presets.get_preset`.
- **Count→weight VST (lift directly):** `mir.density._WEIGHTS` = `{"distinct": 1, "log1p": log1p,
  "anscombe": sqrt(a+0.375)}`; `mir.density._emp_survival` (orphan `P(A≥a)`); the compound-Poisson
  dispersion `φ=E[g²]/E[g]` pattern. These already implement the concave weight `g` of §T.6.9.
- **Diversity / coverage (entire block c is reuse):** `vdjtools.stats.inext` — `inext_coverage`,
  `estimate_d(base="coverage", level=C*)`, `asymptotic_diversity` (Chao1/CWJ-Shannon/MVUE-Simpson),
  `sample_coverage` (Good–Turing Ĉ), `rarefaction_batch`; `vdjtools.stats.diversity.diversity_stats`
  (Shannon `expH`, inverse Simpson, Chao1, Efron–Thisted, d50).
- **Depth control:** `vdjtools.preprocess.downsample(df, size, by="reads")` (hypergeometric, NOT Poisson),
  `select_top`.
- **Radius / bandwidth:** `mir.density.calibrate_radius` gives `r₁` (one-substitution embedding drift) —
  use as the RFF kernel length-scale so the kernel matches the sequence metric.
- **Baselines to beat:** `vdjtools.features.kmer_profile`, `physchem_profile`;
  `vdjtools.overlap.{pairwise_distances, similarity_overlap}` (Morisita-Horn, sequence-weighted
  Leinster–Cobbold), `cluster_samples` (MDS/hclust with metadata join).
- **Cohort loading:** `experiments/_hf.py` `fetch`/`load_repertoire` (collapses to
  `junction_aa/v_call/j_call/duplicate_count`, `top=N`); `vdjtools.io.batch.read_metadata` /
  `read_samples` / `iter_samples`; `io.cohort`.

## 2. Module — `mir/repertoire.py` (torch-free core; vdjtools + optional torch lazy)

```python
@dataclass
class RepertoireSpace:            # one shared basis for a cohort (comparability invariant)
    model: TCREmp | PairedTCREmp
    rff: RandomFourierFeatures    # omega, b; length-scale = r1
    pca: PCA | None               # optional pre-RFF PCA on the pooled clonotype cloud

def fit_repertoire_space(model, cohort_df, *, n_rff=2048, length_scale=None,
                         n_components=None, seed=0) -> RepertoireSpace:
    """length_scale defaults to calibrate_radius(...)=r1. Fit RFF (+optional PCA) ONCE on the
       pooled clonotype cloud so every sample lands in ONE coordinate system (Prop. prop:kme)."""

def sample_embedding(space, sample_df, *, weight="log1p", blocks=("mean","diversity","second"),
                     coverage=None) -> np.ndarray:
    """Φ(S): concat of requested blocks. Reads duplicate_count for weights.
       mean:      Φ1 = Σ w_σ ψ(z_σ)                       (block b; kernel mean embedding)
       diversity: coverage-standardized {0D,1D,2D} + Ĉ    (block c; via vdjtools.stats.inext)
       second:    upper-tri of Σ_σ w_σ ψψᵀ (or top-r eigvals)  (block d; Fisher vector)"""

def mmd_distance(phi_a, phi_b) -> float:            # ||Φ1(a) − Φ1(b)||  ≈ MMD (Eq. eq:kme)
def mmd_matrix(embs) -> np.ndarray                  # sample×sample, feed cluster_samples / a regressor

def hla_stratified_mmd(embs, hla) -> np.ndarray:    # Prop. prop:hla: distance only within HLA-matched pairs
```

Design notes (all traced to a theorem — see appendix Table `tab:sample`):
- **Frequencies, not counts** (scale-freeness, Prop. `prop:kme`); concave `g` for the Zipf tail
  (`weight="log1p"`/`"anscombe"`, §T.6.9).
- **One basis per cohort**: fit RFF (+PCA) on the *pooled* clonotype cloud, embed every sample through it
  — same comparability invariant as `mir/ml/bundle.py` and `density.fit_density_space`. Serialize the
  space (RFF `omega,b` + prototype hash) so embeddings are only ever compared within one basis.
- **Length-scale = `r₁`** (`calibrate_radius`) so the kernel resolves ~one substitution.
- **`n_eff = (Σ w²)⁻¹`** is a Hill number (Prop. `prop:antag`): report it per sample; it predicts each
  sample's own depth-robustness and is the natural x-axis for the depth-robustness benchmark.
- **Multi-chain (RNA-seq):** embed each locus with `PairedTCREmp`-style per-locus `TCREmp`, concatenate
  the per-locus `Φ`. Handle missing chains (some RNA-seq samples lack a locus) by zero-blocks + a
  presence mask.
- **Nuisance decoupling (§T.7.8):** depth = estimation variance (frequencies + coverage-standardized
  diversity handle it, `O(n_eff^{-1/2})`); **batch = a shared shift that cancels in a within-batch
  contrast** (Prop. `prop:batch`, the sample-level image of the T6 differential control). So: prefer
  within-batch/paired designs; else residualize `Φ` on batch indicators / stratify MMD by batch (as for
  HLA); or normalize each sample by its `P_gen` pushforward. Always compare **contrasts, not raw
  positions**. Record `batch`/platform per sample in the metadata and expose a `batch=`/`covariates=` arg.
- **RNA-seq normalization (§T.7.9 `sec:samp-norm`, Table `tab:norm`):** the count factorizes
  `a_σ ≈ g(σ)·R·θ·c·p_σ`; **σ-independent scalars (depth `R`, infiltration `θ`, per-cell expr. `c`) cancel
  under frequency weighting** (`prop:freqquotient`), the **σ-dependent kit gain `g(σ)` is the only surviving
  multiplicative nuisance**. Concrete build items for the heterogeneous-cohort / tissue case:
  - **Infiltration channel (`prop:infiltration`):** in tissue, `N = R·θ·c` conflates depth and T-cell content;
    estimate `θ` from the **TCR read fraction** `N_TCR/N_total ∝ θ` (or a deconvolution signature, Newman 2015)
    and **append it as an explicit scalar channel — never down-sample to a common depth** (that erases the
    prognostic TIL signal). `Φ` on frequencies is already orthogonal to `θ`. Expose `infiltration=`/
    `read_fraction=` per-sample and concat to the embedding; flag joint-low-`R`/low-`θ` samples (coverage-limited).
  - **Read-length / kit (`prop:lengthcensor`):** short reads drop long CDR3s (MNAR length-censoring). Either
    restrict to the common recoverable length range across kits, or inverse-recovery-reweight `1/r_L(ℓ)`.
    Chemistry bias + confounded-batch residual live in the V/J-usage+length subspace → **anchored subspace
    projection** (technical replicate / reference sample = ComBat/Harmony; `P_gen`-predicted V/J usage = RUV
    negative-controls), sparing the CDR3-motif directions. Expose `read_length=`/kit per sample.

## 3. Two co-equal tracks (the learned head is `mir.ml`, torch, `[ml]` extra)

The unsupervised backbone above and a **learned permutation-invariant set network** are co-equal,
benchmarked head-to-head (§T.7.5).

- **Unsupervised** (default, label-free, depth-robust): blocks b+c+d as above. Ship first.
- **Learned** (`mir/ml/set_encoder.py`): a **Set Transformer** (ISAB inducing points + PMA pooling,
  Lee 2019) or **DeepRC**-style attention MIL (Widrich 2020) over the clonotype `φ`-cloud, weighted by
  `g(a_σ)`, supervised by age / CMV / HLA. Inducing points = learned public-cluster detectors (the block-d
  signal). Bundle it like the codecs (`mir/ml/bundle.py`): serialize weights + prototype hash + RFF/PCA
  basis; refuse cross-basis mixing. **Depth-robustness must be engineered in** (frequency weighting +
  subsampling augmentation during training) — set-nets don't get it for free (§T.7.5).

## 4. Benchmarks — `experiments/benchmark_repertoire_*.py` (`RUN_BENCHMARK`-guarded, subsampled first)

Reuse the `experiments/_hf.py` fetch pattern. Fit ONE `RepertoireSpace` per cohort; embed all samples;
train a simple head (ridge / logistic / kNN) with a fixed CV split + seed; report vs baselines.

1. **`benchmark_repertoire_aging.py` — age regression.** Use the **full-depth `aging`** cohort, NOT
   `aging_lite` (see §5). Metadata `metadata_aging.txt` (`file_name, sample_id, sex, age`, 41 samples,
   ages 6–90; Britanova 2014/2016). Also `airr_hip` age. **Gate:** Spearman(age, prediction) beats the
   `kmer_profile` / `physchem_profile` / diversity-only baselines; `¹D` alone is a strong baseline
   (age↓diversity) so `Φ` must add over it. **Batch check (Prop. `prop:batch`, §T.7.8):** the aging cohort
   has sequencing batches (sample IDs `A2/A3/A4-*`) — verify age prediction survives batch-adjustment
   (leave-one-batch-out CV, or batch as a covariate); a real signal must not be a batch artifact.
2. **`benchmark_repertoire_cmvhla.py` — CMV / HLA (the key test).** `airr_hip` (Emerson 2017 HIP; 786
   subjects, CMV serostatus + HLA-A/B typing). (i) CMV⁺ vs CMV⁻ classification AUC. (ii) **HLA-stratified
   MMD** (Prop. `prop:hla`): CMV⁺ samples are close *only* within an HLA-matched stratum — show
   cross-CMV `mmd_distance` separates status **within** HLA-matched pairs and is uninformative across
   HLA-mismatched pairs. (iii) optionally HLA-allele inference (predict presence of common HLA-A/B alleles).
3. **`benchmark_repertoire_depth.py` — depth-robustness (headline).** Take `aging` samples, `downsample`
   to `N ∈ {100, 300, 1000, 3000, 10⁴}` clonotypes (and use `aging_lite` as an independent downsampled
   set). Show: `‖Φ₁(sub) − Φ₁(full)‖` decays like `n_eff^{-1/2}` (Prop. `prop:kme`); age/CMV prediction
   is preserved down into the RNA-seq-shallow regime (~`10²–10³`). This is the motivating use case.
4. **Multi-chain sanity** — a small all-chain (TRA+TRB) sample: per-locus concat works, missing-chain
   mask behaves.

Per progressive-scaling rules: implement + validate on ≤50 samples / subsampled repertoires, run ONE small
end-to-end per benchmark, report timing; leave full-cohort commands documented for the user.

## 5. Datasets & SOURCES (add to `SOURCES.md`)

- **`aging` (full depth)** — HF `isalgo/airr_benchmark`. ⚠ **Use `aging`, not `aging_lite`**: the
  `vdjtools_lite/` subdir is the *downsampled* version (keep it only as a depth-robustness control). Locate
  the full-depth manifest/files in the repo (age cohort, Britanova). Provenance: experimental TRBβ, ages 6–90.
- **`airr_hip`** — HF `isalgo/airr_hip` = **Emerson et al. 2017** HIP cohort (Nat Genet, PMID 28369038),
  786 subjects (666 discovery + 120 validation), TCRβ + CMV serostatus + HLA-A/B typing (+ age/sex in the
  immuneACCESS metadata). Fetch the metadata TSV via `hf_hub_download(repo_id="isalgo/airr_hip",
  filename=..., repo_type="dataset")`; confirm exact column names at build time (Dataset Viewer was 503 at
  spec time). Provenance: experimental.
- **SRA shallow RNA-seq** — `isalgo/airr_benchmark/sra/meta.tsv` (2993 rows, `PMID Run BioProject Sample`;
  all PMID 30830871, BioProject PRJNA511467). The low-coverage-RNA-seq stress set. Provenance: experimental.

## 6. Phased build checklist

- [ ] `mir/repertoire.py`: `RandomFourierFeatures`, `fit_repertoire_space`, `sample_embedding` (mean block
      first), `mmd_distance` + `__main__` self-check → `tests/test_repertoire.py` (bundled prototypes;
      injected two-cohort separation; depth-robustness of `Φ₁` under `downsample`; `n_eff` finite/positive).
- [ ] Diversity block via `vdjtools.stats.inext` (coverage-standardized) + second-moment block.
- [ ] `hla_stratified_mmd`; comparability-invariant serialization of `RepertoireSpace`.
- [ ] `benchmark_repertoire_{aging,cmvhla,depth}.py` (subsampled) + `_hf` loaders; one timed end-to-end each.
- [ ] (co-equal) `mir/ml/set_encoder.py` Set-Transformer/DeepRC + bundle; train scripts; head-to-head vs backbone.
- [ ] Docs: `THEORY.md` "Sample-level embedding" entry (already stubbed), `README` module row, `SOURCES.md`
      entries; `CLAUDE.md` open-loop → done.
- [ ] `pyproject.toml`: core already has numpy/scipy/sklearn/polars; `scipy.fft`/RFF need nothing new.
      Learned track under `[ml]` (torch).

## 7. Acceptance

- Depth-robustness curve matches `n_eff^{-1/2}` (Prop. `prop:kme`); the `n_eff = ^qD` (a Hill number)
  identity holds numerically (Prop. `prop:antag`).
- HLA-stratified CMV separation (Prop. `prop:hla`): CMV signal present within HLA-matched strata, absent
  across mismatched.
- `Φ` beats the `kmer_profile` / diversity-only / overlap-matrix baselines on age and CMV.
- Works at 100–1000 clonotypes/chain (the RNA-seq regime).
