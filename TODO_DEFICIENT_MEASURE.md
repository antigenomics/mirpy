# TODO — sub-probability repertoire embeddings, missing mass, and a naive reference

> **LANDED 2026-08-02** — all five items ship in `mir` (`missing_mass` / `SampleEmbedding.mass` /
> `naive_reference` / `contrast_embedding` in `mir.repertoire`, `preserve_magnitude` in
> `mir.explain.ChannelBuilder.add`, `recovery_report` in `mir.bench`). The measured evidence below is
> reproduced in `CHANGELOG.md` and the README's "Sub-probability embeddings" section; this file is
> kept only as the original hand-off and can be deleted.

Written 2026-08-02 from the eurynome phase24 work. Everything below is **measured on a
134,806-sample AIRR corpus**, not proposed from theory; the evidence lives in
`~/work/industry/bostongene/eurynome/doc/WEIGHTING_AND_DEPTH.md` and
`analysis/phase24_umap_atlas/{30,32,34,36,38,40}_*.py`. This file is the hand-off: what to build in
`mir`, why, and how to test it.

Already landed (commits `76a2ce1`, `8e01dce`): observed-entry standardisation in
`ChannelBuilder.build` / `build_donor_cohort`, plus `mir.cohort.align_loci` and
`missingness_report`. This document is the *next* layer.

---

## The problem, in one paragraph

`mir.repertoire.sample_embedding` builds Φ(S) = Σ_σ w_σ z_σ with weights normalised to sum to 1, so
Φ is the kernel mean embedding of a probability measure. At RNA-seq depth that premise fails: the
**median tissue TRB sample in this corpus holds 21 unique clonotypes** (blood TRB 254, 1st
percentile 1), so w_σ = a_σ/Σa is `1/n` for a technical draw size, not a clonal frequency — true
frequencies live at 1e-5..1e-8. One singleton's weight spans **21,454×** across blood TRB purely
from sample size. Worse, normalising to 1 **forces** a 5-clonotype tumour to assert a full unit of
confidence, so it lands somewhere arbitrary on the unit sphere instead of where it belongs. Callers
respond by imposing a minimum-clonotype floor, which in tumour deletes the *immune desert* — the
phenotype of interest.

---

## 1. `SampleEmbedding.mass` — sub-probability, not renormalised

**What.** Let a sample embedding carry a total mass ≤ 1 instead of always 1.

```python
@dataclass
class SampleEmbedding:
    vector: np.ndarray
    mass: float = 1.0          # NEW: retained probability mass, 1 - M0
    ...
```

with `sample_embedding(..., missing_mass="none" | "turing" | "chao")`.

**Why not negative probability.** It was asked whether the missing mass could be a *negative*
probability. It should not be. Φ's value is that ‖Φ_P − Φ_Q‖_H **is** MMD(P,Q) and that a convex
combination of two Φ's is the Φ of a real pooled repertoire — which is what makes `mir.twin` and
trajectory interpolation meaningful. A measure allowed to go negative on a set is not a probability
measure: MMD stops being a metric and the midpoint stops being the embedding of anything. A
**sub-probability** measure costs none of that.

**The estimators**, and they are the same identity with different M₀:

```
Phi_true = (1 - M0) * sum_seen (p/sum_seen p) z  +  M0 * E[z | unseen]

  none    M0 = 0                      current behaviour
  turing  M0 = f1 / N                 Good-Turing, read-weighted
  chao    M0 = S_u / (N + S_u)        S_u = f1(f1-1) / (2(f2+1))
```

The `chao` form follows from one modelling assumption worth stating in the docstring: *an unseen
clonotype is rare, so had it been drawn it would carry at most one read, and unseen clonotypes do
not overlap.* That fixes each unseen clone's frequency at the detection boundary 1/(N+S_u) and makes
the units legal — N counts reads, S_u counts clonotypes, and adding them is only defensible because
each unseen clone contributes exactly one read. The observed term then collapses to exactly
`a_σ/(N + S_u)`.

The two estimators **coincide when f₁ = 2f₂** and Chao exceeds Turing when f₁ > 2f₂. Measured:
blood TRB 0.552 vs 0.649 (r = 0.93), tissue IGH 0.189 vs 0.458 (r = 0.76).

**Use the bias-corrected Chao1** `f1(f1-1)/(2(f2+1))`, never `f1²/(2f2)` — the classical form is
undefined when no clonotype was seen exactly twice, which is common at these depths.

**Tests.**
- `mass` → ~0 for an all-singleton sample; → ~1 for a deep one. Measured limits: 0.96 and 0.0002.
- `f2 == 0` must not divide by zero.
- Weights plus the unseen block sum to 1 exactly.
- `missing_mass="none"` must be **bit-identical** to today's output — this is the regression gate.

---

## 2. `mir.repertoire.naive_reference` — the unseen has a principled location

**What.** `naive_reference(space, locus, n=20_000, seed=...) -> np.ndarray`, the kernel mean of
sequences drawn from the germline recombination model, cached per (locus, space, n, seed).

**Why.** Once mass is deficient, something must occupy the unseen block, and the choice is not
cosmetic — it decides whether the object is a *shrinkage estimator toward a meaningful point*. Three
candidates were tested:

| unseen prior | verdict |
|---|---|
| the sample's own singletons | begs the question — no new information |
| the corpus mean | this is James–Stein toward the centroid, and it **measurably hurt**: it piles shallow samples into a dense ball that is itself depth-correlated |
| **germline draw** | the one that works |

`vdjtools.model.generate(load_bundled(locus, organism="human"), n, productive_only=True)` produces
20,000 naive V(D)J recombinations in **~8 s**, so this is cheap. mirpy already depends on the same
ecosystem; if a hard vdjtools dependency is unwanted, take an injectable `reference=` array and
document the generate call.

**Measured effect**, filling the unseen block with the germline draw
(`36_hidden_mass_bands.py`, 5,000 samples/view):

| | R²(PC1, depth) before | after |
|---|---:|---:|
| blood TRB | 0.259 | **0.001** |
| tissue IGH | 0.067 | **0.006** |

with kNN label entropy unchanged or better. Verified against a degenerate-arm explanation: the
reference vector is finite, PC1's explained variance is *unchanged* (0.309 → 0.334) so PC1 is a
genuinely different direction rather than a collapsed one, and depth leaves the whole leading block
(best of PC1–5: 0.253 → **0.047**).

**Tests.**
- Deterministic under a fixed seed; cached call returns the identical array.
- `cos(naive_reference, corpus singleton mean)` should be high (0.997–0.9996 measured) — but assert
  on the **centred** comparison, because TCREmp z is an all-positive distance profile and *any* two
  means sit at cos ≈ 0.998 (this artifact caused a wrong conclusion once already).
- Two different loci must give different references.

---

## 3. `contrast_embedding` — where legitimate negativity lives

**What.** `Psi_S = mass * (Phi_S - naive_reference)`.

**Why.** This is the answer to "can missing mass be negative". You do not need a negative *measure*;
a signed **difference of two probability measures** already gives signed coordinates — negative
wherever the sample is depleted relative to unselected recombination — and it is an ordinary RKHS
element with ‖Ψ_S‖ = MMD(S, naive) intact.

Combined with deficient mass, **magnitude = confidence × deviation-from-naive**. An immune desert
has M₀ → 1 and lands at the **origin**, which is the correct place for "no infiltrate detected". A
vague shallow blood sample lands there too and says so by its norm rather than by being dropped.

**Tests.**
- A synthetic all-naive repertoire → ‖Ψ‖ ≈ 0.
- A repertoire with one hyperexpanded clone → large ‖Ψ‖.
- A 3-clonotype sample → ‖Ψ‖ ≈ 0 **without being dropped** (this is the whole point).
- Ψ has both signs; Φ (all-positive z) does not.

---

## 4. ⚠ Scaling: a global scalar, never per-column

**This bit an experiment already and will bite again.** Any representation whose point is that
*magnitude carries information* must be scaled by **one global scalar**, not per-column
standardised. Per-column standardisation forces every coordinate to unit variance across samples, so
a matrix where half the rows are near-zero comes out looking exactly like one where none are — it
deletes the deficiency it was meant to preserve. An earlier deficient arm scored 0.6481 against a
0.6179 baseline purely because of this, and the conclusion drawn from it was worthless.

Suggested: `ChannelBuilder.add(..., preserve_magnitude=True)` marking a block for global rather than
per-column scaling, and a **loud warning** if a block carrying a `mass` attribute is column-scaled.

---

## 5. The evaluation criterion: recoverability, not competition

The scoring rule that should go into the benchmark is **not** "does Φ beat the one-number marker".
It is:

1. **Recovery** — grouped-CV ridge from the embedding's PCs back to each basic repertoire statistic
   (clonotype richness, Shannon, top-clone fraction, singleton fraction, Chao unseen fraction,
   library size, AIRR-per-million), reported as R². High means the statistic is *carried inside* the
   embedding and nothing has to be bolted on beside it.
2. **Increment** — endpoint score for the embedding, for the statistic alone, and for both together.

Renormalising to mass 1 **deletes the magnitude**, so coverage and richness statistics are
unrecoverable from Φ *by construction*. The deficient measure should win question 1 as a design
consequence. Proposed API: `mir.bench.recovery_report(X, stats, groups) -> {stat: r2}`, sitting
beside `missingness_report` as the other "is this object honest" check.

---

## 6. What NOT to do

- **Do not add a minimum-clonotype floor to the library.** It is a *blood* rule, not a tissue rule.
  In blood a low clonotype count is shallow sequencing; in a tumour it is low infiltration, i.e. the
  ImmuneDesert phenotype. A floor applied in tumour deleted 7,179 IE9-labelled donors down to 2,129
  — it removed the stratum the analysis existed to predict. The deficient measure makes the floor
  unnecessary: a 5-clonotype tumour gets M₀ → 1 and collapses onto the naive point, which is exactly
  where it belongs. Let the caller decide; give them `mass` so they can weight instead of filter.
- **Do not shrink toward the corpus centroid.** Measured, and it is worse than doing nothing.
- **Do not sum multiple libraries for one subject** when harmonizing read counts — prefer one and
  record the collapse, or a tumour plus its matched normal invents a library twice the real size.

---

## 7. Suggested order

1. `SampleEmbedding.mass` + `missing_mass=` — self-contained, bit-identical default, big payoff.
2. `naive_reference` — needed by 3, and independently useful as an MMD baseline.
3. `contrast_embedding` — 1 + 2 composed.
4. `preserve_magnitude` scaling guard — small, prevents the recurring mistake.
5. `recovery_report` — the benchmark that shows 1–3 were worth it.

Steps 1–3 are each roughly a day with tests. Step 4 is an hour. Step 5 is where the claim gets
demonstrated rather than asserted.
