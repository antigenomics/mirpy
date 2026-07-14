# Migration guidelines — TCREMP coordinates from squared dissimilarity `d` to the metric `ρ = √d`

> **DECISION (2026-07-13): NOT migrating — this plan is superseded, kept for the record.**
> The `metric="sqrt"` flag was added (commit `4e7c275`) and benchmarked: **`√d` showed no gain (a wash,
> worse on some slices)** on VDJdb retention/F1 — see the "Outcome" section below — so **v3 keeps the
> squared `d`** (default `metric="squared"`). Instead of switching,
> the theory was corrected to document `d` honestly — `(X,d)` is a *negative-type semimetric* with `ρ=√d`
> the induced metric; the coordinate map is Hölder-½ in `d` (constant `2·diam`), not 1-Lipschitz — in
> `appendix/tcremp_theory.tex` and the `junction.py`/`tcremp.py`/`bench/theory.py` docstrings. **Do not
> execute the switch below** unless a future benchmark reverses the verdict.

_Draft plan for a separate session. Benchmark `√d` vs `d` first; switch only if it wins or ties._
_Written 2026-07-13._

## Outcome (benchmarked 2026-07-13) — DO NOT SWITCH; kept `d`

The §2 gate was prototyped and run on human TRB (VDJdb slim, arda coords): a `metric` flag added to
`junction.py` / `TCREmp` / `density` (all three blocks on one scale; `ρ=√d` is the elementwise sqrt
of the assembled squared embedding). Result — a **wash**, so `d` stays the default and the flag was
**reverted** (not shipped):

| Metric | `d` (squared) | `√d` (sqrt) | Verdict |
|---|---|---|---|
| VDJdb mean F1 (PCA-50 preset) | 80% | 80% | tie (±1–2% jitter from seqtree threaded sums) |
| VDJdb mean F1 (best PCA dim, 100–150) | 88% | 90% | `√d` marginal, but needs ~3× the PCs |
| Retention / purity | 17% / 84% | 18% / 86% | `√d` marginal |
| **S2 `D`–`d` correlation** | **0.46** | 0.39 | **`d` clearly better** (stable across runs) |
| DBSCAN `eps` / density `r₁` | 36.6 / 9.6 | 36.6 / 9.2 | both sane |

**Decision (per the §2 rule "if it's a wash, keep `d`"):** clustering is within noise and `√d` is
*worse* on the S2 alignment-tracking diagnostic, so the expensive switch (re-bake germline, re-fit
presets, re-train all 4 codecs, re-calibrate density, break every shipped embedding) is **not
justified**. `d` remains the published-compatible default; the `metric` flag was rolled back. It is
trivial to reinstate (sqrt of the assembled embedding, all blocks share the Gram construction — §3.2)
if the theory-cleanliness case later outweighs the wash. §5's honest `d`=Hölder‑½ / `ρ`=metric
appendix framing stands whichever way this lands.

## 1. Why this exists

TCREMP embeds `φ(x)_k = d(x, p_k)` where the alignment dissimilarity is the **Gram form**

```
d(a,b) = s(a,a) + s(b,b) − 2·s(a,b) = ‖ψ(a) − ψ(b)‖²
```

i.e. a **squared** Hilbert distance (Schoenberg / negative type — see appendix §T.2). A squared
Euclidean distance is **not a metric**: it violates the triangle inequality
(`ψ = 0,1,2 ⇒ d(0,2)=4 > d(0,1)+d(1,2)=2`). The genuine metric is `ρ = √d`.

Consequences of shipping the squared `d`:

- The appendix's clean 1‑Lipschitz / non-expansive results (`prop:lipschitz`, `prop:kuratowski`
  isometry, `prop:energy` bound, `prop:samplecomplexity`, `prop:drift`, `prop:dp`) hold **verbatim for
  `ρ=√d`** but only **weaken to Hölder‑½** for the squared `d`
  (`D ≤ √K·d` → `D ≤ √K·2·diam·√d`; the Kuratowski isometry breaks outright).
- Euclidean distance `D` between squared-distance profiles over-weights far prototypes (the squared tail),
  which **may** hurt DBSCAN / PCA. Unknown until benchmarked.

**This is a theory-cleanliness + possible-clustering improvement, NOT a correctness fix.** The method is a
dissimilarity representation and works fine on non-metric `d`; the energy functional
`D²/K → E_p[(d(x,p)−d(y,p))²]` needs only boundedness, not the triangle inequality.

## 2. Benchmark first — the gate (do this before touching anything else)

Add a **coordinate-mode flag** so one pipeline computes both:

- `mir/distances/junction.py`: `metric: Literal["squared","sqrt"] = "squared"`; when `"sqrt"`, return
  `np.sqrt(np.clip(d, 0, None))`.
- Thread the flag through `TCREmp` / germline lookup so **all three blocks (V, J, junction)** are on the
  same scale (see §3.2 — do not sqrt only the junction).

Compare `d` vs `√d`, fixed seeds, per chain:

| Metric | Where | Pass condition for switching |
|---|---|---|
| Cluster F1 / retention / purity | `mir/bench` on VDJdb (α/β/paired) | `√d` ≥ `d` (no regression) |
| `D`–`d` correlation (S2) | `mir/bench/theory.s2_dissimilarity_distance_correlation` | report both; `√d` should track alignment distance at least as well |
| DBSCAN `eps` stability | `mir/bench/metrics` (kneedle) | `√d` eps not more fragile |
| Density `r₁` calibration | `mir/density.calibrate_radius` | sane, positive; re-tune `eps_factor` if needed |

**Decision rule:** switch only if `√d` matches or beats `d` on cluster F1/retention, or gives materially more
robust distances. If it's a wash, keep `d` (published-compatible) and just fix the appendix wording.

## 3. Code changes (only if the benchmark says switch)

### 3.1 Junction distance
`mir/distances/junction.py:42-51` — after `score_matrix`, clamp then sqrt:
```python
sm = np.asarray(sm, dtype=np.float32)
if metric == "sqrt":
    sm = np.sqrt(np.clip(sm, 0.0, None))   # CND clamp must precede sqrt (indefinite BLOSUM ⇒ tiny negatives)
return sm
```
- `√0 = 0` preserves the zero diagonal. The gap-block "best placement" is a **monotone** argmin, so select on
  `d` (unchanged) and sqrt the chosen value — do **not** sqrt inside the placement search.
- Fix the docstring (`junction.py:3-6`): it currently calls the squared `d` a "genuine metric" — that is only
  true after the sqrt.

### 3.2 Germline distances (V / J / CDR1 / CDR2)
`mir/distances/germline.py` + the baker (`build_germline_dist.py`): the baked `.npz` blocks **must also be
`√d`** so the concatenated embedding is homogeneous across blocks. Two options:
- re-bake `√d` matrices (bump a version tag in the `.npz` / manifest), or
- sqrt at lookup time behind the same flag.
First confirm the germline blocks use the same Gram construction as the junction; if they are raw alignment
scores, reconcile the scale explicitly.

### 3.3 Embedding + prototypes
- `mir/embedding/tcremp.py`: coordinates become `ρ`; the `embed` path is unchanged if junction + germline
  return `ρ`. Assert all three interleaved slots share scale.
- `mir/resources/prototypes/manifest.json`: **bump the coordinate-system version** — `ρ` embeddings are a new,
  incompatible space; consumers must check it.

### 3.4 PCA presets
`mir/embedding/pca.py` + `presets.py`: the variance spectrum changes under `√d` → **re-fit** per-chain
`n_components` (95% / 99%) and update `CHAIN_PRESETS`.

### 3.5 Part-2 codecs (the expensive part)
`mir/ml`: **re-train all** on `√d` embeddings — forward encoder, inverse decoder, Pgen regressor, unified
codec. Bump `CodecBundle` version + prototype hash (`bundle.py` already refuses hash mismatches — keep that
guard; old codecs become uninstallable against `ρ`, which is correct). Re-run `experiments/train_*`.

### 3.6 Density
`mir/density.py`: `E(z)=f_obs/f_gen` is invariant to a monotone reparametrisation **in principle**, but the
balloon radius calibration `r₁` (median one-substitution drift) lives in the coordinate metric → **re-calibrate**
and re-run `experiments/benchmark_density_*`. The `ρ 0.37→−0.05` convergence and the `43%`/`1%`/`Jaccard 0.86`
numbers may shift; update THEORY.md T6.

### 3.7 Benchmarks + docs
Re-run the bench harness and update **THEORY.md** reproduced numbers (S2 `R`, Table S1 F1/retention, T5 drift
slopes, T6 density). Update `CLAUDE.md` (coordinate-system note) and `SOURCES.md` (baked-artifact version).

## 4. Versioning / compatibility

- **Major coordinate-system change.** `d`-embeddings and `ρ`-embeddings are **not comparable**. Bump the
  embedding schema version; the bundle's `(prototype-hash + PCA-rotation)` guard already blocks mixing — rely
  on it.
- Consider keeping `metric="squared"` selectable for backward reproduction of the published-`d` results.

## 5. Theory alignment (appendix — being done now, coordinate-agnostic)

The appendix is being updated **now** to state the 1‑Lipschitz property correctly: it holds for the metric
`ρ=√d` (reverse triangle inequality of the genuine Hilbert metric, from negative type / Schoenberg), with the
current squared `d` documented honestly as Hölder‑½. So the manuscript is correct **whichever way the
benchmark lands**. After a switch, simply drop the squared-`d` caveats — `prop:lipschitz`, `prop:kuratowski`,
`prop:energy`, `prop:samplecomplexity`, `prop:drift`, `prop:dp` then hold verbatim.

## 6. Checklist

- [ ] Add `metric` flag (`junction.py`, threaded through germline + `TCREmp`)
- [ ] Benchmark `d` vs `√d`: F1/retention/purity, S2 correlation, eps stability, `r₁` — **decide**
- [ ] (if switch) sqrt junction + re-bake germline `.npz` (+ version)
- [ ] (if switch) re-fit PCA presets; bump prototype manifest version
- [ ] (if switch) re-train all `mir/ml` codecs; bump bundle version + prototype hash
- [ ] (if switch) re-calibrate density `r₁`; re-run density benchmarks
- [ ] (if switch) re-run bench harness; update THEORY.md / CLAUDE.md / SOURCES.md numbers
- [ ] Fix `junction.py` docstring "genuine metric" claim
- [ ] Drop squared-`d` caveats from the appendix once switched
