<h1 align="center">mirpy — ML embeddings for immune repertoires</h1>

[![PyPI](https://img.shields.io/pypi/v/mirpy-lib.svg)](https://pypi.org/project/mirpy-lib/)
[![Python](https://img.shields.io/pypi/pyversions/mirpy-lib.svg)](https://pypi.org/project/mirpy-lib/)
[![License](https://img.shields.io/badge/license-GPLv3-green)](LICENSE)

**mirpy v3** turns T-/B-cell receptor sequences into fixed-length numeric vectors you can
cluster, visualize, and feed to ML models. It implements **TCREMP** — embedding each receptor
by its alignment distances to a fixed set of *prototype* sequences — so that Euclidean distance
in embedding space approximates pairwise alignment distance (see [`THEORY.md`](THEORY.md)).

> v3 is a slim, embedding-focused rewrite. The classical repertoire toolkit (parsing, overlap,
> diversity, TCRnet, GLIPH, …) lives on the **`legacy-v2`** branch (`mirpy-lib` 2.x) and in the
> sibling tools [`vdjtools`](https://github.com/antigenomics/vdjtools) /
> [`vdjmatch`](https://github.com/antigenomics/vdjmatch).

## Install

```bash
pip install mirpy-lib            # core: numpy, polars, scipy, scikit-learn, seqtree, vdjtools
pip install "mirpy-lib[bench]"   # + benchmark / theory experiments
```

Pure-Python wheel; the heavy lifting (alignment, Pgen, sampling) is reused from
[`seqtree`](https://github.com/antigenomics/seqtree) and `vdjtools`.

## Quick start

```python
import polars as pl
from mir.embedding.tcremp import TCREmp

model = TCREmp.from_defaults("human", "TRB", n_prototypes=3000)   # mode="vjcdr3" | "cdr123"
df = pl.DataFrame({
    "v_call":      ["TRBV10-3*01", "TRBV20-1*01"],
    "j_call":      ["TRBJ2-7*01",  "TRBJ1-2*01"],
    "junction_aa": ["CASSIRSSYEQYF", "CSARVSGYYGYTF"],
})
X = model.embed(df)          # (2, 9000) float32 — 3 distances × 3000 prototypes
```

Downstream (cluster antigen-specific TCRs):

```python
from mir.embedding.pca import pca_denoise
from mir.bench.metrics import cluster, cluster_metrics
labels = cluster(pca_denoise(X, n_components=50))
```

Paired chains concatenate per-chain embeddings via `PairedTCREmp`. Input/output are AIRR polars
frames keyed by `vdjtools.io.schema` column names.

## Recommended presets

`TCREmp.from_defaults(species, locus)` uses the per-chain preset when `n_prototypes` is
omitted. Values are data-driven from the bundled prototypes (prototype geometry saturates by
these counts; PC columns are the PCA dims retaining ~95% / ~99% variance):

| chain | n_prototypes | PCs (95%, clustering) | PCs (99%, reconstruction) |
|---|--:|--:|--:|
| human TRA | 2000 | 65 | 220 |
| human TRB | 2000 | 65 | 260 |
| human TRG | 1000 | 25 | 100 |
| human TRD | 2000 | 65 | 280 |
| human IGH | 2000 | 65 | 300 |
| human IGK | 1000 | 20 | 65 |
| human IGL | 1000 | 20 | 65 |
| mouse TRA | 2000 | 50 | 150 |
| mouse TRB | 2000 | 55 | 225 |

Use **95%** PCs for clustering/visualization (the paper's regime); use **99%** PCs when
*reconstructing* sequences with the neural inverse codec (diverse chains like IGH/TRD/TRA lose
too much sequence detail at 95%). Programmatically: `from mir.embedding import get_preset`.

```python
from mir.embedding import get_preset
from mir.embedding.pca import pca_denoise
p = get_preset("human", "IGH")
Xc = pca_denoise(X, n_components=p.n_components)          # clustering
Xr = pca_denoise(X, n_components=p.n_components_recon)    # codec reconstruction
```

## What's inside

| Module | Purpose |
|---|---|
| `mir.embedding.tcremp` | `TCREmp` / `PairedTCREmp` — the prototype embedding |
| `mir.embedding.pca` | PCA denoising of embeddings |
| `mir.distances` | junction distance (`seqtree.gapblock`; `metric`/`matrix`/`alignment` options) + baked germline distances |
| `mir.bench` | VDJdb loader, clustering (`cluster(method=…)`: DBSCAN/HDBSCAN/OPTICS) + F1/retention, theory experiments (incl. `codec_losslessness`), cohort scorers (`bench.eval`: `cv_auc`/`cv_cindex`/`km_logrank`) |
| `mir.density` | continuous-density TCRNET/ALICE — enrichment (+ clonal-abundance channel, `backend=` exact/kdtree/ann) + noise-filtering (Theory T6) |
| `mir.repertoire` | sample-level (repertoire) embedding — RFF kernel mean ‖ Hill diversity ‖ second moment; MMD / HLA-stratified distance; motif witness; `centroid_atypicality`, multi-locus `fit_repertoire_spaces` (Theory §T.7) |
| `mir.explain` | named-channel fusion (`ChannelBuilder`) + scorer-agnostic ablation (`channel_report`/`channel_drivers`) — which part of Φ carries the signal (§T.7) |
| `mir.cohort` | the **digital donor** — multi-chain `fit_donor_embeddings`/`DonorCohort` (+ `transform`/`save`/`load`) + `residualize` / `cluster_samples` / `incidence_biomarkers` (§T.7) |
| `mir.ml` | neural codecs (forward/inverse/Pgen/unified) + learned repertoire `set_encoder` (Set-Transformer/DeepRC) — Part 2, experimental; `[ml]` extra |

## Background subtraction & clustering (`mir.density`)

TCRNET/ALICE find antigen-driven convergent clusters by *neighbour enrichment*. `mir.density`
does the same test with neighbour-counting in the **embedding space** instead of on a sequence
graph (Theory T6): the enrichment `E(z) = f_obs(z)/f_gen(z)` is estimated by an adaptive-bandwidth
**balloon** estimator with a per-clonotype Poisson/binomial significance test and BH q-values —
no graph, and it scales to whole repertoires.

```python
from mir.density import fit_density_space, neighbor_enrichment, enriched_mask, denoise_and_cluster
from mir.embedding.tcremp import TCREmp

model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
# background = a control repertoire (TCRNET) or generate_background(...) (ALICE, P_gen)
space, obs_emb, bg_emb = fit_density_space(model, obs_df, control_df, n_components=20, space="full")
res  = neighbor_enrichment(obs_emb, bg_emb, test="binomial")   # balloon + water-level calibration
hits = obs_df.filter(enriched_mask(res, alpha=0.05))            # background-subtracted clones
labels, mask = denoise_and_cluster(obs_emb, res)               # noise-filter + DBSCAN the hits
```

Use a **biological control** as the background when you have one (e.g. pre- vs post-vaccination,
patient vs healthy) — differential enrichment cancels generic public convergence and isolates the
antigen-specific response. With no control, `generate_background(locus, n)` samples the vdjtools
P_gen model (the ALICE regime); the "water level" of a naive repertoire is handled by the
empirical-null calibration. See `experiments/benchmark_density_{yfv,ankspond,tcrnet}.py`.

At whole-repertoire scale, pass `neighbor_enrichment(..., backend="kdtree")` (exact scipy cKDTree,
5–9× faster than the default BallTree) or `backend="ann"` (approximate pynndescent, ~30× faster
past ~10⁵ clones, trading a small conservative undercount); see `experiments/benchmark_ann.py`.

## Sample-level (repertoire) embedding (`mir.repertoire`)

One fixed vector `Φ(S)` per **repertoire** — an order-invariant multiset of clonotypes with clone
sizes — depth-robust into the low-coverage bulk-RNA-seq regime (Theory §T.7). `Φ(S)` sketches the
empirical measure `ρ_S = Σ_σ w_σ δ_{φ(σ)}` (concave frequency weights, so one hyperexpanded clone
can't dominate) in three blocks: an RFF **kernel mean** (depth-robust, codebook-free — no `K`, no
clustering), a coverage-standardized **Hill diversity** profile, and a **second-moment** Fisher
vector carrying clonotype co-occurrence (HLA-linked public structure). Repertoire distance is the
**MMD** `‖Φ₁(S) − Φ₁(S')‖`.

```python
from mir.repertoire import fit_repertoire_space, sample_embedding, mmd_matrix, class_witness
from mir.embedding.tcremp import TCREmp
import polars as pl

model  = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
space  = fit_repertoire_space(model, pl.concat(samples))   # ONE basis for the whole cohort
embs   = [sample_embedding(space, s) for s in samples]     # Φ(S): mean ‖ diversity ‖ second moment
D      = mmd_matrix(embs, unbiased=True)                    # pairwise repertoire distance (unbiased MMD²)
motifs = class_witness(space, pos_samples, neg_samples, candidates)   # public clones separating two groups
```

**Comparability invariant** (as with the codecs / density): every sample in a cohort must be
embedded through *one* prototype set and *one* PCA+RFF basis, or the measures are incomparable —
`fit_repertoire_space` fits that basis once and `RepertoireSpace` refuses a prototype-hash mismatch.

Use the **unbiased** MMD (`unbiased=True`) whenever samples differ in depth/diversity — the biased
V-statistic's `1/n_eff` self-term otherwise inflates low-diversity samples and fakes a signal. When a
nuisance batch is present, compare *within-batch* contrasts (residualize `Φ` on the batch indicator):
a batch offset is first-order and cancels, while a batch-orthogonal signal (e.g. HLA) survives. The
empirical rule of thumb — **diversity for how-even, the embedding for which-clones**: clone-size
phenotypes (age, CMV) are a diversity summary's turf, while clonotype identity (HLA — strongest in
TRA and class II) lives in the second moment / witness. A learned co-equal set encoder
(Set-Transformer / DeepRC) is in `mir.ml.set_encoder` (`[ml]` extra). See
`experiments/benchmark_repertoire_*.py`, `BENCHMARKS.md`, and `THEORY.md` T7.

## Reproduce the paper

```bash
python experiments/reproduce_supplementary.py   # supplementary S1–S3
python experiments/benchmark_vdjdb.py           # Table S1 (needs a VDJdb dump)
```

Method: Kremlyakova *et al.*, *TCREMP: a bioinformatic pipeline for efficient embedding of
T-cell receptor sequences*, **J Mol Biol** 437 (2025) 169205.

## Performance & parallelism

mirpy is CPU-parallel by default and uses the GPU for the neural codecs. Knobs, by hot path:

| Stage | Knob | Default | Notes |
|---|---|---|---|
| Embedding (junction distance) | `TCREmp(..., threads=N)` | `0` = **all cores** | The C++ `seqtree.gapblock` scorer; releases the GIL, ~530 M pairs/s @16 cores. `threads=1` for a serial run. |
| Density kNN / balloon | `neighbor_enrichment(..., backend=…)` | `"exact"` (BallTree, **1 core**) | `backend="kdtree"` = exact scipy cKDTree, **all cores** (`workers=-1`), 5–9× faster; `backend="ann"` = pynndescent, auto all-core, ~30× at ≥1e5. Prefer `kdtree` for multicore exact. |
| Clustering | `cluster(..., n_jobs=-1)` | sklearn default (1) | forwarded to DBSCAN/OPTICS/HDBSCAN via `**kwargs`; parallelizes the neighbour search. |
| BLAS (PCA, RFF, matmul) | `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` env | all cores | numpy/sklearn use the platform BLAS; cap via env if oversubscribed. |
| Neural codecs (`mir.ml`) | `pick_device()` / `device=` / `MIR_DEVICE` env | **CUDA → MPS → CPU**, auto | every `train_*` / codec / bundle takes `device=`; e.g. `MIR_DEVICE=cuda:1 python experiments/train_forward_encoder.py`. Torch-free paths (`density`, `repertoire`) never touch the GPU. |

Rule of thumb: leave `threads=0` (all cores) for embedding; switch density to `backend="kdtree"`
for exact multicore or `"ann"` at whole-repertoire scale; the GPU is used only by `mir.ml`.

## Development

Conda env `mirpy` (Python 3.12): `bash setup.sh`. Tests: `python -m pytest tests/ -q`.
See [`CLAUDE.md`](CLAUDE.md) for the architecture and reuse map.
