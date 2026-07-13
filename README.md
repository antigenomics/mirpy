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
| `mir.bench` | VDJdb loader, DBSCAN clustering + F1/retention, theory experiments |
| `mir.density` | continuous-density TCRNET/ALICE — enrichment (+ clonal-abundance channel) + noise-filtering (Theory T6) |
| `mir.ml` | neural codecs — forward/inverse/Pgen/unified (Part 2, experimental; `[ml]` extra) |

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

## Reproduce the paper

```bash
python experiments/reproduce_supplementary.py   # supplementary S1–S3
python experiments/benchmark_vdjdb.py           # Table S1 (needs a VDJdb dump)
```

Method: Kremlyakova *et al.*, *TCREMP: a bioinformatic pipeline for efficient embedding of
T-cell receptor sequences*, **J Mol Biol** 437 (2025) 169205.

## Development

Conda env `mirpy` (Python 3.12): `bash setup.sh`. Tests: `python -m pytest tests/ -q`.
See [`CLAUDE.md`](CLAUDE.md) for the architecture and reuse map.
