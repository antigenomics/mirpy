<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/mirpy_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/mirpy_light.svg">
    <!-- Absolute PNG fallback: PyPI strips <picture>/<source> and cannot render a relative or
         raw-served SVG, so the logo must be an absolute-URL raster here. GitHub uses the SVG sources. -->
    <img alt="mirpy" src="https://raw.githubusercontent.com/antigenomics/mirpy/master/assets/mirpy_light.png" width="360">
  </picture>
</p>

<h1 align="center">mirpy — ML embeddings for immune repertoires</h1>

<p align="center">
  <a href="https://pypi.org/project/mirpy-lib/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mirpy-lib.svg"></a>
  <a href="https://pypi.org/project/mirpy-lib/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mirpy-lib.svg"></a>
  <a href="https://github.com/antigenomics/mirpy/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/antigenomics/mirpy/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://docs.isalgo.dev/mirpy/"><img alt="docs" src="https://github.com/antigenomics/mirpy/actions/workflows/docs.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-GPLv3-green"></a>
</p>

<p align="center"><b>
  <a href="https://docs.isalgo.dev/mirpy/getting-started.html">Get started</a> ·
  <a href="https://docs.isalgo.dev/mirpy/usage.html">Guides</a> ·
  <a href="https://docs.isalgo.dev/mirpy/cli.html">Commands</a> ·
  <a href="https://docs.isalgo.dev/mirpy/api.html">API</a> ·
  <a href="https://docs.isalgo.dev/mirpy/math.html">Theory</a> ·
  <a href="https://docs.isalgo.dev/mirpy/notebooks.html">Examples</a> ·
  <a href="https://docs.isalgo.dev/mirpy/glossary.html">Glossary</a>
</b></p>

Turn T- and B-cell receptor repertoires into fixed-length numeric vectors you can cluster,
visualise, and hand to a machine-learning model — at the scale of one receptor, one repertoire, or a
whole cohort.

mirpy implements **TCREMP**: each receptor is embedded by its alignment distances to a fixed set of
*prototype* sequences, so Euclidean distance in embedding space approximates pairwise alignment
distance. That turns a set of variable-length strings into a coordinate system, which is what every
downstream method actually needs.

PyPI package `mirpy-lib`, import name `mir`. Pure Python; the alignment, Pgen and sampling work is
reused from [seqtree](https://github.com/antigenomics/seqtree) and
[vdjtools](https://github.com/antigenomics/vdjtools).

## Install

```bash
pip install mirpy-lib
```

Optional extras, each for one job:

```bash
pip install "mirpy-lib[ann]"     # pynndescent: approximate neighbours at whole-repertoire scale
pip install "mirpy-lib[ml]"      # torch: the neural codecs and the learned set encoder
pip install "mirpy-lib[bench]"   # the benchmark harness and theory experiments
pip install "mirpy-lib[examples]"  # marimo, matplotlib, umap-learn: to run the notebooks
```

## Three scales

Choosing the right scale is most of the work:

| Scale | One unit is | You get | Module |
|---|---|---|---|
| Clonotype | one receptor | a point in prototype space | `mir.embedding` |
| Repertoire | one sample from one donor | one fingerprint, `Φ(S)` | `mir.repertoire` |
| Cohort | many donors, several chains each | one comparable matrix | `mir.cohort` |

## Quick start

```python
import polars as pl
from mir.embedding.tcremp import TCREmp

model = TCREmp.from_defaults("human", "TRB")     # per-chain preset: 2000 prototypes
df = pl.DataFrame({
    "v_call":      ["TRBV10-3*01",   "TRBV20-1*01"],
    "j_call":      ["TRBJ2-7*01",    "TRBJ1-2*01"],
    "junction_aa": ["CASSIRSSYEQYF", "CSARVSGYYGYTF"],
})
X = model.embed(df)          # (2, 6000) float32 — three distances per prototype
```

Denoise, then cluster:

```python
from mir.embedding.pca import pca_denoise
from mir.bench.metrics import cluster
labels = cluster(pca_denoise(model.embed(vdjdb_df), n_components=65))
```

From the shell:

```bash
mir embed clonotypes  sample.tsv      -o clonotypes.parquet --pca 50
mir embed repertoires cohort/*.tsv.gz -o phi.tsv --mmd mmd.tsv
mir signature --corpus blood cohort/*.tsv.gz -o rsig.parquet
```

**[Getting started](https://docs.isalgo.dev/mirpy/getting-started.html)** walks through all of this
in about fifteen minutes, offline, with nothing to download.

## Where to start

| You want to | Go to |
|---|---|
| Embed receptors and cluster antigen-specific ones | [Getting started](https://docs.isalgo.dev/mirpy/getting-started.html) |
| Pick a prototype count and a PCA dimension | [Which prototypes](https://docs.isalgo.dev/mirpy/usage.html#which-prototypes) |
| Find enriched or antigen-driven neighbourhoods | [`mir.density`](https://docs.isalgo.dev/mirpy/usage.html#density-and-background-subtraction) |
| Compare whole repertoires rather than clonotypes | [`mir.repertoire`](https://docs.isalgo.dev/mirpy/usage.html) |
| Get one fixed, named feature vector per sample | [Signatures](https://docs.isalgo.dev/mirpy/signature.html) |
| Name which part of a vector carries a signal | [Channels](https://docs.isalgo.dev/mirpy/channels.html) |
| Understand why any of this is valid | [Theory](https://docs.isalgo.dev/mirpy/math.html) |
| Look up a command or a function | [Commands](https://docs.isalgo.dev/mirpy/cli.html) · [API](https://docs.isalgo.dev/mirpy/api.html) |
| Check what a term means | [Glossary](https://docs.isalgo.dev/mirpy/glossary.html) |

## What is in the box

| Module | What it does |
|---|---|
| `mir.embedding` | `TCREmp` and `PairedTCREmp`, PCA denoising, the bundled prototypes and per-chain presets |
| `mir.distances` | Junction distance through `seqtree.gapblock`, plus precomputed arda germline distances |
| `mir.density` | Graph-free TCRNET and ALICE neighbourhood enrichment in embedding space, with exact, kdtree and approximate backends |
| `mir.repertoire` | One vector per sample: kernel mean, coverage-standardised Hill diversity, second moment; MMD distance, motif witness, sub-probability measures |
| `mir.signature` | The portable signature — fixed, named, already-standardised columns that join to vdjtools' half on `sample_id` |
| `mir.explain` | Which named channel carries a signal under *your* scorer, and which clonotypes drive it |
| `mir.cohort` | The digital donor — multi-chain donor embeddings, batch residualisation, sample clustering |
| `mir.track` | Exposure trajectory — a latent progression axis disentangled from a known covariate |
| `mir.generate`, `mir.twin` | Sample new synthetic donor states, or perturb and resample one donor's state |
| `mir.ml` | Neural codecs, learned set encoders, diffusion generator — experimental, `[ml]` extra |
| `mir.bench` | VDJdb loader, clustering metrics, cohort scorers, reproduced theory experiments |
| `mir.cli` | The `mir` console script — `embed clonotypes`, `embed repertoires`, `signature`, `corpus` |

## Prototypes are the coordinate system

Every embedding is distances to prototypes, so the prototype set *is* the coordinate system. mirpy
ships **10,000 real receptors per chain** — a fixed-seed uniform sample of unique, productive,
germline-resolvable clonotypes from arda-annotated repertoires, for human TRA/TRB/TRG/TRD/IGH/IGK/IGL
and mouse TRA/TRB. Nothing is downloaded at import or build time.

Real rather than model-generated, because synthetic junctions have degenerate lengths and embed
measurably worse. To ask whether a result depends on *which* prototypes you drew, take a replicate:
the geometry agrees at R = 0.993 between two independent draws at the default `n=1000`, and 0.997 at
`n=2000`, so from about `n≈500` up the draw is not what your result rests on.

Each replicate is a **different coordinate system** — distances within one are comparable, across two
are not, and every container refuses a prototype-hash mismatch rather than mixing them silently.
[Details and the full presets table](https://docs.isalgo.dev/mirpy/usage.html#which-prototypes).

## Repertoire fingerprints

One fixed vector `Φ(S)` per repertoire, robust down into the low-coverage bulk-RNA-seq regime. It
sketches the repertoire as a weighted cloud of receptor points in three blocks: a random-feature
kernel mean, a coverage-standardised Hill diversity profile, and a second-moment block carrying
co-occurrence structure. Distance between two repertoires is the MMD.

```python
from mir.repertoire import fit_repertoire_space, sample_embedding, mmd_matrix

space = fit_repertoire_space(model, pooled_clonotypes)   # ONE basis for the whole cohort
embs  = [sample_embedding(space, s) for s in samples]
D     = mmd_matrix(embs, unbiased=True)
```

Every sample in a cohort must go through one prototype set and one basis, or the distances are not
comparable. Use `unbiased=True` whenever depths differ: the biased estimator's `1/n_eff` self-term
inflates low-diversity samples and manufactures signal with the wrong sign.

## Signatures

`Φ(S)` is fitted on *your* cohort, which is what makes it powerful and also what makes it
incomparable with anyone else's. When you need a vector a collaborator can reproduce independently,
use a **signature** instead: fixed, named columns standardised against a published reference corpus.

```bash
mir       signature --corpus blood cohort/*.tsv.gz -o rsig.tsv   # the geometry half
vdjtools  signature --corpus blood cohort/*.tsv.gz -o vsig.tsv   # the statistics half
```

The two join on `sample_id`. Nine corpora are published, all at 256 components per locus; artifacts
are fetched and digest-verified on first use. `--corpus` is required — a signature is comparable to
another one only if both were rotated through the same corpus.
[Signatures](https://docs.isalgo.dev/mirpy/signature.html) ·
[Channels](https://docs.isalgo.dev/mirpy/channels.html)

## Performance

CPU-parallel by default; the GPU is used only by `mir.ml`. Embedding runs the C++
`seqtree.gapblock` scorer at about 530 million pairs per second on 16 cores and releases the GIL.
Density defaults to an exact multicore kd-tree; the approximate backend is about 30× faster past 1e5
clones and is deliberately asymmetric — only the observed side is approximate, which biases
enrichment down, while the background is always exact.
[Full table of parallelism settings](https://docs.isalgo.dev/mirpy/usage.html#performance-and-parallelism).

## Examples

Six runnable [marimo](https://marimo.io) notebooks — plain Python files, so they diff like
source:

```bash
pip install "mirpy-lib[examples]"
marimo edit examples/signature_pipeline.py     # start here
```

`signature_pipeline` (a folder of AIRR files to a joinable table — **read this one first**) ·
`quickstart` (embed, denoise, cluster, UMAP) · `signature` (how the halves differ, and how many
components survive a refit) · `density` (enrichment against a background model) ·
`trajectory_and_twin` (time courses and counterfactual donors) · `theory` (the results the
library rests on). See the
[gallery](https://docs.isalgo.dev/mirpy/notebooks.html).

## Development

Repo-local `.venv` via [uv](https://docs.astral.sh/uv/):

```bash
bash setup.sh --dev-parents --tests    # editable-installs sibling seqtree / vdjtools checkouts
python -m pytest tests/ -q
```

See [`CLAUDE.md`](CLAUDE.md) for the architecture and the reuse map.

## Citing

Method: Kremlyakova *et al.*, *TCREMP: a bioinformatic pipeline for efficient embedding of T-cell
receptor sequences*, **J Mol Biol** 437 (2025) 169205.

The theory this implements is collected in the
[mathematical foundations](https://docs.isalgo.dev/mirpy/math.html); derivations and recorded
benchmark numbers live in the companion analysis repository `2026-mirpy-analysis`, with the LaTeX
appendix in `2026-mirpy-ms`.

## License

GPL-3.0-or-later. The classical v1.x/v2 repertoire toolkit — parsing, overlap, diversity, TCRnet,
GLIPH — is frozen on the [`legacy-v2`](https://github.com/antigenomics/mirpy/tree/legacy-v2) branch
(`mirpy-lib` 2.x); its functionality lives on in
[vdjtools](https://github.com/antigenomics/vdjtools) and
[vdjmatch](https://github.com/antigenomics/vdjmatch).

### Running signatures across samples

```bash
mir signature --corpus blood --jobs 8 samples/*.tsv -o signatures.parquet
```

`--jobs` controls worker processes. It defaults to 1; use 0 to request all available
cores or a positive count to set the budget explicitly. Each spawned worker uses one
thread per numerical kernel. Workers read one sample at a time, retain their fitted
resources, and return only a feature row. Embeddings use fixed 4,096-row blocks
(24 MiB per float64 block at 256 prototypes). Output follows input order. No sample-count
heuristic changes an explicit worker budget.

For the Python cohort API, pass picklable zero-argument readers to defer file loading
into workers. Passing already-loaded frames keeps those frames in the parent too.
The CLI sets kernel thread limits before imports, including at `--jobs 1`. Direct
Python calls retain the calling process's Polars and BLAS settings at `n_jobs=1`.

Density analysis also accepts an explicit kernel budget:
`neighbor_enrichment(observed, background, threads=8)`. Background counts are exact with both
KD-tree and ANN observation queries, including points tied at the selected radius. Only ANN
observation-neighbor recall is approximate.
