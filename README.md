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
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Academic%20%26%20Non--Commercial-green"></a>
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

Turn T- and B-cell receptor tables into clonotype coordinates, sample RFF vectors,
or published corpus signatures. PyPI package **mirpy-lib**, Python import **mir**.
Native alignment and repertoire IO come from seqtree and vdjtools.

## Install

```bash
pip install mirpy-lib
```

Optional extras: `[parquet]` for streaming clonotype Parquet, `[ann]` for approximate
neighbours, `[ml]` for Torch codecs, `[bench]` for evaluation, and `[examples]` for
marimo notebooks. See [installation and first steps](https://docs.isalgo.dev/mirpy/getting-started.html).

## Choose your output

| Command | Output |
|---|---|
| `mir embed clonotypes` | Clonotype coordinates with sample/clonotype IDs |
| `mir embed repertoires` | One weighted RFF vector per sample, plus separate channels |
| `mir pca` | Optional sample PCs from an RFF table |
| `mir embeddings` / `mir signature` | Corpus-standardised rsig PCs and pass-through channels |

```bash
mir embed clonotypes samples/*.tsv -o clonotypes.tsv --threads 1
mir embed repertoires samples/*.tsv --save-basis basis -o rff.tsv --threads 1
mir embed repertoires held_out/*.tsv --basis basis -o held_out_rff.tsv --threads 1
mir pca rff.tsv --fit rotation.npz --n-components 32 -o pcs.tsv
mir pca held_out_rff.tsv --rotation rotation.npz -o held_out_pcs.tsv
```

The sample path is **clonotype coordinates → shared clonotype basis → RFF → weighted
mean → optional sample PCA**. Every sample must use the same basis and weighting.
Save and reuse the basis and, if fitted, the sample rotation. RFF is applied before
averaging. rsig is a different representation; its corpus rotations cannot rotate RFF.

The [embedding guide](https://docs.isalgo.dev/mirpy/embeddings.html) covers fitting
on input rows, compatible control rows or a saved rotation, output columns and
memory limits. [Signatures](https://docs.isalgo.dev/mirpy/signature.html) explains
published rsig corpora and joining vdjtools' vsig on `sample_id`.

Frozen reference bundles, vdjtools models and examples are hosted in
[AIRR feature signatures](https://huggingface.co/datasets/isalgo/airr_feature_signatures).
Run `mir reference rna-blood` to print a checksum-verified local bundle directory;
use its matching basis and rotation as described in the embedding guide.

## Python quickstart

```python
import mir                       # establish runtime limits before numerical libraries
import polars as pl
from mir.embedding.tcremp import TCREmp

sample = pl.DataFrame({
    "v_call": ["TRBV10-3*01", "TRBV20-1*01"],
    "j_call": ["TRBJ2-7*01", "TRBJ1-2*01"],
    "junction_aa": ["CASSIRSSYEQYF", "CSARVSGYYGYTF"],
})
model = TCREmp.from_defaults("human", "TRB")
coordinates = model.embed(sample)  # interleaved V, J, junction distances per prototype
```

## Find the right page

| Task | Guide |
|---|---|
| RFF tables, shared bases and sample PCA | [Embeddings](https://docs.isalgo.dev/mirpy/embeddings.html) |
| Filtering before embedding | [Pre-processing](https://docs.isalgo.dev/mirpy/preprocessing.html) |
| Density, MMD, digital donors, codecs and CPU budgets | [User guide](https://docs.isalgo.dev/mirpy/usage.html) |
| Corpus signatures and pass-through channels | [Signatures](https://docs.isalgo.dev/mirpy/signature.html) · [Channels](https://docs.isalgo.dev/mirpy/channels.html) |
| Functions, flags, examples and terminology | [API](https://docs.isalgo.dev/mirpy/api.html) · [Commands](https://docs.isalgo.dev/mirpy/cli.html) · [Examples](https://docs.isalgo.dev/mirpy/notebooks.html) · [Glossary](https://docs.isalgo.dev/mirpy/glossary.html) |
| Mathematical derivations | [Theory](https://docs.isalgo.dev/mirpy/math.html) |

## Resource budgets

Import `mir` before numerical libraries in a fresh interpreter. Sample workers use
serial inner kernels and retain one sample each; only compact RFF rows survive.
The final table still grows with cohort size and RFF width, and a deep input sample
can dominate memory. Use `--threads 1` for serial sample processing or a positive
count within your allocation. Known large allocations check container memory.
See the [resource contract](https://docs.isalgo.dev/mirpy/usage.html#mir-resource-limits)
and [embedding memory guide](https://docs.isalgo.dev/mirpy/embeddings.html#memory-use).

## Development

```bash
bash setup.sh --dev-parents --docs --tests
source .venv/bin/activate
python -m pytest tests/ -q -m "not integration and not benchmark"
env -C docs make html
```

Uses a repo-local uv environment. Public APIs and architecture are documented in
[the API reference](https://docs.isalgo.dev/mirpy/api.html); no private agent files
are needed to use the package.

## Citing

Method: Kremlyakova *et al.*, *TCREMP: A Bioinformatic Pipeline for Efficient Embedding of T-cell
Receptor Sequences from Immune Repertoire and Single-cell Sequencing Data*, **J Mol Biol** 437(15)
(2025) 169205. [PubMed](https://pubmed.ncbi.nlm.nih.gov/40368275/) ·
[DOI](https://doi.org/10.1016/j.jmb.2025.169205).

See the [mathematical foundations](https://docs.isalgo.dev/mirpy/math.html) for derivations.

## License

ISALGO LAB Academic and Non-Commercial Research License v1.0. See [`LICENSE`](LICENSE).

The classical v1.x/v2 repertoire toolkit — parsing, overlap, diversity, TCRnet,
GLIPH — is frozen on the [`legacy-v2`](https://github.com/antigenomics/mirpy/tree/legacy-v2) branch
(`mirpy-lib` 2.x); its functionality lives on in
[vdjtools](https://github.com/antigenomics/vdjtools) and
[vdjmatch](https://github.com/antigenomics/vdjmatch).

### Configured statistical references

Reference bundles can opt into vdjtools' `basic` or `joint` statistical features
and their IMGT11/BLOSUM62 motif variants via `feature_definition=`. The bundle
records the definition and vocabulary for compatible saved-PCA application.
Configured fitting delegates to vdjtools' sparse `TablePCA.fit_features`, preserving
raw median-impute/StandardScaler coordinates without a dense training table.
Sample workers join before the BLAS phase. Existing rsig and sample RFF behavior
stays unchanged; see the embedding guide.
