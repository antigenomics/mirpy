# mirpy Agentic Skills

This guide summarizes high-value, reusable analysis skills for programmatic work with mirpy.
Each section highlights the main API surface, recommended patterns, and common pitfalls.

## Skill Categories

### 1. Repertoire I/O & Loading

**What it does**: Load and save immune repertoires in various formats (AIRR, TSV, FASTA, pickle).

**Key Functions**:
- `ClonotypeTableParser.parse_inner()` — Parse AIRR-format DataFrames into clonotypes
- `LocusRepertoire.from_pickle()` / `.to_pickle()` — Pickle serialization for fast cache
- `SampleRepertoire` — Multi-locus repertoire representation

**Common Patterns**:
```python
# Load AIRR TSV
df = pd.read_csv("sample.tsv.gz", sep="\t", compression="infer")
parser = ClonotypeTableParser()
clones = parser.parse_inner(df)
rep = LocusRepertoire(clonotypes=clones, locus="TRB")

# Cache with pickle for fast reload
rep.to_pickle("cache.pkl")
rep = LocusRepertoire.from_pickle("cache.pkl")
```

**Use Cases**:
- Large dataset ingestion and caching
- Multi-sample batch processing
- Format conversion (AIRR → pickle → analysis)

**Parallel Load Defaults (io_parallel)**:
- `load_airr_parallel(..., n_jobs=4)` uses parallel parsing by default.
- Automatic sequential fallback occurs when:
	- `n_jobs == 1`
	- row count `< 10,000` (`parallel_min_rows`)
	- row count `<= chunk_size`
- Practical estimate for narrow AIRR TSVs similar to bundled YFV examples:
	- about 43,000 rows per MB of gzipped file size
	- about 0.23 MB gz ≈ 10,000 rows

---

### 2. Repertoire Filtering & Curation

**What it does**: Filter clonotypes by functional status, CDR3 quality, and sequence characteristics.

**Key Functions**:
- `filter_functional()` — Keep functional V-gene + coding CDR3
- `filter_canonical()` — Keep functional + canonical CDR3
- Gene library lookup via `GeneLibrary.is_functional()`

**Common Patterns**:
```python
from mir.common.filter import filter_functional

# Filter to functional clonotypes
functional_rep = filter_functional(rep)

# Custom filtering with gene library
library = GeneLibrary.load_default(loci={"TRB"}, source="imgt")
functional_subset = filter_functional(rep, gene_library=library)
```

**Use Cases**:
- Remove non-productive / non-coding sequences
- Quality control before statistical analysis
- Create standardized "gold standard" repertoires

---

### 3. Gene Usage Calculation

**What it does**: Compute V and J gene usage frequencies with optional allele normalization.

**Key Functions**:
- `GeneUsage.from_repertoire()` — Build usage table from repertoire
- `.v_usage()`, `.j_usage()`, `.vj_usage()` — Marginal and joint frequencies
- `.v_fraction()`, `.j_fraction()`, `.vj_fraction()` — Laplace-smoothed fractions
- `.usage_comparison()`, `.correction_factors()` — Cross-dataset comparison

**Configuration**:
- `strip_alleles` (default `True`) — Normalize TRBV1*01 → TRBV1
- `count` — Count mode aliases normalized internally:
	- `clonotypes`, `clonotype`, `rearrangement`, `rearrangements`, `count_rearrangement`, `count_rearrangements`
	- `duplicates`, `duplicate`, `count_duplicates`
- `pseudocount` — Laplace smoothing parameter

**Common Patterns**:
```python
# Compute usage with automatic allele stripping
gu = GeneUsage.from_repertoire(rep)
v_freq = gu.v_usage("TRB", count="duplicates")
v_frac = gu.v_fraction("TRB", count="duplicates", pseudocount=1.0)

# Preserve alleles for allele-specific analysis
gu_alleles = GeneUsage.from_repertoire(rep, strip_alleles=False)

# Compare against reference (e.g., OLGA mock)
factors = gu.correction_factors(reference_gu, "TRB", scope="vj")

# factors is Dict[key, float]
enriched = {k: f for k, f in factors.items() if f > 2.0}
```

**Use Cases**:
- Characterize immune selection (compare to generative models)
- Detect biased gene usage in disease vs. control
- Normalize across batches / studies

---

### 4. Repertoire Sampling & Resampling

**What it does**: Downsample repertoires with count preservation and resample to match gene usage distributions.

**Key Functions**:
- `downsample()` — Random downsample with exact count guarantee
- `resample_to_gene_usage()` — Resample to match target V/J frequencies
- `select_top()` — Select top-N clonotypes by duplicate count

**Common Patterns**:
```python
from mir.common.sampling import downsample, resample_to_gene_usage, select_top

# Downsample to 1000 cells, preserving frequency distribution
downsampled = downsample(rep, 1000, random_seed=42)

# Resample day-0 to match day-15 V-gene usage
target_gu = GeneUsage.from_repertoire(rep_d15)
resampled = resample_to_gene_usage(
	rep_d0,
	target_gu.v_usage("TRB", count="duplicates"),
	scope="v",           # or gene_type="v" for backward compatibility
	weighted=True,
	random_seed=42,
)

# Select top 100 clonotypes
top_100 = select_top(rep, 100)
```

**Use Cases**:
- Normalize repertoire sizes for fair comparison
- Match sample sizes in longitudinal studies
- Create bootstraps for confidence intervals

---

### 5. K-mer Analysis

**What it does**: Generate k-mers from CDR3 sequences and compute k-mer statistics.

**Key Functions**:
- `generate_kmers()` — Extract k-mers from junction sequences
- K-mer frequency tables and overlap analysis

**Common Patterns**:
```python
from mir.basic.kmerization import generate_kmers

# Generate 3-mers from all CDR3s
kmers_3 = generate_kmers(rep, k=3)

# Build k-mer overlap profile
kmers_common = [k for k in kmers_3 if kmers_3[k] > 2]
```

**Use Cases**:
- Detect sequence motifs in TCR response
- Assess repertoire similarity via shared k-mers
- Find clonotype clusters

**Control-backed bag-of-k-mers profiles**:
- `mir.embedding.bag_of_kmers.build_control_kmer_profile()` computes control
	k-mer statistics as an in-memory object by default (no profile cache write).
- Set `cache=True` for persisted profile tables under the control cache.
- Profile object fields:
		- `token_stats` (`token`, `n`, `T`, `p`, `idf`)
		- `position_stats` (`token`, `count`, `pos`, `junction_len`)
		- `metadata` (profile name, params, totals, cache mode)

---

### 6. Graph-Based Analysis

**What it does**: Build and analyze CDR3 similarity graphs based on Hamming distance and k-mer similarity.

**Key Functions**:
- `build_edit_distance_graph()` — `igraph` graph from clonotypes with edit-distance edges
- `compute_neighborhood_stats()` — per-clonotype neighbor counts (self/background)
- `add_neighborhood_metadata()` / `add_neighborhood_enrichment_metadata()` — metadata-first enrichment
- Graph metrics via `igraph`: connected components, degree, layouts

**Common Patterns**:
```python
from mir.graph.edit_distance_graph import build_edit_distance_graph
from mir.graph import compute_neighborhood_stats

# Build Hamming graph with threshold 1
g = build_edit_distance_graph(
	rep.clonotypes,
	metric="hamming",
	threshold=1,
	n_jobs=4,
)
print(f"Connected components: {len(g.components())}")

# Compute neighbor stats directly (trie-backed)
stats = compute_neighborhood_stats(rep, metric="hamming", threshold=1, n_jobs=4)
print(stats[next(iter(stats))])
```

**Performance Notes**:
- `build_edit_distance_graph()` and neighborhood/TCRNET methods use `tcrtrie` for fast approximate search.
- Prefer `n_jobs` across APIs for worker control (`nproc` in graph code is backward-compatible only).
- For benchmarking against synthetic controls, use cached controls where possible (`ControlManager`, e.g. `n=1_000_000`).

**Use Cases**:
- Identify clonal families / clusters
- Detect clonal expansions with sequence variants
- Explore TCR cross-reactivity patterns

---

### 7. Gene Sequence Similarity & Clustering

**What it does**: Cluster V/D/J genes by sequence similarity using alignment-free methods.

**Key Functions**:
- `GeneSequenceSimilarity` — Compute pairwise gene sequence similarity
- Hierarchical clustering of genes
- Gene grouping for tolerance analysis

**Common Patterns**:
```python
from mir.embedding.gene_similarity import compute_gene_similarity

# Cluster V genes by sequence
similarity = compute_gene_similarity(library_genes, method="kmer")
clusters = cluster_genes(similarity, threshold=0.85)
```

**Use Cases**:
- Group cross-reactive V genes
- Detect genes with similar HLA interactions
- Build gene families for downstream analysis

---

### 8. P-gen (Probability Generation) Integration

**What it does**: Compute probability of generation for clonotypes using OLGA or precomputed models.

**Key Functions**:
- `OlgaModel` — OLGA sequence generation and p-gen computation
- `PgenGeneUsageAdjustment` — Adjust OLGA mock p-gens by observed gene usage
- `PgenBinPool` — Bin clonotypes by log2 p-gen range

**Common Patterns**:
```python
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.biomarkers.vdjbet import PgenBinPool

# Compute p-gen for repertoire clonotypes
model = OlgaModel(locus="TRB", seed=42)
pgens = model.compute_pgens([c.junction_aa for c in rep.clonotypes])

# Adjust mocks by observed gene usage
pgen_adj = PgenGeneUsageAdjustment(
	observed_gu,
	reference=olga_gu,
	seed=42,
)

# Build binned pool for significance testing
```

---

### 9. Control Data Management (Synthetic + Real)

**What it does**: Build/download and cache background control repertoires for
enrichment workflows, with manifest tracking by species/locus/type.

**Key APIs**:
- `ControlManager.ensure_synthetic_control()` — generate OLGA control and store as pickle
- `ControlManager.ensure_real_control()` — fetch HuggingFace control dataset and convert to pickle
- `ControlManager.ensure_and_load_control_df()` — on-demand setup + immediate load
- `ControlManager.load_control_df()` — load registered control table
- `ControlManager.list_available_controls()` — inspect local registry

**CLI**:
- `mirpy-control-setup --type synthetic --species human,mouse --loci TRA,TRB --n 1000000`
- `mirpy-control-setup --type real --species hsa --loci Tbeta`

**Aliases**:
- Species: `human/hsa/HomoSapiens`, `mouse/mmu/MusMusculus`
- Loci: `TRA/TRB/TRG/TRD/IGH/IGK/IGL`, plus aliases like `Talpha`, `Tbeta`, `Bheavy`, `Bkappa`

**Storage and deployment notes**:
- Default control root: `~/.cache/mirpy/controls`
- Override with `MIRPY_CONTROL_DIR` (recommended for shared HPC caches or node-local scratch)
- Manifest (`manifest.json`) records available controls and their paths for reproducible reuse
- Real controls can be large (multi-GB pickles for large loci like human TRB); prefer fast local scratch/cache on clusters
- Control build/download is lock-protected per `(type, species, locus)` so concurrent workers wait and reuse one produced artifact

**Benchmarking controls**:
- `RUN_BENCHMARK=1 pytest tests/test_control_benchmark.py -s`
- Benchmark covers both synthetic generation and real HuggingFace download/build plus cache-hit timing.

**Common pattern**:
```python
from mir.common.control import ControlManager
from mir.embedding.bag_of_kmers import BagOfKmersParams, build_control_kmer_profile

mgr = ControlManager()
df_bg = mgr.ensure_and_load_control_df("real", "human", "TRB")
```
