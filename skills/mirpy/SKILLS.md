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
    target_gu,
    "TRB",
    gene_type="v",
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

---

### 6. Graph-Based Analysis

**What it does**: Build and analyze CDR3 similarity graphs based on Hamming distance and k-mer similarity.

**Key Functions**:
- `EditDistanceGraph` — Clonotypes as nodes, edges for Hamming distance ≤ threshold
- `KmerGraph` — Clonotypes connected by shared k-mers
- Graph metrics: clustering, diameter, connectivity

**Common Patterns**:
```python
from mir.graph.edit_distance_graph import EditDistanceGraph
from mir.graph.token_graph import KmerGraph

# Build edit-distance graph (1-2 mismatches)
edg = EditDistanceGraph(rep, max_distance=2)
print(f"Clonotype clusters: {edg.n_connected_components()}")

# Build k-mer graph
kg = KmerGraph(rep, k=3, min_shared=1)
print(f"Highly connected clonotypes: {[c for c in kg if kg.degree[c] > 5]}")
```

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
pool = PgenBinPool("TRB", n=50_000, n_jobs=4, seed=42)
```

**Use Cases**:
- Identify clonotypes with unlikely CDR3 sequences
- Control for V/J gene usage bias in significance tests
- Compare antigen-specific vs. naive repertoires

---

### 9. VDJbet Overlap Analysis

**What it does**: Compute repertoire overlap with significance testing via p-generation null models.

**Key Functions**:
- `make_reference_keys()` / `make_query_index()` — Build overlap comparison indices
- `count_overlap()` — Count matching clonotypes (exact, 1mm, or V/J match)
- `compute_overlaps()` — Batch overlap for multiple reference sets
- `VDJBetOverlapAnalysis` — Full significance testing pipeline

**Configuration**:
- Matching modes: exact, 1-mismatch (1mm), or V/J-required
- Normalization: divide by target repertoire size
- Statistical test: empirical p-value from mock null distribution

**Common Patterns**:
```python
from mir.comparative.overlap import make_reference_keys, make_query_index, count_overlap
from mir.biomarkers.vdjbet import VDJBetOverlapAnalysis

# Simple overlap count
ref_keys = make_reference_keys(ref_rep)
query_idx = make_query_index(query_rep)
overlap = count_overlap(ref_keys, query_idx, target_n=len(ref_rep.clonotypes), target_dc=ref_rep.duplicate_count)
print(f"Matched: {overlap.n_normalized:.4f} of target")

# Full VDJbet significance test
analysis = VDJBetOverlapAnalysis(ref_rep, n_mocks=200, seed=42)
results = analysis.analyze_overlap(query_rep)
for r in results:
    print(f"Sample {r.sample_id}: overlap z-score={r.n_z:.2f}, p={r.n_p_emp:.4f}")
```

**Use Cases**:
- Detect TCR sharing between individuals
- Identify public clonotypes in cohort
- Test for unlikely high overlap (potential contamination)

---

### 10. Exact vs. 1-Mismatch Overlap

**What it does**: Distinguish overlaps by matching stringency.

**Configuration**:
- `allow_1mm=False` (default) — Exact junction_aa match only
- `allow_1mm=True` — Include 1-amino-acid substitution variants
- `match_v`, `match_j` — Require V/J gene match (allele-stripped)

**Common Patterns**:
```python
# Exact match only
exact = count_overlap(ref_keys, query_idx, allow_1mm=False)

# 1-mismatch allowed (fuzzy)
fuzzy = count_overlap(ref_keys, query_idx, allow_1mm=True)

# Junction match only, ignore V/J genes
ref_keys_relaxed = make_reference_keys(ref_rep, match_v=False, match_j=False)
vj_free = count_overlap(ref_keys_relaxed, query_idx)
```

**Use Cases**:
- Distinguish true clonal expansion from sequencing noise
- Detect cross-reactive TCRs within 1 mutation
- Focus on core junction identity

---

## Recommended Workflows

### Workflow 1: Quality Control → Gene Usage Characterization

```python
from mir.common.filter import filter_functional
from mir.basic.gene_usage import GeneUsage

# 1. Filter
functional_rep = filter_functional(loaded_rep)

# 2. Compute usage
gu = GeneUsage.from_repertoire(functional_rep)
v_frac = gu.v_fraction("TRB", count="duplicates")

# 3. Compare to reference
factors = gu.correction_factors(reference_gu, "TRB")
print(f"Genes with > 2-fold enrichment: {sum(1 for f in factors.values() if f > 2)}")
```

### Workflow 2: Sample Normalization → Comparison

```python
from mir.common.sampling import downsample, resample_to_gene_usage
from mir.comparative.overlap import make_reference_keys, make_query_index, count_overlap

# 1. Normalize sizes
downsampled = downsample(rep, 1000)

# 2. Resample to match batch
resampled = resample_to_gene_usage(downsampled, batch_gu, "TRB")

# 3. Compute overlap
ref_keys = make_reference_keys(ref_rep)
query_idx = make_query_index(resampled)
overlap = count_overlap(ref_keys, query_idx, target_n=len(ref_rep.clonotypes))
```

### Workflow 3: Significance Testing with VDJbet

```python
from mir.biomarkers.vdjbet import VDJBetOverlapAnalysis

# 1. Filter and resample query
query_rep = filter_functional(query_rep)
query_rep = downsample(query_rep, 1000)

# 2. Run VDJbet
analysis = VDJBetOverlapAnalysis(ref_rep, n_mocks=500, seed=42)
results = analysis.analyze_overlap(query_rep)

# 3. Interpret
for r in results:
    sig = "***" if r.n_p_emp < 0.001 else ("**" if r.n_p_emp < 0.01 else ("*" if r.n_p_emp < 0.05 else ""))
    print(f"{r.sample_id}: overlap={r.n}/{r.n_total}, z={r.n_z:.2f} {sig}")
```

---

## Performance Tips

1. **Caching**: Use `.to_pickle()` / `.from_pickle()` for large repertoires
2. **Filtering**: Apply early to reduce downstream computation
3. **Parallelization**: Use `n_jobs > 1` in VDJBetOverlapAnalysis and compute_overlaps
4. **Gene Usage**: Use `count="clonotypes"` for unweighted analysis (faster)
5. **Graph Construction**: Set reasonable `max_distance` thresholds to avoid explosion

---

## Error Handling & Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` (gene library) | IMGT data not downloaded | `GeneLibrary.load_default()` auto-downloads on first run |
| Empty repertoire after filtering | Too stringent filter | Check V-gene functionality annotations and CDR3 coding status |
| Low overlap p-values despite large n | Sample size not normalized | Apply `downsample()` before overlap analysis |
| Slow k-mer graph | k too small (exponential variants) | Use k=3 or k=4; consider down-sampling |

---

## References

- **AIRR Format**: [AIRR Community Standard](https://docs.airr-community.org/)
- **OLGA**: [Sethna et al. PNAS 2019](https://www.pnas.org/content/116/12/5464)
