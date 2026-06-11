# TCRdist Reference

TCRdist computes weighted V-gene + CDR3 distances between TCR clonotypes, enabling
neighbourhood search, radius-based thresholding, and metaclonotype discovery. The
distance metric and its application to antigen-specific repertoire analysis are
described in:

- Template-based structural modeling: Shcherbinin et al. (2023) Front. Immunol. PMID:37649481 doi:10.3389/fimmu.2023.1224969
- Thymic selection: Luppov et al. (2025) Front. Immunol. PMID:41050667 doi:10.3389/fimmu.2025.1605170
- **GLIPH** CDR3 motif clustering: Glanville *et al.* (2017) *Nature* — PMID:[28636589](https://pubmed.ncbi.nlm.nih.gov/28636589/), doi:[10.1038/nature22976](https://doi.org/10.1038/nature22976)
- **GLIPH2** (large-scale): Huang *et al.* (2020) *Nat. Biotechnol.* — PMID:[32341563](https://pubmed.ncbi.nlm.nih.gov/32341563/), doi:[10.1038/s41587-020-0505-4](https://doi.org/10.1038/s41587-020-0505-4)

## 24. TCRdist — Alignment-Based Clonotype Distance

Use `TcrDist` from `mir.distances.tcrdist` to compute weighted V-gene + CDR3
distances between TCR clonotypes, find per-clonotype radii, and define
metaclonotypes via radius-threshold clustering.

```python
from mir.distances.tcrdist import TcrDist
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire

# Build with defaults (computes V/J germline distances once, ~3–10 s)
td = TcrDist.from_defaults(
    "TRB", "human",
    w_v=1.0,            # V-gene germline weight
    w_j=0.0,            # J-gene weight (0 = ignore)
    w_cdr3=3.0,         # CDR3/junction_aa weight
    fixed_gaps=(3, 4, -4, -3),  # C-accelerated JunctionAligner (default)
    # fixed_gaps="Mid"  → midpoint gap per pair
    # fixed_gaps=None   → full BioPython DP alignment (slowest)
    gap_penalty=-4.0,
)

# One-to-one
cln1 = Clonotype(v_call="TRBV19*01", j_call="TRBJ2-7*01", junction_aa="CASSIRSSYEQYF")
cln2 = Clonotype(v_call="TRBV19*01", j_call="TRBJ2-7*01", junction_aa="CASSIRASYEQYF")
d = td.dist(cln1, cln2)   # float, symmetric, non-negative

# One-to-many
refs = list(rep.clonotypes)
row = td.dist_one_to_many(cln1, refs)  # shape: (K,)

# Many-to-many (N×K matrix, parallel when fixed_gaps is a list)
mat = td.dist_matrix(queries, refs, n_jobs=4)   # shape: (N, K)

# Self-distance matrix
mat_self = td.self_dist_matrix(list(rep.clonotypes), n_jobs=4)  # shape: (N, N)
```

### Radius computation

For each clonotype, compute the p-th percentile of its distances to a
background set (used as the neighbourhood search threshold):

```python
import numpy as np
from mir.basic.pgen import OlgaModel

# Generate OLGA background sequences
model = OlgaModel(locus="TRB", species="human")
bg_seqs, _ = model.generate_sequences_counted(10_000, n_jobs=4, seed=42)
bg_clns = [Clonotype(junction_aa=s, locus="TRB") for s in bg_seqs]

hits = [c for c in rep.clonotypes if c.v_call and c.junction_aa]
radii = td.compute_radius(hits, bg_clns, percentile=50, n_jobs=4)
# radii: float64 array of shape (len(hits),)

# Sequences with small radii are in convergent (antigen-driven) neighbourhoods
threshold = float(np.percentile(radii, 25))
```

### Metaclonotype discovery

```python
from mir.common.metaclonotype import summarize_metaclonotypes

# All-vs-all clustering (each clonotype as its own seed)
meta = td.find_metaclonotypes(
    rep,
    max_distance=threshold,       # float radius threshold
    match_v_call=False,           # optionally restrict to same V
    match_j_call=False,
    cluster_prefix="tcrdist_mc",
    n_jobs=4,
)
print(meta.n_clusters)

# Only cluster around enriched/selected seeds
enriched_ids = [c.sequence_id for c in hits if radius_for_c <= threshold]
meta = td.find_metaclonotypes(
    rep,
    representative_ids=enriched_ids,
    max_distance=threshold,
    n_jobs=4,
)

# Aggregate counts per cluster
summary = summarize_metaclonotypes(rep, meta)
# Columns: cluster_id, n_members, representative_junction_aa,
#           representative_v_gene, representative_j_gene,
#           duplicate_count, umi_count
```

**Scale note**: V-gene distances (BioPython BLOSUM62, unscaled) and CDR3 distances
(JunctionAligner BLOSUM62 × 10) are on different absolute scales. Default weights
w_v=1.0, w_cdr3=3.0 ensure CDR3 divergence dominates. Adjust for a custom balance.

**`cdrs_only=True`** raises `NotImplementedError`. Use `v_alignment_type="full_germline"` (default).

**Performance** (Apple M3, TRB, measured 2026-05-22):

| Mode              | Dataset | n_jobs | Wall time | Pairs/s      |
|-------------------|---------|--------|-----------|--------------|
| `fixed_gaps` list | 1K×1K   | 1      | 0.036 s   | 27.9 M/s     |
| `fixed_gaps` list | 5K×5K   | 1      | 0.894 s   | 28.0 M/s     |
| `fixed_gaps` list | 1K×1K   | 8      | 0.013 s   | 75.9 M/s     |
| `"Mid"` (Python)  | 1K×1K   | 1      | 11.7 s    | 85 k/s       |
| `None` (BioPython)| 100×100 | 1      | 0.278 s   | 36 k/s       |

`fixed_gaps` list is ~330× faster than `"Mid"` and ~780× faster than full BioPython DP.
CDR3 scoring uses JunctionAligner.score_matrix (C, GIL released); true thread parallelism gives ~2.7× speedup at n_jobs=8.

**Notebook**: `notebooks/tcrdist_analysis.ipynb` — influenza GILGFVFTL example, distance heatmap, UMAP, hierarchical clustering, motif logos, gap mode comparison.
