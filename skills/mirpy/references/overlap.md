# Pairwise Sample Overlap Reference

Pairwise overlap metrics quantify TCR repertoire sharing between samples, supporting
both exact clonotype matching and approximate (edit-distance) matching. Key references:

- High-resolution repertoire analysis: Ritvo et al. (2018) Proc. Natl. Acad. Sci. USA PMID:30158170 doi:10.1073/pnas.1808594115
- Aging cohort: Britanova et al. (2016) J. Immunol. PMID:27183615 doi:10.4049/jimmunol.1600005

## Single Pair

```python
from mir.comparative.overlap import pairwise_overlap

# Exact matching
r = pairwise_overlap(rep1, rep2)

# Approximate matching — Hamming distance 1 (1 substitution)
r_h1 = pairwise_overlap(rep1, rep2, metric="hamming", threshold=1)

# Approximate matching — Levenshtein distance 1 (any single edit)
r_l1 = pairwise_overlap(rep1, rep2, metric="levenshtein", threshold=1)

print(r.jaccard, r.d_similarity, r.f_similarity, r.morisita_horn)
print(r.f2_similarity, r.correlation)  # nan for approximate matching
```

`PairwiseOverlapResult` fields:

| Field | Description |
|---|---|
| `n1`, `n2` | unique clonotypes in each sample |
| `n1_matched`, `n2_matched` | clones with ≥1 match in the other sample |
| `f1_overlap`, `f2_overlap` | total frequency of matched clones |
| `jaccard` | n12 / (n1 + n2 − n12) |
| `szymkiewicz_simpson` | min(n1_matched, n2_matched) / min(n1, n2) |
| `d_similarity` | n12 / sqrt(n1 × n2) |
| `f_similarity` | sqrt(f1_overlap × f2_overlap) |
| `morisita_horn` | 2 Σ(p_i q_i) / (D1 + D2) |
| `correlation` | Pearson r of overlap frequencies (NaN for approximate) |
| `f2_similarity` | Σ sqrt(p_i × q_i) over matched pairs (NaN for approximate) |
| `mode` | "exact", "hamming:N", "levenshtein:N" |
| `is_approximate` | True when threshold > 0 |

Use `result.as_dict()` to convert all fields to a plain dict for DataFrame construction.

For approximate matching (threshold > 0), `correlation` and `f2_similarity` are `nan`.
Jaccard and D-metric use the geometric mean of n1_matched and n2_matched for symmetry.

## Pairwise Matrix (Cohort)

```python
from mir.comparative.overlap import pairwise_overlap_matrix

# Returns a long-format DataFrame with one row per ordered pair (i < j)
df = pairwise_overlap_matrix(
    reps, sample_ids=ids,
    metric="exact",     # or "hamming" / "levenshtein"
    threshold=0,
    n_jobs=-1,          # -1 = all physical cores
)

# Pivot to symmetric NxN matrix of a single metric
pivot = df.pivot(index="sample_id_1", columns="sample_id_2", values="f_similarity")
```

## Dissimilarity For UMAP / Clustering

```python
import numpy as np
from umap import UMAP

f_vals = df.pivot(index="sample_id_1", columns="sample_id_2", values="f_similarity")
f_mat  = f_vals.to_numpy()
n = len(reps)
dissim = np.zeros((n, n))
dissim[np.triu_indices(n, 1)] = 1.0 - f_mat[np.triu_indices(n, 1)]
dissim += dissim.T  # symmetric

embedding = UMAP(n_components=2, metric="precomputed", random_state=42).fit_transform(dissim)
```

Dissimilarity conventions:
- **D-metric**: `max(D) − D`
- **F-metric**: `1 − F`

## Parallel Workers (n_jobs)

- `n_jobs=-1` (default): all physical cores (uses `psutil.cpu_count(logical=False)`)
- `n_jobs=1`: serial — useful for deterministic profiling
- In `pairwise_overlap`: parallelises trie search within a single pair (chunk workers)
- In `pairwise_overlap_matrix`: parallelises across pairs (matrix workers)
- For a single pair, forcing many workers can be slower due to process startup; use `n_jobs=1` unless the query side is very large.

## Overlap Spaces

- Approximate matching (`threshold > 0`) is supported only for `aa` and `aavj` spaces.
- In amino-acid overlap spaces, non-coding clonotypes are excluded from overlap matching.
- Similarity outputs are primary; use metric transforms only when distance-like inputs are required.

## VDJBet Harmonisation

The existing `count_overlap` / `compute_overlaps` / `make_reference_keys` / `make_query_index`
API used by `VDJBetOverlapAnalysis` is unchanged. `pairwise_overlap` builds on top of the same
`make_query_index` primitive.
