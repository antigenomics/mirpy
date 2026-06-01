# Gene and Allele Notation in mirpy

## Matching semantics — `genes_match(g1, g2)`

```python
from mir.common.alleles import genes_match
```

| g1 | g2 | Result |
| --- | --- | --- |
| `TRAV1` (bare) | `TRAV1*01` | `True` — bare = wildcard |
| `TRAV1` (bare) | `TRAV1*02` | `True` — bare = wildcard |
| `TRAV1*02` | `TRAV1*02` | `True` — exact match |
| `TRAV1*02` | `TRAV1` (bare) | `True` — bare on either side = wildcard |
| `TRAV1*01` | `TRAV1*02` | `False` — both explicit, different alleles |
| `TRAV1` | `TRAV2` | `False` — different base gene |

Used automatically in all V/J-restricted paths: `build_edit_distance_graph`,
`compute_neighborhood_stats`, `metaclonotypes_from_*`, `associate_clonotype_*`,
and `TcrDist.find_metaclonotypes`.

## Library resolution chain (distance lookups)

When a V/J gene is looked up in a pre-computed distance or embedding library,
mirpy tries three levels in order:

1. **Exact allele** — `TRBV5-1*07` as provided
2. **Major allele** — `TRBV5-1*01` (fallback for minor alleles absent from library)
3. **Bare gene** — `TRBV5-1` (fallback for libraries without allele resolution)
4. **Not found** — returns `float('nan')`; propagates to overall distance/embedding

This applies to `GermlineAligner.gene_dist()` (used by `TcrDist`) and the TCREmp
embedding lookup.

## Utility functions

```python
from mir.common.alleles import strip_allele, allele_with_default, allele_to_major, genes_match

strip_allele("TRBV5-1*02")      # → "TRBV5-1"
allele_with_default("TRBV5-1")  # → "TRBV5-1*01"  (preserves explicit alleles)
allele_to_major("TRBV5-1*02")   # → "TRBV5-1*01"  (always normalises to *01)
genes_match("TRBV5-1", "TRBV5-1*02")  # → True  (bare = wildcard)
```

**Never use `gene.split("*")[0]`** — use `strip_allele()` instead for robustness.
