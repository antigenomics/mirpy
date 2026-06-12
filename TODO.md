# TODO

Open follow-ups. Closed items are removed (history lives in git).

## Benchmarks / CI

- [ ] **Hosted-runner Node 20 deprecation.** `actions/cache@v4`, `actions/checkout@v4`,
  `actions/setup-python@v5` warn about Node 20 EOL. Bump when newer majors land.
- [ ] **COVID association reference concordance.** `ref_overlap` (top-100 Fisher hits
  vs `covid_associated_clonotypes.csv`) is low and varies with sample count
  (0–10 in recent runs). Not asserted today; investigate whether the reference
  clonotype representation matches the scan's `junction_aa` keys.
- [ ] **Total-budget enforcement (optional).** Tiers currently rely on per-test
  timeout guards + measured totals (see `benchmarks.md`). Consider a session-level
  wall-clock guard if a tier starts creeping past its 15 / 30 min budget.
- [ ] **Degenerate z in `TestQ1Q15Integration::test_d15_z_exceeds_d0_z`**
  (very_slow tier). At `pool n=20_000`, `n_mocks=100` the Q1↔LLW mock overlaps
  collapse to a constant, so `VDJBetOverlapAnalysis._z_p` returns `z=inf` for both
  days (`inf > inf` fails the assertion). Pre-existing degeneracy in the mock
  sampling — not the gene-naming change (loaded data is identical: Q1 genes are
  bare → `*01` under both old and new normalization). Fix by sizing the pool /
  `n_mocks` so mock variance > 0, or by asserting on a tie-safe statistic.

## Docs build (pre-existing, unrelated to gene-naming)

- [ ] **Circular import under autodoc.** `mir.basic.gene_usage` ↔
  `mir.common.sampling` (`GeneUsage`) raises during the Sphinx autodoc import of
  `gene_usage`. Build still succeeds; resolve the import cycle to clear the warning.
- [ ] **`docs/mir.ml.rst` not in any toctree.** Add it to a toctree or drop the stub.

## Notebooks

- [ ] **Regenerate notebook outputs.** Code cells were migrated `v_gene`→`v_call`,
  but committed cell *outputs* (column labels in printed frames) were swept textually,
  not re-executed. Re-run notebooks from a clean kernel to refresh real outputs.
