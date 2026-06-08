# TODO

## Resolve `v_call`/`j_call` vs `v_gene`/`j_gene` naming confusion

**Problem.** mirpy uses two names for the same fields:

- **Internal canonical** (`Clonotype.v_gene` / `j_gene` / `d_gene` / `c_gene`,
  used throughout `mir/`): `v_gene`, `j_gene`, …
- **AIRR Rearrangement standard** (input files, `AIRRParser`): `v_call`,
  `j_call`, `d_call`, `c_call`.

The two are bridged ad hoc in `mir/common/parser.py`:

- `_VDJTOOLS_TO_AIRR` maps input column names (`v`, `v_call`, …) → internal
  `v_gene` (note the map's name is itself misleading — it maps *to* mirpy's
  internal names, not to AIRR).
- `AIRRParser.parse_inner` *also* renames `v_call`→`v_gene` separately.
- `single_cell_parser` reads `row.get("v_gene") or row.get("v_call")`.

This duplication is what caused the file-path bug fixed in commit `ebcf23d`
(`AIRRParser.parse(<file>)` skipped the rename + locus filter that only lived in
`parse_inner`). The split keeps inviting that class of bug.

**Why it matters.** AIRR is the interop standard; users expect `v_call`. Internally
everything says `v_gene`. Every new parser / exporter / API has to remember which
side it's on, and the mapping lives in two places.

**Proposed resolution (decide direction first):**

- *Option A — adopt AIRR names internally.* Rename `Clonotype.v_gene` →
  `v_call` (etc.) with deprecated aliases. Most standards-aligned; largest blast
  radius (touches `mir/distances`, `mir/common`, `mir/graph`, tests, notebooks).
- *Option B — keep `v_gene` internally, centralize the boundary.* Make a single
  normalization layer the *only* place AIRR↔internal names are translated (used
  by both the file and DataFrame paths, and by exporters), and rename
  `_VDJTOOLS_TO_AIRR` → something like `_INPUT_TO_INTERNAL`. Smaller change;
  removes the duplicate rename in `AIRRParser.parse_inner`.

**Scope / acceptance:**

- One canonical mapping, referenced everywhere (no per-parser ad-hoc renames).
- `Clonotype` accepts both spellings (alias) regardless of chosen direction.
- AIRR round-trip test: read `v_call` file → embed → export `v_call` unchanged.
- Update `skills/mirpy/references/allele-notation.md` and the parser docstrings.

Tracked because it is cross-cutting tech debt, not a one-line fix.
