---
description: "Use when creating, editing, or running Jupyter notebooks (.ipynb), especially when producing notebook JSON cell content or executing notebook cells in VS Code."
applyTo: "**/*.ipynb"
---
# Notebook And IPYNB Guidelines

- Use JSON notebook structure when generating notebook content.
- Each cell object must include `metadata.language`.
- Existing cells must keep their `metadata.id` values.
- New cells may omit `metadata.id`.
- Keep notebook JSON valid and logically ordered.
- Ensure each code cell includes a brief comment describing what the cell does.

## Reproducibility

- Set and document random seeds for notebook workflows that involve sampling, random initialization, or stochastic methods.
- At notebook start, print key environment versions used by the analysis (Python and core scientific packages in scope).
- Prefer deterministic ordering before comparisons, metrics, or merges.
- Avoid hidden state assumptions: a notebook should run top-to-bottom from a clean kernel.

## User-Facing Communication

- When referring to notebook positions, use cell numbers starting from 1.
- Do not expose internal cell IDs in user-facing messages.

## Execution Workflow

- Configure the notebook kernel before first execution in a session.
- Execute code cells sequentially, one by one, in VS Code.
- Run only code cells, not markdown cells.
- Measure and record runtime for each executed cell.
- Use a 5-10 minute cap for any single cell when no explicit cap is provided.
- If a cell exceeds its cap, stop and optimize before continuing.
- If package imports fail in notebook execution, install missing packages to the notebook kernel and retry.
- Restart the notebook kernel only when needed after package installation or state corruption.
- After significant notebook edits, perform a clean restart and run-all verification before declaring success.

## Editing Scope

- Prefer minimal notebook edits that preserve existing outputs/metadata unless the task asks for cleanup.
- Keep exploratory cells reproducible (imports, setup, deterministic ordering where practical).
- Keep cells single-purpose (load, transform, evaluate, visualize) and split large cells into explicit stages.
- For heavy logic, prefer moving reusable code into Python modules and keeping notebooks orchestration-focused.

## Performance And Diagnostics

- For expensive cells, include timing checkpoints per stage rather than only total runtime.
- When performance claims are made, include both runtime and memory observations when feasible.
- Use bounded parallelism defaults and expose thread/process count as an explicit parameter.

## Plotting And Figure Quality

- For plotting cells, use publication-ready styling: consistent font selection, panel/background style, and color palette.
- Prefer colorblind-safe palettes and journal-style aesthetics suitable for Nature/Science-style figures.
- Ensure labels are readable in scatter plots, volcano plots, and graph visualizations.
- Use label placement strategies that reduce overlap (repel or dodge-like placement).
- For barplots that compare groups, keep binwidth consistent; for overlap-prone grouped bars, prefer dodge-style placement.
- Consider boxplot plus beeswarm-style overlays when both summary and point-level structure are important.
- For notebook cells that produce multiple figures, prefer compact multi-panel layouts with clear spacing and legible axes.

## Graph Visualization Guidance

- Avoid overclumped graph layouts.
- For large graphs, consider showing a representative subset of nodes/edges.
- Prefer force-directed layouts (spring/charge) and include zoomed-in or inset views when useful.
- Consider MDS-like layouts when distance structure is central and force layouts become unreadable.

## End-Of-Notebook Diagnostics

- Provide a diagnostics summary after notebook execution.
- Include per-cell runtime, stalled or capped cells, optimization actions taken, and any remaining performance or rendering risks.

## Source Control And Safety

- Keep notebook outputs intentional; avoid committing noisy or excessively large transient outputs unless outputs are required artifacts.
- Do not hardcode local machine-specific absolute paths or credentials in notebook cells.
- Prefer parameterized paths and environment-driven configuration for data and cache locations.
