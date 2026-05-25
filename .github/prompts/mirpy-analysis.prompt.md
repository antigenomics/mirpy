---
name: mirpy-analysis
description: "Run mirpy-backed notebook analysis from user data paths or public/control data instructions, with hypotheses, benchmark estimation, and execution summary."
argument-hint: "Data path(s)/source + optional metadata + workflow or hypotheses to test"
agent: "mirpy-analysis"
model: "GPT-5 (copilot)"
---
Use the mirpy-analysis agent to build and run a reproducible notebook workflow.

Inputs to collect or validate:
- Local data path(s) in AIRR/VDJtools/Adaptive/other mirpy-supported formats.
- Optional metadata schema or table path.
- If no local data: instructions to generate/load public or control data.
- Workflow definition and/or one or more hypotheses to test.

Execution requirements:
- Build dedicated notebook(s) for setup, ingestion/QC, analysis, and interpretation.
- Include deterministic settings and explicit dependency checks.
- Execute notebook cells sequentially and fix run-time errors.
- If data is large, run benchmark chunks first and extrapolate full runtime + peak memory.
- Prompt before full execution when estimate is above 10-15 minutes on 4-8 cores or above ~12-16 GB RAM.

On request, summarize:
- artifacts created,
- hypothesis outcomes,
- measured/estimated runtime and memory,
- caveats and next actions.
