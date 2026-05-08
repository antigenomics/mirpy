---
description: "Use when running, debugging, or discussing tests in this repository. Covers anti-stall execution, time and memory diagnostics, test tiering, and focused rerun strategy."
---
# Test Execution Guidelines

- Prefer targeted runs before full-suite runs.
- Start with the smallest affected scope, then expand only if needed.
- Run from the repository root unless a test explicitly depends on another working directory.
- Prefer VS Code built-in test tooling (`runTests`) first.
- Use terminal-based pytest invocation only when the VS Code test tool cannot satisfy the task.
- Never leave a test execution unbounded: every explicit run must have a clear scope and time budget.

## Copilot Workflow Quality

- Before running tests, state the acceptance target in one line (what must pass and why).
- Prefer incremental edit-and-validate cycles over large refactors followed by broad reruns.
- After each fix, rerun the narrowest failing scope first, then expand only if needed.
- When uncertain about behavior, surface assumptions before changing test-sensitive logic.

## Progress Reporting In Chat

- Provide a short progress update before starting a run that states: scope, tool choice, and timeout budget.
- If a run is long, report meaningful status updates frequently (for example, when switching tiers, after completion of a focused subset, or on timeout/failure).
- Do not stay silent while work is still running; always report what is happening and what will happen next.
- End each run with a concise diagnostics summary.

## Test Tiers

- Keep unit tests fast and deterministic.
- Split benchmark coverage into explicit fast and slow groups.
- Use this practical default if no explicit cap is defined:
- Unit tests: usually seconds, and never intentionally long-running.
- Fast benchmarks: target completion within 5 minutes.
- Slow benchmarks: cap at 10 minutes.
- If measured runtime exceeds the tier cap, optimize before expanding scope.

## Determinism And Environment

- Prefer deterministic test behavior (fixed seeds, stable ordering, explicit tolerances for numeric assertions).
- Report key environment details when they can influence outcomes (Python version and relevant dependency versions).
- If failures are environment-sensitive, separate code regressions from environment/configuration issues in diagnostics.

## Anti-Stall Policy

- Prevent stall-prone runs with explicit timeout behavior and bounded scope.
- Default timeout budget for non-trivial runs: 5-10 minutes (300000-600000 ms).
- Do not run open-ended benchmark jobs without a defined time budget.
- If a run reaches timeout or appears stuck, treat it as stalled and intervene immediately.
- Intervention order:
- Inspect current output for hang signals.
- Kill the stuck terminal/process.
- Rerun a smaller, targeted subset with the same or tighter timeout.
- If a run approaches the time cap, stop and optimize test setup, data size, parametrization, or algorithmic hot paths.

## Tool Selection Priority

- First choice: VS Code test tool (`runTests`) for focused file/test execution and structured failure output.
- Second choice: terminal commands when special environment setup or custom invocation is required.
- For CMake projects, use dedicated CMake test tools instead of terminal test commands.

## Command Patterns

- Single test file: `pytest -s tests/test_name.py`
- Single test function: `pytest -s tests/test_name.py::test_case`
- Pattern subset: `pytest -s -k "keyword"`
- Verbose details when needed: add `-v`

## Terminal Execution Guardrails

- Use explicit timeouts for test commands (typically 300000-600000 ms).
- Prefer one-shot synchronous runs with timeout for finite test jobs.
- If a command times out and moves to background, check output, then kill it if no forward progress is observed.
- Avoid launching multiple overlapping long-running test commands.

## Benchmarks And Slow Tests

- Do not run benchmark-marked tests by default.
- Only run benchmarks when explicitly requested.
- For benchmark workflows in this repository, set `RUN_BENCHMARK=1` when required by docs or test markers.
- For benchmark runs without explicit limits, enforce a 5-10 minute maximum depending on tier.

## Runtime And Memory Measurement

- Measure and report execution time for every explicit test run.
- When performance is in scope, report memory footprint as well.
- Include enough context to compare runs: selected tests, dataset size, and environment assumptions.
- For regressions, prioritize memory-safe and time-safe optimizations before broad reruns.
- If a run is unexpectedly slow, capture where time is spent (test subset, fixture setup, data loading, algorithmic hot spots) before expanding scope.

## Failure Handling

- If a test fails, report the failing test path and assertion summary first.
- Minimize reruns by rerunning only the failing test(s) after a fix.
- Avoid unrelated formatting or refactors while fixing test failures.
- Include stall signals and timeout symptoms in failure analysis.
- Use structured failure diagnostics when available (for example, dedicated failure-detail tools) before attempting broad reruns.

## Safety And Scope

- Do not modify baseline datasets, fixtures, or generated assets unless the task explicitly requires it.
- If a fix requires broad behavior changes, explain expected blast radius before running broad test sweeps.

## End-Of-Run Diagnostics

- Always provide a concise diagnostics summary at the end of test work.
- Include: tool used (VS Code test tool or terminal), test tier, executed subset, timeout budget, wall-clock runtime, memory footprint (if measured), failures/timeouts, and next optimization step when caps were exceeded.

## Python And Notebook-Backed Logic

- When notebook logic is refactored, ensure critical logic paths are validated by module-level tests when feasible.
- Prefer adding or updating focused unit tests for extracted helper functions instead of relying on notebook-only execution.
