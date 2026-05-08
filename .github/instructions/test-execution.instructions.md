---
description: "Use when running, debugging, or discussing tests in this repository. Covers anti-stall execution, time and memory diagnostics, test tiering, and focused rerun strategy."
---
# Test Execution Guidelines

- Prefer targeted pytest runs before full-suite runs.
- Start with the smallest affected scope, then expand only if needed.
- Run from the repository root unless a test explicitly depends on another working directory.
- Prefer VS Code built-in testing workflows first.
- Use shell-based pytest invocation only if VS Code test tooling cannot satisfy the task.

## Test Tiers

- Keep unit tests fast and deterministic.
- Split benchmark coverage into explicit fast and slow groups.
- Use this practical default if no explicit cap is defined:
- Unit tests: usually seconds, and never intentionally long-running.
- Fast benchmarks: target completion within 5 minutes.
- Slow benchmarks: cap at 10 minutes.
- If measured runtime exceeds the tier cap, optimize before expanding scope.

## Anti-Stall Policy

- Prevent stall-prone runs with explicit timeout behavior or bounded scope.
- Do not leave open-ended benchmark jobs running without a defined time budget.
- If a run approaches the time cap, stop and optimize test setup, data size, parametrization, or algorithmic hot paths.

## Command Patterns

- Single test file: `pytest -s tests/test_name.py`
- Single test function: `pytest -s tests/test_name.py::test_case`
- Pattern subset: `pytest -s -k "keyword"`
- Verbose details when needed: add `-v`

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

## Failure Handling

- If a test fails, report the failing test path and assertion summary first.
- Minimize reruns by rerunning only the failing test(s) after a fix.
- Avoid unrelated formatting or refactors while fixing test failures.
- Include stall signals and timeout symptoms in failure analysis.

## Safety And Scope

- Do not modify baseline datasets, fixtures, or generated assets unless the task explicitly requires it.
- If a fix requires broad behavior changes, explain expected blast radius before running broad test sweeps.

## End-Of-Run Diagnostics

- Always provide a concise diagnostics summary at the end of test work.
- Include: test tier, executed subset, wall-clock runtime, memory footprint (if measured), failures/timeouts, and next optimization step when caps were exceeded.
