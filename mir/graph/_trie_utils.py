"""Shared helpers for tcrtrie-backed neighborhood and graph search."""

from __future__ import annotations

import typing as t


def validate_metric(metric: str) -> None:
    """Validate supported edit-distance metric names."""
    if metric not in ("hamming", "levenshtein"):
        raise ValueError(f"metric must be 'hamming' or 'levenshtein', got {metric!r}")


def search_limits(metric: str, threshold: int) -> tuple[int, int, int, int]:
    """Map metric/threshold to tcrtrie edit constraints."""
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold!r}")
    if metric == "hamming":
        return threshold, 0, 0, threshold
    return threshold, threshold, threshold, threshold


def resolve_n_jobs(*, n_jobs: int | None, nproc: int | None, default: int = 4) -> int:
    """Resolve worker count while keeping backward compatibility with ``nproc``."""
    resolved = default if n_jobs is None else n_jobs
    if n_jobs is None and nproc is not None:
        resolved = nproc
    if resolved <= 0:
        raise ValueError(f"n_jobs must be >= 1, got {resolved!r}")
    return resolved


def hit_index(hit: t.Any) -> int:
    """Extract sequence index from a tcrtrie hit payload."""
    if isinstance(hit, int):
        return hit
    if isinstance(hit, tuple | list):
        if not hit:
            raise ValueError("empty trie hit")
        return int(hit[0])
    if hasattr(hit, "index"):
        return int(hit.index)
    raise TypeError(f"Unsupported trie hit payload: {type(hit).__name__}")
