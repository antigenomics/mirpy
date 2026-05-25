"""Utilities for sharing NumPy arrays across worker processes.

This module provides a thin wrapper around ``multiprocessing.shared_memory``
for read-only fan-out workloads where large arrays are prepared once in the
parent process and attached in child workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np


@dataclass(frozen=True)
class SharedArraySpec:
    """Descriptor needed to attach to a shared-memory NumPy array."""

    name: str
    shape: tuple[int, ...]
    dtype: str


def fixed_bytes_array(values: list[str]) -> np.ndarray:
    """Encode variable-length strings into a fixed-width ASCII byte array."""
    max_len = max((len(v.encode("ascii", errors="ignore")) for v in values), default=1)
    width = max(1, max_len)
    return np.asarray(values, dtype=f"S{width}")


def create_shared_array(array: np.ndarray) -> tuple[SharedArraySpec, shared_memory.SharedMemory]:
    """Copy *array* into shared memory and return attach spec + handle."""
    contiguous = np.ascontiguousarray(array)
    shm = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
    shared_view = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=shm.buf)
    shared_view[...] = contiguous
    spec = SharedArraySpec(name=shm.name, shape=tuple(contiguous.shape), dtype=str(contiguous.dtype))
    return spec, shm


def attach_shared_array(spec: SharedArraySpec) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    """Attach to a shared-memory array described by *spec*."""
    shm = shared_memory.SharedMemory(name=spec.name)
    arr = np.ndarray(spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf)
    return arr, shm


def close_unlink_many(shm_handles: list[shared_memory.SharedMemory]) -> None:
    """Close/unlink shared memory handles, suppressing stale-handle races."""
    for shm in shm_handles:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
