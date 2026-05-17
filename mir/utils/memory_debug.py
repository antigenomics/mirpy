"""Process memory diagnostics helpers for long-lived Python sessions."""

from __future__ import annotations

import os
from typing import Any


def process_memory_snapshot(tag: str = "snapshot") -> dict[str, Any]:
    """Return a lightweight snapshot of the current process memory state."""
    import psutil

    proc = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    rss = proc.memory_info().rss
    try:
        uss = proc.memory_full_info().uss
    except Exception:
        uss = None

    return {
        "tag": tag,
        "pid": proc.pid,
        "rss_gb": rss / (1024 ** 3),
        "uss_gb": (uss / (1024 ** 3)) if uss is not None else None,
        "available_gb": vm.available / (1024 ** 3),
    }


def top_python_processes(limit: int = 10) -> list[dict[str, Any]]:
    """Return top resident Python processes on the host by RSS."""
    import psutil

    rows: list[dict[str, Any]] = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline", "memory_info"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = proc.info.get("cmdline") or []
            cmd = " ".join(cmdline)
            if "python" not in name and "python" not in cmd.lower():
                continue
            rss = proc.info["memory_info"].rss
            rows.append(
                {
                    "pid": int(proc.info["pid"]),
                    "rss_gb": rss / (1024 ** 3),
                    "command": cmd,
                }
            )
        except Exception:
            continue

    rows.sort(key=lambda r: r["rss_gb"], reverse=True)
    return rows[: max(1, int(limit))]
