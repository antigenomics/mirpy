"""Utilities for harmonizing V/D/J/C gene allele notation."""

from __future__ import annotations

import re


_GENE_BASE_RE = re.compile(r"^\s*([^*\s]+)")


def allele_to_major(gene_name: str | None) -> str:
    """Normalize a gene string to major allele form ``*01``.

    Examples
    --------
    - ``TRBV6-5`` -> ``TRBV6-5*01``
    - ``TRBV6-5*02`` -> ``TRBV6-5*01``
    - ``TRBV6-5*01`` -> ``TRBV6-5*01``
    """
    s = str(gene_name or "").strip()
    if not s:
        return ""
    m = _GENE_BASE_RE.match(s)
    if m is None:
        return ""
    return f"{m.group(1)}*01"
