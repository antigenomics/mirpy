"""Utilities for harmonizing V/D/J/C gene allele notation."""

from __future__ import annotations

import re


_GENE_BASE_RE = re.compile(r"^\s*([^*\s]+)")


def strip_allele(gene_name: str | None) -> str:
    """Return gene base without allele suffix.

    Examples
    --------
    - ``TRBV6-5*02`` -> ``TRBV6-5``
    - ``TRBV6-5`` -> ``TRBV6-5``
    """
    s = str(gene_name or "").strip()
    if not s:
        return ""
    m = _GENE_BASE_RE.match(s)
    if m is None:
        return ""
    return m.group(1)


def allele_with_default(gene_name: str | None, default_allele: str = "01") -> str:
    """Ensure allele suffix exists while preserving explicit allele values.

    Examples
    --------
    - ``TRBV6-5`` -> ``TRBV6-5*01``
    - ``TRBV6-5*02`` -> ``TRBV6-5*02``
    """
    s = str(gene_name or "").strip()
    if not s:
        return ""
    if "*" in s:
        base, allele = s.split("*", 1)
        base = strip_allele(base)
        allele = allele.strip()
        if not base:
            return ""
        if not allele:
            allele = default_allele
        return f"{base}*{allele}"
    base = strip_allele(s)
    if not base:
        return ""
    return f"{base}*{default_allele}"


def allele_to_major(gene_name: str | None) -> str:
    """Normalize a gene string to major allele form ``*01``.

    Examples
    --------
    - ``TRBV6-5`` -> ``TRBV6-5*01``
    - ``TRBV6-5*02`` -> ``TRBV6-5*01``
    - ``TRBV6-5*01`` -> ``TRBV6-5*01``
    """
    base = strip_allele(gene_name)
    if not base:
        return ""
    return f"{base}*01"
