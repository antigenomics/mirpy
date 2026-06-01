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


def genes_match(g1: str | None, g2: str | None) -> bool:
    """Match two gene identifiers respecting allele specificity.

    A bare gene (no allele suffix) acts as a wildcard and matches any allele
    of the same base gene.  A specific allele matches only the same allele or
    a bare gene (which, having no allele information, cannot exclude any allele).

    Examples
    --------
    - ``TRAV1``    matches ``TRAV1``, ``TRAV1*01``, ``TRAV1*02``
    - ``TRAV1*02`` matches ``TRAV1*02`` and bare ``TRAV1``
    - ``TRAV1*02`` does NOT match ``TRAV1*01``
    - ``""``       matches only ``""``
    """
    s1 = str(g1 or "").strip()
    s2 = str(g2 or "").strip()
    b1 = strip_allele(s1)
    b2 = strip_allele(s2)
    if b1 != b2:
        return False
    # Same base gene. If either has no allele suffix, treat as wildcard.
    has_allele1 = "*" in s1
    has_allele2 = "*" in s2
    if not has_allele1 or not has_allele2:
        return True
    # Both have explicit alleles — require exact match.
    return s1.split("*", 1)[1] == s2.split("*", 1)[1]
