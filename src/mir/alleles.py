"""Scalar V/D/J/C allele-notation helpers used by the germline distance lookup.

For frame-level allele operations use ``vdjtools.io.schema.strip_allele`` (a polars
expression). These string helpers drive the per-clonotype resolution cascade
(exact allele → ``*01`` → bare gene → fallback) when gathering baked germline
distances.
"""

from __future__ import annotations

import re

_GENE_BASE_RE = re.compile(r"^\s*([^*\s]+)")


def strip_allele(gene_name: str | None) -> str:
    """Return the gene base without allele suffix (``TRBV6-5*02`` → ``TRBV6-5``)."""
    s = str(gene_name or "").strip()
    if not s:
        return ""
    m = _GENE_BASE_RE.match(s)
    return m.group(1) if m else ""


def allele_with_default(gene_name: str | None, default_allele: str = "01") -> str:
    """Append ``*01`` when no allele is given; preserve an explicit allele.

    ``TRBV6-5`` → ``TRBV6-5*01``; ``TRBV6-5*02`` → ``TRBV6-5*02``.
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
        return f"{base}*{allele or default_allele}"
    base = strip_allele(s)
    return f"{base}*{default_allele}" if base else ""


def allele_to_major(gene_name: str | None) -> str:
    """Normalize to major-allele form ``*01`` (``TRBV6-5*02`` → ``TRBV6-5*01``)."""
    base = strip_allele(gene_name)
    return f"{base}*01" if base else ""


if __name__ == "__main__":
    assert strip_allele("TRBV6-5*02") == "TRBV6-5"
    assert allele_with_default("TRBV6-5") == "TRBV6-5*01"
    assert allele_with_default("TRBV6-5*02") == "TRBV6-5*02"
    assert allele_to_major("TRBV6-5*02") == "TRBV6-5*01"
    assert strip_allele("") == "" and allele_with_default(None) == ""
    print("mir.alleles self-check OK")
