"""Region annotation of germline V/J genes via arda (build-time / optional).

`arda <https://github.com/antigenomics/arda>`_ annotates immune-receptor V/J
region boundaries.  mirpy uses it once, at **resource-build time**, to compute
the germline-encoded FR/CDR amino-acid subsequences of each gene-library allele:

* germline **V** gene → ``fwr1``, ``cdr1``, ``fwr2``, ``cdr2``, ``fwr3``
* germline **J** gene → ``fwr4`` plus ``jcdr3`` (the J residues 5' of FR4, i.e.
  the J's contribution to CDR3)

The result is baked into the companion ``region_annotations.txt`` resource by
:mod:`mir.resources.gene_library.build_region_annotations`, so similarity and
embedding at runtime never need arda or mmseqs2.

arda accepts partial (truncated) sequences with **no coverage filter**, so each
bare germline V or J nucleotide sequence is annotated on its own — only the
regions that fall inside the query's coverage are returned, which is exactly the
V-side regions for a V gene and the J-side regions for a J gene.

This module requires the optional ``arda`` extra (``pip install mirpy-lib[arda]``)
and the ``mmseqs2`` binary on ``PATH`` (``conda install -c bioconda mmseqs2``).
Importing this module is cheap; arda is imported lazily inside
:func:`annotate_gene_library`.
"""

from __future__ import annotations

import warnings

from mir.basic.mirseq import translate_linear
from mir.common.gene_library import GeneLibrary

#: Region columns extracted for a V gene, in 5'->3' order.
V_REGIONS: tuple[str, ...] = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3")

#: Region columns extracted for a J gene.
J_REGIONS: tuple[str, ...] = ("jcdr3", "fwr4")

#: Full region column order written to the companion resource.
ALL_REGIONS: tuple[str, ...] = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "jcdr3", "fwr4")

#: Locus pairs treated as equivalent when arda's call differs from the library.
#: TRA/TRD share the dual ``TRAV.../DV`` genes, so a TRD-listed gene mapping to
#: TRA (or vice versa) is correct, not a mismap.
_EQUIVALENT_LOCI: frozenset[frozenset[str]] = frozenset({frozenset({"TRA", "TRD"})})


def _locus_ok(library_locus: str, called_locus: str) -> bool:
    """True if *called_locus* matches *library_locus* or is a known equivalent."""
    if not called_locus or called_locus == library_locus:
        return True
    return frozenset({library_locus, called_locus}) in _EQUIVALENT_LOCI


def _require_arda():
    """Import and return ``arda.annotate.mapper.annotate_records`` or raise."""
    try:
        from arda.annotate.mapper import annotate_records
    except ImportError as exc:  # pragma: no cover - exercised only without arda
        raise ImportError(
            "Region annotation requires the optional 'arda' package and the "
            "mmseqs2 binary. Install with `pip install mirpy-lib[arda]` and "
            "`conda install -c bioconda mmseqs2`. arda is only needed to (re)build "
            "region annotations or annotate a custom library; plain embedding and "
            "similarity use the bundled region_annotations.txt and never import arda."
        ) from exc
    return annotate_records


def _jcdr3_aa(query_nt: str, fwr4_start: int) -> str:
    """Translate the J residues 5' of FR4 (the J contribution to CDR3).

    ``fwr4_start`` is arda's 1-based nt start of FR4 on the J germline query.
    arda translates FR4 in frame 0 of the slice starting at ``fwr4_start``, so
    the J reading frame has codon boundaries at positions ``(fwr4_start - 1) % 3``;
    we translate from there up to (but not including) the FR4 start codon.
    """
    if not fwr4_start or fwr4_start <= 1:
        return ""
    phase = (fwr4_start - 1) % 3
    return translate_linear(query_nt[phase : fwr4_start - 1]).rstrip("_")


def annotate_gene_library(
    lib: GeneLibrary,
    organism: str,
    *,
    sensitivity: float = 7.0,
) -> dict[str, dict[str, str]]:
    """Annotate FR/CDR amino-acid subsequences for a library's V and J alleles.

    Runs arda over every V and J germline nucleotide sequence of *organism*
    present in *lib* and extracts the germline-encoded region amino-acid
    subsequences.

    Args:
        lib: Gene library whose V/J entries (for *organism*) are annotated.
        organism: arda organism name, ``'human'`` or ``'mouse'`` (matches the
            library ``species`` field).
        sensitivity: mmseqs2 search sensitivity passed to arda (arda default 7.0).

    Returns:
        Mapping ``{allele: {region: aa_seq}}``.  V alleles carry keys from
        :data:`V_REGIONS`; J alleles carry keys from :data:`J_REGIONS`.  Regions
        that arda could not resolve are omitted.

    Raises:
        ImportError: If arda (and mmseqs2) are not available.
    """
    annotate_records = _require_arda()

    records: list[tuple[str, str]] = [
        (e.allele, e.sequence)
        for e in lib.get_entries()
        if e.gene in ("V", "J") and e.sequence and e.species == organism
    ]
    if not records:
        return {}

    airr = annotate_records(
        records,
        organism=organism,
        seqtype="nt",
        strand="forward",
        map_d=False,
        sensitivity=sensitivity,
    )

    out: dict[str, dict[str, str]] = {}
    mismatches: list[str] = []
    for rec in airr:
        allele = rec["sequence_id"]
        entry = lib.entries.get(allele)
        if entry is None:
            continue
        called_locus = rec.get("locus") or ""
        if not _locus_ok(entry.locus, called_locus):
            mismatches.append(f"{allele}: library={entry.locus} arda={called_locus}")
            continue

        regions: dict[str, str] = {}
        if entry.gene == "V":
            for r in V_REGIONS:
                aa = rec.get(f"{r}_aa")
                if aa:
                    regions[r] = aa
        elif entry.gene == "J":
            fwr4_aa = rec.get("fwr4_aa")
            if fwr4_aa:
                regions["fwr4"] = fwr4_aa
            fwr4_start = rec.get("fwr4_start")
            if fwr4_start:
                jcdr3 = _jcdr3_aa(rec.get("sequence", ""), int(fwr4_start))
                if jcdr3:
                    regions["jcdr3"] = jcdr3
        if regions:
            out[allele] = regions

    if mismatches:
        warnings.warn(
            f"{len(mismatches)} allele(s) mapped to an unexpected locus and were "
            f"skipped (first few: {mismatches[:5]})",
            UserWarning,
            stacklevel=2,
        )
    return out
