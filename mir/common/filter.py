"""Repertoire filtering helpers.

This module provides clonotype-level filters that depend on both CDR3 amino-acid
quality and gene-library functionality annotations.
"""

from __future__ import annotations

from mir.basic.mirseq_compat import is_canonical, is_coding
from mir.common.gene_library import GeneLibrary
from mir.common.repertoire import LocusRepertoire, SampleRepertoire


def _load_library_for_loci(loci: set[str], source: str) -> GeneLibrary:
    non_empty = {l for l in loci if l}
    if not non_empty:
        non_empty = {"TRA", "TRB"}
    return GeneLibrary.load_default(loci=non_empty, species={"human"}, source=source)


def _filter_locus(
    repertoire: LocusRepertoire,
    gene_library: GeneLibrary,
    *,
    require_canonical: bool,
) -> LocusRepertoire:
    kept = []
    for clonotype in repertoire.clonotypes:
        if not gene_library.is_functional(clonotype.v_gene):
            continue
        if not clonotype.junction_aa or not is_coding(clonotype.junction_aa):
            continue
        if require_canonical and not is_canonical(clonotype.junction_aa):
            continue
        kept.append(clonotype)

    return LocusRepertoire(
        clonotypes=kept,
        locus=repertoire.locus,
        repertoire_id=repertoire.repertoire_id,
        repertoire_metadata=dict(repertoire.repertoire_metadata),
    )


def filter_functional(
    repertoire: LocusRepertoire | SampleRepertoire,
    gene_library: GeneLibrary | None = None,
    *,
    source: str = "imgt",
) -> LocusRepertoire | SampleRepertoire:
    """Keep clonotypes with functional V gene and coding CDR3 amino-acid sequence.

    Functional V-gene status is resolved with :meth:`GeneLibrary.is_functional`.
    CDR3 coding status is checked via :func:`mir.basic.mirseq.is_coding`.
    """
    if isinstance(repertoire, LocusRepertoire):
        lib = gene_library or _load_library_for_loci({repertoire.locus}, source)
        return _filter_locus(repertoire, lib, require_canonical=False)

    loci = set(repertoire.loci.keys())
    lib = gene_library or _load_library_for_loci(loci, source)
    filtered_loci = {
        locus: _filter_locus(lr, lib, require_canonical=False)
        for locus, lr in repertoire.loci.items()
    }
    return SampleRepertoire(
        loci=filtered_loci,
        sample_id=repertoire.sample_id,
        sample_metadata=dict(repertoire.sample_metadata),
    )


def filter_canonical(
    repertoire: LocusRepertoire | SampleRepertoire,
    gene_library: GeneLibrary | None = None,
    *,
    source: str = "imgt",
) -> LocusRepertoire | SampleRepertoire:
    """Keep clonotypes passing functional filter and canonical CDR3 rule.

    Canonical CDR3 is checked with :func:`mir.basic.mirseq.is_canonical`.
    """
    if isinstance(repertoire, LocusRepertoire):
        lib = gene_library or _load_library_for_loci({repertoire.locus}, source)
        return _filter_locus(repertoire, lib, require_canonical=True)

    loci = set(repertoire.loci.keys())
    lib = gene_library or _load_library_for_loci(loci, source)
    filtered_loci = {
        locus: _filter_locus(lr, lib, require_canonical=True)
        for locus, lr in repertoire.loci.items()
    }
    return SampleRepertoire(
        loci=filtered_loci,
        sample_id=repertoire.sample_id,
        sample_metadata=dict(repertoire.sample_metadata),
    )
