"""Backward-compatibility shim — use :mod:`mir.common.gene_library` instead."""

from mir.common.gene_library import (  # noqa: F401
    GeneEntry as Segment,
    GeneLibrary as SegmentLibrary,
    _GENE_LIBRARY_CACHE as _SEGMENT_CACHE,
    _ALLOWED_LOCI as _ALLOWED_GENES,
    _ALLOWED_GENES as _ALLOWED_STYPE,
)
