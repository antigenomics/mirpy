"""Backward-compatible shim for legacy segment imports.

This module preserves older import paths such as::

    from mir.common.segments import Segment, SegmentLibrary
"""

from __future__ import annotations

import warnings

from .gene_library import GeneEntry, GeneLibrary

warnings.warn(
    "mir.common.segments is deprecated; import GeneEntry/GeneLibrary from mir.common.gene_library instead.",
    DeprecationWarning,
    stacklevel=2,
)

Segment = GeneEntry
SegmentLibrary = GeneLibrary

__all__ = ["Segment", "SegmentLibrary", "GeneEntry", "GeneLibrary"]
