"""Smoke tests for compatibility re-export shims.

These modules exist only to preserve old import paths; the tests guard against
the shim silently breaking (e.g. a renamed symbol in the canonical module).
"""

from __future__ import annotations


def test_biomarkers_vdjbet_reexports_comparative():
    import mir.biomarkers.vdjbet as shim
    import mir.comparative.vdjbet as canonical

    # The shim does `from ... import *`; spot-check a public name is re-exported
    # and identical to the canonical implementation.
    public = [n for n in dir(canonical) if not n.startswith("_")]
    assert public, "canonical vdjbet module exposes no public names"
    shared = [n for n in public if hasattr(shim, n)]
    assert shared, "shim re-exported none of the canonical public names"
    for name in shared:
        assert getattr(shim, name) is getattr(canonical, name)


def test_single_cell_util_reexports_pairing_graph():
    from mir.common.single_cell_util import PairingGraph, build_pairing_graph
    from mir.graph.single_cell_pairing import (
        PairingGraph as CanonicalGraph,
        build_pairing_graph as canonical_build,
    )

    assert PairingGraph is CanonicalGraph
    assert build_pairing_graph is canonical_build
