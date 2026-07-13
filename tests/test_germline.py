import numpy as np
import pytest

from mir.distances.germline import (
    COMPONENTS,
    GermlineDistances,
    load_germline_distances,
)


def test_load_has_all_components():
    gd = GermlineDistances.load("human", "TRB")
    for comp in COMPONENTS:
        assert gd.has(comp)


def test_self_distance_zero_and_symmetry():
    gd = GermlineDistances.load("human", "TRB")
    genes = ["TRBV20-1*01", "TRBV6-5*01", "TRBV5-1*01"]
    D = gd.matrix("V", genes, genes)
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)          # V matrix is symmetric
    assert (D >= 0).all()               # valid (semi)metric


def test_bare_gene_resolves_to_major():
    gd = GermlineDistances.load("human", "TRB")
    exact = gd.matrix("V", ["TRBV6-5*01"], ["TRBV20-1*01"])
    bare = gd.matrix("V", ["TRBV6-5"], ["TRBV20-1*01"])
    assert exact[0, 0] == bare[0, 0]


def test_unknown_gene_falls_back():
    gd = GermlineDistances.load("human", "TRB")
    fb = gd._components["V"].fallback
    D = gd.matrix("V", ["TRBVNOPE*99"], ["TRBV20-1*01"])
    assert D[0, 0] == fb


def test_missing_component_raises():
    gd = GermlineDistances.load("human", "TRB")
    with pytest.raises(KeyError):
        gd.matrix("NOPE", ["TRBV20-1*01"], ["TRBV20-1*01"])


def test_cache_returns_same_object():
    a = load_germline_distances("human", "TRB")
    b = load_germline_distances("hsa", "beta")
    assert a is b
