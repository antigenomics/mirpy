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


# --- arda ambiguity groups ---------------------------------------------------------

def test_ambiguity_group_members_resolve_not_fall_back():
    """arda joins alleles with identical germline into one row ("IGHV1-69*01,IGHV1-69D*01").

    Every member must resolve to that row: they were unreachable, so the two most-used human
    IGHV genes (IGHV3-23*01, IGHV1-69*01) silently took the max-distance fallback.
    """
    for species, locus in (("human", "IGH"), ("human", "TRB"), ("mouse", "TRA")):
        gd = load_germline_distances(species, locus)
        for comp in COMPONENTS:
            if not gd.has(comp):
                continue
            c = gd._components[comp]
            for key in [k for k in c.idx if "," in k]:
                row = c.idx[key]
                for member in key.split(","):
                    assert c.resolve(member) == row, f"{species}/{locus}/{comp}: {member}"


def test_common_igh_genes_are_not_max_distance():
    gd = load_germline_distances("human", "IGH")
    fb = gd._components["V"].fallback
    for gene in ("IGHV3-23*01", "IGHV1-69*01"):
        assert gd.matrix("V", [gene], [gene])[0, 0] == 0.0        # self-distance, not fallback
        assert gd.matrix("V", [gene], ["IGHV4-34*01"])[0, 0] != fb
        for comp in ("CDR1", "CDR2"):                             # cdr123 mode uses the same index
            assert gd.matrix(comp, [gene], [gene])[0, 0] == 0.0
