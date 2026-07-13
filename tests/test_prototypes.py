import pytest

from mir.embedding.prototypes import (
    N_PROTOTYPES,
    list_available_prototypes,
    load_prototypes,
)


def test_load_shape_and_columns():
    df = load_prototypes("human", "TRB", n=100)
    assert df.columns == ["v_call", "j_call", "junction_aa"]
    assert df.height == 100


def test_order_is_stable_prefix():
    a = load_prototypes("human", "TRB", n=50)
    b = load_prototypes("human", "TRB", n=100)
    assert a.equals(b.head(50))


def test_alias_resolution():
    a = load_prototypes("human", "TRB", n=10)
    b = load_prototypes("hsa", "beta", n=10)
    assert a.equals(b)


def test_n_cap_raises():
    with pytest.raises(ValueError):
        load_prototypes("human", "TRB", n=N_PROTOTYPES + 1)


def test_unknown_locus_file():
    # IGK has no prototype file? it does; use a locus with no file: mouse IGH
    with pytest.raises(FileNotFoundError):
        load_prototypes("mouse", "IGH")


def test_list_available_includes_human_trb():
    pairs = list_available_prototypes()
    assert ("human", "TRB") in pairs
    assert all(len(p) == 2 for p in pairs)
