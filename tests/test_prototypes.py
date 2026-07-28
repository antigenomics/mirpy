import pytest

from mir.embedding.prototypes import (
    N_PROTOTYPES,
    list_available_prototypes,
    load_prototypes,
    n_replicates,
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


def test_n_nonpositive_raises():
    # a non-positive n silently changes the prototype set (df.head(-k)) and its hash -> reject it.
    for bad in (0, -5):
        with pytest.raises(ValueError):
            load_prototypes("human", "TRB", n=bad)


def test_unknown_locus_file():
    # IGK has no prototype file? it does; use a locus with no file: mouse IGH
    with pytest.raises(FileNotFoundError):
        load_prototypes("mouse", "IGH")


def test_list_available_includes_human_trb():
    pairs = list_available_prototypes()
    assert ("human", "TRB") in pairs
    assert all(len(p) == 2 for p in pairs)


# --- replicate draws ---------------------------------------------------------------

def test_replicate_zero_is_the_default_set():
    # the canonical set must not move: replicate=0 is exactly the historical head(n)
    assert load_prototypes("human", "TRB", n=200).equals(
        load_prototypes("human", "TRB", n=200, replicate=0))


def test_replicates_are_disjoint_and_same_size():
    n = 200
    blocks = [load_prototypes("human", "TRB", n=n, replicate=r) for r in range(5)]
    assert all(b.height == n for b in blocks)
    seen: set[tuple] = set()
    for b in blocks:
        rows = set(map(tuple, b.rows()))
        assert not (rows & seen)                      # disjoint = independent draws
        seen |= rows
    assert len(seen) == 5 * n


def test_n_replicates_matches_pool_and_bounds_the_index():
    assert n_replicates("human", "TRB", 1000) == 10   # the documented "sets 1-10"
    assert n_replicates("human", "TRB", 2000) == 5
    with pytest.raises(ValueError, match="replicates 0..9"):
        load_prototypes("human", "TRB", n=1000, replicate=10)


def test_replicate_requires_explicit_n_and_rejects_negative():
    with pytest.raises(ValueError, match="explicit n"):
        load_prototypes("human", "TRB", replicate=1)
    with pytest.raises(ValueError, match="must be >= 0"):
        load_prototypes("human", "TRB", n=100, replicate=-1)


def test_replicate_changes_the_prototype_hash():
    # the comparability guards key off this hash: two draws must not look alike
    from mir.ml.bundle import prototype_hash

    h0 = prototype_hash("human", "TRB", 100)
    assert h0 == prototype_hash("human", "TRB", 100, replicate=0)
    assert h0 != prototype_hash("human", "TRB", 100, replicate=1)
