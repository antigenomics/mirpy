import numpy as np
import polars as pl
import pytest

from mir.embedding.prototypes import load_prototypes
from mir.embedding.tcremp import MODES, PairedTCREmp, TCREmp


def _df():
    return pl.DataFrame(
        {
            "v_call": ["TRBV10-3*01", "TRBV20-1*01"],
            "j_call": ["TRBJ2-7*01", "TRBJ1-2*01"],
            "junction_aa": ["CASSIRSSYEQYF", "CSARVSGYYGYTF"],
        }
    )


@pytest.mark.parametrize("mode", MODES)
def test_embed_shape_dtype(mode):
    m = TCREmp.from_defaults("human", "TRB", n_prototypes=64, mode=mode)
    X = m.embed(_df())
    assert X.shape == (2, 3 * 64)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert (X >= 0).all()
    assert X.shape[1] == m.n_features


def test_prototype_as_query_zero_self_slot():
    K = 200
    protos = load_prototypes("human", "TRB", n=K)
    m = TCREmp.from_defaults("human", "TRB", n_prototypes=K, mode="vjcdr3")
    X = m.embed(protos.head(4))
    for i in range(4):
        assert tuple(X[i, 3 * i:3 * i + 3]) == (0.0, 0.0, 0.0)


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        TCREmp.from_defaults("human", "TRB", n_prototypes=8, mode="nope")


def test_missing_columns_raises():
    m = TCREmp.from_defaults("human", "TRB", n_prototypes=8)
    with pytest.raises(ValueError):
        m.embed(pl.DataFrame({"v_call": ["TRBV10-3*01"]}))


def test_paired_concat():
    m = PairedTCREmp.from_defaults("human", ("TRA", "TRB"), n_prototypes=32)
    a = pl.DataFrame({"v_call": ["TRAV1-2*01"], "j_call": ["TRAJ33*01"],
                      "junction_aa": ["CAVKDSNYQLIW"]})
    b = pl.DataFrame({"v_call": ["TRBV10-3*01"], "j_call": ["TRBJ2-7*01"],
                      "junction_aa": ["CASSIRSSYEQYF"]})
    X = m.embed({"TRA": a, "TRB": b})
    assert X.shape == (1, 2 * 3 * 32)
    assert X.shape[1] == m.n_features


def test_paired_row_mismatch_raises():
    m = PairedTCREmp.from_defaults("human", ("TRA", "TRB"), n_prototypes=16)
    a = pl.DataFrame({"v_call": ["TRAV1-2*01", "TRAV1-2*01"], "j_call": ["TRAJ33*01", "TRAJ33*01"],
                      "junction_aa": ["CAVKDSNYQLIW", "CAVKDSNYQLIW"]})
    b = pl.DataFrame({"v_call": ["TRBV10-3*01"], "j_call": ["TRBJ2-7*01"],
                      "junction_aa": ["CASSIRSSYEQYF"]})
    with pytest.raises(ValueError):
        m.embed({"TRA": a, "TRB": b})
