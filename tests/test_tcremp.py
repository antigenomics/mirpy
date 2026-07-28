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


def test_null_junction_raises_by_name():
    m = TCREmp.from_defaults("human", "TRB", n_prototypes=8)
    df = _df().with_columns(pl.Series("junction_aa", ["CASSIRSSYEQYF", None]))
    with pytest.raises(ValueError, match="junction_aa"):
        m.embed(df)


def test_metric_sqrt_is_elementwise_root_of_squared():
    kw = dict(species="human", locus="TRB", n_prototypes=64)
    Xd = TCREmp.from_defaults(**kw).embed(_df())
    Xs = TCREmp.from_defaults(**kw, metric="sqrt").embed(_df())
    assert np.allclose(Xs, np.sqrt(np.clip(Xd, 0.0, None)), atol=1e-4)


def test_custom_substitution_matrix_changes_junction_block_only():
    import seqtree

    kw = dict(species="human", locus="TRB", n_prototypes=64)
    Xb = TCREmp.from_defaults(**kw).embed(_df())
    Xp = TCREmp.from_defaults(**kw, matrix=seqtree.SubstitutionMatrix.pam250()).embed(_df())
    assert np.isfinite(Xp).all() and Xp.shape == Xb.shape
    assert np.array_equal(Xp[:, 0::3], Xb[:, 0::3])      # V block is baked BLOSUM62 either way
    assert not np.allclose(Xp[:, 2::3], Xb[:, 2::3])     # junction block follows the matrix


def test_sw_alignment_agrees_with_gapblock_on_near_neighbours():
    pytest.importorskip("Bio", reason="alignment='sw' needs BioPython ([build] extra)")
    from mir.distances.junction import junction_distance_matrix

    # gap-block approximates SW where it matters (close pairs), not globally: SW is a *local*
    # alignment, so distant pairs diverge. Assert the near-neighbour agreement the default relies
    # on, plus a positive but not-interchangeable overall rank correlation.
    close = ["CASSTTGLNTEAFF", "CASSLTAMNTEAFF"]
    g = junction_distance_matrix(close, close)
    w = junction_distance_matrix(close, close, alignment="sw")
    assert np.array_equal(g, w)

    seqs = load_prototypes("human", "TRB", n=60)["junction_aa"].to_list()
    G = junction_distance_matrix(seqs, seqs).astype(float)
    W = junction_distance_matrix(seqs, seqs, alignment="sw").astype(float)
    assert np.isfinite(W).all() and W.shape == G.shape
    assert (np.diag(W) == 0).all()
    iu = np.triu_indices(len(seqs), k=1)
    assert np.corrcoef(G[iu], W[iu])[0, 1] > 0.6


def test_sw_rejects_custom_matrix():
    with pytest.raises(ValueError, match="gapblock"):
        TCREmp.from_defaults("human", "TRB", n_prototypes=8,
                             alignment="sw", matrix=object()).embed(_df())


def test_paired_row_mismatch_raises():
    m = PairedTCREmp.from_defaults("human", ("TRA", "TRB"), n_prototypes=16)
    a = pl.DataFrame({"v_call": ["TRAV1-2*01", "TRAV1-2*01"], "j_call": ["TRAJ33*01", "TRAJ33*01"],
                      "junction_aa": ["CAVKDSNYQLIW", "CAVKDSNYQLIW"]})
    b = pl.DataFrame({"v_call": ["TRBV10-3*01"], "j_call": ["TRBJ2-7*01"],
                      "junction_aa": ["CASSIRSSYEQYF"]})
    with pytest.raises(ValueError):
        m.embed({"TRA": a, "TRB": b})
