import numpy as np

from mir.bench.theory import (
    fit_distributions,
    junction_dissimilarity,
    prototype_source_correlation,
    s2_dissimilarity_distance_correlation,
)
from mir.embedding.prototypes import load_prototypes

_CDR3 = load_prototypes("human", "TRB", n=300)["junction_aa"].to_list()


def test_dissimilarity_matrix_is_metric():
    d = junction_dissimilarity(_CDR3[:50])
    assert d.shape == (50, 50)
    assert np.allclose(np.diag(d), 0.0)      # self-distance 0
    assert np.allclose(d, d.T)               # symmetric
    assert (d >= 0).all()


def test_s2_positive_correlation():
    # T1: embedding distance tracks dissimilarity (positive correlation)
    res = s2_dissimilarity_distance_correlation(_CDR3)
    assert res.n == 300
    assert res.pearson > 0.3


def test_s1_distance_is_extreme_value_not_normal():
    # T4: D_ij fits GEV better than Normal
    res = s2_dissimilarity_distance_correlation(_CDR3)
    fits = fit_distributions(res.d, res.D)
    assert fits["D_gev"]["ks"] < fits["D_normal"]["ks"]


def test_s3_prototype_source_robustness():
    # distances from two disjoint TRB prototype sets are highly correlated
    query = _CDR3[:120]
    protos = load_prototypes("human", "TRB", n=2000)["junction_aa"].to_list()
    r = prototype_source_correlation(query, protos[:1000], protos[1000:])
    assert r["pearson"] > 0.8


def test_shm_drift_monotone_and_bounded():
    # T5: embedding drift increases with mutation load, D_0 == 0
    from mir.bench.theory import shm_embedding_drift

    protos = load_prototypes("human", "TRB", n=400)["junction_aa"].to_list()
    seqs = _CDR3[:120]
    d = shm_embedding_drift(seqs, protos, max_mut=5, n_rep=2, seed=0)
    means = [d[k][0] for k in sorted(d)]
    assert means[0] == 0.0
    assert all(b >= a for a, b in zip(means, means[1:]))   # non-decreasing in k
    assert means[-1] > means[1]                             # real drift accumulates
