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
    # S3: two *independent* prototype draws (disjoint replicate blocks) give the same geometry.
    # Queries are held out of both blocks. Pins the figure published in README/usage.rst:
    # R = 0.993 at the default n=1000 — i.e. results do not rest on which prototypes were drawn.
    query = load_prototypes("human", "TRB", n=400, replicate=24)["junction_aa"].to_list()
    a = load_prototypes("human", "TRB", n=1000, replicate=0)["junction_aa"].to_list()
    b = load_prototypes("human", "TRB", n=1000, replicate=1)["junction_aa"].to_list()
    assert not set(a) & set(b)                       # independent draws, not a re-slice
    r = prototype_source_correlation(query, a, b)
    assert r["pearson"] > 0.98


def test_prototype_draw_agreement_grows_with_n():
    # the other half of the published table: below n~250 the particular draw does start to show
    query = load_prototypes("human", "TRB", n=200, replicate=24)["junction_aa"].to_list()

    def agreement(n):
        a = load_prototypes("human", "TRB", n=n, replicate=0)["junction_aa"].to_list()
        b = load_prototypes("human", "TRB", n=n, replicate=1)["junction_aa"].to_list()
        return prototype_source_correlation(query, a, b)["pearson"]

    small, large = agreement(100), agreement(1000)
    assert small < large                              # more prototypes -> more draw-independent
    assert small > 0.85 and large > 0.98


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


def test_t6_tcrnet_convergence_decays_with_radius():
    # T6: continuous enrichment matches discrete Hamming-1 at the one-substitution scale
    # and decouples as the radius grows (the r->0 graph-enrichment limit).
    from mir.bench.theory import _mutate, tcrnet_convergence

    protos = load_prototypes("human", "TRB", n=1400)["junction_aa"].to_list()
    rng = np.random.default_rng(0)
    # inject a convergent 1-substitution family so a discrete Hamming-1 signal exists
    family = [_mutate(protos[0], 1, rng) for _ in range(20)]
    obs = protos[1:381] + [protos[0], *family]
    bg, ref = protos[400:800], protos[800:1400]
    out = tcrnet_convergence(obs, bg, ref, n_components=20, seed=0)
    assert out["radius_1sub"] > 0 and out["hamming1_mean"] > 0
    corr = out["spearman_by_scale"]
    assert np.isfinite(out["spearman_at_1sub"])
    assert corr[0.5] >= corr[3.0]          # correlation fades as radius grows


def test_theory_helpers_accept_single_pass_iterables():
    """Regression: these walked their sequence argument more than once without materializing it.

    A generator over a polars column -- the natural way to feed them -- arrived empty on the
    second pass, so `tcrnet_convergence` died deep inside StandardScaler with "Found array with
    0 sample(s)" rather than anywhere near the caller.
    """
    from mir.bench.theory import tcrnet_convergence

    seqs = _CDR3[:40]
    ref = junction_dissimilarity(seqs)
    assert np.allclose(junction_dissimilarity(iter(seqs)), ref)      # consumed twice internally

    r_list = prototype_source_correlation(seqs, _CDR3[:60], _CDR3[60:120])
    r_gen = prototype_source_correlation(iter(seqs), _CDR3[:60], _CDR3[60:120])
    assert r_gen["pearson"] == r_list["pearson"]                     # embedded twice

    # obs is walked three times (matrix, mutation calibration, Hamming-1 counts) and prototypes
    # three times; before the fix this raised from inside StandardScaler on the second pass.
    out = tcrnet_convergence(iter(seqs), iter(_CDR3[40:120]), iter(_CDR3[120:200]),
                             n_components=5, scales=(1.0,), seed=0)
    assert out["radius_1sub"] > 0
    assert len(out["spearman_by_scale"]) == 1


def test_sw_dissimilarity_agrees_with_gapblock_on_near_neighbours():
    """The paper-exact Smith-Waterman reference (S1/S2), needed to validate the gapblock default.

    The two are different coordinate systems at long range -- 'sw' is a LOCAL alignment that stops
    extending -- but must agree in the near-neighbour regime clustering and density actually use.
    """
    import pytest

    pytest.importorskip("Bio", reason="junction_dissimilarity_sw needs BioPython ([build] extra)")
    from mir.bench.theory import junction_dissimilarity_sw

    seqs = _CDR3[:24]
    d = junction_dissimilarity_sw(seqs)
    assert d.shape == (24, 24)
    assert np.allclose(np.diag(d), 0.0)          # d(a,a) = s_aa + s_aa - 2 s_aa = 0
    assert np.allclose(d, d.T)                   # symmetric
    assert (d >= -1e-9).all()                    # non-negative (a Gram dissimilarity)

    # a single substitution is closer than an unrelated pair, under BOTH backends
    a = seqs[0]
    b = a[:5] + ("A" if a[5] != "A" else "G") + a[6:]
    near = junction_dissimilarity_sw([a, b])[0, 1]
    far = junction_dissimilarity_sw([a, seqs[7]])[0, 1]
    assert near < far
    assert junction_dissimilarity([a, b])[0, 1] < junction_dissimilarity([a, seqs[7]])[0, 1]

    # and the S2 correlation route accepts the 'sw' backend end-to-end
    r = s2_dissimilarity_distance_correlation(seqs, dissimilarity="sw")
    assert r.n == 24 and np.isfinite(r.pearson)
