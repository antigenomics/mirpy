"""Tests for mir.repertoire (sample-level embedding, Theory §T.7).

Self-contained on bundled resources (no network, no torch): the OLGA TRB sample
``tests/assets/olga_humanTRB_1000.txt.gz`` and the committed TRB prototypes.
"""

import numpy as np
import polars as pl
import pytest

from mir.embedding.tcremp import TCREmp
from mir.repertoire import (
    RepertoireSpace,
    _make_rff,
    class_witness,
    fit_repertoire_space,
    hla_stratified_mmd,
    mmd_distance,
    mmd_matrix,
    sample_embedding,
)

_OLGA = "tests/assets/olga_humanTRB_1000.txt.gz"


def _clonotypes(n: int, offset: int = 0) -> pl.DataFrame:
    df = pl.read_csv(
        _OLGA, separator="\t", has_header=False,
        new_columns=["junction_nt", "junction_aa", "v_call", "j_call"],
    ).select(["junction_aa", "v_call", "j_call"]).slice(offset, n)
    return df.unique(subset=["junction_aa", "v_call", "j_call"])


def _sample(df: pl.DataFrame, counts=None) -> pl.DataFrame:
    """Attach a ``duplicate_count`` column sized to the frame's actual height.

    ``counts`` may be ``None`` (all ones), a callable ``n -> array``, or an array
    matching ``df.height``.
    """
    n = df.height
    if counts is None:
        c = np.ones(n)
    elif callable(counts):
        c = np.asarray(counts(n), dtype=float)
    else:
        c = np.asarray(counts, dtype=float)
    return df.with_columns(pl.Series("duplicate_count", c.astype(np.float64)))


@pytest.fixture(scope="module")
def space():
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=300)
    pool = _clonotypes(600)
    return fit_repertoire_space(model, pool, n_rff=1024, n_rff_second=64, n_components=20, seed=0)


# --- RFF ---------------------------------------------------------------------------

def test_rff_kernel_approximation():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((50, 8))
    rff = _make_rff(8, 40000, length_scale=1.5, seed=0)
    P = rff.transform(Z)
    for i, j in [(0, 1), (0, 10), (3, 7)]:
        exact = np.exp(-np.sum((Z[i] - Z[j]) ** 2) / (2 * 1.5 ** 2))
        assert abs(P[i] @ P[j] - exact) < 0.03


# --- fit + shapes ------------------------------------------------------------------

def test_fit_and_transform_shapes(space):
    df = _clonotypes(40, offset=700)
    Z = space.transform_clonotypes(df)
    assert Z.shape == (df.height, space.meta["n_components"])
    assert space.rff.dim == 1024 and space.rff2.dim == 64


def test_sample_embedding_blocks_and_vector(space):
    df = _sample(_clonotypes(80, offset=100), lambda n: np.arange(1, n + 1))
    emb = sample_embedding(space, df)
    assert emb.mean.shape == (1024,)
    assert emb.diversity.shape == (4,)
    assert emb.second.shape == (64 * 65 // 2,)             # upper triangle
    assert emb.vector.shape == (1024 + 4 + 64 * 65 // 2,)
    assert np.isfinite(emb.vector).all()


def test_spectral_second_block_top_r_eigvals():
    # opt-in n_eigs -> second block is the top-r eigenvalues (non-neg, descending), r-dim
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=300)
    sp = fit_repertoire_space(model, _clonotypes(600), n_rff=512, n_rff_second=64,
                              n_eigs=8, n_components=20, seed=0)
    df = _sample(_clonotypes(80, offset=100), lambda n: np.arange(1, n + 1))
    emb = sample_embedding(sp, df)
    assert emb.second.shape == (8,)
    assert np.all(emb.second >= -1e-9)
    assert np.all(np.diff(emb.second) <= 1e-9)             # descending
    assert np.isfinite(emb.vector).all()
    with pytest.raises(ValueError, match="n_eigs"):
        fit_repertoire_space(model, _clonotypes(600), n_rff_second=64, n_eigs=200)


@pytest.mark.parametrize("weight", ["distinct", "log1p", "anscombe"])
def test_weights_run_and_neff_in_hill_interval(space, weight):
    df = _sample(_clonotypes(120, offset=0), lambda n: np.geomspace(1, 1000, n))
    emb = sample_embedding(space, df, weight=weight)
    d0, d2 = np.exp(emb.diversity[0]), np.exp(emb.diversity[2])
    assert d2 - 1e-6 <= emb.n_eff <= d0 + 1e-6            # n_eff is a Hill number (prop:antag)


def test_neff_equals_richness_under_presence_weighting(space):
    # g≡1 (presence) -> w uniform -> n_eff = #clonotypes = ⁰D (prop:antag boundary)
    df = _sample(_clonotypes(90, offset=200), lambda n: np.geomspace(1, 500, n))
    emb = sample_embedding(space, df, weight="distinct")
    assert abs(emb.n_eff - np.exp(emb.diversity[0])) < 1e-6   # n_eff == observed richness


# --- MMD / cohort separation -------------------------------------------------------

def test_injected_cohort_separation(space):
    base = _clonotypes(400, offset=0)
    spike = _clonotypes(6, offset=0)
    A, B = [], []
    for s in range(4):
        a = _sample(base.sample(150, seed=s))
        A.append(sample_embedding(space, a, blocks=("mean",)))
        b = pl.concat([a, _sample(spike, lambda n: np.full(n, 800.0))])   # public expansion
        B.append(sample_embedding(space, b, blocks=("mean",)))
    within = np.mean([mmd_distance(A[i], A[j]) for i in range(4) for j in range(i + 1, 4)])
    between = np.mean([mmd_distance(a, b) for a in A for b in B])
    assert between > within


def test_mmd_matrix_symmetric_zero_diag(space):
    embs = [sample_embedding(space, _sample(_clonotypes(60, offset=o)), blocks=("mean",))
            for o in (0, 100, 200)]
    D = mmd_matrix(embs)
    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)


def test_unbiased_mmd_removes_self_bias(space):
    """Two independent subsamples of the SAME clonotypes: unbiased MMD² < biased (the 1/n_eff self-bias)."""
    base = _clonotypes(400, offset=0)
    a = sample_embedding(space, _sample(base.sample(200, seed=1)), blocks=("mean",))
    b = sample_embedding(space, _sample(base.sample(200, seed=2)), blocks=("mean",))
    biased = mmd_distance(a, b)
    unbiased = mmd_distance(a, b, unbiased=True)
    assert unbiased < biased                       # diagonal removal shrinks the same-distribution distance
    assert unbiased >= 0.0                          # clamped
    Du = mmd_matrix([a, b], unbiased=True)
    assert np.allclose(Du, Du.T) and np.allclose(np.diag(Du), 0.0)


def test_hla_stratified_masks_mismatched_pairs(space):
    embs = [sample_embedding(space, _sample(_clonotypes(50, offset=o)), blocks=("mean",))
            for o in (0, 100, 200)]
    hla = [{"A*02:01"}, {"A*02:01"}, {"B*07:02"}]
    S = hla_stratified_mmd(embs, hla)
    assert np.isfinite(S[0, 1])                    # matched pair compared
    assert np.isnan(S[0, 2]) and np.isnan(S[1, 2]) # mismatched pairs masked


def test_class_witness_ranks_injected_motif(space):
    # a public motif seeded into every 'pos' sample must surface at the top of the witness
    motif = _clonotypes(1, offset=300)                      # one specific clonotype
    base = _clonotypes(400, offset=0)
    pos = [pl.concat([_sample(base.sample(120, seed=s)),
                      _sample(motif, lambda n: np.full(n, 400.0))]) for s in range(5)]
    neg = [_sample(base.sample(120, seed=s + 50)) for s in range(5)]
    candidates = pl.concat([base.sample(120, seed=0), motif]).unique()
    ranked = class_witness(space, pos, neg, candidates, top=10)
    top_juncs = ranked["junction_aa"].to_list()[:5]
    assert motif["junction_aa"][0] in top_juncs             # discriminative motif is surfaced


# --- depth-robustness (prop:kme) ---------------------------------------------------

def test_phi1_depth_robustness_under_downsample(space):
    from vdjtools.preprocess import downsample

    rng = np.random.default_rng(0)
    full = _sample(_clonotypes(300, offset=0), lambda n: rng.integers(1, 200, n).astype(float))
    phi_full = sample_embedding(space, full, blocks=("mean",)).mean
    reads = int(full["duplicate_count"].sum())
    errs = []
    for frac in (0.02, 0.1, 0.5):
        sub = downsample(full, max(int(reads * frac), 10), by="reads", seed=0)
        phi = sample_embedding(space, sub, blocks=("mean",)).mean
        errs.append(np.linalg.norm(phi - phi_full))
    assert errs[-1] < errs[0]                       # deeper subsample -> closer to Φ₁(full)


# --- coverage-standardized diversity (vdjtools) ------------------------------------

def test_coverage_standardized_diversity_runs(space):
    df = _sample(_clonotypes(120, offset=0), lambda n: np.geomspace(1, 2000, n))
    emb = sample_embedding(space, df, coverage=0.95, blocks=("diversity",))
    assert emb.diversity.shape == (4,)
    assert np.isfinite(emb.diversity).all()
    assert 0.0 < emb.diversity[3] <= 1.0           # Ĉ is a coverage


# --- serialization / comparability invariant ---------------------------------------

def test_save_load_roundtrip_and_cross_basis_refusal(space, tmp_path):
    df = _sample(_clonotypes(50, offset=0))
    before = sample_embedding(space, df, blocks=("mean", "second")).vector

    p = tmp_path / "space.pkl"
    space.save(p)
    reloaded = RepertoireSpace.load(p)
    after = sample_embedding(reloaded, df, blocks=("mean", "second")).vector
    assert np.allclose(before, after, atol=1e-6)

    # tamper the prototype hash -> load must refuse (incomparable basis)
    import pickle
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    d["meta"]["prototype_hash"] = "deadbeefdeadbeef"
    with open(p, "wb") as fh:
        pickle.dump(d, fh)
    with pytest.raises(ValueError, match="prototype hash mismatch"):
        RepertoireSpace.load(p)
    RepertoireSpace.load(p, verify=False)          # explicit override is allowed
