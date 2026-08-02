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
    _sample_weights,
    centroid_atypicality,
    class_witness,
    decode_metrics,
    fit_repertoire_space,
    hla_stratified_mmd,
    mmd_distance,
    mmd_matrix,
    sample_descriptor,
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


def test_descriptor_metrics_derivable_smooth_and_decodable(space):
    # mass-preserving descriptor: infiltration/diversity/clonality are derivable coordinates,
    # decode_metrics inverts the vector, and the scale (infiltration) tracks total reads.
    df = _sample(_clonotypes(120, offset=0), lambda n: np.geomspace(1, 1000, n))
    d = sample_descriptor(space, df)
    m = d.metrics()
    # infiltration = log1p(total reads); diversity = exp(log_neff) in the Hill interval [²D, ⁰D]
    assert abs(m["infiltration"] - np.log1p(df["duplicate_count"].sum())) < 1e-9
    assert m["clonality"] > 0 and m["diversity"] >= 1.0
    # decode_metrics reads the same metrics off the (possibly perturbed) vector
    assert decode_metrics(d.vector)["infiltration"] == m["infiltration"]
    # scale-carrying: a deeper copy raises the infiltration coordinate (mass is retained, not normalised away)
    deep = df.with_columns((pl.col("duplicate_count") * 10).alias("duplicate_count"))
    assert sample_descriptor(space, deep).metrics()["infiltration"] > m["infiltration"]
    # under presence weighting (w uniform) clonality is pure shape → scale-invariant
    dfd = df.with_columns(pl.lit(3.0).alias("duplicate_count"))
    dfd10 = df.with_columns(pl.lit(30.0).alias("duplicate_count"))
    assert abs(sample_descriptor(space, dfd, weight="distinct").metrics()["clonality"]
               - sample_descriptor(space, dfd10, weight="distinct").metrics()["clonality"]) < 1e-9
    # smoothness (continuity): a tiny count perturbation moves the descriptor only slightly (relative)
    pert = df.with_columns((pl.col("duplicate_count") + 1e-3).alias("duplicate_count"))
    rel = np.linalg.norm(sample_descriptor(space, pert).vector - d.vector) / (np.linalg.norm(d.vector) + 1e-9)
    assert rel < 0.02


@pytest.mark.parametrize("weight", ["distinct", "duplicate_count", "log1p", "log2p1", "anscombe"])
def test_weights_run_and_neff_in_hill_interval(space, weight):
    df = _sample(_clonotypes(120, offset=0), lambda n: np.geomspace(1, 1000, n))
    emb = sample_embedding(space, df, weight=weight)
    d0, d2 = np.exp(emb.diversity[0]), np.exp(emb.diversity[2])
    assert d2 - 1e-6 <= emb.n_eff <= d0 + 1e-6            # n_eff is a Hill number (prop:antag)


def test_default_weight_is_log2p1(space):
    df = _sample(_clonotypes(90, offset=400), lambda n: np.geomspace(1, 200, n))
    default = sample_embedding(space, df)
    explicit = sample_embedding(space, df, weight="log2p1")
    old_default = sample_embedding(space, df, weight="log1p")
    assert np.array_equal(default.mean, explicit.mean)
    assert not np.array_equal(default.mean, old_default.mean)


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


def test_hla_stratified_multiallele_and_empty_set(space):
    # vectorized indicator: partial-overlap pairs match; a donor with no HLA matches nobody.
    embs = [sample_embedding(space, _sample(_clonotypes(50, offset=o)), blocks=("mean",))
            for o in (0, 100, 200)]
    hla = [{"A*02:01", "B*07:02"}, {"B*07:02"}, set()]      # 0&1 share B*07; 2 has none
    S = hla_stratified_mmd(embs, hla)
    assert np.isfinite(S[0, 1]) and np.isfinite(S[1, 0])   # partial overlap still matched
    assert np.isnan(S[0, 2]) and np.isnan(S[2, 2])         # empty-HLA donor masked everywhere
    assert np.isfinite(S[0, 0])                            # self compared when it has any allele


def test_unbiased_mmd_rejects_singleton(space):
    """Unbiased MMD is undefined at n_eff ≤ 1 (a point mass): raise instead of silently dividing by 0."""
    single = sample_embedding(space, _sample(_clonotypes(1, offset=0)), blocks=("mean",))
    ok = sample_embedding(space, _sample(_clonotypes(60, offset=100)), blocks=("mean",))
    assert single.n_eff == 1.0
    assert np.isfinite(mmd_distance(single, ok))                      # biased path still works
    with pytest.raises(ValueError, match="single-clonotype"):
        mmd_distance(single, ok, unbiased=True)
    with pytest.raises(ValueError, match="single-clonotype"):
        mmd_matrix([single, ok], unbiased=True)


def test_empty_and_zero_count_samples_raise(space):
    empty = _sample(_clonotypes(0))
    with pytest.raises(ValueError, match="empty repertoire"):
        sample_embedding(space, empty)
    zeros = _sample(_clonotypes(10, offset=0), np.zeros(10))
    with pytest.raises(ValueError, match="degenerate"):
        sample_embedding(space, zeros)
    with pytest.raises(ValueError, match="degenerate"):
        sample_descriptor(space, zeros)


def test_centroid_atypicality_flags_outliers():
    # a point on its group centroid -> ~0; an outlier -> large; grouping is respected.
    X = np.array([[1.0, 0.0], [1.0, 0.02], [1.0, -0.02], [-1.0, 0.0],   # group 0 (last is the outlier)
                  [0.0, 1.0], [0.02, 1.0]])                              # group 1
    g = np.array([0, 0, 0, 0, 1, 1])
    a = centroid_atypicality(X, g)
    assert a[:3].max() < 0.1 and a[3] > 1.5            # in-group tight vs the flipped outlier
    assert a[4] < 0.1 and a[5] < 0.1                   # group 1 is computed against its own centroid


def test_class_witness_precomputed_matches(space):
    """Passing witness= must give identical scores to computing it internally (the sweep fast-path)."""
    base = _clonotypes(400, offset=0)
    motif = _clonotypes(1, offset=300)
    pos = [pl.concat([_sample(base.sample(120, seed=s)),
                      _sample(motif, lambda n: np.full(n, 400.0))]) for s in range(4)]
    neg = [_sample(base.sample(120, seed=s + 50)) for s in range(4)]
    cand = pl.concat([base.sample(120, seed=0), motif]).unique()

    def group_mean(frames):
        return np.mean([w @ space.rff.transform(Z)
                        for Z, w in (space.sample_cloud(f) for f in frames)], axis=0)
    witness = group_mean(pos) - group_mean(neg)

    a = class_witness(space, pos, neg, cand, top=20)
    b = class_witness(space, [], [], cand, top=20, witness=witness)   # pos/neg ignored
    assert np.allclose(a["witness_score"].to_numpy(), b["witness_score"].to_numpy())


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


def test_mean_less_embedding_vectorizes_but_refuses_mmd(space):
    # reachable from `mir embed repertoires --blocks diversity`
    df = _sample(_clonotypes(60, offset=0))
    emb = sample_embedding(space, df, blocks=("diversity",))
    assert emb.mean is None
    assert emb.vector.shape == (4,)                # Φ is still a vector, just a mean-less one
    with pytest.raises(ValueError, match="no mean"):
        mmd_distance(emb, emb)
    with pytest.raises(ValueError, match="no mean"):
        mmd_matrix([emb, emb])


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


def test_replicate_space_is_a_distinct_incomparable_basis(tmp_path):
    """A prototype replicate is a different coordinate system, and save/load knows it."""
    from mir.embedding.tcremp import TCREmp

    pool = _clonotypes(160, offset=0)
    m0 = TCREmp.from_defaults("human", "TRB", 64)
    m1 = TCREmp.from_defaults("human", "TRB", 64, replicate=1)
    assert m0.replicate == 0 and m1.replicate == 1

    s0 = fit_repertoire_space(m0, pool, n_rff=64, n_rff_second=0, n_components=8, seed=0)
    s1 = fit_repertoire_space(m1, pool, n_rff=64, n_rff_second=0, n_components=8, seed=0)
    assert s0.meta["replicate"] == 0 and s1.meta["replicate"] == 1
    assert s0.meta["prototype_hash"] != s1.meta["prototype_hash"]

    # a replicate basis round-trips (the replicate is carried in meta and rebuilt)…
    df = _sample(_clonotypes(50, offset=0))
    p = tmp_path / "rep1.pkl"
    s1.save(p)
    back = RepertoireSpace.load(p)
    assert back.clono.model.replicate == 1
    assert np.allclose(sample_embedding(s1, df, blocks=("mean",)).vector,
                       sample_embedding(back, df, blocks=("mean",)).vector, atol=1e-6)

    # …and the two draws give genuinely different embeddings of the same sample
    assert not np.allclose(sample_embedding(s0, df, blocks=("mean",)).vector,
                           sample_embedding(s1, df, blocks=("mean",)).vector)


def test_correct_batch_reduces_to_residualize_and_beats_it_under_confound():
    """Harmony-lite correct_batch: == residualize at K=1; preserves biology a global
    mean-subtraction destroys when batch is confounded with a biological cluster."""
    import numpy as np
    from mir.repertoire import correct_batch
    from mir.cohort import residualize

    rng = np.random.default_rng(0)
    n, d = 200, 10
    bio = rng.integers(0, 2, n)                              # biological cluster (dominant signal)
    batch = np.where(rng.random(n) < 0.8, bio, 1 - bio)      # 80% confounded with bio
    X = rng.normal(0, 0.1, (n, d))
    X[:, 0] += np.where(bio == 1, 4.0, -4.0)                 # axis 0 = biology
    X[:, 1] += np.where(batch == 1, 1.5, -1.5)              # axis 1 = batch offset

    # K=1 reduces exactly to residualize
    assert np.allclose(correct_batch(X, batch, n_clusters=1), residualize(X, batch))

    Xc = correct_batch(X, batch, n_clusters=2, seed=0)
    Xr = residualize(X, batch)

    def gap(M, lab, ax):
        return abs(M[lab == 1, ax].mean() - M[lab == 0, ax].mean())

    # batch offset (axis 1) is removed by the cluster-aware correction
    assert gap(Xc, batch, 1) < 0.5 * gap(X, batch, 1)
    # biology (axis 0) survives the confound better than plain per-group mean subtraction
    assert gap(Xc, bio, 0) > gap(Xr, bio, 0)


# --- deficient measure: missing mass, naive reference, contrast ---------------------

def test_missing_mass_estimators_and_edge_cases():
    """M₀ limits, the f2==0 guard, and the singleton-vs-doubleton ordering of the estimators."""
    from mir.repertoire import missing_mass

    ones = np.ones(200)                                    # every clone seen exactly once
    assert missing_mass(ones, "turing") == 1.0             # f1/N = 1 -> nothing retained
    assert missing_mass(ones, "chao") > 0.99
    deep = np.full(200, 5000.0)                            # no singletons, no doubletons
    assert missing_mass(deep, "turing") == 0.0
    assert missing_mass(deep, "chao") == 0.0
    assert missing_mass(deep, "none") == 0.0

    # f2 == 0 is common at RNA-seq depth: bias-corrected Chao1 must not divide by zero
    assert np.isfinite(missing_mass(np.array([1.0, 1.0, 1.0, 7.0]), "chao"))

    # singletons dominating doubletons -> Chao above Turing; doubletons abundant -> below
    singleton_rich = np.concatenate([np.ones(100), np.full(1, 2.0), np.full(9, 100.0)])
    assert missing_mass(singleton_rich, "chao") > missing_mass(singleton_rich, "turing")
    doubleton_rich = np.concatenate([np.ones(10), np.full(1000, 2.0), np.full(980, 100.0)])
    assert missing_mass(doubleton_rich, "chao") < missing_mass(doubleton_rich, "turing")

    with pytest.raises(ValueError, match="missing_mass must be"):
        missing_mass(ones, "goodturing")


def test_missing_mass_default_leaves_blocks_untouched(space):
    """The regression gate: missing_mass= only sets .mass, never a block."""
    df = _sample(_clonotypes(120, offset=0), lambda n: np.geomspace(1, 500, n))
    default = sample_embedding(space, df)
    none = sample_embedding(space, df, missing_mass="none")
    chao = sample_embedding(space, df, missing_mass="chao")
    assert default.mass == none.mass == 1.0
    assert np.array_equal(default.vector, none.vector)
    assert np.array_equal(default.vector, chao.vector)     # blocks are bit-identical
    assert chao.mass <= 1.0


@pytest.mark.parametrize("method", ["turing", "chao"])
def test_mass_plus_unseen_block_sums_to_one(space, method):
    from mir.repertoire import missing_mass

    df = _sample(_clonotypes(120, offset=0), lambda n: np.r_[np.ones(n - 5), np.full(5, 900.0)])
    emb = sample_embedding(space, df, missing_mass=method)
    m0 = missing_mass(df["duplicate_count"].to_numpy(), method)
    _, _, w = _sample_weights(df, "log2p1")
    assert emb.mass * w.sum() + m0 == 1.0                  # exactly, not approximately


def test_mass_is_low_when_shallow_and_high_when_deep(space):
    shallow = _sample(_clonotypes(150, offset=0))                      # all singletons
    deep = _sample(_clonotypes(150, offset=0), lambda n: np.full(n, 4000.0))
    assert sample_embedding(space, shallow, missing_mass="chao").mass < 0.02
    assert sample_embedding(space, deep, missing_mass="chao").mass == 1.0


def test_naive_reference_deterministic_and_cached(space):
    from mir.repertoire import naive_reference

    r1 = naive_reference(space, n=1500, seed=3)
    assert r1.shape == (space.rff.dim,) and np.isfinite(r1).all()
    assert naive_reference(space, n=1500, seed=3) is r1                # cached per (n, seed)
    assert not np.array_equal(naive_reference(space, n=1500, seed=4), r1)
    # the injectable path bypasses vdjtools (and the cache) but is still reproducible
    seqs = _clonotypes(200, offset=0)
    assert np.array_equal(naive_reference(space, sequences=seqs),
                          naive_reference(space, sequences=seqs))


def test_naive_reference_tracks_the_germline_draw_not_an_expansion(space):
    """Centred cosine: the reference sits next to an independent naive draw, not an expansion.

    Centred, because an uncentred comparison of two kernel means is dominated by their shared DC
    offset and puts *any* two of them at cos ≈ 1 — an artifact that produced a wrong conclusion once.
    """
    from mir.repertoire import naive_reference

    def cos_centred(u, v):
        u, v = u - u.mean(), v - v.mean()
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    ref = naive_reference(space, n=2000, seed=0)
    other = naive_reference(space, n=2000, seed=11)                     # independent naive draw
    one_clone = _sample(_clonotypes(1, offset=0), lambda n: np.full(n, 500.0))
    spike = sample_embedding(space, one_clone).mean
    assert cos_centred(ref, other) > cos_centred(ref, spike)


def test_naive_reference_differs_between_loci():
    from vdjtools.model import load_bundled
    from vdjtools.model.generate import generate

    from mir.repertoire import naive_reference

    refs = {}
    for locus in ("TRB", "TRA"):
        pool = generate(load_bundled(locus), 400, seed=0, productive_only=True)
        model = TCREmp.from_defaults("human", locus, n_prototypes=300)
        sp = fit_repertoire_space(model, pool, n_rff=256, n_rff_second=0, n_components=15, seed=0)
        refs[locus] = naive_reference(sp, n=600, seed=0)
    assert not np.allclose(refs["TRB"], refs["TRA"])


def test_contrast_embedding_is_confidence_times_deviation(space):
    """Ψ = mass·(Φ − naive): signed, big for an expansion, zero for a sample with no confidence."""
    from mir.repertoire import contrast_embedding, naive_reference

    ref = naive_reference(space, n=2000, seed=0)

    # a deep, naive-looking repertoire: full mass (no singletons), Φ ≈ naive -> small ‖Ψ‖
    naive_like = _sample(_clonotypes(200, offset=0), lambda n: np.full(n, 50.0))
    emb_naive = sample_embedding(space, naive_like, missing_mass="chao")
    psi_naive = contrast_embedding(emb_naive, ref)
    assert emb_naive.mass == 1.0

    # same depth, but one hyperexpanded clone dominates -> large ‖Ψ‖
    expanded = pl.concat([
        _sample(_clonotypes(199, offset=1), lambda n: np.full(n, 50.0)),
        _sample(_clonotypes(1, offset=0), lambda n: np.full(n, 2_000_000.0)),
    ])
    psi_expanded = contrast_embedding(
        sample_embedding(space, expanded, weight="duplicate_count", missing_mass="chao"), ref)
    assert np.linalg.norm(psi_expanded) > 3 * np.linalg.norm(psi_naive)

    # a 3-clonotype "immune desert": no confidence at all, so it lands at the ORIGIN — and is
    # still embedded, not dropped. This is the whole point of the deficient measure.
    desert = _sample(_clonotypes(3, offset=300))
    emb_desert = sample_embedding(space, desert, missing_mass="turing")
    assert emb_desert.mass == 0.0 and np.isfinite(emb_desert.vector).all()
    assert np.linalg.norm(contrast_embedding(emb_desert, ref)) == 0.0

    # Ψ is signed (depletion relative to unselected recombination is a negative coordinate),
    # while the clonotype embedding it is built from is an all-positive distance profile.
    assert (psi_expanded > 0).any() and (psi_expanded < 0).any()
    assert (space.clono.model.embed(naive_like) >= 0).all()
