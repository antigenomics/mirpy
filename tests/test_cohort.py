"""Tests for mir.cohort (the digital-donor multi-chain fusion, §T.7).

Self-contained on bundled TRB+TRA prototypes (no network, no torch).
"""

import numpy as np
import polars as pl
import pytest

from mir.cohort import DonorCohort, cluster_samples, fit_donor_embeddings, residualize
from mir.embedding.prototypes import list_available_prototypes
from mir.embedding.tcremp import TCREmp
from mir.repertoire import centroid_atypicality, fit_repertoire_spaces, sample_embedding

_LOCI = [c for c in ("TRB", "TRA") if ("human", c) in list_available_prototypes()]

pytestmark = pytest.mark.skipif(len(_LOCI) < 2, reason="need >=2 bundled human loci")


@pytest.fixture(scope="module")
def spaces():
    models = {c: TCREmp.from_defaults("human", c, n_prototypes=300) for c in _LOCI}
    protos = {c: pl.DataFrame({"v_call": m._proto_v, "j_call": m._proto_j,
                               "junction_aa": m._proto_junction}).unique() for c, m in models.items()}
    return fit_repertoire_spaces(models, protos, n_rff=512, n_rff_second=0, n_components=15, seed=0), protos


def _cohort_frames(protos, n=24, drop_locus_for=None):
    """n donors in two groups; group 1 gets a public expansion. Optionally drop a locus for one donor."""
    donor_frames, rows = [], []
    for i in range(n):
        grp = i % 2
        frames = {}
        for c in protos:
            if drop_locus_for == (i, c):
                continue                                   # this donor lacks this chain
            base = protos[c].sample(120, seed=i).with_columns(pl.lit(1.0).alias("duplicate_count"))
            if grp == 1:
                spike = protos[c].slice(0, 5).with_columns(pl.lit(400.0).alias("duplicate_count"))
                base = pl.concat([base, spike])
            frames[c] = base
        donor_frames.append(frames)
        rows.append({"group": grp})
    return donor_frames, rows


def test_fit_core_channels_and_shape(spaces):
    sp, protos = spaces
    frames, rows = _cohort_frames(protos)
    coh = fit_donor_embeddings(sp, frames, rows=rows, id_pca=6, seed=0)
    assert coh.X.shape[0] == 24
    for ch in ("identity", "diversity", "coverage"):
        assert ch in coh.spec
    assert "identity" in coh.spec.attributable        # kernel-mean block is attributable
    assert np.isfinite(coh.X).all()                   # imputed + z-scored, no holes
    # identity merges both chains: 2 loci x id_pca=6 columns
    assert len(coh.spec["identity"]) == len(_LOCI) * 6


def test_transform_reproduces_fit_rows(spaces):
    # a core-only cohort: transforming the SAME donors through the stored basis reproduces X exactly.
    sp, protos = spaces
    frames, rows = _cohort_frames(protos)
    coh = fit_donor_embeddings(sp, frames, rows=rows, id_pca=6, seed=0)
    Xt = coh.transform(frames)
    assert np.allclose(Xt, coh.X, atol=1e-6)


def test_missing_chain_is_imputed(spaces):
    sp, protos = spaces
    frames, rows = _cohort_frames(protos, drop_locus_for=(3, _LOCI[1]))   # donor 3 lacks the 2nd locus
    coh = fit_donor_embeddings(sp, frames, rows=rows, id_pca=6, seed=0)
    assert np.isfinite(coh.X).all()                   # the hole is imputed, not NaN


def test_extra_channels_fuse_and_stay_non_attributable(spaces):
    sp, protos = spaces
    frames, rows = _cohort_frames(protos)

    def extra(rows, identity):
        return {"atypicality": centroid_atypicality(identity, np.array([r["group"] for r in rows]))}

    coh = fit_donor_embeddings(sp, frames, rows=rows, id_pca=6, extra_channels=extra, seed=0)
    assert "atypicality" in coh.spec
    assert "atypicality" not in coh.spec.attributable   # a geometry summary has no clonotype pre-image
    # transform now requires the extra block re-supplied
    with pytest.raises(ValueError, match="extra channels"):
        coh.transform(frames[:3])
    Xt = coh.transform(frames[:3], extra={"atypicality": np.zeros(3)})
    assert Xt.shape == (3, coh.X.shape[1])


def test_save_load_roundtrip_and_cross_basis_refusal(spaces, tmp_path):
    sp, protos = spaces
    frames, rows = _cohort_frames(protos)
    coh = fit_donor_embeddings(sp, frames, rows=rows, id_pca=6, seed=0)
    p = tmp_path / "cohort.pkl"
    coh.save(p)
    back = DonorCohort.load(p)
    assert np.allclose(back.X, coh.X)
    assert back.spec.names == coh.spec.names and set(back.spaces) == set(sp)

    import pickle
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    any_locus = next(iter(d["spaces"]))
    d["spaces"][any_locus]["meta"]["prototype_hash"] = "deadbeefdeadbeef"
    with open(p, "wb") as fh:
        pickle.dump(d, fh)
    with pytest.raises(ValueError, match="prototype hash mismatch"):
        DonorCohort.load(p)
    DonorCohort.load(p, verify=False)                 # explicit override allowed


def test_residualize_removes_group_offset():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 5))
    batch = np.array([0, 1, 2] * 10)
    X = X + np.array([0, 5, -5])[batch][:, None]      # a large per-batch offset
    Xr = residualize(X, batch)
    for g in (0, 1, 2):
        assert np.allclose(Xr[batch == g].mean(0), 0.0, atol=1e-9)   # offset removed


def test_cluster_samples_runs(spaces):
    sp, protos = spaces
    frames, _ = _cohort_frames(protos, n=12)
    c0 = _LOCI[0]
    embs = [sample_embedding(sp[c0], f[c0], blocks=("mean",)) for f in frames]
    labels = cluster_samples(embs)
    assert len(labels) == 12 and labels.dtype.kind in "iu"


def test_james_stein_shrink_stops_a_small_n_batch_correction_injecting_noise():
    """Per-group centring in high d with few samples per group injects an offset as big as it removes.

    Evaluated the only way that can see it: fit the offset on half of each batch, apply it to the
    held-out half, and ask how separable the batches are afterwards. In-sample the injected error
    vanishes by construction.
    """
    rng = np.random.default_rng(0)
    d, per_batch, n_batch = 200, 10, 6
    batch = np.repeat(np.arange(n_batch), per_batch)
    n = batch.size

    def batch_separation(M):
        cen = np.stack([M[batch == b].mean(0) for b in range(n_batch)])
        return float(np.mean([np.linalg.norm(cen[i] - cen[j])
                              for i in range(n_batch) for j in range(i + 1, n_batch)]))

    # NO true batch offset: any apparent one is pure estimation error
    X = rng.normal(0, 1, (n, d))
    fit = np.zeros(n, dtype=bool)
    for b in range(n_batch):
        fit[np.flatnonzero(batch == b)[: per_batch // 2]] = True

    def apply_offsets(shrink):
        Xf = residualize(X[fit], batch[fit], shrink=shrink)
        off = X[fit] - Xf                                   # what the correction subtracts per batch
        out = X.copy()
        for b in range(n_batch):
            out[batch == b] -= off[batch[fit] == b][0]
        return out[~fit]

    def held_sep(M):
        return float(np.mean([np.linalg.norm(M[batch[~fit] == i].mean(0) - M[batch[~fit] == j].mean(0))
                              for i in range(n_batch) for j in range(i + 1, n_batch)]))

    sep_raw = held_sep(X[~fit])
    plain = held_sep(apply_offsets(False))
    shrunk = held_sep(apply_offsets(True))
    assert plain > sep_raw                     # the documented failure: correction makes batch louder
    assert shrunk < plain                      # positive-part James-Stein recovers most of it
    assert shrunk <= sep_raw * 1.05

    # with a genuinely large offset, shrinkage keeps ~all of the correction
    Y = X + np.repeat(rng.normal(0, 3, (n_batch, d)), per_batch, axis=0)
    assert batch_separation(residualize(Y, batch, shrink=True)) < 0.2 * batch_separation(Y)
    # and the default is byte-identical to the previous behaviour
    assert np.array_equal(residualize(Y, batch), residualize(Y, batch, shrink=False))


def test_depth_report_flags_a_depth_driven_axis_and_clears_a_clean_one():
    from mir.cohort import depth_report

    rng = np.random.default_rng(0)
    n, d = 300, 40
    log_reads = rng.normal(10, 2, n)
    stats = {"log_reads": log_reads, "singleton_fraction": rng.uniform(0, 1, n)}

    # X whose dominant axis IS depth
    load = np.zeros((n, d))
    load[:, 0] = 6.0
    dirty = rng.normal(0, 1, (n, d)) + (log_reads - log_reads.mean())[:, None] * load[:1].repeat(n, 0)
    rep = depth_report(dirty, stats)
    assert rep["r2_pc1"] > 0.8 and rep["pc_best"] == 1
    assert len(rep["r2_by_pc"]) == rep["n_pc"] == 5
    assert len(rep["explained_variance"]) == 5 and rep["stats"] == ["log_reads", "singleton_fraction"]

    # X independent of the fingerprint
    clean = rng.normal(0, 1, (n, d))
    assert depth_report(clean, stats)["r2_best"] < 0.2

    # the trusted-subset arm, and non-finite stats are filled rather than dropping samples
    holed = {"log_reads": log_reads.copy(), "singleton_fraction": stats["singleton_fraction"]}
    holed["log_reads"][:5] = np.nan
    sub = log_reads > np.median(log_reads)
    r2 = depth_report(dirty, holed, subset=sub)
    assert r2["n_samples"] == n and r2["n_samples_subset"] == int(sub.sum())
    assert np.isfinite(r2["r2_pc1_subset"])

    with pytest.raises(ValueError, match="stats is empty"):
        depth_report(clean, {})
    with pytest.raises(ValueError, match="values for"):
        depth_report(clean, {"bad": np.ones(3)})
