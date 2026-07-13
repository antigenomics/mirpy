"""Tests for mir.density (continuous-density TCRNET/ALICE, Theory T6).

All self-contained on bundled resources (no network, no torch): the OLGA background
sample ``tests/assets/olga_humanTRB_1000.txt.gz`` and the committed TRB prototypes.
"""

import os

import numpy as np
import polars as pl
import pytest

from mir.density import (
    DensitySpace,
    EnrichmentResult,
    _slice,
    calibrate_radius,
    denoise_and_cluster,
    enriched_mask,
    fit_density_space,
    neighbor_enrichment,
)
from mir.embedding.tcremp import TCREmp

_OLGA = "tests/assets/olga_humanTRB_1000.txt.gz"


def _load_olga(n: int, offset: int = 0) -> pl.DataFrame:
    df = pl.read_csv(
        _OLGA, separator="\t", has_header=False,
        new_columns=["junction_nt", "junction_aa", "v_call", "j_call"],
    ).select(["junction_aa", "v_call", "j_call"])
    return df.slice(offset, n)


# --- pure-array tests (no model needed) -------------------------------------------

def test_neighbor_enrichment_detects_injected_cluster():
    rng = np.random.default_rng(0)
    d = 10
    bg = rng.standard_normal((4000, d))
    obs = rng.standard_normal((800, d))
    obs[:40] = np.full(d, 2.5) + 0.05 * rng.standard_normal((40, d))  # convergent family
    res = neighbor_enrichment(obs, bg, radius=0.5)
    mask = enriched_mask(res)
    assert mask[:40].mean() > 0.8            # injected cluster is enriched
    assert mask[40:].mean() < 0.05           # background points are not
    assert res.fold[:40].mean() > res.fold[40:].mean()


def test_poisson_and_binomial_both_run_and_shape():
    rng = np.random.default_rng(1)
    obs, bg = rng.standard_normal((200, 6)), rng.standard_normal((500, 6))
    for test in ("poisson", "binomial"):
        res = neighbor_enrichment(obs, bg, radius=0.8, test=test)
        assert res.n_obs.shape == res.qvalue.shape == (200,)
        assert np.all((res.pvalue >= 0) & (res.pvalue <= 1))
        assert np.all(res.n_obs >= 0)


def test_balloon_mode_detects_injected_cluster():
    # adaptive-bandwidth (balloon) estimator: radius=None, fixed background occupancy
    rng = np.random.default_rng(2)
    bg = rng.standard_normal((5000, 8))
    obs = rng.standard_normal((800, 8))
    obs[:40] = np.full(8, 2.5) + 0.05 * rng.standard_normal((40, 8))
    res = neighbor_enrichment(obs, bg, lambda0=3.0)  # no radius -> balloon
    mask = enriched_mask(res)
    assert mask[:40].mean() > 0.8 and mask[40:].mean() < 0.05
    assert res.radius > 0


def test_unknown_test_raises():
    obs, bg = np.zeros((3, 2)), np.ones((3, 2))
    with pytest.raises(ValueError):
        neighbor_enrichment(obs, bg, radius=1.0, test="chi2")


def test_enriched_mask_criteria():
    res = EnrichmentResult(
        n_obs=np.array([5, 5, 0, 5]),
        n_bg=np.array([0, 0, 0, 100]),
        expected=np.array([0.1, 0.1, 0.1, 9.0]),
        fold=np.array([50.0, 50.0, 0.0, 0.5]),
        pvalue=np.array([1e-6, 0.2, 1.0, 1.0]),
        qvalue=np.array([1e-5, 0.2, 1.0, 1.0]),
        radius=1.0,
    )
    m = enriched_mask(res, alpha=0.05, min_fold=1.0, min_neighbors=2)
    assert m.tolist() == [True, False, False, False]  # only #0 passes q, fold & n_obs


def test_slice_junction_is_every_third_column():
    emb = np.arange(2 * 9).reshape(2, 9).astype(float)  # 3 prototypes
    assert _slice(emb, "junction").tolist() == [[2, 5, 8], [11, 14, 17]]
    assert _slice(emb, "full").shape == (2, 9)
    with pytest.raises(ValueError):
        _slice(emb, "bogus")


# --- model-based tests (real TCREMP embedding on bundled prototypes) ---------------

@pytest.fixture(scope="module")
def model():
    return TCREmp.from_defaults("human", "TRB", n_prototypes=300)


def test_fit_density_space_shapes_and_transform(model):
    obs, bg = _load_olga(120), _load_olga(120, offset=400)
    space, obs_emb, bg_emb = fit_density_space(model, obs, bg, n_components=20, space="junction")
    assert isinstance(space, DensitySpace)
    assert obs_emb.shape == (120, 20) and bg_emb.shape == (120, 20)
    # re-projecting obs through the fitted space reproduces obs_emb
    assert np.allclose(space.transform(obs), obs_emb, atol=1e-6)


def test_calibrate_radius_positive(model):
    obs, bg = _load_olga(150), _load_olga(150, offset=400)
    space, _, _ = fit_density_space(model, obs, bg, n_components=20, space="junction")
    r = calibrate_radius(space, sample=200)
    assert np.isfinite(r) and r > 0


def test_denoise_and_cluster_groups_injected_family(model):
    # observed = OLGA background + a convergent 1-substitution family; background
    # subtraction keeps the family, and clustering recovers it as one (non-noise) group.
    rng = np.random.default_rng(0)
    bgset = _load_olga(300)
    seed_seq = bgset["junction_aa"][0]
    v, j = bgset["v_call"][0], bgset["j_call"][0]
    from mir.density import _mutate1
    family = [_mutate1(seed_seq, rng) for _ in range(30)]  # tight 1-substitution family
    fam_df = pl.DataFrame({"junction_aa": [seed_seq, *family],
                           "v_call": [v] * 31, "j_call": [j] * 31})
    obs = pl.concat([bgset, fam_df])
    bg = _load_olga(400, offset=400)
    space, obs_emb, bg_emb = fit_density_space(model, obs, bg, n_components=20, space="junction")
    r = calibrate_radius(space, sample=200)
    res = neighbor_enrichment(obs_emb, bg_emb, r * 1.5)
    # explicit eps sized to the 1-substitution radius so clustering is deterministic
    labels, mask = denoise_and_cluster(obs_emb, res, eps=r * 2.5, min_samples=3)
    fam_labels = labels[300:]                      # the injected family rows
    assert mask[300:].mean() > mask[:300].mean()   # family enriched more than background
    assert (fam_labels >= 0).mean() > 0.5          # most of the family got clustered
    # the family shares one dominant cluster label
    clustered_fam = fam_labels[fam_labels >= 0]
    _, counts = np.unique(clustered_fam, return_counts=True)
    assert counts.max() >= 0.5 * clustered_fam.size


@pytest.mark.integration
def test_continuous_matches_discrete_hamming1(model):
    """Continuous radius-r neighbour counts correlate with discrete Hamming-1 counts."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    bgset = _load_olga(400)
    seed_seq = bgset["junction_aa"][0]
    v, j = bgset["v_call"][0], bgset["j_call"][0]
    from mir.density import _mutate1
    family = [_mutate1(seed_seq, rng) for _ in range(30)]  # 1-substitution neighbours
    fam_df = pl.DataFrame({"junction_aa": [seed_seq, *family],
                           "v_call": [v] * 31, "j_call": [j] * 31})
    obs = pl.concat([bgset, fam_df])
    bg = _load_olga(400, offset=400)
    space, obs_emb, bg_emb = fit_density_space(model, obs, bg, n_components=25, space="junction")
    r = calibrate_radius(space, sample=200)
    res = neighbor_enrichment(obs_emb, bg_emb, r)

    seqs = obs["junction_aa"].to_list()
    discrete = np.array([
        sum(a != b and len(a) == len(b)
            and sum(x != y for x, y in zip(a, b)) == 1
            for b in seqs)
        for a in seqs
    ])
    rho = spearmanr(res.n_obs, discrete).statistic
    assert rho > 0.4, rho


def test_generate_background_optional():
    from mir.density import generate_background
    try:
        df = generate_background("TRB", 50, seed=0)
    except Exception as e:  # bundled model not loadable in this env
        pytest.skip(f"vdjtools background generation unavailable: {e}")
    assert df.height == 50
    assert set(df.columns) == {"junction_aa", "v_call", "j_call"}
