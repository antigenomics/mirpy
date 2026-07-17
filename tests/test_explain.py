"""Tests for mir.explain (channel registry + scorer-agnostic ablation + attribution, §T.7).

Self-contained on bundled resources (no network, no torch, no lifelines). The ablation half runs on
synthetic matrices; the attribution half reuses the OLGA TRB sample + committed prototypes, as in
``test_repertoire.py``. The scorers here are toy closures defined in this file — which is itself the
proof of the plug-in contract: the library ships none and never sees ``y``.
"""

import numpy as np
import polars as pl
import pytest

from mir.embedding.tcremp import TCREmp
from mir.explain import (
    ChannelBuilder,
    ChannelSpec,
    channel_drivers,
    channel_report,
    stack_embeddings,
)
from mir.repertoire import fit_repertoire_space, sample_embedding

_OLGA = "tests/assets/olga_humanTRB_1000.txt.gz"


def _clonotypes(n: int, offset: int = 0) -> pl.DataFrame:
    df = pl.read_csv(
        _OLGA, separator="\t", has_header=False,
        new_columns=["junction_nt", "junction_aa", "v_call", "j_call"],
    ).select(["junction_aa", "v_call", "j_call"]).slice(offset, n)
    return df.unique(subset=["junction_aa", "v_call", "j_call"])


def _sample(df: pl.DataFrame, counts=None) -> pl.DataFrame:
    n = df.height
    c = np.ones(n) if counts is None else (
        np.asarray(counts(n), dtype=float) if callable(counts) else np.asarray(counts, dtype=float)
    )
    return df.with_columns(pl.Series("duplicate_count", c.astype(np.float64)))


@pytest.fixture(scope="module")
def space():
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=300)
    return fit_repertoire_space(model, _clonotypes(600), n_rff=1024, n_rff_second=64,
                                n_components=20, seed=0)


@pytest.fixture(scope="module")
def planted():
    """A matrix with a known answer: one informative channel, its duplicate, and noise."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    sig = y[:, None] + rng.normal(0, 0.35, (n, 3))
    X, spec = (ChannelBuilder()
               .add("signal", sig, attributable=True)
               .add("dup", sig + rng.normal(0, 0.02, (n, 3)))
               .add("noise", rng.normal(0, 1, (n, 5)))
               .build(standardize=False, impute=False))
    return X, spec, y


def _corr_scorer(y):
    return lambda B: float(max(abs(np.corrcoef(B[:, j], y)[0, 1]) for j in range(B.shape[1])))


# --- channel registry --------------------------------------------------------------


def test_channel_columns_partition_the_matrix(planted):
    X, spec, _ = planted
    cols = [i for g in spec.names for i in spec[g]]
    assert sorted(cols) == list(range(X.shape[1]))       # union == all columns
    assert len(cols) == len(set(cols))                   # pairwise disjoint
    assert spec.width == X.shape[1]


def test_builder_merges_repeated_names_and_accepts_1d():
    a, b = np.zeros((5, 2)), np.ones((5, 3))
    X, spec = ChannelBuilder().add("ident", a).add("ident", b).add("cov", np.arange(5.0)).build(
        standardize=False, impute=False)
    assert len(spec["ident"]) == 5                       # merged, not two channels
    assert spec["ident"] == [0, 1, 2, 3, 4]              # contiguous in add order
    assert len(spec["cov"]) == 1                         # 1-D -> one column
    assert X.shape == (5, 6)
    with pytest.raises(ValueError, match="rows"):
        ChannelBuilder().add("a", np.zeros((5, 1))).add("b", np.zeros((4, 1)))


def test_builder_attributable_is_sticky_per_channel():
    _, spec = (ChannelBuilder()
               .add("ident", np.zeros((4, 2)))
               .add("ident", np.zeros((4, 2)), attributable=True)   # one attributable block...
               .add("div", np.zeros((4, 2)))
               .build(standardize=False, impute=False))
    assert "ident" in spec.attributable                  # ...marks the whole merged channel
    assert "div" not in spec.attributable


def test_builder_standardize_and_impute_invariants():
    rng = np.random.default_rng(1)
    raw = rng.normal(3.0, 7.0, (30, 4))
    holed = raw.copy()
    holed[5, 2] = np.nan
    X, _ = ChannelBuilder().add("a", holed).build(standardize=True, impute=True)
    assert np.isfinite(X).all()
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1, atol=1e-8)
    # the hole becomes the column median of the finite entries
    Xi, _ = ChannelBuilder().add("a", holed).build(standardize=False, impute=True)
    assert Xi[5, 2] == pytest.approx(float(np.median(raw[[i for i in range(30) if i != 5], 2])))
    # raw passthrough
    Xr, _ = ChannelBuilder().add("a", raw).build(standardize=False, impute=False)
    assert np.array_equal(Xr, raw)


def test_build_without_blocks_raises():
    with pytest.raises(ValueError, match="no blocks"):
        ChannelBuilder().build()


def test_spec_getitem_is_dict_compatible(planted):
    _, spec, _ = planted
    assert spec["signal"] == spec.columns("signal")
    assert "signal" in spec and "nope" not in spec
    assert spec["signal"] + spec["noise"] == spec.columns("signal") + spec.columns("noise")
    assert isinstance(spec["signal"][0], int)
    with pytest.raises(ValueError, match="unknown channel"):
        spec["nope"]


# --- the .vector bridge (backward compat) ------------------------------------------


def test_stack_embeddings_matches_vector_concat(space):
    embs = [sample_embedding(space, _sample(_clonotypes(150, offset=o))) for o in (0, 150, 300)]
    X, spec = stack_embeddings(embs)
    for i, e in enumerate(embs):
        assert np.array_equal(X[i], e.vector)            # exact: names attached, nothing transformed
    assert np.array_equal(X[:, spec["mean"]], np.stack([e.mean for e in embs]))
    assert spec.width == X.shape[1]
    assert "mean" in spec.attributable                   # the KME is the attributable block
    assert "diversity" not in spec.attributable


def test_stack_embeddings_rejects_empty_and_mismatched():
    with pytest.raises(ValueError, match="empty"):
        stack_embeddings([])


# --- the ablation ------------------------------------------------------------------


def test_channel_report_ranks_the_informative_channel_first(planted):
    X, spec, y = planted
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0)
    assert rep.best == "signal"
    assert sorted(rep.rank.tolist()) == [1, 2, 3]        # rank is a permutation
    assert rep.n_samples == X.shape[0]


def test_delta_is_score_minus_base_and_gain_is_full_minus_base(planted):
    X, spec, y = planted
    rep = channel_report(X, spec, _corr_scorer(y), base=0.25)
    assert np.allclose(rep.delta, rep.score - 0.25)
    assert rep.gain == pytest.approx(rep.full - 0.25)
    # base=None -> delta is nan, ranking unchanged (delta is a constant shift of score)
    rep2 = channel_report(X, spec, _corr_scorer(y), base=None)
    assert np.isnan(rep2.delta).all()
    assert rep2.rank.tolist() == rep.rank.tolist()


def test_leave_one_out_flags_redundancy(planted):
    """The invariant that justifies mode='both': high delta_in + ~zero delta_out == redundant."""
    X, spec, y = planted
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0, mode="both")
    f = {c: i for i, c in enumerate(rep.channels)}
    assert rep.delta[f["signal"]] > 0.5 and rep.delta[f["dup"]] > 0.5      # both carry it alone...
    assert abs(rep.delta_out[f["signal"]]) < 0.1                           # ...neither is necessary
    assert abs(rep.delta_out[f["dup"]]) < 0.1


def test_leave_one_out_flags_a_uniquely_informative_channel():
    rng = np.random.default_rng(2)
    n = 200
    y = rng.integers(0, 2, n).astype(float)
    X, spec = (ChannelBuilder()
               .add("only", y[:, None] + rng.normal(0, 0.3, (n, 2)))
               .add("noise", rng.normal(0, 1, (n, 3)))
               .build(standardize=False, impute=False))
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0, mode="both")
    f = {c: i for i, c in enumerate(rep.channels)}
    assert rep.delta_out[f["only"]] > 0.3               # no duplicate to cover for it -> necessary


def test_permutation_pvalue_bounds_and_null_channel(planted):
    X, spec, y = planted
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0, n_permutations=50, seed=0)
    assert rep.pvalue is not None
    assert (rep.pvalue >= 1.0 / 51).all()               # add-one smoothed: never 0
    assert (rep.pvalue <= 1.0).all()
    f = {c: i for i, c in enumerate(rep.channels)}
    assert rep.pvalue[f["signal"]] == rep.pvalue.min()
    assert rep.pvalue[f["noise"]] == rep.pvalue.max()


def test_scorer_agnostic_two_scorers_agree(planted):
    """The library sees no labels in either case; only the closure does."""
    X, spec, y = planted

    def auc_like(B):
        s = B.mean(axis=1)
        pos, neg = s[y == 1], s[y == 0]
        return float((pos[:, None] > neg[None, :]).mean())

    a = channel_report(X, spec, _corr_scorer(y), base=0.0)
    b = channel_report(X, spec, auc_like, base=0.5)
    assert a.best == b.best == "signal"


def test_channels_arg_restricts_and_orders_and_validates(planted):
    X, spec, y = planted
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0, channels=["noise", "signal"])
    assert rep.channels == ["noise", "signal"]
    assert rep.frame().height == 2
    assert rep.best == "signal"
    with pytest.raises(ValueError, match="unknown channel"):
        channel_report(X, spec, _corr_scorer(y), channels=["nope"])
    with pytest.raises(ValueError, match="no channels"):
        channel_report(X, spec, _corr_scorer(y), channels=[])
    with pytest.raises(ValueError, match="mode must be"):
        channel_report(X, spec, _corr_scorer(y), mode="sideways")


def test_report_rejects_width_mismatch(planted):
    X, spec, y = planted
    with pytest.raises(ValueError, match="columns"):
        channel_report(X[:, :-1], spec, _corr_scorer(y))


def test_report_frame_schema_and_sort(planted):
    X, spec, y = planted
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0)
    fr = rep.frame()
    assert isinstance(fr, pl.DataFrame)
    assert fr.height == 3
    assert fr["channel"][0] == rep.best
    assert fr["rank"].to_list() == [1, 2, 3]
    for c in ("channel", "n_columns", "score", "delta", "rank", "attributable"):
        assert c in fr.columns
    assert "delta_out" not in fr.columns and "pvalue" not in fr.columns   # absent unless requested
    assert fr.filter(pl.col("channel") == "noise")["n_columns"][0] == 5


def test_single_channel_edge_case():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 50).astype(float)
    X, spec = ChannelBuilder().add("only", rng.normal(0, 1, (50, 2))).build(
        standardize=False, impute=False)
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0, mode="both")
    assert rep.rank.tolist() == [1] and rep.best == "only"


# --- attribution -------------------------------------------------------------------


def test_channel_drivers_surfaces_injected_motif(space):
    motif = _clonotypes(1, offset=300)
    base = _clonotypes(400, offset=0)
    pos = [pl.concat([_sample(base.sample(120, seed=s)),
                      _sample(motif, lambda n: np.full(n, 400.0))]) for s in range(5)]
    neg = [_sample(base.sample(120, seed=s + 50)) for s in range(5)]
    cands = pl.concat([base.sample(120, seed=0), motif]).unique()

    embs = [sample_embedding(space, s) for s in pos + neg]
    X, spec = stack_embeddings(embs)
    y = np.array([1.0] * len(pos) + [0.0] * len(neg))
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0)

    out = channel_drivers(rep, space=space, pos=pos, neg=neg, candidates=cands,
                          channel="mean", top=10)
    assert motif["junction_aa"][0] in out["junction_aa"].to_list()[:5]
    assert out["channel"].unique().to_list() == ["mean"]      # self-describing driver frame


def test_channel_drivers_refuses_non_attributable_channel(space):
    embs = [sample_embedding(space, _sample(_clonotypes(150, offset=o))) for o in (0, 150, 300)]
    X, spec = stack_embeddings(embs)
    y = np.array([1.0, 0.0, 1.0])
    rep = channel_report(X, spec, _corr_scorer(y), base=0.0)
    with pytest.raises(ValueError, match="not clonotype-attributable"):
        channel_drivers(rep, space=space, pos=[], neg=[], candidates=pl.DataFrame(),
                        channel="diversity")
    with pytest.raises(ValueError, match="unknown channel"):
        channel_drivers(rep, space=space, pos=[], neg=[], candidates=pl.DataFrame(), channel="nope")


def test_channel_spec_constructed_directly_is_usable():
    spec = ChannelSpec({"a": [0, 1], "b": [2]}, frozenset({"a"}))
    assert spec.width == 3 and spec.names == ["a", "b"]
    assert spec["a"] == [0, 1] and "a" in spec.attributable
