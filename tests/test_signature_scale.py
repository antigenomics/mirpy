"""The corpus-fitted scale: what it standardises, and what it refuses to.

The reference exists so a collaborator's matrix and ours are on the same scale. Most of what is
protected here is therefore about *not* pretending: a column the corpus barely saw gets no scale,
a hole is not something to centre, and asking for standardisation that is not installed raises
rather than quietly returning raw numbers that look standardised.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mir.signature import scale as S

COLS = ["vsig:depth:TRB:reads", "vsig:div:TRB:1D_c", "rsig:phic:TRB:PC01"]


def cohort(n=400, seed=0, holes=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=[5.0, 1.5, 0.0], scale=[0.3, 0.2, 2.0], size=(n, 3))
    if holes:
        X[:holes, 1] = np.nan
    return pl.DataFrame({"sample_id": [f"s{i}" for i in range(n)],
                         **{c: X[:, i] for i, c in enumerate(COLS)}})


class TestFit:
    def test_scales_columns_with_enough_observations(self):
        ref = S.fit_scale(cohort(), min_n_obs=100)
        assert ref.scaled.all()
        assert ref.report()["scaled"] == 3

    def test_refuses_a_column_the_corpus_barely_saw(self):
        """A location fitted on a handful of samples is not a reference."""
        ref = S.fit_scale(cohort(n=50), min_n_obs=1000)
        assert not ref.scaled.any()
        assert ref.report()["unscaled"] == 3

    def test_holes_are_not_counted_as_observations(self):
        ref = S.fit_scale(cohort(holes=25), min_n_obs=10)
        i = ref.columns.index("vsig:div:TRB:1D_c")
        assert ref.n_obs[i] == 400 - 25

    def test_statistics_come_from_observed_entries_only(self):
        """Imputing first would deflate the scale in proportion to sparsity."""
        full = S.fit_scale(cohort(seed=1), min_n_obs=10)
        holed = S.fit_scale(cohort(seed=1, holes=100), min_n_obs=10)
        i = full.columns.index("vsig:div:TRB:1D_c")
        assert holed.scale[i] == pytest.approx(full.scale[i], rel=0.35)

    def test_scale_is_robust_to_an_outlier(self):
        c = cohort(seed=2)
        spiked = c.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(1e9)
              .otherwise(pl.col(COLS[0])).alias(COLS[0]))
        a = S.fit_scale(c, min_n_obs=10).scale[0]
        b = S.fit_scale(spiked, min_n_obs=10).scale[0]
        assert b == pytest.approx(a, rel=0.05)

    def test_a_constant_column_gets_no_scale(self):
        c = cohort(seed=3).with_columns(pl.lit(1.0).alias(COLS[0]))
        assert S.fit_scale(c, min_n_obs=10).scale[0] == 0.0

    def test_rejects_a_frame_without_columns(self):
        with pytest.raises(ValueError, match="no signature columns"):
            S.fit_scale(pl.DataFrame({"sample_id": ["a"]}))


class TestApply:
    @staticmethod
    @pytest.fixture(scope="class")
    def ref():
        return S.fit_scale(cohort(seed=4), min_n_obs=10)

    def test_the_median_sample_lands_at_the_origin(self, ref):
        c = cohort(seed=4)
        med = {col: float(np.nanmedian(c[col].to_numpy())) for col in COLS}
        assert all(abs(v) < 1e-9 for v in ref.apply(med).values())

    def test_key_set_is_preserved(self, ref):
        row = {c: 1.0 for c in COLS}
        assert set(ref.apply(row)) == set(row)

    def test_a_hole_stays_a_hole(self, ref):
        """nan is not something to centre — it is the absence of a measurement."""
        assert np.isnan(ref.apply({COLS[0]: float("nan")})[COLS[0]])

    def test_an_unknown_column_passes_through(self, ref):
        assert ref.apply({"vsig:invented:XX:col": 3.0})["vsig:invented:XX:col"] == 3.0

    def test_an_unscaled_column_passes_through(self):
        thin = S.fit_scale(cohort(n=40), min_n_obs=10_000)
        row = {c: 7.0 for c in COLS}
        assert thin.apply(row) == row

    def test_clipping_bounds_an_extreme_value(self, ref):
        out = ref.apply({COLS[0]: 1e9}, clip=8.0)
        assert out[COLS[0]] == 8.0

    def test_output_is_roughly_unit_scale(self, ref):
        c = cohort(seed=4)
        vals = [ref.apply({COLS[2]: float(v)})[COLS[2]] for v in c[COLS[2]].to_numpy()]
        assert 0.8 < float(np.std(vals)) < 1.6


class TestRoundTrip:
    def test_save_and_load_preserve_everything(self, tmp_path):
        ref = S.fit_scale(cohort(seed=5), min_n_obs=10,
                          cstar={"TRB": 0.31}, pgen_q05={"TRB": -12.5},
                          meta={"source": "unit test"})
        p = S.save_scale(ref, tmp_path / "sc.npz")
        S.load_scale.cache_clear()
        back = S.load_scale(p)
        assert back.columns == ref.columns
        assert np.allclose(back.loc, ref.loc) and np.allclose(back.scale, ref.scale)
        assert back.cstar == {"TRB": 0.31}
        assert back.pgen_q05 == {"TRB": -12.5}
        assert np.array_equal(back.n_obs, ref.n_obs)

    def test_missing_artifact_returns_none_rather_than_raising(self, tmp_path):
        """A signature without a scale is still a usable raw vector."""
        S.load_scale.cache_clear()
        assert S.load_scale(tmp_path / "absent.npz") is None


class TestStandardizeWiring:
    def test_standardize_none_returns_raw(self):
        from mir.signature import signature

        rng = np.random.default_rng(0)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        df = pl.DataFrame({
            "v_call": ["TRBV20-1"] * 400, "j_call": ["TRBJ2-2"] * 400,
            "junction_aa": ["C" + "".join(rng.choice(aa, 12)) + "F" for _ in range(400)],
            "duplicate_count": rng.integers(1, 200, 400).tolist()})
        v = signature({"TRB": df}, standardize="none")
        assert v["vsig:depth:TRB:reads"] > 3, "raw log10 reads should be on its natural scale"

    def test_unknown_standardize_raises(self):
        from mir.signature import signature

        with pytest.raises(ValueError, match="standardize must be"):
            signature({"TRB": pl.DataFrame()}, standardize="yes-please")


class TestConstantsGuard:
    """Truncated input must not be allowed to freeze a nonsense coverage level."""

    @staticmethod
    def _rep(n, singletons=True, seed=0):
        rng = np.random.default_rng(seed)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        counts = (np.ceil(rng.zipf(1.6, n).clip(1, 500)).astype(int) if singletons
                  else rng.integers(5, 500, n))
        return pl.DataFrame({
            "v_call": ["TRBV20-1"] * n, "j_call": ["TRBJ2-2"] * n,
            "junction_aa": ["C" + "".join(rng.choice(aa, 12)) + "F" for _ in range(n)],
            "duplicate_count": counts.tolist()})

    def test_refuses_input_with_no_singletons(self):
        """A top-N cut removes the singleton tail, and coverage is 1 - f1/n — so it reads 1.0.

        Freezing cstar from that would put every honest, untruncated sample into extrapolation.
        """
        draw = [(f"s{i}", {"TRB": self._rep(300, singletons=False, seed=i)}) for i in range(5)]
        with pytest.raises(ValueError, match="no singletons were observed"):
            S.measure_constants(draw, n_pgen=50)

    def test_accepts_an_untruncated_repertoire(self):
        draw = [(f"s{i}", {"TRB": self._rep(500, singletons=True, seed=i)}) for i in range(5)]
        cstar, q05 = S.measure_constants(draw, n_pgen=50)
        assert 0.0 < cstar["TRB"] < 0.999
        assert q05["TRB"] < 0
