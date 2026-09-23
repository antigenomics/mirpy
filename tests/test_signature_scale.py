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
from vdjtools.signature import layout as LAYOUT

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

    def test_a_missing_default_artifact_returns_none_rather_than_raising(self, tmp_path,
                                                                          monkeypatch):
        """A signature without a scale is still a usable raw vector -- for the DEFAULT path.

        Narrowed from "any missing path returns None". That contract could not tell "you did not
        ask for a reference" from "the reference you named is missing", so a typo in --scale
        produced an unstandardised matrix indistinguishable from a standardised one.
        """
        S.load_scale.cache_clear()
        monkeypatch.setattr(S, "DEFAULT_PATH", tmp_path / "absent.npz")
        assert S.load_scale() is None


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

    def test_drops_a_locus_with_no_singletons(self):
        """A top-N cut removes the singleton tail, and coverage is 1 - f1/n — so it reads ~1.0.

        Freezing cstar from that would put every honest, untruncated sample into extrapolation.
        Dropped per locus rather than raised, so one pre-collapsed arm does not cost the others
        their reference.
        """
        draw = [(f"s{i}", {"TRB": self._rep(300, singletons=False, seed=i)}) for i in range(5)]
        with pytest.warns(RuntimeWarning, match="no singletons observed"):
            cstar, q05 = S.measure_constants(draw, n_pgen=50)
        assert "TRB" not in cstar, "a locus with no singleton tail must get no coverage level"
        assert "TRB" in q05, "Pgen is still measurable there"

    def test_one_bad_locus_does_not_cost_the_others(self):
        draw = [(f"s{i}", {"TRB": self._rep(400, singletons=True, seed=i),
                           "TRA": self._rep(300, singletons=False, seed=i)}) for i in range(5)]
        with pytest.warns(RuntimeWarning):
            cstar, _ = S.measure_constants(draw, n_pgen=50)
        assert "TRB" in cstar and "TRA" not in cstar

    def test_accepts_an_untruncated_repertoire(self):
        draw = [(f"s{i}", {"TRB": self._rep(500, singletons=True, seed=i)}) for i in range(5)]
        cstar, q05 = S.measure_constants(draw, n_pgen=50)
        assert 0.0 < cstar["TRB"] < 0.999
        assert q05["TRB"] < 0


class TestPartialCstar:
    """measure_constants omits loci it cannot establish, so a partial dict is the normal case."""

    @staticmethod
    def _frame(n=500, seed=0):
        rng = np.random.default_rng(seed)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        return pl.DataFrame({
            "v_call": ["TRBV20-1"] * n, "j_call": ["TRBJ2-2"] * n,
            "junction_aa": ["C" + "".join(rng.choice(aa, 12)) + "F" for _ in range(n)],
            "duplicate_count": np.ceil(rng.zipf(1.5, n).clip(1, 400)).astype(int).tolist()})

    def test_a_partial_dict_passed_explicitly_does_not_crash(self):
        """The path that broke the first real fit: cstar handed straight from measure_constants."""
        from mir.signature import signature

        v = signature({"TRB": self._frame()}, standardize="none", cstar={"TRB": 0.3})
        assert np.isfinite(v["vsig:div:TRB:1D_c"])

    def test_a_locus_without_a_level_masks_out_rather_than_borrowing_one(self):
        from mir.signature import signature

        v = signature({"TRB": self._frame()}, standardize="none", cstar={"TRA": 0.3})
        assert v["vsig:mask:TRB:estimable"] == 0.0
        assert np.isnan(v["vsig:div:TRB:1D_c"])
        assert np.isfinite(v["vsig:depth:TRB:reads"]), "depth is still measurable"


class TestGroupArgumentForms:
    """``group=`` is documented as "a column name in ``frame``, or a sequence" — so both must work."""

    @staticmethod
    def _frame(n: int = 400):
        rng = np.random.default_rng(0)
        return pl.DataFrame({"sample_id": [f"s{i}" for i in range(n)],
                             **{f"vsig:depth:TRB:c{i}": rng.normal(size=n) for i in range(5)}})

    @pytest.mark.parametrize("as_list", [False, True])
    def test_a_sequence_of_labels_is_accepted(self, as_list):
        """The column filter used to compare each name against ``group`` itself.

        With a sequence that broadcasts to an array, and ``and`` on an array raises before the fit
        starts — so the documented sequence form crashed while only the column-name form worked.
        """
        g = np.array(["A"] * 200 + ["B"] * 200)
        ref = S.fit_scale(self._frame(), group=list(g) if as_list else g, min_n_groups=0)
        assert len(ref.columns) == 5

    def test_a_column_name_is_excluded_from_the_signature_columns(self):
        g = np.array(["A"] * 200 + ["B"] * 200)
        frame = self._frame().with_columns(pl.Series("study_group", g))
        ref = S.fit_scale(frame, group="study_group", min_n_groups=0)
        assert "study_group" not in ref.columns
        assert len(ref.columns) == 5


class TestBlockPolicy:
    """``exempt`` and ``magnitude`` are layout *contract*; the scaler has to honour them.

    Both flags were declared, documented and asserted in the layout tests from the start, and
    nothing on the emission path read either — so ``rsig:contrast:*`` shipped median-centred in
    ``rsig_scale_v2.npz`` (66 columns, ``loc = 7.04`` on ``contrast:TRA:norm``). That inverts the
    block's stated meaning: an immune desert is ``Ψ = 0``, which centring maps to ``-87`` robust
    deviations and clips onto the far tail beside the most deviant samples in the corpus.
    """

    @staticmethod
    def _frame(n: int = 2000):
        rng = np.random.default_rng(0)
        return pl.DataFrame({
            "sample_id": [f"s{i}" for i in range(n)],
            "vsig:mask:TRB:present": rng.integers(0, 2, n).astype(float),
            "rsig:contrast:TRB:norm": 7.0 + rng.normal(0, 0.08, n),
            "rsig:contrast:TRB:PC01": -2.0 + rng.normal(0, 0.5, n),
            "rsig:contrast:TRB:PC02": 0.5 + rng.normal(0, 0.13, n),
            "vsig:depth:TRB:reads": 5.0 + rng.normal(0, 0.3, n),
        })

    def test_a_null_contrast_stays_at_the_origin(self):
        ref = S.fit_scale(self._frame(), min_n_obs=100)
        out = ref.apply({"rsig:contrast:TRB:PC01": 0.0, "rsig:contrast:TRB:PC02": 0.0})
        assert out == {"rsig:contrast:TRB:PC01": 0.0, "rsig:contrast:TRB:PC02": 0.0}

    def test_contrast_coordinates_share_one_scale_per_locus(self):
        """One uncentred number for the block, not a per-column z.

        Per-column scaling forces every coordinate to unit spread, which erases the *relative*
        sizes the contrast direction is made of.
        """
        ref = S.fit_scale(self._frame(), min_n_obs=100)
        i = ref.columns.index("rsig:contrast:TRB:PC01")
        j = ref.columns.index("rsig:contrast:TRB:PC02")
        assert ref.scale[i] == ref.scale[j] > 0

    def test_the_norm_keeps_ordinary_reference_z(self):
        """It is a log1p of a length, not a magnitude — sharing the coordinates' RMS would
        divide a value near 7 by the same constant as one near 0."""
        ref = S.fit_scale(self._frame(), min_n_obs=100)
        i = ref.columns.index("rsig:contrast:TRB:norm")
        assert abs(ref.loc[i] - 7.0) < 0.05 and 0 < ref.scale[i] < 0.2

    def test_a_mask_is_not_standardised(self):
        ref = S.fit_scale(self._frame(), min_n_obs=100)
        i = ref.columns.index("vsig:mask:TRB:present")
        assert ref.loc[i] == 0.0 and ref.scale[i] == 0.0
        assert ref.apply({"vsig:mask:TRB:present": 1.0}) == {"vsig:mask:TRB:present": 1.0}

    def test_an_ordinary_column_is_still_centred_and_scaled(self):
        ref = S.fit_scale(self._frame(), min_n_obs=100)
        i = ref.columns.index("vsig:depth:TRB:reads")
        assert abs(ref.loc[i] - 5.0) < 0.05 and ref.scale[i] > 0


def test_measure_constants_plumbs_threads_to_the_pgen_batch():
    """Pgen is the whole cost of a reference fit; unplumbed, the machine's cores go unused.

    Measured: a 400-sample seven-locus draw sat at 2 cores for 17 minutes because this argument
    stopped at the function boundary. ``vsig``'s ``pgen_block`` has always taken it.
    """
    import inspect

    from vdjtools.model.native import pgen_aa_batch

    assert "threads" in inspect.signature(S.measure_constants).parameters
    assert "threads" in inspect.signature(pgen_aa_batch).parameters

    seen = {}
    df = pl.DataFrame({"junction_aa": ["CASSLGQAYEQYF"] * 4, "duplicate_count": [3, 2, 1, 1],
                       "v_call": ["TRBV20-1"] * 4, "j_call": ["TRBJ2-2"] * 4})

    def spy(model, seqs, *, v=None, j=None, threads=0, **kw):
        seen["threads"] = threads
        return [1e-9] * len(seqs)

    import vdjtools.model.native as native
    real = native.pgen_aa_batch
    native.pgen_aa_batch = spy
    try:
        S.measure_constants([("s0", {"TRB": df})], loci=("TRB",), threads=7)
    finally:
        native.pgen_aa_batch = real
    assert seen.get("threads") == 7, "threads stopped at the function boundary"


class TestTheStudyCountIsAFloorOfItsOwn:
    """Samples inside a study are not independent observations of the population.

    A column seen a thousand times across three studies is a claim about three studies, and
    ``min_n_obs`` alone cannot see the difference. Drawing whole studies from the 947-group blood
    reference, 20 studies put 0.16 of columns inside the acceptance gate and 40 put 0.25
    (analysis repo, ``benchmarks/SIGNATURE_SCALE_N.md``), so a corpus below that floor produces a
    reference worse than none.
    """

    @staticmethod
    def _frame(groups: int, per_group: int = 40, thin_col: bool = False):
        rng = np.random.default_rng(0)
        n = groups * per_group
        g = [f"study{i // per_group}" for i in range(n)]
        cols = {f"vsig:depth:TRB:c{i}": rng.normal(size=n) for i in range(3)}
        if thin_col:                                   # observed in one study only
            v = np.full(n, np.nan)
            v[:per_group] = rng.normal(size=per_group)
            cols["vsig:depth:TRB:thin"] = v
        return pl.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "study_group": g, **cols})

    def test_a_corpus_of_too_few_studies_is_refused(self):
        with pytest.raises(ValueError, match="min_n_groups"):
            S.fit_scale(self._frame(6), group="study_group", min_n_obs=10)

    def test_the_floor_is_opt_outable(self):
        ref = S.fit_scale(self._frame(6), group="study_group", min_n_obs=10, min_n_groups=0)
        assert ref.scaled.all()

    def test_a_column_seen_in_too_few_studies_stays_unscaled(self):
        ref = S.fit_scale(self._frame(30, thin_col=True), group="study_group", min_n_obs=10)
        i = list(ref.columns).index("vsig:depth:TRB:thin")
        assert not ref.scaled[i], "one study cannot support a scale, however many samples it has"
        assert ref.scaled.sum() == 3
        assert ref.meta["n_groups"] == 30

    def test_without_group_nothing_changes(self):
        frame = self._frame(6).drop("study_group")
        assert S.fit_scale(frame, min_n_obs=10).scaled.all()


class TestTheConstantsAreQuantilesToo:
    """``cstar`` and ``pgen_q05`` are corpus quantiles, so they inherit the same voting problem.

    ``pgen_q05`` inherits it twice over: it pools ``n_pgen`` junctions *per sample*, so a large
    study contributes both more samples and, through them, proportionally more junctions.
    """

    def test_one_dominant_group_sets_the_unweighted_quantile(self):
        chunks = [np.array([0.9])] * 100 + [np.array([0.1])] * 5     # 100 deep, 5 shallow
        labels = ["big"] * 100 + [f"s{i}" for i in range(5)]
        assert S._quantile_by_group(chunks, labels, 0.10, weighted=False) == 0.9
        assert S._quantile_by_group(chunks, labels, 0.10, weighted=True) == 0.1

    def test_a_deeply_drawn_sample_does_not_outvote_a_shallow_one(self):
        # same two studies, but one sample contributed 2,000 junctions and the other 10
        chunks = [np.full(2000, -6.0), np.full(10, -12.0)]
        labels = ["studyA", "studyB"]
        assert S._quantile_by_group(chunks, labels, 0.05, weighted=False) == -6.0
        assert S._quantile_by_group(chunks, labels, 0.05, weighted=True) == -12.0

    def test_unweighted_matches_numpy(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=500)
        chunks = [np.array([x]) for x in v]
        got = S._quantile_by_group(chunks, ["a"] * 500, 0.10, weighted=False)
        assert got == pytest.approx(float(np.quantile(v, 0.10)))


class TestBatchRatioIsReportedHonestly:
    """``batch_dominated`` is all-False both when the reference is clean and when nothing could
    be measured. Those are opposite statements and the report has to tell them apart.

    The case is not hypothetical: a task-disjoint reference draws a handful of samples per study
    on purpose, so every group falls under ``_batch_ratio``'s floor and the ratio is NaN
    throughout — which would print ``batch_dominated: 0`` and read as a clean bill of health.
    """

    @staticmethod
    def _frame(n: int, per_group: int):
        rng = np.random.default_rng(0)
        g = [f"study{i // per_group}" for i in range(n)]
        return pl.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "study_group": g,
                             **{f"vsig:depth:TRB:c{i}": rng.normal(size=n) for i in range(4)}})

    def test_groups_too_small_to_measure_are_not_reported_as_clean(self):
        ref = S.fit_scale(self._frame(400, per_group=8), group="study_group", min_n_obs=100)
        rep = ref.report()
        assert rep["batch_measured"] == 0, "nothing was measurable at 8 samples per group"
        assert rep["batch_dominated"] == 0
        assert np.isnan(ref.batch_ratio).all()

    def test_groups_large_enough_are_measured(self):
        ref = S.fit_scale(self._frame(400, per_group=200), group="study_group", min_n_obs=100,
                          min_n_groups=0)
        assert ref.report()["batch_measured"] == 4

    def test_the_report_says_how_many_groups_voted(self):
        """"266 dominated" means one thing over 40 studies and another over 238."""
        ref = S.fit_scale(self._frame(400, per_group=50), group="study_group", min_n_obs=100,
                          min_n_groups=0)
        assert ref.report()["batch_groups"] == 8         # 400/50, all clearing MIN_PER_GROUP=25
        thin = S.fit_scale(self._frame(400, per_group=8), group="study_group", min_n_obs=100,
                           min_n_groups=0)
        assert thin.report()["batch_groups"] == 0

    def test_sampling_noise_in_a_group_centre_is_not_counted_as_batch(self):
        """Groups drawn from ONE distribution have no batch effect, however noisy their centres.

        Without the moment correction the ratio is ~1/sqrt(n_g) per group and rises as the floor
        drops, which is what forced the old floor of 100 and cost 6/7 of the corpus its vote.
        """
        rng = np.random.default_rng(0)
        n, per = 3000, 30                                # 100 groups of 30 — all i.i.d. N(0,1)
        frame = pl.DataFrame({"sample_id": [f"s{i}" for i in range(n)],
                              "study_group": [f"g{i // per}" for i in range(n)],
                              "vsig:depth:TRB:reads": rng.normal(size=n)})
        ref = S.fit_scale(frame, group="study_group", min_n_obs=100, min_n_groups=0)
        # uncorrected this same draw reads 0.234; corrected, 0.063. The truth is 0.
        assert ref.batch_ratio[0] < 0.12, (
            f"pure sampling noise read as batch: ratio {ref.batch_ratio[0]:.3f}")


class TestSignatureColumnsAreSelectedByTheContract:
    """A frame carries more than its signature. Only parseable names may be fitted."""

    @staticmethod
    def _frame(extra: dict) -> pl.DataFrame:
        cols = LAYOUT.columns("core")
        rng = np.random.default_rng(0)
        d = {c: rng.normal(size=64) for c in cols}
        d["sample_id"] = [f"s{i}" for i in range(64)]
        d.update(extra)
        return pl.DataFrame(d)

    def test_a_string_metadata_column_does_not_break_the_fit(self):
        ref = S.fit_scale(self._frame({"loci": ["TRA+TRB"] * 64}))
        assert "loci" not in ref.columns
        assert set(ref.columns) == set(LAYOUT.columns("core"))

    def test_a_numeric_metadata_column_is_not_silently_fitted(self):
        """The one that hides: it never raises, it just joins the frozen artifact.

        `age` has a finite median and MAD, so a deny-list based selection gives it a loc/scale,
        writes it into the npz, and rescales it on every future sample -- under a name
        `layout.parse` cannot read, so nothing downstream can even name what went wrong.
        """
        ref = S.fit_scale(self._frame({"age": np.arange(64, dtype=float)}))
        assert "age" not in ref.columns
        assert len(ref.columns) == len(LAYOUT.columns("core"))

    def test_the_group_column_is_still_excluded(self):
        ref = S.fit_scale(self._frame({"study": ["a", "b"] * 32}), group="study", min_n_groups=0)
        assert "study" not in ref.columns
        assert ref.batch_ratio.size == len(LAYOUT.columns("core"))

    def test_a_frame_of_only_metadata_is_refused(self):
        with pytest.raises(ValueError, match="no signature columns"):
            S.fit_scale(pl.DataFrame({"sample_id": ["a"], "age": [1.0]}))


class TestLoadScaleDistinguishesAbsentFromMisnamed:
    def test_an_explicit_missing_path_raises(self, tmp_path):
        """A typo in --scale must not quietly yield an unstandardised matrix."""
        S.load_scale.cache_clear()
        with pytest.raises(FileNotFoundError, match="no scale reference"):
            S.load_scale(tmp_path / "typo.npz")

    def test_an_explicit_present_path_loads(self, tmp_path):
        S.load_scale.cache_clear()
        ref = S.fit_scale(cohort(n=80))
        S.save_scale(ref, tmp_path / "s.npz")
        assert S.load_scale(tmp_path / "s.npz").columns == ref.columns


def test_measure_constants_pools_the_same_draw_the_block_scores():
    """The reference and the statistic compared to it must come off one distribution.

    ``pgen_block`` scores a random draw of junctions; ``measure_constants`` fits the q05 those
    scores are compared against. Head-slicing the reference instead shifted pooled q05 by -0.185
    log10 -- a fixed offset in every sample's ``frac_atypical``, landing in a wholly plausible
    range with nothing to flag it. Both sides must call the one shared draw.

    Lives here rather than in vdjtools: ``measure_constants`` is mirpy's, and mirpy may import
    vdjtools while the reverse would invert the dependency.
    """
    import inspect

    from vdjtools.signature.blocks import pgen_junctions

    src = inspect.getsource(S.measure_constants)
    assert "pgen_junctions(df, locus, n_pgen)" in src, "the reference draws its own pool"
    assert "to_list()[:n_pgen]" not in src, "the reference head-slices a sorted frame"
    assert callable(pgen_junctions)


class TestNamedModels:
    """`load_scale` resolves a model name, and refuses a name it does not know.

    The rotation is one artifact for every model; only the per-column scale and the per-locus
    coverage constant differ, and those are assay-specific. A name that resolves to an artifact
    which is not installed must raise rather than fall back to the bundled one -- silently reading
    bulk RNA-seq against a reference fitted on targeted TCR libraries is a 3.2x error in the
    coverage level every Hill number is compared at.
    """

    def test_the_bundled_model_loads_by_name(self):
        from mir.signature.scale import load_scale

        ref = load_scale("deep-tcr")
        assert ref is not None
        assert set(ref.cstar) == {"TRA", "TRB"}, "the bundled reference covers TRA and TRB only"

    def test_a_model_with_no_installed_artifact_raises(self):
        import pytest

        from mir.signature.scale import MODELS, load_scale, _RES

        missing = [n for n, f in MODELS.items() if not (_RES / f).exists()]
        if not missing:
            pytest.skip("every declared model is installed")
        with pytest.raises(FileNotFoundError, match="not installed"):
            load_scale(missing[0])

    def test_an_unknown_name_raises_rather_than_being_read_as_a_path(self):
        import pytest

        from mir.signature.scale import load_scale

        with pytest.raises(ValueError, match="unknown scale reference"):
            load_scale("bloood")


class TestBloodV3:
    """`blood-v3` covers the five loci the bundled amplicon reference does not.

    The shipped `deep-tcr` reference was fitted on targeted TCR libraries, so it carries `cstar`
    for TRA and TRB and nothing else, and every coverage-standardised diversity column on the
    other five loci comes back nan. `blood-v3` is the 947-study bulk-blood fit; it has the SAME
    column order, so it is a drop-in -- no coordinate moves, only holes get filled.
    """

    def test_it_covers_all_seven_loci(self):
        from mir.signature.scale import load_scale

        ref = load_scale("blood-v3")
        assert set(ref.cstar) == {"IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG"}

    def test_it_scales_far_more_columns_than_the_amplicon_reference(self):
        import numpy as np

        from mir.signature.scale import load_scale

        deep, blood = load_scale("deep-tcr"), load_scale("blood-v3")
        n_deep = int(np.asarray(deep.scaled).sum())
        n_blood = int(np.asarray(blood.scaled).sum())
        assert n_deep == 394, f"the bundled amplicon fit scales 394 columns, got {n_deep}"
        assert n_blood >= 1377, f"the blood fit scales >= 1377 columns, got {n_blood}"

    def test_the_column_order_is_identical_so_it_is_a_drop_in(self):
        """The layout must not move between references, or column i changes meaning."""
        from mir.signature.scale import load_scale

        assert load_scale("deep-tcr").columns == load_scale("blood-v3").columns


class TestV4Models:
    """All six named models resolve, and weighted is not the same fit as unweighted.

    The weighting is the whole point of v4 -- v3 was fitted under mirpy 3.9.0, which predates
    `weight_by_group` entirely, so it is a per-sample fit whether or not anyone intended one. If
    the two v4 arms came out identical the parameter would not be doing anything.
    """

    def test_every_declared_model_is_installed(self):
        from mir.signature.scale import MODELS, load_scale

        missing = []
        for name in MODELS:
            try:
                load_scale(name)
            except FileNotFoundError:
                missing.append(name)
        assert not missing, f"declared but not bundled: {missing}"

    def test_every_rna_seq_model_covers_all_seven_loci(self):
        from mir.signature.scale import load_scale

        seven = {"IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG"}
        for name in ("blood", "blood-unweighted", "blood-v3", "tissue", "tissue-unweighted"):
            assert set(load_scale(name).cstar) == seven, f"{name} does not cover all seven loci"

    def test_weighted_and_unweighted_are_different_fits(self):
        import numpy as np

        from mir.signature.scale import load_scale

        for w, u in (("blood", "blood-unweighted"), ("tissue", "tissue-unweighted")):
            a, b = load_scale(w), load_scale(u)
            both = np.asarray(a.scaled) & np.asarray(b.scaled)
            assert both.sum() > 1000, "the two arms should share most columns"
            assert not np.array_equal(np.asarray(a.loc)[both], np.asarray(b.loc)[both]), (
                f"{w} and {u} have identical locations -- weight_by_group did nothing")

    def test_the_column_layout_never_moves_between_models(self):
        """Column i must mean the same thing under every reference, or the contract is void."""
        from mir.signature.scale import MODELS, load_scale

        ref = load_scale("deep-tcr").columns
        for name in MODELS:
            assert load_scale(name).columns == ref, f"{name} moves the column layout"


class TestKmerSpaces:
    """The k-mer spaces ship, and registering them cannot move an existing coordinate."""

    def test_the_bundled_spaces_load_for_all_seven_loci(self):
        from mir.signature import load_kmer_spaces

        sp = load_kmer_spaces()
        assert set(sp) == {"IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG"}
        for locus, s in sp.items():
            assert s.components.shape[0] == 32, f"{locus} was fitted at rank 32"
            assert s.components.shape[1] == len(s.codes)

    def test_they_load_without_pickle(self, tmp_path):
        """A space is meant to travel, so reading one must not be able to execute code."""
        import numpy as np

        from mir.signature import KMER_PATH

        np.load(KMER_PATH, allow_pickle=False)   # raises if any array is dtype=object

    def test_registering_appends_and_leaves_the_narrow_tiers_alone(self):
        from vdjtools.signature import layout as L
        from vdjtools.signature.kmer import register_kmer

        from mir.signature import load_kmer_spaces

        before = {t: L.columns(t) for t in ("core", "standard", "full")}
        register_kmer(load_kmer_spaces(), tier="full")
        after = {t: L.columns(t) for t in ("core", "standard", "full")}

        assert after["core"] == before["core"], "core tier moved"
        assert after["standard"] == before["standard"], "standard tier moved"
        assert after["full"][:len(before["full"])] == before["full"], (
            "full tier moved -- an existing vector's column i would change meaning")
        assert len(after["full"]) == len(before["full"]) + 231
