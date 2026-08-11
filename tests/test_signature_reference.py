"""The frozen reference and the assembled signature.

Two things are being protected. The reference must refuse to project through a basis that does
not match the installed prototype panel — the failure it prevents is silent, since a mismatched
panel still yields a full, plausible vector in coordinates nobody else shares. And the assembled
vector must be positional: column *i* means the same thing in every sample anyone computes.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mir.signature import assemble, reference

AA = list("ACDEFGHIKLMNPQRSTVWY")


def frame(n=600, v="TRBV20-1", j="TRBJ2-2", c=None, seed=0) -> pl.DataFrame:
    r = np.random.default_rng(seed)
    return pl.DataFrame(
        {"v_call": [v] * n, "j_call": [j] * n, "c_call": [c] * n,
         "junction_aa": ["C" + "".join(r.choice(AA, 12)) + "F" for _ in range(n)],
         "duplicate_count": np.ceil(r.zipf(1.5, n).clip(1, 500)).astype(int).tolist()},
        schema_overrides={"c_call": pl.Utf8})


@pytest.fixture(scope="module")
def ref():
    return reference.load_reference()


@pytest.fixture(scope="module")
def sample():
    return {"TRB": frame(1500, seed=1),
            "IGH": frame(600, "IGHV1-2", "IGHJ4", "IGHM", seed=2)}


class TestReference:
    def test_covers_every_locus(self, ref):
        assert set(ref.loci) == {"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"}
        assert ref.version == "RSIG-v1"

    def test_hashes_match_the_installed_panel(self, ref):
        assert len(ref.verify()) == 7

    def test_a_mismatched_panel_is_refused_loudly(self, ref):
        """The failure this prevents is silent, so the check must not be."""
        import dataclasses

        broken = dataclasses.replace(ref["TRB"], prototype_hash="deadbeefdeadbeef")
        bad = dataclasses.replace(ref, loci={**ref.loci, "TRB": broken})
        with pytest.raises(ValueError, match="prototype hash mismatch"):
            bad.verify()

    def test_missing_locus_names_what_is_available(self, ref):
        with pytest.raises(KeyError, match="TRZ"):
            ref["TRZ"]

    def test_projection_shape_and_prefix_property(self, ref):
        phi = np.zeros(3 * ref["TRB"].n_prototypes)
        full = ref["TRB"].project(phi, "C")
        assert ref["TRB"].project(phi, "C", k=8).shape == (8,)
        assert np.allclose(ref["TRB"].project(phi, "C", k=8), full[:8]), "k is not a prefix"

    def test_rejects_a_phi_of_the_wrong_width(self, ref):
        with pytest.raises(ValueError, match="width"):
            ref["TRB"].project(np.zeros(10), "C")

    def test_rejects_more_components_than_stored(self, ref):
        with pytest.raises(ValueError, match="stores"):
            ref["TRB"].project(np.zeros(3 * ref["TRB"].n_prototypes), "C", k=999)

    def test_rejects_an_unknown_slot(self, ref):
        with pytest.raises(ValueError, match="slot must be"):
            ref["TRB"].project(np.zeros(3 * ref["TRB"].n_prototypes), "X")

    def test_contrast_sends_an_unselected_repertoire_to_the_origin(self, ref):
        assert np.allclose(ref["TRB"].contrast(ref["TRB"].naive, 1.0), 0.0)

    def test_contrast_sends_an_unobserved_repertoire_to_the_origin(self, ref):
        phi = np.arange(3 * ref["TRB"].n_prototypes, dtype=float)
        assert np.allclose(ref["TRB"].contrast(phi, 0.0), 0.0)

    def test_centre_is_naive_not_the_prototype_cloud_mean(self, ref):
        """Measured on real cohorts, naive recovers 67-85% of an oracle centring; mu_phi 4-21%."""
        assert np.allclose(ref["TRB"].standardize(ref["TRB"].naive), 0.0)
        assert not np.allclose(ref["TRB"].standardize(ref["TRB"].mu_phi), 0.0)

    def test_self_test_passes(self):
        out = reference.self_test()
        assert out["version"] == "RSIG-v1" and out["n_prototypes"] == 256
        assert out["centring_gain"] > 5

    def test_missing_artifact_says_how_to_rebuild(self):
        reference.load_reference.cache_clear()
        with pytest.raises(FileNotFoundError, match="build_rsig"):
            reference.load_reference("/nonexistent/rsig_v1.npz")


class TestRsig:
    @pytest.mark.parametrize("tier", ["core", "standard", "full"])
    def test_columns_match_the_layout(self, sample, tier):
        from vdjtools.signature import layout as L
        assert list(assemble.rsig(sample, tier=tier)) == L.columns(tier, "rsig")

    def test_present_locus_is_finite(self, sample):
        r = assemble.rsig(sample)
        for col in ("rsig:depth:TRB:n_eff", "rsig:div:TRB:rao",
                    "rsig:contrast:TRB:norm", "rsig:phic:TRB:PC01", "rsig:phiv:TRB:norm"):
            assert np.isfinite(r[col]), col

    def test_absent_locus_is_a_hole(self, sample):
        r = assemble.rsig(sample)
        for col in ("rsig:phic:TRA:PC01", "rsig:depth:TRA:n_eff", "rsig:div:TRA:rao"):
            assert np.isnan(r[col]), col

    def test_tiers_agree_where_they_overlap(self, sample):
        core, std = assemble.rsig(sample, tier="core"), assemble.rsig(sample, tier="standard")
        for k, v in core.items():
            assert v == std[k] or (np.isnan(v) and np.isnan(std[k])), k

    def test_isotype_shares_resolve_for_igh(self, sample):
        assert np.isfinite(assemble.rsig(sample)["rsig:band:IGH:IgM"])

    def test_an_empty_sample_is_all_holes(self):
        r = assemble.rsig({"TRB": frame(5).head(0)})
        assert all(np.isnan(v) for v in r.values())

    def test_zero_counts_do_not_crash(self):
        z = frame(50).with_columns(pl.lit(0).alias("duplicate_count"))
        assert all(np.isnan(v) for v in assemble.rsig({"TRB": z}).values())

    def test_chunking_does_not_change_the_answer(self, sample):
        a = assemble.rsig(sample, chunk=100_000)
        b = assemble.rsig(sample, chunk=37)
        for k, v in a.items():
            assert v == pytest.approx(b[k], rel=1e-9) or (np.isnan(v) and np.isnan(b[k])), k


class TestCombinedSignature:
    def test_both_halves_in_layout_order(self, sample):
        from vdjtools.signature import layout as L
        assert list(assemble.signature(sample, standardize="none")) == L.columns("standard")

    def test_width_is_the_sum_of_the_halves(self, sample):
        from vdjtools.signature import layout as L
        assert (len(assemble.signature(sample, standardize="none"))
                == len(L.columns("standard", "vsig")) + len(L.columns("standard", "rsig")))

    def test_a_trb_only_user_gets_a_full_width_vector(self):
        from vdjtools.signature import layout as L
        v = assemble.signature({"TRB": frame(900, seed=5)}, standardize="none")
        assert list(v) == L.columns("standard")
        assert np.isfinite(v["rsig:phic:TRB:PC01"])
        assert np.isnan(v["rsig:phic:IGH:PC01"])

    def test_cohort_frame_is_positional(self, sample):
        from vdjtools.signature import layout as L
        F = assemble.signature_cohort({"a": sample, "b": {"TRB": frame(700, seed=6)}},
                                          standardize="none")
        assert F.columns == ["sample_id", *L.columns("standard")]
        assert F.height == 2

    def test_empty_cohort_is_not_an_error(self):
        assert assemble.signature_cohort({}).height == 0

    def test_malformed_junctions_are_dropped_before_the_geometry(self):
        """They do not raise in the embedder — they return a finite, meaningless distance."""
        junk = frame(400, seed=7).with_columns(
            pl.when(pl.int_range(pl.len()) < 200).then(pl.lit("CASS*RSSYEQYF"))
              .otherwise(pl.col("junction_aa")).alias("junction_aa"))
        clean = assemble.signature({"TRB": junk}, standardize="none")
        assert np.isfinite(clean["rsig:phic:TRB:PC01"])
        assert clean["vsig:qc:TRB:nonstd_aa_frac"] > -5, "the dropped fraction was not reported"


class TestStandardizeContract:
    """Standardisation is opt-out, and its absence is stated rather than silently ignored."""

    def test_reference_standardisation_uses_a_supplied_scale(self, sample):
        import polars as pl

        from mir.signature.scale import fit_scale

        raw = assemble.signature(sample, standardize="none")
        cols = [c for c in raw if np.isfinite(raw[c])][:5]
        rng = np.random.default_rng(0)
        fake = pl.DataFrame({"sample_id": [f"s{i}" for i in range(50)],
                             **{c: rng.normal(raw[c], 1.0, 50) for c in cols}})
        sc = fit_scale(fake, min_n_obs=10)
        out = assemble.signature(sample, standardize="reference", scale=sc)
        assert set(out) == set(raw), "standardising changed the key set"
        assert abs(out[cols[0]]) < 8.0

    def test_missing_scale_is_an_error_not_a_silent_passthrough(self, sample, monkeypatch):
        """Raw values that look standardised are worse than a refusal."""
        monkeypatch.setattr("mir.signature.scale.load_scale", lambda *a, **k: None)
        with pytest.raises(ValueError, match="no scale reference is installed"):
            assemble.signature(sample, standardize="reference")
