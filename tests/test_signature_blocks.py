"""The geometry blocks: the algebraic identities they rest on, and the centring they need.

Every rsig column is a functional of ``Φ = Σ w_σ z_σ``, so the tests here are mostly identity
checks — linearity, exact strides, the telescoped Rao sum, the mixture identity. The exception
is :func:`test_centring_is_what_makes_phi_discriminative`, which records *why* the frozen
reference mean is load-bearing rather than cosmetic.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mir.embedding.tcremp import TCREmp
from mir.repertoire import rao_dispersion
from mir.signature import blocks as B
from mir.signature import reference as R

AA = list("ACDEFGHIKLMNPQRSTVWY")


def repertoire(n: int = 300, seed: int = 0, locus: str = "TRB") -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    v, j = ("TRBV20-1", "TRBJ2-2") if locus == "TRB" else ("IGHV1-2", "IGHJ4")
    return pl.DataFrame({
        "v_call": [v] * n, "j_call": [j] * n, "c_call": [None] * n,
        "junction_aa": ["C" + "".join(rng.choice(AA, 12)) + "F" for _ in range(n)],
        "duplicate_count": np.ceil(rng.zipf(1.6, n).clip(1, 5000)).astype(int).tolist(),
    })


@pytest.fixture(scope="module")
def model():
    return TCREmp.from_defaults("human", "TRB", n_prototypes=64)


class TestWeights:
    def test_weights_close(self):
        w = B.weights(repertoire()["duplicate_count"].to_numpy())
        assert w.sum() == pytest.approx(1.0)

    def test_unknown_weight_raises(self):
        with pytest.raises(ValueError, match="unknown weight"):
            B.weights(np.array([1, 2]), weight="magic")

    def test_all_zero_counts_raise(self):
        with pytest.raises(ValueError, match="no usable counts"):
            B.weights(np.zeros(5))


class TestPrototypeSum:
    def test_phi_is_the_weighted_mean_of_the_rows(self, model):
        df = repertoire()
        w = B.weights(df["duplicate_count"].to_numpy())
        phi, _ = B.prototype_sum(df, model, w)
        assert np.allclose(phi, w @ model.embed(df).astype(np.float64))

    def test_chunking_does_not_change_the_answer(self, model):
        df = repertoire()
        w = B.weights(df["duplicate_count"].to_numpy())
        a, sa = B.prototype_sum(df, model, w)
        b, sb = B.prototype_sum(df, model, w, chunk=37)
        assert np.allclose(a, b) and sa == pytest.approx(sb)

    def test_rao_telescopes_out_of_the_accumulators(self, model):
        """Two running sums, never an n x n Gram matrix."""
        df = repertoire()
        w = B.weights(df["duplicate_count"].to_numpy())
        phi, mean_sq = B.prototype_sum(df, model, w)
        z = model.embed(df).astype(np.float64)
        assert 2.0 * (mean_sq - phi @ phi) == pytest.approx(
            rao_dispersion(z, w, correct=False), rel=1e-9)

    def test_linearity_in_the_clone_weight_measure(self, model):
        """The identity the compartment shares depend on: Phi(S) = sum_c pi_c Phi(c)."""
        df = repertoire(200, seed=4)
        w = B.weights(df["duplicate_count"].to_numpy())
        phi, _ = B.prototype_sum(df, model, w)
        half = df.height // 2
        parts = []
        for lo, hi in ((0, half), (half, df.height)):
            sub, ws = df.slice(lo, hi - lo), w[lo:hi]
            p, _ = B.prototype_sum(sub, model, ws / ws.sum())
            parts.append(ws.sum() * p)
        assert np.allclose(phi, sum(parts))


class TestSlots:
    def test_slots_are_exact_strides(self, model):
        df = repertoire()
        w = B.weights(df["duplicate_count"].to_numpy())
        phi, _ = B.prototype_sum(df, model, w)
        s = B.slots(phi)
        assert np.allclose(s["phiv"], phi[0::3])
        assert np.allclose(s["phij"], phi[1::3])
        assert np.allclose(s["phic"], phi[2::3])

    def test_slots_partition_the_coordinates(self, model):
        phi = np.arange(3 * model.n_features // 3, dtype=float)
        s = B.slots(phi)
        assert sum(v.size for v in s.values()) == phi.size


class TestShares:
    def test_compartment_shares_close(self, model):
        df = repertoire(400, seed=2)
        w = B.weights(df["duplicate_count"].to_numpy())
        phi, _ = B.prototype_sum(df, model, w)
        shares = B.band_shares(df, w, min_clonotypes=1)
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_a_small_compartment_is_absent_not_zero(self, model):
        """Zero is a measurement; absent is not."""
        df = repertoire(400, seed=2)
        w = B.weights(df["duplicate_count"].to_numpy())
        phi, _ = B.prototype_sum(df, model, w)
        shares = B.band_shares(df, w, min_clonotypes=10_000)
        assert set(shares) == {"_residual"}
        assert shares["_residual"] == pytest.approx(1.0)

    def test_isotype_shares_need_a_c_call(self):
        df = repertoire(60, locus="IGH")
        w = B.weights(df["duplicate_count"].to_numpy())
        assert B.isotype_shares(df.drop("c_call"), w) == {}

    def test_isotype_shares_close_with_an_uncalled_part(self):
        n = 60
        df = repertoire(n, locus="IGH").with_columns(
            pl.Series("c_call", ["IGHM"] * 20 + ["IGHG1"] * 20 + [None] * (n - 40)))
        w = B.weights(df["duplicate_count"].to_numpy())
        out = B.isotype_shares(df, w)
        assert sum(out.values()) == pytest.approx(1.0)
        assert out["_uncalled"] > 0


class TestContrast:
    """Psi is a method of the reference, not a free function.

    It needs the frozen ``naive`` to subtract, so it belongs to the object that carries one. A
    second copy taking ``naive`` as an argument used to live in ``blocks``, called by nothing but
    its own self-check -- two spellings of one formula, only one of which can be kept honest
    against the artifact.
    """

    @staticmethod
    def _ref(naive):
        return R.LocusReference(
            mu_phi=np.zeros_like(naive), sd_phi=np.ones_like(naive), naive=np.asarray(naive,
            dtype=float), naive_sem=np.zeros_like(naive), rotations={}, prototype_hash="test")

    def test_an_unobserved_repertoire_lands_at_the_origin(self):
        """Not at minus-the-median: that is the whole reason the block is magnitude-scaled."""
        phi = np.arange(12, dtype=float)
        assert np.allclose(self._ref(np.zeros(12)).contrast(phi, mass=0.0), 0.0)

    def test_a_repertoire_matching_the_naive_reference_is_at_the_origin(self):
        phi = np.arange(12, dtype=float)
        assert np.allclose(self._ref(phi).contrast(phi, mass=1.0), 0.0)

    def test_magnitude_scales_with_retained_mass(self):
        phi, ref = np.ones(12), self._ref(np.zeros(12))
        assert np.linalg.norm(ref.contrast(phi, 1.0)) == pytest.approx(
            2 * np.linalg.norm(ref.contrast(phi, 0.5)))


class TestCentring:
    def test_centring_is_what_makes_phi_discriminative(self, model):
        """The measured reason mu_phi is frozen into the artifact rather than skipped.

        Every prototype distance is large and positive, so all repertoires sit in nearly the
        same place. On real donors the raw between-donor cosine spans about 0.001; centred, it
        spans well over 1. If this ever stops holding, the identity block has quietly become a
        constant and the rotation is fitting noise.
        """
        P = []
        for seed in range(8):
            df = repertoire(400, seed=seed)
            w = B.weights(df["duplicate_count"].to_numpy())
            phi, _ = B.prototype_sum(df, model, w)
            P.append(phi)
        P = np.array(P)

        def spread(X):
            Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
            c = Xn @ Xn.T
            iu = np.triu_indices(len(X), 1)
            return c[iu].max() - c[iu].min()

        raw, centred = spread(P), spread(P - P.mean(0))
        assert raw < 0.05, "raw Phi was unexpectedly discriminative; recheck the premise"
        assert centred > 20 * raw, f"centring bought only {centred / raw:.1f}x"

    def test_the_shared_offset_dominates_the_signal(self, model):
        P = []
        for seed in range(6):
            df = repertoire(400, seed=seed)
            w = B.weights(df["duplicate_count"].to_numpy())
            phi, _ = B.prototype_sum(df, model, w)
            P.append(phi)
        P = np.array(P)
        offset = np.linalg.norm(P.mean(0))
        signal = np.linalg.norm(P - P.mean(0), axis=1).mean()
        assert offset > 10 * signal


class TestDepthBlock:
    def test_n_eff_tracks_effective_not_nominal_size(self, model):
        even = np.ones(100)
        skewed = np.array([10_000.0] + [1.0] * 99)
        n_even = 1 / (B.weights(even) @ B.weights(even))
        n_skew = 1 / (B.weights(skewed) @ B.weights(skewed))
        assert n_even > n_skew

    def test_mass_is_denominator_aware(self):
        """A retained mass of 0.9 means less on ten clonotypes than on a thousand.

        Measured away from 0.5 on purpose: the correction shrinks *toward* the midpoint, so at
        exactly 0.5 there is nothing to shrink and every denominator gives 0.
        """
        shallow = B.depth_block(np.ones(10), B.weights(np.ones(10)), 0.9)["mass"]
        deep = B.depth_block(np.ones(1000), B.weights(np.ones(1000)), 0.9)["mass"]
        assert shallow < deep, "mass ignored how many clonotypes it was estimated from"

    def test_mass_at_the_midpoint_is_zero_for_any_depth(self):
        for n in (10, 1000):
            assert B.depth_block(np.ones(n), B.weights(np.ones(n)), 0.5)["mass"] == 0.0


class TestPortability:
    def test_the_signature_computes_without_scikit_learn(self, monkeypatch):
        """``mir.signature`` must not need the ML extra.

        The signature is the one part of the library meant to run anywhere from a plain install,
        and the import that broke this sat two modules away: ``assemble`` reaches ``missing_mass``
        in ``mir.repertoire``, which imported ``mir.density``, which imported sklearn at module
        scope. Nothing failed until a sample was actually assembled, so a cluster run computed
        every block for 4,000 samples and then emitted zero rows. Blocking the import is the only
        check that would have caught it -- sklearn is installed in every dev environment.
        """
        import builtins

        real = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "sklearn" or name.startswith("sklearn."):
                raise ModuleNotFoundError("No module named 'sklearn'")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)

        from mir.signature import signature

        rng = np.random.default_rng(0)
        aa = ["CASS" + "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 8)) + "F"
              for _ in range(60)]
        df = pl.DataFrame({"junction_aa": aa, "v_call": ["TRBV5-1*01"] * 60,
                           "j_call": ["TRBJ2-7*01"] * 60,
                           "duplicate_count": rng.integers(1, 50, 60)})
        v = signature({"TRB": df}, tier="full", standardize="none", threads=1)
        assert any(np.isfinite(x) for x in v.values() if isinstance(x, float))


class TestMissingMassIsNotSwallowed:
    """A frequency column must raise, not silently become "we observed everything".

    ``missing_mass`` raises a ValueError when handed values in (0, 1] that are not whole numbers,
    specifically so a shallow sample is not declared a complete probability measure. The assembler
    used to catch it and substitute 0.0 -- mass = 1.0 -- reaching the exact outcome the guard
    exists to prevent, on every sample, with the contrast block then scaled at full magnitude.
    """

    @staticmethod
    def _sample(counts):
        n = len(counts)
        r = np.random.default_rng(3)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        return {"TRB": pl.DataFrame({
            "v_call": ["TRBV20-1"] * n, "j_call": ["TRBJ2-2"] * n, "c_call": [None] * n,
            "junction_aa": ["C" + "".join(r.choice(aa, 12)) + "F" for _ in range(n)],
            "duplicate_count": list(counts)}, schema_overrides={"c_call": pl.Utf8})}

    def test_a_frequency_column_raises(self):
        from mir.signature import rsig

        freqs = list(np.full(200, 1.0 / 200))
        with pytest.raises(ValueError, match="integer clonotype counts"):
            rsig(self._sample(freqs), tier="core")

    def test_honest_counts_still_compute(self):
        from mir.signature import rsig

        out = rsig(self._sample(np.random.default_rng(0).integers(1, 60, 200).tolist()),
                   tier="core")
        assert np.isfinite(out["rsig:depth:TRB:mass"])
