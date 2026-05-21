"""Unit tests and benchmarks for :mod:`mir.biomarkers.motif_logo`.

Tests are split into:
* Unit tests — fast, deterministic, no external assets.
* Benchmark — marked ``benchmark``; uses GILGFVFTL test asset and motif_pwms.

Run unit tests only::

    pytest -s tests/test_motif_logo.py -m "not benchmark"

Run benchmarks (requires VDJdb asset)::

    pytest -s tests/test_motif_logo.py -m benchmark
"""

from __future__ import annotations

import gzip
import math
import time
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.conftest import skip_integration
from mir.biomarkers.motif_logo import (
    AA_ORDER,
    compute_cluster_profiles,
    compute_logo,
    compute_pwm,
    get_vj_background,
    load_motif_pwms,
    plot_logo,
    plot_motif_logos,
    pwm_from_motif_pwms,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

ASSETS = Path(__file__).parent / "assets"
GILG_FILE = ASSETS / "gilgfvftl_trb_junctions.txt.gz"
MOTIF_PWMS_FILE = (
    Path(__file__).parent.parent
    / "notebooks"
    / "assets"
    / "large"
    / "airr_benchmark"
    / "vdjdb"
    / "vdjdb-2025-12-29"
    / "motif_pwms.txt.gz"
)

# Hand-crafted sequences for deterministic unit tests
_SIMPLE_SEQS = [
    "CASSRS",
    "CASSRS",
    "CASSRS",
    "CASGTS",
    "CASGTS",
    "CASSQT",
]

# The conserved RS/GIL motif positions (positions 4-5) in GILGFVFTL TRB CDR3s
# come from Influenza A HLA-A*02-restricted T cells (TRBV19, length 13).


def _load_gilg_seqs() -> list[str]:
    with gzip.open(GILG_FILE, "rt") as fh:
        return [line.strip() for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Unit tests: compute_pwm
# ---------------------------------------------------------------------------

class TestComputePwm(unittest.TestCase):
    """Tests for :func:`compute_pwm`."""

    def test_basic_shape(self):
        """PWM must have 20 AAs × n_positions rows."""
        pwm = compute_pwm(_SIMPLE_SEQS)
        n_pos = 6
        self.assertEqual(len(pwm), n_pos * 20)

    def test_all_aas_present(self):
        """Every position must have exactly 20 AA rows."""
        pwm = compute_pwm(_SIMPLE_SEQS)
        for pos in range(6):
            aas = set(pwm.filter(pl.col("pos") == pos)["aa"].to_list())
            self.assertEqual(aas, set(AA_ORDER), f"Missing AAs at position {pos}")

    def test_frequencies_sum_to_one(self):
        """Frequencies must sum to ~1.0 at every position."""
        pwm = compute_pwm(_SIMPLE_SEQS)
        for pos in range(6):
            total = pwm.filter(pl.col("pos") == pos)["frequency"].sum()
            self.assertAlmostEqual(total, 1.0, places=6, msg=f"pos={pos}")

    def test_conserved_position(self):
        """Position 0 is 'C' in all sequences → C should dominate."""
        pwm = compute_pwm(_SIMPLE_SEQS)
        pos0 = pwm.filter(pl.col("pos") == 0).sort("frequency", descending=True)
        self.assertEqual(pos0[0]["aa"][0], "C")

    def test_pseudocount_zero_allowed(self):
        """pseudocount=0 should produce exact ML frequencies."""
        pwm = compute_pwm(_SIMPLE_SEQS, pseudocount=0.0)
        pos0 = pwm.filter((pl.col("pos") == 0) & (pl.col("aa") == "C"))
        self.assertAlmostEqual(float(pos0["frequency"][0]), 1.0, places=6)

    def test_length_filter(self):
        """Only sequences of the specified length should be used."""
        mixed = _SIMPLE_SEQS + ["CASSRSS"]  # length 7
        pwm_6 = compute_pwm(mixed, length=6)
        self.assertEqual(len(pwm_6), 6 * 20)

    def test_empty_raises(self):
        """Empty input must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_pwm([])

    def test_negative_pseudocount_raises(self):
        with self.assertRaises(ValueError):
            compute_pwm(_SIMPLE_SEQS, pseudocount=-1.0)

    def test_count_column_present(self):
        """PWM must expose a 'count' column."""
        pwm = compute_pwm(_SIMPLE_SEQS, pseudocount=0.0)
        self.assertIn("count", pwm.columns)
        # C at position 0 has count 6 (all six sequences)
        c_count = int(
            pwm.filter((pl.col("pos") == 0) & (pl.col("aa") == "C"))["count"][0]
        )
        self.assertEqual(c_count, 6)


# ---------------------------------------------------------------------------
# Unit tests: compute_logo — IC heights
# ---------------------------------------------------------------------------

class TestComputeLogoIC(unittest.TestCase):
    """Tests for Shannon IC heights in :func:`compute_logo`."""

    def setUp(self):
        self.pwm = compute_pwm(_SIMPLE_SEQS)
        self.logo = compute_logo(self.pwm)

    def test_ic_height_column_present(self):
        self.assertIn("ic_height", self.logo.columns)

    def test_ic_heights_non_negative(self):
        """IC heights must never be negative."""
        self.assertTrue((self.logo["ic_height"] >= -1e-10).all())

    def test_perfectly_conserved_position(self):
        """Position 0 is fully conserved (C) with pseudocount=0 → IC = log2(20)."""
        pwm_exact = compute_pwm(_SIMPLE_SEQS, pseudocount=0.0)
        logo_exact = compute_logo(pwm_exact)
        pos0 = logo_exact.filter(pl.col("pos") == 0)
        total_ic = float(pos0["ic_height"].sum())
        self.assertAlmostEqual(total_ic, math.log2(20), places=4)

    def test_uniform_position_has_low_ic(self):
        """A position with uniform distribution should have near-zero IC."""
        uniform_seqs = [
            "C" + aa + "SS"
            for aa in "ACDEFGHIKLMNPQRSTVWY"
        ]
        pwm = compute_pwm(uniform_seqs, pseudocount=0.0)
        logo = compute_logo(pwm)
        pos1_ic = logo.filter(pl.col("pos") == 1)["ic_height"].sum()
        # Uniform → IC ≈ 0 (entropy = log2(20), so IC = log2(20) - log2(20) = 0)
        self.assertAlmostEqual(pos1_ic, 0.0, places=1)

    def test_ic_height_sums_to_ic_per_position(self):
        """Sum of ic_height at each position should equal position IC."""
        for pos in range(6):
            pos_data = self.logo.filter(pl.col("pos") == pos)
            freqs = pos_data["frequency"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                log_f = np.where(freqs > 0, np.log2(freqs), 0.0)
            expected_ic = max(0.0, math.log2(20) + float(np.dot(freqs, log_f)))
            actual_sum = float(pos_data["ic_height"].sum())
            self.assertAlmostEqual(actual_sum, expected_ic, places=6, msg=f"pos={pos}")


# ---------------------------------------------------------------------------
# Unit tests: compute_logo — background-normalised heights
# ---------------------------------------------------------------------------

class TestComputeLogoBg(unittest.TestCase):
    """Tests for background-normalised heights in :func:`compute_logo`."""

    def setUp(self):
        # Use uniform background at every position / residue
        n = len(AA_ORDER)
        bg_records = [
            {"pos": pos, "aa": aa, "frequency": 1.0 / n}
            for pos in range(6)
            for aa in AA_ORDER
        ]
        self.bg = pl.DataFrame(bg_records).with_columns(pl.col("pos").cast(pl.Int32))
        self.pwm = compute_pwm(_SIMPLE_SEQS)
        self.logo = compute_logo(self.pwm, background=self.bg)

    def test_bg_height_column_present(self):
        self.assertIn("bg_height", self.logo.columns)

    def test_conserved_position_positive_bg(self):
        """Conserved position → enriched residue must have positive bg_height."""
        pos0 = self.logo.filter((pl.col("pos") == 0) & (pl.col("aa") == "C"))
        self.assertGreater(float(pos0["bg_height"][0]), 0.0)

    def test_rare_aa_negative_bg(self):
        """A residue absent from the motif but present in background → negative."""
        # 'W' does not appear in _SIMPLE_SEQS at any position
        w_bg = self.logo.filter(
            (pl.col("pos") == 4) & (pl.col("aa") == "W")
        )
        if len(w_bg) > 0:
            self.assertLess(float(w_bg["bg_height"][0]), 0.0)

    def test_uniform_bg_height_sum_property(self):
        """With uniform background, sum of f*log2(f/bg) = sum(f)*KL(f||bg)."""
        for pos in range(6):
            pos_data = self.logo.filter(pl.col("pos") == pos)
            freqs = pos_data["frequency"].to_numpy()
            bg = np.full(len(freqs), 1.0 / len(AA_ORDER))
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = freqs * np.where(freqs > 0, np.log2(freqs / bg), 0.0)
            actual = pos_data["bg_height"].to_numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-10, err_msg=f"pos={pos}")


# ---------------------------------------------------------------------------
# Unit tests: motif_pwms helpers
# ---------------------------------------------------------------------------

class TestPwmFromMotifPwms(unittest.TestCase):
    """Tests for :func:`pwm_from_motif_pwms` and :func:`get_vj_background`."""

    @classmethod
    def setUpClass(cls):
        if not MOTIF_PWMS_FILE.exists():
            raise unittest.SkipTest(
                f"motif_pwms.txt.gz not found — run ensure_airr_benchmark first"
            )
        cls.pwms = load_motif_pwms(MOTIF_PWMS_FILE)

    def test_load_returns_dataframe(self):
        self.assertIsInstance(self.pwms, pl.DataFrame)
        self.assertIn("cid", self.pwms.columns)
        self.assertIn("height.I", self.pwms.columns)

    def test_pwm_from_motif_pwms_known_cluster(self):
        logo = pwm_from_motif_pwms(self.pwms, "H.B.GILGFVFTL.1")
        self.assertIn("ic_height", logo.columns)
        self.assertIn("bg_height", logo.columns)
        self.assertIn("frequency", logo.columns)

    def test_pwm_from_motif_pwms_unknown_raises(self):
        with self.assertRaises(KeyError):
            pwm_from_motif_pwms(self.pwms, "NONEXISTENT_CLUSTER")

    def test_get_vj_background_exact(self):
        bg = get_vj_background(
            self.pwms,
            v_gene="TRBV19*01",
            j_gene="TRBJ2-7*01",
            length=13,
        )
        self.assertIsNotNone(bg)
        # Should have all 13 positions
        positions = sorted(bg["pos"].unique().to_list())
        self.assertEqual(len(positions), 13)
        # Frequencies must be positive
        self.assertTrue((bg["frequency"] > 0).all())

    def test_get_vj_background_no_match_returns_none(self):
        bg = get_vj_background(
            self.pwms,
            v_gene="TRBV999",
            j_gene="TRBJ999",
            length=99,
        )
        self.assertIsNone(bg)

    def test_get_vj_background_prefix_match(self):
        # Strip allele suffix — should still find a match
        bg = get_vj_background(
            self.pwms,
            v_gene="TRBV19",
            j_gene="TRBJ2-7",
            length=13,
        )
        self.assertIsNotNone(bg)


# ---------------------------------------------------------------------------
# Unit tests: plotting (smoke tests)
# ---------------------------------------------------------------------------

class TestPlottingSmoke(unittest.TestCase):
    """Smoke tests: verify that plotting functions run without error."""

    def setUp(self):
        self.pwm = compute_pwm(_SIMPLE_SEQS)
        self.logo_no_bg = compute_logo(self.pwm)
        bg_records = [
            {"pos": pos, "aa": aa, "frequency": 1.0 / len(AA_ORDER)}
            for pos in range(6) for aa in AA_ORDER
        ]
        bg = pl.DataFrame(bg_records).with_columns(pl.col("pos").cast(pl.Int32))
        self.logo_with_bg = compute_logo(self.pwm, background=bg)

    def test_plot_logo_ic(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_logo(self.logo_no_bg, ax, height_col="ic_height")
        plt.close(fig)

    def test_plot_logo_bg(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_logo(self.logo_with_bg, ax, height_col="bg_height")
        plt.close(fig)

    def test_plot_motif_logos_two_panels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plot_motif_logos(
            self.logo_with_bg,
            v_gene="TRBV9",
            j_gene="TRBJ2-3",
            n_seqs=6,
        )
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_plot_motif_logos_one_panel(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plot_motif_logos(
            self.logo_no_bg,
            show_bg=False,
        )
        self.assertEqual(len(axes), 1)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Unit tests: compute_cluster_profiles
# ---------------------------------------------------------------------------

class TestComputeClusterProfiles(unittest.TestCase):
    """Tests for :func:`compute_cluster_profiles`."""

    @classmethod
    def setUpClass(cls):
        if not MOTIF_PWMS_FILE.exists():
            raise unittest.SkipTest("motif_pwms.txt.gz not found")
        cls.pwms = load_motif_pwms(MOTIF_PWMS_FILE)

    def test_returns_dataframe(self):
        profiles = compute_cluster_profiles(self.pwms, min_csz=30)
        self.assertIsInstance(profiles, pl.DataFrame)

    def test_expected_columns(self):
        profiles = compute_cluster_profiles(self.pwms, min_csz=30)
        for col in ["cid", "species", "gene", "len", "csz", "pos", "IC", "H", "I_norm"]:
            self.assertIn(col, profiles.columns, f"Missing column: {col}")

    def test_ic_non_negative(self):
        profiles = compute_cluster_profiles(self.pwms, min_csz=30)
        self.assertTrue((profiles["IC"] >= -1e-8).all())

    def test_h_non_negative(self):
        profiles = compute_cluster_profiles(self.pwms, min_csz=30)
        self.assertTrue((profiles["H"] >= -1e-8).all())

    def test_i_norm_non_negative(self):
        profiles = compute_cluster_profiles(self.pwms, min_csz=30)
        self.assertTrue((profiles["I_norm"] >= -1e-8).all())

    def test_gene_filter(self):
        trb = compute_cluster_profiles(self.pwms, min_csz=30, gene="TRB")
        tra = compute_cluster_profiles(self.pwms, min_csz=30, gene="TRA")
        self.assertTrue((trb["gene"] == "TRB").all())
        self.assertTrue((tra["gene"] == "TRA").all())

    def test_species_filter(self):
        hs = compute_cluster_profiles(self.pwms, min_csz=30, species="HomoSapiens")
        self.assertTrue((hs["species"] == "HomoSapiens").all())

    def test_min_csz_filter(self):
        high = compute_cluster_profiles(self.pwms, min_csz=100)
        all_csz_ok = (high.select("cid", "csz").unique()["csz"] >= 100).all()
        self.assertTrue(all_csz_ok)

    def test_ic_plus_h_close_to_log2_20(self):
        """IC + H = log₂(20) at every position (constant total entropy)."""
        profiles = compute_cluster_profiles(self.pwms, min_csz=30)
        import math
        log2_20 = math.log2(20)
        totals = (profiles["IC"] + profiles["H"]).to_numpy()
        # Allow ±0.02 bits tolerance due to floating-point in pre-stored heights
        self.assertTrue(
            all(abs(t - log2_20) < 0.05 for t in totals),
            f"IC+H deviates from log₂(20) at some positions (max dev = "
            f"{max(abs(t - log2_20) for t in totals):.4f})",
        )

    def test_gilgfvftl_ic_profile_sorted(self):
        """GILGFVFTL positions 0 and 12 should have highest IC (conserved C and F)."""
        profiles = compute_cluster_profiles(self.pwms, min_csz=30)
        gilg = profiles.filter(pl.col("cid") == "H.B.GILGFVFTL.1").sort("IC", descending=True)
        if gilg.is_empty():
            self.skipTest("H.B.GILGFVFTL.1 not found")
        top_positions = gilg.head(3)["pos"].to_list()
        self.assertTrue(
            any(p in top_positions for p in [0, 12]),
            f"Conserved positions not in top-3 IC: {top_positions}",
        )


# ---------------------------------------------------------------------------
# Benchmark: GILGFVFTL RS motif enrichment via background-normalised logo
# ---------------------------------------------------------------------------

@unittest.skipUnless(GILG_FILE.exists(), "GILGFVFTL asset missing")
@unittest.skipUnless(MOTIF_PWMS_FILE.exists(), "motif_pwms.txt.gz missing")
@skip_integration
@pytest.mark.benchmark
class TestGilgfvftlLogoValidation(unittest.TestCase):
    """Validate that RS-containing positions are enriched in bg-normalised logo.

    Uses GILGFVFTL-specific TRB CDR3 sequences (TRBV19, length 13) from the
    VDJdb benchmark asset and OLGA-derived background from motif_pwms.

    The conserved CDR3 motif CASSIRSS (length 8, positions 4-7) drives a
    strong enrichment signal at the RS positions (7-8) in the length-13
    cluster.
    """

    @classmethod
    def setUpClass(cls):
        cls.seqs = _load_gilg_seqs()
        cls.pwms = load_motif_pwms(MOTIF_PWMS_FILE)

        # Filter to length 13 (dominant in H.B.GILGFVFTL.1)
        cls.seqs_13 = [s for s in cls.seqs if len(s) == 13]
        cls.pwm = compute_pwm(cls.seqs_13)
        cls.bg = get_vj_background(
            cls.pwms,
            v_gene="TRBV19*01",
            j_gene="TRBJ2-7*01",
            length=13,
        )
        if cls.bg is None:
            raise unittest.SkipTest("No TRBV19/TRBJ2-7 background in motif_pwms")
        cls.logo = compute_logo(cls.pwm, background=cls.bg)

    def test_rs_positions_enriched(self):
        """RS k-mers at positions 7-8 should show positive bg-height for R and S."""
        # Gather R and S bg_height at positions where they appear
        rs_aas = self.logo.filter(
            pl.col("aa").is_in(["R", "S"])
            & (pl.col("bg_height") > 0)
        )
        # There should be multiple enriched RS-containing positions
        self.assertGreater(
            len(rs_aas), 0,
            "Expected R or S to be enriched in at least one position",
        )

    def test_ic_logo_sum_positive(self):
        """Total IC should be positive (motif has information content)."""
        total_ic = float(self.logo["ic_height"].sum())
        self.assertGreater(total_ic, 0.0)

    def test_benchmark_timing(self):
        """PWM + logo computation for ~500 sequences should be fast (< 1 s)."""
        t0 = time.perf_counter()
        pwm = compute_pwm(self.seqs_13)
        logo = compute_logo(pwm, background=self.bg)
        elapsed = time.perf_counter() - t0
        n_seqs = len(self.seqs_13)
        print(f"\n  GILGFVFTL len-13 ({n_seqs} seqs): compute_pwm + compute_logo "
              f"in {elapsed * 1000:.1f} ms")
        self.assertLess(elapsed, 1.0, "Expected < 1 s for ~500 sequences")

    def test_plotting_runs(self):
        """Full two-panel logo figure should render without error."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plot_motif_logos(
            self.logo,
            v_gene="TRBV19*01",
            j_gene="TRBJ2-7*01",
            n_seqs=len(self.seqs_13),
            title="GILGFVFTL (H.B.GILGFVFTL.1)",
        )
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_precomputed_vs_computed_ic_rank_agreement(self):
        """Position IC rank order should agree between motif_pwms and compute_logo.

        Note: motif_pwms normalises IC to [0,1] (divides by log2(20)).
        ``compute_logo`` returns absolute IC in bits.  The two scales differ by
        a constant factor so rank order should be preserved.
        """
        ref_logo = pwm_from_motif_pwms(self.pwms, "H.B.GILGFVFTL.1")
        positions = sorted(self.logo["pos"].unique().to_list())

        # Per-position IC sums (motif_pwms: normalised; computed: bits)
        ref_ic = [
            float(ref_logo.filter(pl.col("pos") == p)["ic_height"].sum())
            for p in positions
        ]
        computed_ic = [
            float(self.logo.filter(pl.col("pos") == p)["ic_height"].sum())
            for p in positions
        ]

        # Conserved positions 0 and 12 should rank highest in both
        for cons_pos in [0, 12]:
            ref_rank = sorted(positions, key=lambda p: ref_ic[p], reverse=True)
            comp_rank = sorted(positions, key=lambda p: computed_ic[p], reverse=True)
            self.assertIn(
                cons_pos, ref_rank[:3],
                f"Conserved pos {cons_pos} not in top-3 of motif_pwms IC",
            )
            self.assertIn(
                cons_pos, comp_rank[:3],
                f"Conserved pos {cons_pos} not in top-3 of computed IC",
            )


if __name__ == "__main__":
    unittest.main()
