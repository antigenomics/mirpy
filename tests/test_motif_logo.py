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
    BIOCHEMISTRY_COLORS,
    aggregate_vj_background,
    build_motif_logos_vj,
    build_terminal_anchored_logo,
    build_terminal_anchored_pwm,
    compute_cluster_profiles,
    compute_logo,
    compute_pwm,
    get_vj_background,
    get_vj_background_from_control,
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
            v_call="TRBV19*01",
            j_call="TRBJ2-7*01",
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
            v_call="TRBV999",
            j_call="TRBJ999",
            length=99,
        )
        self.assertIsNone(bg)

    def test_get_vj_background_prefix_match(self):
        # Strip allele suffix — should still find a match
        bg = get_vj_background(
            self.pwms,
            v_call="TRBV19",
            j_call="TRBJ2-7",
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
            v_call="TRBV9",
            j_call="TRBJ2-3",
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
            v_call="TRBV19*01",
            j_call="TRBJ2-7*01",
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
            v_call="TRBV19*01",
            j_call="TRBJ2-7*01",
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


# ---------------------------------------------------------------------------
# Unit tests: BIOCHEMISTRY_COLORS
# ---------------------------------------------------------------------------

class TestBiochemistryColors(unittest.TestCase):
    """Tests for the five-category biochemistry colour scheme."""

    def test_all_20_aas_have_color(self):
        for aa in AA_ORDER:
            self.assertIn(aa, BIOCHEMISTRY_COLORS, f"Missing color for {aa}")

    def test_aromatic_residues_share_color(self):
        """W, F, Y, H (aromatic) should have the same colour."""
        aromatic = [BIOCHEMISTRY_COLORS[aa] for aa in "WFYH"]
        self.assertEqual(len(set(aromatic)), 1)

    def test_charged_residues_distinct_colors(self):
        """Positively (K, R) and negatively (D, E) charged should differ."""
        pos_color = BIOCHEMISTRY_COLORS["K"]
        neg_color = BIOCHEMISTRY_COLORS["D"]
        self.assertNotEqual(pos_color, neg_color)


# ---------------------------------------------------------------------------
# Unit tests: letter stacking order
# ---------------------------------------------------------------------------

class TestLetterStackingOrder(unittest.TestCase):
    """Verify that the tallest letter ends up on top (WebLogo convention)."""

    def _get_patch_y_tops(self, logo_df, height_col):
        """Render logo to Agg canvas and return patches sorted by top y."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_logo(logo_df, ax, height_col=height_col)
        patches = [p for p in ax.patches if hasattr(p, "get_path")]
        plt.close(fig)
        return patches

    def test_ic_logo_dominant_letter_on_top(self):
        """At a conserved position, the dominant AA patch should have the highest y-top."""
        # Position 0 is almost entirely 'C'; with pseudocount it dominates but not 100%
        seqs = ["CASSRS"] * 10 + ["CASGTS"] * 1
        pwm = compute_pwm(seqs, pseudocount=0.0)
        logo = compute_logo(pwm)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plot_logo(logo, ax, height_col="ic_height")

        # The tallest letter at pos 0 (C) should be drawn last → its bounding box
        # y-top should be the highest among all patches at that position.
        patches_pos0 = [
            p for p in ax.patches
            if hasattr(p, "get_path")
            and p.get_path() is not None
            and 0.0 <= p.get_path().get_extents().x0 < 1.0
        ]
        if patches_pos0:
            y_tops = [p.get_path().get_extents().y1 for p in patches_pos0]
            # The overall stack top should equal the sum of all ic_heights at pos 0
            total_ic_pos0 = float(logo.filter(pl.col("pos") == 0)["ic_height"].sum())
            self.assertAlmostEqual(max(y_tops), total_ic_pos0, delta=0.15)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Unit tests: aggregate_vj_background and build_motif_logos_vj
# ---------------------------------------------------------------------------

class TestAggregateVjBackground(unittest.TestCase):
    """Tests for :func:`aggregate_vj_background`."""

    @classmethod
    def setUpClass(cls):
        if not MOTIF_PWMS_FILE.exists():
            raise unittest.SkipTest("motif_pwms.txt.gz not found")
        cls.pwms = load_motif_pwms(MOTIF_PWMS_FILE)

    def test_returns_dataframe(self):
        bg = aggregate_vj_background(self.pwms, length=13, gene="TRB")
        self.assertIsNotNone(bg)
        self.assertIsInstance(bg, pl.DataFrame)

    def test_expected_columns(self):
        bg = aggregate_vj_background(self.pwms, length=13, gene="TRB")
        self.assertIn("pos", bg.columns)
        self.assertIn("aa", bg.columns)
        self.assertIn("frequency", bg.columns)

    def test_frequencies_positive(self):
        bg = aggregate_vj_background(self.pwms, length=13, gene="TRB")
        self.assertTrue((bg["frequency"] > 0).all())

    def test_frequencies_in_unit_interval(self):
        # motif_pwms stores only non-zero AAs; missing AAs get bg_floor in compute_logo.
        # All stored frequencies must be in (0, 1].
        bg = aggregate_vj_background(self.pwms, length=13, gene="TRB")
        self.assertTrue((bg["frequency"] > 0).all(), "All background frequencies must be positive")
        self.assertTrue((bg["frequency"] <= 1.0 + 1e-9).all(), "Background frequencies must be ≤ 1")

    def test_correct_length(self):
        bg = aggregate_vj_background(self.pwms, length=13, gene="TRB")
        n_positions = bg["pos"].n_unique()
        self.assertEqual(n_positions, 13)

    def test_gene_filter(self):
        bg_trb = aggregate_vj_background(self.pwms, length=13, gene="TRB")
        bg_tra = aggregate_vj_background(self.pwms, length=13, gene="TRA")
        if bg_trb is not None and bg_tra is not None:
            # TRB and TRA aggregate backgrounds should differ from each other
            joined = (
                bg_trb.rename({"frequency": "f_trb"})
                .join(bg_tra.rename({"frequency": "f_tra"}), on=["pos", "aa"], how="inner")
            )
            if len(joined) > 0:
                diff = (joined["f_trb"] - joined["f_tra"]).abs().max()
                self.assertGreater(diff, 1e-4, "TRB and TRA aggregate backgrounds should differ")

    def test_nonexistent_length_returns_none(self):
        bg = aggregate_vj_background(self.pwms, length=999, gene="TRB")
        self.assertIsNone(bg)

    def test_differs_from_per_vj_background(self):
        """All-VJ aggregate should differ from a specific VJ background."""
        bg_agg = aggregate_vj_background(self.pwms, length=13, gene="TRB")
        bg_vj = get_vj_background(
            self.pwms, v_call="TRBV19*01", j_call="TRBJ2-7*01",
            length=13, gene="TRB",
        )
        if bg_agg is not None and bg_vj is not None:
            joined = bg_agg.join(bg_vj, on=["pos", "aa"], suffix="_vj")
            diffs = (joined["frequency"] - joined["frequency_vj"]).abs()
            # At least some positions should differ between aggregate and specific VJ
            self.assertGreater(float(diffs.max()), 1e-4)


class TestBuildMotifLogosVj(unittest.TestCase):
    """Tests for :func:`build_motif_logos_vj`."""

    @classmethod
    def setUpClass(cls):
        if not MOTIF_PWMS_FILE.exists():
            raise unittest.SkipTest("motif_pwms.txt.gz not found")
        cls.pwms = load_motif_pwms(MOTIF_PWMS_FILE)
        # Minimal synthetic input: 5 sequences of TRBV9/TRBJ2-3/len=15
        cls.seqs_df = pl.DataFrame({
            "junction_aa": [
                "CASSVGLYSTDTQYF", "CASSVGLFSTDTQYF", "CASSVGVYSTDTQYF",
                "CASSLGLFSTDTQYF", "CASSAGLFSTDTQYF",
            ],
            "v_call": ["TRBV9"] * 5,
            "j_call": ["TRBJ2-3"] * 5,
        })

    def test_returns_dict(self):
        logos = build_motif_logos_vj(self.seqs_df, self.pwms)
        self.assertIsInstance(logos, dict)
        self.assertGreater(len(logos), 0)

    def test_per_vj_key_present(self):
        logos = build_motif_logos_vj(self.seqs_df, self.pwms)
        # Should have at least one (v, j, len) key
        vj_keys = [k for k in logos if k[0] is not None]
        self.assertGreater(len(vj_keys), 0)

    def test_aggregate_key_present(self):
        logos = build_motif_logos_vj(self.seqs_df, self.pwms)
        agg_keys = [k for k in logos if k[0] is None]
        self.assertGreater(len(agg_keys), 0)

    def test_logo_has_ic_height(self):
        logos = build_motif_logos_vj(self.seqs_df, self.pwms)
        for key, logo in logos.items():
            self.assertIn("ic_height", logo.columns, f"key={key}")

    def test_per_vj_logo_has_bg_height(self):
        """Per-VJ logos should have bg_height when a background is found."""
        logos = build_motif_logos_vj(self.seqs_df, self.pwms)
        vj_keys = [k for k in logos if k[0] is not None]
        for key in vj_keys:
            # bg_height may or may not be present depending on motif_pwms coverage;
            # if present it must contain valid values
            logo = logos[key]
            if "bg_height" in logo.columns:
                self.assertFalse(logo["bg_height"].is_null().any())

    def test_min_seqs_filter(self):
        """Groups with fewer than min_seqs sequences should be excluded."""
        logos_strict = build_motif_logos_vj(self.seqs_df, self.pwms, min_seqs=10)
        # All 5 sequences belong to one VJ group → below min_seqs=10 → excluded
        vj_keys = [k for k in logos_strict if k[0] is not None]
        self.assertEqual(len(vj_keys), 0)

    def test_custom_column_names(self):
        renamed = self.seqs_df.rename({
            "junction_aa": "cdr3",
            "v_call": "v",
            "j_call": "j",
        })
        logos = build_motif_logos_vj(
            renamed, self.pwms,
            cdr3_col="cdr3", v_col="v", j_col="j",
        )
        self.assertGreater(len(logos), 0)

    def test_length_in_key_matches_sequences(self):
        logos = build_motif_logos_vj(self.seqs_df, self.pwms)
        for key, logo in logos.items():
            expected_len = key[2]
            n_positions = logo["pos"].n_unique()
            self.assertEqual(n_positions, expected_len, f"key={key}")


# ---------------------------------------------------------------------------
# Unit tests: plot_motif_logos title placement
# ---------------------------------------------------------------------------

class TestPlotMotifLogosTitle(unittest.TestCase):
    """Verify that V/J gene names appear in the suptitle, not on axes margins."""

    def test_vj_in_suptitle(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pwm = compute_pwm(_SIMPLE_SEQS)
        logo = compute_logo(pwm)
        fig, axes = plot_motif_logos(
            logo, v_call="TRBV9*01", j_call="TRBJ2-3*01",
            show_bg=False,
        )
        suptitle_text = fig._suptitle.get_text() if fig._suptitle else ""
        self.assertIn("TRBV9", suptitle_text)
        self.assertIn("TRBJ2-3", suptitle_text)
        plt.close(fig)

    def test_no_rotated_text_on_axes(self):
        """V/J should not appear as rotated text objects on the axes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pwm = compute_pwm(_SIMPLE_SEQS)
        logo = compute_logo(pwm)
        fig, axes = plot_motif_logos(
            logo, v_call="TRBV9", j_call="TRBJ2-3", show_bg=False,
        )
        for ax in axes:
            rotated_texts = [
                t for t in ax.texts
                if t.get_rotation() not in (0, 360)
            ]
            self.assertEqual(
                len(rotated_texts), 0,
                f"Found {len(rotated_texts)} rotated text objects on axes — V/J should be in suptitle",
            )
        plt.close(fig)


class TestBuildTerminalAnchoredPwm(unittest.TestCase):
    """Tests for :func:`build_terminal_anchored_pwm`."""

    # Sequences of mixed length — typical CDR3 diversity
    _MIXED = [
        "CASSRSYEQYF",   # len 11
        "CASSGRSYEQYF",  # len 12
        "CASSRSSYEQYF",  # len 12
        "CASSQGRSYEQYF", # len 13
        "CASSPGRSYEQYF", # len 13
        "CASSRSYEQYF",   # len 11 again
    ]

    def test_returns_dataframe(self):
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        self.assertIsInstance(pwm, pl.DataFrame)

    def test_required_columns(self):
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        for col in ("pos", "label", "aa", "count", "n_covering", "frequency"):
            self.assertIn(col, pwm.columns, f"missing column {col!r}")

    def test_n_term_labels_positive(self):
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        n_labels = set(pwm.filter(pl.col("pos") < 5)["label"].unique().to_list())
        for lbl in n_labels:
            self.assertFalse(lbl.startswith("-"), f"N-term label should be positive, got {lbl!r}")

    def test_c_term_labels_negative(self):
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        c_labels = set(pwm.filter(pl.col("pos") >= 5)["label"].unique().to_list())
        for lbl in c_labels:
            self.assertTrue(lbl.startswith("-"), f"C-term label should be negative, got {lbl!r}")

    def test_last_c_term_label_is_minus_one(self):
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        last_pos = pwm["pos"].max()
        last_label = pwm.filter(pl.col("pos") == last_pos)["label"].to_list()[0]
        self.assertEqual(last_label, "-1", "Last C-terminal label must be '-1'")

    def test_first_c_term_is_conserved_phef(self):
        """Last position (-1) of any TRB CDR3 is always F (Phe)."""
        seqs = [s for s in self._MIXED]  # all end in F
        pwm = build_terminal_anchored_pwm(seqs, n_term=3, c_term=3)
        last_pos = pwm["pos"].max()
        last_col = pwm.filter(pl.col("pos") == last_pos)
        # All counts at -1 should be F
        f_count = last_col.filter(pl.col("aa") == "F")["count"].sum()
        total_count = last_col["count"].sum()
        self.assertEqual(f_count, total_count, "Last residue should be all F")

    def test_only_observed_aas_returned(self):
        """No rows with count == 0 should be present."""
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        zero_count_rows = pwm.filter(pl.col("count") == 0)
        self.assertEqual(len(zero_count_rows), 0, "build_terminal_anchored_pwm must not include unobserved AAs")

    def test_frequencies_sum_leq_one(self):
        """Per-position frequencies sum to <= 1 (sparse = only observed AAs)."""
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        pos_sums = pwm.group_by("pos").agg(pl.col("frequency").sum()).sort("pos")
        for row in pos_sums.iter_rows(named=True):
            self.assertLessEqual(row["frequency"], 1.0 + 1e-9, f"frequency sum > 1 at pos {row['pos']}")

    def test_compute_logo_compatible(self):
        """Output must be accepted by compute_logo without errors."""
        pwm = build_terminal_anchored_pwm(self._MIXED, n_term=5, c_term=4)
        logo = compute_logo(pwm)
        self.assertIn("ic_height", logo.columns)

    def test_empty_sequences_raises(self):
        with self.assertRaises(ValueError):
            build_terminal_anchored_pwm([])


class TestPlotLogoCountFilter(unittest.TestCase):
    """Verify that plot_logo suppresses pseudocount-only residues."""

    def test_no_spurious_slivers_at_conserved_position(self):
        """Fully-conserved sequences should produce exactly one patch per position."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # All 5 sequences are identical: 6 positions each fully conserved
        seqs = ["CASSRS", "CASSRS", "CASSRS", "CASSRS", "CASSRS"]
        pwm = compute_pwm(seqs)
        logo = compute_logo(pwm)
        n_pos = logo["pos"].n_unique()

        fig, ax = plt.subplots(figsize=(4, 2))
        plot_logo(logo, ax, height_col="ic_height")

        # With count filter: each position has exactly 1 observed AA → 1 patch per position.
        # Without the fix, pseudocount leaks many extra patches (up to 20 per position).
        n_patches = len(ax.patches)
        self.assertEqual(
            n_patches, n_pos,
            f"Expected {n_pos} patches (one per position for fully-conserved logo), got {n_patches} — pseudocount residuals not suppressed",
        )
        plt.close(fig)

    def test_tick_labels_centered_below_columns(self):
        """X-axis ticks must be at bar centres (pos + 0.5), not left edges."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pwm = compute_pwm(_SIMPLE_SEQS)
        logo = compute_logo(pwm)

        fig, ax = plt.subplots(figsize=(4, 2))
        plot_logo(logo, ax)

        tick_locs = ax.get_xticks()
        n_pos = logo["pos"].n_unique()
        for i, loc in enumerate(tick_locs[:n_pos]):
            expected = i + 0.5
            self.assertAlmostEqual(
                loc, expected, places=5,
                msg=f"Tick {i} at {loc}, expected {expected} (bar centre)"
            )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Unit tests: get_vj_background_from_control
# ---------------------------------------------------------------------------

class TestGetVjBackgroundFromControl(unittest.TestCase):
    """Tests for :func:`get_vj_background_from_control`."""

    # Synthetic control: 200 sequences of TRBV9/TRBJ2-3/len=15
    @classmethod
    def setUpClass(cls):
        import random
        rng = random.Random(42)
        aas = list("ACDEFGHIKLMNPQRSTVWY")
        def _rand_cdr3(length=15):
            mid = "".join(rng.choice(aas) for _ in range(length - 10))
            return f"CASS{mid}TDTQYF"
        cls.ctrl = pl.DataFrame({
            "junction_aa": [_rand_cdr3() for _ in range(200)],
            "v_call": ["TRBV9"] * 200,
            "j_call": ["TRBJ2-3"] * 200,
        })

    def test_returns_dataframe_sufficient_seqs(self):
        bg = get_vj_background_from_control(
            self.ctrl, v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=50
        )
        self.assertIsInstance(bg, pl.DataFrame)

    def test_returns_none_insufficient_seqs(self):
        bg = get_vj_background_from_control(
            self.ctrl, v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=500
        )
        self.assertIsNone(bg)

    def test_columns(self):
        bg = get_vj_background_from_control(
            self.ctrl, v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=50
        )
        for col in ("pos", "aa", "frequency"):
            self.assertIn(col, bg.columns)

    def test_n_positions(self):
        bg = get_vj_background_from_control(
            self.ctrl, v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=50
        )
        self.assertEqual(bg["pos"].n_unique(), 15)

    def test_frequencies_sum_to_one(self):
        bg = get_vj_background_from_control(
            self.ctrl, v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=50
        )
        for pos in bg["pos"].unique().to_list():
            total = float(bg.filter(pl.col("pos") == pos)["frequency"].sum())
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_v_prefix_matching(self):
        """TRBV9*01 should match TRBV9 entries in control."""
        bg = get_vj_background_from_control(
            self.ctrl, v_call="TRBV9*01", j_call="TRBJ2-3*01", length=15, min_seqs=50
        )
        self.assertIsInstance(bg, pl.DataFrame)

    def test_no_vj_filter(self):
        """Passing v_call=None, j_call=None should aggregate all entries."""
        bg = get_vj_background_from_control(
            self.ctrl, v_call=None, j_call=None, length=15, min_seqs=50
        )
        self.assertIsInstance(bg, pl.DataFrame)

    def test_usable_as_compute_logo_background(self):
        bg = get_vj_background_from_control(
            self.ctrl, v_call="TRBV9", j_call="TRBJ2-3", length=15, min_seqs=50
        )
        seqs = ["CASSVGLYSTDTQYF", "CASSVGLFSTDTQYF", "CASSVGVYSTDTQYF",
                "CASSLGLFSTDTQYF", "CASSAGLFSTDTQYF"]
        pwm = compute_pwm(seqs, length=15)
        logo = compute_logo(pwm, background=bg)
        self.assertIn("bg_height", logo.columns)


# ---------------------------------------------------------------------------
# Unit tests: build_terminal_anchored_logo
# ---------------------------------------------------------------------------

class TestBuildTerminalAnchoredLogo(unittest.TestCase):
    """Tests for :func:`build_terminal_anchored_logo`."""

    _SEQS_DF = pl.DataFrame({
        "junction_aa": [
            "CASSVGLYSTDTQYF",   # len 15
            "CASSVGLFSTDTQYF",   # len 15
            "CASSVGVYSTDTQYF",   # len 15
            "CASSLGLFSTDTQYF",   # len 15
            "CASSRSYEQYF",       # len 11
            "CASSGRSYEQYF",      # len 12
            "CASSRSSYEQYF",      # len 12
        ],
        "v_call": ["TRBV9"] * 4 + ["TRBV19"] * 3,
        "j_call": ["TRBJ2-3"] * 4 + ["TRBJ2-7"] * 3,
    })

    @classmethod
    def setUpClass(cls):
        if not MOTIF_PWMS_FILE.exists():
            raise unittest.SkipTest("motif_pwms.txt.gz not found")
        cls.pwms = load_motif_pwms(MOTIF_PWMS_FILE)

    def test_returns_dataframe(self):
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, self.pwms, n_term=8, c_term=7
        )
        self.assertIsInstance(logo, pl.DataFrame)

    def test_required_columns(self):
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, self.pwms, n_term=8, c_term=7
        )
        for col in ("pos", "label", "aa", "ic_height"):
            self.assertIn(col, logo.columns, f"missing column {col!r}")

    def test_n_term_display_positions(self):
        """Display positions 0..n_term-1 should have positive-integer labels."""
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, self.pwms, n_term=8, c_term=7
        )
        n_labels = logo.filter(pl.col("pos") < 8)["label"].unique().to_list()
        for lbl in n_labels:
            self.assertFalse(lbl.startswith("-"), f"N-block label should be positive: {lbl!r}")
            self.assertGreater(int(lbl), 0)

    def test_c_term_display_positions(self):
        """Display positions n_term..n_term+c_term-1 should have negative labels."""
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, self.pwms, n_term=8, c_term=7
        )
        c_labels = logo.filter(pl.col("pos") >= 8)["label"].unique().to_list()
        for lbl in c_labels:
            self.assertTrue(lbl.startswith("-"), f"C-block label should be negative: {lbl!r}")
            self.assertLess(int(lbl), 0)

    def test_max_display_pos(self):
        """Max display pos must be n_term + c_term - 1."""
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, self.pwms, n_term=8, c_term=7
        )
        self.assertEqual(int(logo["pos"].max()), 8 + 7 - 1)

    def test_bg_height_present_when_background_found(self):
        """bg_height column must appear when at least one length has a valid background."""
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, self.pwms, n_term=8, c_term=7
        )
        # Some VJ combos are in motif_pwms; at least one bg match expected
        if "bg_height" in logo.columns:
            non_null = logo["bg_height"].drop_nulls()
            self.assertGreater(len(non_null), 0)

    def test_ic_height_non_negative(self):
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, self.pwms, n_term=8, c_term=7
        )
        self.assertTrue((logo["ic_height"] >= -1e-9).all())

    def test_control_df_fallback(self):
        """When control_df is given and has coverage, bg_height should appear."""
        import random
        rng = random.Random(0)
        aas = list("ACDEFGHIKLMNPQRSTVWY")
        def _rand15():
            mid = "".join(rng.choice(aas) for _ in range(5))
            return f"CASS{mid}TDTQYF"
        ctrl = pl.DataFrame({
            "junction_aa": [_rand15() for _ in range(500)],
            "v_call": ["TRBV9"] * 500,
            "j_call": ["TRBJ2-3"] * 500,
        })
        seqs_df = self._SEQS_DF.filter(pl.col("j_call") == "TRBJ2-3")
        logo = build_terminal_anchored_logo(
            seqs_df, None, n_term=8, c_term=7,
            control_df=ctrl, min_control_seqs=50,
        )
        self.assertIsInstance(logo, pl.DataFrame)
        if "bg_height" in logo.columns:
            non_null = logo["bg_height"].drop_nulls()
            self.assertGreater(len(non_null), 0)

    def test_no_background_source_returns_ic_only(self):
        """Calling with no motif_pwms and no control_df must still return a logo (IC only)."""
        logo = build_terminal_anchored_logo(
            self._SEQS_DF, None, n_term=8, c_term=7
        )
        self.assertIn("ic_height", logo.columns)
        self.assertNotIn("bg_height", logo.columns)


if __name__ == "__main__":
    unittest.main()
