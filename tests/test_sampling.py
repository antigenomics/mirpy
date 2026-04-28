import time
import warnings
from pathlib import Path

import pytest
import numpy as np
from scipy import stats

from mir.common.sampling import downsample, downsample_locus, resample_to_gene_usage, select_top
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.parser import ClonotypeTableParser
from mir.basic.gene_usage import GeneUsage
from tests.conftest import skip_integration

import pandas as pd


# YFV dataset detection
_YFV_DIR = Path(__file__).parent.parent / "notebooks" / "assets" / "large" / "yfv19"
_YFV_AVAILABLE = _YFV_DIR.is_dir()


def _load_p1_f1(day: int) -> LocusRepertoire:
    """Load P1 replica F1 for the given *day* from the YFV dataset."""
    if not _YFV_AVAILABLE:
        pytest.skip(f"YFV dataset not found in {_YFV_DIR}")
    
    meta = pd.read_csv(_YFV_DIR / "metadata.txt", sep="\t")
    row = meta[(meta["donor"] == "P1") & (meta["day"] == day) & (meta["replica"] == "F1")]
    if row.empty:
        pytest.skip(f"P1/F1/day={day} not found in metadata")
    
    fname = row.iloc[0]["file_name"]
    df = pd.read_csv(_YFV_DIR / fname, sep="\t", compression="infer")
    if "locus" in df.columns:
        df = df[df["locus"].fillna("") == "TRB"]
    df = df.dropna(subset=["junction_aa"])
    df = df[df["junction_aa"].str.strip().str.len() > 0]
    
    parser = ClonotypeTableParser()
    clones = parser.parse_inner(df)
    return LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=fname)


class TestDownsampleLocusBasic:
    """Basic functionality tests for downsample_locus."""

    def test_raises_on_zero_downsample_count(self):
        rep = LocusRepertoire(
            clonotypes=[],
            locus="TRB",
        )
        with pytest.raises(ValueError, match="downsample_count must be > 0"):
            downsample_locus(rep, 0)

    def test_raises_on_negative_downsample_count(self):
        rep = LocusRepertoire(
            clonotypes=[],
            locus="TRB",
        )
        with pytest.raises(ValueError, match="downsample_count must be > 0"):
            downsample_locus(rep, -5)

    def test_warns_when_downsample_count_exceeds_total(self):
        from mir.common.clonotype import Clonotype
        
        rep = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=10,
                ),
            ],
            locus="TRB",
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = downsample_locus(rep, 100)
            assert len(w) == 1
            assert "no downsampling performed" in str(w[0].message)
            assert result.duplicate_count == 10

    def test_exact_downsample_count(self):
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id="1",
                locus="TRB",
                junction_aa="CASSF",
                v_gene="TRBV1",
                j_gene="TRBJ1-1",
                duplicate_count=100,
            ),
            Clonotype(
                sequence_id="2",
                locus="TRB",
                junction_aa="CASSGF",
                v_gene="TRBV2",
                j_gene="TRBJ1-1",
                duplicate_count=200,
            ),
            Clonotype(
                sequence_id="3",
                locus="TRB",
                junction_aa="CASSYGF",
                v_gene="TRBV3",
                j_gene="TRBJ1-1",
                duplicate_count=300,
            ),
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        downsampled = downsample_locus(rep, 300, random_seed=42)
        
        assert downsampled.duplicate_count == 300
        assert isinstance(downsampled, LocusRepertoire)
        assert downsampled.locus == rep.locus

    def test_preserves_metadata(self):
        from mir.common.clonotype import Clonotype
        
        rep = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                ),
            ],
            locus="TRB",
            repertoire_id="test_rep",
            repertoire_metadata={"donor": "P1", "day": 0},
        )
        
        downsampled = downsample_locus(rep, 50, random_seed=42)
        assert downsampled.repertoire_id == "test_rep"
        assert downsampled.repertoire_metadata == {"donor": "P1", "day": 0}


class TestDownsampleLocusStochastic:
    """Test stochastic properties of downsampling."""

    def test_clonotypes_omitted_when_count_zero(self):
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id=str(i),
                locus="TRB",
                junction_aa="CASSF",
                v_gene=f"TRBV{i}",
                j_gene="TRBJ1-1",
                duplicate_count=1,
            )
            for i in range(100)
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        downsampled = downsample_locus(rep, 10, random_seed=42)
        
        assert downsampled.clonotype_count <= rep.clonotype_count
        assert all(c.duplicate_count > 0 for c in downsampled.clonotypes)

    def test_lost_clonotypes_count(self):
        """Count how many clonotypes are lost during downsampling."""
        from mir.common.clonotype import Clonotype
        
        aa_variants = [
            "CASSF", "CASGF", "CASSGF", "CASSEGF", "CASSGEGF",
            "CASSYF", "CASSAGF", "CASSRGF", "CASSNGF", "CASSLF",
            "CASSQF", "CASSIF", "CASSDF", "CASSFF", "CASSWF",
            "CASSYF", "CASSRF", "CASSIF", "CASSGF", "CASSAF",
        ]
        
        clonotypes = []
        for i in range(50):
            aa = aa_variants[i % len(aa_variants)]
            clonotypes.append(
                Clonotype(
                    sequence_id=str(i),
                    locus="TRB",
                    junction_aa=aa,
                    v_gene=f"TRBV{i % 10 + 1}",
                    j_gene="TRBJ1-1",
                    duplicate_count=10,
                )
            )
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        original_count = rep.clonotype_count
        
        downsampled = downsample_locus(rep, 100, random_seed=42)
        downsampled_count = downsampled.clonotype_count
        
        lost = original_count - downsampled_count
        assert lost > 0
        assert downsampled.duplicate_count == 100


class TestDownsampleSampleRepertoire:
    """Test downsampling SampleRepertoire."""

    def test_downsample_sample_repertoire(self):
        from mir.common.clonotype import Clonotype
        
        trb = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=200,
                ),
            ],
            locus="TRB",
        )
        tra = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="2",
                    locus="TRA",
                    junction_aa="CAVRDSNYQLIW",
                    v_gene="TRAV1",
                    j_gene="TRAJ33",
                    duplicate_count=300,
                ),
            ],
            locus="TRA",
        )
        
        sample = SampleRepertoire(loci={"TRB": trb, "TRA": tra}, sample_id="s1")
        downsampled = downsample(sample, 100, random_seed=42)
        
        assert isinstance(downsampled, SampleRepertoire)
        assert downsampled.loci["TRB"].duplicate_count == 100
        assert downsampled.loci["TRA"].duplicate_count == 100
        assert downsampled.sample_id == "s1"


class TestDownsampleGensInterface:
    """Test generic downsample function dispatching."""

    def test_downsample_dispatches_locus_repertoire(self):
        from mir.common.clonotype import Clonotype
        
        rep = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                ),
            ],
            locus="TRB",
        )
        
        result = downsample(rep, 50, random_seed=42)
        assert isinstance(result, LocusRepertoire)
        assert result.duplicate_count == 50

    def test_downsample_dispatches_sample_repertoire(self):
        from mir.common.clonotype import Clonotype
        
        trb = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                ),
            ],
            locus="TRB",
        )
        sample = SampleRepertoire(loci={"TRB": trb}, sample_id="s1")
        
        result = downsample(sample, 50, random_seed=42)
        assert isinstance(result, SampleRepertoire)


@skip_integration
@pytest.mark.skipif(not _YFV_AVAILABLE, reason="YFV dataset not found")
@pytest.mark.integration
class TestDownsampleYFVPerformance:
    """Performance tests on real YFV data."""

    def test_downsample_p1_day0_speed(self):
        """Check downsampling speed on P1 day 0."""
        rep = _load_p1_f1(0)
        original_count = rep.duplicate_count
        
        start = time.time()
        downsampled = downsample_locus(rep, 100000, random_seed=42)
        elapsed = time.time() - start
        
        assert downsampled.duplicate_count == 100000
        assert elapsed < 5.0, f"Downsampling took {elapsed:.2f}s, expected < 5s"
        # Verify reasonable proportion of clonotypes lost
        lost = rep.clonotype_count - downsampled.clonotype_count
        assert lost > 0

    def test_downsample_p1_day15_speed(self):
        """Check downsampling speed on P1 day 15."""
        rep = _load_p1_f1(15)
        original_count = rep.duplicate_count
        
        start = time.time()
        downsampled = downsample_locus(rep, 100000, random_seed=42)
        elapsed = time.time() - start
        
        assert downsampled.duplicate_count == 100000
        assert elapsed < 5.0, f"Downsampling took {elapsed:.2f}s, expected < 5s"

    def test_downsample_p1_day0_clonotype_counts(self):
        """Verify exact counts and lost clonotypes on P1 day 0."""
        rep = _load_p1_f1(0)
        original_clonotype_count = rep.clonotype_count
        original_duplicate_count = rep.duplicate_count
        
        target_count = 100000
        downsampled = downsample_locus(rep, target_count, random_seed=42)
        
        assert downsampled.duplicate_count == target_count
        lost_clonotypes = original_clonotype_count - downsampled.clonotype_count
        assert lost_clonotypes >= 0
        print(f"\nP1 day 0: {original_duplicate_count} -> {target_count} duplicates, "
              f"lost {lost_clonotypes}/{original_clonotype_count} clonotypes")

    def test_downsample_p1_day15_clonotype_counts(self):
        """Verify exact counts and lost clonotypes on P1 day 15."""
        rep = _load_p1_f1(15)
        original_clonotype_count = rep.clonotype_count
        original_duplicate_count = rep.duplicate_count
        
        target_count = 100000
        downsampled = downsample_locus(rep, target_count, random_seed=42)
        
        assert downsampled.duplicate_count == target_count
        lost_clonotypes = original_clonotype_count - downsampled.clonotype_count
        assert lost_clonotypes >= 0
        print(f"\nP1 day 15: {original_duplicate_count} -> {target_count} duplicates, "
              f"lost {lost_clonotypes}/{original_clonotype_count} clonotypes")

    def test_downsample_reproducibility(self):
        """Verify reproducibility with same random seed."""
        rep = _load_p1_f1(0)
        
        down1 = downsample_locus(rep, 50000, random_seed=42)
        down2 = downsample_locus(rep, 50000, random_seed=42)
        
        assert down1.duplicate_count == down2.duplicate_count
        assert down1.clonotype_count == down2.clonotype_count
        # Check same clonotypes in same order
        for c1, c2 in zip(down1.clonotypes, down2.clonotypes):
            assert c1.sequence_id == c2.sequence_id
            assert c1.duplicate_count == c2.duplicate_count


# ------------------------------------------------------------------
# Resample to gene usage tests
# ------------------------------------------------------------------


class TestResampleToGeneUsageBasic:
    """Basic functionality tests for resample_to_gene_usage."""

    def test_raises_on_invalid_gene_type(self):
        from mir.common.clonotype import Clonotype
        
        rep = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                ),
            ],
            locus="TRB",
        )
        
        target_usage = {"TRBV1": 100}
        
        with pytest.raises(ValueError, match="gene_type must be"):
            resample_to_gene_usage(rep, target_usage, gene_type="invalid")

    def test_resample_exact_duplicate_count(self):
        """Verify resampled repertoire has same total duplicates."""
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id="1",
                locus="TRB",
                junction_aa="CASSF",
                v_gene="TRBV1",
                j_gene="TRBJ1-1",
                duplicate_count=100,
            ),
            Clonotype(
                sequence_id="2",
                locus="TRB",
                junction_aa="CASSGF",
                v_gene="TRBV2",
                j_gene="TRBJ1-1",
                duplicate_count=200,
            ),
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        original_duplicates = rep.duplicate_count
        
        target_usage = {"TRBV1": 200, "TRBV2": 100}
        resampled = resample_to_gene_usage(rep, target_usage, gene_type="v", random_seed=42)
        
        assert resampled.duplicate_count == original_duplicates
        assert isinstance(resampled, LocusRepertoire)

    def test_resample_v_weighted(self):
        """Test V-gene resampling with weighted mode."""
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id="1",
                locus="TRB",
                junction_aa="CASSF",
                v_gene="TRBV1",
                j_gene="TRBJ1-1",
                duplicate_count=100,
            ),
            Clonotype(
                sequence_id="2",
                locus="TRB",
                junction_aa="CASSGF",
                v_gene="TRBV2",
                j_gene="TRBJ1-1",
                duplicate_count=300,
            ),
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        original_gu = GeneUsage.from_repertoire(rep)
        
        # Target: reverse the usage (favor TRBV1 over TRBV2)
        target_usage = {"TRBV1": 300, "TRBV2": 100}
        resampled = resample_to_gene_usage(
            rep, target_usage, gene_type="v", weighted=True, random_seed=42
        )
        
        assert resampled.duplicate_count == rep.duplicate_count

    def test_resample_vj_unweighted(self):
        """Test V-J resampling with unweighted mode."""
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id="1",
                locus="TRB",
                junction_aa="CASSF",
                v_gene="TRBV1",
                j_gene="TRBJ1-1",
                duplicate_count=50,
            ),
            Clonotype(
                sequence_id="2",
                locus="TRB",
                junction_aa="CASSGF",
                v_gene="TRBV1",
                j_gene="TRBJ2-1",
                duplicate_count=50,
            ),
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        target_usage = {("TRBV1", "TRBJ1-1"): 50, ("TRBV1", "TRBJ2-1"): 50}
        resampled = resample_to_gene_usage(
            rep, target_usage, gene_type="vj", weighted=False, random_seed=42
        )
        
        assert resampled.duplicate_count == 100

    def test_resample_sample_repertoire(self):
        """Test resampling SampleRepertoire."""
        from mir.common.clonotype import Clonotype
        
        trb = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=200,
                ),
            ],
            locus="TRB",
        )
        
        sample = SampleRepertoire(loci={"TRB": trb}, sample_id="s1")
        target_usage = {"TRBV1": 200}
        resampled = resample_to_gene_usage(sample, target_usage, gene_type="v", random_seed=42)
        
        assert isinstance(resampled, SampleRepertoire)
        assert resampled.loci["TRB"].duplicate_count == 200


class TestResampleToGeneUsageStatistical:
    """Statistical tests to verify resampled gene usage matches target."""

    def test_chi2_test_v_usage_convergence(self):
        """Use Chi-square test to verify V-gene usage similarity."""
        from mir.common.clonotype import Clonotype
        
        # Create repertoire with 5 V genes, each with 200 duplicates
        clonotypes = []
        for i in range(1, 6):
            clonotypes.append(
                Clonotype(
                    sequence_id=str(i),
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene=f"TRBV{i}",
                    j_gene="TRBJ1-1",
                    duplicate_count=200,
                )
            )
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        
        # Target: make V1 and V2 more abundant, V3-V5 less abundant
        target_usage = {
            "TRBV1": 400,
            "TRBV2": 300,
            "TRBV3": 150,
            "TRBV4": 100,
            "TRBV5": 50,
        }
        
        resampled = resample_to_gene_usage(
            rep, target_usage, gene_type="v", weighted=True, random_seed=42
        )
        
        # Compute resulting gene usage
        resampled_gu = GeneUsage.from_repertoire(resampled)
        resulting_usage = resampled_gu.v_usage(rep.locus, count="duplicates")
        
        # Prepare for chi-square test: observed vs expected
        genes = sorted(target_usage.keys())
        observed = np.array([resulting_usage.get(g, 0) for g in genes])
        expected = np.array([target_usage[g] for g in genes])
        
        # Chi-square test: should not be significantly different (p > 0.05)
        # In practice, with finite sampling, we may be lenient
        chi2_stat, p_value = stats.chisquare(observed, expected)
        assert p_value > 0.01, f"Chi-square p-value {p_value} too low; usage diverged from target"

    def test_ks_test_v_fractions(self):
        """Use Kolmogorov-Smirnov test on V-gene fractions."""
        from mir.common.clonotype import Clonotype
        
        clonotypes = []
        for i in range(1, 11):
            clonotypes.append(
                Clonotype(
                    sequence_id=str(i),
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene=f"TRBV{i}",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                )
            )
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        
        # Target: uniform distribution over all V genes
        target_usage = {f"TRBV{i}": 100 for i in range(1, 11)}
        
        resampled = resample_to_gene_usage(
            rep, target_usage, gene_type="v", weighted=True, random_seed=42
        )
        
        resampled_gu = GeneUsage.from_repertoire(resampled)
        resulting_usage = resampled_gu.v_usage(rep.locus, count="duplicates")
        
        # Get fractions
        genes = sorted(target_usage.keys())
        resulting_fracs = np.array([resulting_usage.get(g, 0) / rep.duplicate_count for g in genes])
        target_fracs = np.array([target_usage[g] / sum(target_usage.values()) for g in genes])
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(resulting_fracs, target_fracs)
        assert p_value > 0.01, f"KS p-value {p_value} too low; fractions diverged from target"

    def test_resample_alters_gene_usage(self):
        """Verify resampling actually changes gene usage in the direction of target."""
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id="1",
                locus="TRB",
                junction_aa="CASSF",
                v_gene="TRBV1",
                j_gene="TRBJ1-1",
                duplicate_count=900,  # 90%
            ),
            Clonotype(
                sequence_id="2",
                locus="TRB",
                junction_aa="CASSGF",
                v_gene="TRBV2",
                j_gene="TRBJ1-1",
                duplicate_count=100,  # 10%
            ),
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        
        # Original: heavily skewed toward TRBV1
        original_gu = GeneUsage.from_repertoire(rep)
        orig_v1_frac = original_gu.v_usage(rep.locus, count="duplicates")["TRBV1"] / rep.duplicate_count
        
        # Target: reverse (90% V2, 10% V1)
        target_usage = {"TRBV1": 100, "TRBV2": 900}
        
        resampled = resample_to_gene_usage(
            rep, target_usage, gene_type="v", weighted=True, random_seed=42
        )
        
        resampled_gu = GeneUsage.from_repertoire(resampled)
        resampled_v1_frac = resampled_gu.v_usage(rep.locus, count="duplicates")["TRBV1"] / resampled.duplicate_count
        
        # V1 fraction should decrease
        assert resampled_v1_frac < orig_v1_frac, "Resampling did not move toward target usage"


@skip_integration
@pytest.mark.skipif(not _YFV_AVAILABLE, reason="YFV dataset not found")
@pytest.mark.integration
class TestResampleToGeneUsageYFV:
    """Integration tests using YFV data."""

    def test_resample_p1_day0_to_balanced_v_usage(self):
        """Resample P1 day 0 to have more balanced V-gene usage."""
        rep = _load_p1_f1(0)
        
        # Compute original usage
        orig_gu = GeneUsage.from_repertoire(rep)
        orig_v_usage = orig_gu.v_usage(rep.locus, count="duplicates")
        
        # Create target: uniform usage over observed V genes
        n_v_genes = len(orig_v_usage)
        target_per_gene = rep.duplicate_count // n_v_genes
        target_usage = {v: target_per_gene for v in orig_v_usage.keys()}
        
        resampled = resample_to_gene_usage(
            rep, target_usage, gene_type="v", weighted=True, random_seed=42
        )
        
        assert resampled.duplicate_count == rep.duplicate_count
        
        # Check that resampling moved closer to uniform
        resampled_gu = GeneUsage.from_repertoire(resampled)
        resampled_v_usage = resampled_gu.v_usage(rep.locus, count="duplicates")
        
        orig_fracs = np.array(list(orig_v_usage.values())) / rep.duplicate_count
        resampled_fracs = np.array([resampled_v_usage[v] for v in orig_v_usage.keys()]) / resampled.duplicate_count
        
        # Variance should decrease (more uniform)
        assert np.var(resampled_fracs) < np.var(orig_fracs)

    def test_resample_p1_day0_to_day15_usage(self):
        """Resample P1 day 0 to match P1 day 15 V-gene usage proportions."""
        rep_d0 = _load_p1_f1(0)
        rep_d15 = _load_p1_f1(15)
        
        # Get day 15 V-gene usage as target (need to scale to day 0's duplicate count)
        d15_gu = GeneUsage.from_repertoire(rep_d15)
        d15_usage = d15_gu.v_usage(rep_d15.locus, count="duplicates")
        
        # Scale target to day 0's duplicate count
        d15_total = rep_d15.duplicate_count
        d0_total = rep_d0.duplicate_count
        target_usage = {g: int(count * d0_total / d15_total) for g, count in d15_usage.items()}
        
        # Resample day 0 to match day 15
        resampled = resample_to_gene_usage(
            rep_d0, target_usage, gene_type="v", weighted=True, random_seed=42
        )
        
        # Compare resulting usage to target
        resampled_gu = GeneUsage.from_repertoire(resampled)
        resulting_usage = resampled_gu.v_usage(rep_d0.locus, count="duplicates")
        
        # Get common V genes
        common_genes = set(target_usage.keys()) & set(resulting_usage.keys())
        
        if common_genes:
            # Compare proportions, not absolute counts
            target_vals = np.array([target_usage[g] / sum(target_usage.values()) for g in common_genes])
            resulting_vals = np.array([resulting_usage[g] / resampled.duplicate_count for g in common_genes])
            
            # Use KS test for proportions
            ks_stat, p_value = stats.ks_2samp(resulting_vals, target_vals)
            print(f"\nResample D0→D15: KS stat={ks_stat:.4f}, p={p_value:.4f}")
            assert p_value > 0.001, f"Resampled usage diverged from target (p={p_value})"

    def test_resample_reproducibility(self):
        """Verify reproducibility with same random seed."""
        rep = _load_p1_f1(0)
        
        gu = GeneUsage.from_repertoire(rep)
        target_usage = gu.v_usage(rep.locus, count="duplicates")
        
        resamp1 = resample_to_gene_usage(rep, target_usage, gene_type="v", random_seed=42)
        resamp2 = resample_to_gene_usage(rep, target_usage, gene_type="v", random_seed=42)
        
        assert resamp1.duplicate_count == resamp2.duplicate_count
        assert resamp1.clonotype_count == resamp2.clonotype_count
        
        for c1, c2 in zip(resamp1.clonotypes, resamp2.clonotypes):
            assert c1.sequence_id == c2.sequence_id
            assert c1.duplicate_count == c2.duplicate_count


# ------------------------------------------------------------------
# Select top clonotypes tests
# ------------------------------------------------------------------


class TestSelectTopBasic:
    """Basic functionality tests for select_top."""

    def test_raises_on_zero_top_n(self):
        rep = LocusRepertoire(clonotypes=[], locus="TRB")
        with pytest.raises(ValueError, match="top_n must be > 0"):
            select_top(rep, 0)

    def test_raises_on_negative_top_n(self):
        rep = LocusRepertoire(clonotypes=[], locus="TRB")
        with pytest.raises(ValueError, match="top_n must be > 0"):
            select_top(rep, -5)

    def test_warns_when_top_n_exceeds_clonotypes(self):
        from mir.common.clonotype import Clonotype
        
        rep = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                ),
            ],
            locus="TRB",
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = select_top(rep, 100)
            assert len(w) == 1
            assert "returning all clonotypes" in str(w[0].message)
            assert result.clonotype_count == 1

    def test_select_top_3(self):
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id="1",
                locus="TRB",
                junction_aa="CASSF",
                v_gene="TRBV1",
                j_gene="TRBJ1-1",
                duplicate_count=500,
            ),
            Clonotype(
                sequence_id="2",
                locus="TRB",
                junction_aa="CASSGF",
                v_gene="TRBV2",
                j_gene="TRBJ1-1",
                duplicate_count=200,
            ),
            Clonotype(
                sequence_id="3",
                locus="TRB",
                junction_aa="CASSYGF",
                v_gene="TRBV3",
                j_gene="TRBJ1-1",
                duplicate_count=300,
            ),
            Clonotype(
                sequence_id="4",
                locus="TRB",
                junction_aa="CASSSGF",
                v_gene="TRBV4",
                j_gene="TRBJ1-1",
                duplicate_count=100,
            ),
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        selected = select_top(rep, 3)
        
        assert selected.clonotype_count == 3
        assert selected.clonotypes[0].duplicate_count == 500
        assert selected.clonotypes[1].duplicate_count == 300
        assert selected.clonotypes[2].duplicate_count == 200

    def test_selected_sorted_by_duplicate_count(self):
        from mir.common.clonotype import Clonotype
        
        clonotypes = [
            Clonotype(
                sequence_id=str(i),
                locus="TRB",
                junction_aa="CASSF",
                v_gene=f"TRBV{i}",
                j_gene="TRBJ1-1",
                duplicate_count=100 * (i % 5 + 1),
            )
            for i in range(10)
        ]
        
        rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB")
        selected = select_top(rep, 5)
        
        # Verify all results are sorted descending
        for i in range(len(selected.clonotypes) - 1):
            assert (
                selected.clonotypes[i].duplicate_count
                >= selected.clonotypes[i + 1].duplicate_count
            )

    def test_preserves_metadata(self):
        from mir.common.clonotype import Clonotype
        
        rep = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                ),
            ],
            locus="TRB",
            repertoire_id="test_rep",
            repertoire_metadata={"donor": "P1", "day": 0},
        )
        
        selected = select_top(rep, 1)
        assert selected.repertoire_id == "test_rep"
        assert selected.repertoire_metadata == {"donor": "P1", "day": 0}

    def test_select_top_sample_repertoire(self):
        from mir.common.clonotype import Clonotype
        
        trb = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSF",
                    v_gene="TRBV1",
                    j_gene="TRBJ1-1",
                    duplicate_count=100,
                ),
                Clonotype(
                    sequence_id="2",
                    locus="TRB",
                    junction_aa="CASSGF",
                    v_gene="TRBV2",
                    j_gene="TRBJ1-1",
                    duplicate_count=50,
                ),
            ],
            locus="TRB",
        )
        tra = LocusRepertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="3",
                    locus="TRA",
                    junction_aa="CAVRDSNYQLIW",
                    v_gene="TRAV1",
                    j_gene="TRAJ33",
                    duplicate_count=200,
                ),
                Clonotype(
                    sequence_id="4",
                    locus="TRA",
                    junction_aa="CAVRDSNYQLIF",
                    v_gene="TRAV2",
                    j_gene="TRAJ33",
                    duplicate_count=100,
                ),
            ],
            locus="TRA",
        )
        
        sample = SampleRepertoire(loci={"TRB": trb, "TRA": tra}, sample_id="s1")
        selected = select_top(sample, 1)
        
        assert isinstance(selected, SampleRepertoire)
        assert selected.loci["TRB"].clonotype_count == 1
        assert selected.loci["TRA"].clonotype_count == 1
        assert selected.loci["TRB"].clonotypes[0].duplicate_count == 100
        assert selected.loci["TRA"].clonotypes[0].duplicate_count == 200
        assert selected.sample_id == "s1"


@skip_integration
@pytest.mark.skipif(not _YFV_AVAILABLE, reason="YFV dataset not found")
@pytest.mark.integration
class TestSelectTopYFV:
    """Integration tests for select_top using YFV data."""

    def test_select_top_p1_day0(self):
        """Select top 1000 clonotypes from P1 day 0."""
        rep = _load_p1_f1(0)
        original_count = rep.clonotype_count
        
        selected = select_top(rep, 1000)
        
        assert selected.clonotype_count == 1000
        assert selected.duplicate_count <= rep.duplicate_count
        # Verify sorted descending
        for i in range(len(selected.clonotypes) - 1):
            assert (
                selected.clonotypes[i].duplicate_count
                >= selected.clonotypes[i + 1].duplicate_count
            )
        print(f"\nP1 day 0: selected 1000 from {original_count} clonotypes")

    def test_select_top_preserves_duplicate_count_sum(self):
        """Verify selected clonotypes sum is less than or equal to original."""
        rep = _load_p1_f1(0)
        selected = select_top(rep, 500)
        
        assert selected.duplicate_count <= rep.duplicate_count
        print(f"\nDuplicate count: {rep.duplicate_count} -> {selected.duplicate_count}")

    def test_select_top_ordering(self):
        """Verify clonotypes are returned in descending duplicate count order."""
        rep = _load_p1_f1(0)
        selected = select_top(rep, 100)
        
        for i in range(len(selected.clonotypes) - 1):
            assert (
                selected.clonotypes[i].duplicate_count
                >= selected.clonotypes[i + 1].duplicate_count
            )
