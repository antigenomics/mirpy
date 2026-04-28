import time
import warnings
from pathlib import Path

import pytest

from mir.common.sampling import downsample, downsample_locus
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.parser import ClonotypeTableParser
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
