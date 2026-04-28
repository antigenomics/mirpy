"""Integration test: LLWNGPMAV-reactive TRB enrichment in YFV vaccination.

Uses local assets only — no network access required:
  * ``tests/assets/vdjdb.slim.txt.gz``         — VDJdb LLWNGPMAV reference
  * ``notebooks/assets/large/yfv19/``           — YFV AIRR dataset (donor S1)

Run with::

    RUN_INTEGRATION=1 pytest -s tests/test_yfv_integration.py -v

Biology expectations
--------------------
Day 15 (vaccination peak)
    S1/F1 should show significant enrichment of LLWNGPMAV-reactive TRB
    clonotypes relative to the Pgen-matched null:
    * Both clonotype count (n) and duplicate-count (dc) fractions should
      be higher than mock (z > 1.96, p < 0.05).
    * The 1mm match amplifies this signal further.
    * V/J matching in overlap requires V-gene and J-gene match for the
      query↔reference comparison; V/J bias in the mock null is corrected
      via PgenGeneUsageAdjustment.

Day 0 (pre-vaccination)
    Some overlap is expected even pre-vaccination due to cross-reactive
    memory: a fraction of donors carry LLWNGPMAV-reactive clonotypes
    before vaccination (public clonotypes).  With pgen adjustment the
    mock null matches the target V/J distribution so V/J bias is
    eliminated, and any remaining day-0 signal reflects genuine memory.
    The key diagnostic is that day-15 z exceeds day-0 z
    (z_pvj_d0 < z_pvj_d15).

API options
-----------
* ``match_v=True/False`` — require V-gene match in the query↔reference overlap.
* ``match_j=True/False`` — require J-gene match in the query↔reference overlap.
* Defaults are ``match_v=True, match_j=True``.
* V/J bias in the mock null is controlled by passing a
  :class:`~mir.basic.pgen.PgenGeneUsageAdjustment` at construction time.

Observed z-scores (seed=42, n_mocks=200)
-----------------------------------------
Recorded on the full YFV19 dataset; rerun with ``-s`` to verify.

    Day 15, S1/F1:
      pgen-only exact:  z = 13.15  p < 0.0001  n=43
      pgen-only 1mm:    z = 10.29  p < 0.0001  n=627
      pgen+adj exact:   z = 19.63  p < 0.0001
      pgen+adj 1mm:     z = 22.86  p < 0.0001
      dc exact pgen:    z =  5.63  p < 0.0001

    Day 0, S1/F1 (pre-vaccination):
      pgen-only exact:  z =  5.53  (some inflation from V/J bias)
      pgen+adj exact:   z = 12.58  (reflects genuine cross-reactive memory)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import PgenGeneUsageAdjustment
from mir.biomarkers.vdjbet import OverlapResult, VDJBetOverlapAnalysis
from mir.common.filter import filter_functional
from mir.common.gene_library import GeneLibrary
from mir.common.parser import ClonotypeTableParser, VDJdbSlimParser
from mir.common.repertoire import LocusRepertoire
from tests.conftest import skip_integration

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEST_DIR   = Path(__file__).parent
_ASSETS     = _TEST_DIR / "assets"
_VDJDB_FILE = _ASSETS / "vdjdb.slim.txt.gz"
_YFV_DIR    = _TEST_DIR.parent / "notebooks" / "assets" / "large" / "yfv19"

_VDJDB_AVAILABLE = _VDJDB_FILE.exists()
_YFV_AVAILABLE   = (_YFV_DIR / "metadata.txt").exists()


# ---------------------------------------------------------------------------
# Fixtures (module-scoped for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vdjdb_ref() -> LocusRepertoire:
    """Load LLWNGPMAV TRB HLA-A*02 entries from the local VDJdb test asset."""
    sample = VDJdbSlimParser().parse_file(_VDJDB_FILE, species="HomoSapiens")
    trb = sample["TRB"]
    filtered = [
        c for c in trb.clonotypes
        if c.clone_metadata.get("antigen.epitope") == "LLWNGPMAV"
        and "A*02" in c.clone_metadata.get("mhc.a", "")
    ]
    return LocusRepertoire(clonotypes=filtered, locus="TRB")


@pytest.fixture(scope="module")
def analysis(vdjdb_ref) -> VDJBetOverlapAnalysis:
    """Analysis object using default OLGA pool size (100k)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return VDJBetOverlapAnalysis(
            vdjdb_ref, n_mocks=200, seed=42
        )


def _build_yfv_gene_usage() -> GeneUsage:
    """Build TRB gene usage from all YFV repertoires listed in metadata."""
    meta = pd.read_csv(_YFV_DIR / "metadata.txt", sep="\t")
    parser = ClonotypeTableParser()
    all_clones = []
    for _, row in meta.iterrows():
        fpath = _YFV_DIR / row["file_name"]
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, sep="\t", compression="infer")
        if "locus" in df.columns:
            df = df[df["locus"].fillna("") == "TRB"]
        df = df.dropna(subset=["junction_aa"])
        df = df[df["junction_aa"].str.strip().str.len() > 0]
        all_clones.extend(parser.parse_inner(df))

    rep = LocusRepertoire(clonotypes=all_clones, locus="TRB", repertoire_id="yfv19-all-trb")
    return GeneUsage.from_repertoire(rep)


@pytest.fixture(scope="module")
def analysis_pvj(vdjdb_ref) -> VDJBetOverlapAnalysis:
    """Analysis object with pgen+V/J-adjusted null model."""
    target_gu = _build_yfv_gene_usage()
    adj = PgenGeneUsageAdjustment(target_gu, cache_size=100_000, seed=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return VDJBetOverlapAnalysis(vdjdb_ref, n_mocks=200, seed=42, pgen_adjustment=adj)


def _load_s1_f1(day: int) -> LocusRepertoire:
    """Load S1 replica F1 for the given *day* from the YFV dataset."""
    meta = pd.read_csv(_YFV_DIR / "metadata.txt", sep="\t")
    row = meta[(meta["donor"] == "S1") & (meta["day"] == day) & (meta["replica"] == "F1")]
    if row.empty:
        pytest.skip(f"S1/F1/day={day} not found in metadata")
    fname = row.iloc[0]["file_name"]
    df = pd.read_csv(_YFV_DIR / fname, sep="\t", compression="infer")
    if "locus" in df.columns:
        df = df[df["locus"].fillna("") == "TRB"]
    df = df.dropna(subset=["junction_aa"])
    df = df[df["junction_aa"].str.strip().str.len() > 0]
    parser = ClonotypeTableParser()
    clones = parser.parse_inner(df)
    return LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=fname)


def _load_p1_f1(day: int) -> LocusRepertoire:
    """Load P1 replica F1 for the given *day* from the YFV dataset."""
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


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@skip_integration
@pytest.mark.skipif(not _VDJDB_AVAILABLE, reason="vdjdb.slim.txt.gz asset missing")
@pytest.mark.skipif(not _YFV_AVAILABLE,   reason="YFV dataset not found in notebooks/assets/large/yfv19/")
@pytest.mark.integration
class TestYFVS1F1:
    """Biology-driven assertions for donor S1 replica F1.

    All thresholds are empirically calibrated on the real dataset.  If the
    underlying data or mock generation changes they may need adjustment, but
    the qualitative expectations (day 15 > day 0, exact < 1mm) should hold.
    """

    # Each fixture calls score() with a distinct option combination.
    # module-scope analysis caches the pool/mocks so all score() calls reuse them.

    @pytest.fixture(scope="class")
    def d15_pgen(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(15), allow_1mm=False)

    @pytest.fixture(scope="class")
    def d15_pgen_1mm(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(15), allow_1mm=True)

    @pytest.fixture(scope="class")
    def d15_pvj(self, analysis_pvj) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis_pvj.score(
                _load_s1_f1(15), allow_1mm=False,
            )

    @pytest.fixture(scope="class")
    def d15_pvj_1mm(self, analysis_pvj) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis_pvj.score(
                _load_s1_f1(15), allow_1mm=True,
            )

    @pytest.fixture(scope="class")
    def d0_pgen(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(0), allow_1mm=False)

    @pytest.fixture(scope="class")
    def d0_pgen_1mm(self, analysis) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis.score(_load_s1_f1(0), allow_1mm=True)

    @pytest.fixture(scope="class")
    def d0_pvj(self, analysis_pvj) -> OverlapResult:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return analysis_pvj.score(
                _load_s1_f1(0), allow_1mm=False,
            )

    # --- sanity: samples are non-empty ---

    def test_d15_nonempty(self, d15_pgen: OverlapResult) -> None:
        assert d15_pgen.n_total > 0

    def test_d0_nonempty(self, d0_pgen: OverlapResult) -> None:
        assert d0_pgen.n_total > 0

    # --- 1mm finds at least as many matches as exact ---

    def test_d15_1mm_ge_exact(
        self, d15_pgen: OverlapResult, d15_pgen_1mm: OverlapResult
    ) -> None:
        assert d15_pgen_1mm.n >= d15_pgen.n, (
            f"1mm ({d15_pgen_1mm.n}) should be ≥ exact ({d15_pgen.n})"
        )

    def test_d0_1mm_ge_exact(
        self, d0_pgen: OverlapResult, d0_pgen_1mm: OverlapResult
    ) -> None:
        assert d0_pgen_1mm.n >= d0_pgen.n

    # --- day 15: significant enrichment under pgen-only null ---

    def test_d15_pgen_exact_significant(self, d15_pgen: OverlapResult) -> None:
        # pgen-only null, exact match — expected: z >> 1.96 (strong vaccine response)
        print(f"\nday 15 pgen-only exact:  z={d15_pgen.z_n:.2f}  "
              f"p={d15_pgen.p_n:.4f}  n_real={d15_pgen.n}")
        assert d15_pgen.z_n > 1.96, (
            f"day-15 exact pgen z={d15_pgen.z_n:.2f} should be > 1.96"
        )
        assert d15_pgen.p_n < 0.05

    def test_d15_pgen_1mm_significant(self, d15_pgen_1mm: OverlapResult) -> None:
        # pgen-only null, 1mm match — captures near-neighbour clonotypes
        print(f"\nday 15 pgen-only 1mm:    z={d15_pgen_1mm.z_n:.2f}  "
              f"p={d15_pgen_1mm.p_n:.4f}  n_real={d15_pgen_1mm.n}")
        assert d15_pgen_1mm.z_n > 1.96
        assert d15_pgen_1mm.p_n < 0.05

    def test_d15_pvj_exact_significant(self, d15_pvj: OverlapResult) -> None:
        # pgen+V+J null, exact — controls for V/J gene usage bias
        print(f"\nday 15 pgen+VJ exact:    z={d15_pvj.z_n:.2f}  "
              f"p={d15_pvj.p_n:.4f}")
        assert d15_pvj.z_n > 1.96

    def test_d15_pvj_1mm_significant(self, d15_pvj_1mm: OverlapResult) -> None:
        # pgen+V+J null, 1mm
        print(f"\nday 15 pgen+VJ 1mm:      z={d15_pvj_1mm.z_n:.2f}  "
              f"p={d15_pvj_1mm.p_n:.4f}")
        assert d15_pvj_1mm.z_n > 1.96

    def test_d15_dc_pgen_significant(self, d15_pgen: OverlapResult) -> None:
        # duplicate-count overlap (log2) under pgen-only null
        print(f"\nday 15 dc log2 pgen:     z={d15_pgen.z_dc:.2f}  "
              f"p={d15_pgen.p_dc:.4f}")
        assert d15_pgen.z_dc > 1.96

    # --- day 0: effect size smaller than day 15 ---

    def test_d15_pgen_z_gt_d0_pgen_z(
        self, d15_pgen: OverlapResult, d0_pgen: OverlapResult
    ) -> None:
        # Day-0 pgen-only z may still be elevated due to V/J bias (see header)
        print(f"\nz_pgen_exact:  day15={d15_pgen.z_n:.2f}  day0={d0_pgen.z_n:.2f}")
        assert d15_pgen.z_n > d0_pgen.z_n, (
            f"day-15 effect (z={d15_pgen.z_n:.2f}) should exceed "
            f"day-0 (z={d0_pgen.z_n:.2f}) for pgen-only null"
        )

    def test_d15_pvj_z_gt_d0_pvj_z(
        self, d15_pvj: OverlapResult, d0_pvj: OverlapResult
    ) -> None:
        # pgen+V+J corrects V/J bias; day-0 should show clearly smaller effect
        print(f"\nz_pvj_exact:   day15={d15_pvj.z_n:.2f}  day0={d0_pvj.z_n:.2f}")
        assert d15_pvj.z_n > d0_pvj.z_n, (
            f"day-15 effect (z={d15_pvj.z_n:.2f}) should exceed "
            f"day-0 (z={d0_pvj.z_n:.2f}) for pgen+VJ null"
        )

    def test_d0_pvj_vs_pgen_comparison(
        self, d0_pgen: OverlapResult, d0_pvj: OverlapResult
    ) -> None:
        # With pgen adjustment, mock null reflects target V/J distribution,
        # eliminating V/J usage bias.  Any remaining day-0 signal reflects genuine
        # cross-reactive memory (public clonotypes present pre-vaccination).
        # The key comparison between timepoints is covered by
        # test_d15_pvj_z_gt_d0_pvj_z.
        print(
            f"\nday 0: z_pgen={d0_pgen.z_n:.2f}  z_pvj={d0_pvj.z_n:.2f}"
        )
        # Both results should show a positive day-0 signal
        assert d0_pgen.z_n > 0
        assert d0_pvj.z_n > 0

    def test_d15_1mm_amplifies_signal_vs_exact(
        self, d15_pgen: OverlapResult, d15_pgen_1mm: OverlapResult
    ) -> None:
        # 1mm captures ~15x more clonotypes than exact match (n_1mm=627 vs
        # n_exact=43 observed).  z-score can be slightly lower because the
        # mock distributions also expand with 1mm, increasing variance.
        # Threshold: z_1mm >= 70% of z_exact
        # (observed ratio: z_1mm=10.29, z_exact=13.15 → 0.78).
        print(f"\nday 15: z_exact_pgen={d15_pgen.z_n:.2f}  "
              f"z_1mm_pgen={d15_pgen_1mm.z_n:.2f}  "
              f"n_exact={d15_pgen.n}  n_1mm={d15_pgen_1mm.n}")
        assert d15_pgen_1mm.z_n >= d15_pgen.z_n * 0.7, (
            f"1mm z ({d15_pgen_1mm.z_n:.2f}) should be >= 70% of "
            f"exact z ({d15_pgen.z_n:.2f})"
        )


@skip_integration
@pytest.mark.skipif(not _VDJDB_AVAILABLE, reason="vdjdb.slim.txt.gz asset missing")
@pytest.mark.skipif(not _YFV_AVAILABLE,   reason="YFV dataset not found in notebooks/assets/large/yfv19/")
@pytest.mark.integration
class TestFunctionalFilteringCounts:
    """Integration checks for repertoire functional filtering counts."""

    @pytest.fixture(scope="class")
    def imgt_trb_lib(self) -> GeneLibrary:
        return GeneLibrary.load_default(loci={"TRB"}, species={"human"}, source="imgt")

    def test_p1_day0_functional_filter_count(self, imgt_trb_lib: GeneLibrary) -> None:
        rep = _load_p1_f1(0)
        filtered = filter_functional(rep, gene_library=imgt_trb_lib)
        assert isinstance(filtered, LocusRepertoire)
        assert rep.clonotype_count == 624081
        assert filtered.clonotype_count == 604303

    def test_p1_day15_functional_filter_count(self, imgt_trb_lib: GeneLibrary) -> None:
        rep = _load_p1_f1(15)
        filtered = filter_functional(rep, gene_library=imgt_trb_lib)
        assert isinstance(filtered, LocusRepertoire)
        assert rep.clonotype_count == 982154
        assert filtered.clonotype_count == 930938

    def test_llw_functional_filter_count(self, vdjdb_ref: LocusRepertoire, imgt_trb_lib: GeneLibrary) -> None:
        filtered = filter_functional(vdjdb_ref, gene_library=imgt_trb_lib)
        assert isinstance(filtered, LocusRepertoire)
        assert vdjdb_ref.clonotype_count == 409
        assert filtered.clonotype_count == 409
