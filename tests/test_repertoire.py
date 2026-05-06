import unittest
import warnings
import time
import tracemalloc
from pathlib import Path

import pandas as pd
import pytest

from mir.common.gene_library import GeneLibrary
from mir.common.clonotype import Clonotype
from mir.common.parser import VDJtoolsParser, ClonotypeTableParser
from mir.common.repertoire import Repertoire, SampleRepertoire
from tests.conftest import skip_benchmarks

ASSETS_DIR = Path(__file__).parent / "assets"
REAL_DIR = Path(__file__).parent / "assets" / "real_repertoires"

# metadata_hsct.txt has 4 time-series samples in the older per-sample export format
# metadata_aging.txt has 41 HSCT cohort samples in VDJtools format

# Column mapping for the older per-sample format (4months.txt.gz, etc.)
_TIMESERIES_TO_AIRR = {
    'Seq. Count': 'duplicate_count',
    'N Sequence': 'junction',
    'AA Sequence': 'junction_aa',
    'V segments': 'v_gene',
    'D segment': 'd_gene',
    'J segments': 'j_gene',
}


def read_vdjtools_df(path: str | Path) -> pd.DataFrame:
    """Read a VDJtools file (gzipped or plain), normalise column names to AIRR."""
    df = pd.read_csv(path, sep='\t', compression='infer')
    return ClonotypeTableParser.normalize_df(df)


def read_timeseries_df(path: str | Path) -> pd.DataFrame:
    """Auto-detect VDJtools or older per-sample format; return AIRR-normalised DataFrame."""
    df = pd.read_csv(path, sep='\t', compression='infer')
    df.columns = [c.lstrip('#') for c in df.columns]
    first_col = df.columns[0]
    if first_col == 'Seq. Count':
        # Older per-sample export format
        df = df.rename(columns=_TIMESERIES_TO_AIRR)
    else:
        # VDJtools format (count/cdr3nt/cdr3aa/v/d/j) — use standard normalization
        from mir.common.parser import _VDJTOOLS_TO_AIRR
        df = df.rename(columns=_VDJTOOLS_TO_AIRR)
    if 'd_gene' in df.columns:
        df['d_gene'] = df['d_gene'].fillna('.')
    return df


def build_repertoire(df: pd.DataFrame, metadata: dict, parser: VDJtoolsParser) -> Repertoire:
    """Parse a normalised DataFrame and wrap it in a Repertoire."""
    clonotypes = parser.parse_inner(df)
    return Repertoire(clonotypes=clonotypes, metadata=metadata)


def _olga_inconsistencies(repertoires, loci=('TRB',)):
    """Return (unknown_v, unknown_j) sets: gene base names absent from OLGA."""
    olga_lib = GeneLibrary.load_default(loci=set(loci), species={'human'}, source='olga')
    olga_bases = {a.split('*')[0] for a in olga_lib.entries}
    unknown_v, unknown_j = set(), set()
    for rep in repertoires:
        for clone in rep.clonotypes:
            v_base = str(clone.v_gene).split('*')[0]
            j_base = str(clone.j_gene).split('*')[0]
            if v_base not in olga_bases:
                unknown_v.add(v_base)
            if j_base not in olga_bases:
                unknown_j.add(j_base)
    return unknown_v, unknown_j



class TestHSCTRepertoires(unittest.TestCase):
    """Load and validate the HSCT cohort (metadata_aging.txt, VDJtools format, 41 samples)."""

    @classmethod
    def setUpClass(cls):
        meta = pd.read_csv(REAL_DIR / "metadata_aging.txt", sep='\t')
        parser = VDJtoolsParser(sep='\t')
        cls.repertoires = []
        for _, row in meta.iterrows():
            path = REAL_DIR / row.file_name
            df = read_vdjtools_df(path)
            rep = build_repertoire(df, dict(row), parser)
            cls.repertoires.append(rep)
        cls.meta = meta

    def test_all_files_loaded(self):
        assert len(self.repertoires) == len(self.meta)

    def test_repertoires_non_empty(self):
        for rep in self.repertoires:
            assert rep.number_of_clones > 0, f"Empty repertoire: {rep.metadata}"

    def test_duplicate_count_positive(self):
        for rep in self.repertoires:
            assert rep.number_of_reads > 0

    def test_genes_have_junction_aa(self):
        for rep in self.repertoires:
            for clone in rep.clonotypes[:5]:
                assert clone.junction_aa, f"Missing junction_aa in {rep.metadata}"

    def test_olga_trb_allele_consistency(self):
        """Report V/J gene names absent from OLGA TRB library (warning only)."""
        unknown_v, unknown_j = _olga_inconsistencies(self.repertoires, loci=('TRB',))
        if unknown_v or unknown_j:
            warnings.warn(
                f"HSCT cohort genes absent from OLGA TRB library — "
                f"V genes: {sorted(unknown_v)}, J genes: {sorted(unknown_j)}"
            )
        # Informational — passes regardless so results are always reported


class TestTimeSeriesRepertoires(unittest.TestCase):
    """Load and validate the time-series cohort (metadata_hsct.txt, 4 samples)."""

    @classmethod
    def setUpClass(cls):
        meta = pd.read_csv(REAL_DIR / "metadata_hsct.txt", sep='\t')
        parser = VDJtoolsParser(sep='\t')
        cls.repertoires = []
        for _, row in meta.iterrows():
            # paths in metadata are ../samples/; files live in assets/real_repertoires/
            fname = Path(row['file.name']).name
            path = REAL_DIR / fname
            df = read_timeseries_df(path)
            rep = build_repertoire(df, dict(row), parser)
            cls.repertoires.append(rep)
        cls.meta = meta

    def test_all_files_loaded(self):
        assert len(self.repertoires) == len(self.meta)

    def test_repertoires_non_empty(self):
        for rep in self.repertoires:
            assert rep.number_of_clones > 0, f"Empty repertoire: {rep.metadata}"

    def test_duplicate_count_positive(self):
        for rep in self.repertoires:
            assert rep.number_of_reads > 0

    def test_genes_have_junction_aa(self):
        for rep in self.repertoires:
            for clone in rep.clonotypes[:5]:
                assert clone.junction_aa, f"Missing junction_aa in {rep.metadata}"

    def test_olga_trb_allele_consistency(self):
        """Report V/J gene names absent from OLGA TRB library (warning only)."""
        unknown_v, unknown_j = _olga_inconsistencies(self.repertoires, loci=('TRB',))
        if unknown_v or unknown_j:
            warnings.warn(
                f"Time-series cohort genes absent from OLGA TRB library — "
                f"V genes: {sorted(unknown_v)}, J genes: {sorted(unknown_j)}"
            )


class TestRepertoirePickleRoundTrip:
    def test_locus_repertoire_pickle_round_trip(self, tmp_path) -> None:
        rep = Repertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRB",
                    junction_aa="CASSLGQETQYF",
                    v_gene="TRBV5-1*01",
                    j_gene="TRBJ2-7*01",
                    duplicate_count=7,
                )
            ],
            locus="TRB",
            repertoire_id="rep1",
            repertoire_metadata={"run": "sample-1"},
        )
        path = tmp_path / "rep.pkl"
        rep.to_pickle(path)
        loaded = Repertoire.from_pickle(path)

        assert loaded.locus == rep.locus
        assert loaded.repertoire_id == rep.repertoire_id
        assert loaded.repertoire_metadata == rep.repertoire_metadata
        assert loaded.clonotype_count == rep.clonotype_count
        assert loaded.duplicate_count == rep.duplicate_count
        assert loaded.clonotypes[0].junction_aa == rep.clonotypes[0].junction_aa

    def test_sample_repertoire_pickle_round_trip(self, tmp_path) -> None:
        rep = Repertoire(
            clonotypes=[
                Clonotype(
                    sequence_id="1",
                    locus="TRA",
                    junction_aa="CAVRDSNYQLIW",
                    v_gene="TRAV1-2*01",
                    j_gene="TRAJ33*01",
                    duplicate_count=3,
                )
            ],
            locus="TRA",
            repertoire_id="rep-tra",
        )
        sample = SampleRepertoire(
            loci={"TRA": rep},
            sample_id="sample-1",
            sample_metadata={"cohort": "test"},
        )
        path = tmp_path / "sample.pkl"
        sample.to_pickle(path)
        loaded = SampleRepertoire.from_pickle(path)

        assert loaded.sample_id == sample.sample_id
        assert loaded.sample_metadata == sample.sample_metadata
        assert set(loaded.loci) == {"TRA"}
        assert loaded.loci["TRA"].clonotype_count == 1


@skip_benchmarks
@pytest.mark.benchmark
class TestRepertoireIOBenchmarks:
    """Benchmark repertoire file I/O + parsing diagnostics.

    Reports total wall-clock, per-file latency distribution, throughput,
    and peak memory while loading the full HSCT cohort.
    """

    def test_hsct_io_parse_runtime_and_memory(self):
        meta = pd.read_csv(REAL_DIR / "metadata_aging.txt", sep='\t')
        parser = VDJtoolsParser(sep='\t')

        per_file_sec: list[float] = []
        total_clones = 0
        total_reads = 0

        tracemalloc.start()
        t0 = time.perf_counter()

        for _, row in meta.iterrows():
            f0 = time.perf_counter()
            path = REAL_DIR / row.file_name
            df = read_vdjtools_df(path)
            rep = build_repertoire(df, dict(row), parser)
            per_file_sec.append(time.perf_counter() - f0)
            total_clones += rep.number_of_clones
            total_reads += rep.number_of_reads

        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        p50 = float(pd.Series(per_file_sec).quantile(0.50))
        p95 = float(pd.Series(per_file_sec).quantile(0.95))
        files = len(per_file_sec)
        throughput = files / max(elapsed, 1e-9)

        print(
            "\nRepertoire I/O benchmark: "
            f"files={files} total={elapsed:.2f}s throughput={throughput:.2f} files/s "
            f"p50={p50:.3f}s p95={p95:.3f}s peak_mem={peak/(1024**2):.1f}MiB "
            f"clones={total_clones:,} reads={total_reads:,}"
        )

        # Data-integrity and performance guardrails.
        assert files == len(meta), "Not all metadata files were benchmarked"
        assert total_clones > 0 and total_reads > 0, "Parsed cohort appears empty"
        assert elapsed < 180.0, f"HSCT repertoire I/O benchmark too slow: {elapsed:.2f}s"
        assert p95 < 8.0, f"Per-file parse p95 too slow: {p95:.2f}s"
        assert peak < 1_500 * 1024 * 1024, "Repertoire I/O peak memory unexpectedly high"


if __name__ == "__main__":
    unittest.main()
