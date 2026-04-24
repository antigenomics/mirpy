import unittest
import warnings
from pathlib import Path

import pandas as pd

from mir.common.gene_library import GeneLibrary
from mir.common.parser import VDJtoolsParser, ClonotypeTableParser
from mir.common.repertoire import Repertoire

ASSETS_DIR = Path(__file__).parent / "assets"
REAL_DIR = Path(__file__).parent / "real_repertoires"

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
            # paths in metadata are ../samples/; files live in real_repertoires/
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


if __name__ == "__main__":
    unittest.main()
