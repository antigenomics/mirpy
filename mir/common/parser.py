import logging
import typing as t
import warnings
from collections import namedtuple

import pandas as pd

from mir.common.clonotype import JunctionMarkup, Clonotype, ClonotypeAA, ClonotypeNT
from mir.common.gene_library import GeneLibrary, GeneEntry

# Legacy aliases kept for callers that reference old names
SegmentLibrary = GeneLibrary
Segment = GeneEntry

_AIRR_LOCUS_ALIASES = {
    'alpha': {'alpha', 'tra'},
    'beta': {'beta', 'trb'},
    'gamma': {'gamma', 'trg'},
    'delta': {'delta', 'trd'},
    'heavy': {'heavy', 'igh'},
    'kappa': {'kappa', 'igk'},
    'lambda': {'lambda', 'igl'},
}

# Mapping from VDJtools / legacy column names to AIRR column names
_VDJTOOLS_TO_AIRR: dict[str, str] = {
    'count':    'duplicate_count',
    '#count':   'duplicate_count',
    'cdr3nt':   'junction',
    'cdr3aa':   'junction_aa',
    'v':        'v_gene',
    'd':        'd_gene',
    'j':        'j_gene',
    'VEnd':     'v_sequence_end',
    'DStart':   'd_sequence_start',
    'DEnd':     'd_sequence_end',
    'JStart':   'j_sequence_start',
}


def _gene_str(val) -> str:
    """Normalise a gene field value to a plain string."""
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return ""
    s = str(val).strip().split(',')[0].split(';')[0]
    return s if s not in ('.', '') else ""


class SegmentParser:
    """Resolve raw V/J gene strings from a parser row into :class:`GeneEntry` objects.

    Parameters
    ----------
    lib:
        Gene library to resolve allele names against.
    select_most_probable:
        When ``True``, take only the first candidate from comma- or
        semicolon-delimited multi-hit strings (e.g. ``'TRBV1*01,TRBV1*02'``).
    mock_allele:
        When ``True``, calls :meth:`GeneLibrary.get_or_create_noallele` so
        that bare gene names without ``*NN`` suffixes are accepted.
    remove_allele:
        When ``True``, strips the allele suffix before lookup, returning the
        base gene name entry.
    """

    def __init__(self, lib: SegmentLibrary,
                 select_most_probable=True,
                 mock_allele: bool = True,
                 remove_allele: bool = False) -> None:
        self.lib = lib
        self.mock_allele = mock_allele
        self.remove_allele = remove_allele
        self.select_most_probable = select_most_probable

    def parse(self, id: str) -> GeneEntry | None:
        """Resolve *id* to a :class:`GeneEntry`, or ``None`` if unparseable."""
        id = id.strip()
        if pd.isna(id) or len(id) < 5:
            return None
        if self.select_most_probable:
            id = id.split(',')[0]   # keep first of comma-delimited candidates
            id = id.split(';')[0]   # keep first of semicolon-delimited candidates
        if self.remove_allele:
            id = id.split('*', 1)[0]
        if self.mock_allele:
            return self.lib.get_or_create_noallele(id)
        return self.lib.get_or_create(id)


class ClonotypeTableParser:
    """Parse clonotype tables into lists of :class:`Clonotype` objects.

    Accepts both file paths (string) and pre-loaded :class:`pd.DataFrame`.
    Column names are normalised via :meth:`normalize_df` before parsing so
    that both VDJtools-style (``count``, ``cdr3nt``, ``cdr3aa``, ``v``, ``j``)
    and AIRR-style (``duplicate_count``, ``junction``, ``junction_aa``,
    ``v_gene``, ``j_gene``) input files are handled transparently.
    """

    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t') -> None:
        self.segment_parser = SegmentParser(lib)
        self.sep = sep

    @staticmethod
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names: strip leading ``#``, map VDJtools → AIRR."""
        df = df.copy()
        df.columns = [c.lstrip('#') for c in df.columns]
        return df.rename(columns=_VDJTOOLS_TO_AIRR)

    def parse(self, source: str | pd.DataFrame, n: int = None, sample: bool = False) -> list[Clonotype]:
        """Parse *source* (path or DataFrame) into a list of clonotypes.

        Parameters
        ----------
        source:
            Either a file path string or a :class:`pd.DataFrame`.
        n:
            Maximum number of rows to parse (``None`` = all).
        sample:
            When ``True`` and *n* is set, sample randomly instead of taking
            the first *n* rows.
        """
        if isinstance(source, str):
            if n is None or not sample:
                source = pd.read_csv(source, sep=self.sep, nrows=n)
            else:
                source = pd.read_csv(source, sep=self.sep).sample(n=n, random_state=42)
            source = self.normalize_df(source)
        else:
            if not sample:
                source = source.head(n)
            elif sample and n is not None:
                source = source.sample(n=n, random_state=42)
        return self.parse_inner(source)

    def parse_inner(self, source: pd.DataFrame) -> list[Clonotype]:
        """Parse an already-loaded DataFrame with AIRR-normalised column names."""
        cols = set(source.columns)
        if 'junction_aa' not in cols or 'v_gene' not in cols or 'j_gene' not in cols:
            raise ValueError(
                f'Critical columns missing in df {list(source.columns)}. '
                f'Expected junction_aa, v_gene, j_gene (AIRR names) or '
                f'cdr3aa, v, j (VDJtools names — pass through normalize_df first).')
        has_markup = {'v_sequence_end', 'd_sequence_start',
                      'd_sequence_end', 'j_sequence_start'}.issubset(cols)
        clonotypes = []
        for index, row in source.iterrows():
            clonotypes.append(Clonotype(
                sequence_id=str(index),
                duplicate_count=int(row['duplicate_count']) if 'duplicate_count' in cols else 1,
                junction=str(row['junction']) if 'junction' in cols else "",
                junction_aa=str(row['junction_aa']),
                v_gene=_gene_str(row.get('v_gene')),
                d_gene=_gene_str(row.get('d_gene')) if 'd_gene' in cols else "",
                j_gene=_gene_str(row.get('j_gene')),
                v_sequence_end=int(row['v_sequence_end']) if has_markup else -1,
                d_sequence_start=int(row['d_sequence_start']) if has_markup else -1,
                d_sequence_end=int(row['d_sequence_end']) if has_markup else -1,
                j_sequence_start=int(row['j_sequence_start']) if has_markup else -1,
            ))
        return clonotypes


VdjdbPayload = namedtuple(
    'VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')


class VDJdbSlimParser(ClonotypeTableParser):
    """Parse VDJdb slim export files."""

    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 species: str = 'HomoSapiens',
                 gene: str = 'TRB',
                 filter: t.Callable[[pd.DataFrame],
                                    pd.DataFrame] = lambda x: x,
                 warn: int = 0) -> None:
        super().__init__(lib)
        self.species = species
        self.gene = gene
        self.filter = filter
        self.warn = warn

    def parse_inner(self, source: pd.DataFrame) -> list[Clonotype]:
        if self.species:
            source = source[source['species'] == self.species]
        if self.gene:
            source = source[source['gene'] == self.gene]
        if self.filter:
            source = self.filter(source)
        res = []
        wrn = 0
        for idx, row in source.iterrows():
            try:
                c = Clonotype(
                    sequence_id=str(idx),
                    junction_aa=str(row['cdr3']),
                    v_gene=_gene_str(row.get('v.segm')),
                    j_gene=_gene_str(row.get('j.segm')),
                )
                c.clone_metadata['vdjdb'] = VdjdbPayload(
                    row['mhc.a'], row['mhc.b'], row['mhc.class'],
                    row['antigen.epitope'], row['antigen.species'])
                res.append(c)
            except Exception as e:
                if wrn < self.warn:
                    wrn += 1
                    warnings.warn(f'Error parsing VDJdb line {row} - {e}')
        return res


class OlgaParser(ClonotypeTableParser):
    """Parse OLGA-generated output files."""

    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary()) -> None:
        super().__init__(lib)

    def read_table(self, path: str, n: int = None) -> pd.DataFrame:
        return pd.read_csv(path,
                           header=None,
                           names=['junction', 'junction_aa', 'v_gene', 'j_gene'], sep='\t',
                           nrows=n)

    def parse_inner(self, source: pd.DataFrame) -> list[Clonotype]:
        return [Clonotype(
                    sequence_id=str(index),
                    junction=str(row['junction']),
                    junction_aa=str(row['junction_aa']),
                    v_gene=_gene_str(row.get('v_gene')),
                    j_gene=_gene_str(row.get('j_gene')),
                ) for index, row in source.iterrows()]


class VDJtoolsParser(ClonotypeTableParser):
    """Parse VDJtools output files.

    Expected columns (after :meth:`normalize_df`):
    ``duplicate_count, junction, junction_aa, v_gene, d_gene, j_gene,
    v_end, d_start, d_end, j_start``.

    The raw VDJtools header uses ``#count`` (or ``count``), ``cdr3nt``,
    ``cdr3aa``, ``v``, ``d``, ``j``, ``VEnd``, ``DStart``, ``DEnd``,
    ``JStart`` — these are automatically mapped via :meth:`normalize_df`.
    """

    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t') -> None:
        super().__init__(lib, sep)

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        cols = set(df.columns)
        has_markup = {'v_sequence_end', 'd_sequence_start',
                      'd_sequence_end', 'j_sequence_start'}.issubset(cols)
        clonotypes = []
        for index, row in df.iterrows():
            clonotypes.append(Clonotype(
                sequence_id=str(index),
                duplicate_count=int(row['duplicate_count']),
                junction=str(row['junction']),
                junction_aa=str(row['junction_aa']),
                v_gene=_gene_str(row.get('v_gene')),
                d_gene=_gene_str(row.get('d_gene')) if 'd_gene' in cols else "",
                j_gene=_gene_str(row.get('j_gene')),
                v_sequence_end=int(row['v_sequence_end']) if has_markup else -1,
                d_sequence_start=int(row['d_sequence_start']) if has_markup else -1,
                d_sequence_end=int(row['d_sequence_end']) if has_markup else -1,
                j_sequence_start=int(row['j_sequence_start']) if has_markup else -1,
            ))
        return clonotypes


class DoubleChainVDJtoolsParser(ClonotypeTableParser):
    """Parse paired-chain VDJtools-style files with alpha and beta columns."""

    def __init__(self,
                 column_mapping=None,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t'
                 ):
        super().__init__(lib, sep)
        if column_mapping is None:
            column_mapping = {
                'epitope': 'Peptide',
                'mhc.a': 'HLA',
                'Va': 'Va',
                'Ja': 'Ja',
                'cdr3a': 'CDR3a_extended',
                'Vb': 'Vb',
                'Jb': 'Jb',
                'cdr3b': 'CDR3b_extended',
            }
        self.column_mapping = column_mapping

    def parse_inner(self, source: pd.DataFrame) -> list[Clonotype]:
        raise NotImplementedError("DoubleChainVDJtoolsParser is not supported in this version")


class AIRRParser(ClonotypeTableParser):
    """Parse AIRR-format files.

    Mandatory columns: ``locus``, ``v_call``, ``j_call``, ``junction_aa``.
    """

    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t',
                 locus='beta') -> None:
        super().__init__(lib, sep)
        self.locus = locus
        self.mandatory_columns = ['locus', 'v_call', 'j_call', 'junction_aa']

    def get_locus_aliases(self) -> set[str]:
        locus = str(self.locus).strip()
        locus_normalized = locus.lower()
        if locus_normalized in _AIRR_LOCUS_ALIASES:
            return _AIRR_LOCUS_ALIASES[locus_normalized]
        for aliases in _AIRR_LOCUS_ALIASES.values():
            if locus_normalized in aliases:
                return aliases
        return {locus_normalized}

    def validate_columns(self, df: pd.DataFrame):
        for col in self.mandatory_columns:
            if col not in df.columns:
                raise KeyError(f'Mandatory column {col} is missing! '
                               f'List of mandatory columns is {self.mandatory_columns}.')

    def check_not_na_columns(self, row, index):
        for k, v in row.items():
            if pd.isna(v):
                logging.warning(f'Filtered out row {index}, because it has None in {k}')
                return False
        return True

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        self.validate_columns(df)
        locus_aliases = self.get_locus_aliases()
        df = df[df.locus.astype(str).str.strip().str.lower().isin(locus_aliases)]
        clonotypes = []
        for i, row in df.iterrows():
            try:
                v = _gene_str(row.get('v_call'))
                j = _gene_str(row.get('j_call'))
                if not v or not j:
                    raise ValueError(f'Missing v_call or j_call in row {i}')
                seq_id = str(row['clone_id']) if 'clone_id' in df.columns else str(i)
                clonotypes.append(Clonotype(
                    sequence_id=seq_id,
                    locus=str(row.get('locus', '')),
                    junction_aa=str(row['junction_aa']),
                    v_gene=v,
                    j_gene=j,
                ))
            except Exception as e:
                logging.warning(f"Error parsing row {i + 1}: {e}")
        return clonotypes


class DoubleChainAIRRParser(AIRRParser):
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t', mapping_column='clone_id') -> None:
        super().__init__(lib, sep)
        self.alpha_parser = AIRRParser(lib, sep, 'alpha')
        self.beta_parser = AIRRParser(lib, sep, 'beta')
        self.mapping_column = mapping_column

    def validate_columns(self, df: pd.DataFrame):
        super().validate_columns(df)
        if self.mapping_column not in df.columns:
            raise KeyError(f'Mapping column {self.mapping_column} is missing!')
        if df[self.mapping_column].isna().sum() > 0:
            raise ValueError(f'Mapping column {self.mapping_column} cannot be null!')

    def get_tcr_ids_for_chain(self, clonotypes, chain):
        clone_ids = {x.payload[self.mapping_column]: x for x in clonotypes}
        id_set = set(clone_ids.keys())
        if len(clone_ids) != len(id_set):
            raise ValueError(f'TCR ids are not unique for {chain} chain!')
        return clone_ids

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        raise NotImplementedError("DoubleChainAIRRParser is not supported in this version")
