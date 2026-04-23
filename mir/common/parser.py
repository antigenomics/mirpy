import logging
import typing as t
import warnings
from collections import namedtuple

import pandas as pd

from mir.common.clonotype import JunctionMarkup, Clonotype, ClonotypeAA, ClonotypeNT, PairedChainClone
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
    'VEnd':     'v_end',
    'DStart':   'd_start',
    'DEnd':     'd_end',
    'JStart':   'j_start',
}


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
        if {'duplicate_count'}.issubset(source.columns):
            def get_cells(r):
                return r['duplicate_count']
        else:
            def get_cells(_):
                return 1
        if {'v_end', 'd_start', 'd_end', 'j_start'}.issubset(source.columns):
            def get_junction(r):
                return JunctionMarkup(r['v_end'],
                                      r['d_start'],
                                      r['d_end'],
                                      r['j_start'])
        else:
            def get_junction(_):
                return None
        if {'junction_aa', 'v_gene', 'j_gene'}.issubset(source.columns):
            if 'junction' in source.columns:
                return [ClonotypeNT(duplicate_count=get_cells(row),
                                    junction_aa=row['junction_aa'],
                                    v_gene=self.segment_parser.parse(row['v_gene']),
                                    d_gene=self.segment_parser.parse(row['d_gene']) if 'd_gene' in source.columns else None,
                                    j_gene=self.segment_parser.parse(row['j_gene']),
                                    junction=row['junction'],
                                    junction_markup=get_junction(row),
                                    id=index)
                        for index, row in source.iterrows()]
            else:
                return [ClonotypeAA(duplicate_count=get_cells(row),
                                    junction_aa=row['junction_aa'],
                                    v_gene=self.segment_parser.parse(row['v_gene']),
                                    d_gene=self.segment_parser.parse(row['d_gene']) if 'd_gene' in source.columns else None,
                                    j_gene=self.segment_parser.parse(row['j_gene']),
                                    id=index)
                        for index, row in source.iterrows()]
        else:
            raise ValueError(
                f'Critical columns missing in df {list(source.columns)}. '
                f'Expected junction_aa, v_gene, j_gene (AIRR names) or '
                f'cdr3aa, v, j (VDJtools names — pass through normalize_df first).')


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

    def parse_inner(self, source: pd.DataFrame) -> list[ClonotypeAA]:
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
                res.append(ClonotypeAA(junction_aa=row['cdr3'],
                                       v_gene=self.segment_parser.parse(row['v.segm']),
                                       j_gene=self.segment_parser.parse(row['j.segm']),
                                       id=idx,
                                       payload={'vdjdb': VdjdbPayload(row['mhc.a'],
                                                                      row['mhc.b'],
                                                                      row['mhc.class'],
                                                                      row['antigen.epitope'],
                                                                      row['antigen.species'])}))
            except Exception as e:
                if wrn < self.warn:
                    wrn = wrn + 1
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

    def parse_inner(self, source: pd.DataFrame) -> list[ClonotypeNT]:
        return [ClonotypeNT(junction=row['junction'],
                            junction_aa=row['junction_aa'],
                            v_gene=self.segment_parser.parse(row['v_gene']),
                            j_gene=self.segment_parser.parse(row['j_gene']),
                            id=index)
                for index, row in source.iterrows()]


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

    def parse_inner(self, df: pd.DataFrame) -> list[ClonotypeNT]:
        has_markup = {'v_end', 'd_start', 'd_end', 'j_start'}.issubset(df.columns)
        if has_markup:
            def get_junction(r):
                return JunctionMarkup(r['v_end'], r['d_start'], r['d_end'], r['j_start'])
        else:
            def get_junction(_):
                return JunctionMarkup()

        return list(df.apply(lambda x: ClonotypeNT(
            duplicate_count=x['duplicate_count'],
            junction=x['junction'],
            junction_aa=x['junction_aa'],
            v_gene=self.segment_parser.parse(x['v_gene']),
            d_gene=self.segment_parser.parse(x['d_gene']) if 'd_gene' in df.columns else None,
            j_gene=self.segment_parser.parse(x['j_gene']),
            junction_markup=get_junction(x),
            id=x.name,
        ), axis=1))


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

    def parse_inner(self, source: pd.DataFrame) -> list[PairedChainClone]:
        alpha_clonotypes = source.apply(
            lambda x: ClonotypeAA(
                junction_aa=x[self.column_mapping['cdr3a']],
                v_gene=self.segment_parser.parse(x[self.column_mapping['Va']]),
                j_gene=self.segment_parser.parse(x[self.column_mapping['Ja']]),
                payload={'HLA': x[self.column_mapping['mhc.a']] if 'mhc.a' in self.column_mapping else None,
                         'epitope': x[self.column_mapping['epitope']] if 'epitope' in self.column_mapping else None}),
            axis=1)
        beta_clonotypes = source.apply(
            lambda x: ClonotypeAA(
                junction_aa=x[self.column_mapping['cdr3b']],
                v_gene=self.segment_parser.parse(x[self.column_mapping['Vb']]),
                j_gene=self.segment_parser.parse(x[self.column_mapping['Jb']]),
                payload={'HLA': x[self.column_mapping['mhc.a']] if 'mhc.a' in self.column_mapping else None,
                         'epitope': x[self.column_mapping['epitope']] if 'epitope' in self.column_mapping else None}),
            axis=1)
        return [PairedChainClone(chainA=alpha, chainB=beta)
                for alpha, beta in zip(alpha_clonotypes, beta_clonotypes)]


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

    def parse_inner(self, df: pd.DataFrame) -> list[ClonotypeAA]:
        self.validate_columns(df)
        locus_aliases = self.get_locus_aliases()
        df = df[df.locus.astype(str).str.strip().str.lower().isin(locus_aliases)]
        clonotypes = []
        for i, row in df.iterrows():
            try:
                if self.check_not_na_columns(row, i):
                    clonotype = ClonotypeAA(
                        junction_aa=row['junction_aa'],
                        v_gene=self.segment_parser.parse(row['v_call']),
                        j_gene=self.segment_parser.parse(row['j_call']),
                        id=i if 'clone_id' not in df.columns else row['clone_id'],
                        payload={x: y for x, y in row.items() if x not in self.mandatory_columns}
                    )
                    if clonotype.v_gene is None or clonotype.j_gene is None:
                        raise ValueError(f'Error parsing {clonotype}')
                    clonotypes.append(clonotype)
            except Exception as e:
                logging.warn(f"Error parsing row {i + 1}: {e}")
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

    def parse_inner(self, df: pd.DataFrame) -> list[ClonotypeNT]:
        self.validate_columns(df)

        logging.info('Started processing TCR alpha chain clonotypes.')
        alpha_clonotypes = self.alpha_parser.parse_inner(df)
        alpha_ids_to_clonotype = self.get_tcr_ids_for_chain(alpha_clonotypes, 'alpha')

        logging.info('Started processing TCR beta chain clonotypes.')
        beta_clonotypes = self.beta_parser.parse_inner(df)
        beta_ids_to_clonotype = self.get_tcr_ids_for_chain(beta_clonotypes, 'beta')

        paired_clonotypes = []
        all_ids = set(beta_ids_to_clonotype).union(set(alpha_ids_to_clonotype))
        for id in all_ids:
            if id not in alpha_ids_to_clonotype:
                logging.warning(f'Filtered out clonotype {id}: not found in TRA data')
            elif id not in beta_ids_to_clonotype:
                logging.warning(f'Filtered out clonotype {id}: not found in TRB data')
            else:
                paired_clonotypes.append(PairedChainClone(
                    chainA=alpha_ids_to_clonotype[id],
                    chainB=beta_ids_to_clonotype[id],
                    id=id))
        return paired_clonotypes
