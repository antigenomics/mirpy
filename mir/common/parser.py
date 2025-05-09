import logging
import typing as t
import warnings
from collections import namedtuple

import pandas as pd

from mir.common.clonotype import JunctionMarkup, Clonotype, ClonotypeAA, ClonotypeNT, PairedChainClone
from mir.common.segments import SegmentLibrary, Segment


class SegmentParser:
    """
    A parser which processes the segment. It uses the segment library as input and has some other parameters
    """
    def __init__(self, lib: SegmentLibrary,
                 select_most_probable=True,
                 mock_allele: bool = True,
                 remove_allele: bool = False) -> None:
        """
        The initialization function for the `SegmentParser`.
        :param lib: the `SegmentLibrary` object which stores the info about the segments to map to
        :param select_most_probable: whether to select the most probable segment out of a sequence or not
        :param mock_allele: whether to substitute the allele with *01 or not
        :param remove_allele: whether to remove the allele from a segment name or not
        """
        self.lib = lib
        self.mock_allele = mock_allele
        self.remove_allele = remove_allele
        self.select_most_probable = select_most_probable

    def parse(self, id: str) -> Segment:
        """
        creates the `Segment` object
        :param id: the name of a segment. should be a string. if the name cannot be parsed would return None
        :return: a `Segment` object. if the name cannot be parsed would return None
        """
        id = id.strip()
        if pd.isna(id) or len(id) < 5:
            return None
            # raise ValueError(
            #     f"Id {id} is null or too short")
        if self.select_most_probable:
            id = id.split(',')[0]
            id = id.split(';')[0]
        if self.remove_allele:
            id = id.split('*', 1)[0]
        if self.mock_allele:
            if not '*' in id:
                id += '*01'
        return self.lib.get_or_create(id)


class ClonotypeTableParser:
    """
    The object which parses clonotype tables.
    Creates a list of clonotypes
    """
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t') -> None:
        self.segment_parser = SegmentParser(lib)
        self.sep = sep

    def parse(self, source: str | pd.DataFrame, n: int = None, sample: bool = False) -> list[Clonotype]:
        """
        Parses the dataset.
        :param source: Should be either a `pd.DataFrame` or a string with filename
        :param n: either None or number of rows to parse
        :param sample: whether the file should be sampled into a smaller one or not
        :return: a list of clonotypes in a file
        """
        if isinstance(source, str):
            if n is None or not sample:
                source = pd.read_csv(source, sep=self.sep, nrows=n)
            else:
                source = pd.read_csv(source, sep=self.sep).sample(n=n, random_state=42)
        else:
            if not sample:
                source = source.head(n)
            elif sample and n is not None:
                source = source.sample(n=n, random_state=42)
        return self.parse_inner(source)

    def parse_inner(self, source: pd.DataFrame) -> list[Clonotype]:
        """
        Reads the file and clonotypes in it. Takes counts and junction in account.
        :param source: the dataframe to perform the parsing on
        :return: list of clonotypes
        """
        if {'cells'}.issubset(source.columns):
            def get_cells(r):
                return r['cells']
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
        if {'cdr3aa', 'v', 'j'}.issubset(source.columns):
            if 'cdr3nt' in source.columns:
                return [ClonotypeNT(cells=get_cells(row),
                                    cdr3aa=row['cdr3aa'],
                                    v=self.segment_parser.parse(row['v']),
                                    d=self.segment_parser.parse(row['d']) if 'd' in source.columns else None,
                                    j=self.segment_parser.parse(row['j']),
                                    cdr3nt=row['cdr3nt'],
                                    junction=get_junction(row),
                                    id=index)
                        for index, row in source.iterrows()]
            else:
                return [ClonotypeAA(cells=get_cells(row),
                                    cdr3aa=row['cdr3aa'],
                                    v=self.segment_parser.parse(row['v']),
                                    d=self.segment_parser.parse(row['d']) if 'd' in source.columns else None,
                                    j=self.segment_parser.parse(row['j']),
                                    id=index)
                        for index, row in source.iterrows()]
        else:
            raise ValueError(
                f'Critical columns missing in df {source.columns}')


# TODO: TCRNET
# TcrnetPayload = namedtuple(
#    'TcrnetPayload', 'degree_s degree_c total_s total_c p_value')

VdjdbPayload = namedtuple(
    'VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')


class VDJdbSlimParser(ClonotypeTableParser):
    """
    The parser which is made to parse VDJdb.
    Has a filtering parameter which can be a lambda function.
    It has also got a parameter `warn` which is made to skip a number of exceptions in reading the file
    """
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 species: str = 'HomoSapiens',
                 gene: str = 'TRB',
                 filter: t.Callable[[pd.DataFrame],
                                    pd.DataFrame] = lambda x: x,
                 warn: int = 0) -> None:
        """
        The initializing function for the parser
        :param lib: the segment library object to parse with
        :param species: by default HomoSapiens, can also be MusMusculus
        :param gene: TRB by default; can be None if you want to parse all the rows in a file
        :param filter: the lambda function to perform file filtering on; should return the boolean
        :param warn: the number of errors to skip while reading a file
        """
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
                res.append(ClonotypeAA(cdr3aa=row['cdr3'],
                                       v=self.segment_parser.parse(
                                           row['v.segm']),
                                       j=self.segment_parser.parse(
                                           row['j.segm']),
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
    """
    An object to parse the OLGA software generated data. Only accepts the `SegmentLibrary` as input
    """
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary()) -> None:
        super().__init__(lib)

    def read_table(self, path: str, n: int = None) -> pd.DataFrame:
        return pd.read_csv(path,
                           header=None,
                           names=['cdr3nt', 'cdr3aa', 'v', 'j'], sep='\t',
                           nrows=n)

    def parse_inner(self, source: pd.DataFrame) -> list[ClonotypeNT]:
        return [ClonotypeNT(cdr3nt=row['cdr3nt'],
                            cdr3aa=row['cdr3aa'],
                            v=self.segment_parser.parse(row['v']),
                            j=self.segment_parser.parse(row['j']),
                            id=index)
                for index, row in source.iterrows()]


class VDJtoolsParser(ClonotypeTableParser):
    """
    A parser to process the result of VDJtools. It is one of the most common formats which includes the following \
    columns: `cdr3aa, cdr3nt, count, v, d, j, VEnd, DStart, DEnd, JStart`.
    """
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t') -> None:
        """
        The initializing function of the parser. Important to put the correct SegmentLibrary and separator
        :param lib: `SegmentLibrary` object, can be default or made using `SegmentLibrary.load_from_imgt`
        :param sep: either tabulation or comma usually
        """
        super().__init__(lib, sep)

    def parse_inner(self, df: pd.DataFrame) -> list[ClonotypeNT]:
        if len(df.columns) == 7:
            def get_junction(_):
                return JunctionMarkup()
        else:
            def get_junction(r):
                return JunctionMarkup(r['VEnd'], r['DStart'], r['DEnd'], r['JStart'])
        return list(df.apply(lambda x: ClonotypeNT(cells=x['count'],
                                                   cdr3nt=x['cdr3nt'],
                                                   cdr3aa=x['cdr3aa'],
                                                   v=self.segment_parser.parse(x['v']),
                                                   d=self.segment_parser.parse(x['d']),
                                                   j=self.segment_parser.parse(x['j']),
                                                   junction=get_junction(x),
                                                   id=x.index
                                                   ), axis=1))


class DoubleChainVDJtoolsParser(ClonotypeTableParser):
    """
    A parser which is used if you need to rename columns in `VDJtoolsParser`. \
    You can pass the name mapping as a parameter. The file should contain HLA information \
    (parameter `mhc.a` in `column_mapping`) and should have both chains information for each row \
    (parameters `cdr3a` and `cdr3b` in `column_mapping`)
    """
    def __init__(self,
                 column_mapping=None,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t'
                 ):
        """
        The initializing function of the parser. Important to put the correct SegmentLibrary and separator. You should \
        also specify column mapping information
        :param lib: `SegmentLibrary` object, can be default or made using `SegmentLibrary.load_from_imgt`
        :param sep: either tabulation or comma usually
        :param column_mapping: the dictionary which maps the columns names to the column names in the initial file.\
        The default mapping is the following: {
                'epitope': 'Peptide',
                'mhc.a': 'HLA',
                'Va': 'Va',
                'Ja': 'Ja',
                'cdr3a': 'CDR3a_extended',
                'Vb': 'Vb',
                'Jb': 'Jb',
                'cdr3b': 'CDR3b_extended',
            }
        """
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

        alpha_clonotypes = source.apply(lambda x: ClonotypeAA(cdr3aa=x[self.column_mapping['cdr3a']],
                                                              v=self.segment_parser.parse(x[self.column_mapping['Va']]),
                                                              j=self.segment_parser.parse(x[self.column_mapping['Ja']]),
                                                              payload={'HLA': x[self.column_mapping['mhc.a']] if
                                                                            'mhc.a' in self.column_mapping else None,
                                                                       'epitope': x[self.column_mapping['epitope']] if
                                                                            'epitope' in self.column_mapping else None}),
                                        axis=1)
        beta_clonotypes = source.apply(lambda x: ClonotypeAA(cdr3aa=x[self.column_mapping['cdr3b']],
                                                             v=self.segment_parser.parse(x[self.column_mapping['Vb']]),
                                                             j=self.segment_parser.parse(x[self.column_mapping['Jb']]),
                                                             payload={'HLA': x[self.column_mapping['mhc.a']] if
                                                                            'mhc.a' in self.column_mapping else None,
                                                                       'epitope': x[self.column_mapping['epitope']] if
                                                                            'epitope' in self.column_mapping else None}),
                                       axis=1)
        return [PairedChainClone(chainA=alpha, chainB=beta) for alpha, beta in zip(alpha_clonotypes, beta_clonotypes)]


class AIRRParser(ClonotypeTableParser):
    """
    A parser to process data in AIRR format. It is one of the most common formats which includes the following \
    columns: `locus, v_call, j_call, junction_aa`. All the other columns including the column for
    """
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t',
                 locus='beta') -> None:
        """
        The initializing function of the parser. Important to put the correct SegmentLibrary and separator
        :param lib: `SegmentLibrary` object, can be default or made using `SegmentLibrary.load_from_imgt`
        :param sep: either tabulation or comma usually
        """
        super().__init__(lib, sep)
        self.locus = locus
        self.mandatory_columns = ['locus', 'v_call', 'j_call', 'junction_aa']


    def validate_columns(self, df: pd.DataFrame):
        for col in self.mandatory_columns:
            if col not in df.columns:
                raise KeyError(f'Mandatory column {col} is missing! List of mandatory columns is {self.mandatory_columns}.')

    def check_not_na_columns(self, row, index):
        for k, v in row.items():
            if pd.isna(v):
                logging.warning(f'Filtered out row {index}, because it has None in {k}')
                return False
        return True

    def parse_inner(self, df: pd.DataFrame) -> list[ClonotypeNT]:
        self.validate_columns(df)
        df = df[df.locus == self.locus]
        clonotypes = []
        for i, row in df.iterrows():
            try:
                if self.check_not_na_columns(row, i):
                    clonotype = ClonotypeAA(
                        cdr3aa=row['junction_aa'],
                        v=self.segment_parser.parse(row['v_call']),
                        j=self.segment_parser.parse(row['j_call']),
                        id=i if 'clone_id' not in df.columns else row['clone_id'],
                        payload={x: y for x, y in row.items() if x not in self.mandatory_columns}
                    )
                    if clonotype.v is None or clonotype.j is None:
                        raise ValueError(f'Error parsing {clonotype}')
                    clonotypes.append(clonotype)
            except Exception as e:
                logging.warn(f"Error parsing row {i + 1}: {e}")
        return clonotypes

class DoubleChainAIRRParser(AIRRParser):
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 sep='\t', mapping_column='clone_id') -> None:
        """
        The initializing function of the parser. Important to put the correct SegmentLibrary and separator
        :param lib: `SegmentLibrary` object, can be default or made using `SegmentLibrary.load_from_imgt`
        :param sep: either tabulation or comma usually
        """
        super().__init__(lib, sep)
        self.alpha_parser = AIRRParser(lib, sep, 'alpha')
        self.beta_parser = AIRRParser(lib, sep, 'beta')
        self.mapping_column = mapping_column

    def validate_columns(self, df: pd.DataFrame):
        super().validate_columns(df)
        if self.mapping_column not in df.columns:
            raise KeyError(f'Mapping column {self.mapping_column} is missing! It is mandatory for paired chain data.')
        if df[self.mapping_column].isna().sum() > 0:
            raise ValueError(f'Mapping column {self.mapping_column} cannot be null! Check your data.')

    def get_tcr_ids_for_chain(self, clonotypes, chain):
        clone_ids = {x.payload[self.mapping_column]: x for x in clonotypes}
        id_set = set(clone_ids.keys())
        if len(clone_ids) != len(id_set):
            raise ValueError(f'TCR ids are not unique for {chain} chain! Check your dataset.')
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
                logging.warning(f'Have filtered out clonotype with id {id} as it is not found in TRA data')
            elif id not in beta_ids_to_clonotype:
                logging.warning(f'Have filtered out clonotype with id {id} as it is not found in TRB data')
            else:
                paired_clonotypes.append(PairedChainClone(chainA=alpha_ids_to_clonotype[id],
                                                          chainB=beta_ids_to_clonotype[id],
                                                          id=id))
        return paired_clonotypes
