from collections import namedtuple
import typing as t
import pandas as pd
import warnings
from . import JunctionMarkup, Clonotype, ClonotypeAA, ClonotypeNT, SegmentLibrary, Segment


class SegmentParser:
    def __init__(self, lib: SegmentLibrary,
                 mock_allele: bool = True,
                 remove_allele: bool = False) -> None:
        self.lib = lib
        self.mock_allele = mock_allele
        self.remove_allele = remove_allele

    def parse(self, id: str) -> Segment:
        if not id or len(id) < 5:
            return None
        if self.remove_allele:
            id = id.split('*', 1)[0]
        if self.mock_allele:
            id += '*01'
        return self.lib.get_or_create(id)


class ClonotypeTableParser:
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary()) -> None:
        self.segment_parser = SegmentParser(lib)

    def parse(self, source: str | pd.DataFrame, n: int = None) -> list[Clonotype]:
        if isinstance(source, str):
            source = self.read_table(source, n)
        else:
            source = source.head(n)
        return self.parse_inner(source)

    def read_table(self, path: str, n: int = None) -> pd.DataFrame:
        return pd.read_csv(path, sep='\t', nrows=n)

    def parse_inner(self, source: str | pd.DataFrame) -> list[Clonotype]:
        if {'cells'}.issubset(source.columns):
            def get_cells(r): return r['cells']
        else:
            def get_cells(_): return 1
        if {'v_end', 'd_start', 'd_end', 'j_start'}.issubset(source.columns):
            def get_junction(r): return JunctionMarkup(r['v_end'],
                                                       r['d_start'],
                                                       r['d_end'],
                                                       r['j_start'])
        else:
            def get_junction(_): return None
        if {'cdr3aa', 'v', 'j'}.issubset(source.columns):
            if 'cdr3nt' in source.columns:
                return [ClonotypeNT(cells=get_cells(row),
                                    cdr3aa=row['cdr3aa'],
                                    v=self.segment_parser.parse(row['v']),
                                    d=self.segment_parser.parse(row['d']),
                                    j=self.segment_parser.parse(row['j']),
                                    cdr3nt=row['cdr3nt'],
                                    junction=get_junction(row),
                                    id=index)
                        for index, row in source.iterrows()]
            else:
                return [ClonotypeAA(cells=get_cells(row),
                                    cdr3aa=row['cdr3aa'],
                                    v=self.segment_parser.parse(row['v']),
                                    d=self.segment_parser.parse(row['d']),
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
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary(),
                 species: str = "HomoSapiens", gene: str = "TRB",
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
            source = source[(source['species'] == self.species)]
        if self.gene:
            source = source[(source['gene'] == self.gene)]
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
    def __init__(self,
                 lib: SegmentLibrary = SegmentLibrary()) -> None:
        super().__init__(lib)

    def parse_inner(self, source: pd.DataFrame) -> list[ClonotypeNT]:
        if len(source.columns) == 7:
            def get_junction(_): return JunctionMarkup()
        else:
            def get_junction(r): return JunctionMarkup(r[7], r[8], r[9], r[10])
        return [ClonotypeNT(cells=row[0],
                            cdr3nt=row[2],
                            cdr3aa=row[3],
                            v=self.segment_parser.parse(row[4]),
                            d=self.segment_parser.parse(row[5]),
                            j=self.segment_parser.parse(row[6]),
                            junction=get_junction(row),
                            id=index)
                for index, row in source.iterrows()]


# ---- legacy


def parse_vdjdb_slim(fname: str,
                     lib: SegmentLibrary = SegmentLibrary(),
                     species: str = "HomoSapiens", gene: str = "TRB",
                     filter: t.Callable[[pd.DataFrame],
                                        pd.DataFrame] = lambda x: x,
                     warn: bool = False,
                     n: int = None) -> list[ClonotypeAA]:
    df = pd.read_csv(fname, sep='\t', nrows=n)
    df = df[(df['species'] == species) & (df['gene'] == gene)]
    df = filter(df)
    segment_parser = SegmentParser(lib)
    res = []
    for index, row in df.iterrows():
        try:
            res.append(ClonotypeAA(cdr3aa=row['cdr3'],
                                   v=segment_parser.parse(row['v.segm']),
                                   j=segment_parser.parse(row['j.segm']),
                                   id=index,
                                   payload={'vdjdb': VdjdbPayload(row['mhc.a'],
                                                                  row['mhc.b'],
                                                                  row['mhc.class'],
                                                                  row['antigen.epitope'],
                                                                  row['antigen.species'])}))
        except Exception as e:
            if warn:
                warnings.warn(f'Error parsing VDJdb line {row} - {e}')
    return res


def parse_olga(fname: str,
               lib: SegmentLibrary = SegmentLibrary(),
               n: int = None) -> list[ClonotypeAA]:
    df = pd.read_csv(fname,
                     header=None,
                     names=['cdr3nt', 'cdr3aa', 'v', 'j'], sep='\t',
                     nrows=n)
    segment_parser = SegmentParser(lib)
    return [ClonotypeNT(cdr3nt=row['cdr3nt'],
                        cdr3aa=row['cdr3aa'],
                        v=segment_parser.parse(row['v']),
                        j=segment_parser.parse(row['j']),
                        id=index)
            for index, row in df.iterrows()]


def parse_vdjtools(fname: str,
                   lib: SegmentLibrary = SegmentLibrary(),
                   n: int = None) -> list[ClonotypeAA]:
    df = pd.read_csv(fname, sep='\t', nrows=n)
    if len(df.columns) == 7:
        def get_junction(_): return JunctionMarkup()
    else:
        def get_junction(r): return JunctionMarkup(r[7], r[8], r[9], r[10])
    segment_parser = SegmentParser(lib)
    return [ClonotypeNT(cells=row[0],
                        cdr3nt=row[2],
                        cdr3aa=row[3],
                        v=segment_parser.parse(row[4]),
                        d=segment_parser.parse(row[5]),
                        j=segment_parser.parse(row[6]),
                        junction=get_junction(row),
                        id=index)
            for index, row in df.iterrows()]


# TODO payload
def from_df(df: pd.DataFrame,
            lib: SegmentLibrary = SegmentLibrary(),
            n: int = None) -> list[ClonotypeAA]:
    if n:
        df = df.head(n)
    if {'cells'}.issubset(df.columns):
        def get_cells(r): return r['cells']
    else:
        def get_cells(_): return 1
    if {'v_end', 'd_start', 'd_end', 'j_start'}.issubset(df.columns):
        def get_junction(r): return JunctionMarkup(r['v_end'],
                                                   r['d_start'],
                                                   r['d_end'],
                                                   r['j_start'])
    else:
        def get_junction(_): return None
    segment_parser = SegmentParser(lib)
    if {'cdr3aa', 'v', 'j'}.issubset(df.columns):
        if 'cdr3nt' in df.columns:
            return [ClonotypeNT(cells=get_cells(row),
                                cdr3aa=row['cdr3aa'],
                                v=segment_parser.parse(row['v']),
                                d=segment_parser.parse(row['d']),
                                j=segment_parser.parse(row['j']),
                                cdr3nt=row['cdr3nt'],
                                junction=get_junction(row),
                                id=index)
                    for index, row in df.iterrows()]
        else:
            return [ClonotypeAA(cells=get_cells(row),
                                cdr3aa=row['cdr3aa'],
                                v=segment_parser.parse(row['v']),
                                d=segment_parser.parse(row['d']),
                                j=segment_parser.parse(row['j']),
                                id=index)
                    for index, row in df.iterrows()]
    else:
        raise ValueError(f'Critical columns missing in df {df.columns}')
