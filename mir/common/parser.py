from collections import namedtuple
import typing as t
import pandas as pd
import warnings

from . import JunctionMarkup, ClonotypeAA, ClonotypeNT, SegmentLibrary

VdjdbPayload = namedtuple(
    'VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')


def parse_vdjdb_slim(fname: str, lib: SegmentLibrary = SegmentLibrary(),
                     species: str = "HomoSapiens", gene: str = "TRB",
                     filter: t.Callable[[pd.DataFrame],
                                        pd.DataFrame] = lambda x: x,
                     warn: bool = False, n: int = None) -> list[ClonotypeAA]:
    df = pd.read_csv(fname, sep='\t', nrows=n)
    df = df[(df['species'] == species) & (df['gene'] == gene)]
    df = filter(df)
    res = []
    for index, row in df.iterrows():
        try:
            res.append(ClonotypeAA(cdr3aa=row['cdr3'],
                                   v=lib.get_or_create(row['v.segm']),
                                   j=lib.get_or_create(row['j.segm']),
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


def parse_olga(fname: str, lib: SegmentLibrary = SegmentLibrary(), n: int = None) -> list[ClonotypeAA]:
    df = pd.read_csv(fname, header=None, names=['cdr3nt', 'cdr3aa', 'v', 'j'], sep='\t',
                     nrows=n)
    return [ClonotypeNT(cdr3nt=row['cdr3nt'],
                        cdr3aa=row['cdr3aa'],
                        v=lib.get_or_create_noallele(row['v'] + '*01'),
                        j=lib.get_or_create_noallele(row['j'] + '*01'),
                        id=index)
            for index, row in df.iterrows()]


def parse_vdjtools(fname: str, lib: SegmentLibrary = SegmentLibrary(), 
                   n: int = None, mock_allele: bool = True) -> list[ClonotypeAA]:
    df = pd.read_csv(fname, sep='\t', nrows=n)

    if len(df.columns) == 7:
        get_junction = lambda _: JunctionMarkup()
    else:
        get_junction = lambda r: JunctionMarkup(r[7], r[8], r[9], r[10])

    if mock_allele:
        lgc = lambda x: lib.get_or_create_noallele(x)
    else:
        lgc = lambda x: lib.get_or_create(x)

    def lgcd(d): # vdjtools allow '' and '.' for missing D
        if len(d) < 5:
            return None
        else:
            return lgc(d)

    return [ClonotypeNT(cells=row[0],
                        cdr3nt=row[2],
                        cdr3aa=row[3],
                        v=lgc(row[4]),
                        d=lgcd(row[5]),
                        j=lgc(row[6]),
                        junction=get_junction(row),
                        id=index)
            for index, row in df.iterrows()]


def from_df(df: pd.DataFrame, lib: SegmentLibrary = SegmentLibrary(), n: int = None):
    # todo: payload
    if {'cdr3aa', 'v', 'j'}.issubset(df.columns):
        if 'cdr3nt' in df.columns:
            if {'d', 'v_end', 'd_start', 'd_end', 'j_start'}.issubset(df.columns):
                return [ClonotypeNT(cdr3nt=row['cdr3nt'],
                                    cdr3aa=row['cdr3aa'],
                                    v=lib.get_or_create(row['v']),
                                    d=lib.get_or_create(row['d']),
                                    j=lib.get_or_create(row['j']),
                                    junction=JunctionMarkup(row['v_end'],
                                                            row['d_start'],
                                                            row['d_end'],
                                                            row['j_start']),
                                    id=index)
                        for index, row in df.iterrows()]
            else:
                return [ClonotypeNT(cdr3nt=row['cdr3nt'],
                                    cdr3aa=row['cdr3aa'],
                                    v=lib.get_or_create(row['v']),
                                    j=lib.get_or_create(row['j']),
                                    id=index)
                        for index, row in df.iterrows()]
        else:
            return [ClonotypeAA(cdr3aa=row['cdr3aa'],
                                v=lib.get_or_create(row['v']),
                                j=lib.get_or_create(row['j']),
                                id=index)
                    for index, row in df.iterrows()]
    else:
        raise ValueError(f'Critical columns missing in df {df.columns}')
