from collections import namedtuple
from typing import Callable
import pandas as pd
import warnings

from . import JunctionMarkup, ClonotypeAA, ClonotypeNT, SegmentLibrary

VdjdbPayload = namedtuple(
    'VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')


def parse_vdjdb_slim(fname: str, lib: SegmentLibrary = SegmentLibrary(),
                     species: str = "HomoSapiens", gene: str = "TRB",
                     filter: Callable[[pd.DataFrame],
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
                        v=lib.get_or_create(row['v'] + '*01'),
                        j=lib.get_or_create(row['j'] + '*01'),
                        id=index)
            for index, row in df.iterrows()]
# todo: fix segment name TCRB -> TRB and without *01 -> *01


def from_df(df: pd.DataFrame, lib: SegmentLibrary = SegmentLibrary(), n: int = None):
    # todo: payload
    if {'cdr3aa', 'v', 'j'}.issubset(df.columns):
        if 'cdr3nt' in df.columns:
            if {'v_end', 'd_start', 'd_end', 'j_start'}.issubset(df.columns):
                return [ClonotypeNT(cdr3nt=row['cdr3nt'],
                                    cdr3aa=row['cdr3aa'],
                                    v=lib.get_or_create(row['v']),
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
