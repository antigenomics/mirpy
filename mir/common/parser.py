from collections import namedtuple
import pandas as pd
import warnings
from . import ClonotypeAA, ClonotypeNT, SegmentLibrary

VdjdbPayload = namedtuple('VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')

def parse_vdjdb_slim(fname : str, lib : SegmentLibrary,
                     species : str = "HomoSapiens", gene : str = "TRB") -> list[ClonotypeAA]:
    df = pd.read_csv(fname, sep = '\t')
    df = df[(df['species'] == species) & (df['gene'] == gene)]
    res = []
    for index, row in df.iterrows():
        try:
            res.append(ClonotypeAA(cdr3aa = row['cdr3'], 
                        v = lib.get_or_create(row['v.segm']),
                        j = lib.get_or_create(row['j.segm']),
                        id = index,
                        payload = VdjdbPayload(row['mhc.a'],
                                               row['mhc.b'],
                                               row['mhc.class'],
                                               row['antigen.epitope'],
                                               row['antigen.species'])))
        except Exception as e:
            warnings.warn(f'Error parsing VDJdb line {row} - {e}')
    return res


def parse_olga(fname : str, lib : SegmentLibrary) -> list[ClonotypeAA]:
    df = pd.read_csv(fname, header = None, names = ['cdr3nt', 'cdr3aa', 'v', 'j'], sep = '\t')
    return [ClonotypeNT(cdr3nt = row['cdr3nt'], 
                        cdr3aa = row['cdr3aa'], 
                        v = lib.get_or_create(row['v'] + "*01"),
                        j = lib.get_or_create(row['j'] + "*01"),
                        id = index)
                        for index, row in df.iterrows()]