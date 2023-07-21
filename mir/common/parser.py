from collections import namedtuple
import pandas as pd
from . import ClonotypeAA, SegmentLibrary

VdjdbPayload = namedtuple('VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')

def parse_vdjdb_slim(fname : str, lib : SegmentLibrary,
                     species : str = "HomoSapiens", gene : str = "TRB") -> list[ClonotypeAA]:
    df = pd.read_csv(fname, sep = '\t')
    df = df[(df['species'] == species) & (df['gene'] == gene)]
    return [ClonotypeAA(row['cdr3'], 
                        lib.get_or_create(row['v.segm']),
                        lib.get_or_create(row['j.segm']),
                        index,
                        payload = VdjdbPayload(row['mhc.a'],
                                               row['mhc.b'],
                                               row['mhc.class'],
                                               row['antigen.epitope'],
                                               row['antigen.species'])) 
                                               for index, row in df.iterrows()]
        
