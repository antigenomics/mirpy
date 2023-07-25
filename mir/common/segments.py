from collections import Counter
from Bio.Seq import translate
mir = __import__(__name__.split('.')[0])


_ALL_AV2DV = True
_ALLOWED_GENES = {'TRA', 'TRB', 'TRG', 'TRD', 'IGL', 'IGK', 'IGH'}
_ALLOWED_STYPE = {'V', 'D', 'J', 'C'}


class Segment:
    def __init__(self, 
                 id : str,
                 organism : str = 'Unknown',
                 gene : str = None,
                 stype : str = None,
                 seqnt : str = None,
                 seqaa : str = None,
                 refpoint : int = -1, # 0-based right after Cys or right before F/W
                 featnt : dict[str, tuple[int, int]] = {},
                 feataa : dict[str, tuple[int, int]] = {}):
        self.id = id
        self.organism = organism
        if not gene:
            self.gene = id[0:3]
        else:
            self.gene = gene
        if not self.gene in _ALLOWED_GENES:
            raise ValueError(f'Bad gene {self.gene}')
        if not stype:
            self.stype = id[3]
        else:
            self.stype = stype
        if not self.stype in _ALLOWED_STYPE:
            raise ValueError(f'Bad segment type {self.stype}')
        self.seqnt = seqnt
        if not seqaa and self.seqnt:
                if stype == 'J':
                    offset = (refpoint + 1) % 3
                    ss = seqnt[offset:]
                else:
                    ss = seqnt                
                trim = len(ss) % 3
                self.seqaa = translate(ss[:len(ss) - trim])
        else:
            self.seqaa = seqaa
        self.refpoint = refpoint
        self.featnt = dict([(k, i) for (k, i) in featnt.items() if i[1] > i[0]])
        self.feataa = dict([(k, i) for (k, i) in feataa.items() if i[1] > i[0]])
        if not feataa and self.featnt:
            self.feataa = dict([(k, (i[0]//3, i[1]//3)) for (k, i) in 
                       featnt.items()])
            
    def __str__(self):
        return self.id

    def __repr__(self):
        if self.seqaa:
            if self.stype == 'V':
                seq = ".." + self.seqaa[-10:]
            elif self.stype == 'D':
                seq = "_" + self.seqaa + "_"
            else:
                seq = self.seqaa[:10] + ".."
        else:
            seq = "?"
        return f"{self.organism} {self.id}:{self.refpoint}:{seq}"


class SegmentLibrary:
    def __init__(self, 
                 segments : dict[str, Segment] = {},
                 complete : bool = False):
        self.segments = segments
        self.complete = complete
        
    @classmethod
    def load_default(cls,
                     genes : set[str] = {'TRB'},
                     organisms : set[str] = {'HomoSapiens'},
                     fname : str = "segments.txt"):
        try:
            file = open(mir.get_resource_path(fname))
            lines = file.readlines()
        finally:
            file.close()
        header = lines[0].split()
        organism_col = header.index('organism')
        id_col = header.index('id')
        gene_col = header.index('gene')
        stype_col = header.index('stype')
        seqnt_col = header.index('seqnt')        
        refpoint_col = header.index('refpoint')
        cdr1_start_col = header.index('cdr1_start')
        cdr1_end_col = header.index('cdr1_end')
        cdr2_start_col = header.index('cdr2_start')
        cdr2_end_col = header.index('cdr2_end')
        cdr25_start_col = header.index('cdr2.5_start')
        cdr25_end_col = header.index('cdr2.5_end')
        segments = {}
        for line in lines[1:]:
            splitline = line.split()
            organism = splitline[organism_col]
            if organism in organisms:
                id = splitline[id_col]
                gene = splitline[gene_col]
                stype = splitline[stype_col][0]
                seqnt = splitline[seqnt_col]
                refpoint = int(splitline[refpoint_col])
                featnt = {
                    'cdr1': (int(splitline[cdr1_start_col]), int(splitline[cdr1_end_col])),
                    'cdr2': (int(splitline[cdr2_start_col]), int(splitline[cdr2_end_col])),
                    'cdr2.5': (int(splitline[cdr25_start_col]), int(splitline[cdr25_end_col])),
                    }
                if gene in genes:  
                    segment = Segment(id=id,
                                      organism=organism,
                                      gene=gene,
                                      stype=stype,
                                      seqnt=seqnt,
                                      refpoint=refpoint,
                                      featnt=featnt)              
                    segments[segment.id] = segment
                if _ALL_AV2DV and gene == 'TRA' and 'TRD' in genes and stype == 'V':
                    segment = Segment(id=id + 'd',
                                      organism=organism,
                                      gene='TRD',
                                      stype=stype,
                                      seqnt=seqnt,
                                      refpoint=refpoint,
                                      featnt=featnt)              
                    segments[segment.id] = segment
        return cls(segments, True)
    
    def get_segments(self, gene : str = None, stype : str = None) -> list[Segment]:
        return [x for x in self.segments.values() if (not gene or x.gene == gene) & 
                      (not stype or x.stype == stype)]
    
    def get_seqaas(self, gene : str = None, stype : str = None) -> list[tuple[str, str]]:
        return [(s.id, s.seqaa) for s in self.get_segments(gene, stype)]
    
    def get_seqnts(self, gene : str = None, stype : str = None) -> list[tuple[str, str]]:
        return [(s.id, s.seqnt) for s in self.get_segments(gene, stype)]

    def get_summary(self) -> Counter[tuple[str, str, str]]:
        return Counter(((s.organism, s.gene, s.stype) for s in self.segments.values()))
    
    def get_organisms(self) -> set[str]:
        return {s.organism for s in self.segments.values()}
    
    def get_genes(self) -> set[str]:
        return {s.gene for s in self.segments.values()}
    
    def get_stypes(self) -> set[str]:
        return {s.stype for s in self.segments.values()}
    
    def __getitem__(self, id : str) -> Segment:
        return self.segments[id]
    
    def get_or_create(self, s : str | Segment,
                      seqaa : str = None, 
                      seqnt : str = None) -> Segment:                
        if isinstance(s, Segment):
            res = self.segments.get(s.id)
            if not res:
                if self.complete:
                    raise ValueError(f'Segment {s} not found in a complete library')
                res = s
                self.segments[s.id] = s
        else:
            s = str(s)
            res = self.segments.get(s)
            if not res:
                if self.complete:
                    raise ValueError(f'Segment {s} not found in a complete library')
                res = Segment(s, seqnt = seqnt, seqaa = seqaa)
                self.segments[s] = res
        return res
    
    def __repr__(self):
        return f"Library of {len(self.segments)} segments: " + \
            f"{[x[1] for x in self.segments.items()][:10]}"
    

_SEGMENT_CACHE = SegmentLibrary()