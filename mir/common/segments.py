mir = __import__(__name__.split('.')[0])
DEFAULT_SEGMENTS = mir.get_resource_path("segments.txt")

from typing import Iterable
from Bio.Seq import Seq

class Segment:
    def __init__(self, 
                 id : str, 
                 gene : str = None,
                 stype : str = None,
                 seqnt : str = None,
                 refpoint : int = -1, # 0-based right after Cys or right before F/W
                 seqaa : str = None):
        # todo: cdrs
        self.id = id
        if not gene:
            self.gene = id[0:3]
        else:
            self.gene = gene
        if not stype:
            self.stype = id[3]
        else:
            self.stype = stype
        self.seqnt = seqnt
        if not seqaa:
            if seqnt:
                if stype == 'J':
                    offset = (refpoint + 1) % 3
                    ss = seqnt[offset:]
                else:
                    ss = seqnt                
                trim = len(ss) % 3
                self.seqaa = str(Seq.translate(ss[:len(ss) - trim]))
        else:
            self.seqaa = seqaa
        self.refpoint = refpoint

    def __repr__(self):
        if self.stype == "V":
            seq = ".." + self.seqaa[-10:]
        elif self.stype == "D":
            seq = "_" + self.seqaa + "_"
        else:
            seq = self.seqaa[:10] + ".."
        return f"{self.id}:{self.refpoint}:{seq}"


class Library:
    def __init__(self, 
                 segments : dict[str, Segment] = {},
                 species : list[str] = None):
        self.segments = segments
        self.species = species

    def get_or_create(self, s : str | Segment, seqaa : str = None, seqnt : str = None) -> Segment:        
        if type(s) == "str":
            res = self.segments[s]
            if not res:
                res = Segment(s, seqaa, seqnt)
                self.segments[s] = res
            return res
        else:
            res = self.segments[s.id]
            if not res:
                res = s
                self.segments[s.id] = s
            return res
        
    @classmethod
    def load_default(cls,
                     genes = ["TRB"],
                     species = ["HomoSapiens"],
                     fname = "segments.txt"):
        try:
            file = open(mir.get_resource_path(fname))
            lines = file.readlines()
        finally:
            file.close()
        header = lines[0].split()
        species_col = header.index("species")
        id_col = header.index("id")
        gene_col = header.index("gene")
        stype_col = header.index("segment")
        seqnt_col = header.index("sequence")        
        refpoint_col = header.index("reference_point")
        segments = {}
        for line in lines[1:]:
            splitline = line.split()
            if splitline[species_col] in species:
                gene = splitline[gene_col]
                if gene in genes:
                    segment = Segment(splitline[id_col], gene,
                                      splitline[stype_col][0], 
                                      splitline[seqnt_col],
                                      int(splitline[refpoint_col]))                
                    segments[segment.id] = segment
        return cls(segments, species)
    
    def get_seqaas(self) -> Iterable[tuple[str, str]]:
        return ((s.id, s.seqaa) for s in self.segments.values())
    
    def get_seqnts(self) -> Iterable[tuple[str, str]]:
        return ((s.id, s.seqnt) for s in self.segments.values())
    
    def __repr__(self):
        return f"Library of {len(self.segments)} segments: " + \
            f"{[x[1] for x in self.segments.items()][:10]}"