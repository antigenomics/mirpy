mir = __import__(__name__.split('.')[0])
DEFAULT_SEGMENTS = mir.get_resource_path("segments.txt")


class Segment:
    def __init__(self, id : str, seqaa : str = None, seqnt : str = None):
        self.id = id
        self.gene = id[0:3]
        if "DV" in id:
            self.gene = "TRD"
        else:
            self.gene = id[0:3]
        self.type = id[3]
        self.seqaa = seqaa
        self.seqnt = seqnt

    def __repr__(self):
        if self.type == "V":
            seq = ".." + self.seqaa[-10:]
        elif self.type == "D":
            seq = "..." + self.seqnt[3:-3] + "..."
        else:
            seq = self.seqaa[:10] + ".."
        return f"{self.id}:{seq}"


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
        seqaa_col = header.index("seqaa")
        seqnt_col = header.index("seqnt")
        segments = {}
        for line in lines[1:]:
            splitline = line.split()
            if splitline[species_col] in species:
                segment = Segment(splitline[id_col], splitline[seqaa_col], splitline[seqnt_col])
                if segment.gene in genes:
                    segments[segment.id] = segment
        return cls(segments, species)
    
    def get_seqaas(self) -> list[tuple[str, str]]:
        return [(s.id, s.seqaa) for s in self.segments.values]
    
    def get_seqnts(self) -> list[tuple[str, str]]:
        return [(s.id, s.seqnt) for s in self.segments.values]
    
    def __repr__(self):
        return f"Library of {len(self.segments)} segments: " + \
            f"{[x[1] for x in self.segments.items()][:10]}"