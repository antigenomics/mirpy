mir = __import__(__name__.split('.')[0])
DEFAULT_SEGMENTS = mir.get_resource_path("segments.txt")

class SegmentLibrary:
    def __init__(self, seqs : dict[str, str]):
        self.seqs = seqs

    @classmethod
    def load_default(cls,
                     genes = ["TRB"],
                     species = ["HomoSapiens"]):
        handle = mir.get_resource_path("segments.txt")
        try:
            fp = open(handle)
            lines = fp.readlines()
        finally:
            fp.close()
        header = lines[0].split()
        species_col = header.index("species")
        name_col = header.index("id")
        seq_col = header.index("seqaa")
        seqs = {}
        for line in lines[1:]:
            splitline = line.split()
            if splitline[species_col] in species and splitline[name_col][0:3] in genes:
                seqs[splitline[name_col]] = splitline[seq_col]
        return cls(seqs)
        