import urllib
from collections import Counter
import pandas as pd

from Bio.Seq import translate

from .. import get_resource_path

_ALL_AV2DV = True
_ALLOWED_GENES = {'TRA', 'TRB', 'TRG', 'TRD', 'IGL', 'IGK', 'IGH'}
_ALLOWED_STYPE = {'V', 'D', 'J', 'C'}


class Segment:
    def __init__(self,
                 id: str,
                 organism: str = 'Unknown',
                 gene: str = None,
                 stype: str = None,
                 seqnt: str = None,
                 seqaa: str = None,
                 refpoint: int = -1,  # 0-based right after Cys or right before F/W
                 featnt: dict[str, tuple[int, int]] = {},
                 feataa: dict[str, tuple[int, int]] = {}):
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
        self.featnt = dict([(k, i)
                            for (k, i) in featnt.items() if i[1] > i[0]])
        self.feataa = dict([(k, i)
                            for (k, i) in feataa.items() if i[1] > i[0]])
        if not feataa and self.featnt:
            self.feataa = dict([(k, (i[0] // 3, i[1] // 3)) for (k, i) in
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
                 segments: dict[str, Segment] = {},
                 complete: bool = False):
        self.segments = segments
        self.complete = complete


    @classmethod
    def load_default(cls,
                         genes: set[str] = {'TRB', 'TRA'},
                         organisms: set[str] = {'HomoSapiens'},
                         fname: str = 'segments.txt'
                         ):
        df = pd.read_csv(get_resource_path(fname), sep='\t')
        segments = {}
        for _, row in df.iterrows():
            organism = row['organism']
            if organism in organisms:
                id = row['id']
                gene = row['gene']
                stype = row['stype'][0]
                seqnt = row['seqnt']
                # refpoint = int(row['refpoint'])
                # featnt = {
                #     'cdr1': (int(row['cdr1_start']), int(row['cdr1_end'])),
                #     'cdr2': (int(row['cdr2_start']), int(row['cdr2_end'])),
                #     'cdr2.5': (int(row['cdr2.5_start']), int(row['cdr2.5_end'])),
                # }
                if gene in genes:
                    segment = Segment(id=id,
                                      organism=organism,
                                      gene=gene,
                                      stype=stype,
                                      seqnt=seqnt)
                    segments[segment.id] = segment
                if _ALL_AV2DV and gene == 'TRA' and 'TRD' in genes and stype == 'V':
                    segment = Segment(id=id + 'd',
                                      organism=organism,
                                      gene='TRD',
                                      stype=stype,
                                      seqnt=seqnt)
                    segments[segment.id] = segment
        return cls(segments, True)
    @classmethod
    def load_from_imgt(cls,
                       genes: set[str] = {'TRA', 'TRB'},
                       organisms: set[str] = {'Homo_sapiens'}):
        segments = {}
        for organism in organisms:
            for gene_type in genes:
                gene_prefix = gene_type[:2]
                for segment_type in ['V', 'J']:
                    data = urllib.request.urlopen(
                        f'https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/{organism}/{gene_prefix}/{gene_type}{segment_type}.fasta')
                    fasta_parsed = data.read().decode("utf-8").replace('\n', '').split('>')
                    segments_list = [Segment(id=x.split('|')[1] + ('*01' if '*' not in x.split('|')[1] else ''),
                                        organism=organism,
                                        gene=gene_type,
                                        stype=segment_type,
                                        seqnt=x.split('|')[15].replace('.', '').upper()) for x in fasta_parsed[1:]]
                    for segment in segments_list:
                        segments[segment.id] = segment
        return cls(segments, True)

    def get_segments(self, gene: str = None, stype: str = None) -> list[Segment]:
        return [x for x in self.segments.values() if (not gene or x.gene == gene) &
                (not stype or x.stype == stype)]

    def get_seqaas(self, gene: str = None, stype: str = None) -> list[tuple[str, str]]:
        return [(s.id, s.seqaa) for s in self.get_segments(gene, stype)]

    def get_seqnts(self, gene: str = None, stype: str = None) -> list[tuple[str, str]]:
        return [(s.id, s.seqnt) for s in self.get_segments(gene, stype)]

    def get_summary(self) -> Counter[tuple[str, str, str]]:
        return Counter(((s.organism, s.gene, s.stype) for s in self.segments.values()))

    def get_organisms(self) -> set[str]:
        return {s.organism for s in self.segments.values()}

    def get_genes(self) -> set[str]:
        return {s.gene for s in self.segments.values()}

    def get_stypes(self) -> set[str]:
        return {s.stype for s in self.segments.values()}

    def __getitem__(self, id: str) -> Segment:
        return self.segments[id]

    def get_or_create(self, s: str | Segment,
                      seqaa: str = None,
                      seqnt: str = None) -> Segment:
        if isinstance(s, Segment):
            res = self.segments.get(s.id)
            if not res:
                if self.complete:
                    raise ValueError(
                        f"Segment {s} not found in a complete library")
                res = s
                self.segments[s.id] = s
        else:
            s = str(s).replace('/', '')
            res = self.segments.get(s)
            if not res:
                if self.complete:
                    raise ValueError(
                        f"Segment {s} not found in a complete library")
                res = Segment(s, seqnt=seqnt, seqaa=seqaa)
                self.segments[s] = res
        return res

    def get_or_create_noallele(self, id: str) -> Segment:
        if '*' in id:
            return self.get_or_create(id)
        else:
            return self.get_or_create(id + '*01')

    def __repr__(self):
        return f"Library of {len(self.segments)} segments: " + \
               f"{[x[1] for x in self.segments.items()][:10]}"

    def serialize(self):
        segments_serialized = []
        for segment_id, segment in self.segments.items():
            segments_serialized.append(
                [segment_id, segment.gene, segment.organism, segment.seqnt, segment.stype]
            )
        return pd.DataFrame(segments_serialized, columns=['id', 'gene', 'organism', 'seqnt', 'stype'])


_SEGMENT_CACHE = SegmentLibrary()
