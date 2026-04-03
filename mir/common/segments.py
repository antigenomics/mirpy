import urllib
from collections import Counter
from pathlib import Path
import pandas as pd

from Bio.Seq import translate

from .. import get_resource_path

_ALL_AV2DV = True
_ALLOWED_GENES = {'TRA', 'TRB', 'TRG', 'TRD', 'IGL', 'IGK', 'IGH'}
_ALLOWED_STYPE = {'V', 'D', 'J', 'C'}
_DEFAULT_SEGMENTS_COLUMNS = ['organism', 'gene', 'stype', 'id', 'seqnt']
_IMGT_SPECIES_ALIASES = {
    'HomoSapiens': 'Homo_sapiens',
    'MusMusculus': 'Mus_musculus',
    'MacacaMulatta': 'Macaca_mulatta',
}
_IMGT_SPECIES_REVERSE_ALIASES = {v: k for k, v in _IMGT_SPECIES_ALIASES.items()}


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
        default_path = Path(get_resource_path(fname))
        df = cls._load_segments_dataframe(default_path)
        missing_pairs = cls._get_missing_vj_pairs(df=df, genes=genes, organisms=organisms)
        if missing_pairs:
            downloaded = cls._download_missing_segments(missing_pairs)
            if not downloaded.empty:
                df = cls._merge_segments_dataframes(df, downloaded)
                cls._write_segments_dataframe(df, default_path)
        segments = {}
        for _, row in df.iterrows():
            organism = row['organism']
            if organism in organisms:
                id = row['id']
                gene = row['gene']
                stype = row['stype'][0]
                seqnt = row['seqnt']
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
            imgt_organism = cls._to_imgt_organism(organism)
            mir_organism = cls._from_imgt_organism(imgt_organism)
            for gene_type in genes:
                gene_prefix = gene_type[:2]
                for segment_type in ['V', 'J']:
                    data = urllib.request.urlopen(
                        f'https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/{imgt_organism}/{gene_prefix}/{gene_type}{segment_type}.fasta')
                    fasta_parsed = data.read().decode("utf-8").replace('\n', '').split('>')
                    segments_list = [Segment(id=x.split('|')[1] + ('*01' if '*' not in x.split('|')[1] else ''),
                                        organism=mir_organism,
                                        gene=gene_type,
                                        stype=segment_type,
                                        seqnt=x.split('|')[15].replace('.', '').upper()) for x in fasta_parsed[1:] if x]
                    for segment in segments_list:
                        segments[segment.id] = segment
        return cls(segments, True)

    @staticmethod
    def _to_imgt_organism(organism: str) -> str:
        return _IMGT_SPECIES_ALIASES.get(organism, organism)

    @staticmethod
    def _from_imgt_organism(organism: str) -> str:
        return _IMGT_SPECIES_REVERSE_ALIASES.get(organism, organism.replace('_', ''))

    @classmethod
    def _load_segments_dataframe(cls, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t')
        return cls._normalize_segments_dataframe(df)

    @classmethod
    def _normalize_segments_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        missing_columns = [c for c in _DEFAULT_SEGMENTS_COLUMNS if c not in normalized.columns]
        if missing_columns:
            raise ValueError(f'Missing segment columns: {missing_columns}')
        normalized = normalized[_DEFAULT_SEGMENTS_COLUMNS]
        normalized['organism'] = normalized['organism'].astype(str)
        normalized['gene'] = normalized['gene'].astype(str)
        normalized['stype'] = normalized['stype'].astype(str)
        normalized['id'] = normalized['id'].astype(str)
        normalized['seqnt'] = normalized['seqnt'].astype(str).str.upper()
        normalized = normalized.drop_duplicates(subset=['organism', 'gene', 'id'])
        return normalized

    @classmethod
    def _merge_segments_dataframes(cls, current: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
        merged = pd.concat([current, extra], ignore_index=True)
        merged = cls._normalize_segments_dataframe(merged)
        return merged.sort_values(['organism', 'gene', 'stype', 'id']).reset_index(drop=True)

    @classmethod
    def _write_segments_dataframe(cls, df: pd.DataFrame, path: Path) -> None:
        cls._normalize_segments_dataframe(df).to_csv(path, sep='\t', index=False)

    @classmethod
    def _get_missing_vj_pairs(cls, df: pd.DataFrame, genes: set[str], organisms: set[str]) -> list[tuple[str, str]]:
        missing = []
        requested = {(organism, gene) for organism in organisms for gene in genes}
        for organism, gene in sorted(requested):
            subset = df[(df['organism'] == organism) & (df['gene'] == gene)]
            stypes = set(subset['stype'].astype(str).str[0])
            if 'V' not in stypes or 'J' not in stypes:
                missing.append((organism, gene))
        return missing

    @classmethod
    def _download_missing_segments(cls, missing_pairs: list[tuple[str, str]]) -> pd.DataFrame:
        if not missing_pairs:
            return pd.DataFrame(columns=_DEFAULT_SEGMENTS_COLUMNS)
        downloaded_frames = []
        for organism, gene in missing_pairs:
            lib = cls.load_from_imgt(genes={gene}, organisms={organism})
            serialized = cls.serialize_library(lib)
            if not serialized.empty:
                downloaded_frames.append(serialized)
        if not downloaded_frames:
            return pd.DataFrame(columns=_DEFAULT_SEGMENTS_COLUMNS)
        return cls._merge_segments_dataframes(
            pd.DataFrame(columns=_DEFAULT_SEGMENTS_COLUMNS),
            pd.concat(downloaded_frames, ignore_index=True),
        )

    @staticmethod
    def serialize_library(lib: 'SegmentLibrary') -> pd.DataFrame:
        rows = []
        for segment in lib.segments.values():
            stype_name = {'V': 'Variable', 'J': 'Joining', 'D': 'Diversity', 'C': 'Constant'}.get(segment.stype, segment.stype)
            rows.append([segment.organism, segment.gene, stype_name, segment.id, segment.seqnt])
        return pd.DataFrame(rows, columns=_DEFAULT_SEGMENTS_COLUMNS)

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
