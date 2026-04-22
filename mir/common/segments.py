import urllib
from collections import Counter
from pathlib import Path
import pandas as pd

from Bio.Seq import translate

from .. import get_resource_path

_ALLOWED_GENES = {'TRA', 'TRB', 'TRG', 'TRD', 'IGL', 'IGK', 'IGH'}
_ALLOWED_STYPE = {'V', 'D', 'J', 'C'}
_DEFAULT_SEGMENTS_COLUMNS = ['organism', 'gene', 'stype', 'id', 'seqnt']
_IMGT_SPECIES_ALIASES = {
    'HomoSapiens': 'Homo_sapiens',
    'MusMusculus': 'Mus_musculus',
    'MacacaMulatta': 'Macaca_mulatta',
}
_SEGMENT_ALLELE_CACHE = None


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
        genes = cls._as_set(genes)
        organisms = cls._as_set(organisms)
        default_path = Path(get_resource_path('segments/' + fname))
        df = cls._load_segments_dataframe(default_path)
        missing_pairs = cls._get_missing_vj_pairs(df=df, genes=genes, organisms=organisms)
        if missing_pairs:
            downloaded_frames = []
            for organism, gene in missing_pairs:
                lib = cls.load_from_imgt(genes={gene}, organisms={organism})
                serialized = cls.serialize_library(lib)
                if not serialized.empty:
                    downloaded_frames.append(serialized)
            if downloaded_frames:
                downloaded = cls._merge_segments_dataframes(
                    pd.DataFrame(columns=_DEFAULT_SEGMENTS_COLUMNS),
                    pd.concat(downloaded_frames, ignore_index=True),
                )
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
        return cls(segments, True)

    @classmethod
    def load_from_imgt(cls,
                       genes: set[str] = {'TRA', 'TRB'},
                       organisms: set[str] = {'Homo_sapiens'}):
        genes = cls._as_set(genes)
        organisms = cls._as_set(organisms)
        segments = {}
        for organism in organisms:
            imgt_organism = _IMGT_SPECIES_ALIASES.get(organism, organism)
            for gene_type in genes:
                gene_prefix = gene_type[:2]
                for segment_type in ['V', 'J']:
                    data = urllib.request.urlopen(
                        f'https://www.imgt.org/download/V-QUEST/IMGT_V-QUEST_reference_directory/{imgt_organism}/{gene_prefix}/{gene_type}{segment_type}.fasta')
                    fasta_parsed = data.read().decode("utf-8").replace('\n', '').split('>')
                    segments_list = [Segment(id=x.split('|')[1] + ('*01' if '*' not in x.split('|')[1] else ''),
                                        organism=organism,
                                        gene=gene_type,
                                        stype=segment_type,
                                        seqnt=x.split('|')[15].replace('.', '').upper()) for x in fasta_parsed[1:] if x]
                    for segment in segments_list:
                        segments[segment.id] = segment
        return cls(segments, True)

    @staticmethod
    def _as_set(values) -> set[str]:
        if isinstance(values, str):
            return {values}
        return set(values)

    @staticmethod
    def _allele_sort_key(segment_id: str) -> tuple[int, str]:
        allele = str(segment_id).split('*', 1)[1]
        try:
            return (int(allele), allele)
        except ValueError:
            return (10**9, allele)

    @classmethod
    def _get_resource_allele_cache(cls) -> dict[tuple[str, str, str], str]:
        global _SEGMENT_ALLELE_CACHE
        if _SEGMENT_ALLELE_CACHE is not None:
            return _SEGMENT_ALLELE_CACHE

        path = Path(get_resource_path('segments/segments.txt'))
        df = cls._load_segments_dataframe(path)
        cache = {}
        df = df[df['id'].astype(str).str.contains(r'\*', regex=True)]
        for _, row in df.iterrows():
            segment_id = str(row['id'])
            base_id = segment_id.split('*', 1)[0]
            key = (str(row['organism']), str(row['gene']), base_id)
            current = cache.get(key)
            if current is None or cls._allele_sort_key(segment_id) < cls._allele_sort_key(current):
                cache[key] = segment_id
        _SEGMENT_ALLELE_CACHE = cache
        return cache

    @classmethod
    def _resolve_allele_id(cls, segment_id: str, organism: str = None, gene: str = None) -> str:
        segment_id = str(segment_id).strip()
        if '*' in segment_id:
            return segment_id

        if gene is None and len(segment_id) >= 4:
            gene = segment_id[:3]
        if organism is not None and gene is not None:
            cached = cls._get_resource_allele_cache().get((organism, gene, segment_id))
            if cached is not None:
                return cached
        return segment_id + '*01'

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
            s = str(s).strip()
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
            for organism in sorted(self.get_organisms()):
                gene = id[:3] if len(id) >= 3 else None
                resolved = self._resolve_allele_id(id, organism=organism, gene=gene)
                if resolved in self.segments:
                    return self.get_or_create(resolved)
            # Cache miss — find the minimum available allele with this base name.
            candidates = sorted(s for s in self.segments if s.startswith(id + "*"))
            if candidates:
                return self.get_or_create(candidates[0])
            return self.get_or_create(self._resolve_allele_id(id))

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
