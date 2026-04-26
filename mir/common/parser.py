"""Clonotype table parsers and AIRR writer.

Parsers
-------
* :class:`ClonotypeTableParser` — generic VDJtools / AIRR table parser
  (returns ``list[Clonotype]``).
* :class:`VDJtoolsParser` — VDJtools-format tables.
* :class:`AIRRParser` — AIRR-format tables.
* :class:`OldMiXCRParser` — legacy MiXCR clone tables → :class:`SampleRepertoire`.
* :class:`VDJdbSlimParser` — VDJdb slim export → :class:`SampleRepertoire`.
* :class:`OlgaParser` — OLGA sequence generation output → :class:`SampleRepertoire`.

Writer
------
* :class:`AIRRWriter` — serialise a repertoire to a tab-separated AIRR file.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import re
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Union

# VDJdb slim files can contain fields larger than the default 131 072-byte limit
# (e.g. long reference.id lists).  Set the limit as high as the platform allows.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:          # 32-bit Windows: sys.maxsize overflows C long
    csv.field_size_limit(2**31 - 1)

import pandas as pd
import polars as pl

from mir.basic.alphabets import back_translate
from mir.common.clonotype import Clonotype, ClonotypeAA, ClonotypeNT, JunctionMarkup
from mir.common.gene_library import GeneLibrary, GeneEntry
from mir.common.repertoire import SampleRepertoire, LocusRepertoire

# Mapping from VDJtools / legacy column names to AIRR column names.
# Kept public because downstream test code imports it.
_VDJTOOLS_TO_AIRR: dict[str, str] = {
    'count':    'duplicate_count',
    '#count':   'duplicate_count',
    'cdr3nt':   'junction',
    'cdr3aa':   'junction_aa',
    'v':        'v_gene',
    'd':        'd_gene',
    'j':        'j_gene',
    'VEnd':     'v_sequence_end',
    'DStart':   'd_sequence_start',
    'DEnd':     'd_sequence_end',
    'JStart':   'j_sequence_start',
}

_AIRR_LOCUS_ALIASES: dict[str, set[str]] = {
    'alpha':  {'alpha', 'tra'},
    'beta':   {'beta',  'trb'},
    'gamma':  {'gamma', 'trg'},
    'delta':  {'delta', 'trd'},
    'heavy':  {'heavy', 'igh'},
    'kappa':  {'kappa', 'igk'},
    'lambda': {'lambda','igl'},
}

# Backward-compat namedtuple kept as a public export.
VdjdbPayload = namedtuple('VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')


def _gene_str(val) -> str:
    """Normalise a gene field value to a plain string."""
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return ""
    s = str(val).strip().split(',')[0].split(';')[0]
    return s if s not in ('.', '') else ""


# ---------------------------------------------------------------------------
# SegmentParser (resolves raw gene strings into GeneEntry objects)
# ---------------------------------------------------------------------------

class SegmentParser:
    """Resolve raw V/J gene strings from a parser row into :class:`GeneEntry` objects."""

    def __init__(
        self,
        lib: GeneLibrary,
        select_most_probable: bool = True,
        mock_allele: bool = True,
        remove_allele: bool = False,
    ) -> None:
        self.lib = lib
        self.mock_allele = mock_allele
        self.remove_allele = remove_allele
        self.select_most_probable = select_most_probable

    def parse(self, id: str) -> GeneEntry | None:
        """Resolve *id* to a :class:`GeneEntry`, or ``None`` if unparseable."""
        id = id.strip()
        if pd.isna(id) or len(id) < 5:
            return None
        if self.select_most_probable:
            id = id.split(',')[0]
            id = id.split(';')[0]
        if self.remove_allele:
            id = id.split('*', 1)[0]
        if self.mock_allele:
            return self.lib.get_or_create_noallele(id)
        return self.lib.get_or_create(id)


# ---------------------------------------------------------------------------
# ClonotypeTableParser — generic (VDJtools / AIRR column names)
# ---------------------------------------------------------------------------

class ClonotypeTableParser:
    """Parse clonotype tables into lists of :class:`Clonotype` objects.

    Accepts both file paths (string) and pre-loaded :class:`pd.DataFrame`.
    Column names are normalised via :meth:`normalize_df` before parsing so
    that both VDJtools-style and AIRR-style inputs are handled transparently.
    """

    def __init__(
        self,
        lib: GeneLibrary = GeneLibrary(),
        sep: str = '\t',
    ) -> None:
        self.segment_parser = SegmentParser(lib)
        self.sep = sep

    @staticmethod
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names: strip leading ``#``, map VDJtools → AIRR."""
        df = df.copy()
        df.columns = [c.lstrip('#') for c in df.columns]
        return df.rename(columns=_VDJTOOLS_TO_AIRR)

    def parse(
        self,
        source: str | pd.DataFrame,
        n: int | None = None,
        sample: bool = False,
    ) -> list[Clonotype]:
        """Parse *source* (path or DataFrame) into a list of clonotypes."""
        if isinstance(source, str):
            if n is None or not sample:
                source = pd.read_csv(source, sep=self.sep, nrows=n)
            else:
                source = pd.read_csv(source, sep=self.sep).sample(n=n, random_state=42)
            source = self.normalize_df(source)
        else:
            if not sample:
                source = source.head(n)
            elif sample and n is not None:
                source = source.sample(n=n, random_state=42)
        return self.parse_inner(source)

    def parse_inner(self, source: pd.DataFrame) -> list[Clonotype]:
        """Parse an already-loaded DataFrame with AIRR-normalised column names."""
        cols = set(source.columns)
        if 'junction_aa' not in cols or 'v_gene' not in cols or 'j_gene' not in cols:
            raise ValueError(
                f'Critical columns missing in df {list(source.columns)}. '
                f'Expected junction_aa, v_gene, j_gene (AIRR names) or '
                f'cdr3aa, v, j (VDJtools names — pass through normalize_df first).')
        has_markup = {'v_sequence_end', 'd_sequence_start',
                      'd_sequence_end', 'j_sequence_start'}.issubset(cols)
        clonotypes = []
        for index, row in source.iterrows():
            clonotypes.append(Clonotype(_validate=False,
                sequence_id=str(index),
                duplicate_count=int(row['duplicate_count']) if 'duplicate_count' in cols else 1,
                junction=str(row['junction']) if 'junction' in cols else "",
                junction_aa=str(row['junction_aa']),
                v_gene=_gene_str(row.get('v_gene')),
                d_gene=_gene_str(row.get('d_gene')) if 'd_gene' in cols else "",
                j_gene=_gene_str(row.get('j_gene')),
                v_sequence_end=int(row['v_sequence_end']) if has_markup else -1,
                d_sequence_start=int(row['d_sequence_start']) if has_markup else -1,
                d_sequence_end=int(row['d_sequence_end']) if has_markup else -1,
                j_sequence_start=int(row['j_sequence_start']) if has_markup else -1,
            ))
        return clonotypes


# ---------------------------------------------------------------------------
# VDJtoolsParser
# ---------------------------------------------------------------------------

class VDJtoolsParser(ClonotypeTableParser):
    """Parse VDJtools output files.

    The raw VDJtools header uses ``#count`` (or ``count``), ``cdr3nt``,
    ``cdr3aa``, ``v``, ``d``, ``j``, ``VEnd``, ``DStart``, ``DEnd``,
    ``JStart`` — these are automatically mapped via :meth:`normalize_df`.
    """

    def __init__(
        self,
        lib: GeneLibrary = GeneLibrary(),
        sep: str = '\t',
    ) -> None:
        super().__init__(lib, sep)

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        cols = set(df.columns)
        has_markup = {'v_sequence_end', 'd_sequence_start',
                      'd_sequence_end', 'j_sequence_start'}.issubset(cols)
        clonotypes = []
        for index, row in df.iterrows():
            clonotypes.append(Clonotype(_validate=False,
                sequence_id=str(index),
                duplicate_count=int(row['duplicate_count']),
                junction=str(row['junction']),
                junction_aa=str(row['junction_aa']),
                v_gene=_gene_str(row.get('v_gene')),
                d_gene=_gene_str(row.get('d_gene')) if 'd_gene' in cols else "",
                j_gene=_gene_str(row.get('j_gene')),
                v_sequence_end=int(row['v_sequence_end']) if has_markup else -1,
                d_sequence_start=int(row['d_sequence_start']) if has_markup else -1,
                d_sequence_end=int(row['d_sequence_end']) if has_markup else -1,
                j_sequence_start=int(row['j_sequence_start']) if has_markup else -1,
            ))
        return clonotypes


# ---------------------------------------------------------------------------
# AIRRParser
# ---------------------------------------------------------------------------

class AIRRParser(ClonotypeTableParser):
    """Parse AIRR-format files.

    Mandatory columns: ``locus``, ``v_call``, ``j_call``, ``junction_aa``.
    """

    def __init__(
        self,
        lib: GeneLibrary = GeneLibrary(),
        sep: str = '\t',
        locus: str = 'beta',
    ) -> None:
        super().__init__(lib, sep)
        self.locus = locus
        self.mandatory_columns = ['locus', 'v_call', 'j_call', 'junction_aa']

    def get_locus_aliases(self) -> set[str]:
        locus_normalized = str(self.locus).strip().lower()
        if locus_normalized in _AIRR_LOCUS_ALIASES:
            return _AIRR_LOCUS_ALIASES[locus_normalized]
        for aliases in _AIRR_LOCUS_ALIASES.values():
            if locus_normalized in aliases:
                return aliases
        return {locus_normalized}

    def validate_columns(self, df: pd.DataFrame) -> None:
        for col in self.mandatory_columns:
            if col not in df.columns:
                raise KeyError(
                    f'Mandatory column {col!r} is missing! '
                    f'Mandatory columns: {self.mandatory_columns}')

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        self.validate_columns(df)
        locus_aliases = self.get_locus_aliases()
        df = df[df.locus.astype(str).str.strip().str.lower().isin(locus_aliases)]
        clonotypes = []
        for i, row in df.iterrows():
            try:
                v = _gene_str(row.get('v_call'))
                j = _gene_str(row.get('j_call'))
                if not v or not j:
                    raise ValueError(f'Missing v_call or j_call in row {i}')
                seq_id = str(row['clone_id']) if 'clone_id' in df.columns else str(i)
                clonotypes.append(Clonotype(_validate=False,
                    sequence_id=seq_id,
                    locus=str(row.get('locus', '')),
                    junction_aa=str(row['junction_aa']),
                    v_gene=v,
                    j_gene=j,
                ))
            except Exception as e:
                logging.warning(f"Error parsing row {i + 1}: {e}")
        return clonotypes


# ---------------------------------------------------------------------------
# OldMiXCRParser — legacy MiXCR clone tables → SampleRepertoire
# ---------------------------------------------------------------------------

class OldMiXCRParser:
    """Parse legacy MiXCR clone tables (tab-delimited, optionally gzipped).

    Produces a :class:`~mir.common.repertoire.SampleRepertoire` whose
    :class:`~mir.common.repertoire.LocusRepertoire` entries are split by
    locus.  Locus is inferred automatically from the J-gene prefix
    (``TRBJ…`` → ``"TRB"``, ``TRAJ…`` → ``"TRA"``, etc.).

    Field mapping
    -------------
    ``Clone count``        → ``duplicate_count``
    ``All V/D/J/C hits``   → gene fields (first hit, ``*00`` → ``*01``)
    ``N. Seq. CDR3``       → ``junction``
    ``AA. Seq. CDR3``      → ``junction_aa``
    ``Ref. points[11]``    → ``v_sequence_end``
    ``Ref. points[12]``    → ``d_sequence_start``
    ``Ref. points[15]``    → ``d_sequence_end``
    ``Ref. points[16]``    → ``j_sequence_start``
    """

    _COL_CLONE_ID    = 0
    _COL_COUNT       = 1
    _COL_V_HITS      = 5
    _COL_D_HITS      = 6
    _COL_J_HITS      = 7
    _COL_C_HITS      = 8
    _COL_JUNCTION    = 23   # N. Seq. CDR3
    _COL_JUNCTION_AA = 32   # AA. Seq. CDR3
    _COL_REF_POINTS  = 34   # Ref. points

    _RP_V_END   = 11
    _RP_D_START = 12
    _RP_D_END   = 15
    _RP_J_START = 16

    @staticmethod
    def _parse_gene(hits: str) -> str:
        """Return the first gene hit, normalising ``*00`` allele to ``*01``."""
        if not hits:
            return ""
        return re.sub(r"\*00.*", "*01", hits.split(",")[0])

    @staticmethod
    def _ref_point(parts: list[str], idx: int) -> int:
        """Return the integer ref-point at *idx*, or ``-1`` when absent."""
        if idx < len(parts) and parts[idx]:
            return int(parts[idx])
        return -1

    def parse_file(self, path: str | Path, sample_id: str = "") -> SampleRepertoire:
        """Parse *path* and return a :class:`~mir.common.repertoire.SampleRepertoire`.

        Parameters
        ----------
        path:
            Path to a legacy MiXCR clone table (plain or gzipped ``.gz``).
        sample_id:
            Identifier for the resulting sample.  Defaults to the file stem
            with ``.txt`` / ``.gz`` suffixes stripped.
        """
        path = Path(path)
        if not sample_id:
            stem = path.stem
            if stem.endswith(".txt"):
                stem = stem[:-4]
            sample_id = stem

        opener = gzip.open if path.suffix == ".gz" else open
        clonotypes: list[Clonotype] = []

        with opener(path, "rt", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            next(reader, None)  # skip header
            for row in reader:
                if len(row) <= self._COL_REF_POINTS:
                    continue
                junction_aa = row[self._COL_JUNCTION_AA].strip()
                if not junction_aa:
                    continue
                ref = row[self._COL_REF_POINTS].split(":")
                clonotypes.append(Clonotype(_validate=False,
                    sequence_id=      row[self._COL_CLONE_ID],
                    duplicate_count=  int(row[self._COL_COUNT]),
                    junction=         row[self._COL_JUNCTION],
                    junction_aa=      junction_aa,
                    v_gene=           self._parse_gene(row[self._COL_V_HITS]),
                    d_gene=           self._parse_gene(row[self._COL_D_HITS]),
                    j_gene=           self._parse_gene(row[self._COL_J_HITS]),
                    c_gene=           self._parse_gene(row[self._COL_C_HITS]),
                    v_sequence_end=   self._ref_point(ref, self._RP_V_END),
                    d_sequence_start= self._ref_point(ref, self._RP_D_START),
                    d_sequence_end=   self._ref_point(ref, self._RP_D_END),
                    j_sequence_start= self._ref_point(ref, self._RP_J_START),
                ))

        return SampleRepertoire.from_clonotypes(clonotypes, sample_id=sample_id)


# ---------------------------------------------------------------------------
# VDJdbSlimParser — VDJdb slim export → SampleRepertoire
# ---------------------------------------------------------------------------

class VDJdbSlimParser:
    """Parse VDJdb slim export files (tab-delimited, optionally gzipped).

    Returns a :class:`~mir.common.repertoire.SampleRepertoire` grouped by
    locus (``gene`` column).

    Field mapping
    -------------
    ``gene``              → ``locus``
    ``cdr3``              → ``junction_aa``
    (back-translated)     → ``junction``  (most-likely human codon per AA)
    ``v.segm``            → ``v_gene``
    ``j.segm``            → ``j_gene``
    ``v.end * 3``         → ``v_sequence_end``  (AA→NT position)
    ``j.start * 3``       → ``j_sequence_start``
    metadata columns      → ``clone_metadata`` dict keys

    The following fields are stored in each clonotype's ``clone_metadata`` dict:
    ``mhc.a``, ``mhc.b``, ``mhc.class``, ``antigen.species``,
    ``antigen.gene``, ``antigen.epitope``.
    """

    _METADATA_COLS = (
        "mhc.a", "mhc.b", "mhc.class",
        "antigen.species", "antigen.gene", "antigen.epitope",
    )

    def parse_file(
        self,
        path: str | Path,
        sample_id: str = "",
        species: str = "",
    ) -> SampleRepertoire:
        """Parse *path* and return a :class:`~mir.common.repertoire.SampleRepertoire`.

        Parameters
        ----------
        path:
            Path to a VDJdb slim export (plain or gzipped ``.gz``).
        sample_id:
            Identifier for the resulting sample.
        species:
            When non-empty, keep only rows whose ``species`` column matches.
        """
        path = Path(path)
        if not sample_id:
            stem = path.stem
            if stem.endswith(".txt") or stem.endswith(".tsv"):
                stem = Path(stem).stem
            sample_id = stem

        opener = gzip.open if path.suffix == ".gz" else open
        clonotypes: list[Clonotype] = []

        with opener(path, "rt", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                if species and row.get("species", "") != species:
                    continue
                junction_aa = row.get("cdr3", "").strip()
                if not junction_aa:
                    continue

                try:
                    v_end_aa = int(row.get("v.end", -1))
                    v_sequence_end = v_end_aa * 3 if v_end_aa >= 0 else -1
                except (ValueError, TypeError):
                    v_sequence_end = -1

                try:
                    j_start_aa = int(row.get("j.start", -1))
                    j_sequence_start = j_start_aa * 3 if j_start_aa >= 0 else -1
                except (ValueError, TypeError):
                    j_sequence_start = -1

                clone = Clonotype(_validate=False,
                    junction_aa=      junction_aa,
                    junction=         back_translate(junction_aa),
                    locus=            row.get("gene", "").strip(),
                    v_gene=           row.get("v.segm", "").strip(),
                    j_gene=           row.get("j.segm", "").strip(),
                    v_sequence_end=   v_sequence_end,
                    j_sequence_start= j_sequence_start,
                )
                clone.clone_metadata.update({
                    col: row.get(col, "") for col in self._METADATA_COLS
                })
                clonotypes.append(clone)

        return SampleRepertoire.from_clonotypes(clonotypes, sample_id=sample_id)


# ---------------------------------------------------------------------------
# OlgaParser — OLGA sequence generation output → SampleRepertoire
# ---------------------------------------------------------------------------

class OlgaParser:
    """Parse OLGA-generated sequence files (tab-delimited, optionally gzipped).

    The file has **no header row**.  Columns (in order):
    ``junction``, ``junction_aa``, ``v_gene``, ``j_gene``.

    Locus is inferred from the J-gene prefix (e.g. ``TRBJ…`` → ``"TRB"``)
    via :class:`~mir.common.clonotype.Clonotype`'s ``__post_init__``.
    """

    def parse_file(
        self,
        path: str | Path,
        sample_id: str = "",
        locus: str = "",
    ) -> SampleRepertoire:
        """Parse *path* and return a :class:`~mir.common.repertoire.SampleRepertoire`.

        Parameters
        ----------
        path:
            Path to an OLGA output file (plain or gzipped ``.gz``).
        sample_id:
            Identifier for the resulting sample.
        locus:
            Override locus code.  When empty (default), locus is inferred
            from each row's J-gene prefix.
        """
        path = Path(path)
        if not sample_id:
            stem = path.stem
            if stem.endswith(".txt"):
                stem = stem[:-4]
            sample_id = stem

        opener = gzip.open if path.suffix == ".gz" else open
        clonotypes: list[Clonotype] = []

        with opener(path, "rt", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for i, row in enumerate(reader):
                if len(row) < 4:
                    continue
                clonotypes.append(Clonotype(_validate=False,
                    sequence_id=str(i),
                    junction=    row[0].strip(),
                    junction_aa= row[1].strip(),
                    v_gene=      row[2].strip(),
                    j_gene=      row[3].strip(),
                    locus=       locus,
                ))

        return SampleRepertoire.from_clonotypes(clonotypes, sample_id=sample_id)


# ---------------------------------------------------------------------------
# AIRRWriter — repertoire → tab-separated AIRR file
# ---------------------------------------------------------------------------

class AIRRWriter:
    """Write a repertoire to a tab-separated AIRR file.

    Accepts both :class:`~mir.common.repertoire.SampleRepertoire` and
    :class:`~mir.common.repertoire.LocusRepertoire`.

    Empty strings are written as blank fields (not quoted ``""``) by
    converting them to null before serialising.

    Parameters
    ----------
    compress:
        ``True`` → always gzip; ``False`` → never; ``None`` (default) →
        infer from the output file extension (``.gz`` → compressed).
    """

    def __init__(self, compress: bool | None = None) -> None:
        self._compress = compress

    def _should_compress(self, path: Path) -> bool:
        if self._compress is not None:
            return self._compress
        return path.suffix == ".gz"

    @staticmethod
    def _clean(df: pl.DataFrame) -> pl.DataFrame:
        """Replace empty strings with null so they serialise as blank fields."""
        str_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        if not str_cols:
            return df
        return df.with_columns([
            pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col)
            for col in str_cols
        ])

    def write(
        self,
        repertoire: Union[SampleRepertoire, LocusRepertoire],
        path: str | Path,
    ) -> None:
        """Serialise *repertoire* to *path* as a tab-separated AIRR file.

        Parameters
        ----------
        repertoire:
            Repertoire to serialise.
        path:
            Destination file.  Files ending in ``.gz`` are gzip-compressed
            unless overridden by the constructor *compress* argument.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self._clean(repertoire.to_polars())

        if self._should_compress(path):
            with gzip.open(path, "wb") as fh:
                df.write_csv(fh, separator="\t", null_value="")
        else:
            df.write_csv(path, separator="\t", null_value="")


# ---------------------------------------------------------------------------
# load_vdjdb_latest — download latest VDJdb, return filtered LocusRepertoire
# ---------------------------------------------------------------------------

def load_vdjdb_latest(
    epitope: str,
    locus: str = "TRB",
    species: str = "HomoSapiens",
    mhc_a_contains: str = "",
) -> LocusRepertoire:
    """Download the latest VDJdb release and return a filtered LocusRepertoire.

    Fetches the latest release ZIP from the antigenomics/vdjdb-db GitHub
    repository, parses the slim TSV file inside it, and returns a
    deduplicated :class:`~mir.common.repertoire.LocusRepertoire`.

    Parameters
    ----------
    epitope:
        Antigen epitope sequence to keep (``antigen.epitope`` column).
    locus:
        Receptor gene locus to keep (``gene`` column, e.g. ``"TRB"``).
    species:
        Host species to keep (``species`` column).  Empty string = no filter.
    mhc_a_contains:
        Keep only rows where the ``mhc.a`` field contains this substring
        (e.g. ``"A*02"``).  Empty string = no filter.

    Returns
    -------
    LocusRepertoire
        Unique entries (deduplicated by junction_aa + V-gene base + J-gene
        base), each with ``duplicate_count=1``.
    """
    # --- resolve latest release ZIP URL via GitHub API ----------------------
    api_url = "https://api.github.com/repos/antigenomics/vdjdb-db/releases/latest"
    req = urllib.request.Request(api_url, headers={"User-Agent": "mirpy"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        release = json.loads(resp.read())

    zip_url = next(
        (a["browser_download_url"] for a in release.get("assets", [])
         if a["name"].endswith(".zip")),
        None,
    )
    if not zip_url:
        raise RuntimeError(
            f"No .zip asset found in VDJdb release {release.get('tag_name')!r}"
        )

    # --- download to a temp file, parse in memory ---------------------------
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"Downloading: {zip_url}")
        urllib.request.urlretrieve(zip_url, str(tmp_path))

        with zipfile.ZipFile(tmp_path) as zf:
            txt_entries = [n for n in zf.namelist() if n.endswith(".txt")]
            slim_entries = [n for n in txt_entries if "slim" in n.lower()]
            target = slim_entries[0] if slim_entries else (txt_entries[0] if txt_entries else None)
            if target is None:
                raise RuntimeError(f"No .txt file found inside VDJdb ZIP. Contents: {zf.namelist()}")

            with zf.open(target) as raw:
                content = io.TextIOWrapper(raw, encoding="utf-8")
                reader = csv.DictReader(content, delimiter="\t")

                seen: set[tuple] = set()
                clonotypes: list[Clonotype] = []

                for row in reader:
                    if row.get("gene", "").strip() != locus:
                        continue
                    if species and row.get("species", "").strip() != species:
                        continue
                    if row.get("antigen.epitope", "").strip() != epitope:
                        continue
                    if mhc_a_contains and mhc_a_contains not in row.get("mhc.a", ""):
                        continue

                    junction_aa = row.get("cdr3", "").strip()
                    if not junction_aa:
                        continue

                    v_gene = row.get("v.segm", "").strip()
                    j_gene = row.get("j.segm", "").strip()

                    dedup_key = (junction_aa, v_gene.split("*")[0], j_gene.split("*")[0])
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    try:
                        v_end_aa = int(row.get("v.end", -1))
                        v_seq_end = v_end_aa * 3 if v_end_aa >= 0 else -1
                    except (ValueError, TypeError):
                        v_seq_end = -1
                    try:
                        j_start_aa = int(row.get("j.start", -1))
                        j_seq_start = j_start_aa * 3 if j_start_aa >= 0 else -1
                    except (ValueError, TypeError):
                        j_seq_start = -1

                    clone = Clonotype(_validate=False,
                        sequence_id=str(len(clonotypes)),
                        duplicate_count=1,
                        locus=locus,
                        junction_aa=junction_aa,
                        junction=back_translate(junction_aa),
                        v_gene=v_gene,
                        j_gene=j_gene,
                        v_sequence_end=v_seq_end,
                        j_sequence_start=j_seq_start,
                    )
                    clone.clone_metadata.update({
                        col: row.get(col, "")
                        for col in VDJdbSlimParser._METADATA_COLS
                    })
                    clonotypes.append(clone)
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"{epitope}: {len(clonotypes)} unique {locus} clonotypes")
    return LocusRepertoire(clonotypes=clonotypes, locus=locus)
