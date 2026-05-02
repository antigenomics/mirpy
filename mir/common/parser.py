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
from mir.basic.aliases import airr_aliases_for_locus, normalize_airr_locus_value
from mir.common.clonotype import Clonotype, ClonotypeAA, ClonotypeNT, JunctionMarkup
from mir.common.repertoire import SampleRepertoire, LocusRepertoire

# GeneLibrary is only imported lazily when needed (functional checks, germline sequences).
# Do NOT instantiate it inside parsers — load it explicitly via GeneLibrary.load_default().

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

# Backward-compat namedtuple kept as a public export.
VdjdbPayload = namedtuple('VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')


def _gene_str(val) -> str:
    """Normalise a gene field value to a plain string."""
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return ""
    s = str(val).strip().split(',')[0].split(';')[0]
    return s if s not in ('.', '') else ""


# ---------------------------------------------------------------------------
# ClonotypeTableParser — generic (VDJtools / AIRR column names)
# ---------------------------------------------------------------------------

class ClonotypeTableParser:
    """Parse clonotype tables into lists of :class:`Clonotype` objects.

    Accepts file paths (string/Path) or pre-loaded :class:`pd.DataFrame`.
    Gene names (v_gene, d_gene, j_gene, c_gene) are stored as plain strings —
    consult :class:`~mir.common.gene_library.GeneLibrary` explicitly when you
    need germline sequences or functional annotations.

    Column names are normalised so both VDJtools-style and AIRR-style inputs
    are handled transparently.  File reading uses polars for speed; gzip files
    are decompressed automatically.
    """

    def __init__(self, sep: str = '\t') -> None:
        self.sep = sep

    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise pandas column names: strip leading ``#``, map VDJtools → AIRR."""
        df = df.copy()
        df.columns = [c.lstrip('#') for c in df.columns]
        return df.rename(columns=_VDJTOOLS_TO_AIRR)

    @staticmethod
    def _normalize_pl(df: pl.DataFrame) -> pl.DataFrame:
        """Normalise polars column names: strip leading ``#``, map VDJtools → AIRR."""
        rename = {}
        for col in df.columns:
            target = _VDJTOOLS_TO_AIRR.get(col.lstrip('#'), col.lstrip('#'))
            if target != col:
                rename[col] = target
        return df.rename(rename) if rename else df

    # ------------------------------------------------------------------
    # Vectorised gene normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_gene_col(s: pl.Series) -> list[str]:
        """Vectorised gene normalisation: first comma/semicolon token, strip, '.' → ''."""
        return (
            s.cast(pl.Utf8).fill_null("")
            .str.split_exact(",", 1).struct.field("field_0")
            .str.split_exact(";", 1).struct.field("field_0")
            .str.strip_chars()
            .str.replace("^\\.$", "")
            .to_list()
        )

    @staticmethod
    def _normalize_locus_col(s: pl.Series) -> list[str]:
        """Normalize AIRR locus aliases to canonical IMGT codes."""
        raw = s.cast(pl.Utf8).fill_null("").str.strip_chars().str.to_lowercase().to_list()
        return [normalize_airr_locus_value(v) for v in raw]

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_polars(self, path: str) -> pl.DataFrame:
        """Read a TSV/CSV file (plain or gzipped) into a polars DataFrame with
        normalised AIRR column names.  All values are kept as strings."""
        df = pl.read_csv(
            path,
            separator=self.sep,
            infer_schema_length=0,
            null_values=["", "NA"],
            truncate_ragged_lines=True,
        )
        return self._normalize_pl(df)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(
        self,
        source: str | pd.DataFrame,
        n: int | None = None,
        sample: bool = False,
    ) -> list[Clonotype]:
        """Parse *source* (file path or DataFrame) into a list of clonotypes."""
        if isinstance(source, str):
            df = self._read_polars(source)
            if n is not None and not sample:
                df = df.head(n)
            elif n is not None:
                df = df.sample(n=n, seed=42)
            return self._polars_to_clonotypes(df)
        else:
            # pandas DataFrame passed directly (e.g. from tests)
            if not sample:
                source = source.head(n)
            elif n is not None:
                source = source.sample(n=n, random_state=42)
            return self.parse_inner(self.normalize_df(source))

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        """Parse an already-normalised pandas DataFrame. Converts to polars internally."""
        # Avoid pl.from_pandas here because nullable pandas extension dtypes
        # (e.g. Int64) require pyarrow during conversion.
        data = {
            col: [None if pd.isna(v) else v for v in df[col].tolist()]
            for col in df.columns
        }
        # Mixed pandas dtypes (e.g. mostly-null gene columns) can otherwise be
        # inferred as numeric by polars construction and fail on first string.
        return self._polars_to_clonotypes(pl.DataFrame(data, strict=False))

    def _polars_to_clonotypes(self, df: pl.DataFrame) -> list[Clonotype]:
        """Vectorised conversion of a normalised polars DataFrame → list[Clonotype].

        All field extraction and normalisation happens in polars (no Python loops
        over rows).  Clonotype objects are constructed with ``_validate=False`` so
        ``__post_init__`` is a no-op; every field must be fully clean before this
        call returns.
        """
        cols = set(df.columns)
        if 'junction_aa' not in cols or 'v_gene' not in cols or 'j_gene' not in cols:
            raise ValueError(
                f'Critical columns missing: {list(df.columns)}. '
                f'Expected junction_aa, v_gene, j_gene (AIRR) or '
                f'cdr3aa, v, j (VDJtools — pass through normalize first).')
        n = len(df)
        has_markup = {'v_sequence_end', 'd_sequence_start',
                      'd_sequence_end', 'j_sequence_start'}.issubset(cols)

        dup_counts   = df['duplicate_count'].cast(pl.Int64).fill_null(1).to_list() \
                       if 'duplicate_count' in cols else [1] * n
        junctions    = df['junction'].cast(pl.Utf8).fill_null("").to_list() \
                       if 'junction' in cols else [""] * n
        junction_aas = df['junction_aa'].cast(pl.Utf8).fill_null("").to_list()
        v_genes      = self._norm_gene_col(df['v_gene'])
        j_genes      = self._norm_gene_col(df['j_gene'])
        d_genes      = self._norm_gene_col(df['d_gene']) if 'd_gene' in cols else [""] * n

        # Vectorise locus inference in polars: take first 3 chars of j_gene
        # and keep only known IMGT locus codes (TRA/TRB/TRG/TRD/IGH/IGK/IGL).
        _valid_loci: frozenset[str] = frozenset(
            ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
        )
        if 'locus' in cols:
            loci = self._normalize_locus_col(df['locus'])
        else:
            _j_prefix = (
                df['j_gene'].cast(pl.Utf8).fill_null("")
                    .str.slice(0, 3).str.to_uppercase().to_list()
            )
            loci = [p if p in _valid_loci else "" for p in _j_prefix]

        # junction_aa fallback: translate junction where junction_aa is blank.
        from mir.basic.mirseq_compat import translate_bidi
        junction_aas = [
            jaa if jaa else (translate_bidi(jnt) if jnt else "")
            for jaa, jnt in zip(junction_aas, junctions)
        ]

        if has_markup:
            v_ends   = df['v_sequence_end'].cast(pl.Int64).fill_null(-1).to_list()
            d_starts = df['d_sequence_start'].cast(pl.Int64).fill_null(-1).to_list()
            d_ends   = df['d_sequence_end'].cast(pl.Int64).fill_null(-1).to_list()
            j_starts = df['j_sequence_start'].cast(pl.Int64).fill_null(-1).to_list()
        else:
            v_ends = d_starts = d_ends = j_starts = [-1] * n

        seq_ids = [str(i) for i in range(n)]
        return [
            Clonotype(_validate=False,
                sequence_id=sid,
                locus=loc,
                duplicate_count=dup,
                junction=jnt,
                junction_aa=jaa,
                v_gene=vg,
                d_gene=dg,
                j_gene=jg,
                v_sequence_end=ve,
                d_sequence_start=ds,
                d_sequence_end=de,
                j_sequence_start=js,
            )
            for sid, loc, dup, jnt, jaa, vg, dg, jg, ve, ds, de, js in zip(
                seq_ids, loci, dup_counts, junctions, junction_aas,
                v_genes, d_genes, j_genes, v_ends, d_starts, d_ends, j_starts,
            )
        ]


    def _polars_to_col_groups(self, df: "pl.DataFrame") -> "dict[str, dict]":
        """Extract column lists from a normalised DataFrame, grouped by locus.

        Returns a dict mapping each locus string to a column-list dict.
        No Clonotype objects are constructed — the caller stores the dict and
        materialises lazily via :meth:`LocusRepertoire._from_lazy_cols`.
        """
        cols = set(df.columns)
        n = len(df)
        has_markup = {'v_sequence_end', 'd_sequence_start',
                      'd_sequence_end', 'j_sequence_start'}.issubset(cols)

        dup_counts   = df['duplicate_count'].cast(pl.Int64).fill_null(1).to_list() \
                       if 'duplicate_count' in cols else [1] * n
        junctions    = df['junction'].cast(pl.Utf8).fill_null("").to_list() \
                       if 'junction' in cols else [""] * n
        junction_aas_raw = df['junction_aa'].cast(pl.Utf8).fill_null("").to_list()
        v_genes      = self._norm_gene_col(df['v_gene'])
        j_genes      = self._norm_gene_col(df['j_gene'])
        d_genes      = self._norm_gene_col(df['d_gene']) if 'd_gene' in cols else [""] * n

        _valid_loci: frozenset[str] = frozenset(
            ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
        )
        if 'locus' in cols:
            loci = self._normalize_locus_col(df['locus'])
        else:
            _j_prefix = (
                df['j_gene'].cast(pl.Utf8).fill_null("")
                    .str.slice(0, 3).str.to_uppercase().to_list()
            )
            loci = [p if p in _valid_loci else "" for p in _j_prefix]

        from mir.basic.mirseq_compat import translate_bidi
        junction_aas = [
            jaa if jaa else (translate_bidi(jnt) if jnt else "")
            for jaa, jnt in zip(junction_aas_raw, junctions)
        ]

        if has_markup:
            v_ends   = df['v_sequence_end'].cast(pl.Int64).fill_null(-1).to_list()
            d_starts = df['d_sequence_start'].cast(pl.Int64).fill_null(-1).to_list()
            d_ends   = df['d_sequence_end'].cast(pl.Int64).fill_null(-1).to_list()
            j_starts = df['j_sequence_start'].cast(pl.Int64).fill_null(-1).to_list()
        else:
            v_ends = d_starts = d_ends = j_starts = [-1] * n

        seq_ids = [str(i) for i in range(n)]

        # Group rows by locus
        groups: dict[str, dict] = {}
        for i, loc in enumerate(loci):
            if loc not in groups:
                groups[loc] = {
                    'locus': loc,
                    'seq_ids': [], 'dup_counts': [], 'junctions': [],
                    'junction_aas': [], 'v_genes': [], 'd_genes': [], 'j_genes': [],
                    'v_ends': [], 'd_starts': [], 'd_ends': [], 'j_starts': [],
                }
            g = groups[loc]
            g['seq_ids'].append(seq_ids[i])
            g['dup_counts'].append(dup_counts[i])
            g['junctions'].append(junctions[i])
            g['junction_aas'].append(junction_aas[i])
            g['v_genes'].append(v_genes[i])
            g['d_genes'].append(d_genes[i])
            g['j_genes'].append(j_genes[i])
            g['v_ends'].append(v_ends[i])
            g['d_starts'].append(d_starts[i])
            g['d_ends'].append(d_ends[i])
            g['j_starts'].append(j_starts[i])
        return groups


# ---------------------------------------------------------------------------
# VDJtoolsParser
# ---------------------------------------------------------------------------

class VDJtoolsParser(ClonotypeTableParser):
    """Parse VDJtools output files.

    The raw VDJtools header (``#count``, ``cdr3nt``, ``cdr3aa``, ``v``, ``d``,
    ``j``, ``VEnd``, ``DStart``, ``DEnd``, ``JStart``) is automatically mapped
    to AIRR names.  Gene names are stored as plain strings.
    """
    pass  # All functionality is inherited from ClonotypeTableParser.


# ---------------------------------------------------------------------------
# AIRRParser
# ---------------------------------------------------------------------------

class AIRRParser(ClonotypeTableParser):
    """Parse AIRR-format files.

    Mandatory columns: ``locus``, ``v_call``, ``j_call``, ``junction_aa``.
    Gene names are stored as plain strings.
    """

    def __init__(
        self,
        sep: str = '\t',
        locus: str = 'beta',
    ) -> None:
        super().__init__(sep)
        self.locus = locus
        self.mandatory_columns = ['locus', 'v_call', 'j_call', 'junction_aa']

    def get_locus_aliases(self) -> set[str]:
        return airr_aliases_for_locus(str(self.locus))

    def validate_columns(self, df: pd.DataFrame) -> None:
        for col in self.mandatory_columns:
            if col not in df.columns:
                raise KeyError(
                    f'Mandatory column {col!r} is missing! '
                    f'Mandatory columns: {self.mandatory_columns}')

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        self.validate_columns(df)
        locus_aliases = self.get_locus_aliases()
        df = df[df['locus'].astype(str).str.strip().str.lower().isin(locus_aliases)].copy()
        # Rename AIRR call columns to AIRR gene names for the base parser
        df = df.rename(columns={'v_call': 'v_gene', 'j_call': 'j_gene'})
        if 'clone_id' in df.columns:
            df.index = df['clone_id'].astype(str)
        return super().parse_inner(df)


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
                j = self._parse_gene(row[self._COL_J_HITS])
                clonotypes.append(Clonotype(_validate=False,
                    sequence_id=      row[self._COL_CLONE_ID],
                    duplicate_count=  int(row[self._COL_COUNT]),
                    junction=         row[self._COL_JUNCTION],
                    junction_aa=      junction_aa,
                    locus=            j[:3].upper() if j else "",
                    v_gene=           self._parse_gene(row[self._COL_V_HITS]),
                    d_gene=           self._parse_gene(row[self._COL_D_HITS]),
                    j_gene=           j,
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

        _loci_map = {"TRA": "TRA", "TRB": "TRB", "TRG": "TRG", "TRD": "TRD",
                     "IGH": "IGH", "IGK": "IGK", "IGL": "IGL"}
        with opener(path, "rt", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for i, row in enumerate(reader):
                if len(row) < 4:
                    continue
                j = row[3].strip()
                clonotypes.append(Clonotype(_validate=False,
                    sequence_id=str(i),
                    junction=    row[0].strip(),
                    junction_aa= row[1].strip(),
                    v_gene=      row[2].strip(),
                    j_gene=      j,
                    locus=       locus or _loci_map.get(j[:3].upper(), ""),
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
