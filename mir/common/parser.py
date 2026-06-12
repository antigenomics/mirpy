"""Clonotype table parsers and AIRR writer.

Parsers
-------
* :class:`ClonotypeTableParser` — generic VDJtools / AIRR table parser
  (returns ``list[Clonotype]``).
* :class:`VDJtoolsParser` — VDJtools-format tables.
* :class:`AIRRParser` — AIRR-format tables.
* :class:`AdaptiveParser` — Adaptive immunoSEQ / MLR tables.
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
import re
import sys
import tempfile
import urllib.request
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
from mir.common.alleles import allele_with_default, strip_allele as _strip_gene
from mir.common.clonotype import Clonotype
from mir.common.repertoire import SampleRepertoire, LocusRepertoire
from mir.common.single_cell import PairedRepertoire, build_tenx_sample_from_cell_clonotypes

# GeneLibrary is only imported lazily when needed (functional checks, germline sequences).
# Do NOT instantiate it inside parsers — load it explicitly via GeneLibrary.load_default().

# Canonical mapping from any accepted input column name to mirpy's internal
# (AIRR Rearrangement) schema names.  Internal gene fields use the AIRR
# ``*_call`` spelling, so this is the single place where VDJtools (``v``),
# legacy internal (``v_gene``) and AIRR (``v_call``) input columns converge.
# Kept public because downstream test code imports it.
_INPUT_TO_INTERNAL: dict[str, str] = {
    'count':    'duplicate_count',
    '#count':   'duplicate_count',
    'cdr3nt':   'junction',
    'cdr3aa':   'junction_aa',
    'v':        'v_call',
    'd':        'd_call',
    'j':        'j_call',
    # Legacy internal gene names (pre-AIRR-unification) accepted as input only.
    'v_gene':   'v_call',
    'd_gene':   'd_call',
    'j_gene':   'j_call',
    'c_gene':   'c_call',
    'VEnd':     'v_sequence_end',
    'DStart':   'd_sequence_start',
    'DEnd':     'd_sequence_end',
    'JStart':   'j_sequence_start',
    # Alternative locus column names used in some tools / AIRR variants.
    'chain':                      'locus',
    # MiXCR v2/v3 export format (e.g. alice benchmark files).
    'Read.count':                 'duplicate_count',
    'CDR3.amino.acid.sequence':   'junction_aa',
    'CDR3.nucleotide.sequence':   'junction',
    'bestVGene':                  'v_call',
    'bestJGene':                  'j_call',
}

# Backward-compat namedtuple kept as a public export.
VdjdbPayload = namedtuple('VdjdbPayload', 'mhc_a mhc_b mhc_class epitope pathogen')


def _gene_str(val) -> str:
    """Normalise a gene field value to a plain string."""
    if val is None or (not isinstance(val, str) and pd.isna(val)):
        return ""
    s = str(val).strip().split(',')[0].split(';')[0]
    if s in ('.', ''):
        return ""
    return allele_with_default(s)


# ---------------------------------------------------------------------------
# ClonotypeTableParser — generic (VDJtools / AIRR column names)
# ---------------------------------------------------------------------------

class ClonotypeTableParser:
    """Parse clonotype tables into lists of :class:`Clonotype` objects.

    Accepts file paths (string/Path) or pre-loaded :class:`pd.DataFrame`.
    Gene names (v_call, d_call, j_call, c_call) are stored as plain strings —
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
        return df.rename(columns=_INPUT_TO_INTERNAL)

    @staticmethod
    def _normalize_pl(df: pl.DataFrame) -> pl.DataFrame:
        """Normalise polars column names: strip leading ``#``, map VDJtools → AIRR."""
        rename = {}
        for col in df.columns:
            target = _INPUT_TO_INTERNAL.get(col.lstrip('#'), col.lstrip('#'))
            if target != col:
                rename[col] = target
        return df.rename(rename) if rename else df

    # ------------------------------------------------------------------
    # Vectorised gene normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_gene_col(s: pl.Series) -> list[str]:
        """Vectorised gene normalisation.

        Bare genes (no allele suffix) receive the major allele ``*01``; explicit
        alleles (e.g. ``*02``) are preserved (see :func:`allele_with_default`).
        """
        values = (
            s.cast(pl.Utf8).fill_null("")
            .str.split_exact(",", 1).struct.field("field_0")
            .str.split_exact(";", 1).struct.field("field_0")
            .str.strip_chars()
            .str.replace("^\\.$", "")
            .to_list()
        )
        return [allele_with_default(v) for v in values]

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
            df = self._filter_locus_pl(df)
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

    def _filter_locus_pl(self, df: pl.DataFrame) -> pl.DataFrame:
        """Restrict a polars frame to the parser's locus (no-op in the base class).

        Subclasses that target a single locus (e.g. :class:`AIRRParser`) override
        this so the file-parsing path applies the same locus filter as the
        DataFrame path — important for mixed-chain files where a TSV holds
        several loci.
        """
        return df

    def parse_inner(self, df: pd.DataFrame) -> list[Clonotype]:
        """Parse a pandas DataFrame into clonotypes (converts to polars internally).

        Column names are normalised to internal AIRR names first, so callers may
        pass raw tables using legacy ``v_gene``/VDJtools ``v`` spellings directly.
        The rename is idempotent, so passing an already-normalised frame is fine.
        """
        df = self.normalize_df(df)
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
        if 'junction_aa' not in cols or 'v_call' not in cols or 'j_call' not in cols:
            raise ValueError(
                f'Critical columns missing: {list(df.columns)}. '
                f'Expected junction_aa, v_call, j_call (AIRR) or '
                f'cdr3aa, v, j (VDJtools — pass through normalize first).')
        n = len(df)
        has_markup = {'v_sequence_end', 'd_sequence_start',
                      'd_sequence_end', 'j_sequence_start'}.issubset(cols)

        dup_counts   = df['duplicate_count'].cast(pl.Int64).fill_null(1).to_list() \
                       if 'duplicate_count' in cols else [1] * n
        umi_counts   = df['umi_count'].cast(pl.Int64).fill_null(0).to_list() \
                   if 'umi_count' in cols else [0] * n
        junctions    = df['junction'].cast(pl.Utf8).fill_null("").to_list() \
                       if 'junction' in cols else [""] * n
        junction_aas = df['junction_aa'].cast(pl.Utf8).fill_null("").to_list()
        v_calls      = self._norm_gene_col(df['v_call'])
        j_calls      = self._norm_gene_col(df['j_call'])
        d_calls      = self._norm_gene_col(df['d_call']) if 'd_call' in cols else [""] * n

        # Vectorise locus inference in polars: take first 3 chars of j_call
        # and keep only known IMGT locus codes (TRA/TRB/TRG/TRD/IGH/IGK/IGL).
        _valid_loci: frozenset[str] = frozenset(
            ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
        )
        if 'locus' in cols:
            loci = self._normalize_locus_col(df['locus'])
        else:
            _j_prefix = (
                df['j_call'].cast(pl.Utf8).fill_null("")
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
                umi_count=umi,
                junction=jnt,
                junction_aa=jaa,
                v_call=vg,
                d_call=dg,
                j_call=jg,
                v_sequence_end=ve,
                d_sequence_start=ds,
                d_sequence_end=de,
                j_sequence_start=js,
            )
            for sid, loc, dup, umi, jnt, jaa, vg, dg, jg, ve, ds, de, js in zip(
                seq_ids, loci, dup_counts, umi_counts, junctions, junction_aas,
                v_calls, d_calls, j_calls, v_ends, d_starts, d_ends, j_starts,
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
        umi_counts   = df['umi_count'].cast(pl.Int64).fill_null(0).to_list() \
                   if 'umi_count' in cols else [0] * n
        junctions    = df['junction'].cast(pl.Utf8).fill_null("").to_list() \
                       if 'junction' in cols else [""] * n
        junction_aas_raw = df['junction_aa'].cast(pl.Utf8).fill_null("").to_list()
        v_calls      = self._norm_gene_col(df['v_call'])
        j_calls      = self._norm_gene_col(df['j_call'])
        d_calls      = self._norm_gene_col(df['d_call']) if 'd_call' in cols else [""] * n

        _valid_loci: frozenset[str] = frozenset(
            ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
        )
        if 'locus' in cols:
            loci = self._normalize_locus_col(df['locus'])
        else:
            _j_prefix = (
                df['j_call'].cast(pl.Utf8).fill_null("")
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
                    'umi_counts': [],
                    'junction_aas': [], 'v_calls': [], 'd_calls': [], 'j_calls': [],
                    'v_ends': [], 'd_starts': [], 'd_ends': [], 'j_starts': [],
                }
            g = groups[loc]
            g['seq_ids'].append(seq_ids[i])
            g['dup_counts'].append(dup_counts[i])
            g['umi_counts'].append(umi_counts[i])
            g['junctions'].append(junctions[i])
            g['junction_aas'].append(junction_aas[i])
            g['v_calls'].append(v_calls[i])
            g['d_calls'].append(d_calls[i])
            g['j_calls'].append(j_calls[i])
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

    def _filter_locus_pl(self, df: pl.DataFrame) -> pl.DataFrame:
        """Keep only rows for this parser's locus (file path; mixed-chain TSVs)."""
        if 'locus' not in df.columns:
            return df  # locus inferred per-row downstream
        aliases = self.get_locus_aliases()
        keep = [
            v in aliases
            for v in df['locus'].cast(pl.Utf8).fill_null("").str.strip_chars().str.to_lowercase().to_list()
        ]
        return df.filter(pl.Series(keep))

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
        # No gene-column rename needed: internal schema is AIRR (v_call/j_call).
        if 'clone_id' in df.columns:
            df.index = df['clone_id'].astype(str)
        return super().parse_inner(df)


# ---------------------------------------------------------------------------
# AdaptiveParser — Adaptive immunoSEQ / MLR tables → LocusRepertoire
# ---------------------------------------------------------------------------

class AdaptiveParser(ClonotypeTableParser):
    """Parse Adaptive immunoSEQ / MLR tables into a single-locus repertoire.

    The MLR benchmark files expose Adaptive-style annotations.  This parser
    maps the relevant columns to AIRR names, normalizes gene naming, and then
    returns a :class:`~mir.common.repertoire.LocusRepertoire` for a chosen
    locus (default: ``TRB``).

    Field mapping
    -------------
    ``nucleotide``     → ``junction``
    ``aminoAcid``      → ``junction_aa``
    ``count``          → ``duplicate_count``
    ``vMaxResolved``   → ``v_call``
    ``jMaxResolved``   → ``j_call``
    ``dMaxResolved``   → ``d_call``

    Gene-name normalization keeps the file names usable as IMGT-style gene
    strings:

    * ``TCRBV01`` → ``TRBV1``
    * ``TCR`` → ``TR``
    * bare genes receive the major allele ``*01``; explicit alleles are preserved

    Boundary reconstruction from insertion/deletion annotations is deferred to
    a later pass that uses the reference library.
    """

    _ADAPTIVE_TO_AIRR: dict[str, str] = {
        "nucleotide": "junction",
        "aminoAcid": "junction_aa",
        "count": "duplicate_count",
        "vMaxResolved": "v_call",
        "jMaxResolved": "j_call",
        "dMaxResolved": "d_call",
    }

    def __init__(self, sep: str = '\t', locus: str = 'TRB') -> None:
        super().__init__(sep=sep)
        self.locus = locus

    @staticmethod
    def _normalize_gene_value(value) -> str:
        """Return a cleaned Adaptive gene string.

        Normalizations:
        - TCRBV → TRBV (TCR → TR)
        - TRBV05-01 → TRBV5-1 (remove leading zeros from gene number and subtype)
        - allele suffix harmonised via :func:`allele_with_default` (bare → ``*01``,
          explicit alleles preserved)
        """
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return ""
        gene = str(value).strip().split(',')[0].split(';')[0]
        if not gene or gene == '.':
            return ""
        gene = gene.replace('TCR', 'TR')
        # Remove leading zeros from gene number (e.g., TRBV05 → TRBV5)
        gene = re.sub(r'^(TR[ABDG][VDJ])0+(\d+)', r'\1\2', gene)
        # Remove leading zeros from subtype after dash (e.g., V5-01 → V5-1)
        gene = re.sub(r'-0+(\d+)', r'-\1', gene)
        return allele_with_default(gene)

    @classmethod
    def _normalize_pl(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Normalise Adaptive table columns to AIRR-style names in polars."""
        df = df.rename({str(col).strip().lstrip('#'): str(col).strip().lstrip('#') for col in df.columns})
        rename = {src: dst for src, dst in cls._ADAPTIVE_TO_AIRR.items() if src in df.columns}
        if rename:
            df = df.rename(rename)
        for col in ('v_call', 'd_call', 'j_call'):
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col)
                    .cast(pl.Utf8)
                    .fill_null('')
                    .map_elements(cls._normalize_gene_value, return_dtype=pl.Utf8)
                    .alias(col)
                )
        if 'duplicate_count' in df.columns:
            df = df.with_columns(
                pl.col('duplicate_count').cast(pl.Int64, strict=False).fill_null(1).alias('duplicate_count')
            )
        if 'junction_aa' in df.columns:
            df = df.with_columns(pl.col('junction_aa').cast(pl.Utf8).fill_null('').alias('junction_aa'))
        if 'junction' in df.columns:
            df = df.with_columns(pl.col('junction').cast(pl.Utf8).fill_null('').alias('junction'))
        return df

    @classmethod
    def normalize_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise Adaptive table columns to AIRR-style names."""
        return cls._normalize_pl(pl.from_pandas(df, include_index=False, strict=False)).to_pandas()

    def parse_file(
        self,
        path: str | Path,
        sample_id: str = '',
        locus: str = '',
    ) -> LocusRepertoire:
        """Parse *path* and return a single-locus repertoire."""
        path = Path(path)
        if not sample_id:
            sample_id = path.stem

        df = pl.read_csv(
            path,
            separator=self.sep,
            infer_schema_length=0,
            null_values=['', 'NA'],
            truncate_ragged_lines=True,
        )
        df = self._normalize_pl(df)

        target_locus = locus or self.locus or ''
        clonotypes = self._polars_to_clonotypes(df)
        if target_locus:
            clonotypes = [clonotype for clonotype in clonotypes if clonotype.locus == target_locus]
        return LocusRepertoire(clonotypes=clonotypes, locus=target_locus, repertoire_id=sample_id)

    def parse_inner(self, df: pd.DataFrame) -> LocusRepertoire:
        """Parse an already-loaded Adaptive table into a locus repertoire."""
        normalized = self.normalize_df(df)
        target_locus = self.locus or ''
        clonotypes = self._polars_to_clonotypes(pl.from_pandas(normalized, include_index=False, strict=False))
        if target_locus:
            clonotypes = [clonotype for clonotype in clonotypes if clonotype.locus == target_locus]
        return LocusRepertoire(clonotypes=clonotypes, locus=target_locus)


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
        """Return the first gene hit with a harmonised allele suffix.

        MiXCR's ``*00`` "no-allele" marker is dropped and the bare gene then
        receives the major allele ``*01``; explicit alleles are preserved.
        """
        if not hits:
            return ""
        first = re.sub(r"\*00.*$", "", hits.split(",")[0])
        return allele_with_default(first)

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
                    v_call=           self._parse_gene(row[self._COL_V_HITS]),
                    d_call=           self._parse_gene(row[self._COL_D_HITS]),
                    j_call=           j,
                    c_call=           self._parse_gene(row[self._COL_C_HITS]),
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
    ``v.segm``            → ``v_call``
    ``j.segm``            → ``j_call``
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
                    v_call=           allele_with_default(row.get("v.segm", "").strip()),
                    j_call=           allele_with_default(row.get("j.segm", "").strip()),
                    v_sequence_end=   v_sequence_end,
                    j_sequence_start= j_sequence_start,
                )
                clone.clone_metadata.update({
                    col: row.get(col, "") for col in self._METADATA_COLS
                })
                clonotypes.append(clone)

        return SampleRepertoire.from_clonotypes(clonotypes, sample_id=sample_id)


class VDJdbFullPairedParser:
    """Parse VDJdb full export rows into paired-cell style structures.

    Each source row corresponds to one VDJdb record. TRA and TRB chains are
    extracted from the alpha/beta columns on that row and represented as a
    synthetic single-cell barcode keyed by the VDJdb row index.

    The resulting table is compatible with
    :func:`mir.common.single_cell_repair.impute_missing_chains` and
    :func:`mir.common.single_cell.build_tenx_sample_from_cell_clonotypes`.
    """

    _BARCODE_METADATA_COLS = (
        "vdjdb_record_id",
        "species",
        "mhc.a",
        "mhc.b",
        "mhc.class",
        "antigen.epitope",
        "antigen.gene",
        "antigen.species",
        "reference.id",
        "method.identification",
    )
    _CELL_SCHEMA = {
        "sample_id": pl.Utf8,
        "barcode": pl.Utf8,
        "raw_pair_id": pl.Utf8,
        "sequence_id": pl.Utf8,
        "duplicate_count": pl.Int64,
        "umi_count": pl.Int64,
        "locus": pl.Utf8,
        "junction": pl.Utf8,
        "junction_aa": pl.Utf8,
        "v_call": pl.Utf8,
        "d_call": pl.Utf8,
        "j_call": pl.Utf8,
        "c_call": pl.Utf8,
        "vdjdb_record_id": pl.Utf8,
        "species": pl.Utf8,
        "mhc.a": pl.Utf8,
        "mhc.b": pl.Utf8,
        "mhc.class": pl.Utf8,
        "antigen.epitope": pl.Utf8,
        "antigen.gene": pl.Utf8,
        "antigen.species": pl.Utf8,
        "reference.id": pl.Utf8,
        "method.identification": pl.Utf8,
    }

    @staticmethod
    def _clean_field(row: dict[str, str], key: str) -> str:
        value = row.get(key, "")
        if value is None:
            return ""
        return str(value).strip()

    def _metadata_for_row(self, row: dict[str, str], record_id: int) -> dict[str, str]:
        return {
            "vdjdb_record_id": str(record_id),
            "species": self._clean_field(row, "species"),
            "mhc.a": self._clean_field(row, "mhc.a"),
            "mhc.b": self._clean_field(row, "mhc.b"),
            "mhc.class": self._clean_field(row, "mhc.class"),
            "antigen.epitope": self._clean_field(row, "antigen.epitope"),
            "antigen.gene": self._clean_field(row, "antigen.gene"),
            "antigen.species": self._clean_field(row, "antigen.species"),
            "reference.id": self._clean_field(row, "reference.id"),
            "method.identification": self._clean_field(row, "method.identification"),
        }

    def _build_chain_row(
        self,
        *,
        row: dict[str, str],
        record_id: int,
        sample_id: str,
        locus: str,
        suffix: str,
        metadata: dict[str, str],
    ) -> dict[str, object] | None:
        junction_aa = self._clean_field(row, f"cdr3.{suffix}")
        if not junction_aa:
            return None

        cell_row: dict[str, object] = {
            "sample_id": sample_id,
            "barcode": str(record_id),
            "raw_pair_id": str(record_id),
            "sequence_id": f"{record_id}_{locus}",
            "duplicate_count": 1,
            "umi_count": 1,
            "locus": locus,
            "junction": back_translate(junction_aa),
            "junction_aa": junction_aa,
            "v_call": _gene_str(row.get(f"v.{suffix}")),
            "d_call": _gene_str(row.get(f"d.{suffix}")) if suffix == "beta" else "",
            "j_call": _gene_str(row.get(f"j.{suffix}")),
            "c_call": "",
        }
        cell_row.update(metadata)
        return cell_row

    def parse_cell_clonotypes_file(
        self,
        path: str | Path,
        *,
        sample_id: str = "",
        species: str = "HomoSapiens",
        include_incomplete: bool = False,
    ) -> tuple[pl.DataFrame, dict[str, dict[str, str]]]:
        """Return a single-cell style table plus per-barcode metadata.

        Parameters
        ----------
        path:
            Path to ``vdjdb_full.txt.gz`` or its uncompressed TSV equivalent.
        sample_id:
            Identifier for the resulting synthetic sample.
        species:
            Host species filter. Use an empty string to disable filtering.
        include_incomplete:
            When ``False`` (default), keep only rows with both TRA and TRB.
            When ``True``, keep single-chain rows as well so they can be passed
            through single-cell imputation before paired repertoire assembly.
        """
        path = Path(path)
        if not sample_id:
            stem = path.stem
            if stem.endswith(".txt") or stem.endswith(".tsv"):
                stem = Path(stem).stem
            sample_id = stem

        opener = gzip.open if path.suffix == ".gz" else open
        rows: list[dict[str, object]] = []
        barcode_metadata: dict[str, dict[str, str]] = {}

        with opener(path, "rt", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for record_id, row in enumerate(reader):
                row_species = self._clean_field(row, "species")
                if species and row_species != species:
                    continue

                metadata = self._metadata_for_row(row, record_id)
                alpha_row = self._build_chain_row(
                    row=row,
                    record_id=record_id,
                    sample_id=sample_id,
                    locus="TRA",
                    suffix="alpha",
                    metadata=metadata,
                )
                beta_row = self._build_chain_row(
                    row=row,
                    record_id=record_id,
                    sample_id=sample_id,
                    locus="TRB",
                    suffix="beta",
                    metadata=metadata,
                )
                if alpha_row is None and beta_row is None:
                    continue
                if not include_incomplete and (alpha_row is None or beta_row is None):
                    continue

                barcode = str(record_id)
                barcode_metadata[barcode] = metadata
                if alpha_row is not None:
                    rows.append(alpha_row)
                if beta_row is not None:
                    rows.append(beta_row)

        if not rows:
            return pl.DataFrame(schema=self._CELL_SCHEMA), barcode_metadata
        return pl.DataFrame(rows, schema=self._CELL_SCHEMA), barcode_metadata

    def parse_file(
        self,
        path: str | Path,
        *,
        sample_id: str = "",
        species: str = "HomoSapiens",
        include_incomplete: bool = False,
    ) -> PairedRepertoire:
        """Parse VDJdb full rows into a :class:`PairedRepertoire`."""
        cell_df, barcode_metadata = self.parse_cell_clonotypes_file(
            path,
            sample_id=sample_id,
            species=species,
            include_incomplete=include_incomplete,
        )
        return build_tenx_sample_from_cell_clonotypes(
            cell_df,
            sample_id=sample_id or Path(path).stem,
            barcode_metadata=barcode_metadata,
        )


# ---------------------------------------------------------------------------
# OlgaParser — OLGA sequence generation output → SampleRepertoire
# ---------------------------------------------------------------------------

class OlgaParser:
    """Parse OLGA-generated sequence files (tab-delimited, optionally gzipped).

    The file has **no header row**.  Columns (in order):
    ``junction``, ``junction_aa``, ``v_call``, ``j_call``.

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
                    v_call=      allele_with_default(row[2].strip()),
                    j_call=      allele_with_default(j),
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
            slim_entries = [n for n in txt_entries if "slim" in n.lower() and "meta" not in n.lower()]
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

                    v_call = allele_with_default(row.get("v.segm", "").strip())
                    j_call = allele_with_default(row.get("j.segm", "").strip())

                    dedup_key = (junction_aa, _strip_gene(v_call), _strip_gene(j_call))
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
                        v_call=v_call,
                        j_call=j_call,
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


def load_10x_vdj_v1_donor(
    consensus_annotations_path: str | Path,
    all_contig_annotations_path: str | Path,
    donor_id: str = "",
):
    """Load one donor from 10x_vdj_v1 files into paired single-cell objects."""
    from mir.common.single_cell import load_10x_vdj_v1_donor as _load

    return _load(
        consensus_annotations_path=consensus_annotations_path,
        all_contig_annotations_path=all_contig_annotations_path,
        donor_id=donor_id,
    )
