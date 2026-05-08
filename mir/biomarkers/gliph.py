"""GLIPH-style k-mer token artifact extraction for TCR repertoires.

Provides:
- :class:`GliphTokenArtifacts` — counts and bipartite adjacency for one token family.
- :func:`deduplicate_clonotype_rows` — aggregate repeated clonotype rows.
- :func:`extract_v3mer_artifacts` — V-gene anchored 3-mer extraction.
- :func:`extract_pos3mer_artifacts` — V-gene + junction position + 3-mer extraction.
- :func:`extract_vpos3mer_artifacts` — deprecated alias for :func:`extract_pos3mer_artifacts`.
- :func:`extract_u4mer_artifacts` — ungapped 4-mer extraction.
- :func:`extract_g4mer_artifacts` — gapped 4-mer extraction.
- :func:`extract_g5mer_artifacts` — gapped 5-mer extraction.
- :func:`normalize_control_v` — resample control to match sample unweighted V usage.
- :func:`normalize_control_vj` — resample control to match sample unweighted VJ usage.

Graph construction helpers (moved to :mod:`mir.graph.token_graph`)
-------------------------------------------------------------------
- ``combine_enriched_token_maps`` — merge enriched token neighborhoods across families.
- ``build_full_gliph_clonotype_graph`` — build a combined k-mer/Hamming clonotype graph.
- ``build_kmer_projection_graph`` — project token co-occurrence graph.

Threaded tokenisation
---------------------
The extraction helpers accept a ``threads`` argument. When ``threads > 1`` the
input DataFrame is split into chunks and processed via
:class:`concurrent.futures.ThreadPoolExecutor`.  The underlying tokeniser spends
most of its time in the C-extension, so threads retain a lightweight API while
keeping naming consistent with the rest of the codebase.

CDR3 trimming
-------------
Token extraction supports optional CDR3 trimming (first N and last M amino acids).
Use ``trim_first`` and ``trim_last`` parameters to enable.  Position tracking
is preserved: position in tokens reflects offset into trimmed sequence.
"""

from __future__ import annotations

import concurrent.futures
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
import pandas as pd

from mir.basic.token_tables import tokenize_rearrangements
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class GliphTokenArtifacts:
    """Token counts and bidirectional clonotype ↔ token mappings.

    Attributes
    ----------
    counts : dict[str, int]
        Active token count dictionary used for enrichment denominators.
    token_to_clone : dict[str, set[str]]
        Maps each token string to the set of clonotype row-IDs containing it.
    clone_to_tokens : dict[str, set[str]]
        Maps each clonotype row-ID to the set of tokens it contains.
    occurrence_counts : dict[str, int]
        Raw token occurrence counts across all windows.
    clonotype_counts : dict[str, int]
        Number of unique clonotypes containing each token.
    count_mode : {"occurrence", "clonotype"}
        Which count dictionary populates ``counts``.
    """

    counts: dict[str, int]
    token_to_clone: dict[str, set[str]]
    clone_to_tokens: dict[str, set[str]]
    occurrence_counts: dict[str, int]
    clonotype_counts: dict[str, int]
    count_mode: Literal["occurrence", "clonotype"] = "clonotype"


TOKEN_FAMILY = Literal["v3", "pos3", "u4", "g4", "g5", "vpos3"]
COUNT_MODE = Literal["occurrence", "clonotype"]

_DEFAULT_UNIQUE_CLONOTYPE_COLUMNS = (
    "reference_id",
    "junction_aa",
    "v_gene",
    "j_gene",
)


def _locus_repertoire_from_dataframe(df: pd.DataFrame, *, locus: str = "TRB") -> LocusRepertoire:
    """Build a LocusRepertoire from a pandas DataFrame with backward compatibility."""
    from_pandas = getattr(LocusRepertoire, "from_pandas", None)
    if callable(from_pandas):
        return from_pandas(df, locus=locus)

    tmp = df.copy()
    if "sequence_id" not in tmp.columns and "row_id" in tmp.columns:
        tmp = tmp.rename(columns={"row_id": "sequence_id"})
    seq_ids = tmp["sequence_id"].astype(str).tolist() if "sequence_id" in tmp.columns else [str(i) for i in range(len(tmp))]
    jaa = tmp["junction_aa"].astype(str).tolist()
    vg = tmp["v_gene"].astype(str).tolist()
    jg = tmp["j_gene"].astype(str).tolist() if "j_gene" in tmp.columns else [""] * len(tmp)
    dc = pd.to_numeric(tmp.get("duplicate_count", 1), errors="coerce").fillna(1).astype(int).tolist()
    clones = [
        Clonotype(
            sequence_id=sid,
            locus=locus,
            junction_aa=jaa_i,
            v_gene=vg_i,
            j_gene=jg_i,
            duplicate_count=dc_i,
            _validate=False,
        )
        for sid, jaa_i, vg_i, jg_i, dc_i in zip(seq_ids, jaa, vg, jg, dc)
    ]
    return LocusRepertoire(clones, locus=locus)


# ---------------------------------------------------------------------------
# Row → Clonotype conversion and CDR3 trimming
# ---------------------------------------------------------------------------


def _trim_junction_aa(junction_aa: str, trim_first: int = 3, trim_last: int = 4) -> str:
    """Trim junction_aa and return trimmed sequence.
    
    Parameters
    ----------
    junction_aa : str
        Full junction amino acid sequence.
    trim_first : int
        Number of amino acids to trim from the start (default 0).
    trim_last : int
        Number of amino acids to trim from the end (default 0).
    
    Returns
    -------
    str
        Trimmed junction_aa sequence.
    """
    if not (trim_first or trim_last):
        return junction_aa
    if trim_first >= len(junction_aa):
        return ""
    end = len(junction_aa) - trim_last if trim_last else len(junction_aa)
    if trim_first >= end:
        return ""
    return junction_aa[trim_first:end]


def repertoire_to_clonotypes(
    repertoire: LocusRepertoire,
    *,
    trim_first: int = 3,
    trim_last: int = 4,
) -> list[Clonotype]:
    """Convert a LocusRepertoire to a list of Clonotype objects.
    
    Optionally trims CDR3 (junction_aa) sequences before returning.
    
    Parameters
    ----------
    repertoire : LocusRepertoire
        Source repertoire.
    trim_first : int
        Number of amino acids to trim from start (default 0).
    trim_last : int
        Number of amino acids to trim from end (default 0).
    
    Returns
    -------
    list[Clonotype]
        List of clonotypes, with junction_aa trimmed if requested.
    """
    clonotypes = repertoire.clonotypes
    if not (trim_first or trim_last):
        return clonotypes
    
    trimmed = []
    for c in clonotypes:
        trimmed_jaa = _trim_junction_aa(c.junction_aa, trim_first=trim_first, trim_last=trim_last)
        if trimmed_jaa:  # Skip clonotypes that trim to empty
            trimmed_c = Clonotype(
                sequence_id=c.sequence_id,
                locus=c.locus,
                junction_aa=trimmed_jaa,
                junction=c.junction,
                v_gene=c.v_gene,
                d_gene=c.d_gene,
                j_gene=c.j_gene,
                v_sequence_end=c.v_sequence_end,
                d_sequence_start=c.d_sequence_start,
                d_sequence_end=c.d_sequence_end,
                j_sequence_start=c.j_sequence_start,
                duplicate_count=c.duplicate_count,
                _validate=False,
            )
            trimmed.append(trimmed_c)
    return trimmed


def rows_to_clonotypes(df: pd.DataFrame) -> list[Clonotype]:
    """Convert a DataFrame of AIRR-schema rows to :class:`Clonotype` objects.

    **Deprecated**: Use :func:`repertoire_to_clonotypes` with :class:`LocusRepertoire` instead.

    Uses column-level list extraction (not :meth:`DataFrame.iterrows`) for
    substantially faster conversion on large tables.

    Required columns
    ----------------
    ``row_id``, ``junction_aa``, ``v_gene``, ``duplicate_count``.
    Optional: ``j_gene`` (defaults to ``""`` if absent).
    """
    warnings.warn(
        "rows_to_clonotypes is deprecated; use repertoire_to_clonotypes with LocusRepertoire.",
        DeprecationWarning,
        stacklevel=2,
    )
    repertoire = _locus_repertoire_from_dataframe(df, locus="TRB")
    return repertoire_to_clonotypes(repertoire, trim_first=0, trim_last=0)


def _first_nonempty(series: pd.Series):
    """Return the first non-empty value from a grouped series."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    if non_null.dtype == object:
        stripped = non_null.astype(str).str.strip()
        valid = stripped[~stripped.str.lower().isin({"", "nan", "none", "na"})]
        if not valid.empty:
            return valid.iloc[0]
    return non_null.iloc[0]


def deduplicate_clonotype_rows(
    df: pd.DataFrame,
    *,
    subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    duplicate_count_col: str = "duplicate_count",
) -> pd.DataFrame:
    """Aggregate repeated clonotype rows to one row per unique clonotype.

    ``duplicate_count`` values are summed; all remaining metadata columns keep
    the first non-empty value observed within each group.
    """
    group_cols = [col for col in subset if col in df.columns]
    if not group_cols:
        return df.copy()

    agg: dict[str, str | callable] = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if col == duplicate_count_col:
            agg[col] = "sum"
        else:
            agg[col] = _first_nonempty

    dedup = df.groupby(group_cols, sort=False, dropna=False, as_index=False).agg(agg)
    if duplicate_count_col not in dedup.columns:
        dedup[duplicate_count_col] = 1
    dedup[duplicate_count_col] = (
        pd.to_numeric(dedup[duplicate_count_col], errors="coerce").fillna(1).astype(int)
    )
    dedup = dedup.reset_index(drop=True)
    dedup["row_id"] = dedup.index.astype(str)
    return dedup


# ---------------------------------------------------------------------------
# Internal builders (pure Python/C, no closures — needed for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _token_table_for_family(
    clones: list[Clonotype],
    family: TOKEN_FAMILY,
):
    if family == "vpos3":
        family = "pos3"
    if family == "v3":
        return tokenize_rearrangements(clones, k=3, mask_byte=None)
    if family == "pos3":
        return tokenize_rearrangements(clones, k=3, mask_byte=None)
    if family == "u4":
        return tokenize_rearrangements(clones, k=4, mask_byte=None)
    if family == "g4":
        return tokenize_rearrangements(clones, k=4, mask_byte=ord("X"))
    if family == "g5":
        return tokenize_rearrangements(clones, k=5, mask_byte=ord("X"))
    raise ValueError(f"Unknown GLIPH token family: {family}")


def _token_from_match(
    family: TOKEN_FAMILY,
    kmer,
    match,
    *,
    position_offset: int = 0,
) -> str:
    if family == "vpos3":
        family = "pos3"
    seq = kmer.seq.decode("ascii")
    v_base = (kmer.v_gene or "").split("*")[0]
    if family == "v3":
        return f"v3::{v_base}::{seq}"
    if family == "pos3":
        return f"pos3::{v_base}::{position_offset + match.position}::{seq}"
    if family == "u4":
        return f"u4::{seq}"
    if family == "g4":
        return f"g4::{seq}"
    if family == "g5":
        return f"g5::{seq}"
    raise ValueError(f"Unknown GLIPH token family: {family}")


def _build_artifacts_from_clones(
    clones: list[Clonotype],
    family: TOKEN_FAMILY,
    count_mode: COUNT_MODE,
) -> GliphTokenArtifacts:
    """Build token artifacts for one family from a clonotype list."""
    token_table = _token_table_for_family(clones, family)

    occurrence_counts: Counter[str] = Counter()
    token_to_clone: dict[str, set[str]] = defaultdict(set)
    clone_to_tokens: dict[str, set[str]] = defaultdict(set)

    for kmer, matches in token_table.items():
        for match in matches:
            token = _token_from_match(family, kmer, match)
            rid = str(match.rearrangement.id)
            occurrence_counts[token] += 1
            token_to_clone[token].add(rid)
            clone_to_tokens[rid].add(token)

    clonotype_counts = {token: len(cloneset) for token, cloneset in token_to_clone.items()}
    counts = dict(clonotype_counts if count_mode == "clonotype" else occurrence_counts)

    return GliphTokenArtifacts(
        counts=counts,
        token_to_clone=dict(token_to_clone),
        clone_to_tokens=dict(clone_to_tokens),
        occurrence_counts=dict(occurrence_counts),
        clonotype_counts=clonotype_counts,
        count_mode=count_mode,
    )


# ---------------------------------------------------------------------------
# Process-pool worker functions — must be importable at module level
# ---------------------------------------------------------------------------


def _worker_extract(
    chunk_df: pd.DataFrame,
    family: TOKEN_FAMILY,
    count_mode: COUNT_MODE,
    trim_first: int = 3,
    trim_last: int = 4,
) -> GliphTokenArtifacts:
    """ThreadPoolExecutor worker for GLIPH token extraction."""
    repertoire = _locus_repertoire_from_dataframe(chunk_df, locus="TRB")
    clones = repertoire_to_clonotypes(
        repertoire,
        trim_first=trim_first,
        trim_last=trim_last,
    )
    token_table = _token_table_for_family(clones, family)

    occurrence_counts: Counter[str] = Counter()
    token_to_clone: dict[str, set[str]] = defaultdict(set)
    clone_to_tokens: dict[str, set[str]] = defaultdict(set)

    for kmer, matches in token_table.items():
        for match in matches:
            token = _token_from_match(
                family,
                kmer,
                match,
                position_offset=trim_first,
            )
            rid = str(match.rearrangement.id)
            occurrence_counts[token] += 1
            token_to_clone[token].add(rid)
            clone_to_tokens[rid].add(token)

    clonotype_counts = {token: len(cloneset) for token, cloneset in token_to_clone.items()}
    counts = dict(clonotype_counts if count_mode == "clonotype" else occurrence_counts)

    return GliphTokenArtifacts(
        counts=counts,
        token_to_clone=dict(token_to_clone),
        clone_to_tokens=dict(clone_to_tokens),
        occurrence_counts=dict(occurrence_counts),
        clonotype_counts=clonotype_counts,
        count_mode=count_mode,
    )


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def _merge_artifact_parts(
    parts: list[GliphTokenArtifacts],
) -> GliphTokenArtifacts:
    """Merge partial artifact objects produced by threaded workers."""
    if not parts:
        return GliphTokenArtifacts(
            counts={},
            token_to_clone={},
            clone_to_tokens={},
            occurrence_counts={},
            clonotype_counts={},
            count_mode="clonotype",
        )

    count_mode = parts[0].count_mode
    merged_occurrences: Counter[str] = Counter()
    merged_t2c: dict[str, set[str]] = defaultdict(set)
    merged_c2t: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        merged_occurrences.update(part.occurrence_counts)
        for tok, rids in part.token_to_clone.items():
            merged_t2c[tok].update(rids)
        for rid, toks in part.clone_to_tokens.items():
            merged_c2t[rid].update(toks)

    merged_clonotype_counts = {
        token: len(rids) for token, rids in merged_t2c.items()
    }
    merged_counts = (
        dict(merged_clonotype_counts)
        if count_mode == "clonotype"
        else dict(merged_occurrences)
    )
    return GliphTokenArtifacts(
        counts=merged_counts,
        token_to_clone=dict(merged_t2c),
        clone_to_tokens=dict(merged_c2t),
        occurrence_counts=dict(merged_occurrences),
        clonotype_counts=merged_clonotype_counts,
        count_mode=count_mode,
    )


def _split_dataframe(df: pd.DataFrame, threads: int) -> list[pd.DataFrame]:
    """Split a DataFrame into row-wise chunks while preserving DataFrame type."""
    if threads <= 1 or len(df) == 0:
        return [df]

    boundaries = np.linspace(0, len(df), num=threads + 1, dtype=int)
    chunks: list[pd.DataFrame] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if start == stop:
            continue
        chunks.append(df.iloc[start:stop].reset_index(drop=True))
    return chunks


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------


def extract_gliph_token_artifacts(
    df: pd.DataFrame,
    family: TOKEN_FAMILY,
    *,
    threads: int = 1,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    trim_first: int = 3,
    trim_last: int = 4,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract GLIPH-style token artifacts for one token family.

    Parameters
    ----------
    df : pd.DataFrame
        Clonotype table with columns ``row_id``, ``junction_aa``, ``v_gene``,
        ``j_gene`` (optional), ``duplicate_count``.
    family : {"v3", "pos3", "u4", "g4", "g5"}
        Token family to extract.
    threads : int, optional
        Number of worker threads (default ``1``).
    count_mode : {"occurrence", "clonotype"}, optional
        Whether enrichment counts should reflect raw token occurrences or token
        presence per unique clonotype (default ``"clonotype"``).
    unique_clonotypes : bool, optional
        When ``True``, aggregate repeated clonotype rows before tokenisation.
    unique_subset : tuple[str, ...], optional
        Columns defining clonotype uniqueness when ``unique_clonotypes=True``.
    trim_first : int, optional
        Number of amino acids to trim from start of junction_aa (default 3).
    trim_last : int, optional
        Number of amino acids to trim from end of junction_aa (default 4).
    n_workers : int, optional
        Backward-compatible alias for ``threads``.
    """
    if n_workers is not None:
        warnings.warn(
            "n_workers is deprecated for GLIPH extraction; use threads instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        threads = n_workers

    if unique_clonotypes:
        df = deduplicate_clonotype_rows(df, subset=unique_subset)
    else:
        df = df.copy()

    if threads <= 1:
        return _worker_extract(df, family=family, count_mode=count_mode, trim_first=trim_first, trim_last=trim_last)

    chunks = _split_dataframe(df.reset_index(drop=True), threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [
            pool.submit(_worker_extract, chunk, family, count_mode, trim_first, trim_last)
            for chunk in chunks
        ]
        parts = [future.result() for future in futures]
    return _merge_artifact_parts(parts)


def extract_v3mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    trim_first: int = 3,
    trim_last: int = 4,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract V-gene anchored 3-mer (V+3-mer) token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="v3",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        trim_first=trim_first,
        trim_last=trim_last,
        n_workers=n_workers,
    )


def extract_vpos3mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Deprecated alias for :func:`extract_pos3mer_artifacts`."""
    warnings.warn(
        "extract_vpos3mer_artifacts is deprecated; use extract_pos3mer_artifacts.",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_pos3mer_artifacts(
        df,
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        n_workers=n_workers,
    )


def extract_pos3mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    trim_first: int = 3,
    trim_last: int = 4,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract pos+3-mer token artifacts (V-gene + junction position + 3-mer).
    
    Position in returned tokens is reported against original CDR3 coordinates:
    ``reported_pos = trim_first + pos_in_trimmed_cdr3``.
    """
    return extract_gliph_token_artifacts(
        df,
        family="pos3",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        trim_first=trim_first,
        trim_last=trim_last,
        n_workers=n_workers,
    )


def extract_u4mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    trim_first: int = 3,
    trim_last: int = 4,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract ungapped 4-mer token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="u4",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        trim_first=trim_first,
        trim_last=trim_last,
        n_workers=n_workers,
    )


def extract_g4mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    trim_first: int = 3,
    trim_last: int = 4,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract gapped 4-mer token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="g4",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        trim_first=trim_first,
        trim_last=trim_last,
        n_workers=n_workers,
    )


def extract_g5mer_artifacts(
    df: pd.DataFrame,
    threads: int = 1,
    *,
    count_mode: COUNT_MODE = "clonotype",
    unique_clonotypes: bool = False,
    unique_subset: tuple[str, ...] = _DEFAULT_UNIQUE_CLONOTYPE_COLUMNS,
    trim_first: int = 3,
    trim_last: int = 4,
    n_workers: int | None = None,
) -> GliphTokenArtifacts:
    """Extract gapped 5-mer token artifacts."""
    return extract_gliph_token_artifacts(
        df,
        family="g5",
        threads=threads,
        count_mode=count_mode,
        unique_clonotypes=unique_clonotypes,
        unique_subset=unique_subset,
        trim_first=trim_first,
        trim_last=trim_last,
        n_workers=n_workers,
    )


# ---------------------------------------------------------------------------
# Full-GLIPH graph helpers (moved to mir.graph.token_graph)
# ---------------------------------------------------------------------------

# For backward compatibility, import and re-export graph construction helpers.
# These are now in mir.graph.token_graph:
#  - combine_enriched_token_maps
#  - build_full_gliph_clonotype_graph  
#  - build_kmer_projection_graph

from mir.graph.token_graph import (
    combine_enriched_token_maps,
    build_full_gliph_clonotype_graph,
    build_kmer_projection_graph,
)

__all__ = [
    "GliphTokenArtifacts",
    "extract_v3mer_artifacts",
    "extract_pos3mer_artifacts",
    "extract_vpos3mer_artifacts",
    "extract_u4mer_artifacts",
    "extract_g4mer_artifacts",
    "extract_g5mer_artifacts",
    "extract_gliph_token_artifacts",
    "combine_enriched_token_maps",
    "build_full_gliph_clonotype_graph",
    "build_kmer_projection_graph",
    "normalize_control_v",
    "normalize_control_vj",
    "deduplicate_clonotype_rows",
    "repertoire_to_clonotypes",
    "rows_to_clonotypes",  # deprecated
]


# ---------------------------------------------------------------------------
# Gene-usage-normalised control resampling
# ---------------------------------------------------------------------------


def _normalize_gene_usage_series(df: pd.DataFrame, gene_col: str) -> pd.Series:
    return df[gene_col].fillna("").astype(str).str.strip().str.split("*").str[0]


def _normalize_control_by_gene_columns(
    sample_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    n: int,
    *,
    gene_columns: tuple[str, ...],
    seed: int = 42,
) -> pd.DataFrame:
    """Resample *control_pool_df* to match unweighted gene usage in *sample_df*.

    **Unweighted** means every clonotype (row) counts once regardless of its
    ``duplicate_count``.  The returned rows all receive ``duplicate_count=1``.

    Algorithm
    ---------
     1. Compute the frequency of the requested base-gene columns in *sample_df*
         (one count per row).
     2. For each gene-usage bucket, sample from the matching rows in
         *control_pool_df* proportionally; use replacement when the control has
         fewer rows than needed.
     3. If a bucket present in the sample is absent from the control, those slots
         are back-filled from the global control at random.
    4. Truncate / pad to exactly *n* rows.

    Parameters
    ----------
    sample_df, control_pool_df : pd.DataFrame
        Must contain the requested gene columns (alleles are stripped).
    n : int
        Target number of control clonotypes after resampling.
    gene_columns : tuple[str, ...]
        Columns whose stripped base-gene usage should be matched.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Resampled control rows with ``duplicate_count=1`` and freshly assigned
        ``row_id`` values (``ctrl_0``, ``ctrl_1``, …).
    """
    rng = np.random.default_rng(seed)

    if not gene_columns:
        raise ValueError("gene_columns must not be empty")

    sample_keys = pd.DataFrame(
        {
            f"g{i}": _normalize_gene_usage_series(sample_df, col)
            for i, col in enumerate(gene_columns)
        }
    )
    for col in sample_keys.columns:
        sample_keys = sample_keys[sample_keys[col] != ""]
    key_counts = sample_keys.groupby(list(sample_keys.columns)).size()
    key_freq = key_counts / float(key_counts.sum())

    # --- Group control by stripped gene-usage key ---
    ctrl = control_pool_df.copy()
    key_cols = []
    for i, col in enumerate(gene_columns):
        key_col = f"_g{i}"
        key_cols.append(key_col)
        ctrl[key_col] = _normalize_gene_usage_series(ctrl, col)
        ctrl = ctrl[ctrl[key_col] != ""]
    ctrl_groups = {
        key: grp for key, grp in ctrl.groupby(key_cols, sort=False)
    }

    # --- Sample per gene-usage bucket ---
    sampled: list[pd.DataFrame] = []
    for key, freq in key_freq.items():
        if not isinstance(key, tuple):
            key = (key,)
        n_target = max(1, round(n * float(freq)))
        grp = ctrl_groups.get(tuple(key))
        if grp is None or len(grp) == 0:
            continue
        replace = n_target > len(grp)
        n_draw = n_target if replace else min(n_target, len(grp))
        samp = grp.sample(n=n_draw, replace=replace, random_state=int(rng.integers(0, 2**31)))
        sampled.append(samp)

    if not sampled:
        # Fallback: pure random sample from whole control
        result = control_pool_df.sample(
            n=min(n, len(control_pool_df)), replace=n > len(control_pool_df),
            random_state=seed,
        )
    else:
        result = pd.concat(sampled, ignore_index=True)
        # Adjust to exactly n rows
        if len(result) > n:
            result = result.sample(n=n, random_state=seed, replace=False)
        elif len(result) < n:
            extra = control_pool_df.sample(
                n=n - len(result), replace=True, random_state=seed,
            )
            result = pd.concat([result, extra], ignore_index=True)

    result = result.drop(columns=key_cols, errors="ignore").reset_index(drop=True)
    result["duplicate_count"] = 1
    result["row_id"] = ["ctrl_" + str(i) for i in range(len(result))]
    return result


def normalize_control_v(
    sample_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    n: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample *control_pool_df* to match the unweighted V usage of *sample_df*."""
    return _normalize_control_by_gene_columns(
        sample_df,
        control_pool_df,
        n,
        gene_columns=("v_gene",),
        seed=seed,
    )


def normalize_control_vj(
    sample_df: pd.DataFrame,
    control_pool_df: pd.DataFrame,
    n: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample *control_pool_df* to match the unweighted VJ usage of *sample_df*."""
    return _normalize_control_by_gene_columns(
        sample_df,
        control_pool_df,
        n,
        gene_columns=("v_gene", "j_gene"),
        seed=seed,
    )
