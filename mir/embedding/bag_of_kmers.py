"""Bag-of-k-mers utilities for repertoires and control backgrounds.

This module provides:
- repertoire token-table builders (locus/sample/dataset),
- control-background k-mer profile materialization in memory (default),
- optional cache persistence for repeated workflows,
- enrichment-ready statistics (n, T, P, idf) and positional tables.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mir.basic.alphabets import aa_to_reduced
from mir.basic.tokens import tokenize_gapped_str, tokenize_str
from mir.common.alleles import allele_to_major
from mir.common.control import (
    _DEFAULT_LOCK_POLL_S,
    _DEFAULT_LOCK_TIMEOUT_S,
    _file_lock,
    _wait_for_lock_release,
    ControlManager,
)
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset


@dataclass(frozen=True)
class BagOfKmersParams:
    """Parameters controlling tokenization behavior."""

    use_v: bool = False
    k: int = 3
    gapped: bool = False
    reduced_alphabet: bool = False
    mask_char: str = "X"
    weight_by_duplicate_count: bool = True

    def validate(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if len(self.mask_char) != 1:
            raise ValueError("mask_char must be a single character")

    def suffix(self) -> str:
        mode = "v" if self.use_v else "kmer"
        gap = "gapped" if self.gapped else "plain"
        alpha = "reduced" if self.reduced_alphabet else "full"
        return f"{mode}_{self.k}mer_{gap}_{alpha}"


@dataclass(frozen=True)
class ControlKmerProfile:
    """In-memory representation of a control k-mer profile."""

    token_stats: pd.DataFrame
    position_stats: pd.DataFrame
    metadata: dict[str, Any]


def _strip_allele(gene: str) -> str:
    return allele_to_major(gene)


def _normalize_seq(seq: str, reduced_alphabet: bool) -> str:
    aa = str(seq or "").strip().upper()
    if not aa:
        return ""
    if not reduced_alphabet:
        return aa
    return aa_to_reduced(aa).decode("ascii")


def _token_weight(duplicate_count: Any, weighted: bool) -> int:
    if not weighted:
        return 1
    try:
        w = int(duplicate_count)
    except Exception:
        w = 1
    return 1 if w < 1 else w


def _iter_tokens_for_clonotype(
    *,
    junction_aa: str,
    v_gene: str,
    duplicate_count: Any,
    params: BagOfKmersParams,
):
    seq = _normalize_seq(junction_aa, params.reduced_alphabet)
    n = len(seq)
    if n < params.k:
        return

    v_prefix = _strip_allele(v_gene)
    w = _token_weight(duplicate_count, params.weight_by_duplicate_count)

    if params.gapped:
        tokens = tokenize_gapped_str(seq, params.k, params.mask_char)
        per_window = params.k
        for idx, token in enumerate(tokens):
            pos = idx // per_window
            key = f"{v_prefix}|{token}" if params.use_v else token
            yield key, pos, n, w
        return

    for pos, token in enumerate(tokenize_str(seq, params.k)):
        key = f"{v_prefix}|{token}" if params.use_v else token
        yield key, pos, n, w


def _build_tables_from_df(df: pd.DataFrame, params: BagOfKmersParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_counts: Counter[str] = Counter()
    pos_counts: Counter[tuple[str, int, int]] = Counter()

    if df.empty:
        stats_empty = pd.DataFrame(columns=["token", "n", "T", "p", "idf"])
        pos_empty = pd.DataFrame(columns=["token", "count", "pos", "junction_len"])
        return stats_empty, pos_empty

    has_v = "v_gene" in df.columns
    for rec in df.to_dict(orient="records"):
        for token, pos, junction_len, weight in _iter_tokens_for_clonotype(
            junction_aa=str(rec.get("junction_aa", "")),
            v_gene=str(rec.get("v_gene", "") if has_v else ""),
            duplicate_count=rec.get("duplicate_count", 1),
            params=params,
        ):
            token_counts[token] += weight
            pos_counts[(token, pos, junction_len)] += weight

    if not token_counts:
        stats_empty = pd.DataFrame(columns=["token", "n", "T", "p", "idf"])
        pos_empty = pd.DataFrame(columns=["token", "count", "pos", "junction_len"])
        return stats_empty, pos_empty

    total_kmers = int(sum(token_counts.values()))
    stats_rows = []
    for token, n in token_counts.items():
        p = float(n) / float(total_kmers)
        stats_rows.append(
            {
                "token": token,
                "n": int(n),
                "T": total_kmers,
                "p": p,
                "idf": float(-math.log(p)),
            }
        )

    pos_rows = [
        {
            "token": token,
            "count": int(count),
            "pos": int(pos),
            "junction_len": int(jlen),
        }
        for (token, pos, jlen), count in pos_counts.items()
    ]

    token_df = pd.DataFrame.from_records(stats_rows).sort_values(
        ["n", "token"], ascending=[False, True]
    ).reset_index(drop=True)
    pos_df = pd.DataFrame.from_records(pos_rows).sort_values(
        ["token", "junction_len", "pos"], ascending=[True, True, True]
    ).reset_index(drop=True)
    return token_df, pos_df


def tokenize_locus_repertoire_to_table(
    repertoire: LocusRepertoire,
    *,
    params: BagOfKmersParams,
) -> pd.DataFrame:
    """Build a bag-of-k-mers table for one locus repertoire."""
    params.validate()
    rows = [
        {
            "junction_aa": c.junction_aa,
            "v_gene": c.v_gene,
            "duplicate_count": c.duplicate_count,
        }
        for c in repertoire.clonotypes
    ]
    df = pd.DataFrame.from_records(rows)
    token_df, _ = _build_tables_from_df(df, params)
    if not token_df.empty:
        token_df["locus"] = repertoire.locus
    return token_df


def tokenize_sample_repertoire_by_locus(
    repertoire: SampleRepertoire,
    *,
    params: BagOfKmersParams,
) -> dict[str, pd.DataFrame]:
    """Build per-locus bag-of-k-mers tables for a sample repertoire."""
    out: dict[str, pd.DataFrame] = {}
    for locus, locus_rep in repertoire.loci.items():
        out[locus] = tokenize_locus_repertoire_to_table(locus_rep, params=params)
    return out


def tokenize_dataset_by_sample_and_locus(
    dataset: RepertoireDataset,
    *,
    params: BagOfKmersParams,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build per-sample, per-locus bag-of-k-mers tables for a dataset."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for sample_id, sample in dataset.samples.items():
        out[sample_id] = tokenize_sample_repertoire_by_locus(sample, params=params)
    return out


def control_kmer_profile_name(
    control_type: str,
    species: str,
    locus: str,
    *,
    params: BagOfKmersParams,
) -> str:
    """Return stable control k-mer profile key string.

    Example: ``real_human_TRB_v_3mer_gapped_reduced``.
    """
    ctype = str(control_type or "").strip().lower()
    sp = str(species or "").strip().lower()
    lc = str(locus or "").strip().upper()
    return f"{ctype}_{sp}_{lc}_{params.suffix()}"


def _profile_dir(
    manager: ControlManager,
    *,
    control_type: str,
    species: str,
    locus: str,
    params: BagOfKmersParams,
) -> Path:
    return (
        manager.control_dir
        / "kmer_profiles"
        / control_type
        / species
        / locus
        / params.suffix()
    )


def _profile_lock_path(
    manager: ControlManager,
    *,
    control_type: str,
    species: str,
    locus: str,
    params: BagOfKmersParams,
) -> Path:
    safe = control_kmer_profile_name(control_type, species, locus, params=params).replace("/", "_")
    return manager.control_dir / ".locks" / f"kmer_profile_{safe}.lock"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_df_tsv_gz_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, sep="\t", index=False, compression="gzip")
    os.replace(tmp, path)


def _load_profile_tables(profile_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    token_stats = pd.read_csv(profile_dir / "token_stats.tsv.gz", sep="\t")
    pos_stats = pd.read_csv(profile_dir / "position_stats.tsv.gz", sep="\t")
    with (profile_dir / "profile.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return token_stats, pos_stats, meta


def _build_profile_metadata(
    *,
    control_type: str,
    species: str,
    locus: str,
    params: BagOfKmersParams,
    token_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    cache_enabled: bool,
    paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    total_kmers = int(token_df["T"].iloc[0]) if not token_df.empty else 0
    return {
        "profile_name": control_kmer_profile_name(control_type, species, locus, params=params),
        "control_type": control_type,
        "species": species,
        "locus": locus,
        "params": asdict(params),
        "total_kmers": total_kmers,
        "n_tokens": int(len(token_df)),
        "n_position_rows": int(len(pos_df)),
        "created_at_epoch_s": time.time(),
        "cache_enabled": cache_enabled,
        "paths": dict(paths or {}),
    }


def _build_control_profile_in_memory(
    manager: ControlManager,
    *,
    control_type: str,
    species: str,
    locus: str,
    params: BagOfKmersParams,
    control_kwargs: dict[str, Any],
) -> ControlKmerProfile:
    manager.ensure_control(control_type, species, locus, **control_kwargs)
    control_df = manager.load_control_df(control_type, species, locus)
    token_df, pos_df = _build_tables_from_df(control_df, params)
    meta = _build_profile_metadata(
        control_type=control_type,
        species=species,
        locus=locus,
        params=params,
        token_df=token_df,
        pos_df=pos_df,
        cache_enabled=False,
    )
    return ControlKmerProfile(token_stats=token_df, position_stats=pos_df, metadata=meta)


def build_control_kmer_profile(
    manager: ControlManager,
    *,
    control_type: str,
    species: str,
    locus: str,
    params: BagOfKmersParams,
    control_kwargs: dict[str, Any] | None = None,
    cache: bool = False,
    overwrite_cache: bool = False,
    wait_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
) -> ControlKmerProfile:
    """Build a control k-mer profile.

    Default behavior computes and returns an in-memory profile object without
    writing profile tables to disk.

    Set ``cache=True`` to persist profile tables and metadata under the control
    cache directory and reuse existing cached tables when available.
    """
    params.validate()
    species_c = manager.canonical_species(species)
    locus_c = manager.canonical_locus(locus)
    ctype = str(control_type or "").strip().lower()
    if ctype not in {"real", "synthetic"}:
        raise ValueError("control_type must be 'real' or 'synthetic'")

    kwargs = dict(control_kwargs or {})
    if not cache:
        return _build_control_profile_in_memory(
            manager,
            control_type=ctype,
            species=species_c,
            locus=locus_c,
            params=params,
            control_kwargs=kwargs,
        )

    profile_dir = _profile_dir(
        manager,
        control_type=ctype,
        species=species_c,
        locus=locus_c,
        params=params,
    )
    token_stats_path = profile_dir / "token_stats.tsv.gz"
    pos_stats_path = profile_dir / "position_stats.tsv.gz"
    meta_path = profile_dir / "profile.json"

    lock_path = _profile_lock_path(
        manager,
        control_type=ctype,
        species=species_c,
        locus=locus_c,
        params=params,
    )

    with _file_lock(lock_path, timeout_s=wait_timeout_s):
        if (
            not overwrite_cache
            and token_stats_path.exists()
            and pos_stats_path.exists()
            and meta_path.exists()
        ):
            token_df, pos_df, meta = _load_profile_tables(profile_dir)
            return ControlKmerProfile(token_stats=token_df, position_stats=pos_df, metadata=meta)

        profile = _build_control_profile_in_memory(
            manager,
            control_type=ctype,
            species=species_c,
            locus=locus_c,
            params=params,
            control_kwargs=kwargs,
        )

        profile_dir.mkdir(parents=True, exist_ok=True)
        _write_df_tsv_gz_atomic(profile.token_stats, token_stats_path)
        _write_df_tsv_gz_atomic(profile.position_stats, pos_stats_path)
        meta = _build_profile_metadata(
            control_type=ctype,
            species=species_c,
            locus=locus_c,
            params=params,
            token_df=profile.token_stats,
            pos_df=profile.position_stats,
            cache_enabled=True,
            paths={
                "token_stats": str(token_stats_path),
                "position_stats": str(pos_stats_path),
                "profile_json": str(meta_path),
            },
        )
        _atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True))
        return ControlKmerProfile(
            token_stats=profile.token_stats,
            position_stats=profile.position_stats,
            metadata=meta,
        )


def ensure_control_kmer_profile(
    manager: ControlManager,
    *,
    control_type: str,
    species: str,
    locus: str,
    params: BagOfKmersParams,
    control_kwargs: dict[str, Any] | None = None,
    cache: bool = False,
    overwrite: bool = False,
    wait_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
) -> ControlKmerProfile:
    """Build a control k-mer profile with in-memory default behavior.

    Args:
        cache: If ``True``, persist/reuse profile tables in control cache.
        overwrite: When ``cache=True``, force recompute and overwrite cache.
    """
    return build_control_kmer_profile(
        manager,
        control_type=control_type,
        species=species,
        locus=locus,
        params=params,
        control_kwargs=control_kwargs,
        cache=cache,
        overwrite_cache=overwrite,
        wait_timeout_s=wait_timeout_s,
    )


def load_control_kmer_profile(
    manager: ControlManager,
    *,
    control_type: str,
    species: str,
    locus: str,
    params: BagOfKmersParams,
    wait_if_building: bool = True,
    wait_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S,
) -> ControlKmerProfile:
    """Load persisted control k-mer profile tables and metadata from cache."""
    params.validate()
    species_c = manager.canonical_species(species)
    locus_c = manager.canonical_locus(locus)
    ctype = str(control_type or "").strip().lower()

    profile_dir = _profile_dir(
        manager,
        control_type=ctype,
        species=species_c,
        locus=locus_c,
        params=params,
    )
    lock_path = _profile_lock_path(
        manager,
        control_type=ctype,
        species=species_c,
        locus=locus_c,
        params=params,
    )
    if wait_if_building:
        _wait_for_lock_release(lock_path, timeout_s=wait_timeout_s, poll_s=_DEFAULT_LOCK_POLL_S)

    if not profile_dir.exists():
        raise FileNotFoundError(
            "Control k-mer profile not found; run ensure_control_kmer_profile first"
        )
    token_stats, pos_stats, meta = _load_profile_tables(profile_dir)
    return ControlKmerProfile(token_stats=token_stats, position_stats=pos_stats, metadata=meta)
