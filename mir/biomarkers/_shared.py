"""Shared helpers for biomarker-style repertoire analyses."""

from __future__ import annotations

import typing as t

from mir.common.repertoire import LocusRepertoire, SampleRepertoire

MatchMode = t.Literal["none", "v", "j", "vj"]


def normalize_match_mode(match_mode: str) -> MatchMode:
    """Normalize public match-mode aliases."""
    mode = str(match_mode).strip().lower().replace("_", "")
    if mode in {"none", "v", "j", "vj"}:
        return t.cast(MatchMode, mode)
    raise ValueError("match_mode must be one of: none, v, j, vj (or v_j)")


def match_flags(match_mode: MatchMode) -> tuple[bool, bool]:
    """Return (match_v, match_j) flags for a normalized match mode."""
    return match_mode in {"v", "vj"}, match_mode in {"j", "vj"}


def iter_loci(
    repertoire: LocusRepertoire | SampleRepertoire,
) -> dict[str, LocusRepertoire]:
    """Expose a uniform locus->repertoire mapping."""
    if isinstance(repertoire, SampleRepertoire):
        return dict(repertoire.loci)
    if isinstance(repertoire, LocusRepertoire):
        return {repertoire.locus: repertoire}
    raise TypeError("repertoire must be LocusRepertoire or SampleRepertoire")
