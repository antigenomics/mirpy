"""Shared test factories for building Clonotype and LocusRepertoire objects."""

from __future__ import annotations

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire


def make_trb_clone(
    sid: str,
    aa: str,
    *,
    v: str = "TRBV5-1*01",
    j: str = "TRBJ2-7*01",
    dup: int = 1,
) -> Clonotype:
    return Clonotype(
        sequence_id=sid,
        locus="TRB",
        junction_aa=aa,
        v_call=v,
        j_call=j,
        duplicate_count=dup,
        _validate=False,
    )


def make_trb_repertoire(rows: list[tuple[str, str]], **clone_kwargs) -> LocusRepertoire:
    """Build a TRB LocusRepertoire from (sequence_id, junction_aa) pairs."""
    clonotypes = [make_trb_clone(sid, aa, **clone_kwargs) for sid, aa in rows]
    return LocusRepertoire(clonotypes=clonotypes, locus="TRB")
