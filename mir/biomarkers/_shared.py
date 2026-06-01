"""Shared helpers for biomarker-style repertoire analyses."""

from __future__ import annotations

import typing as t

from mir.common.alleles import strip_allele
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.utils.stats import bh_fdr

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


def lookup_gene_frac(
    match_mode: str,
    v_gene: str,
    j_gene: str,
    fracs: "dict[str, dict]",
    *,
    floor: float = 1e-10,
) -> float:
    """Return P(V), P(J), or P(V,J) from *fracs*, with a floor to avoid division by zero.

    *fracs* must have keys ``"v"``, ``"j"``, and ``"vj"`` mapping to dicts of
    gene-name → probability.  Allele suffixes are stripped before lookup.
    For ``"vj"`` mode, falls back to ``P(V) × P(J)`` when the pair is absent.
    """
    vf = strip_allele(v_gene)
    jf = strip_allele(j_gene)
    if match_mode == "v":
        p = fracs["v"].get(vf, 0.0)
    elif match_mode == "j":
        p = fracs["j"].get(jf, 0.0)
    else:  # "vj"
        p = fracs["vj"].get((vf, jf), 0.0)
        if p == 0.0:
            p = fracs["v"].get(vf, 0.0) * fracs["j"].get(jf, 0.0)
    return max(float(p), floor)


def apply_bh_qvalues_to_metadata(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    metadata_prefix: str,
) -> None:
    """Compute BH-adjusted q-values from per-clonotype p-values in metadata.

    Writes ``{metadata_prefix}_q_value`` in-place for each clonotype per locus.
    """
    p_key = f"{metadata_prefix}_p_value"
    q_key = f"{metadata_prefix}_q_value"
    for _, lrep in iter_loci(repertoire).items():
        clonotypes = list(lrep.clonotypes)
        if not clonotypes:
            continue
        pvals = [float(c.clone_metadata.get(p_key, 1.0)) for c in clonotypes]
        qvals = bh_fdr(pvals)
        for clonotype, q in zip(clonotypes, qvals):
            clonotype.clone_metadata[q_key] = float(q)
