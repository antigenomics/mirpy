"""Germline-retention profiles from the OLGA recombination model.

The probability that a given CDR3 position (amino-acid offset from the conserved V-Cys, or from the
J-Phe/Trp anchor) is *germline-encoded* rather than trimmed away and replaced by non-template (N)
insertions. This is derived from the OLGA V/J deletion distributions and the germline CDR3-portion
segment lengths, and is the basis for region-aware CDR3 scoring: germline-retained flank positions are
near-invariant, while the trimmed/insert core is where antigen-driven substitution variation lives.

    from mir.basic.trimming import retention_profiles
    V, J = retention_profiles(locus="TRB", species="human")
    V["TRBV19*01"]   # [P(offset 0 germline), P(offset 1 germline), ...] from the V-Cys anchor
    J["TRBJ2-7*01"]  # [...] from the J anchor (C-terminus) inward
"""
from __future__ import annotations

import numpy as np

from .pgen import OlgaModel


def _profile(seg_len_nt: int, del_dist: np.ndarray, max_offset: int) -> list[float]:
    """P(amino-acid offset k is germline-retained) = P(deletions <= seg_len - 3*(k+1)).

    ``del_dist`` is the deletion-count distribution (index = number of trimmed nucleotides). OLGA
    P-nucleotide (negative-deletion) entries, if present at the start of the array, are folded into
    deletion 0 since they only *extend* the germline segment.
    """
    out = []
    for k in range(max_offset):
        thr = seg_len_nt - 3 * (k + 1)  # max deletions that still keep codon k
        out.append(float(del_dist[: thr + 1].sum()) if thr >= 0 else 0.0)
    return out


def retention_profiles(locus: str = "TRB", species: str = "human", max_offset: int = 14
                       ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Per-gene germline-retention profiles for V (from the Cys anchor) and J (from the J anchor).

    Returns ``(V, J)`` dicts mapping gene name -> list of per-offset retention probabilities.
    """
    m = OlgaModel(locus=locus, species=species)
    gd = m.genomic_data
    cutV = gd.cutV_genomic_CDR3_segs
    cutJ = gd.cutJ_genomic_CDR3_segs
    pdelV = np.asarray(m.gen_model.PdelV_given_V)  # [n_del, n_V]
    pdelJ = np.asarray(m.gen_model.PdelJ_given_J)  # [n_del, n_J]
    V = {name: _profile(len(cutV[i]), pdelV[:, i], max_offset)
         for i, name in enumerate(m.v_names)}
    J = {name: _profile(len(cutJ[i]), pdelJ[:, i], max_offset)
         for i, name in enumerate(m.j_names)}
    return V, J
