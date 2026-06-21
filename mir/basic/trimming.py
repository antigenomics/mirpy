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

from .mirseq import translate_linear
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
        # clamp: a normalized distribution can sum to 1 + epsilon (FP), and callers
        # rely on retention in [0, 1] so that (1 - retention) stays non-negative.
        out.append(min(1.0, float(del_dist[: thr + 1].sum())) if thr >= 0 else 0.0)
    return out


def _retention_from_model(m: OlgaModel, max_offset: int
                          ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Per-gene V/J retention profiles from an already-loaded OLGA model."""
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


def retention_profiles(locus: str = "TRB", species: str = "human", max_offset: int = 14
                       ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Per-gene germline-retention profiles for V (from the Cys anchor) and J (from the J anchor).

    Returns ``(V, J)`` dicts mapping gene name -> list of per-offset retention probabilities.
    """
    return _retention_from_model(OlgaModel(locus=locus, species=species), max_offset)


def retention_anchor_depths(retV: dict[str, list[float]], retJ: dict[str, list[float]],
                            v_call: str, j_call: str, cutoff: float = 0.5) -> tuple[int, int]:
    """Germline-anchored N/C block depths for a V/J pair.

    Counts the leading offsets whose retention is ``>= cutoff`` — i.e. how many
    CDR3 positions are reliably germline-encoded from each anchor.  Use as
    data-driven ``n_term``/``c_term`` for :func:`~mir.biomarkers.motif_logo.build_terminal_anchored_logo`
    instead of a fixed guess.
    """
    def _depth(profile: list[float] | None) -> int:
        if not profile:
            return 0
        d = 0
        for p in profile:
            if p < cutoff:
                break
            d += 1
        return d

    return _depth(retV.get(v_call)), _depth(retJ.get(j_call))


_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX = {a: i for i, a in enumerate(_AA20)}

def _germline_aa_from_model(m: OlgaModel, max_offset: int
                            ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Germline CDR3 residues per gene, via the C-native ``mir.basic.mirseq.translate_linear``:
    V from the N-anchor (Cys, frame 0); J from the C-anchor (frame ``len%3`` aligns the reading to
    the C-terminal F/W), indexed inward from the C-terminus. ``translate_linear`` marks an
    incomplete trailing codon with ``'_'``, which we strip."""
    gd = m.genomic_data
    germV = {name: list(translate_linear(gd.cutV_genomic_CDR3_segs[i]).rstrip("_"))[:max_offset]
             for i, name in enumerate(m.v_names)}
    germJ = {}
    for i, name in enumerate(m.j_names):
        seg = gd.cutJ_genomic_CDR3_segs[i]
        aa = translate_linear(seg[len(seg) % 3:]).rstrip("_")
        germJ[name] = list(reversed(aa))[:max_offset]  # [0] = C-terminal (J anchor)
    return germV, germJ


def _position_retention(rv: list[float] | None, rj: list[float] | None, L: int) -> list[float]:
    """Per-position germline-retention: max of V-side (from N-term) and J-side (from C-term)."""
    return [pr for pr, _ in _position_model(rv, rj, None, None, L)]


def _position_model(rv: list[float] | None, rj: list[float] | None,
                    gv: list[str] | None, gj: list[str] | None, L: int
                    ) -> list[tuple[float, str | None]]:
    """Per CDR3 position, return (germline-retention, expected germline residue). The owning side
    is the anchor (V from N-term, J from C-term) with the higher retention at that position."""
    out = []
    for i in range(L):
        a = rv[i] if (rv is not None and i < len(rv)) else 0.0
        k = L - 1 - i
        b = rj[k] if (rj is not None and k < len(rj)) else 0.0
        if a >= b:
            out.append((a, gv[i] if (gv is not None and i < len(gv)) else None))
        else:
            out.append((b, gj[k] if (gj is not None and k < len(gj)) else None))
    return out


class PgenLite:
    """Fast approximate log10 Pgen from germline-retention + insert composition.

    Factorized position model::

        log10 Pgen ≈ log10 P(V) + log10 P(J)
                     + Σ_pos log10( ret_pos + (1 - ret_pos) · f_insert[aa_pos] )

    where ``ret_pos`` is the germline-retention probability at that CDR3 position
    (from :func:`retention_profiles`) and ``f_insert`` is the amino-acid
    composition of the trimmed/insert core.  Germline-flank positions
    (``ret ≈ 1``) are "free" — their contribution is already paid for by the V/J
    gene choice — while insert-core positions (``ret ≈ 0``) cost their insertion
    frequency.  All tables are calibrated once from a single OLGA-generated
    sample; scoring is then O(L) arithmetic per sequence with **no per-query OLGA
    Pgen call**.  Intended for ranking/filtering, not exact Pgen.
    """

    def __init__(self, retV, retJ, germV, germJ, logPV, logPJ, f_insert,
                 fallback_logp, fallback_finsert, germline_match=True):
        self.retV = retV
        self.retJ = retJ
        self.germV = germV  # dict v -> germline CDR3 residues from the N-anchor
        self.germJ = germJ  # dict j -> germline CDR3 residues from the C-anchor (inward)
        self.logPV = logPV
        self.logPJ = logPJ
        self.f_insert = f_insert  # dict aa -> linear insert-region frequency
        self.fallback_logp = fallback_logp
        self.fallback_finsert = fallback_finsert
        # credit a flank position as germline only when its residue matches the translated
        # germline; a mismatch can arise only via insertion and is scored as such (the lever that
        # adds same-length V/J resolution beyond raw length).
        self.germline_match = germline_match

    @classmethod
    def calibrate(cls, locus: str = "TRB", species: str = "human", *, n_calib: int = 20000,
                  max_offset: int = 14, insert_cutoff: float = 0.5, seed: int = 42) -> "PgenLite":
        """Calibrate V/J usage and insert-region composition from one OLGA sample."""
        m = OlgaModel(locus=locus, species=species)
        retV, retJ = _retention_from_model(m, max_offset)
        germV, germJ = _germline_aa_from_model(m, max_offset)
        np.random.seed(seed)
        sg = m.seq_gen_model

        v_counts: dict[str, int] = {}
        j_counts: dict[str, int] = {}
        insert_aa = np.zeros(20)
        n_ok = 0
        for _ in range(n_calib):
            r = sg.gen_rnd_prod_CDR3()
            if r is None:
                continue
            aa, v, j = r[1], m.v_names[r[2]], m.j_names[r[3]]
            v_counts[v] = v_counts.get(v, 0) + 1
            j_counts[j] = j_counts.get(j, 0) + 1
            ret = _position_retention(retV.get(v), retJ.get(j), len(aa))
            for i, a in enumerate(aa):
                idx = _AA_IDX.get(a)
                if idx is not None and ret[i] < insert_cutoff:
                    insert_aa[idx] += 1
            n_ok += 1

        nV, nJ = len(m.v_names), len(m.j_names)
        logPV = {v: float(np.log10((c + 1) / (n_ok + nV))) for v, c in v_counts.items()}
        logPJ = {j: float(np.log10((c + 1) / (n_ok + nJ))) for j, c in j_counts.items()}
        insert_aa += 1.0  # Laplace
        f = insert_aa / insert_aa.sum()
        f_insert = {a: float(f[i]) for a, i in _AA_IDX.items()}
        return cls(
            retV, retJ, germV, germJ, logPV, logPJ, f_insert,
            fallback_logp=float(np.log10(1.0 / (n_ok + max(nV, nJ)))),
            fallback_finsert=float(f.min()),
        )

    def score(self, seqs: list[str], v_calls: list[str], j_calls: list[str]) -> np.ndarray:
        """Approximate log10 Pgen for each ``(seq, v_call, j_call)`` triple."""
        f_insert, f0 = self.f_insert, self.fallback_finsert
        out = np.empty(len(seqs))
        for n, (seq, v, j) in enumerate(zip(seqs, v_calls, j_calls)):
            s = self.logPV.get(v, self.fallback_logp) + self.logPJ.get(j, self.fallback_logp)
            pm = _position_model(self.retV.get(v), self.retJ.get(j),
                                 self.germV.get(v), self.germJ.get(j), len(seq))
            for i, aa in enumerate(seq):
                r, gl = pm[i]
                if self.germline_match and not (gl is not None and gl == aa):
                    r = 0.0  # germline credit only on a residue match; else it is an insertion
                s += np.log10(r + (1.0 - r) * f_insert.get(aa, f0))
            out[n] = s
        return out
