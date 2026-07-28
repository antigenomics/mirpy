"""Junction (CDR3) distance matrices.

The default backend is :mod:`seqtree.gapblock` — a fast contiguous-gap-block approximation to
Smith-Waterman. ``seqtree``'s ``blosum62()`` substitution matrix is already the Gram transform
``s_aa + s_bb − 2·s_ab``, so ``gapblock.score_matrix`` returns the squared dissimilarity ``d``
directly (``d(a,a)=0``, symmetric, non-negative), selecting the best of contiguous gap-block
placements at ``gap_positions``.

Three knobs, all defaulting to the published v3 coordinate system:

* ``alignment`` — ``"gapblock"`` (default, fast, ~5·10⁸ pairs/s) or ``"sw"`` (paper-exact
  Smith-Waterman via BioPython; O(n·K) pairwise, so validation / small-scale only). gap-block
  approximates SW **in the near-neighbour regime that clustering and density use**: close pairs
  agree exactly (e.g. ``CASSTTGLNTEAFF``/``CASSLTAMNTEAFF`` → 26 under both). They part company on
  distant pairs, because ``"sw"`` is a *local* alignment that simply stops extending while gap-block
  scores the whole junction — over all 44 850 pairs of the 300 bundled human-TRB prototypes the two
  rank-correlate ρ≈0.78, with equal-length outliers as far apart as 181 vs 77. The two are therefore
  **different coordinate systems, not interchangeable scales**: use ``"sw"`` to reproduce the paper's
  distances, ``"gapblock"`` (default) for everything else, and never mix them in one embedding.
* ``matrix`` — the substitution matrix for the gap-block backend: any ``seqtree.SubstitutionMatrix``
  (``blosum62()`` default, plus ``pam250()``, ``structural()``, ``unit()``, or ``from_similarity``
  for a fully custom matrix). Changing it is a coordinate-system change (the baked germline blocks
  stay BLOSUM62, so mix scales only deliberately).
* ``metric`` — ``"squared"`` (default) returns the Gram dissimilarity ``d``; ``"sqrt"`` returns the
  induced metric ``ρ=√d``. ``d`` is a *squared* Hilbert distance, so it is not itself a metric (it
  violates the triangle inequality); ``ρ`` is. Benchmarked a wash vs ``d``, so ``d`` is the
  default (the published v3 space). The gap-block placement is a monotone argmin,
  identical under ``d`` and ``√d``; the sqrt is applied to the chosen value.
"""

from __future__ import annotations

import numpy as np
import seqtree
from seqtree.gapblock import positions_prior, score_matrix

DEFAULT_GAP_POSITIONS: tuple[int, ...] = (3, 4, -4, -3)

_BLOSUM62 = None


def _blosum62():
    global _BLOSUM62
    if _BLOSUM62 is None:
        _BLOSUM62 = seqtree.SubstitutionMatrix.blosum62()
    return _BLOSUM62


def _apply_metric(sm, metric: str) -> np.ndarray:
    sm = np.asarray(sm).astype(np.float32)
    if metric == "sqrt":
        return np.sqrt(np.clip(sm, 0.0, None))  # clamp: indefinite BLOSUM ⇒ tiny negatives
    if metric != "squared":
        raise ValueError(f"metric must be 'squared' or 'sqrt', got {metric!r}")
    return sm


def _sw_distance_matrix(queries, refs) -> np.ndarray:
    """Paper-exact Smith-Waterman ``d(a,b)=s(a,a)+s(b,b)−2·s(a,b)`` (local BLOSUM62, linear gap).

    Lazy BioPython (``[build]`` extra). O(n·K) pairwise alignments — for validation and
    small-scale comparison against the gap-block approximation, not whole-repertoire embedding.
    """
    try:
        from Bio.Align import PairwiseAligner, substitution_matrices
    except ImportError as exc:  # BioPython is not a core dep — it ships in [build]
        raise ImportError(
            "alignment='sw' needs BioPython: pip install biopython "
            '(or pip install "mirpy-lib[build]")'
        ) from exc

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0  # linear gap penalty (paper S1/S2)
    q, r = list(queries), list(refs)
    sq = np.array([aligner.score(s, s) for s in q], dtype=np.float64)
    sr = np.array([aligner.score(s, s) for s in r], dtype=np.float64)
    cross = np.array([[aligner.score(a, b) for b in r] for a in q], dtype=np.float64)
    return sq[:, None] + sr[None, :] - 2.0 * cross


def junction_distance_matrix(
    queries,
    refs,
    gap_positions: tuple[int, ...] = DEFAULT_GAP_POSITIONS,
    threads: int = 0,
    *,
    metric: str = "squared",
    matrix=None,
    alignment: str = "gapblock",
) -> np.ndarray:
    """Return the ``(len(queries), len(refs))`` junction distance matrix (float32).

    Args:
        queries: Query junction (CDR3) amino-acid strings.
        refs: Reference/prototype junction strings.
        gap_positions: Candidate contiguous gap-block placements; best is chosen (gap-block only).
        threads: Worker threads (``0`` = all cores, GIL released; gap-block only).
        metric: ``"squared"`` (Gram dissimilarity ``d``, default) or ``"sqrt"`` (metric ``ρ=√d``).
        matrix: A ``seqtree.SubstitutionMatrix`` for the gap-block backend; ``None`` = ``blosum62()``.
        alignment: ``"gapblock"`` (default, fast) or ``"sw"`` (paper-exact Smith-Waterman, slow).
    """
    if alignment == "sw":
        if matrix is not None:  # the SW backend is fixed to BioPython BLOSUM62 — don't silently ignore
            raise ValueError("matrix= is only supported for alignment='gapblock'; "
                             "the 'sw' backend is fixed to BioPython BLOSUM62")
        return _apply_metric(_sw_distance_matrix(queries, refs), metric)
    if alignment != "gapblock":
        raise ValueError(f"alignment must be 'gapblock' or 'sw', got {alignment!r}")
    bl = matrix if matrix is not None else _blosum62()
    sm = score_matrix(
        list(queries),
        list(refs),
        matrix=bl,
        gap_open=2 * bl.scale(),
        gap_extend=1,
        gap_prior=positions_prior(gap_positions),
        threads=threads,
    )
    return _apply_metric(sm, metric)
