"""Junction (CDR3) distance matrices via :mod:`seqtree.gapblock`.

``seqtree``'s ``blosum62()`` substitution matrix is already the Gram transform
``s_aa + s_bb − 2·s_ab``, so ``gapblock.score_matrix`` returns the TCREMP junction
*distance* directly (a genuine metric: ``d(a,a)=0``, symmetric, non-negative),
selecting the best of contiguous gap-block placements at ``gap_positions``.
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


def junction_distance_matrix(
    queries,
    refs,
    gap_positions: tuple[int, ...] = DEFAULT_GAP_POSITIONS,
    threads: int = 0,
) -> np.ndarray:
    """Return the ``(len(queries), len(refs))`` junction distance matrix (float32).

    Args:
        queries: Query junction (CDR3) amino-acid strings.
        refs: Reference/prototype junction strings.
        gap_positions: Candidate contiguous gap-block placements; best is chosen.
        threads: Worker threads (``0`` = all cores, GIL released).
    """
    bl = _blosum62()
    sm = score_matrix(
        list(queries),
        list(refs),
        matrix=bl,
        gap_open=2 * bl.scale(),
        gap_extend=1,
        gap_prior=positions_prior(gap_positions),
        threads=threads,
    )
    return np.asarray(sm).astype(np.float32)
