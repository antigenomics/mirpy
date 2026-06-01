"""TCRdist: alignment-based distance metric for TCR clonotypes.

Computes a weighted sum of:

* **V-gene germline distance** — pre-computed from the full V-gene amino-acid
  sequence alignment via :class:`~mir.distances.aligner.GermlineAligner`.
* **CDR3/junction_aa alignment distance** — BLOSUM62 scoring with a
  configurable gap model.

Three CDR3 gap models are available via ``fixed_gaps``:

* ``None`` — full dynamic-programming alignment (BioPython BLOSUM62).
* ``"Mid"`` — a single gap block inserted at the midpoint of the length
  difference between two CDR3s.
* ``[p1, p2, ...]`` — a list of candidate gap positions; the position that
  maximises the alignment score is kept.  This path uses the C-accelerated
  :class:`~mir.distances.aligner.JunctionAligner` and releases the GIL,
  enabling true thread-level parallelism in :meth:`TcrDist.dist_matrix`.

All distance components use the symmetric formula
``d(a, b) = s(a, a) + s(b, b) − 2 · s(a, b)`` which guarantees
``d(a, a) = 0`` and ``d(a, b) ≥ 0``.

**Scale note**: V-gene distances (from :class:`~mir.distances.aligner.BioAlignerWrapper`,
not multiplied by 10) are on a different absolute scale than CDR3 distances
(from :class:`~mir.distances.aligner.JunctionAligner`, multiplied by 10).
The default weights ``w_v=1.0, w_cdr3=3.0`` were chosen so that CDR3
divergence dominates, which is appropriate for identifying convergent
antigen-specific sequences.

References
----------
Dash P, Fiore-Gartland AJ, Hertz T, et al.
Quantifiable predictive features define epitope-specific T cell receptor
repertoires. *Nature.* 2017;547(7661):89-93. doi:10.1038/nature22383.

Mayer-Blackwell K, Schattgen S, Cohen-Lavi L, et al.
TCR meta-clonotypes for biomarker discovery with tcrdist3 and publicly
available repositories of T cell receptor sequences. *eLife.* 2021;10:e68605.
doi:10.7554/eLife.68605.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import polars as pl
from Bio.Align import substitution_matrices

from mir.common.alleles import allele_with_default
from mir.common.clonotype import Clonotype
from mir.common.gene_library import GeneLibrary
from mir.common.metaclonotype import MetaClonotypeClustering
from mir.common.repertoire import LocusRepertoire
from mir.distances.aligner import (
    BioAlignerWrapper,
    GermlineAligner,
    JunctionAligner,
)

_logger = logging.getLogger(__name__)
_BLOSUM62 = substitution_matrices.load("BLOSUM62")

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_seqdist_c():
    """Return the ``seqdist_c`` C extension module or *None* if unavailable."""
    try:
        from mir.distances import seqdist_c as _mod  # type: ignore[import]
        if hasattr(_mod, "score_max"):
            return _mod
    except ImportError:
        pass
    return None


class _MidGapScorer:
    """CDR3 alignment with a gap inserted at the midpoint of length difference.

    When *s1* and *s2* have different lengths, a gap block of the required
    size is inserted at position ``len(shorter) // 2`` in the shorter
    sequence.  Uses BLOSUM62 by default; delegates to the C extension when
    available for the gap-aware path.

    Parameters
    ----------
    mat :
        Amino-acid substitution matrix (BLOSUM62 by default).
    gap_penalty :
        Per-position gap penalty (applied once per inserted gap column).
    """

    _factor = 10.0

    def __init__(self, mat=_BLOSUM62, gap_penalty: float = -4.0):
        self._mat256 = JunctionAligner._build_dense_mat(mat)
        self._mat = mat
        self.gap_penalty = float(gap_penalty)
        self._use_mat = mat is not None
        self._self_cache: dict[str, float] = {}

    def score(self, s1: str, s2: str) -> float:
        """Raw alignment score between *s1* and *s2*."""
        n1, n2 = len(s1), len(s2)
        if n1 == n2:
            return self._score_equal(s1, s2)
        shorter, longer = (s1, s2) if n1 < n2 else (s2, s1)
        p = len(shorter) // 2
        _sd = _get_seqdist_c()
        if _sd is not None:
            return _sd.score_max(
                shorter, longer,
                self._mat256,
                np.array([p], dtype=np.int32),
                self.gap_penalty, 0, 0,
                self._factor, self._use_mat,
            )
        return self._score_with_gap_py(shorter, longer, p)

    def _score_equal(self, s1: str, s2: str) -> float:
        mat = self._mat
        if mat is None:
            return 0.0
        return self._factor * sum(mat[a, b] for a, b in zip(s1, s2))

    def _score_with_gap_py(self, shorter: str, longer: str, p: int) -> float:
        gap_len = len(longer) - len(shorter)
        g0, g1 = p, p + gap_len
        mat = self._mat
        x = 0.0
        for i in range(len(longer)):
            if g0 <= i < g1:
                x += self.gap_penalty
            else:
                j = i if i < g0 else i - gap_len
                if mat is not None:
                    x += mat[shorter[j], longer[i]]
        return self._factor * x

    def score_dist(self, s1: str, s2: str) -> float:
        """Symmetric non-negative distance: ``s(a,a) + s(b,b) − 2·s(a,b)``."""
        ss1 = self._self_cache.get(s1)
        if ss1 is None:
            ss1 = self.score(s1, s1)
            self._self_cache[s1] = ss1
        ss2 = self._self_cache.get(s2)
        if ss2 is None:
            ss2 = self.score(s2, s2)
            self._self_cache[s2] = ss2
        return ss1 + ss2 - 2.0 * self.score(s1, s2)


# ─────────────────────────────────────────────────────────────────────────────
# TcrDist
# ─────────────────────────────────────────────────────────────────────────────

class TcrDist:
    """TCRdist: alignment-based clonotype distance metric.

    Computes a weighted sum of V-gene germline distance and CDR3/junction_aa
    alignment distance:

    .. math::

        d(a, b) = w_v \\cdot d_V(a, b) + w_j \\cdot d_J(a, b)
                + w_{CDR3} \\cdot d_{CDR3}(a, b)

    All components use the symmetric formula
    :math:`d(a, b) = s(a,a) + s(b,b) - 2 \\cdot s(a,b)`.

    V-gene distances are pre-computed at construction from the full V-gene
    amino-acid sequence alignment (``v_alignment_type="full_germline"``).
    ``cdrs_only=True`` requires CDR1/2/2.5 positional markup and raises
    :class:`NotImplementedError`.

    Parameters
    ----------
    locus :
        Receptor locus, e.g. ``"TRB"``.
    species :
        Species, e.g. ``"human"``.
    germline_aligner :
        Pre-built multi-locus germline distance store (built via
        :meth:`~mir.distances.aligner.GermlineAligner.from_library`).
    v_alignment_type :
        V-gene distance method.  Currently only ``"full_germline"``
        is supported.
    cdrs_only :
        Reserved for CDR1/2/2.5-based V scoring; always raises
        :class:`NotImplementedError`.
    w_v, w_j, w_cdr3 :
        Relative weights for the V-gene, J-gene, and CDR3 components.
        Defaults: ``w_v=1.0``, ``w_j=0.0``, ``w_cdr3=3.0``.
    fixed_gaps :
        CDR3 gap model.  ``None`` → BioPython full DP; ``"Mid"`` → midpoint
        gap; list of ints → candidate positions (C-accelerated).
    gap_penalty :
        Per-position gap penalty for CDR3 alignment.
    """

    def __init__(
        self,
        locus: str,
        species: str,
        germline_aligner: GermlineAligner,
        *,
        v_alignment_type: str = "full_germline",
        cdrs_only: bool = False,
        w_v: float = 1.0,
        w_j: float = 0.0,
        w_cdr3: float = 3.0,
        fixed_gaps: None | str | Iterable[int] = (3, 4, -4, -3),
        gap_penalty: float = -4.0,
    ):
        if cdrs_only:
            raise NotImplementedError(
                "cdrs_only requires CDR1/2/2.5 positional markup in the gene "
                "library and is not yet implemented."
            )
        if v_alignment_type != "full_germline":
            raise ValueError(
                f"v_alignment_type must be 'full_germline'; got {v_alignment_type!r}. "
                "CDR-only V scoring is reserved (cdrs_only=True, not implemented)."
            )

        self.locus = locus
        self.species = species
        self.germline_aligner = germline_aligner
        self.v_alignment_type = v_alignment_type
        self.w_v = float(w_v)
        self.w_j = float(w_j)
        self.w_cdr3 = float(w_cdr3)
        self.gap_penalty = float(gap_penalty)
        self.fixed_gaps = fixed_gaps

        if fixed_gaps is None:
            self._cdr3: BioAlignerWrapper | _MidGapScorer | JunctionAligner = BioAlignerWrapper("blastp")
        elif fixed_gaps == "Mid":
            self._cdr3 = _MidGapScorer(gap_penalty=gap_penalty)
        else:
            self._cdr3 = JunctionAligner(
                gap_positions=list(fixed_gaps),
                gap_penalty=gap_penalty,
                v_offset=0,
                j_offset=0,
            )

    @classmethod
    def from_defaults(
        cls,
        locus: str = "TRB",
        species: str = "human",
        *,
        v_alignment_type: str = "full_germline",
        w_v: float = 1.0,
        w_j: float = 0.0,
        w_cdr3: float = 3.0,
        fixed_gaps: None | str | Iterable[int] = (3, 4, -4, -3),
        gap_penalty: float = -4.0,
    ) -> "TcrDist":
        """Construct :class:`TcrDist` using the default OLGA gene library.

        Loads V/J gene sequences for *locus*/*species* and pre-computes all
        pairwise germline distances via
        :meth:`~mir.distances.aligner.GermlineAligner.from_library`.
        The germline aligner build is O(n²) in the number of V/J genes and
        takes a few seconds on first call.

        Args:
            locus: Receptor locus, e.g. ``"TRB"``.
            species: Species, e.g. ``"human"``.
            v_alignment_type: V-gene distance method (``"full_germline"``).
            w_v, w_j, w_cdr3: Component weights.
            fixed_gaps: CDR3 gap model.
            gap_penalty: Per-position gap penalty.

        Returns:
            Ready-to-use :class:`TcrDist` instance.

        Example:
            >>> td = TcrDist.from_defaults("TRB", "human")
            >>> a = Clonotype(v_gene="TRBV19*01", j_gene="TRBJ2-7*01",
            ...               junction_aa="CASSIRSSYEQYF")
            >>> b = Clonotype(v_gene="TRBV19*01", j_gene="TRBJ2-7*01",
            ...               junction_aa="CASSIRASYEQYF")
            >>> td.dist(a, a)
            0.0
        """
        lib = GeneLibrary.load_default(loci={locus}, species={species})
        ga = GermlineAligner.from_library(lib, loci=[locus])
        return cls(
            locus=locus,
            species=species,
            germline_aligner=ga,
            v_alignment_type=v_alignment_type,
            w_v=w_v,
            w_j=w_j,
            w_cdr3=w_cdr3,
            fixed_gaps=fixed_gaps,
            gap_penalty=gap_penalty,
        )

    # ──────────────────────────────────────────────────────────────────
    # Per-pair distance
    # ──────────────────────────────────────────────────────────────────

    def dist(self, cln1: Clonotype, cln2: Clonotype) -> float:
        """Compute TCRdist between two clonotypes.

        Args:
            cln1, cln2: Clonotypes to compare.

        Returns:
            Non-negative distance value (0 when ``cln1 == cln2``).
        """
        d = 0.0
        ga = self.germline_aligner
        locus = self.locus

        if self.w_v and cln1.v_gene and cln2.v_gene:
            d += self.w_v * ga.gene_dist(locus, cln1.v_gene, cln2.v_gene)
        if self.w_j and cln1.j_gene and cln2.j_gene:
            d += self.w_j * ga.gene_dist(locus, cln1.j_gene, cln2.j_gene)
        if self.w_cdr3 and cln1.junction_aa and cln2.junction_aa:
            d += self.w_cdr3 * self._cdr3.score_dist(cln1.junction_aa, cln2.junction_aa)
        return d

    # ──────────────────────────────────────────────────────────────────
    # Batch distances
    # ──────────────────────────────────────────────────────────────────

    def dist_one_to_many(
        self,
        query: Clonotype,
        refs: list[Clonotype],
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Compute TCRdist from one query to many references.

        Args:
            query: Single query clonotype.
            refs: K reference clonotypes.
            n_jobs: Number of threads (``-1`` = all cores).

        Returns:
            Float64 array of shape ``(K,)``.
        """
        return self.dist_matrix([query], refs, n_jobs=n_jobs)[0]

    def dist_matrix(
        self,
        queries: list[Clonotype],
        refs: list[Clonotype],
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Compute N×K TCRdist matrix.

        When ``fixed_gaps`` is a position list, the CDR3 component uses
        :meth:`~mir.distances.aligner.JunctionAligner.score_matrix` which
        releases the GIL, enabling true CPU parallelism across threads.

        Args:
            queries: N query clonotypes.
            refs: K reference clonotypes.
            n_jobs: Number of threads.  ``-1`` uses all physical CPU cores;
                    ``1`` forces serial execution.

        Returns:
            Float64 array of shape ``(N, K)``.
        """
        N, K = len(queries), len(refs)
        if N == 0 or K == 0:
            return np.zeros((N, K), dtype=np.float64)

        q_junc = [c.junction_aa or "" for c in queries]
        r_junc = [c.junction_aa or "" for c in refs]
        q_v = [c.v_gene or "" for c in queries]
        r_v = [c.v_gene or "" for c in refs]
        q_j = [c.j_gene or "" for c in queries]
        r_j = [c.j_gene or "" for c in refs]

        result = np.zeros((N, K), dtype=np.float64)

        if self.w_v:
            result += self.w_v * self._gene_dist_matrix(q_v, r_v)
        if self.w_j:
            result += self.w_j * self._gene_dist_matrix(q_j, r_j)
        if self.w_cdr3:
            result += self.w_cdr3 * self._cdr3_dist_matrix(q_junc, r_junc, n_jobs)

        return result

    def self_dist_matrix(
        self,
        clonotypes: list[Clonotype],
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Compute symmetric N×N TCRdist matrix.

        Args:
            clonotypes: N clonotypes.
            n_jobs: Number of threads.

        Returns:
            Float64 array of shape ``(N, N)``.
        """
        return self.dist_matrix(clonotypes, clonotypes, n_jobs=n_jobs)

    # ──────────────────────────────────────────────────────────────────
    # Radius
    # ──────────────────────────────────────────────────────────────────

    def compute_radius(
        self,
        clonotypes: list[Clonotype],
        background: list[Clonotype],
        *,
        percentile: float = 50.0,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """Compute per-clonotype distance radius against a background set.

        For each query clonotype, computes the *percentile*-th percentile of
        distances to all *background* clonotypes.  The standard TCRdist3
        workflow uses a 50th-percentile (median) background radius to define
        the neighbourhood search threshold.

        Args:
            clonotypes: Query clonotypes to compute radii for.
            background: Background repertoire (random subset or OLGA synthetic
                        sequences from :class:`~mir.basic.pgen.OlgaModel`).
            percentile: Percentile for the radius (50 = median, 95 = 95th).
            n_jobs: Number of threads.

        Returns:
            Float64 array of shape ``(len(clonotypes),)`` with one radius per
            query clonotype.

        Example:
            >>> background = list(LocusRepertoire(...).clonotypes)
            >>> radii = td.compute_radius(hits, background, percentile=50)
        """
        d = self.dist_matrix(clonotypes, background, n_jobs=n_jobs)
        return np.percentile(d, percentile, axis=1)

    # ──────────────────────────────────────────────────────────────────
    # Metaclonotypes
    # ──────────────────────────────────────────────────────────────────

    def find_metaclonotypes(
        self,
        repertoire: LocusRepertoire,
        *,
        max_distance: float,
        representative_ids: list[str] | None = None,
        match_v_gene: bool = False,
        match_j_gene: bool = False,
        cluster_prefix: str = "tcrdist_mc",
        n_jobs: int = 1,
    ) -> MetaClonotypeClustering:
        """Find metaclonotypes via TCRdist radius-threshold clustering.

        Each representative clonotype seeds a cluster containing all
        repertoire members within ``max_distance``.  This mirrors the
        TCRdist3 meta-clonotype workflow (Mayer-Blackwell *et al.* 2021).

        When ``representative_ids`` is ``None``, every clonotype in the
        repertoire is used as a seed (all-vs-all mode, O(N²)).

        Unlike :func:`~mir.common.metaclonotype.metaclonotypes_from_radius_threshold`
        which calls :meth:`dist` per pair, this method computes the full
        distance matrix in one batch operation for efficiency.

        Args:
            repertoire: Target repertoire to cluster.
            max_distance: Maximum TCRdist for cluster membership.
            representative_ids: Seed clonotype IDs.  Defaults to all.
            match_v_gene: Only admit clonotypes with the same V gene as the
                representative into its cluster.
            match_j_gene: Only admit clonotypes with the same J gene.
            cluster_prefix: Prefix for cluster ID strings.
            n_jobs: Number of threads for distance computation.

        Returns:
            :class:`~mir.common.metaclonotype.MetaClonotypeClustering` with
            one cluster per representative.
        """
        clonotypes = repertoire.clonotypes
        by_id = {c.sequence_id: c for c in clonotypes}

        if representative_ids is None:
            representative_ids = [c.sequence_id for c in clonotypes]

        reps = [by_id[rid] for rid in representative_ids if rid in by_id]
        n_missing = len(representative_ids) - len(reps)
        if n_missing:
            _logger.warning(
                "%d representative IDs not found in repertoire; skipping",
                n_missing,
            )

        if not reps:
            return MetaClonotypeClustering(
                pl.DataFrame({
                    "cluster_id": pl.Series([], dtype=pl.Utf8),
                    "clonotype_id": pl.Series([], dtype=pl.Utf8),
                    "is_representative": pl.Series([], dtype=pl.Boolean),
                }),
                paired=False,
            )

        dist_mat = self.dist_matrix(reps, clonotypes, n_jobs=n_jobs)

        # Pre-extract V/J genes for optional V/J filtering
        all_v = [c.v_gene or "" for c in clonotypes]
        all_j = [c.j_gene or "" for c in clonotypes]
        all_ids = [c.sequence_id for c in clonotypes]

        rows: list[dict] = []
        valid_rep_ids = [
            rid for rid in representative_ids if rid in by_id
        ]
        for i, rep_id in enumerate(valid_rep_ids):
            rep = by_id[rep_id]
            cluster_id = f"{cluster_prefix}_{i}"
            rep_v = rep.v_gene or ""
            rep_j = rep.j_gene or ""
            for k, cln_id in enumerate(all_ids):
                if match_v_gene and rep_v != all_v[k]:
                    continue
                if match_j_gene and rep_j != all_j[k]:
                    continue
                if dist_mat[i, k] <= max_distance:
                    rows.append({
                        "cluster_id": cluster_id,
                        "clonotype_id": cln_id,
                        "is_representative": cln_id == rep_id,
                    })

        if not rows:
            return MetaClonotypeClustering(
                pl.DataFrame({
                    "cluster_id": pl.Series([], dtype=pl.Utf8),
                    "clonotype_id": pl.Series([], dtype=pl.Utf8),
                    "is_representative": pl.Series([], dtype=pl.Boolean),
                }),
                paired=False,
            )
        return MetaClonotypeClustering(pl.DataFrame(rows), paired=False)

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _gene_dist_matrix(
        self, q_genes: list[str], r_genes: list[str]
    ) -> np.ndarray:
        """N×K gene distance matrix using pre-computed O(1) lookups.

        Deduplicates query and reference genes, builds a compact unique-pair
        sub-matrix, then expands via numpy advanced indexing.
        """
        q_norm = [allele_with_default(g) if g else "" for g in q_genes]
        r_norm = [allele_with_default(g) if g else "" for g in r_genes]

        N, K = len(q_norm), len(r_norm)
        mat = np.zeros((N, K), dtype=np.float64)

        # Deduplicate while preserving order
        seen_q: dict[str, int] = {}
        for g in q_norm:
            if g and g not in seen_q:
                seen_q[g] = len(seen_q)
        seen_r: dict[str, int] = {}
        for g in r_norm:
            if g and g not in seen_r:
                seen_r[g] = len(seen_r)

        if not seen_q or not seen_r:
            return mat

        unique_q = list(seen_q)
        unique_r = list(seen_r)

        # Compact sub-matrix
        ga = self.germline_aligner
        locus = self.locus
        sub = np.zeros((len(unique_q), len(unique_r)), dtype=np.float64)
        for i, g1 in enumerate(unique_q):
            for j, g2 in enumerate(unique_r):
                sub[i, j] = ga.gene_dist(locus, g1, g2)

        # Map original lists to unique indices (-1 = absent/empty)
        q_idx = np.array([seen_q.get(g, -1) for g in q_norm])
        r_idx = np.array([seen_r.get(g, -1) for g in r_norm])

        valid_q = np.where(q_idx >= 0)[0]
        valid_r = np.where(r_idx >= 0)[0]
        if valid_q.size and valid_r.size:
            mat[np.ix_(valid_q, valid_r)] = sub[
                np.ix_(q_idx[valid_q], r_idx[valid_r])
            ]
        return mat

    def _cdr3_dist_matrix(
        self,
        q_seqs: list[str],
        r_seqs: list[str],
        n_jobs: int,
    ) -> np.ndarray:
        """CDR3 alignment distance matrix using the configured gap model."""
        cdr3 = self._cdr3

        if isinstance(cdr3, JunctionAligner):
            # C-accelerated path; GIL released inside score_matrix
            score_mat = self._junc_score_matrix_threaded(q_seqs, r_seqs, cdr3, n_jobs)
            q_self = cdr3.selfscore_batch(q_seqs)   # shape: (N,)
            r_self = cdr3.selfscore_batch(r_seqs)   # shape: (K,)
            # d(a,b) = s(a,a) + s(b,b) - 2*s(a,b) via broadcasting
            return q_self[:, None] + r_self[None, :] - 2.0 * score_mat

        # Python scorers: BioAlignerWrapper or _MidGapScorer
        N, K = len(q_seqs), len(r_seqs)
        result = np.empty((N, K), dtype=np.float64)

        # Cache self-scores to avoid redundant computation
        self_cache: dict[str, float] = {}
        for s in set(q_seqs) | set(r_seqs):
            if s:
                self_cache[s] = cdr3.score(s, s)

        def _row(i: int) -> None:
            s1 = q_seqs[i]
            if not s1:
                result[i, :] = 0.0
                return
            ss1 = self_cache.get(s1, 0.0)
            for k, s2 in enumerate(r_seqs):
                if not s2:
                    result[i, k] = 0.0
                else:
                    result[i, k] = ss1 + self_cache.get(s2, 0.0) - 2.0 * cdr3.score(s1, s2)

        if n_jobs == 1 or N < 16:
            for i in range(N):
                _row(i)
        else:
            n_workers = os.cpu_count() if n_jobs < 0 else n_jobs
            with ThreadPoolExecutor(max_workers=min(n_workers, N)) as ex:
                list(ex.map(_row, range(N)))

        return result

    def _junc_score_matrix_threaded(
        self,
        q_seqs: list[str],
        r_seqs: list[str],
        scorer: JunctionAligner,
        n_jobs: int,
    ) -> np.ndarray:
        """JunctionAligner.score_matrix with optional thread-level parallelism.

        ``score_matrix`` releases the GIL; splitting queries into chunks and
        running one thread per chunk achieves near-linear scaling up to the
        number of physical CPU cores.
        """
        if n_jobs == 1 or len(q_seqs) < 64:
            return scorer.score_matrix(q_seqs, r_seqs)

        n_workers = os.cpu_count() if n_jobs < 0 else n_jobs
        n_workers = min(n_workers, len(q_seqs))
        chunk_sz = max(1, (len(q_seqs) + n_workers - 1) // n_workers)
        chunks = [q_seqs[i: i + chunk_sz] for i in range(0, len(q_seqs), chunk_sz)]

        result = np.empty((len(q_seqs), len(r_seqs)), dtype=np.float64)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(scorer.score_matrix, chunk, r_seqs) for chunk in chunks]
            offset = 0
            for f, chunk in zip(futures, chunks):
                part = f.result()
                result[offset: offset + len(chunk)] = part
                offset += len(chunk)
        return result
