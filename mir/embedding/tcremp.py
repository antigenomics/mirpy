"""TCREMP: distance-vector embeddings for TCR/BCR clonotypes.

TCREMP (T-Cell Receptor EMbedding with Prototypes) embeds each clonotype
as a flat vector of distances to a fixed set of *prototype* clonotypes.
For clonotype *i* and prototype *k*, three distances are computed:

* **v_ik** — V-germline distance (pre-computed from gene sequences).
* **j_ik** — J-germline distance (pre-computed from gene sequences).
* **junction_ik** — junction amino-acid alignment distance.

The resulting embedding vector is::

    [v_i1, j_i1, junc_i1, v_i2, j_i2, junc_i2, ..., v_iK, j_iK, junc_iK]

where K is the number of prototypes.  The output matrix has shape
``(n_clonotypes, 3 * K)`` and is stored as ``float32`` for
TensorFlow/Keras compatibility.

All distances use the formula ``d(a, b) = s(a,a) + s(b,b) − 2·s(a,b)``
where *s* is the raw alignment score (BLOSUM62 by default).  This ensures
``d(a, a) = 0`` and ``d(a, b) = d(b, a)``.

Typical usage::

    from mir.embedding.tcremp import TCREmp
    from mir.common.clonotype import Clonotype

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
    clonotypes = [Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01",
                            junction_aa="CASSIRSSYEQYF")]
    X = model.embed(clonotypes)  # shape: (1, 3000)
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from mir.common.clonotype import Clonotype
from mir.common.gene_library import GeneLibrary
from mir.distances.aligner import BioAlignerWrapper, JunctionAligner, GermlineAligner, Scoring
from mir.embedding.prototypes import N_PROTOTYPES, load_prototypes

# Backward-compat alias kept for pickled workers from previous API versions
CDRAligner = JunctionAligner


def _build_gene_matrix(
    germline_aligner: GermlineAligner,
    locus: str,
    gene_type: str,
    proto_genes: list[str],
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Build gene→row index, extended distance matrix, and fallback index.

    The returned matrix has the fallback row appended as the last row so
    that a single advanced-index lookup handles both known and unknown
    genes without a copy.

    Args:
        germline_aligner: Aligner built via :meth:`GermlineAligner.from_library`.
        locus: Receptor locus, e.g. ``'TRB'``.
        gene_type: ``'V'`` or ``'J'``.
        proto_genes: Prototype gene alleles (columns of the output matrix).

    Returns:
        Tuple of ``(gene_idx, mat_ext, fallback_idx)`` where *gene_idx* maps
        allele → row index, *mat_ext* has shape ``(n_genes + 1, n_protos)``
        with the fallback row appended, and *fallback_idx* is the index of
        the fallback row (``n_genes``).
    """
    all_genes = sorted(
        germline_aligner._locus_gene_sets.get((locus, gene_type), frozenset())
    )
    gene_idx = {g: i for i, g in enumerate(all_genes)}
    n_protos = len(proto_genes)
    n_genes = len(all_genes)
    mat = np.empty((n_genes, n_protos), dtype=np.float32)
    for i, g in enumerate(all_genes):
        for k, pg in enumerate(proto_genes):
            mat[i, k] = germline_aligner.gene_dist(locus, g, pg)

    fallback_val = germline_aligner._fallback_dist.get((locus, gene_type), 0.0)
    fallback_row = np.full((1, n_protos), fallback_val, dtype=np.float32)
    mat_ext = np.vstack([mat, fallback_row])
    return gene_idx, mat_ext, n_genes  # n_genes is the fallback row index


_REQUIRED_PROTO_COLS = {"v_gene", "j_gene", "junction_aa"}


def _validate_prototypes(df: pl.DataFrame, source: str) -> None:
    missing = _REQUIRED_PROTO_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Prototype file {source!r} is missing required columns: {sorted(missing)}"
        )


class TCREmp:
    """TCREMP distance-vector embedding for immune receptor clonotypes.

    Embeds each input clonotype as a flat float32 vector of distances
    against all prototype clonotypes.

    Use :meth:`from_defaults` to construct an instance for a given
    species/locus with one call.  Use :meth:`from_file` to supply a custom
    prototype set (with a comparability warning).

    Args:
        germline_aligner: Multi-locus :class:`~mir.distances.aligner.GermlineAligner`
            built via :meth:`~mir.distances.aligner.GermlineAligner.from_library`.
        junction_aligner: Scoring instance for junction comparison.
        prototypes: Prototype DataFrame with columns ``v_gene``,
            ``j_gene``, ``junction_aa``.
        locus: Receptor locus, e.g. ``'TRB'``.
        species: Species identifier, e.g. ``'human'``.
    """

    def __init__(
        self,
        germline_aligner: GermlineAligner,
        junction_aligner: Scoring,
        prototypes: pl.DataFrame,
        locus: str,
        species: str,
    ) -> None:
        self.germline_aligner = germline_aligner
        self.junction_aligner = junction_aligner
        self.prototypes = prototypes
        self.locus = locus
        self.species = species
        self._n_prototypes = len(prototypes)
        self._proto_v = prototypes["v_gene"].to_list()
        self._proto_j = prototypes["j_gene"].to_list()
        self._proto_junction = prototypes["junction_aa"].to_list()

        # Pre-compute proto junction self-scores using the windowed score(s,s),
        # which applies v/j offsets.  selfscore_batch uses the full diagonal and
        # must NOT be used here.
        self._proto_junction_selfscores = np.array(
            [junction_aligner.score(s, s) for s in self._proto_junction], dtype=np.float64
        )

        # Build extended V/J lookup tables (last row = fallback for unknown genes).
        self._v_gene_idx, self._v_dist_mat_ext, self._v_fallback_idx = _build_gene_matrix(
            germline_aligner, locus, "V", self._proto_v
        )
        self._j_gene_idx, self._j_dist_mat_ext, self._j_fallback_idx = _build_gene_matrix(
            germline_aligner, locus, "J", self._proto_j
        )

    # Auto-parallelization threshold for default n_jobs=None behavior.
    # Work is chunked by queries (input clonotypes), but each query is scored
    # against all prototypes, so total work still scales with N_queries * N_prototypes.
    # If that workload exceeds this threshold, use all available CPUs.
    AUTO_PARALLEL_WORKLOAD_THRESHOLD = 10_000_000

    @classmethod
    def from_defaults(
        cls,
        species: str = "human",
        locus: str = "TRB",
        n_prototypes: int | None = None,
        junction_method: str = "fixed_gap",
        germline_scoring: Scoring | None = None,
    ) -> TCREmp:
        """Build a :class:`TCREmp` from library defaults.

        Loads the gene library and prototype file for *species*/*locus*,
        computes all pairwise V/J germline distances, and returns a fully
        configured instance ready to embed clonotypes.

        Args:
            species: Species identifier (e.g. ``'human'``).  Aliases such
                as ``'hsa'`` are resolved automatically.
            locus: Receptor locus (e.g. ``'TRB'``).  Aliases such as
                ``'beta'`` are resolved automatically.
            n_prototypes: Number of prototypes to use (first *n* rows from
                the resource file).  Uses all :data:`N_PROTOTYPES` when
                ``None``.
            junction_method: Junction alignment method:

                * ``'fixed_gap'`` (default) — :class:`~mir.distances.aligner.JunctionAligner`
                  with BLOSUM62 and fixed-position gap model (~25 M pairs/s,
                  C-accelerated).
                * ``'biopython'`` — :class:`~mir.distances.aligner.BioAlignerWrapper`
                  full DP alignment (~270 k pairs/s).

            germline_scoring: Scoring function used to compute pairwise
                germline distances.  Defaults to
                :class:`~mir.distances.aligner.BioAlignerWrapper`.

        Returns:
            Configured :class:`TCREmp` instance.

        Raises:
            ValueError: If *junction_method* is not ``'fixed_gap'`` or
                ``'biopython'``.
            FileNotFoundError: If no prototype file exists for the given
                species/locus.

        Example:
            >>> model = TCREmp.from_defaults("human", "TRB", n_prototypes=100)
            >>> model.n_prototypes
            100
        """
        from mir.basic.aliases import normalize_locus_alias, normalize_species_alias
        species_c = normalize_species_alias(species)
        locus_c = normalize_locus_alias(locus)

        if junction_method == "fixed_gap":
            junction_aligner: Scoring = JunctionAligner()
        elif junction_method == "biopython":
            junction_aligner = BioAlignerWrapper()
        else:
            raise ValueError(
                f"Unknown junction_method: {junction_method!r}. "
                "Use 'fixed_gap' or 'biopython'."
            )

        lib = GeneLibrary.load_default(loci={locus_c}, species={species_c})
        germline_aligner = GermlineAligner.from_library(
            lib, loci=[locus_c], scoring=germline_scoring
        )
        prototypes = load_prototypes(species_c, locus_c, n=n_prototypes)
        return cls(germline_aligner, junction_aligner, prototypes, locus_c, species_c)

    @classmethod
    def from_file(
        cls,
        prototypes_path: str | Path,
        species: str = "human",
        locus: str = "TRB",
        junction_method: str = "fixed_gap",
        germline_scoring: Scoring | None = None,
    ) -> TCREmp:
        """Build a :class:`TCREmp` from a user-supplied prototype file.

        .. warning::
            Embeddings produced with a custom prototype set are **not
            comparable** to embeddings produced with a different prototype set.
            It is strongly recommended to use the pre-set default prototypes
            via :meth:`from_defaults`, which are sampled from real control
            repertoires and versioned for reproducibility.

        The file must be tab-separated (or comma-separated) with at least
        the columns ``v_gene``, ``j_gene``, and ``junction_aa``.

        Args:
            prototypes_path: Path to a TSV/CSV file with prototype clonotypes.
            species: Species identifier (e.g. ``'human'``).
            locus: Receptor locus (e.g. ``'TRB'``).
            junction_method: Junction alignment method (``'fixed_gap'`` or
                ``'biopython'``).
            germline_scoring: Scoring for germline distances.

        Returns:
            Configured :class:`TCREmp` instance.

        Raises:
            ValueError: If the file is missing required columns.
            FileNotFoundError: If *prototypes_path* does not exist.
        """
        warnings.warn(
            "TCREmp built from a custom prototype file. "
            "Embeddings produced with different prototype sets are NOT comparable. "
            "Use TCREmp.from_defaults() for reproducible, versioned embeddings.",
            UserWarning,
            stacklevel=2,
        )

        path = Path(prototypes_path)
        if not path.exists():
            raise FileNotFoundError(f"Prototype file not found: {path}")

        sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
        df = pl.read_csv(path, separator=sep)
        _validate_prototypes(df, str(path))

        from mir.basic.aliases import normalize_locus_alias, normalize_species_alias
        species_c = normalize_species_alias(species)
        locus_c = normalize_locus_alias(locus)

        if junction_method == "fixed_gap":
            junction_aligner: Scoring = JunctionAligner()
        elif junction_method == "biopython":
            junction_aligner = BioAlignerWrapper()
        else:
            raise ValueError(
                f"Unknown junction_method: {junction_method!r}. "
                "Use 'fixed_gap' or 'biopython'."
            )

        lib = GeneLibrary.load_default(loci={locus_c}, species={species_c})
        germline_aligner = GermlineAligner.from_library(
            lib, loci=[locus_c], scoring=germline_scoring
        )
        return cls(germline_aligner, junction_aligner, df, locus_c, species_c)

    @property
    def n_prototypes(self) -> int:
        """Number of prototype clonotypes."""
        return self._n_prototypes

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality: ``3 * n_prototypes``."""
        return 3 * self._n_prototypes

    @property
    def cdr3_aligner(self) -> Scoring:
        """Backward-compatibility alias for :attr:`junction_aligner`."""
        return self.junction_aligner

    @property
    def _proto_cdr3(self) -> list[str]:
        """Backward-compatibility alias for :attr:`_proto_junction`."""
        return self._proto_junction

    def _resolve_n_jobs(self, n_queries: int, n_jobs: int | None) -> int:
        """Resolve effective worker count for embedding.

        Policy:
        - Explicit ``n_jobs`` always wins (clamped to at least 1).
        - ``n_jobs=None`` uses workload-aware auto selection.
                - BioPython backend defaults to serial because the Python-level
                    PairwiseAligner calls are relatively fine-grained and thread overhead
                    typically dominates.
        """
        if n_jobs is not None:
            return max(1, int(n_jobs))

        if isinstance(self.junction_aligner, BioAlignerWrapper):
            return 1

        workload = n_queries * self._n_prototypes
        if workload >= self.AUTO_PARALLEL_WORKLOAD_THRESHOLD:
            return os.cpu_count() or 1
        return 1

    def embed(
        self,
        clonotypes: list[Clonotype],
        n_jobs: int | None = None,
    ) -> np.ndarray:
        """Embed clonotypes as distance vectors against all prototypes.

        Uses a fully vectorized path: V/J distances via numpy advanced
        indexing and junction distances via :meth:`~.JunctionAligner.score_matrix`
        (a single C call with the GIL released).  When *n_jobs* > 1 the
        query list is split into chunks processed by a :class:`ThreadPoolExecutor`;
        since ``score_matrix`` releases the GIL, the threads run in true
        parallel.

        Args:
            clonotypes: List of clonotypes to embed.
            n_jobs: Number of parallel worker threads:

                                * ``None`` (default) — auto-select based on workload
                                    ``len(clonotypes) * n_prototypes``:

                                    * uses ``1`` for small workloads.
                                    * uses ``os.cpu_count()`` when workload is at least
                                        ``AUTO_PARALLEL_WORKLOAD_THRESHOLD``.

                                * ``1`` — force serial execution.
                                * ``> 1`` — force explicit worker count.

                                Note: chunking is by clonotypes, but each chunk computes scores
                                against all prototypes, so prototype count still contributes
                                linearly to total cost. For BioPython junction alignment,
                                auto mode always uses serial execution.

        Returns:
            Float32 array of shape ``(n_clonotypes, 3 * n_prototypes)``.
            Column layout::

                [v_1, j_1, junc_1, v_2, j_2, junc_2, ..., v_K, j_K, junc_K]

        Example:
            >>> from mir.common.clonotype import Clonotype
            >>> model = TCREmp.from_defaults("human", "TRB", n_prototypes=10)
            >>> c = Clonotype(v_gene="TRBV10-3*01", j_gene="TRBJ2-7*01",
            ...               junction_aa="CASSIRSSYEQYF")
            >>> X = model.embed([c])
            >>> X.shape
            (1, 30)
            >>> X.dtype
            dtype('float32')
        """
        n = len(clonotypes)
        if n == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        n_jobs = self._resolve_n_jobs(n_queries=n, n_jobs=n_jobs)

        v_genes = [c.v_gene for c in clonotypes]
        j_genes = [c.j_gene for c in clonotypes]
        junctions = [c.junction_aa for c in clonotypes]

        # Vectorized V/J: map to row indices (unknown → fallback row)
        v_idx = np.array([self._v_gene_idx.get(v, self._v_fallback_idx) for v in v_genes])
        j_idx = np.array([self._j_gene_idx.get(j, self._j_fallback_idx) for j in j_genes])
        v_mat = self._v_dist_mat_ext[v_idx]   # (n, n_protos) float32
        j_mat = self._j_dist_mat_ext[j_idx]   # (n, n_protos) float32

        # Junction self-scores for query clonotypes — must use windowed score(s,s)
        # to be consistent with score_matrix (which also applies v/j offsets).
        junc_aligner = self.junction_aligner
        junc_ss = np.array(
            [junc_aligner.score(s, s) for s in junctions], dtype=np.float64
        )

        # Junction cross-scores: (n, n_protos) float64
        if n_jobs == 1 or n == 1:
            cross = junc_aligner.score_matrix(junctions, self._proto_junction)
        else:
            actual_jobs = min(n_jobs, n)
            chunk_size = max(1, (n + actual_jobs - 1) // actual_jobs)
            chunks = [junctions[i: i + chunk_size] for i in range(0, n, chunk_size)]
            cross = np.empty((n, self._n_prototypes), dtype=np.float64)
            with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                futures = [
                    (i * chunk_size, pool.submit(
                        junc_aligner.score_matrix, chunk, self._proto_junction
                    ))
                    for i, chunk in enumerate(chunks)
                ]
                for offset, fut in futures:
                    sub = fut.result()
                    cross[offset: offset + len(sub)] = sub

        junc_mat = (
            junc_ss[:, None] + self._proto_junction_selfscores[None, :] - 2.0 * cross
        ).astype(np.float32)

        result = np.empty((n, self.embedding_dim), dtype=np.float32)
        result[:, 0::3] = v_mat
        result[:, 1::3] = j_mat
        result[:, 2::3] = junc_mat
        return result
