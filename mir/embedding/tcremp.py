"""TCREMP: distance-vector embeddings for TCR/BCR clonotypes.

TCREMP (T-Cell Receptor EMbedding with Prototypes) embeds each clonotype
as a flat vector of distances to a fixed set of *prototype* clonotypes.
For clonotype *i* and prototype *k*, three distances are computed:

* **v_ik** — V-germline distance (pre-computed from gene sequences).
* **j_ik** — J-germline distance (pre-computed from gene sequences).
* **cdr3_ik** — CDR3 amino-acid alignment distance.

The resulting embedding vector is::

    [v_i1, j_i1, cdr3_i1, v_i2, j_i2, cdr3_i2, ..., v_iK, j_iK, cdr3_iK]

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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import polars as pl

from mir.common.clonotype import Clonotype
from mir.common.gene_library import GeneLibrary
from mir.distances.aligner import BioAlignerWrapper, CDRAligner, GermlineAligner, Scoring
from mir.embedding.prototypes import N_PROTOTYPES, load_prototypes


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be at module level for spawn to work)
# ---------------------------------------------------------------------------

def _embed_chunk_worker(
    clono_tuples: list[tuple[str, str, str]],
    proto_cdr3: list[str],
    proto_cdr3_selfscores: np.ndarray,
    v_gene_idx: dict[str, int],
    v_dist_mat: np.ndarray,
    j_gene_idx: dict[str, int],
    j_dist_mat: np.ndarray,
    v_fallback: np.ndarray,
    j_fallback: np.ndarray,
    cdr3_aligner: Scoring,
) -> np.ndarray:
    """Embed a chunk of clonotypes — runs in a subprocess."""
    n_protos = len(proto_cdr3)
    n_clono = len(clono_tuples)
    result = np.empty((n_clono, 3 * n_protos), dtype=np.float32)
    for i, (v, j, cdr3) in enumerate(clono_tuples):
        vi = v_gene_idx.get(v)
        v_dists = v_dist_mat[vi] if vi is not None else v_fallback
        ji = j_gene_idx.get(j)
        j_dists = j_dist_mat[ji] if ji is not None else j_fallback

        cdr3_ss = cdr3_aligner.score(cdr3, cdr3)
        cross = np.fromiter(
            (cdr3_aligner.score(cdr3, s) for s in proto_cdr3),
            dtype=np.float64,
            count=n_protos,
        )
        cdr3_dists = (cdr3_ss + proto_cdr3_selfscores - 2.0 * cross).astype(np.float32)

        result[i, 0::3] = v_dists
        result[i, 1::3] = j_dists
        result[i, 2::3] = cdr3_dists
    return result


def _build_gene_matrix(
    germline_aligner: GermlineAligner,
    locus: str,
    gene_type: str,
    proto_genes: list[str],
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Build gene→row index, distance matrix, and fallback row for one gene type.

    Args:
        germline_aligner: Aligner built via :meth:`GermlineAligner.from_library`.
        locus: Receptor locus, e.g. ``'TRB'``.
        gene_type: ``'V'`` or ``'J'``.
        proto_genes: Prototype gene alleles (columns of the output matrix).

    Returns:
        Tuple of ``(gene_idx, mat, fallback_row)`` where *gene_idx* maps
        allele → row index, *mat* has shape ``(n_genes, n_protos)``, and
        *fallback_row* has shape ``(n_protos,)`` with the max-distance values
        used for any allele absent from the library.
    """
    all_genes = sorted(
        germline_aligner._locus_gene_sets.get((locus, gene_type), frozenset())
    )
    gene_idx = {g: i for i, g in enumerate(all_genes)}
    n_protos = len(proto_genes)
    mat = np.empty((len(all_genes), n_protos), dtype=np.float32)
    for i, g in enumerate(all_genes):
        for k, pg in enumerate(proto_genes):
            mat[i, k] = germline_aligner.gene_dist(locus, g, pg)

    # Fallback row: max distance for any unknown gene
    fallback_val = germline_aligner._fallback_dist.get((locus, gene_type), 0.0)
    fallback_row = np.full(n_protos, fallback_val, dtype=np.float32)
    return gene_idx, mat, fallback_row


class TCREmp:
    """TCREMP distance-vector embedding for immune receptor clonotypes.

    Embeds each input clonotype as a flat float32 vector of distances
    against all :data:`~mir.embedding.prototypes.N_PROTOTYPES` prototype
    clonotypes.

    Use :meth:`from_defaults` to construct an instance for a given
    species/locus with one call.  Alternatively pass pre-built components
    to ``__init__`` directly (e.g. for custom prototype sets or scorings).

    Args:
        germline_aligner: Multi-locus :class:`~mir.distances.aligner.GermlineAligner`
            built via :meth:`~mir.distances.aligner.GermlineAligner.from_library`.
        cdr3_aligner: Scoring instance for CDR3 comparison.  Must implement
            :meth:`~mir.distances.aligner.Scoring.score_dist`.
        prototypes: Prototype DataFrame with columns ``v_gene``,
            ``j_gene``, ``junction_aa``.
        locus: Receptor locus, e.g. ``'TRB'``.
        species: Species identifier, e.g. ``'human'``.
    """

    def __init__(
        self,
        germline_aligner: GermlineAligner,
        cdr3_aligner: Scoring,
        prototypes: pl.DataFrame,
        locus: str,
        species: str,
    ) -> None:
        self.germline_aligner = germline_aligner
        self.cdr3_aligner = cdr3_aligner
        self.prototypes = prototypes
        self.locus = locus
        self.species = species
        self._n_prototypes = len(prototypes)
        self._proto_v = prototypes["v_gene"].to_list()
        self._proto_j = prototypes["j_gene"].to_list()
        self._proto_cdr3 = prototypes["junction_aa"].to_list()

        # Pre-compute proto CDR3 self-scores: avoids K redundant calls per clonotype.
        self._proto_cdr3_selfscores = np.array(
            [cdr3_aligner.score(s, s) for s in self._proto_cdr3], dtype=np.float64
        )

        # Build vectorized V/J lookup tables.
        self._v_gene_idx, self._v_dist_mat, self._v_fallback = _build_gene_matrix(
            germline_aligner, locus, "V", self._proto_v
        )
        self._j_gene_idx, self._j_dist_mat, self._j_fallback = _build_gene_matrix(
            germline_aligner, locus, "J", self._proto_j
        )

    @classmethod
    def from_defaults(
        cls,
        species: str = "human",
        locus: str = "TRB",
        n_prototypes: int | None = None,
        cdr3_method: str = "fixed_gap",
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
            cdr3_method: CDR3 alignment method:

                * ``'fixed_gap'`` (default) — :class:`~mir.distances.aligner.CDRAligner`
                  with BLOSUM62 and fixed-position gap model (~1 M pairs/s,
                  C-accelerated).
                * ``'biopython'`` — :class:`~mir.distances.aligner.BioAlignerWrapper`
                  full DP alignment (~270 k pairs/s).

            germline_scoring: Scoring function used to compute pairwise
                germline distances.  Defaults to
                :class:`~mir.distances.aligner.BioAlignerWrapper`.

        Returns:
            Configured :class:`TCREmp` instance.

        Raises:
            ValueError: If *cdr3_method* is not ``'fixed_gap'`` or
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

        if cdr3_method == "fixed_gap":
            cdr3_aligner: Scoring = CDRAligner()
        elif cdr3_method == "biopython":
            cdr3_aligner = BioAlignerWrapper()
        else:
            raise ValueError(
                f"Unknown cdr3_method: {cdr3_method!r}. "
                "Use 'fixed_gap' or 'biopython'."
            )

        lib = GeneLibrary.load_default(loci={locus_c}, species={species_c})
        germline_aligner = GermlineAligner.from_library(
            lib, loci=[locus_c], scoring=germline_scoring
        )
        prototypes = load_prototypes(species_c, locus_c, n=n_prototypes)
        return cls(germline_aligner, cdr3_aligner, prototypes, locus_c, species_c)

    @property
    def n_prototypes(self) -> int:
        """Number of prototype clonotypes."""
        return self._n_prototypes

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality: ``3 * n_prototypes``."""
        return 3 * self._n_prototypes

    def _embed_one(self, clonotype: Clonotype) -> np.ndarray:
        """Embed a single clonotype; returns float32 array of length ``3*K``."""
        v = clonotype.v_gene
        j = clonotype.j_gene
        cdr3 = clonotype.junction_aa

        vi = self._v_gene_idx.get(v)
        v_dists = self._v_dist_mat[vi] if vi is not None else self._v_fallback

        ji = self._j_gene_idx.get(j)
        j_dists = self._j_dist_mat[ji] if ji is not None else self._j_fallback

        cdr3_ss = self.cdr3_aligner.score(cdr3, cdr3)
        cross = np.fromiter(
            (self.cdr3_aligner.score(cdr3, s) for s in self._proto_cdr3),
            dtype=np.float64,
            count=self._n_prototypes,
        )
        cdr3_dists = (cdr3_ss + self._proto_cdr3_selfscores - 2.0 * cross).astype(np.float32)

        vec = np.empty(3 * self._n_prototypes, dtype=np.float32)
        vec[0::3] = v_dists
        vec[1::3] = j_dists
        vec[2::3] = cdr3_dists
        return vec

    def embed(
        self,
        clonotypes: list[Clonotype],
        n_jobs: int | None = None,
    ) -> np.ndarray:
        """Embed clonotypes as distance vectors against all prototypes.

        Args:
            clonotypes: List of clonotypes to embed.  Each must have
                ``v_gene``, ``j_gene``, and ``junction_aa`` attributes that
                are present in the gene library used at construction.
            n_jobs: Number of parallel worker processes.  Uses
                ``os.cpu_count()`` when ``None``.  Pass ``1`` to disable
                multiprocessing.

        Returns:
            Float32 array of shape ``(n_clonotypes, 3 * n_prototypes)``.
            Column layout::

                [v_1, j_1, cdr3_1, v_2, j_2, cdr3_2, ..., v_K, j_K, cdr3_K]

        Raises:
            KeyError: If a clonotype contains a V/J gene not covered by the
                gene library used at construction.

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

        if n_jobs is None:
            n_jobs = os.cpu_count() or 1

        result = np.empty((n, self.embedding_dim), dtype=np.float32)

        if n_jobs == 1 or n == 1:
            for i, cln in enumerate(clonotypes):
                result[i] = self._embed_one(cln)
            return result

        # Extract raw tuples for pickling — avoids sending full Clonotype objects.
        clono_tuples = [(c.v_gene, c.j_gene, c.junction_aa) for c in clonotypes]
        chunk_size = max(1, (n + n_jobs - 1) // n_jobs)
        chunks = [
            clono_tuples[i : i + chunk_size]
            for i in range(0, n, chunk_size)
        ]

        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as pool:
            futures = [
                pool.submit(
                    _embed_chunk_worker,
                    chunk,
                    self._proto_cdr3,
                    self._proto_cdr3_selfscores,
                    self._v_gene_idx,
                    self._v_dist_mat,
                    self._j_gene_idx,
                    self._j_dist_mat,
                    self._v_fallback,
                    self._j_fallback,
                    self.cdr3_aligner,
                )
                for chunk in chunks
            ]
            offset = 0
            for future, chunk in zip(futures, chunks):
                sub = future.result()
                result[offset : offset + len(chunk)] = sub
                offset += len(chunk)

        return result
