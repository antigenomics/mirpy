"""TCREMP: distance-vector embeddings for TCR/BCR clonotypes.

TCREMP (T-Cell Receptor EMbedding with Prototypes) embeds each clonotype
as a flat vector of distances to a fixed set of *prototype* clonotypes.
For clonotype *i* and prototype *k*, three distances are computed.  The first
two depend on the embedding ``mode``:

* ``mode="vjcdr3"`` (default): **V-germline**, **J-germline**, **junction**.
* ``mode="cdr123"``: **CDR1**, **CDR2** (both germline V-gene-determined,
  precomputed from the library's region annotations), **CDR3/junction**.

The resulting embedding vector (vjcdr3 shown) is::

    [v_i1, j_i1, junc_i1, v_i2, j_i2, junc_i2, ..., v_iK, j_iK, junc_iK]

where K is the number of prototypes.  The output matrix has shape
``(n_clonotypes, 3 * K)`` regardless of mode and is stored as ``float32`` for
TensorFlow/Keras compatibility.

All distances use the formula ``d(a, b) = s(a,a) + s(b,b) − 2·s(a,b)``
where *s* is the raw alignment score (BLOSUM62 by default).  This ensures
``d(a, a) = 0`` and ``d(a, b) = d(b, a)``.

Typical usage::

    from mir.embedding.tcremp import TCREmp
    from mir.common.clonotype import Clonotype

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
    clonotypes = [Clonotype(v_call="TRBV10-3*01", j_call="TRBJ2-7*01",
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

from mir.common.alleles import allele_to_major, allele_with_default, strip_allele
from mir.common.clonotype import Clonotype
from mir.common.single_cell import LOCUS_PAIR_TO_LOCI, PairedClonotype
from mir.common.gene_library import GeneLibrary
from mir.distances.aligner import BioAlignerWrapper, JunctionAligner, GermlineAligner, Scoring
from mir.embedding.prototypes import N_PROTOTYPES, load_prototypes
from mir.common.metaclonotype import MetaClonotypeClustering
from mir.utils.metaclonotype_clustering import (
    metaclonotypes_from_cluster_labels,
    paired_metaclonotypes_from_pair_labels,
)

# Backward-compat alias kept for pickled workers from previous API versions
CDRAligner = JunctionAligner


def _resolve_gene_idx(gene: str | None, idx_map: dict[str, int], fallback: int) -> int:
    """Look up *gene* in *idx_map* using the resolution cascade.

    Tries: exact allele → major (*01) → bare gene → fallback index.
    """
    for candidate in (allele_with_default(gene), allele_to_major(gene), strip_allele(gene)):
        if candidate and candidate in idx_map:
            return idx_map[candidate]
    return fallback


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
    # A prototype gene absent from the aligner (e.g. a V gene without region
    # annotation in cdr123 mode) yields NaN; treat it as the fallback distance.
    np.nan_to_num(mat, copy=False, nan=fallback_val)
    fallback_row = np.full((1, n_protos), fallback_val, dtype=np.float32)
    mat_ext = np.vstack([mat, fallback_row])
    return gene_idx, mat_ext, n_genes  # n_genes is the fallback row index


_REQUIRED_PROTO_COLS = {"v_call", "j_call", "junction_aa"}


def _validate_prototypes(df: pl.DataFrame, source: str) -> None:
    missing = _REQUIRED_PROTO_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Prototype file {source!r} is missing required columns: {sorted(missing)}"
        )


def _make_junction_aligner(junction_method: str) -> Scoring:
    """Resolve a junction scoring backend by name."""
    if junction_method == "fixed_gap":
        return JunctionAligner()
    if junction_method == "biopython":
        return BioAlignerWrapper()
    raise ValueError(
        f"Unknown junction_method: {junction_method!r}. Use 'fixed_gap' or 'biopython'."
    )


def _build_mode_aligners(
    lib: GeneLibrary,
    locus: str,
    mode: str,
    germline_scoring: Scoring | None,
) -> tuple[GermlineAligner | None, dict[str, GermlineAligner] | None]:
    """Build the germline distance aligners required by *mode*.

    Returns ``(germline_aligner, region_aligners)``: for ``'vjcdr3'`` a single
    V/J :class:`GermlineAligner` and ``None``; for ``'cdr123'`` ``None`` and a
    ``{'cdr1': ..., 'cdr2': ...}`` mapping of region aligners.
    """
    if mode == "vjcdr3":
        return GermlineAligner.from_library(lib, loci=[locus], scoring=germline_scoring), None
    if mode == "cdr123":
        region_aligners = {
            region: GermlineAligner.from_library_region(
                lib, loci=[locus], region=region, scoring=germline_scoring
            )
            for region in ("cdr1", "cdr2")
        }
        return None, region_aligners
    raise ValueError(f"Unknown mode {mode!r}. Use 'vjcdr3' or 'cdr123'.")


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
        prototypes: Prototype DataFrame with columns ``v_call``,
            ``j_call``, ``junction_aa``.
        locus: Receptor locus, e.g. ``'TRB'``.
        species: Species identifier, e.g. ``'human'``.
    """

    #: Supported embedding modes.
    MODES = ("vjcdr3", "cdr123")

    def __init__(
        self,
        germline_aligner: GermlineAligner | None,
        junction_aligner: Scoring,
        prototypes: pl.DataFrame,
        locus: str,
        species: str,
        mode: str = "vjcdr3",
        region_aligners: dict[str, GermlineAligner] | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode {mode!r}. Use one of {self.MODES}.")
        self.germline_aligner = germline_aligner
        self.region_aligners = region_aligners or {}
        self.junction_aligner = junction_aligner
        self.prototypes = prototypes
        self.locus = locus
        self.species = species
        self.mode = mode
        self._n_prototypes = len(prototypes)
        self._proto_v = [allele_with_default(g) for g in prototypes["v_call"].to_list()]
        self._proto_j = [allele_with_default(g) for g in prototypes["j_call"].to_list()]
        self._proto_junction = prototypes["junction_aa"].to_list()

        # Pre-compute proto junction self-scores using the windowed score(s,s),
        # which applies v/j offsets.  selfscore_batch uses the full diagonal and
        # must NOT be used here.
        self._proto_junction_selfscores = np.array(
            [junction_aligner.score(s, s) for s in self._proto_junction], dtype=np.float64
        )

        # Two germline distance components (the third is always the junction).
        # vjcdr3: [V (by v_call), J (by j_call)]; cdr123: [CDR1, CDR2] both
        # V-gene-determined and looked up by v_call.  Component matrices have a
        # fallback row appended so unknown genes index the last row.
        if mode == "vjcdr3":
            self._comp1_gene = "v_call"
            self._comp2_gene = "j_call"
            self._comp1_idx, self._comp1_mat_ext, self._comp1_fallback_idx = _build_gene_matrix(
                germline_aligner, locus, "V", self._proto_v
            )
            self._comp2_idx, self._comp2_mat_ext, self._comp2_fallback_idx = _build_gene_matrix(
                germline_aligner, locus, "J", self._proto_j
            )
        else:  # cdr123
            self._comp1_gene = "v_call"
            self._comp2_gene = "v_call"
            self._comp1_idx, self._comp1_mat_ext, self._comp1_fallback_idx = _build_gene_matrix(
                self.region_aligners["cdr1"], locus, "V", self._proto_v
            )
            self._comp2_idx, self._comp2_mat_ext, self._comp2_fallback_idx = _build_gene_matrix(
                self.region_aligners["cdr2"], locus, "V", self._proto_v
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
        mode: str = "vjcdr3",
    ) -> TCREmp:
        """Build a :class:`TCREmp` from library defaults.

        Loads the gene library and prototype file for *species*/*locus*,
        computes the pairwise germline distances required by *mode*, and returns
        a fully configured instance ready to embed clonotypes.

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
            mode: Embedding feature set:

                * ``'vjcdr3'`` (default) — V-gene, J-gene, and CDR3/junction
                  distances.
                * ``'cdr123'`` — CDR1, CDR2 (both germline V-gene-determined,
                  precomputed from the library's region annotations) and
                  CDR3/junction distances.  Requires ``region_annotations.txt``.

        Returns:
            Configured :class:`TCREmp` instance.

        Raises:
            ValueError: If *junction_method* / *mode* is invalid, or (for
                ``'cdr123'``) the library lacks region annotations.
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
        junction_aligner = _make_junction_aligner(junction_method)

        lib = GeneLibrary.load_default(loci={locus_c}, species={species_c})
        germline_aligner, region_aligners = _build_mode_aligners(
            lib, locus_c, mode, germline_scoring
        )
        prototypes = load_prototypes(species_c, locus_c, n=n_prototypes)
        return cls(germline_aligner, junction_aligner, prototypes, locus_c, species_c,
                   mode=mode, region_aligners=region_aligners)

    @classmethod
    def from_file(
        cls,
        prototypes_path: str | Path,
        species: str = "human",
        locus: str = "TRB",
        junction_method: str = "fixed_gap",
        germline_scoring: Scoring | None = None,
        mode: str = "vjcdr3",
    ) -> TCREmp:
        """Build a :class:`TCREmp` from a user-supplied prototype file.

        .. warning::
            Embeddings produced with a custom prototype set are **not
            comparable** to embeddings produced with a different prototype set.
            It is strongly recommended to use the pre-set default prototypes
            via :meth:`from_defaults`, which are sampled from real control
            repertoires and versioned for reproducibility.

        The file must be tab-separated (or comma-separated) with at least
        the columns ``v_call``, ``j_call``, and ``junction_aa``.

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
        junction_aligner = _make_junction_aligner(junction_method)

        lib = GeneLibrary.load_default(loci={locus_c}, species={species_c})
        germline_aligner, region_aligners = _build_mode_aligners(
            lib, locus_c, mode, germline_scoring
        )
        return cls(germline_aligner, junction_aligner, df, locus_c, species_c,
                   mode=mode, region_aligners=region_aligners)

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
                ``v_call`` and ``j_call`` values without an explicit allele
                suffix are normalized to ``*01`` before matrix lookup.
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
            Column layout (per prototype, interleaved)::

                vjcdr3: [v_1, j_1, junc_1, ..., v_K, j_K, junc_K]
                cdr123: [cdr1_1, cdr2_1, junc_1, ..., cdr1_K, cdr2_K, junc_K]

        Example:
            >>> from mir.common.clonotype import Clonotype
            >>> model = TCREmp.from_defaults("human", "TRB", n_prototypes=10)
            >>> c = Clonotype(v_call="TRBV10-3*01", j_call="TRBJ2-7*01",
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

        junctions = [c.junction_aa for c in clonotypes]

        # Vectorized germline components: resolve each gene via cascade
        # (exact → *01 → bare → fallback).  Component lookup gene depends on mode:
        # vjcdr3 → (v_call, j_call); cdr123 → (v_call, v_call).
        comp1_genes = [getattr(c, self._comp1_gene) for c in clonotypes]
        comp2_genes = [getattr(c, self._comp2_gene) for c in clonotypes]
        c1_idx = np.array([_resolve_gene_idx(g, self._comp1_idx, self._comp1_fallback_idx) for g in comp1_genes])
        c2_idx = np.array([_resolve_gene_idx(g, self._comp2_idx, self._comp2_fallback_idx) for g in comp2_genes])
        v_mat = self._comp1_mat_ext[c1_idx]   # (n, n_protos) float32
        j_mat = self._comp2_mat_ext[c2_idx]   # (n, n_protos) float32

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

        # Blockwise junction distance matrix computation to avoid giant float64 intermediate.
        # Process in chunks sized to fit in L3 cache; convert to float32 immediately.
        junc_mat = np.empty((n, self._n_prototypes), dtype=np.float32)
        chunk_size = max(1, min(50_000, (n + 7) // 8))  # ~64 MiB per chunk for float32
        for offset in range(0, n, chunk_size):
            end = min(offset + chunk_size, n)
            chunk_junc_ss = junc_ss[offset:end, None]  # (chunk_n, 1)
            chunk_cross = cross[offset:end, :]          # (chunk_n, n_protos)
            junc_mat[offset:end, :] = (
                chunk_junc_ss + self._proto_junction_selfscores[None, :] - 2.0 * chunk_cross
            ).astype(np.float32)

        result = np.empty((n, self.embedding_dim), dtype=np.float32)
        result[:, 0::3] = v_mat
        result[:, 1::3] = j_mat
        result[:, 2::3] = junc_mat
        return result


class PairedTCREmp:
    """TCREmp embedding for paired clonotypes.

    A paired embedding is the concatenation of per-chain TCREmp embeddings in
    canonical locus order for the configured ``locus_pair``.
    """

    def __init__(
        self,
        chain1_model: TCREmp,
        chain2_model: TCREmp,
        locus_pair: str = "TRA_TRB",
    ) -> None:
        if locus_pair not in LOCUS_PAIR_TO_LOCI:
            raise ValueError(
                f"Unsupported locus_pair {locus_pair!r}; "
                f"expected one of {sorted(LOCUS_PAIR_TO_LOCI)}"
            )
        expected_chain1, expected_chain2 = LOCUS_PAIR_TO_LOCI[locus_pair]
        if chain1_model.locus != expected_chain1 or chain2_model.locus != expected_chain2:
            raise ValueError(
                "Chain embedder loci must match locus_pair order: "
                f"expected {(expected_chain1, expected_chain2)}, got "
                f"{(chain1_model.locus, chain2_model.locus)}"
            )

        self.chain1_model = chain1_model
        self.chain2_model = chain2_model
        self.locus_pair = locus_pair
        self.species = chain1_model.species

    @property
    def n_prototypes(self) -> tuple[int, int]:
        return self.chain1_model.n_prototypes, self.chain2_model.n_prototypes

    @property
    def embedding_dim(self) -> int:
        return self.chain1_model.embedding_dim + self.chain2_model.embedding_dim

    @classmethod
    def from_defaults(
        cls,
        species: str = "human",
        locus_pair: str = "TRA_TRB",
        n_prototypes: int | None = None,
        junction_method: str = "fixed_gap",
        germline_scoring: Scoring | None = None,
        mode: str = "vjcdr3",
    ) -> PairedTCREmp:
        """Build a paired embedder by composing two default chain embedders.

        *mode* (``'vjcdr3'`` or ``'cdr123'``) is applied to both chains.
        """
        if locus_pair not in LOCUS_PAIR_TO_LOCI:
            raise ValueError(
                f"Unsupported locus_pair {locus_pair!r}; "
                f"expected one of {sorted(LOCUS_PAIR_TO_LOCI)}"
            )
        locus1, locus2 = LOCUS_PAIR_TO_LOCI[locus_pair]
        return cls(
            chain1_model=TCREmp.from_defaults(
                species=species,
                locus=locus1,
                n_prototypes=n_prototypes,
                junction_method=junction_method,
                germline_scoring=germline_scoring,
                mode=mode,
            ),
            chain2_model=TCREmp.from_defaults(
                species=species,
                locus=locus2,
                n_prototypes=n_prototypes,
                junction_method=junction_method,
                germline_scoring=germline_scoring,
                mode=mode,
            ),
            locus_pair=locus_pair,
        )

    def _ordered_pair(self, paired_clonotype: PairedClonotype) -> tuple[Clonotype, Clonotype]:
        locus1, locus2 = LOCUS_PAIR_TO_LOCI[self.locus_pair]
        by_locus = {
            paired_clonotype.clonotype1.locus: paired_clonotype.clonotype1,
            paired_clonotype.clonotype2.locus: paired_clonotype.clonotype2,
        }
        if locus1 not in by_locus or locus2 not in by_locus:
            raise ValueError(
                f"Paired clonotype {paired_clonotype.pair_id!r} does not contain "
                f"required loci {(locus1, locus2)}"
            )
        return by_locus[locus1], by_locus[locus2]

    def embed(
        self,
        paired_clonotypes: list[PairedClonotype],
        n_jobs: int | None = None,
    ) -> np.ndarray:
        """Embed paired clonotypes as concatenated chain embeddings."""
        if not paired_clonotypes:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        chain1_clonotypes: list[Clonotype] = []
        chain2_clonotypes: list[Clonotype] = []
        for paired_clonotype in paired_clonotypes:
            chain1, chain2 = self._ordered_pair(paired_clonotype)
            chain1_clonotypes.append(chain1)
            chain2_clonotypes.append(chain2)

        chain1_emb = self.chain1_model.embed(chain1_clonotypes, n_jobs=n_jobs)
        chain2_emb = self.chain2_model.embed(chain2_clonotypes, n_jobs=n_jobs)
        return np.concatenate([chain1_emb, chain2_emb], axis=1, dtype=np.float32)


def metaclonotypes_from_tcremp_labels(
    clonotypes: list[Clonotype],
    labels: list[int | str] | np.ndarray,
    *,
    include_noise: bool = False,
    noise_labels: set[int | str] | None = None,
) -> MetaClonotypeClustering:
    """Build single-chain metaclonotypes from TCREmp clustering labels.

    This wrapper is agnostic to clustering backend: DBSCAN, OPTICS, and
    VDBSCAN-like algorithms can pass their label arrays directly.
    """
    clonotype_ids = [str(c.sequence_id) for c in clonotypes]
    return metaclonotypes_from_cluster_labels(
        clonotype_ids,
        labels,
        include_noise=include_noise,
        noise_labels=noise_labels,
    )


def paired_metaclonotypes_from_tcremp_labels(
    paired_clonotypes: list[PairedClonotype],
    labels: list[int | str] | np.ndarray,
    *,
    include_noise: bool = False,
    noise_labels: set[int | str] | None = None,
    mock_chain_1_by_pair: dict[str, bool] | None = None,
    mock_chain_2_by_pair: dict[str, bool] | None = None,
) -> MetaClonotypeClustering:
    """Build paired metaclonotypes from paired TCREmp clustering labels.

    Optional mock-chain maps allow preserving imputation provenance from
    paired workflows where one chain was synthetically imputed.
    """
    pair_ids = [str(p.pair_id) for p in paired_clonotypes]
    chain1_ids = [str(p.clonotype1.sequence_id) for p in paired_clonotypes]
    chain2_ids = [str(p.clonotype2.sequence_id) for p in paired_clonotypes]
    return paired_metaclonotypes_from_pair_labels(
        pair_ids,
        chain1_ids,
        chain2_ids,
        labels,
        mock_chain_1_by_pair=mock_chain_1_by_pair,
        mock_chain_2_by_pair=mock_chain_2_by_pair,
        include_noise=include_noise,
        noise_labels=noise_labels,
    )
