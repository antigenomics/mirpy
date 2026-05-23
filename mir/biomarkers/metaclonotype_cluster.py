"""Unified metaclonotype clustering interface.

This module provides a single entry-point for metaclonotype clustering that
dispatches to any supported backend (ALICE, TCRNET, TCRdist, edit-distance graph,
TCREmp embeddings, or GLIPH) via a :class:`MetaclonotypeClusterConfig` dataclass.

Paired-chain clustering is supported for all single-chain methods by combining
per-chain cluster IDs (see :func:`cluster_paired_metaclonotypes`).  This lets
you compare the built-in paired TCREmp clustering against a "single-chain then
combine" approach with any other method.

Typical usage::

    from mir.biomarkers.metaclonotype_cluster import (
        MetaclonotypeClusterConfig,
        cluster_metaclonotypes,
        cluster_paired_metaclonotypes,
    )

    # TCRdist single-chain
    cfg = MetaclonotypeClusterConfig(method="tcrdist", locus="TRB", max_distance=24.5)
    meta = cluster_metaclonotypes(rep, cfg)

    # Edit-distance graph, Leiden communities
    cfg_ed = MetaclonotypeClusterConfig(method="edit_distance", graph_algo="leiden")
    meta_ed = cluster_metaclonotypes(rep, cfg_ed)

    # ALICE (requires add_alice_metadata to have been called on rep already)
    cfg_alice = MetaclonotypeClusterConfig(method="alice", q_value_max=0.05)
    meta_alice = cluster_metaclonotypes(rep, cfg_alice)

    # TCREmp paired vs single-chain-combined comparison
    cfg_tcremp = MetaclonotypeClusterConfig(method="tcremp", locus_pair="TRA_TRB")
    meta_paired = cluster_paired_metaclonotypes(paired_locus_rep, cfg_tcremp)

    cfg_tcrnet = MetaclonotypeClusterConfig(method="tcrnet")
    meta_combined = cluster_paired_metaclonotypes(
        paired_locus_rep, cfg_tcrnet, method_chain1=cfg_tcrnet, method_chain2=cfg_tcrnet
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mir.common.metaclonotype import MetaClonotypeClustering
from mir.common.repertoire import LocusRepertoire
from mir.common.single_cell import PairedLocusRepertoire

_VALID_METHODS = frozenset(
    {"alice", "tcrnet", "tcrdist", "edit_distance", "tcremp", "gliph"}
)
_VALID_EMBED_CLUSTER = frozenset({"dbscan", "optics"})
_VALID_GRAPH_ALGO = frozenset({"components", "leiden", "louvain"})


@dataclass
class MetaclonotypeClusterConfig:
    """Configuration for unified metaclonotype clustering.

    All parameters have sensible defaults.  Only ``method`` is required.

    Args:
        method: Clustering backend.  One of ``"alice"``, ``"tcrnet"``,
            ``"tcrdist"``, ``"edit_distance"``, ``"tcremp"``, ``"gliph"``.
        locus: Receptor locus (e.g. ``"TRB"``). Used by TCRdist and TCREmp.
        species: Species string (e.g. ``"human"``). Used by TCRdist and TCREmp.
        locus_pair: Locus pair for paired TCREmp (e.g. ``"TRA_TRB"``).
        n_jobs: Worker count for parallelisable methods.
        cluster_prefix: Cluster ID prefix.  Auto-generated from method if ``None``.
        min_cluster_size: Minimum cluster size retained (graph/embedding methods).

        q_value_max: Maximum q-value for enriched clonotype selection
            (ALICE, TCRNET).  Requires metadata to be pre-set via
            ``add_alice_metadata`` / ``add_tcrnet_metadata``.
        metric: Edit-distance metric (``"hamming"`` or ``"levenshtein"``).
            Used by ALICE, TCRNET, and edit-distance graph.
        threshold: Maximum edit distance for cluster membership (ALICE, TCRNET,
            edit-distance graph).
        match_mode: Gene matching mode for neighborhood expansion
            (``"vj"``, ``"v"``, ``"j"``, ``"none"``).  ALICE / TCRNET only.
        metadata_prefix: Metadata key prefix written by
            ``add_alice_metadata`` / ``add_tcrnet_metadata``.  Defaults to
            the method name if ``None``.

        max_distance: Maximum TCRdist score for cluster membership.
        match_v_gene: Restrict TCRdist neighbors to same V gene.
        match_j_gene: Restrict TCRdist neighbors to same J gene.

        n_prototypes: Number of prototype clonotypes for TCREmp.
        embed_cluster_algo: Embedding-space clustering algorithm for TCREmp
            (``"dbscan"`` or ``"optics"``).
        dbscan_eps: DBSCAN neighbourhood radius (L2 distance in embedding space).
            Only used when ``embed_cluster_algo="dbscan"``.
        dbscan_min_samples: DBSCAN minimum neighbourhood size.
        optics_min_samples: OPTICS minimum cluster size.

        graph_algo: Graph community detection algorithm for edit-distance graph
            and GLIPH (``"components"``, ``"leiden"``, or ``"louvain"``).

        gliph_hamming_threshold: Hamming expansion threshold for GLIPH graph.
        gliph_min_kmer_edge_weight: Minimum k-mer edge weight for GLIPH graph.
    """

    method: str
    locus: str = "TRB"
    species: str = "human"
    locus_pair: str = "TRA_TRB"
    n_jobs: int = 1
    cluster_prefix: str | None = None
    min_cluster_size: int = 2

    # ALICE / TCRNET
    q_value_max: float = 0.05
    metric: str = "hamming"
    threshold: int = 1
    match_mode: str = "vj"
    metadata_prefix: str | None = None

    # TCRdist
    max_distance: float = 24.5
    match_v_gene: bool = False
    match_j_gene: bool = False

    # TCREmp
    n_prototypes: int = 300
    embed_cluster_algo: str = "dbscan"
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 3
    optics_min_samples: int = 3

    # Graph methods
    graph_algo: str = "components"

    # GLIPH extras
    gliph_hamming_threshold: int = 1
    gliph_min_kmer_edge_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method {self.method!r} not in {sorted(_VALID_METHODS)}"
            )
        if self.embed_cluster_algo not in _VALID_EMBED_CLUSTER:
            raise ValueError(
                f"embed_cluster_algo {self.embed_cluster_algo!r} not in "
                f"{sorted(_VALID_EMBED_CLUSTER)}"
            )
        if self.graph_algo not in _VALID_GRAPH_ALGO:
            raise ValueError(
                f"graph_algo {self.graph_algo!r} not in {sorted(_VALID_GRAPH_ALGO)}"
            )

    @property
    def _prefix(self) -> str:
        return self.cluster_prefix or f"{self.method}_mc"


def _cluster_alice(
    rep: LocusRepertoire, cfg: MetaclonotypeClusterConfig
) -> MetaClonotypeClustering:
    from mir.biomarkers.alice import metaclonotypes_from_alice

    return metaclonotypes_from_alice(
        rep,
        metadata_prefix=cfg.metadata_prefix or "alice",
        q_value_max=cfg.q_value_max,
        metric=cfg.metric,  # type: ignore[arg-type]
        threshold=cfg.threshold,
        match_mode=cfg.match_mode,  # type: ignore[arg-type]
    )


def _cluster_tcrnet(
    rep: LocusRepertoire, cfg: MetaclonotypeClusterConfig
) -> MetaClonotypeClustering:
    from mir.biomarkers.tcrnet import metaclonotypes_from_tcrnet

    return metaclonotypes_from_tcrnet(
        rep,
        metadata_prefix=cfg.metadata_prefix or "tcrnet",
        q_value_max=cfg.q_value_max,
        metric=cfg.metric,  # type: ignore[arg-type]
        threshold=cfg.threshold,
        match_mode=cfg.match_mode,  # type: ignore[arg-type]
    )


def _cluster_tcrdist(
    rep: LocusRepertoire, cfg: MetaclonotypeClusterConfig
) -> MetaClonotypeClustering:
    from mir.distances.tcrdist import TcrDist

    tcrdist = TcrDist.from_defaults(cfg.locus, cfg.species)
    return tcrdist.find_metaclonotypes(
        rep,
        max_distance=cfg.max_distance,
        match_v_gene=cfg.match_v_gene,
        match_j_gene=cfg.match_j_gene,
        cluster_prefix=cfg._prefix,
        n_jobs=cfg.n_jobs,
    )


def _cluster_edit_distance(
    rep: LocusRepertoire, cfg: MetaclonotypeClusterConfig
) -> MetaClonotypeClustering:
    from mir.graph.edit_distance_graph import (
        build_edit_distance_graph,
        metaclonotypes_from_edit_distance_graph,
    )

    graph = build_edit_distance_graph(
        rep.clonotypes,
        metric=cfg.metric,
        threshold=cfg.threshold,
        n_jobs=cfg.n_jobs,
    )
    return metaclonotypes_from_edit_distance_graph(
        graph,
        method=cfg.graph_algo,
        min_cluster_size=cfg.min_cluster_size,
    )


def _apply_embed_clustering(
    X: "np.ndarray",  # noqa: F821
    clonotype_ids: list[str],
    cfg: MetaclonotypeClusterConfig,
) -> MetaClonotypeClustering:
    import numpy as np
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.preprocessing import normalize as l2normalize

    from mir.utils.metaclonotype_clustering import metaclonotypes_from_cluster_labels

    X_norm = l2normalize(X.astype(np.float64))
    if cfg.embed_cluster_algo == "dbscan":
        labels = DBSCAN(
            eps=cfg.dbscan_eps,
            min_samples=cfg.dbscan_min_samples,
            metric="euclidean",
        ).fit_predict(X_norm)
    else:
        labels = OPTICS(
            min_samples=cfg.optics_min_samples,
            metric="euclidean",
        ).fit_predict(X_norm)

    return metaclonotypes_from_cluster_labels(
        clonotype_ids,
        labels,
        include_noise=False,
        noise_labels={-1},
    )


def _cluster_tcremp_single(
    rep: LocusRepertoire, cfg: MetaclonotypeClusterConfig
) -> MetaClonotypeClustering:
    from mir.embedding.tcremp import TCREmp

    model = TCREmp.from_defaults(cfg.species, cfg.locus, n_prototypes=cfg.n_prototypes)
    clonotypes = rep.clonotypes
    X = model.embed(clonotypes, n_jobs=cfg.n_jobs)
    ids = [str(c.sequence_id) for c in clonotypes]
    return _apply_embed_clustering(X, ids, cfg)


def _cluster_tcremp_paired(
    paired_rep: PairedLocusRepertoire, cfg: MetaclonotypeClusterConfig
) -> MetaClonotypeClustering:
    from mir.embedding.tcremp import PairedTCREmp, paired_metaclonotypes_from_tcremp_labels

    model = PairedTCREmp.from_defaults(cfg.species, cfg.locus_pair, n_prototypes=cfg.n_prototypes)
    pairs = paired_rep.paired_clonotypes
    X = model.embed(pairs, n_jobs=cfg.n_jobs)

    import numpy as np
    from sklearn.cluster import DBSCAN, OPTICS
    from sklearn.preprocessing import normalize as l2normalize

    X_norm = l2normalize(X.astype(np.float64))
    if cfg.embed_cluster_algo == "dbscan":
        labels = DBSCAN(
            eps=cfg.dbscan_eps,
            min_samples=cfg.dbscan_min_samples,
            metric="euclidean",
        ).fit_predict(X_norm)
    else:
        labels = OPTICS(
            min_samples=cfg.optics_min_samples,
            metric="euclidean",
        ).fit_predict(X_norm)

    return paired_metaclonotypes_from_tcremp_labels(
        pairs,
        labels,
        include_noise=False,
        noise_labels={-1},
    )


def _cluster_gliph(
    rep: LocusRepertoire,
    cfg: MetaclonotypeClusterConfig,
    extra: dict[str, Any],
) -> MetaClonotypeClustering:
    from mir.graph.token_graph import build_gliph_metaclonotypes

    study_df = extra.get("study_df")
    token_to_clones = extra.get("token_to_clones")
    if study_df is None or token_to_clones is None:
        raise ValueError(
            "method='gliph' requires extra={'study_df': ..., 'token_to_clones': ...}. "
            "Run extract_gliph_artifacts_batch_from_repertoire and "
            "combine_enriched_token_maps first."
        )
    return build_gliph_metaclonotypes(
        study_df,
        token_to_clones,
        method=cfg.graph_algo,
        min_cluster_size=cfg.min_cluster_size,
        hamming_threshold=cfg.gliph_hamming_threshold,
        min_kmer_edge_weight=cfg.gliph_min_kmer_edge_weight,
    )


def cluster_metaclonotypes(
    repertoire: LocusRepertoire,
    config: MetaclonotypeClusterConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> MetaClonotypeClustering:
    """Cluster a single-chain repertoire using the specified method.

    Dispatches to the appropriate backend based on ``config.method``:

    * ``"alice"`` — Requires ALICE metadata pre-set via
      :func:`~mir.biomarkers.alice.add_alice_metadata`.
    * ``"tcrnet"`` — Requires TCRNET metadata pre-set via
      :func:`~mir.biomarkers.tcrnet.add_tcrnet_metadata`.
    * ``"tcrdist"`` — Builds :class:`~mir.distances.tcrdist.TcrDist` and
      finds radius clusters.
    * ``"edit_distance"`` — Builds edit-distance graph then applies graph
      community detection.
    * ``"tcremp"`` — Embeds with :class:`~mir.embedding.tcremp.TCREmp` and
      clusters in embedding space.
    * ``"gliph"`` — Requires pre-computed GLIPH data passed via ``extra``
      (keys ``"study_df"`` and ``"token_to_clones"``).

    Args:
        repertoire: Input :class:`~mir.common.repertoire.LocusRepertoire`.
        config: Method and parameter configuration.
        extra: Method-specific extra data.  Currently used only for GLIPH.

    Returns:
        :class:`~mir.common.metaclonotype.MetaClonotypeClustering` with
        single-chain membership table.
    """
    _extra = extra or {}
    method = config.method
    if method == "alice":
        return _cluster_alice(repertoire, config)
    if method == "tcrnet":
        return _cluster_tcrnet(repertoire, config)
    if method == "tcrdist":
        return _cluster_tcrdist(repertoire, config)
    if method == "edit_distance":
        return _cluster_edit_distance(repertoire, config)
    if method == "tcremp":
        return _cluster_tcremp_single(repertoire, config)
    if method == "gliph":
        return _cluster_gliph(repertoire, config, _extra)
    raise ValueError(f"Unknown method {method!r}")  # unreachable after __post_init__


def cluster_paired_metaclonotypes(
    paired_locus_rep: PairedLocusRepertoire,
    config: MetaclonotypeClusterConfig,
    *,
    config_chain1: MetaclonotypeClusterConfig | None = None,
    config_chain2: MetaclonotypeClusterConfig | None = None,
    cluster_separator: str = ".",
    include_unassigned: bool = False,
    extra_chain1: dict[str, Any] | None = None,
    extra_chain2: dict[str, Any] | None = None,
) -> MetaClonotypeClustering:
    """Cluster paired clonotypes using per-chain single-chain methods.

    For TCREmp, this function can optionally use the built-in paired
    embedding (:class:`~mir.embedding.tcremp.PairedTCREmp`) by setting
    ``config.method="tcremp"`` without overriding chain configs.  For all
    other methods, it runs the single-chain clusterer on each chain
    independently and combines cluster IDs via
    :func:`~mir.utils.metaclonotype_clustering.paired_metaclonotypes_from_single_chain`.

    When ``config_chain1`` / ``config_chain2`` are provided they override
    ``config`` for the respective chain.  This lets you use different methods
    per chain or tweak per-chain parameters.

    Args:
        paired_locus_rep: Paired clonotypes for one locus-pair family.
        config: Base configuration applied to both chains (or to paired TCREmp).
        config_chain1: Optional per-chain override for chain 1.
        config_chain2: Optional per-chain override for chain 2.
        cluster_separator: Separator between chain 1 and chain 2 cluster IDs.
        include_unassigned: Include pairs where one chain has no cluster.
        extra_chain1: Extra data forwarded to chain 1 clusterer (e.g. GLIPH data).
        extra_chain2: Extra data forwarded to chain 2 clusterer.

    Returns:
        Paired :class:`~mir.common.metaclonotype.MetaClonotypeClustering`.
    """
    from mir.common.single_cell import LOCUS_PAIR_TO_LOCI
    from mir.utils.metaclonotype_clustering import paired_metaclonotypes_from_single_chain

    pairs = paired_locus_rep.paired_clonotypes

    # Native paired TCREmp — use built-in paired embedding directly.
    if config.method == "tcremp" and config_chain1 is None and config_chain2 is None:
        return _cluster_tcremp_paired(paired_locus_rep, config)

    # Per-chain configs
    cfg1 = config_chain1 or config
    cfg2 = config_chain2 or config

    # Extract locus names from the locus_pair
    locus_pair = paired_locus_rep.locus_pair
    loci = LOCUS_PAIR_TO_LOCI.get(locus_pair)
    if loci is None:
        raise ValueError(f"Unknown locus_pair {locus_pair!r}")
    locus1, locus2 = loci

    # Build per-chain LocusRepertoires from the paired clonotypes.
    from mir.common.repertoire import LocusRepertoire

    chain1_clones = [p.clonotype1 for p in pairs]
    chain2_clones = [p.clonotype2 for p in pairs]
    rep1 = LocusRepertoire(clonotypes=chain1_clones, locus=locus1)
    rep2 = LocusRepertoire(clonotypes=chain2_clones, locus=locus2)

    # Override locus in chain configs if not explicitly set by the caller.
    if config_chain1 is None and cfg1.locus == config.locus:
        from dataclasses import replace
        cfg1 = replace(cfg1, locus=locus1)
    if config_chain2 is None and cfg2.locus == config.locus:
        from dataclasses import replace
        cfg2 = replace(cfg2, locus=locus2)

    meta1 = cluster_metaclonotypes(rep1, cfg1, extra=extra_chain1)
    meta2 = cluster_metaclonotypes(rep2, cfg2, extra=extra_chain2)

    return paired_metaclonotypes_from_single_chain(
        pairs,
        meta1,
        meta2,
        cluster_separator=cluster_separator,
        include_unassigned=include_unassigned,
    )
