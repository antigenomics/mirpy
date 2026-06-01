"""Metaclonotype definitions and core analytics.

A *metaclonotype* is a lightweight cluster layer over an existing
:class:`~mir.common.repertoire.LocusRepertoire`.  It is stored as a membership
table (a Polars DataFrame) that maps cluster identifiers to sequence identifiers,
without duplicating or rebuilding the underlying clonotype objects.  This design
allows cluster-level count aggregation, functional-diversity computation, and
cross-repertoire overlap without the memory cost of re-instantiating repertoires.

Metaclonotypes can represent the output of any clustering backend:

* DBSCAN / Leiden / Louvain community detection on an edit-distance graph
* ALICE / TCRNET enriched-cluster sets
* TCRdist-style radius clusters around representative sequences
* Pre-computed connected components from any graph

Membership table schemas
------------------------
Single-chain (``paired=False``):

- ``cluster_id``     (str)
- ``clonotype_id``   (str)
- ``is_representative`` (bool, defaults to False)

Paired-chain (``paired=True``):

- ``cluster_id``     (str)
- ``clonotype_id_1`` (str)
- ``clonotype_id_2`` (str)
- ``is_representative`` (bool, defaults to False)
- ``mock_chain_1``   (bool, defaults to False)
- ``mock_chain_2``   (bool, defaults to False)

Functional-diversity methods delegate to :mod:`mir.common.diversity`,
implementing Hill (1973) profiles and iNEXT-style rarefaction (Hsieh *et al.*
2016) at the metaclonotype count level.

References
----------
Pogorelyy MV, Minervina AA, Shugay M, Chudakov DM, Lebedev YB, Mora T,
Walczak AM. Detecting T cell receptors involved in immune responses from
single repertoire snapshots. *PLoS Biol.* 2019;17(6):e3000314.
PMID:31194732. https://pubmed.ncbi.nlm.nih.gov/31194732/

Hsieh TC, Ma KH, Chao A. iNEXT: an R package for rarefaction and
extrapolation of species diversity (Hill numbers). *Methods Ecol Evol.*
2016;7(12):1451-1456. doi:10.1111/2041-210X.12613.

Hill MO. Diversity and evenness: a unifying notation and its consequences.
*Ecology.* 1973;54(2):427-432. doi:10.2307/1934352.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import log
from typing import Callable, Iterable

_logger = logging.getLogger(__name__)

import polars as pl

from mir.common.alleles import allele_to_major, strip_allele
from mir.common.clonotype import Clonotype
from mir.common.diversity import (
    CountField,
    DiversitySummary,
    hill_curve,
    rarefaction_curve,
    summarize_counts,
)
from mir.common.repertoire import LocusRepertoire
from mir.common.single_cell import PairedRepertoire


_SINGLE_REQUIRED = {"cluster_id", "clonotype_id"}
_PAIRED_REQUIRED = {"cluster_id", "clonotype_id_1", "clonotype_id_2"}


def _ensure_utf8(df: pl.DataFrame, cols: Iterable[str]) -> pl.DataFrame:
    out = df
    for col in cols:
        if col in out.columns:
            out = out.with_columns(pl.col(col).cast(pl.Utf8))
    return out


def _ensure_bool_col(df: pl.DataFrame, col: str, default: bool = False) -> pl.DataFrame:
    if col in df.columns:
        return df.with_columns(pl.col(col).cast(pl.Boolean).fill_null(default))
    return df.with_columns(pl.lit(default).alias(col))


@dataclass(frozen=True)
class MetaClonotypeClustering:
    """Cluster membership table for single-chain or paired clonotypes.

    Args:
        table: Membership table. Required columns depend on ``paired``.
        paired: If True, validates paired schema; otherwise single-chain schema.
    """

    table: pl.DataFrame
    paired: bool = False

    def __post_init__(self) -> None:
        required = _PAIRED_REQUIRED if self.paired else _SINGLE_REQUIRED
        missing = required - set(self.table.columns)
        if missing:
            raise ValueError(
                f"Missing required metaclonotype columns: {sorted(missing)}"
            )

        normalized = _ensure_utf8(self.table, required)
        normalized = _ensure_bool_col(normalized, "is_representative", default=False)
        if self.paired:
            normalized = _ensure_bool_col(normalized, "mock_chain_1", default=False)
            normalized = _ensure_bool_col(normalized, "mock_chain_2", default=False)

        subset = ["cluster_id", "clonotype_id_1", "clonotype_id_2"] if self.paired else ["cluster_id", "clonotype_id"]
        normalized = normalized.unique(subset=subset, keep="first").sort(subset)
        object.__setattr__(self, "table", normalized)

    @property
    def n_clusters(self) -> int:
        """Return number of unique clusters."""
        return int(self.table["cluster_id"].n_unique())

    @property
    def cluster_ids(self) -> list[str]:
        """Return sorted unique cluster identifiers."""
        return sorted(self.table["cluster_id"].unique().to_list())

    def members_of(self, cluster_id: str) -> pl.DataFrame:
        """Return membership rows for one cluster."""
        return self.table.filter(pl.col("cluster_id") == cluster_id)

    def representatives(self) -> pl.DataFrame:
        """Return representative membership rows."""
        return self.table.filter(pl.col("is_representative"))


def metaclonotypes_from_labels(
    clonotype_ids: list[str],
    labels: list[int | str],
    *,
    include_noise: bool = False,
    noise_labels: set[int | str] | None = None,
    representatives: set[str] | None = None,
) -> MetaClonotypeClustering:
    """Build single-chain metaclonotypes from per-clonotype labels.

    Args:
        clonotype_ids: Sequence IDs in the same order as labels.
        labels: Cluster labels (e.g. connected components, DBSCAN labels).
        include_noise: Keep labels in ``noise_labels`` if True.
        noise_labels: Label values treated as noise (default ``{-1}``).
        representatives: Optional set of representative clonotype IDs.
    """
    if len(clonotype_ids) != len(labels):
        raise ValueError("clonotype_ids and labels must have equal length")

    reps = representatives or set()
    noise = noise_labels if noise_labels is not None else {-1}
    rows: list[dict] = []
    for cid, label in zip(clonotype_ids, labels, strict=True):
        if (label in noise) and not include_noise:
            continue
        rows.append(
            {
                "cluster_id": str(label),
                "clonotype_id": str(cid),
                "is_representative": str(cid) in reps,
            }
        )
    if not rows:
        return MetaClonotypeClustering(
            pl.DataFrame(
                {
                    "cluster_id": pl.Series([], dtype=pl.Utf8),
                    "clonotype_id": pl.Series([], dtype=pl.Utf8),
                    "is_representative": pl.Series([], dtype=pl.Boolean),
                }
            ),
            paired=False,
        )
    return MetaClonotypeClustering(pl.DataFrame(rows), paired=False)


def metaclonotypes_from_components(
    components: list[list[str]],
    *,
    cluster_prefix: str = "mc",
) -> MetaClonotypeClustering:
    """Build single-chain metaclonotypes from connected components.

    The first member of each component is marked as representative.
    """
    rows: list[dict] = []
    for idx, members in enumerate(components):
        cluster_id = f"{cluster_prefix}_{idx}"
        for j, member in enumerate(members):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "clonotype_id": str(member),
                    "is_representative": j == 0,
                }
            )
    return MetaClonotypeClustering(pl.DataFrame(rows), paired=False)


def metaclonotypes_from_igraph(
    graph,
    *,
    membership: list[int] | None = None,
    vertex_id_attr: str = "r_id",
    cluster_prefix: str = "mc",
) -> MetaClonotypeClustering:
    """Build single-chain metaclonotypes from an igraph graph.

    If ``membership`` is omitted, connected components are used.
    """
    if membership is None:
        comps = graph.components()
        members = [[str(graph.vs[i][vertex_id_attr]) for i in comp] for comp in comps]
        return metaclonotypes_from_components(members, cluster_prefix=cluster_prefix)

    if len(membership) != graph.vcount():
        raise ValueError("membership size must equal graph.vcount()")
    ids = [str(v[vertex_id_attr]) for v in graph.vs]
    return metaclonotypes_from_labels(
        ids,
        membership,
        include_noise=False,
        representatives=None,
    )


def metaclonotypes_from_seed_neighbors(
    repertoire: LocusRepertoire,
    *,
    seed_clonotype_ids: list[str],
    metric: str = "hamming",
    threshold: int = 1,
    match_v_gene: bool = True,
    match_j_gene: bool = True,
    cluster_prefix: str = "mc",
) -> MetaClonotypeClustering:
    """Build one metaclonotype per seed clonotype using edit-distance neighbors."""
    if metric not in {"hamming", "levenshtein"}:
        raise ValueError("metric must be 'hamming' or 'levenshtein'")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    from mir.graph.distance_utils import is_within_threshold

    by_id = {c.sequence_id: c for c in repertoire.clonotypes}
    rows: list[dict] = []
    for idx, seed_id in enumerate(seed_clonotype_ids):
        seed = by_id.get(seed_id)
        if seed is None:
            _logger.warning("seed_id %r not found in repertoire; skipping", seed_id)
            continue
        cluster_id = f"{cluster_prefix}_{idx}"
        seed_v = strip_allele(seed.v_gene)
        seed_j = strip_allele(seed.j_gene)
        for candidate in repertoire.clonotypes:
            if match_v_gene and seed_v != strip_allele(candidate.v_gene):
                continue
            if match_j_gene and seed_j != strip_allele(candidate.j_gene):
                continue
            if is_within_threshold(seed.junction_aa, candidate.junction_aa, metric, threshold):
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "clonotype_id": candidate.sequence_id,
                        "is_representative": candidate.sequence_id == seed_id,
                    }
                )
    if not rows:
        return MetaClonotypeClustering(
            pl.DataFrame(
                {
                    "cluster_id": pl.Series([], dtype=pl.Utf8),
                    "clonotype_id": pl.Series([], dtype=pl.Utf8),
                    "is_representative": pl.Series([], dtype=pl.Boolean),
                }
            ),
            paired=False,
        )
    return MetaClonotypeClustering(pl.DataFrame(rows), paired=False)


def metaclonotypes_from_search_scope(
    representative_ids: list[str],
    *,
    neighbor_selector: Callable[[str], Iterable[str]],
    cluster_prefix: str = "scope_mc",
) -> MetaClonotypeClustering:
    """Build representative-centered metaclonotypes from a custom search scope.

    This is suitable for tcrtrie-backed scope search (substitutions/indels/
    max edits) and any custom neighborhood API that returns member IDs for a
    representative query.
    """
    from mir.utils.metaclonotype_clustering import metaclonotypes_from_search_scope as _build

    return _build(
        representative_ids,
        neighbor_selector=neighbor_selector,
        cluster_prefix=cluster_prefix,
    )


def metaclonotypes_from_radius_threshold(
    repertoire: LocusRepertoire,
    *,
    representative_ids: list[str],
    score_distance_fn: Callable[[Clonotype, Clonotype], float],
    max_distance: float,
    match_v_gene: bool = False,
    match_j_gene: bool = False,
    cluster_prefix: str = "radius_mc",
) -> MetaClonotypeClustering:
    """Build metaclonotypes via a continuous-radius score threshold.

    This helper targets TCRdist-like workflows where clonotypes join a
    representative-centered cluster when distance/score <= ``max_distance``.
    """
    if max_distance < 0:
        raise ValueError("max_distance must be >= 0")

    by_id = {c.sequence_id: c for c in repertoire.clonotypes}
    rows: list[dict[str, object]] = []
    for idx, rep_id in enumerate(representative_ids):
        rep = by_id.get(rep_id)
        if rep is None:
            continue
        cluster_id = f"{cluster_prefix}_{idx}"
        rep_v = strip_allele(rep.v_gene)
        rep_j = strip_allele(rep.j_gene)
        for candidate in repertoire.clonotypes:
            if match_v_gene and rep_v != strip_allele(candidate.v_gene):
                continue
            if match_j_gene and rep_j != strip_allele(candidate.j_gene):
                continue
            distance = float(score_distance_fn(rep, candidate))
            if distance <= max_distance:
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "clonotype_id": candidate.sequence_id,
                        "is_representative": candidate.sequence_id == rep_id,
                    }
                )
    return MetaClonotypeClustering(pl.DataFrame(rows), paired=False)


def summarize_metaclonotypes(
    repertoire: LocusRepertoire,
    metaclonotypes: MetaClonotypeClustering,
) -> pl.DataFrame:
    """Aggregate duplicate/UMI counts per metaclonotype for single-chain data."""
    if metaclonotypes.paired:
        raise ValueError("summarize_metaclonotypes expects a single-chain definition")

    rep_df = repertoire.to_polars().select(
        pl.col("sequence_id").cast(pl.Utf8),
        pl.col("duplicate_count").cast(pl.Int64).fill_null(0),
        pl.col("umi_count").cast(pl.Int64).fill_null(0),
        pl.col("junction_aa").cast(pl.Utf8),
        pl.col("v_gene").cast(pl.Utf8),
        pl.col("j_gene").cast(pl.Utf8),
    )

    joined = metaclonotypes.table.join(
        rep_df,
        left_on="clonotype_id",
        right_on="sequence_id",
        how="left",
    )

    reps = (
        joined
        .filter(pl.col("is_representative"))
        .group_by("cluster_id")
        .agg(
            pl.col("clonotype_id").first().alias("representative_clonotype_id"),
            pl.col("junction_aa").first().alias("representative_junction_aa"),
            pl.col("v_gene").first().alias("representative_v_gene"),
            pl.col("j_gene").first().alias("representative_j_gene"),
        )
    )

    out = (
        joined
        .group_by("cluster_id")
        .agg(
            pl.len().alias("n_members"),
            pl.col("duplicate_count").sum().alias("duplicate_count"),
            pl.col("umi_count").sum().alias("umi_count"),
        )
        .join(reps, on="cluster_id", how="left")
        .with_columns(
            pl.col("duplicate_count").cast(pl.Int64).fill_null(0),
            pl.col("umi_count").cast(pl.Int64).fill_null(0),
        )
        .sort("cluster_id")
    )
    return out


def metaclonotype_count_vector(
    repertoire: LocusRepertoire,
    metaclonotypes: MetaClonotypeClustering,
    *,
    count_field: CountField = "duplicate_count",
) -> list[int]:
    """Return per-metaclonotype count vector using duplicate or UMI counts."""
    if count_field not in {"duplicate_count", "umi_count"}:
        raise ValueError("count_field must be 'duplicate_count' or 'umi_count'")
    summary = summarize_metaclonotypes(repertoire, metaclonotypes)
    return [int(x) for x in summary[count_field].to_list()]


def functional_diversity(
    repertoire: LocusRepertoire,
    metaclonotypes: MetaClonotypeClustering,
    *,
    count_field: CountField = "duplicate_count",
    expanded_threshold: float = 1e-3,
    hyperexpanded_threshold: float = 1e-2,
) -> DiversitySummary:
    """Compute diversity summary over metaclonotype-level aggregated counts."""
    counts = metaclonotype_count_vector(
        repertoire,
        metaclonotypes,
        count_field=count_field,
    )
    return summarize_counts(
        counts,
        expanded_threshold=expanded_threshold,
        hyperexpanded_threshold=hyperexpanded_threshold,
    )


def functional_hill_curve(
    repertoire: LocusRepertoire,
    metaclonotypes: MetaClonotypeClustering,
    *,
    count_field: CountField = "duplicate_count",
    q_values: list[float] | None = None,
) -> pl.DataFrame:
    """Compute Hill diversity profile over metaclonotype-level counts."""
    counts = metaclonotype_count_vector(
        repertoire,
        metaclonotypes,
        count_field=count_field,
    )
    return hill_curve(counts, q_values=q_values)


def functional_rarefaction_curve(
    repertoire: LocusRepertoire,
    metaclonotypes: MetaClonotypeClustering,
    *,
    count_field: CountField = "duplicate_count",
    m_steps: list[int] | None = None,
    include_exact: bool = True,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Compute rarefaction over metaclonotype-level aggregated counts."""
    counts = metaclonotype_count_vector(
        repertoire,
        metaclonotypes,
        count_field=count_field,
    )
    return rarefaction_curve(
        counts,
        m_steps=m_steps,
        include_exact=include_exact,
        confidence=confidence,
    )


def metaclonotype_junctions(
    repertoire: LocusRepertoire,
    metaclonotypes: MetaClonotypeClustering,
    *,
    cluster_id: str,
    representatives_only: bool = False,
) -> list[str]:
    """Return junction_aa sequences for one metaclonotype cluster."""
    if metaclonotypes.paired:
        raise ValueError("metaclonotype_junctions expects a single-chain definition")

    members = metaclonotypes.members_of(cluster_id)
    if representatives_only:
        members = members.filter(pl.col("is_representative"))
    ids = set(members["clonotype_id"].to_list())
    return [c.junction_aa for c in repertoire.clonotypes if c.sequence_id in ids and c.junction_aa]


def default_clonotype_identity(clonotype: Clonotype) -> tuple[str, str, str]:
    """Return default clonotype identity key used in functional overlap.

    Identity key: (junction_aa, major-v-gene, major-j-gene)
    """
    return (
        clonotype.junction_aa,
        allele_to_major(clonotype.v_gene or ""),
        allele_to_major(clonotype.j_gene or ""),
    )


def _cluster_identity_sets(
    repertoire: LocusRepertoire,
    metaclonotypes: MetaClonotypeClustering,
    identity_fn: Callable[[Clonotype], object],
) -> dict[str, set[object]]:
    by_id = {c.sequence_id: c for c in repertoire.clonotypes}
    out: dict[str, set[object]] = {}
    for row in metaclonotypes.table.iter_rows(named=True):
        cid = row["cluster_id"]
        seq_id = row["clonotype_id"]
        clone = by_id.get(seq_id)
        if clone is None:
            continue
        out.setdefault(cid, set()).add(identity_fn(clone))
    return out


def functional_overlap_1(
    repertoire_a: LocusRepertoire,
    metaclonotypes_a: MetaClonotypeClustering,
    repertoire_b: LocusRepertoire,
    metaclonotypes_b: MetaClonotypeClustering,
    *,
    identity_fn: Callable[[Clonotype], object] = default_clonotype_identity,
) -> pl.DataFrame:
    """Compute functional overlap-1 between two independently clustered repertoires.

    Two clusters are considered matching if they share at least one clonotype
    identity according to ``identity_fn``.
    """
    if metaclonotypes_a.paired or metaclonotypes_b.paired:
        raise ValueError("functional_overlap_1 currently supports single-chain metaclonotypes")

    ida = _cluster_identity_sets(repertoire_a, metaclonotypes_a, identity_fn)
    idb = _cluster_identity_sets(repertoire_b, metaclonotypes_b, identity_fn)

    # Build a flat set of all identities present in B to allow O(1) lookup.
    all_b_identities: set[object] = set()
    for bset in idb.values():
        all_b_identities.update(bset)

    all_a_identities: set[object] = set()
    for aset in ida.values():
        all_a_identities.update(aset)

    shared_a = {a for a, aset in ida.items() if aset & all_b_identities}
    shared_b = {b for b, bset in idb.items() if bset & all_a_identities}

    return pl.DataFrame(
        {
            "a_clusters": [len(ida)],
            "b_clusters": [len(idb)],
            "a_shared_clusters": [len(shared_a)],
            "b_shared_clusters": [len(shared_b)],
            "a_overlap_fraction": [float(len(shared_a) / len(ida)) if ida else 0.0],
            "b_overlap_fraction": [float(len(shared_b) / len(idb)) if idb else 0.0],
        }
    )


def _shannon_entropy_from_counts(counts: list[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = float(c) / total
        h -= p * log(p)
    return h


def pooled_entropy_difference(
    pooled_counts: list[int],
    counts_a: list[int],
    counts_b: list[int],
) -> pl.DataFrame:
    """Compute entropy decomposition summary for functional overlap analyses.

    Returns ``H_pooled - H_a - H_b`` where each entropy is Shannon entropy over
    metaclonotype abundances.
    """
    h_pool = _shannon_entropy_from_counts(pooled_counts)
    h_a = _shannon_entropy_from_counts(counts_a)
    h_b = _shannon_entropy_from_counts(counts_b)
    return pl.DataFrame(
        {
            "h_pooled": [h_pool],
            "h_a": [h_a],
            "h_b": [h_b],
            "h_pooled_minus_h_a_minus_h_b": [h_pool - h_a - h_b],
        }
    )


def summarize_paired_metaclonotypes(
    paired_repertoire: PairedRepertoire,
    metaclonotypes: MetaClonotypeClustering,
    *,
    count_field: CountField = "duplicate_count",
) -> pl.DataFrame:
    """Aggregate paired metaclonotype counts from a PairedRepertoire."""
    if not metaclonotypes.paired:
        raise ValueError("summarize_paired_metaclonotypes expects paired metaclonotypes")
    if count_field not in {"duplicate_count", "umi_count"}:
        raise ValueError("count_field must be 'duplicate_count' or 'umi_count'")

    flat_lookup: dict[str, Clonotype] = {}
    for pair_rep in paired_repertoire.paired_locus_repertoires.values():
        for pair in pair_rep.paired_clonotypes:
            flat_lookup[pair.clonotype1.sequence_id] = pair.clonotype1
            flat_lookup[pair.clonotype2.sequence_id] = pair.clonotype2

    def _find_clone(seq_id: str) -> Clonotype | None:
        return flat_lookup.get(seq_id)

    rows: list[dict] = []
    for row in metaclonotypes.table.iter_rows(named=True):
        c1 = _find_clone(str(row["clonotype_id_1"]))
        c2 = _find_clone(str(row["clonotype_id_2"]))
        v1 = int(getattr(c1, count_field, 0) or 0)
        v2 = int(getattr(c2, count_field, 0) or 0)
        rows.append(
            {
                "cluster_id": row["cluster_id"],
                "pair_count": v1 + v2,
                "is_representative": bool(row.get("is_representative", False)),
                "mock_chain_1": bool(row.get("mock_chain_1", False)),
                "mock_chain_2": bool(row.get("mock_chain_2", False)),
            }
        )
    return (
        pl.DataFrame(rows)
        .group_by("cluster_id")
        .agg(
            pl.len().alias("n_members"),
            pl.col("pair_count").sum().alias(count_field),
            pl.col("mock_chain_1").sum().alias("n_mock_chain_1"),
            pl.col("mock_chain_2").sum().alias("n_mock_chain_2"),
        )
        .sort("cluster_id")
    )
