"""Common clustering-to-metaclonotype conversion helpers.

This module centralizes conversion of clustering outputs (labels, graph
components, Leiden/Louvain communities, search neighborhoods) into
``MetaClonotypeDefinition`` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from mir.common.metaclonotype import MetaClonotypeDefinition


def metaclonotypes_from_cluster_labels(
    clonotype_ids: list[str],
    labels: list[int | str] | np.ndarray,
    *,
    include_noise: bool = False,
    noise_labels: set[int | str] | None = None,
    representatives: set[str] | None = None,
) -> MetaClonotypeDefinition:
    """Convert arbitrary cluster labels to a metaclonotype definition."""
    from mir.common.metaclonotype import metaclonotypes_from_labels

    if isinstance(labels, np.ndarray):
        label_list = labels.tolist()
    else:
        label_list = list(labels)
    return metaclonotypes_from_labels(
        clonotype_ids,
        label_list,
        include_noise=include_noise,
        noise_labels=noise_labels,
        representatives=representatives,
    )


def _membership_for_method(
    graph,
    *,
    method: str,
    weights: str | None,
    leiden_objective_function: str,
    leiden_n_iterations: int,
) -> list[int]:
    if method == "components":
        membership = np.full(graph.vcount(), -1, dtype=int)
        for component_id, vertices in enumerate(graph.components()):
            for vertex in vertices:
                membership[vertex] = component_id
        return membership.tolist()
    if method == "leiden":
        return list(
            graph.community_leiden(
                weights=weights,
                objective_function=leiden_objective_function,
                n_iterations=leiden_n_iterations,
            ).membership
        )
    if method == "louvain":
        return list(graph.community_multilevel(weights=weights).membership)
    raise ValueError("method must be one of: components, leiden, louvain")


def metaclonotypes_from_graph_communities(
    graph,
    *,
    vertex_id_attr: str = "r_id",
    method: str = "components",
    min_cluster_size: int = 1,
    weights: str | None = "weight",
    leiden_objective_function: str = "modularity",
    leiden_n_iterations: int = 5,
    include_unclustered: bool = False,
    unclustered_label: int | str = -1,
) -> MetaClonotypeDefinition:
    """Convert graph components/communities to metaclonotypes.

    Args:
        graph: igraph graph with clonotype ids in ``vertex_id_attr``.
        vertex_id_attr: Vertex attribute that stores clonotype ids.
        method: One of ``components``, ``leiden``, or ``louvain``.
        min_cluster_size: Drop clusters below this size (set as noise).
        weights: Edge weight attribute for Leiden/Louvain.
        include_unclustered: Keep dropped/noise nodes as singleton/noise rows.
        unclustered_label: Label assigned to dropped/noise nodes.
    """
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be >= 1")

    labels = _membership_for_method(
        graph,
        method=method,
        weights=weights,
        leiden_objective_function=leiden_objective_function,
        leiden_n_iterations=leiden_n_iterations,
    )
    labels_arr = np.asarray(labels, dtype=object)
    if min_cluster_size > 1:
        for label in np.unique(labels_arr):
            mask = labels_arr == label
            if int(mask.sum()) < min_cluster_size:
                labels_arr[mask] = unclustered_label

    ids = [str(v[vertex_id_attr]) for v in graph.vs]
    return metaclonotypes_from_cluster_labels(
        ids,
        labels_arr.tolist(),
        include_noise=include_unclustered,
        noise_labels={unclustered_label},
    )


def metaclonotypes_from_search_scope(
    representative_ids: list[str],
    *,
    neighbor_selector: Callable[[str], Iterable[str]],
    cluster_prefix: str = "mc",
) -> MetaClonotypeDefinition:
    """Build metaclonotypes from representative-centered search neighborhoods.

    The ``neighbor_selector`` callback can be backed by tcrtrie search scopes,
    edit-distance scopes, or continuous-radius score thresholds.
    """
    import polars as pl
    from mir.common.metaclonotype import MetaClonotypeDefinition

    rows: list[dict[str, object]] = []
    for i, rep_id in enumerate(representative_ids):
        cluster_id = f"{cluster_prefix}_{i}"
        neigh = [str(x) for x in neighbor_selector(str(rep_id))]
        if str(rep_id) not in neigh:
            neigh = [str(rep_id)] + neigh
        # Preserve order but deduplicate.
        seen: set[str] = set()
        uniq = [x for x in neigh if not (x in seen or seen.add(x))]
        for member_id in uniq:
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "clonotype_id": member_id,
                    "is_representative": member_id == str(rep_id),
                }
            )

    return MetaClonotypeDefinition(pl.DataFrame(rows), paired=False)


def paired_metaclonotypes_from_pair_labels(
    pair_ids: list[str],
    clonotype_id_1: list[str],
    clonotype_id_2: list[str],
    labels: list[int | str] | np.ndarray,
    *,
    mock_chain_1_by_pair: dict[str, bool] | None = None,
    mock_chain_2_by_pair: dict[str, bool] | None = None,
    include_noise: bool = False,
    noise_labels: set[int | str] | None = None,
) -> MetaClonotypeDefinition:
    """Convert paired-clonotype labels (e.g. paired TCREmp DBSCAN) to metaclonotypes."""
    if isinstance(labels, np.ndarray):
        label_list = labels.tolist()
    else:
        label_list = list(labels)

    if not (len(pair_ids) == len(clonotype_id_1) == len(clonotype_id_2) == len(label_list)):
        raise ValueError("All paired label inputs must have equal length")

    noise = noise_labels if noise_labels is not None else {-1}
    mock1 = mock_chain_1_by_pair or {}
    mock2 = mock_chain_2_by_pair or {}
    rows: list[dict[str, object]] = []
    for pid, c1, c2, label in zip(pair_ids, clonotype_id_1, clonotype_id_2, label_list, strict=True):
        if (label in noise) and not include_noise:
            continue
        rows.append(
            {
                "cluster_id": str(label),
                "clonotype_id_1": str(c1),
                "clonotype_id_2": str(c2),
                "is_representative": False,
                "mock_chain_1": bool(mock1.get(str(pid), False)),
                "mock_chain_2": bool(mock2.get(str(pid), False)),
            }
        )

    # Pick first row per cluster as representative.
    first_seen: dict[str, int] = {}
    for idx, row in enumerate(rows):
        cid = str(row["cluster_id"])
        if cid not in first_seen:
            first_seen[cid] = idx
    for idx in first_seen.values():
        rows[idx]["is_representative"] = True

    import polars as pl
    from mir.common.metaclonotype import MetaClonotypeDefinition

    return MetaClonotypeDefinition(pl.DataFrame(rows), paired=True)
