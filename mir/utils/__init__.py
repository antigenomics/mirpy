"""Utility helpers shared across notebooks and scripts."""

from .notebook_assets import (
    ensure_airr_benchmark,
    ensure_airr_covid19,
    ensure_airr_yfv19,
    find_airr_benchmark_dcode_10x_vdj_v1_donor,
    find_airr_benchmark_dcode_10x_vdj_v1_donor_matrix,
    find_airr_benchmark_sra_meta,
    find_airr_benchmark_tcrnet_file,
    find_airr_benchmark_vdjdb_slim,
    find_repo_root,
    notebook_assets_root,
    notebook_large_assets_root,
)
from .embedding_diagnostics import (
    analyze_embedding_dbscan,
    classification_scores_by_label,
    cluster_purity_consistency,
    majority_vote_cluster_predictions,
    select_eps_kneedle_stable,
)
from .stats import bh_fdr
from .memory_debug import process_memory_snapshot, top_python_processes

__all__ = [
    "ensure_airr_benchmark",
    "ensure_airr_covid19",
    "ensure_airr_yfv19",
    "find_airr_benchmark_dcode_10x_vdj_v1_donor",
    "find_airr_benchmark_dcode_10x_vdj_v1_donor_matrix",
    "find_airr_benchmark_sra_meta",
    "find_airr_benchmark_tcrnet_file",
    "find_airr_benchmark_vdjdb_slim",
    "find_repo_root",
    "notebook_assets_root",
    "notebook_large_assets_root",
    "analyze_embedding_dbscan",
    "classification_scores_by_label",
    "cluster_purity_consistency",
    "majority_vote_cluster_predictions",
    "select_eps_kneedle_stable",
    "bh_fdr",
    "process_memory_snapshot",
    "top_python_processes",
]