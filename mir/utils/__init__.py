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
    cluster_purity_consistency,
    select_eps_kneedle,
)
from .stats import bh_fdr

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
    "cluster_purity_consistency",
    "select_eps_kneedle",
    "bh_fdr",
]