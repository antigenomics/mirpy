"""Benchmark and parameter search for GLIPH-style clustering on AIRR benchmark.

Runs only with ``RUN_BENCHMARK=1``. The test bootstraps the GLIPH dataset from
``airr_benchmark/gliph/gliph_trb.tsv.gz`` and evaluates multiple token families
and cluster methods against the provided ``gliph_cluster_id`` labels.
"""

from __future__ import annotations

from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_mutual_info_score

from mir.biomarkers.gliph import (
    deduplicate_clonotype_rows,
    extract_g4mer_artifacts,
    extract_u4mer_artifacts,
    extract_v3mer_artifacts,
    extract_vpos3mer_artifacts,
    normalize_control_vj,
)
from mir.biomarkers.kmer_stats import compare_kmer_counts
from mir.common.control import ControlManager
from tests.conftest import skip_benchmarks


GLIPH_PATH = Path(__file__).resolve().parents[1] / "airr_benchmark" / "gliph" / "gliph_trb.tsv.gz"
AA_RE = r"^[ACDEFGHIKLMNPQRSTVWY]+$"
THREADS = 4
CONTROL_SAMPLE = 100_000
FAMILIES = ("v3", "vpos3", "u4", "g4")
METHODS = ("components", "leiden")
MIN_CLONOTYPE_SUPPORT = 2
SIG_FDR = 0.05
SIG_ODDS = 2.0


def _normalize_gliph_df(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "junction_aa": raw["junction_aa"].astype(str).str.strip(),
            "v_gene": raw["v_gene"].astype(str).str.strip(),
            "j_gene": raw["j_gene"].astype(str).str.strip(),
            "duplicate_count": pd.to_numeric(raw["duplicate_count"], errors="coerce").fillna(1).astype(int),
            "reference_id": raw["reference_id"].astype(str).str.strip(),
            "stimulus": raw["stimulus"].astype(str).str.strip(),
            "epitope": raw["epitope"].astype(str).str.strip(),
            "gliph_cluster_id": raw["gliph_cluster_id"].astype(str).str.strip(),
        }
    )
    out = out[out["junction_aa"].str.len() >= 5].copy()
    out = out[out["junction_aa"].str.match(AA_RE)].copy()
    out = deduplicate_clonotype_rows(out)
    return out


def _extract_family(df: pd.DataFrame, family: str):
    kwargs = dict(threads=THREADS, count_mode="clonotype", unique_clonotypes=False)
    if family == "v3":
        return extract_v3mer_artifacts(df, **kwargs)
    if family == "vpos3":
        return extract_vpos3mer_artifacts(df, **kwargs)
    if family == "u4":
        return extract_u4mer_artifacts(df, **kwargs)
    if family == "g4":
        return extract_g4mer_artifacts(df, **kwargs)
    raise ValueError(f"Unknown family {family}")


def _significant_tokens(sample_art, ctrl_art) -> pd.DataFrame:
    df = compare_kmer_counts(
        sample_art.counts,
        ctrl_art.counts,
        test="fisher",
        p_adj_method="fdr_bh",
        pseudocount=1,
    )
    support = pd.Series(sample_art.clonotype_counts, name="sample_clonotypes")
    df = df.join(support, how="left").fillna({"sample_clonotypes": 0})
    df["sample_clonotypes"] = df["sample_clonotypes"].astype(int)
    return df


def _cluster_labels(study_df: pd.DataFrame, sample_art, enriched_tokens: set[str], method: str) -> dict[str, int]:
    clone_ids = sorted(study_df["row_id"].astype(str).tolist())
    labels = {cid: -1 for cid in clone_ids}
    token_to_clones = {
        tok: sorted(sample_art.token_to_clone.get(tok, set()))
        for tok in enriched_tokens
        if sample_art.token_to_clone.get(tok)
    }
    if not token_to_clones:
        return labels

    token_nodes = sorted(token_to_clones)
    active_clone_ids = sorted({cid for ids in token_to_clones.values() for cid in ids})
    token_idx = {tok: i for i, tok in enumerate(token_nodes)}
    clone_idx = {cid: len(token_nodes) + i for i, cid in enumerate(active_clone_ids)}
    graph = ig.Graph(n=len(token_nodes) + len(active_clone_ids), directed=False)
    graph.add_edges(
        (token_idx[tok], clone_idx[cid])
        for tok, ids in token_to_clones.items()
        for cid in ids
    )

    if method == "components":
        membership = np.full(graph.vcount(), -1, dtype=int)
        for comp_id, verts in enumerate(graph.components()):
            for vertex in verts:
                membership[vertex] = comp_id
    elif method == "leiden":
        membership = np.array(
            graph.community_leiden(objective_function="modularity", n_iterations=5).membership,
            dtype=int,
        )
    else:
        raise ValueError(method)

    clone_membership = np.array([membership[clone_idx[cid]] for cid in active_clone_ids], dtype=int)
    for cluster_id in np.unique(clone_membership):
        members = clone_membership == cluster_id
        if int(members.sum()) < 3:
            clone_membership[members] = -1

    for cid, cluster_id in zip(active_clone_ids, clone_membership):
        labels[cid] = int(cluster_id)
    return labels


@skip_benchmarks
@pytest.mark.slow_benchmark
def test_gliph_parameter_search_benchmark() -> None:
    if not GLIPH_PATH.exists():
        pytest.skip(f"Missing GLIPH dataset: {GLIPH_PATH}")

    raw = pd.read_csv(GLIPH_PATH, sep="\t")
    df = _normalize_gliph_df(raw)

    ctrl_raw = ControlManager().ensure_and_load_control_df("real", "human", "TRB")
    ctrl_all = pd.DataFrame(
        {
            "junction_aa": ctrl_raw["junction_aa"].astype(str).str.strip(),
            "v_gene": ctrl_raw["v_gene"].astype(str).str.strip(),
            "j_gene": ctrl_raw["j_gene"].astype(str).str.strip(),
            "duplicate_count": pd.to_numeric(ctrl_raw.get("duplicate_count", 1), errors="coerce").fillna(1).astype(int),
            "reference_id": "control",
            "stimulus": "control",
            "epitope": "",
            "gliph_cluster_id": "",
        }
    )
    ctrl_all = ctrl_all[ctrl_all["junction_aa"].str.match(AA_RE) & (ctrl_all["junction_aa"].str.len() >= 5)].copy()

    results: list[dict[str, object]] = []
    baseline_sig: dict[str, int] = {}

    for study, sdf in df.groupby("reference_id", sort=True):
        ctrl_df = normalize_control_vj(sdf, ctrl_all, n=CONTROL_SAMPLE, seed=42)
        for family in FAMILIES:
            sample_art = _extract_family(sdf, family)
            ctrl_art = _extract_family(ctrl_df, family)
            comp = _significant_tokens(sample_art, ctrl_art)
            sig_mask = (
                (comp["p_val_adj"] < SIG_FDR)
                & (comp["odds_ratio"] > SIG_ODDS)
                & (comp["sample_clonotypes"] >= MIN_CLONOTYPE_SUPPORT)
            )
            n_sig = int(sig_mask.sum())
            if family == "v3":
                baseline_sig[str(study)] = n_sig

            enriched_tokens = set(comp.index[sig_mask])
            for method in METHODS:
                labels = _cluster_labels(sdf, sample_art, enriched_tokens, method)
                eval_df = sdf[["row_id", "gliph_cluster_id"]].copy()
                eval_df["target"] = eval_df["gliph_cluster_id"].fillna("").astype(str).str.strip()
                eval_df["pred"] = eval_df["row_id"].map(labels).fillna(-1).astype(int)
                eval_df = eval_df[
                    (eval_df["target"] != "")
                    & (eval_df["pred"] >= 0)
                ].copy()

                if len(eval_df) >= 2 and eval_df["target"].nunique() >= 2 and eval_df["pred"].nunique() >= 2:
                    ami = adjusted_mutual_info_score(eval_df["target"], eval_df["pred"])
                else:
                    ami = np.nan

                results.append(
                    {
                        "study": str(study),
                        "family": family,
                        "method": method,
                        "n_sig": n_sig,
                        "ami_gliph": ami,
                        "n_eval": len(eval_df),
                    }
                )

    result_df = pd.DataFrame(results).sort_values(["study", "ami_gliph", "n_sig"], ascending=[True, False, True])
    print("\nGLIPH benchmark search summary:")
    print(result_df.to_string(index=False))

    assert not result_df.empty
    assert result_df["study"].nunique() == 2
    assert set(result_df["family"]) == set(FAMILIES)

    for study, sdf in result_df.groupby("study"):
        assert study in baseline_sig
        improved = sdf[sdf["n_sig"] < baseline_sig[study]]
        assert not improved.empty, f"Expected a config with fewer significant tokens than v3 baseline for {study}"
        assert sdf["ami_gliph"].notna().any(), f"Expected at least one evaluable AMI result for {study}"