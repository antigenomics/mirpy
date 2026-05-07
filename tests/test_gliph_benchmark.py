"""Benchmark and parameter search for GLIPH-style clustering on AIRR benchmark.

Runs only with ``RUN_BENCHMARK=1``. The benchmark now mirrors the notebook's
expanded workflow:

- deduplicate GLIPH rows to unique ``(reference_id, v_gene, junction_aa)``
  clonotypes,
- match controls on V usage only,
- run separate Fisher tests for five token families,
- combine all enriched tokens into one projected clonotype graph,
- evaluate concordance to ``gliph_cluster_id`` and ``stimulus``.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from mir.biomarkers.gliph import (
    GliphTokenArtifacts,
    deduplicate_clonotype_rows,
    extract_g4mer_artifacts,
    extract_g5mer_artifacts,
    extract_u4mer_artifacts,
    extract_v3mer_artifacts,
    extract_vpos3mer_artifacts,
    normalize_control_v,
)
from mir.biomarkers.kmer_stats import compare_kmer_counts
from mir.common.control import ControlManager
from tests.conftest import skip_benchmarks


GLIPH_PATH = Path(__file__).resolve().parents[1] / "airr_benchmark" / "gliph" / "gliph_trb.tsv.gz"
AA_RE = r"^[ACDEFGHIKLMNPQRSTVWY]+$"
THREADS = 4
CONTROL_SAMPLE = 1_000_000
FAMILIES = ("v3", "vpos3", "u4", "g4", "g5")
METHODS = ("components", "leiden")
MIN_CLONOTYPE_SUPPORT = 2
MIN_CLUSTER_SIZE = 3
SIG_FDR = 0.05
SIG_ODDS = 1.0
CLONE_EDGE_MIN_WEIGHT = 0.35


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
    return deduplicate_clonotype_rows(out, subset=("reference_id", "v_gene", "junction_aa"))


def _extract_family(df: pd.DataFrame, family: str) -> GliphTokenArtifacts:
    kwargs = dict(threads=THREADS, count_mode="clonotype", unique_clonotypes=False)
    if family == "v3":
        return extract_v3mer_artifacts(df, **kwargs)
    if family == "vpos3":
        return extract_vpos3mer_artifacts(df, **kwargs)
    if family == "u4":
        return extract_u4mer_artifacts(df, **kwargs)
    if family == "g4":
        return extract_g4mer_artifacts(df, **kwargs)
    if family == "g5":
        return extract_g5mer_artifacts(df, **kwargs)
    raise ValueError(f"Unknown family {family}")


def _comparison_df(sample_art: GliphTokenArtifacts, ctrl_art: GliphTokenArtifacts) -> pd.DataFrame:
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


def _family_payloads(study_df: pd.DataFrame, ctrl_df: pd.DataFrame) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    payloads: dict[str, dict[str, object]] = {}
    n_sig_by_family: dict[str, int] = {}
    for family in FAMILIES:
        sample_art = _extract_family(study_df, family)
        ctrl_art = _extract_family(ctrl_df, family)
        comp = _comparison_df(sample_art, ctrl_art)
        sig_mask = (
            (comp["p_val_adj"] < SIG_FDR)
            & (comp["odds_ratio"] > SIG_ODDS)
            & (comp["sample_clonotypes"] >= MIN_CLONOTYPE_SUPPORT)
        )
        enriched_tokens = set(comp.index[sig_mask])
        payloads[family] = {
            "sample_art": sample_art,
            "comparison": comp,
            "enriched_tokens": enriched_tokens,
        }
        n_sig_by_family[family] = int(sig_mask.sum())
    return payloads, n_sig_by_family


def _combine_enriched_tokens(payloads: dict[str, dict[str, object]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    token_to_clones: dict[str, set[str]] = defaultdict(set)
    clone_to_tokens: dict[str, set[str]] = defaultdict(set)
    for family, payload in payloads.items():
        sample_art = payload["sample_art"]
        for token in payload["enriched_tokens"]:
            clone_ids = set(sample_art.token_to_clone.get(token, set()))
            if not clone_ids:
                continue
            token_to_clones[token].update(clone_ids)
            for clone_id in clone_ids:
                clone_to_tokens[clone_id].add(token)
    return dict(token_to_clones), dict(clone_to_tokens)


def _project_clonotype_graph(study_df: pd.DataFrame, token_to_clones: dict[str, set[str]]) -> tuple[ig.Graph, list[str]]:
    clone_ids = sorted(study_df["row_id"].astype(str).tolist())
    active_clone_ids = sorted({clone_id for clone_ids_for_token in token_to_clones.values() for clone_id in clone_ids_for_token})
    if not active_clone_ids:
        graph = ig.Graph(n=0, directed=False)
        graph.vs["name"] = []
        return graph, clone_ids

    clone_idx = {clone_id: i for i, clone_id in enumerate(active_clone_ids)}
    edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    for clone_ids_for_token in token_to_clones.values():
        ordered = sorted(clone_ids_for_token)
        if len(ordered) < 2:
            continue
        for left, right in combinations(ordered, 2):
            edge = tuple(sorted((clone_idx[left], clone_idx[right])))
            edge_weights[edge] += 1

    graph = ig.Graph(n=len(active_clone_ids), directed=False)
    graph.vs["name"] = active_clone_ids
    if edge_weights:
        graph.add_edges(list(edge_weights.keys()))
        graph.es["weight"] = [float(weight) for weight in edge_weights.values()]
    return graph, clone_ids


def _build_weighted_clonotype_graph(
    study_df: pd.DataFrame,
    token_to_clones: dict[str, set[str]],
    *,
    min_weight: float = CLONE_EDGE_MIN_WEIGHT,
) -> tuple[ig.Graph, list[str]]:
    raw_graph, all_clone_ids = _project_clonotype_graph(study_df, token_to_clones)
    if raw_graph.vcount() == 0:
        return raw_graph, all_clone_ids

    clone_idx = {clone_id: idx for idx, clone_id in enumerate(raw_graph.vs["name"])}
    edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    for clone_ids_for_token in token_to_clones.values():
        ordered = sorted(clone_ids_for_token)
        degree = len(ordered)
        if degree < 2:
            continue
        contribution = 1.0 / max(1.0, float(degree - 1))
        for left, right in combinations(ordered, 2):
            edge = tuple(sorted((clone_idx[left], clone_idx[right])))
            edge_weights[edge] += contribution

    keep_edges = [(edge, weight) for edge, weight in edge_weights.items() if weight >= min_weight]
    weighted = ig.Graph(n=raw_graph.vcount(), directed=False)
    weighted.vs["name"] = raw_graph.vs["name"]
    if keep_edges:
        weighted.add_edges([edge for edge, _ in keep_edges])
        weighted.es["weight"] = [float(weight) for _, weight in keep_edges]
    return weighted, all_clone_ids


def _community_labels(study_df: pd.DataFrame, token_to_clones: dict[str, set[str]], method: str) -> tuple[dict[str, int], dict[str, int]]:
    graph, all_clone_ids = _build_weighted_clonotype_graph(study_df, token_to_clones)
    labels = {clone_id: -1 for clone_id in all_clone_ids}
    if graph.vcount() == 0:
        return labels, {"n_clusters": 0, "n_clustered": 0, "n_total": len(all_clone_ids)}

    if graph.ecount() == 0:
        return labels, {"n_clusters": 0, "n_clustered": 0, "n_total": len(all_clone_ids)}

    if method == "components":
        membership = np.full(graph.vcount(), -1, dtype=int)
        for component_id, vertices in enumerate(graph.components()):
            for vertex in vertices:
                membership[vertex] = component_id
    elif method == "leiden":
        membership = np.array(
            graph.community_leiden(weights="weight", objective_function="modularity", n_iterations=5).membership,
            dtype=int,
        )
    else:
        raise ValueError(method)

    keep_mask = np.zeros_like(membership, dtype=bool)
    for label in np.unique(membership):
        if int((membership == label).sum()) >= MIN_CLUSTER_SIZE:
            keep_mask |= membership == label
    membership = np.where(keep_mask, membership, -1)

    for clone_id, label in zip(graph.vs["name"], membership):
        labels[str(clone_id)] = int(label)

    stats = {
        "n_clusters": int(len(set(membership[membership >= 0]))),
        "n_clustered": int((membership >= 0).sum()),
        "n_total": len(all_clone_ids),
    }
    return labels, stats


def _metric_row(study_df: pd.DataFrame, labels: dict[str, int], target_col: str) -> dict[str, float]:
    eval_df = study_df[["row_id", target_col]].copy()
    eval_df["target"] = eval_df[target_col].fillna("").astype(str).str.strip()
    eval_df = eval_df[~eval_df["target"].isin({"", "nan", "None", "none", "NA", "na"})].copy()
    eval_df["pred"] = eval_df["row_id"].map(labels).fillna(-1).astype(int)
    coverage = float((eval_df["pred"] >= 0).mean()) if len(eval_df) else 0.0
    eval_df = eval_df[eval_df["pred"] >= 0].copy()

    if len(eval_df) < 2 or eval_df["pred"].nunique() < 2 or eval_df["target"].nunique() < 2:
        return {"ami": np.nan, "nmi": np.nan, "coverage": coverage, "n_eval": len(eval_df)}

    return {
        "ami": adjusted_mutual_info_score(eval_df["target"], eval_df["pred"]),
        "nmi": normalized_mutual_info_score(eval_df["target"], eval_df["pred"]),
        "coverage": coverage,
        "n_eval": len(eval_df),
    }


@skip_benchmarks
@pytest.mark.slow_benchmark
def test_gliph_combined_graph_benchmark() -> None:
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

    rows: list[dict[str, object]] = []
    for study, study_df in df.groupby("reference_id", sort=True):
        ctrl_df = normalize_control_v(study_df, ctrl_all, n=CONTROL_SAMPLE, seed=42)
        payloads, n_sig_by_family = _family_payloads(study_df, ctrl_df)
        token_to_clones, _clone_to_tokens = _combine_enriched_tokens(payloads)

        for method in METHODS:
            labels, stats = _community_labels(study_df, token_to_clones, method)
            gliph_metrics = _metric_row(study_df, labels, "gliph_cluster_id")
            stimulus_metrics = _metric_row(study_df, labels, "stimulus")
            rows.append(
                {
                    "study": str(study),
                    "method": method,
                    "n_enriched_total": int(sum(n_sig_by_family.values())),
                    "n_enriched_v3": n_sig_by_family["v3"],
                    "n_enriched_vpos3": n_sig_by_family["vpos3"],
                    "n_enriched_u4": n_sig_by_family["u4"],
                    "n_enriched_g4": n_sig_by_family["g4"],
                    "n_enriched_g5": n_sig_by_family["g5"],
                    "n_clusters": stats["n_clusters"],
                    "n_clustered": stats["n_clustered"],
                    "ami_gliph": gliph_metrics["ami"],
                    "nmi_gliph": gliph_metrics["nmi"],
                    "coverage_gliph": gliph_metrics["coverage"],
                    "n_eval_gliph": gliph_metrics["n_eval"],
                    "ami_stimulus": stimulus_metrics["ami"],
                    "nmi_stimulus": stimulus_metrics["nmi"],
                    "coverage_stimulus": stimulus_metrics["coverage"],
                    "n_eval_stimulus": stimulus_metrics["n_eval"],
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["study", "ami_gliph", "coverage_gliph"], ascending=[True, False, False])
    print("\nCombined GLIPH graph benchmark summary:")
    print(result_df.to_string(index=False))

    assert not result_df.empty
    assert set(result_df["study"]) == {"Glanville2017", "Huang2020"}
    assert result_df["n_enriched_g5"].ge(0).all()
    assert result_df["n_enriched_total"].gt(0).all()
    assert result_df["coverage_gliph"].gt(0).any()

    for study, study_rows in result_df.groupby("study"):
        assert study_rows["ami_gliph"].notna().any(), f"Expected evaluable gliph_cluster_id AMI for {study}"
        best_row = study_rows.sort_values(["ami_gliph", "coverage_gliph"], ascending=[False, False]).iloc[0]
        assert best_row["method"] == "leiden"
        assert float(best_row["ami_gliph"]) > 0.2, f"Expected combined weighted Leiden AMI > 0.2 for {study}"
        if study == "Glanville2017":
            assert study_rows["ami_stimulus"].notna().any(), "Expected evaluable stimulus AMI for Glanville2017"