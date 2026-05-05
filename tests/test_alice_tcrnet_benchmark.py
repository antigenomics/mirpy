"""Benchmark ALICE (OLGA/Pgen) vs TCRNET on B35+/CMV+ style samples.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_alice_tcrnet_benchmark.py -s

This benchmark compares:
- ALICE (hamming, thresholds 0/1, match modes none/v/j/v_j)
- TCRNET (hamming + levenshtein, thresholds 0/1, match modes none/v/j/v_j)
- TCRNET controls: real and synthetic

For each run we compute:
- runtime
- neighbor-enriched clonotypes
- connected components among enriched clonotypes
- enrichment of target epitope/HLA groups from VDJdb

Then we summarize concordance between ALICE and TCRNET (synthetic/real) by
component-size profiles and target-hit overlap.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from mir.biomarkers.alice import compute_alice
from mir.biomarkers.tcrnet import compute_tcrnet
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire
from mir.graph.edit_distance_graph import build_edit_distance_graph
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import benchmark_max_seconds, skip_benchmarks

_ASSETS = Path(__file__).parent / "assets"
_B35_FILE = _ASSETS / "B35+.txt.gz"
_CMV_FILE = _ASSETS / "CMV+.txt.gz"
_VDJDB_FILE = _ASSETS / "vdjdb.slim.txt.gz"

_TARGETS_BY_SAMPLE: dict[str, list[tuple[str, str, str]]] = {
    "CMV+": [
        ("A*02", "NLV", "NLVPMVATV"),
        ("B*07", "RPH", "RPHERNGFTVL"),
        ("B*07", "TPR", "TPRVTGGGAM"),
    ],
    "B35+": [
        ("B*35", "EPL", "EPLPQGQLTAY"),
        ("B*35", "HPV", "HPVTKYIM"),
    ],
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return max(1, val)


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.Series(p_values, dtype=float)
    if values.empty:
        return values

    order = np.argsort(values.to_numpy())
    ranked = values.to_numpy()[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, ranked[i] * n / (i + 1))
        adjusted[i] = running

    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(out, index=values.index)


def _top_clone_count_limited(rep: LocusRepertoire, *, max_clonotypes: int) -> LocusRepertoire:
    clones = sorted(rep.clonotypes, key=lambda c: int(c.duplicate_count or 0), reverse=True)
    clones = clones[:max_clonotypes]
    out = [
        Clonotype(
            sequence_id=str(i),
            locus="TRB",
            junction_aa=c.junction_aa,
            junction=c.junction,
            v_gene=c.v_gene,
            j_gene=c.j_gene,
            duplicate_count=int(c.duplicate_count or 1),
            _validate=False,
        )
        for i, c in enumerate(clones)
    ]
    return LocusRepertoire(clonotypes=out, locus="TRB", repertoire_id=rep.repertoire_id)


def _load_trb_from_table(path: Path, sample_id: str) -> LocusRepertoire:
    parser = ClonotypeTableParser()
    rep = LocusRepertoire(parser.parse(str(path)), locus="TRB", repertoire_id=sample_id)
    rep = filter_functional(rep)
    return LocusRepertoire(clonotypes=list(rep.clonotypes), locus="TRB", repertoire_id=sample_id)


def _load_b35_sample(max_clonotypes: int) -> LocusRepertoire:
    if not _B35_FILE.exists():
        pytest.skip("B35+ benchmark asset missing: tests/assets/B35+.txt.gz")
    rep = _load_trb_from_table(_B35_FILE, "B35+")
    return _top_clone_count_limited(rep, max_clonotypes=max_clonotypes)


def _parse_vdjdb_df() -> pd.DataFrame:
    if not _VDJDB_FILE.exists():
        pytest.skip("VDJdb benchmark asset missing: tests/assets/vdjdb.slim.txt.gz")
    raw = pd.read_csv(_VDJDB_FILE, sep="\t", compression="gzip")
    raw = raw[raw["species"].eq("HomoSapiens") & raw["gene"].eq("TRB")].copy()
    raw["cdr3"] = raw["cdr3"].fillna("").astype(str)
    raw["mhc.a"] = raw["mhc.a"].fillna("").astype(str)
    raw = raw[raw["cdr3"].str.len().gt(0)].copy()
    return raw.reset_index(drop=True)


def _load_cmv_sample(max_clonotypes: int, vdjdb_df: pd.DataFrame) -> tuple[LocusRepertoire, str]:
    if _CMV_FILE.exists():
        rep = _load_trb_from_table(_CMV_FILE, "CMV+")
        return _top_clone_count_limited(rep, max_clonotypes=max_clonotypes), "file"

    # Fallback: build a CMV+ proxy from VDJdb CMV epitopes + heavy-tail background.
    cmv_targets = {"NLVPMVATV", "RPHERNGFTVL", "TPRVTGGGAM"}
    df = vdjdb_df[vdjdb_df["antigen.epitope"].isin(cmv_targets)].copy()
    if df.empty:
        pytest.skip("Cannot build CMV+ proxy: requested CMV epitopes not found in VDJdb asset")

    df = df.drop_duplicates(subset=["cdr3", "v.segm", "j.segm", "antigen.epitope"]).copy()

    top_target = df.head(max(400, max_clonotypes // 2)).copy()
    rows: list[Clonotype] = []
    for i, (_, row) in enumerate(top_target.iterrows()):
        ep = str(row["antigen.epitope"])
        dup = 40 if ep == "NLVPMVATV" else 20
        rows.append(
            Clonotype(
                sequence_id=f"cmv_target_{i}",
                locus="TRB",
                junction_aa=str(row["cdr3"]),
                v_gene=str(row.get("v.segm", "") or ""),
                j_gene=str(row.get("j.segm", "") or ""),
                duplicate_count=dup,
                _validate=False,
            )
        )

    # Add noisy background from non-CMV VDJdb TRB to mimic realistic tails.
    bg = vdjdb_df[~vdjdb_df["antigen.epitope"].isin(cmv_targets)].copy()
    bg = bg.drop_duplicates(subset=["cdr3", "v.segm", "j.segm"]).head(max(200, max_clonotypes // 2))
    for i, (_, row) in enumerate(bg.iterrows()):
        rows.append(
            Clonotype(
                sequence_id=f"cmv_bg_{i}",
                locus="TRB",
                junction_aa=str(row["cdr3"]),
                v_gene=str(row.get("v.segm", "") or ""),
                j_gene=str(row.get("j.segm", "") or ""),
                duplicate_count=1,
                _validate=False,
            )
        )

    # De-duplicate by sequence/V/J while preserving largest duplicate_count.
    best: dict[tuple[str, str, str], Clonotype] = {}
    for c in rows:
        key = (c.junction_aa, c.v_gene, c.j_gene)
        prev = best.get(key)
        if prev is None or int(c.duplicate_count) > int(prev.duplicate_count):
            best[key] = c

    uniq = sorted(best.values(), key=lambda c: int(c.duplicate_count), reverse=True)
    uniq = uniq[:max_clonotypes]
    for i, c in enumerate(uniq):
        c.sequence_id = str(i)
        c.locus = "TRB"
    return LocusRepertoire(clonotypes=uniq, locus="TRB", repertoire_id="CMV+"), "proxy"


def _build_target_reference(vdjdb_df: pd.DataFrame, sample_id: str) -> tuple[pd.DataFrame, list[str]]:
    target_specs = _TARGETS_BY_SAMPLE[sample_id]
    parts: list[pd.DataFrame] = []
    missing: list[str] = []

    for hla_sub, short_name, epitope in target_specs:
        sub = vdjdb_df[
            vdjdb_df["antigen.epitope"].eq(epitope)
            & vdjdb_df["mhc.a"].str.contains(hla_sub, regex=False)
        ].copy()
        label = f"{hla_sub} {short_name}"
        if sub.empty:
            missing.append(label)
            continue
        sub["target_label"] = label
        parts.append(sub)

    if not parts:
        return pd.DataFrame(columns=["cdr3", "target_label"]), missing

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["cdr3", "target_label"])
    return out, missing


def _enriched_table_from_method_output(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()

    df = table.copy()
    df["p.adj"] = _bh_adjust(df["p_value"])
    return df[(df["p.adj"] <= 0.05) & (df["fold_enrichment"] > 1.0)].copy()


def _component_sizes(
    enriched: pd.DataFrame,
    *,
    metric: str,
    threshold: int,
) -> list[int]:
    if enriched.empty:
        return []

    rows: list[Clonotype] = []
    uniq = enriched.drop_duplicates(subset=["junction_aa", "v_gene", "j_gene"]).copy()
    for i, row in enumerate(uniq.itertuples(index=False)):
        rows.append(
            Clonotype(
                sequence_id=str(i),
                locus="TRB",
                junction_aa=str(row.junction_aa),
                v_gene=str(row.v_gene or ""),
                j_gene=str(row.j_gene or ""),
                duplicate_count=1,
                _validate=False,
            )
        )

    graph = build_edit_distance_graph(
        rows,
        metric=metric,
        threshold=threshold,
        n_jobs=4,
    )
    sizes = sorted(graph.components().sizes(), reverse=True)
    return [int(x) for x in sizes]


def _target_hits(
    enriched: pd.DataFrame,
    ref_df: pd.DataFrame,
    *,
    threshold: int,
) -> tuple[dict[str, int], set[str]]:
    if enriched.empty or ref_df.empty:
        return {}, set()

    target_to_seqs: dict[str, list[str]] = {}
    for label, grp in ref_df.groupby("target_label"):
        target_to_seqs[label] = grp["cdr3"].dropna().astype(str).drop_duplicates().tolist()

    enriched_seqs = enriched["junction_aa"].dropna().astype(str).drop_duplicates().tolist()
    e_clones = [
        Clonotype(sequence_id=str(i), locus="TRB", junction_aa=s, duplicate_count=1, _validate=False)
        for i, s in enumerate(enriched_seqs)
    ]
    erep = LocusRepertoire(clonotypes=e_clones, locus="TRB")

    hits_by_target: dict[str, int] = {}
    all_hit_sequences: set[str] = set()
    for label, refs in target_to_seqs.items():
        hit: set[str] = set()
        for seq in refs:
            matches = erep.trie.SearchIndices(
                query=seq,
                maxSubstitution=int(threshold),
                maxInsertion=0,
                maxDeletion=0,
                maxEdits=int(threshold),
            )
            for idx, _ in matches:
                hit.add(erep.clonotypes[int(idx)].junction_aa)
        hits_by_target[label] = len(hit)
        all_hit_sequences.update(hit)

    return hits_by_target, all_hit_sequences


def _pad_vector(vals: list[int], *, k: int = 10) -> np.ndarray:
    out = np.zeros(k, dtype=float)
    if vals:
        lim = min(k, len(vals))
        out[:lim] = np.asarray(vals[:lim], dtype=float)
    return out


@dataclass
class RunResult:
    sample_id: str
    method: str
    control_kind: str
    metric: str
    threshold: int
    match_mode: str
    elapsed_s: float
    ok: bool
    error: str
    n_total: int
    n_enriched: int
    n_components: int
    largest_component: int
    target_hits: dict[str, int]
    component_sizes: list[int]
    hit_sequences: set[str]


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_benchmark_alice_vs_tcrnet_components_and_targets(capsys) -> None:
    t_start = time.perf_counter()

    max_clonotypes = _env_int("MIRPY_BENCH_MAX_CLONOTYPES", 2500)
    synthetic_n = _env_int("MIRPY_BENCH_SYNTHETIC_N", 250_000)
    n_jobs = _env_int("MIRPY_BENCH_N_JOBS", 4)
    gene_modes = ["none", "v", "j", "v_j"]
    manager = ControlManager()

    vdjdb_df = _parse_vdjdb_df()
    b35 = _load_b35_sample(max_clonotypes=max_clonotypes)
    cmv, cmv_source = _load_cmv_sample(max_clonotypes=max_clonotypes, vdjdb_df=vdjdb_df)

    samples = {
        "B35+": b35,
        "CMV+": cmv,
    }

    runs: list[RunResult] = []
    missing_targets: dict[str, list[str]] = {}

    for sample_id, rep in samples.items():
        assert rep.locus == "TRB"

        ref_df, missing = _build_target_reference(vdjdb_df, sample_id)
        missing_targets[sample_id] = missing

        for threshold in (0, 1):
            for match_mode in gene_modes:
                # ALICE (hamming only)
                t0 = time.perf_counter()
                try:
                    alice = compute_alice(
                        rep,
                        species="human",
                        threshold=threshold,
                        match_mode=match_mode,
                        metric="hamming",
                        pgen_mode="exact",
                        gene_usage_synthetic_n=synthetic_n,
                        control_manager=manager,
                        n_jobs=n_jobs,
                    )
                    elapsed = time.perf_counter() - t0
                    enr = _enriched_table_from_method_output(alice.table)
                    comp_sizes = _component_sizes(enr, metric="hamming", threshold=threshold)
                    hits, hit_sequences = _target_hits(enr, ref_df, threshold=1)
                    runs.append(
                        RunResult(
                            sample_id=sample_id,
                            method="alice",
                            control_kind="synthetic",
                            metric="hamming",
                            threshold=threshold,
                            match_mode=match_mode,
                            elapsed_s=elapsed,
                            ok=True,
                            error="",
                            n_total=len(alice.table),
                            n_enriched=len(enr),
                            n_components=len(comp_sizes),
                            largest_component=(comp_sizes[0] if comp_sizes else 0),
                            target_hits=hits,
                            component_sizes=comp_sizes,
                            hit_sequences=hit_sequences,
                        )
                    )
                except Exception as exc:  # pragma: no cover - diagnostic path
                    elapsed = time.perf_counter() - t0
                    runs.append(
                        RunResult(
                            sample_id=sample_id,
                            method="alice",
                            control_kind="synthetic",
                            metric="hamming",
                            threshold=threshold,
                            match_mode=match_mode,
                            elapsed_s=elapsed,
                            ok=False,
                            error=f"{type(exc).__name__}: {exc}",
                            n_total=0,
                            n_enriched=0,
                            n_components=0,
                            largest_component=0,
                            target_hits={},
                            component_sizes=[],
                            hit_sequences=set(),
                        )
                    )

                # TCRNET: hamming + levenshtein, real and synthetic controls.
                for metric in ("hamming", "levenshtein"):
                    for control_kind in ("real", "synthetic"):
                        t1 = time.perf_counter()
                        try:
                            kwargs = {}
                            if control_kind == "synthetic":
                                kwargs = {
                                    "control_kwargs": {
                                        "n": synthetic_n,
                                        "seed": 42,
                                        "n_jobs": n_jobs,
                                        "progress": False,
                                    }
                                }
                            tcr = compute_tcrnet(
                                rep,
                                control_type=control_kind,
                                species="human",
                                control_manager=manager,
                                metric=metric,
                                threshold=threshold,
                                match_mode=match_mode,
                                pvalue_mode="binomial",
                                n_jobs=n_jobs,
                                **kwargs,
                            )
                            elapsed = time.perf_counter() - t1
                            enr = _enriched_table_from_method_output(tcr.table)
                            comp_sizes = _component_sizes(enr, metric=metric, threshold=threshold)
                            hits, hit_sequences = _target_hits(enr, ref_df, threshold=1)
                            runs.append(
                                RunResult(
                                    sample_id=sample_id,
                                    method="tcrnet",
                                    control_kind=control_kind,
                                    metric=metric,
                                    threshold=threshold,
                                    match_mode=match_mode,
                                    elapsed_s=elapsed,
                                    ok=True,
                                    error="",
                                    n_total=len(tcr.table),
                                    n_enriched=len(enr),
                                    n_components=len(comp_sizes),
                                    largest_component=(comp_sizes[0] if comp_sizes else 0),
                                    target_hits=hits,
                                    component_sizes=comp_sizes,
                                    hit_sequences=hit_sequences,
                                )
                            )
                        except Exception as exc:  # pragma: no cover - diagnostic path
                            elapsed = time.perf_counter() - t1
                            runs.append(
                                RunResult(
                                    sample_id=sample_id,
                                    method="tcrnet",
                                    control_kind=control_kind,
                                    metric=metric,
                                    threshold=threshold,
                                    match_mode=match_mode,
                                    elapsed_s=elapsed,
                                    ok=False,
                                    error=f"{type(exc).__name__}: {exc}",
                                    n_total=0,
                                    n_enriched=0,
                                    n_components=0,
                                    largest_component=0,
                                    target_hits={},
                                    component_sizes=[],
                                    hit_sequences=set(),
                                )
                            )

    assert runs, "Benchmark produced no runs"

    run_df = pd.DataFrame(
        [
            {
                "sample_id": r.sample_id,
                "method": r.method,
                "control_kind": r.control_kind,
                "metric": r.metric,
                "threshold": r.threshold,
                "match_mode": r.match_mode,
                "elapsed_s": r.elapsed_s,
                "ok": r.ok,
                "error": r.error,
                "n_total": r.n_total,
                "n_enriched": r.n_enriched,
                "n_components": r.n_components,
                "largest_component": r.largest_component,
                "target_hits_total": int(sum(r.target_hits.values())) if r.target_hits else 0,
            }
            for r in runs
        ]
    )

    # Concordance: compare ALICE vs TCRNET synthetic/real in hamming settings.
    idx = {
        (r.sample_id, r.method, r.control_kind, r.metric, r.threshold, r.match_mode): r
        for r in runs
        if r.ok
    }

    conc_rows: list[dict[str, float | str]] = []
    for sample_id in samples:
        for threshold in (0, 1):
            for match_mode in gene_modes:
                key_alice = (sample_id, "alice", "synthetic", "hamming", threshold, match_mode)
                a = idx.get(key_alice)
                if a is None:
                    continue
                for control_kind in ("synthetic", "real"):
                    key_t = (sample_id, "tcrnet", control_kind, "hamming", threshold, match_mode)
                    t = idx.get(key_t)
                    if t is None:
                        continue
                    x = _pad_vector(a.component_sizes, k=10)
                    y = _pad_vector(t.component_sizes, k=10)
                    corr = spearmanr(x, y).statistic
                    if corr is None or math.isnan(float(corr)):
                        corr = 0.0

                    union = a.hit_sequences | t.hit_sequences
                    inter = a.hit_sequences & t.hit_sequences
                    jacc = float(len(inter) / len(union)) if union else 1.0

                    conc_rows.append(
                        {
                            "sample_id": sample_id,
                            "threshold": threshold,
                            "match_mode": match_mode,
                            "tcrnet_control": control_kind,
                            "spearman_component_sizes": float(corr),
                            "jaccard_target_hits": jacc,
                            "alice_hits": len(a.hit_sequences),
                            "tcrnet_hits": len(t.hit_sequences),
                        }
                    )

    conc_df = pd.DataFrame(conc_rows)

    elapsed_total = time.perf_counter() - t_start
    ok_df = run_df[run_df["ok"]].copy()
    err_df = run_df[~run_df["ok"]].copy()

    benchmark_log_line(
        "alice_tcrnet_compare "
        f"elapsed_total_s={elapsed_total:.3f} "
        f"runs_total={len(run_df)} runs_ok={len(ok_df)} runs_error={len(err_df)} "
        f"b35_n={len(b35.clonotypes)} cmv_n={len(cmv.clonotypes)} cmv_source={cmv_source} "
        f"synthetic_n={synthetic_n}"
    )

    with capsys.disabled():
        print("\n" + "=" * 88)
        print("ALICE vs TCRNET benchmark (TRB/human)")
        print(f"total elapsed: {elapsed_total:.2f}s")
        print(f"B35+ clonotypes: {len(b35.clonotypes)}")
        print(f"CMV+ clonotypes: {len(cmv.clonotypes)} (source={cmv_source})")
        print(f"synthetic control n: {synthetic_n}")
        if missing_targets:
            for sid, miss in missing_targets.items():
                if miss:
                    print(f"missing target references for {sid}: {', '.join(miss)}")

        print("\nRun summary (ok runs):")
        if ok_df.empty:
            print("  no successful runs")
        else:
            print(
                ok_df.groupby(["sample_id", "method", "control_kind", "metric", "threshold"], as_index=False)
                .agg(
                    elapsed_s=("elapsed_s", "mean"),
                    n_enriched=("n_enriched", "median"),
                    largest_component=("largest_component", "median"),
                    target_hits_total=("target_hits_total", "median"),
                )
                .sort_values(["sample_id", "method", "control_kind", "metric", "threshold"]) 
                .to_string(index=False)
            )

        print("\nConcordance (ALICE vs TCRNET, hamming):")
        if conc_df.empty:
            print("  no comparable successful pairs")
        else:
            print(
                conc_df.groupby(["sample_id", "tcrnet_control"], as_index=False)
                .agg(
                    spearman_component_sizes=("spearman_component_sizes", "mean"),
                    jaccard_target_hits=("jaccard_target_hits", "mean"),
                    alice_hits=("alice_hits", "mean"),
                    tcrnet_hits=("tcrnet_hits", "mean"),
                )
                .sort_values(["sample_id", "tcrnet_control"])
                .to_string(index=False)
            )

        if not err_df.empty:
            print("\nErrors:")
            cols = [
                "sample_id",
                "method",
                "control_kind",
                "metric",
                "threshold",
                "match_mode",
                "elapsed_s",
                "error",
            ]
            print(err_df[cols].to_string(index=False))
        print("=" * 88)

    # Acceptance: benchmark must execute enough successful runs and stay in time budget.
    assert len(ok_df) >= 12

    # If CMV+ file is present, enforce no proxy usage.
    if _CMV_FILE.exists():
        assert cmv_source == "file"

    max_s = benchmark_max_seconds(default=2400.0)
    assert elapsed_total < max_s
