"""Benchmarks comparing ALICE and TCRNET on B35+/CMV+ style samples.

Run with:
    RUN_BENCHMARK=1 pytest tests/test_alice_tcrnet_benchmark.py -s

The suite is split into bounded tests:
- fast hamming ALICE vs synthetic-control TCRNET concordance,
- fast synthetic-control Levenshtein TCRNET runs,
- slower real-control hamming TCRNET runs,
- slower real-control Levenshtein TCRNET runs.

This avoids one monolithic >1 h test and makes the slowest configurations
visible in per-test summaries.
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
from mir.common.alleles import allele_to_major
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire
from mir.graph.edit_distance_graph import build_edit_distance_graph
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import skip_benchmarks

_ASSETS = Path(__file__).parent / "assets"
_B35_FILE = _ASSETS / "real_repertoires" / "B35+.txt.gz"
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


@dataclass(frozen=True)
class RunSpec:
    sample_id: str
    method: str
    control_kind: str
    metric: str
    threshold: int
    match_mode: str


@dataclass
class RunResult:
    spec: RunSpec
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(1.0, value)


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


def _clone_from_row(row: pd.Series, *, sid: str, duplicate_count: int = 1) -> Clonotype:
    return Clonotype(
        sequence_id=sid,
        locus="TRB",
        junction_aa=str(row.get("cdr3", "") or row.get("junction_aa", "")),
        junction=str(row.get("cdr3nt", "") or row.get("junction", "")),
        v_gene=str(row.get("v.segm", row.get("v_gene", row.get("v", ""))) or ""),
        j_gene=str(row.get("j.segm", row.get("j_gene", row.get("j", ""))) or ""),
        duplicate_count=max(1, int(duplicate_count)),
        _validate=False,
    )


def _df_to_repertoire(df: pd.DataFrame, *, repertoire_id: str, limit: int | None = None) -> LocusRepertoire:
    work = df.copy()
    if "duplicate_count" not in work.columns:
        work["duplicate_count"] = 1
    if limit is not None and limit > 0:
        work = work.sort_values("duplicate_count", ascending=False).head(int(limit)).copy()
    clones = [
        Clonotype(
            sequence_id=str(i),
            locus="TRB",
            junction_aa=str(row.get("junction_aa", "")),
            junction=str(row.get("junction", "")),
            v_gene=allele_to_major(str(row.get("v_gene", "") or "")),
            j_gene=allele_to_major(str(row.get("j_gene", "") or "")),
            duplicate_count=max(1, int(row.get("duplicate_count", 1) or 1)),
            _validate=False,
        )
        for i, row in enumerate(work.to_dict(orient="records"))
    ]
    return LocusRepertoire(clonotypes=clones, locus="TRB", repertoire_id=repertoire_id)


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
        pytest.skip("B35+ benchmark asset missing: tests/assets/real_repertoires/B35+.txt.gz")
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

    cmv_targets = {"NLVPMVATV", "RPHERNGFTVL", "TPRVTGGGAM"}
    df = vdjdb_df[vdjdb_df["antigen.epitope"].isin(cmv_targets)].copy()
    if df.empty:
        pytest.skip("Cannot build CMV+ proxy: requested CMV epitopes not found in VDJdb asset")

    df = df.drop_duplicates(subset=["cdr3", "v.segm", "j.segm", "antigen.epitope"]).copy()
    top_target = df.head(max(120, max_clonotypes // 2)).copy()
    rows: list[Clonotype] = []
    for i, (_, row) in enumerate(top_target.iterrows()):
        dup = 40 if str(row["antigen.epitope"]) == "NLVPMVATV" else 20
        rows.append(_clone_from_row(row, sid=f"cmv_target_{i}", duplicate_count=dup))

    bg = vdjdb_df[~vdjdb_df["antigen.epitope"].isin(cmv_targets)].copy()
    bg = bg.drop_duplicates(subset=["cdr3", "v.segm", "j.segm"]).head(max(80, max_clonotypes // 2))
    for i, (_, row) in enumerate(bg.iterrows()):
        rows.append(_clone_from_row(row, sid=f"cmv_bg_{i}", duplicate_count=1))

    best: dict[tuple[str, str, str], Clonotype] = {}
    for c in rows:
        key = (c.junction_aa, c.v_gene, c.j_gene)
        prev = best.get(key)
        if prev is None or int(c.duplicate_count) > int(prev.duplicate_count):
            best[key] = c
    uniq = sorted(best.values(), key=lambda c: int(c.duplicate_count), reverse=True)[:max_clonotypes]
    for i, c in enumerate(uniq):
        c.sequence_id = str(i)
    return LocusRepertoire(clonotypes=uniq, locus="TRB", repertoire_id="CMV+"), "proxy"


def _build_target_reference(vdjdb_df: pd.DataFrame, sample_id: str) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for hla_sub, short_name, epitope in _TARGETS_BY_SAMPLE[sample_id]:
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


def _enriched_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    df = table.copy()
    df["p.adj"] = _bh_adjust(df["p_value"])
    return df[(df["p.adj"] <= 0.05) & (df["fold_enrichment"] > 1.0)].copy()


def _component_sizes(enriched: pd.DataFrame, *, metric: str, threshold: int) -> list[int]:
    if enriched.empty:
        return []
    uniq = enriched.drop_duplicates(subset=["junction_aa", "v_gene", "j_gene"]).copy()
    rows = [
        Clonotype(
            sequence_id=str(i),
            locus="TRB",
            junction_aa=str(row.junction_aa),
            v_gene=str(row.v_gene or ""),
            j_gene=str(row.j_gene or ""),
            duplicate_count=1,
            _validate=False,
        )
        for i, row in enumerate(uniq.itertuples(index=False))
    ]
    graph = build_edit_distance_graph(rows, metric=metric, threshold=threshold, n_jobs=4)
    return [int(x) for x in sorted(graph.components().sizes(), reverse=True)]


def _target_hits(enriched: pd.DataFrame, ref_df: pd.DataFrame, *, threshold: int) -> tuple[dict[str, int], set[str]]:
    if enriched.empty or ref_df.empty:
        return {}, set()
    target_to_seqs = {
        label: grp["cdr3"].dropna().astype(str).drop_duplicates().tolist()
        for label, grp in ref_df.groupby("target_label")
    }
    enriched_seqs = enriched["junction_aa"].dropna().astype(str).drop_duplicates().tolist()
    erep = LocusRepertoire(
        clonotypes=[Clonotype(sequence_id=str(i), locus="TRB", junction_aa=s, duplicate_count=1, _validate=False) for i, s in enumerate(enriched_seqs)],
        locus="TRB",
    )
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


def _build_context(
    *,
    max_clonotypes: int,
    synthetic_n: int,
    real_control_limit: int,
) -> tuple[dict[str, LocusRepertoire], dict[str, pd.DataFrame], dict[str, list[str]], dict[str, LocusRepertoire], str, ControlManager]:
    manager = ControlManager()
    vdjdb_df = _parse_vdjdb_df()
    b35 = _load_b35_sample(max_clonotypes=max_clonotypes)
    cmv, cmv_source = _load_cmv_sample(max_clonotypes=max_clonotypes, vdjdb_df=vdjdb_df)
    samples = {"B35+": b35, "CMV+": cmv}
    refs: dict[str, pd.DataFrame] = {}
    missing: dict[str, list[str]] = {}
    for sample_id in samples:
        refs[sample_id], missing[sample_id] = _build_target_reference(vdjdb_df, sample_id)

    control_df_real = manager.ensure_and_load_control_df("real", "human", "TRB")
    control_df_syn = manager.ensure_and_load_control_df(
        "synthetic",
        "human",
        "TRB",
        n=synthetic_n,
        seed=42,
        n_jobs=4,
        progress=False,
    )
    controls = {
        "real": _df_to_repertoire(control_df_real, repertoire_id="real-control", limit=real_control_limit),
        "synthetic": _df_to_repertoire(control_df_syn, repertoire_id="synthetic-control", limit=synthetic_n),
    }
    return samples, refs, missing, controls, cmv_source, manager


def _execute_run(
    spec: RunSpec,
    *,
    rep: LocusRepertoire,
    ref_df: pd.DataFrame,
    controls: dict[str, LocusRepertoire],
    synthetic_n: int,
    manager: ControlManager,
    n_jobs: int,
) -> RunResult:
    t0 = time.perf_counter()
    try:
        if spec.method == "alice":
            result = compute_alice(
                rep,
                species="human",
                threshold=spec.threshold,
                match_mode=spec.match_mode,
                metric="hamming",
                pgen_mode="exact",
                gene_usage_synthetic_n=synthetic_n,
                control_manager=manager,
                n_jobs=n_jobs,
            )
            table = result.table
        else:
            result = compute_tcrnet(
                rep,
                control=controls[spec.control_kind],
                species="human",
                metric=spec.metric,
                threshold=spec.threshold,
                match_mode=spec.match_mode,
                pvalue_mode="binomial",
                n_jobs=n_jobs,
            )
            table = result.table

        elapsed = time.perf_counter() - t0
        enriched = _enriched_table(table)
        component_sizes = _component_sizes(enriched, metric=spec.metric, threshold=spec.threshold)
        target_hits, hit_sequences = _target_hits(enriched, ref_df, threshold=1)
        return RunResult(
            spec=spec,
            elapsed_s=elapsed,
            ok=True,
            error="",
            n_total=len(table),
            n_enriched=len(enriched),
            n_components=len(component_sizes),
            largest_component=(component_sizes[0] if component_sizes else 0),
            target_hits=target_hits,
            component_sizes=component_sizes,
            hit_sequences=hit_sequences,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return RunResult(
            spec=spec,
            elapsed_s=time.perf_counter() - t0,
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


def _results_to_frame(runs: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": r.spec.sample_id,
                "method": r.spec.method,
                "control_kind": r.spec.control_kind,
                "metric": r.spec.metric,
                "threshold": r.spec.threshold,
                "match_mode": r.spec.match_mode,
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


def _print_summary(
    *,
    title: str,
    elapsed_total: float,
    runs: list[RunResult],
    missing_targets: dict[str, list[str]],
    cmv_source: str,
    extra_df: pd.DataFrame | None = None,
) -> None:
    run_df = _results_to_frame(runs)
    ok_df = run_df[run_df["ok"]].copy()
    err_df = run_df[~run_df["ok"]].copy()
    print("\n" + "=" * 88)
    print(title)
    print(f"total elapsed: {elapsed_total:.2f}s")
    print(f"CMV source: {cmv_source}")
    for sid, miss in missing_targets.items():
        if miss:
            print(f"missing target references for {sid}: {', '.join(miss)}")
    print("\nRun summary:")
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
        print("\nLongest runs:")
        print(
            ok_df.sort_values("elapsed_s", ascending=False)
            .head(6)[["sample_id", "method", "control_kind", "metric", "threshold", "match_mode", "elapsed_s"]]
            .to_string(index=False)
        )
    if extra_df is not None and not extra_df.empty:
        print("\nConcordance:")
        print(extra_df.to_string(index=False))
    if not err_df.empty:
        print("\nErrors:")
        print(err_df[["sample_id", "method", "control_kind", "metric", "threshold", "match_mode", "elapsed_s", "error"]].to_string(index=False))
    print("=" * 88)


def _compute_hamming_concordance(runs: list[RunResult]) -> pd.DataFrame:
    idx = {
        (r.spec.sample_id, r.spec.method, r.spec.control_kind, r.spec.metric, r.spec.threshold, r.spec.match_mode): r
        for r in runs
        if r.ok
    }
    rows: list[dict[str, float | str]] = []
    for sample_id in {r.spec.sample_id for r in runs}:
        for threshold in (0, 1):
            for match_mode in ("none", "v", "j", "v_j"):
                a = idx.get((sample_id, "alice", "synthetic", "hamming", threshold, match_mode))
                t = idx.get((sample_id, "tcrnet", "synthetic", "hamming", threshold, match_mode))
                if a is None or t is None:
                    continue
                x = _pad_vector(a.component_sizes)
                y = _pad_vector(t.component_sizes)
                if np.std(x) == 0.0 or np.std(y) == 0.0:
                    corr = 0.0
                else:
                    corr = spearmanr(x, y).statistic
                    if corr is None or math.isnan(float(corr)):
                        corr = 0.0
                union = a.hit_sequences | t.hit_sequences
                inter = a.hit_sequences & t.hit_sequences
                rows.append(
                    {
                        "sample_id": sample_id,
                        "threshold": threshold,
                        "match_mode": match_mode,
                        "spearman_component_sizes": float(corr),
                        "jaccard_target_hits": (len(inter) / len(union)) if union else 1.0,
                        "alice_hits": len(a.hit_sequences),
                        "tcrnet_hits": len(t.hit_sequences),
                    }
                )
    return pd.DataFrame(rows)


def _run_specs(
    *,
    samples: dict[str, LocusRepertoire],
    refs: dict[str, pd.DataFrame],
    controls: dict[str, LocusRepertoire],
    manager: ControlManager,
    synthetic_n: int,
    specs: list[RunSpec],
    n_jobs: int,
) -> list[RunResult]:
    return [
        _execute_run(
            spec,
            rep=samples[spec.sample_id],
            ref_df=refs[spec.sample_id],
            controls=controls,
            synthetic_n=synthetic_n,
            manager=manager,
            n_jobs=n_jobs,
        )
        for spec in specs
    ]


@skip_benchmarks
@pytest.mark.benchmark
def test_alice_tcrnet_synthetic_hamming_concordance(capsys) -> None:
    t_start = time.perf_counter()
    max_clonotypes = _env_int("MIRPY_BENCH_FAST_MAX_CLONOTYPES", 300)
    synthetic_n = _env_int("MIRPY_BENCH_FAST_SYNTHETIC_N", 100_000)
    real_control_limit = _env_int("MIRPY_BENCH_REAL_CONTROL_LIMIT", 50_000)
    n_jobs = _env_int("MIRPY_BENCH_N_JOBS", 4)
    match_modes = ["none", "v", "j", "v_j"]

    samples, refs, missing, controls, cmv_source, manager = _build_context(
        max_clonotypes=max_clonotypes,
        synthetic_n=synthetic_n,
        real_control_limit=real_control_limit,
    )
    specs = []
    for sample_id in samples:
        for threshold in (0, 1):
            for match_mode in match_modes:
                specs.append(RunSpec(sample_id, "alice", "synthetic", "hamming", threshold, match_mode))
                specs.append(RunSpec(sample_id, "tcrnet", "synthetic", "hamming", threshold, match_mode))

    runs = _run_specs(
        samples=samples,
        refs=refs,
        controls=controls,
        manager=manager,
        synthetic_n=synthetic_n,
        specs=specs,
        n_jobs=n_jobs,
    )
    elapsed_total = time.perf_counter() - t_start
    concordance = _compute_hamming_concordance(runs)
    benchmark_log_line(
        f"alice_tcrnet_synth_hamming elapsed_total_s={elapsed_total:.3f} max_clonotypes={max_clonotypes} synthetic_n={synthetic_n}"
    )
    with capsys.disabled():
        _print_summary(
            title="ALICE vs TCRNET synthetic-control hamming concordance",
            elapsed_total=elapsed_total,
            runs=runs,
            missing_targets=missing,
            cmv_source=cmv_source,
            extra_df=concordance,
        )
    assert all(r.ok for r in runs)
    assert elapsed_total < _env_float("MIRPY_BENCH_ALICE_SYNTH_MAX_SECONDS", 60.0)


@skip_benchmarks
@pytest.mark.benchmark
def test_tcrnet_synthetic_levenshtein_matrix(capsys) -> None:
    t_start = time.perf_counter()
    max_clonotypes = _env_int("MIRPY_BENCH_FAST_MAX_CLONOTYPES", 300)
    synthetic_n = _env_int("MIRPY_BENCH_FAST_SYNTHETIC_N", 100_000)
    real_control_limit = _env_int("MIRPY_BENCH_REAL_CONTROL_LIMIT", 50_000)
    n_jobs = _env_int("MIRPY_BENCH_N_JOBS", 4)
    match_modes = ["none", "v", "j", "v_j"]

    samples, refs, missing, controls, cmv_source, manager = _build_context(
        max_clonotypes=max_clonotypes,
        synthetic_n=synthetic_n,
        real_control_limit=real_control_limit,
    )
    specs = [
        RunSpec(sample_id, "tcrnet", "synthetic", "levenshtein", threshold, match_mode)
        for sample_id in samples
        for threshold in (0, 1)
        for match_mode in match_modes
    ]
    runs = _run_specs(
        samples=samples,
        refs=refs,
        controls=controls,
        manager=manager,
        synthetic_n=synthetic_n,
        specs=specs,
        n_jobs=n_jobs,
    )
    elapsed_total = time.perf_counter() - t_start
    benchmark_log_line(
        f"tcrnet_synth_levenshtein elapsed_total_s={elapsed_total:.3f} max_clonotypes={max_clonotypes} synthetic_n={synthetic_n}"
    )
    with capsys.disabled():
        _print_summary(
            title="TCRNET synthetic-control levenshtein matrix",
            elapsed_total=elapsed_total,
            runs=runs,
            missing_targets=missing,
            cmv_source=cmv_source,
        )
    assert all(r.ok for r in runs)
    assert elapsed_total < _env_float("MIRPY_BENCH_TCRNET_SYNTH_MAX_SECONDS", 60.0)


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_tcrnet_real_hamming_matrix(capsys) -> None:
    t_start = time.perf_counter()
    max_clonotypes = _env_int("MIRPY_BENCH_REAL_MAX_CLONOTYPES", 100)
    synthetic_n = _env_int("MIRPY_BENCH_REAL_SYNTHETIC_N", 100_000)
    real_control_limit = _env_int("MIRPY_BENCH_REAL_CONTROL_LIMIT", 50_000)
    n_jobs = _env_int("MIRPY_BENCH_N_JOBS", 4)
    match_modes = ["none", "v", "j", "v_j"]

    samples, refs, missing, controls, cmv_source, manager = _build_context(
        max_clonotypes=max_clonotypes,
        synthetic_n=synthetic_n,
        real_control_limit=real_control_limit,
    )
    specs = [
        RunSpec(sample_id, "tcrnet", "real", "hamming", threshold, match_mode)
        for sample_id in samples
        for threshold in (0, 1)
        for match_mode in match_modes
    ]
    runs = _run_specs(
        samples=samples,
        refs=refs,
        controls=controls,
        manager=manager,
        synthetic_n=synthetic_n,
        specs=specs,
        n_jobs=n_jobs,
    )
    elapsed_total = time.perf_counter() - t_start
    benchmark_log_line(
        f"tcrnet_real_hamming elapsed_total_s={elapsed_total:.3f} max_clonotypes={max_clonotypes} real_control_limit={real_control_limit}"
    )
    with capsys.disabled():
        _print_summary(
            title="TCRNET real-control hamming matrix",
            elapsed_total=elapsed_total,
            runs=runs,
            missing_targets=missing,
            cmv_source=cmv_source,
        )
    assert all(r.ok for r in runs)
    assert elapsed_total < _env_float("MIRPY_BENCH_TCRNET_REAL_MAX_SECONDS", 300.0)


@skip_benchmarks
@pytest.mark.benchmark
@pytest.mark.slow_benchmark
def test_tcrnet_real_levenshtein_matrix(capsys) -> None:
    t_start = time.perf_counter()
    max_clonotypes = _env_int("MIRPY_BENCH_REAL_MAX_CLONOTYPES", 100)
    synthetic_n = _env_int("MIRPY_BENCH_REAL_SYNTHETIC_N", 100_000)
    real_control_limit = _env_int("MIRPY_BENCH_REAL_CONTROL_LIMIT", 50_000)
    n_jobs = _env_int("MIRPY_BENCH_N_JOBS", 4)
    match_modes = ["none", "v", "j", "v_j"]

    samples, refs, missing, controls, cmv_source, manager = _build_context(
        max_clonotypes=max_clonotypes,
        synthetic_n=synthetic_n,
        real_control_limit=real_control_limit,
    )
    specs = [
        RunSpec(sample_id, "tcrnet", "real", "levenshtein", threshold, match_mode)
        for sample_id in samples
        for threshold in (0, 1)
        for match_mode in match_modes
    ]
    runs = _run_specs(
        samples=samples,
        refs=refs,
        controls=controls,
        manager=manager,
        synthetic_n=synthetic_n,
        specs=specs,
        n_jobs=n_jobs,
    )
    elapsed_total = time.perf_counter() - t_start
    benchmark_log_line(
        f"tcrnet_real_levenshtein elapsed_total_s={elapsed_total:.3f} max_clonotypes={max_clonotypes} real_control_limit={real_control_limit}"
    )
    with capsys.disabled():
        _print_summary(
            title="TCRNET real-control levenshtein matrix",
            elapsed_total=elapsed_total,
            runs=runs,
            missing_targets=missing,
            cmv_source=cmv_source,
        )
    assert all(r.ok for r in runs)
    assert elapsed_total < _env_float("MIRPY_BENCH_TCRNET_REAL_MAX_SECONDS", 300.0)
