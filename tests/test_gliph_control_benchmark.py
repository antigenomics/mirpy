"""Benchmarks for GLIPH control tokenization and rare-token coverage.

These benchmarks are opt-in and run only with RUN_BENCHMARK=1.
Default sizes are intentionally bounded to avoid stall-prone runs.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
import re
import sys
import time
import tracemalloc

if sys.platform != "win32":
    import resource

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mir.biomarkers.gliph import (
    deduplicate_clonotype_rows,
    extract_gliph_artifacts_batch_from_repertoire,
)
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire
from tests.benchmark_helpers import benchmark_log_line
from tests.conftest import skip_benchmarks


GLIPH_PATH = Path(__file__).resolve().parents[1] / "airr_benchmark" / "gliph" / "gliph_trb.tsv.gz"
DEFAULT_FAMILIES = ("v3", "pos3", "u3", "u4", "g4", "g5")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
SEED = 42
TRIM_FIRST = 3
TRIM_LAST = 4
MIN_TOKEN_K = 3
CHUNK_SIZE = int(os.getenv("MIRPY_GLIPH_CHUNK_SIZE", "200000"))


def _control_sizes() -> list[int]:
    raw = os.getenv("MIRPY_GLIPH_CONTROL_SIZES", "10000,100000")
    values: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            value = int(tok)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return sorted(set(values)) if values else [10_000, 100_000]


def _min_raw_len_for_tokenization() -> int:
    return TRIM_FIRST + TRIM_LAST + MIN_TOKEN_K


def _to_mb(bytes_value: int) -> float:
    return float(bytes_value) / (1024.0 * 1024.0)


def _maxrss_mb() -> float:
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        return _to_mb(int(rss))
    return rss / 1024.0


_REAL_CONTROL_CAP = int(os.getenv("MIRPY_BENCH_REAL_CONTROL_N", "2000000"))


def _canonical_control_df() -> pd.DataFrame:
    ctrl_raw_full = ControlManager().ensure_and_load_control_df("real", "human", "TRB")
    if _REAL_CONTROL_CAP < len(ctrl_raw_full):
        ctrl_raw = ctrl_raw_full.sample(n=_REAL_CONTROL_CAP, seed=42).to_pandas()
    else:
        ctrl_raw = ctrl_raw_full.to_pandas()
    del ctrl_raw_full
    df = pd.DataFrame(
        {
            "junction_aa": ctrl_raw["junction_aa"].astype(str).str.strip(),
            "v_gene": ctrl_raw["v_gene"].astype(str).str.strip(),
            "j_gene": ctrl_raw["j_gene"].astype(str).str.strip(),
            "duplicate_count": pd.to_numeric(ctrl_raw.get("duplicate_count", 1), errors="coerce").fillna(1).astype(int),
        }
    )
    del ctrl_raw
    min_len = _min_raw_len_for_tokenization()
    mask = df["junction_aa"].str.len().ge(min_len) & df["junction_aa"].str.match(AA_RE)
    return df.loc[mask].reset_index(drop=True)


def _sample_to_repertoire(control_df: pd.DataFrame, idx: np.ndarray) -> LocusRepertoire:
    sampled = control_df.iloc[idx].copy()
    sampled.insert(0, "sequence_id", np.arange(len(sampled), dtype=np.int64).astype(str))
    pl_df = pl.from_pandas(sampled, include_index=False)
    return LocusRepertoire.from_polars(pl_df, locus="TRB")


def _measure_batch_extraction(
    repertoire: LocusRepertoire,
    *,
    families: tuple[str, ...],
    chunk_size: int,
) -> dict[str, float | int]:
    gc.collect()
    rss_before = _maxrss_mb()

    tracemalloc.start()
    t0 = time.perf_counter()
    artifacts = extract_gliph_artifacts_batch_from_repertoire(
        repertoire,
        list(families),
        count_mode="clonotype",
        build_mappings=False,
        trim_first=TRIM_FIRST,
        trim_last=TRIM_LAST,
        chunk_size=chunk_size,
    )
    elapsed_s = time.perf_counter() - t0
    _current, py_peak_after = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rss_after = _maxrss_mb()

    return {
        "elapsed_s": float(elapsed_s),
        "py_peak_mb": float(_to_mb(int(py_peak_after))),
        "rss_delta_mb": float(max(0.0, rss_after - rss_before)),
        "tokens_total": int(sum(len(art.counts) for art in artifacts.values())),
    }


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
    min_len = _min_raw_len_for_tokenization()
    out = out[out["junction_aa"].str.len() >= min_len].copy()
    out = out[out["junction_aa"].str.match(AA_RE)].copy()
    out = deduplicate_clonotype_rows(out, subset=("reference_id", "v_gene", "junction_aa")).to_pandas()
    out = out.reset_index(drop=True)
    out["row_id"] = out.index.astype(str)
    return out


def _zipf_fit(counts: dict[str, int]) -> dict[str, float]:
    freqs = np.array(sorted((int(v) for v in counts.values() if int(v) > 0), reverse=True), dtype=float)
    if len(freqs) < 5:
        return {"zipf_slope": np.nan, "zipf_r2": np.nan}

    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    x = np.log(ranks)
    y = np.log(freqs)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"zipf_slope": float(slope), "zipf_r2": float(r2)}


def _fit_missing_powerlaw(sizes: np.ndarray, missing_fraction: np.ndarray) -> dict[str, float]:
    mask = (sizes > 0) & (missing_fraction > 0) & np.isfinite(missing_fraction)
    if int(mask.sum()) < 2:
        return {"log_a": np.nan, "b": np.nan, "r2": np.nan}

    x = np.log(sizes[mask].astype(float))
    y = np.log(missing_fraction[mask].astype(float))
    b, log_a = np.polyfit(x, y, 1)
    y_hat = b * x + log_a
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"log_a": float(log_a), "b": float(b), "r2": float(r2)}


def _n_for_coverage(log_a: float, b: float, coverage: float) -> float:
    if not np.isfinite(log_a) or not np.isfinite(b) or b >= 0:
        return np.nan
    miss = 1.0 - coverage
    if miss <= 0:
        return np.nan
    return float(np.exp((np.log(miss) - log_a) / b))


@skip_benchmarks
@pytest.mark.very_slow_benchmark
def test_gliph_control_tokenization() -> None:
    sizes = _control_sizes()
    control_df = _canonical_control_df()

    max_n = max(sizes)
    if max_n > len(control_df):
        pytest.skip(f"Requested max size {max_n} exceeds available control clonotypes {len(control_df)}")

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(control_df), size=max_n, replace=False)

    rows: list[dict[str, object]] = []
    benchmark_log_line(
        "GLIPH control tokenization start: "
        f"sizes={sizes}, families={list(DEFAULT_FAMILIES)}, trim=({TRIM_FIRST},{TRIM_LAST}), chunk_size={CHUNK_SIZE}"
    )

    for n in sizes:
        idx = np.sort(chosen[:n])
        rep = _sample_to_repertoire(control_df, idx)

        single_total = 0.0
        single_peak_py = 0.0
        single_peak_rss = 0.0
        for family in DEFAULT_FAMILIES:
            stats = _measure_batch_extraction(
                rep,
                families=(family,),
                chunk_size=CHUNK_SIZE,
            )
            single_total += float(stats["elapsed_s"])
            single_peak_py = max(single_peak_py, float(stats["py_peak_mb"]))
            single_peak_rss = max(single_peak_rss, float(stats["rss_delta_mb"]))
            rows.append(
                {
                    "size": n,
                    "mode": "single_family_ctrl",
                    "family": family,
                    "elapsed_s": float(stats["elapsed_s"]),
                    "py_peak_mb": float(stats["py_peak_mb"]),
                    "rss_delta_mb": float(stats["rss_delta_mb"]),
                    "tokens_total": int(stats["tokens_total"]),
                }
            )

        batch_ctrl = _measure_batch_extraction(
            rep,
            families=DEFAULT_FAMILIES,
            chunk_size=CHUNK_SIZE,
        )
        rows.append(
            {
                "size": n,
                "mode": "batch_ctrl",
                "family": "all",
                "elapsed_s": float(batch_ctrl["elapsed_s"]),
                "py_peak_mb": float(batch_ctrl["py_peak_mb"]),
                "rss_delta_mb": float(batch_ctrl["rss_delta_mb"]),
                "tokens_total": int(batch_ctrl["tokens_total"]),
                "single_family_total_s": single_total,
                "single_family_peak_py_mb": single_peak_py,
                "single_family_peak_rss_mb": single_peak_rss,
                "speedup_vs_single_total": (
                    single_total / float(batch_ctrl["elapsed_s"])
                    if float(batch_ctrl["elapsed_s"]) > 0
                    else np.nan
                ),
            }
        )

        benchmark_log_line(
            "GLIPH control tokenization "
            f"n={n}: batch_s={batch_ctrl['elapsed_s']:.3f}, single_total_s={single_total:.3f}, "
            f"batch_py_peak_mb={batch_ctrl['py_peak_mb']:.1f}, batch_rss_delta_mb={batch_ctrl['rss_delta_mb']:.1f}"
        )

    result_df = pd.DataFrame(rows)
    ctrl_df = result_df[result_df["mode"] == "batch_ctrl"].copy()

    print("\nGLIPH control tokenization benchmark (per family + batch):")
    print(result_df.sort_values(["size", "mode", "family"]).to_string(index=False))
    print("\nControl-mode batch summary by control size:")
    print(
        ctrl_df[
            [
                "size",
                "elapsed_s",
                "single_family_total_s",
                "speedup_vs_single_total",
                "py_peak_mb",
                "rss_delta_mb",
                "tokens_total",
            ]
        ].to_string(index=False)
    )

    assert not ctrl_df.empty
    assert set(ctrl_df["size"]) == set(sizes)
    assert ctrl_df["tokens_total"].gt(0).all()


@skip_benchmarks
@pytest.mark.very_slow_benchmark
def test_gliph_rare_token_coverage() -> None:
    if not GLIPH_PATH.exists():
        pytest.skip(f"Missing GLIPH dataset: {GLIPH_PATH}")

    sizes = _control_sizes()
    control_df = _canonical_control_df()
    max_n = max(sizes)
    if max_n > len(control_df):
        pytest.skip(f"Requested max size {max_n} exceeds available control clonotypes {len(control_df)}")

    raw = pd.read_csv(GLIPH_PATH, sep="\t")
    study_df = _normalize_gliph_df(raw)

    study_pl = pl.from_pandas(
        study_df[["row_id", "junction_aa", "v_gene", "j_gene", "duplicate_count"]]
        .rename(columns={"row_id": "sequence_id"}),
        include_index=False,
    )
    study_rep = LocusRepertoire.from_polars(study_pl, locus="TRB")
    study_artifacts = extract_gliph_artifacts_batch_from_repertoire(
        study_rep,
        list(DEFAULT_FAMILIES),
        count_mode="clonotype",
        build_mappings=False,
        trim_first=TRIM_FIRST,
        trim_last=TRIM_LAST,
        chunk_size=CHUNK_SIZE,
    )

    rare_sets: dict[str, dict[str, set[str]]] = {}
    zipf_rows: list[dict[str, object]] = []
    for family, art in study_artifacts.items():
        counts = {k: int(v) for k, v in art.clonotype_counts.items()}
        rare_sets[family] = {
            "n1": {tok for tok, c in counts.items() if c == 1},
            "n2": {tok for tok, c in counts.items() if c == 2},
            "n3p": {tok for tok, c in counts.items() if c >= 3},
        }
        fit = _zipf_fit(counts)
        zipf_rows.append(
            {
                "family": family,
                "n_tokens": len(counts),
                "zipf_slope": fit["zipf_slope"],
                "zipf_r2": fit["zipf_r2"],
            }
        )

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(control_df), size=max_n, replace=False)

    rows: list[dict[str, object]] = []
    for n in sizes:
        idx = np.sort(chosen[:n])
        rep = _sample_to_repertoire(control_df, idx)
        ctrl_artifacts = extract_gliph_artifacts_batch_from_repertoire(
            rep,
            list(DEFAULT_FAMILIES),
            count_mode="clonotype",
            build_mappings=False,
            trim_first=TRIM_FIRST,
            trim_last=TRIM_LAST,
            chunk_size=CHUNK_SIZE,
        )

        for family in DEFAULT_FAMILIES:
            ctrl_tokens = set(ctrl_artifacts[family].counts)
            for bucket, target_tokens in rare_sets[family].items():
                if not target_tokens:
                    missing = 0
                    missing_frac = np.nan
                    coverage = np.nan
                else:
                    missing = len(target_tokens - ctrl_tokens)
                    missing_frac = missing / len(target_tokens)
                    coverage = 1.0 - missing_frac

                rows.append(
                    {
                        "size": n,
                        "family": family,
                        "bucket": bucket,
                        "target_tokens": len(target_tokens),
                        "missing_tokens": missing,
                        "missing_fraction": missing_frac,
                        "coverage": coverage,
                    }
                )

        benchmark_log_line(f"GLIPH rare-token coverage computed for n={n}")

    coverage_df = pd.DataFrame(rows)
    zipf_df = pd.DataFrame(zipf_rows).sort_values("family")

    aggregate = (
        coverage_df.groupby(["size", "bucket"], as_index=False)
        .agg(target_tokens=("target_tokens", "sum"), missing_tokens=("missing_tokens", "sum"))
        .assign(
            missing_fraction=lambda df_: np.where(
                df_["target_tokens"] > 0,
                df_["missing_tokens"] / df_["target_tokens"],
                np.nan,
            )
        )
    )

    model_rows: list[dict[str, object]] = []
    for bucket in ("n1", "n2", "n3p"):
        sub = aggregate[aggregate["bucket"] == bucket].sort_values("size")
        fit = _fit_missing_powerlaw(
            sub["size"].to_numpy(dtype=float),
            sub["missing_fraction"].to_numpy(dtype=float),
        )
        model_rows.append(
            {
                "bucket": bucket,
                "fit_b": fit["b"],
                "fit_r2": fit["r2"],
                "n_for_90pct": _n_for_coverage(fit["log_a"], fit["b"], 0.90),
                "n_for_95pct": _n_for_coverage(fit["log_a"], fit["b"], 0.95),
                "n_for_99pct": _n_for_coverage(fit["log_a"], fit["b"], 0.99),
            }
        )
    model_df = pd.DataFrame(model_rows)

    print("\nGLIPH token-frequency Zipf fit by family:")
    print(zipf_df.to_string(index=False))
    print("\nRare-token coverage vs control size:")
    print(coverage_df.sort_values(["size", "family", "bucket"]).to_string(index=False))
    print("\nAggregate missing fraction across families:")
    print(aggregate.to_string(index=False))
    print("\nPower-law fit for missing fraction vs control size:")
    print(model_df.to_string(index=False))

    benchmark_log_line("GLIPH rare-token coverage benchmark complete")

    assert not coverage_df.empty
    assert set(coverage_df["size"]) == set(sizes)
    for family in DEFAULT_FAMILIES:
        fam = coverage_df[coverage_df["family"] == family].copy()
        for bucket in ("n1", "n2", "n3p"):
            sub = fam[fam["bucket"] == bucket].sort_values("size")
            vals = [int(v) for v in sub["missing_tokens"].tolist() if pd.notna(v)]
            if len(vals) >= 2:
                assert all(later <= earlier for earlier, later in zip(vals[:-1], vals[1:]))
