"""Reusable workflow helpers for VDJBet analyses.

This module extracts common notebook logic into importable functions so
analyses can be reproduced in scripts, tests, and notebooks with less
duplication.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mir.basic.gene_usage import GeneUsage
from mir.basic.pgen import OlgaModel, PgenGeneUsageAdjustment
from mir.common.control import ControlManager
from mir.common.filter import filter_functional
from mir.common.parser import ClonotypeTableParser
from mir.common.repertoire import LocusRepertoire, Repertoire, infer_locus
from mir.comparative.overlap import compute_overlaps, count_overlap, make_query_index
from mir.comparative.vdjbet import PgenBinPool, VDJBetOverlapAnalysis
from mir.utils.stats import bh_fdr


_WORKER_REF_KEYS: frozenset[tuple[str, str, str]] | None = None
_WORKER_MOCK_KEYS: list[frozenset[tuple[str, str, str]]] | None = None


def _init_score_worker(
    ref_keys: frozenset[tuple[str, str, str]],
    mock_key_sets: list[frozenset[tuple[str, str, str]]],
) -> None:
    """Initialize process worker state for per-sample VDJBet scoring."""
    global _WORKER_REF_KEYS, _WORKER_MOCK_KEYS
    _WORKER_REF_KEYS = ref_keys
    _WORKER_MOCK_KEYS = mock_key_sets


def _score_one_sample_row(sample: dict[str, Any]) -> dict[str, Any]:
    """Worker-safe sample scoring using precomputed reference/mock key sets."""
    if _WORKER_REF_KEYS is None or _WORKER_MOCK_KEYS is None:
        raise RuntimeError("VDJBet workflow score worker not initialized")

    qi = make_query_index(sample["repertoire"], match_v=True, match_j=True)
    n_total = len(qi)
    dc_total = sum(qi.values())

    real = count_overlap(_WORKER_REF_KEYS, qi, allow_1mm=False)
    mock_res = compute_overlaps(_WORKER_MOCK_KEYS, qi, allow_1mm=False, n_jobs=1)

    mn = np.array([r.n for r in mock_res], dtype=float)
    mdc = np.array([r.dc for r in mock_res], dtype=float)
    mdc_log2 = np.log2(mdc + 1.0)
    real_dc_log2 = math.log2(real.dc + 1.0)

    mn_mean = float(np.mean(mn))
    mn_sd = float(np.std(mn, ddof=1)) if len(mn) > 1 else 0.0
    mdc_mean = float(np.mean(mdc_log2))
    mdc_sd = float(np.std(mdc_log2, ddof=1)) if len(mdc_log2) > 1 else 0.0

    z_n = (real.n - mn_mean) / mn_sd if mn_sd > 0 else 0.0
    z_dc = (real_dc_log2 - mdc_mean) / mdc_sd if mdc_sd > 0 else 0.0

    p_n_emp = (np.sum(mn >= real.n) + 1.0) / (len(mn) + 1.0)
    p_dc_emp = (np.sum(mdc_log2 >= real_dc_log2) + 1.0) / (len(mdc_log2) + 1.0)

    return {
        "donor": sample["donor"],
        "day": sample["day"],
        "replica": sample["replica"],
        "sample_label": f"{sample['donor']} {sample['replica']}",
        "n_total": n_total,
        "dc_total": dc_total,
        "matched_n_real": float(real.n),
        "matched_dc_real": float(real.dc),
        "matched_n_fraction": float(real.n) / max(n_total, 1),
        "matched_dc_fraction": float(real.dc) / max(dc_total, 1),
        "matched_n_mock_mean": mn_mean,
        "matched_n_mock_sd": mn_sd,
        "matched_n_z": z_n,
        "matched_n_p_emp": p_n_emp,
        "matched_n_cohen_d": (real.n - mn_mean) / mn_sd if mn_sd > 0 else 0.0,
        "matched_dc_log2_real": real_dc_log2,
        "matched_dc_log2_mock_mean": mdc_mean,
        "matched_dc_log2_mock_sd": mdc_sd,
        "matched_dc_log2_z": z_dc,
        "matched_dc_log2_p_emp": p_dc_emp,
        "matched_dc_log2_cohen_d": (real_dc_log2 - mdc_mean) / mdc_sd if mdc_sd > 0 else 0.0,
        "mock_n": list(mn),
        "mock_dc_log2": list(mdc_log2),
    }


@dataclass
class UsageAdjustmentResult:
    """YF vs OLGA usage comparison outputs and adjustment object."""

    olga_model: OlgaModel
    olga_usage: GeneUsage
    v_cmp: dict
    vj_cmp: dict
    v_df: pd.DataFrame
    vj_df: pd.DataFrame
    pgen_adj_olga: PgenGeneUsageAdjustment


@dataclass
class RealControlAnalysisResult:
    """Artifacts produced when building the real-control null analysis."""

    control_df: pd.DataFrame
    control_usage: GeneUsage
    pgen_adj_real: PgenGeneUsageAdjustment
    pool: PgenBinPool
    analysis: VDJBetOverlapAnalysis
    elapsed_s: float


def parse_yfv_sample_filename(path: Path) -> tuple[str, int, str]:
    """Parse YFV sample file names like ``S2_pre0_F1.airr.tsv.gz``."""
    stem = path.name
    if stem.endswith(".airr.tsv.gz"):
        stem = stem[: -len(".airr.tsv.gz")]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected sample filename format: {path.name}")
    donor = parts[0]
    day_token = parts[1].lower()
    replica = parts[2]
    day = -1 if day_token.startswith("pre") else int(day_token)
    return donor, day, replica


def load_yfv_trb_samples(yfv_dir: str | Path) -> tuple[list[dict[str, Any]], GeneUsage]:
    """Load AIRR YFV samples and return TRB-only sample records + usage."""
    parser = ClonotypeTableParser()
    yfv_path = Path(yfv_dir)
    airr_files = sorted(yfv_path.glob("*.airr.tsv.gz"))
    if not airr_files:
        raise FileNotFoundError(f"No AIRR files found in {yfv_path}")

    samples: list[dict[str, Any]] = []
    for fp in airr_files:
        donor, day, replica = parse_yfv_sample_filename(fp)
        clonotypes = parser.parse(str(fp))
        for clonotype in clonotypes:
            if not clonotype.locus:
                clonotype.locus = infer_locus(clonotype.j_gene or clonotype.v_gene or "")

        trb_clonotypes = [c for c in clonotypes if c.locus == "TRB"]
        trb_rep = LocusRepertoire(clonotypes=trb_clonotypes, locus="TRB", repertoire_id=fp.stem)
        trb_rep = filter_functional(trb_rep)
        if trb_rep.clonotype_count == 0:
            continue

        samples.append(
            {
                "donor": donor,
                "day": day,
                "replica": replica,
                "sample_id": fp.stem,
                "repertoire": trb_rep,
            }
        )

    samples = sorted(samples, key=lambda x: (x["donor"], x["replica"], x["day"]))
    yfv_usage = GeneUsage.from_list([s["repertoire"] for s in samples])
    return samples, yfv_usage


def compute_olga_usage_adjustment(
    yfv_usage: GeneUsage,
    *,
    seed: int,
    olga_usage_n: int,
    n_jobs: int = 1,
    count_mode: str,
    pseudocount: float,
) -> UsageAdjustmentResult:
    """Build OLGA usage cache, comparison tables, and OLGA adjustment."""
    olga_model = OlgaModel(locus="TRB", seed=seed)
    control_mgr = ControlManager()
    olga_control_df = control_mgr.ensure_and_load_control_df(
        "synthetic",
        "human",
        "TRB",
        n=int(olga_usage_n),
        n_jobs=max(1, int(n_jobs)),
        seed=int(seed),
        progress=False,
    )
    olga_usage = GeneUsage.from_dataframe(olga_control_df, locus="TRB")

    v_cmp = yfv_usage.usage_comparison(
        olga_usage,
        "TRB",
        scope="v",
        count=count_mode,
        pseudocount=pseudocount,
    )
    vj_cmp = yfv_usage.usage_comparison(
        olga_usage,
        "TRB",
        scope="vj",
        count=count_mode,
        pseudocount=pseudocount,
    )

    v_df = pd.DataFrame(
        [
            {
                "v_gene": k,
                "p_yf": vals["p_self"],
                "p_olga": vals["p_reference"],
                "factor_v": vals["factor"],
            }
            for k, vals in v_cmp.items()
        ]
    )
    v_df["log2_factor_v"] = np.log2(np.clip(v_df["factor_v"].values, 1e-300, None))

    vj_df = pd.DataFrame(
        [
            {
                "v_gene": k[0],
                "j_gene": k[1],
                "p_yf": vals["p_self"],
                "p_olga": vals["p_reference"],
                "factor_vj": vals["factor"],
            }
            for k, vals in vj_cmp.items()
        ]
    )
    vj_df["log2_factor_vj"] = np.log2(np.clip(vj_df["factor_vj"].values, 1e-300, None))

    pgen_adj_olga = PgenGeneUsageAdjustment(
        yfv_usage,
        cache_size=olga_usage_n,
        seed=seed,
        olga_n_jobs=n_jobs,
        count=count_mode,
        pseudocount=pseudocount,
        reference=olga_usage,
    )

    return UsageAdjustmentResult(
        olga_model=olga_model,
        olga_usage=olga_usage,
        v_cmp=v_cmp,
        vj_cmp=vj_cmp,
        v_df=v_df,
        vj_df=vj_df,
        pgen_adj_olga=pgen_adj_olga,
    )


def build_real_control_analysis(
    reference: Repertoire,
    yfv_usage: GeneUsage,
    *,
    seed: int,
    count_mode: str,
    pseudocount: float,
    pool_size: int,
    n_mocks: int,
    n_jobs: int,
    control_manager: ControlManager | None = None,
) -> RealControlAnalysisResult:
    """Construct real-control adjustment, pool, and VDJBet analysis object."""
    t0 = pd.Timestamp.now()
    control_mgr = control_manager or ControlManager()
    real_control_df = control_mgr.ensure_and_load_control_df("real", "human", "TRB")
    real_control_usage = GeneUsage.from_dataframe(real_control_df, locus="TRB")

    pgen_adj_real = PgenGeneUsageAdjustment(
        yfv_usage,
        seed=seed,
        count=count_mode,
        pseudocount=pseudocount,
        reference=real_control_usage,
    )

    pool = PgenBinPool.from_control(
        locus="TRB",
        control_type="real",
        species="human",
        n=pool_size,
        n_jobs=n_jobs,
        seed=seed,
        pgen_adjustment=pgen_adj_real,
    )
    analysis = VDJBetOverlapAnalysis(
        reference,
        pool=pool,
        n_mocks=n_mocks,
        n_jobs=n_jobs,
        seed=seed,
    )
    elapsed_s = float((pd.Timestamp.now() - t0).total_seconds())

    return RealControlAnalysisResult(
        control_df=real_control_df,
        control_usage=real_control_usage,
        pgen_adj_real=pgen_adj_real,
        pool=pool,
        analysis=analysis,
        elapsed_s=elapsed_s,
    )


def compute_bin_alignment_diagnostics(
    analysis: VDJBetOverlapAnalysis,
) -> dict[str, np.ndarray | list[int] | float]:
    """Return LLW-vs-mock histogram alignment diagnostics for plotting/reporting."""
    ref_bins = analysis.get_reference_bin_sample()
    mock_bins = analysis.get_mock_bin_samples()

    ref_counts = Counter(ref_bins)
    all_bins = sorted(ref_counts.keys())
    if not all_bins:
        raise RuntimeError("No reference bins available for diagnostics.")

    mock_mat = []
    for mb in mock_bins:
        mc = Counter(mb)
        mock_mat.append([mc.get(b, 0) for b in all_bins])
    mock_mat_np = np.asarray(mock_mat, dtype=float)

    ref_vec = np.asarray([ref_counts.get(b, 0) for b in all_bins], dtype=float)
    mock_mean = mock_mat_np.mean(axis=0)
    mock_q10 = np.quantile(mock_mat_np, 0.10, axis=0)
    mock_q90 = np.quantile(mock_mat_np, 0.90, axis=0)

    ref_prob = ref_vec / max(float(ref_vec.sum()), 1.0)
    mock_probs = mock_mat_np / np.clip(mock_mat_np.sum(axis=1, keepdims=True), 1.0, None)
    max_abs_diff = np.max(np.abs(mock_probs - ref_prob[None, :]), axis=1)
    rmsd = np.sqrt(np.mean((mock_probs - ref_prob[None, :]) ** 2, axis=1))

    return {
        "all_bins": all_bins,
        "ref_vec": ref_vec,
        "mock_mat": mock_mat_np,
        "mock_mean": mock_mean,
        "mock_q10": mock_q10,
        "mock_q90": mock_q90,
        "max_abs_diff": max_abs_diff,
        "rmsd": rmsd,
    }


def score_samples_dataframe(
    analysis_obj: VDJBetOverlapAnalysis,
    samples_list: list[dict[str, Any]],
    *,
    progress_every: int = 10,
    sample_n_jobs: int = 1,
) -> pd.DataFrame:
    """Score every sample and return a sorted DataFrame matching notebook schema."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ref_keys = analysis_obj._get_reference_keys_for_match(match_v=True, match_j=True)
        mock_key_sets = analysis_obj._get_mock_key_sets_for_match(match_v=True, match_j=True)

    rows_local: list[dict[str, Any]] = []
    if sample_n_jobs <= 1:
        _init_score_worker(ref_keys, mock_key_sets)
        for i, s in enumerate(samples_list, start=1):
            rows_local.append(_score_one_sample_row(s))
            if progress_every and i % progress_every == 0:
                print(f"Processed {i}/{len(samples_list)} samples")
    else:
        with ProcessPoolExecutor(
            max_workers=sample_n_jobs,
            initializer=_init_score_worker,
            initargs=(ref_keys, mock_key_sets),
        ) as pool:
            for i, row in enumerate(pool.map(_score_one_sample_row, samples_list, chunksize=1), start=1):
                rows_local.append(row)
                if progress_every and i % progress_every == 0:
                    print(f"Processed {i}/{len(samples_list)} samples")

    out = pd.DataFrame(rows_local).sort_values(["donor", "replica", "day"]).reset_index(drop=True)
    out["matched_n_p_adj"] = bh_fdr(out["matched_n_p_emp"].values)
    out["matched_dc_log2_p_adj"] = bh_fdr(out["matched_dc_log2_p_emp"].values)
    return out


def build_synthetic_comparison(
    reference: Repertoire,
    samples: list[dict[str, Any]],
    *,
    pgen_adj_olga: PgenGeneUsageAdjustment,
    pool_size: int,
    n_mocks: int,
    n_jobs: int,
    seed: int,
    df_res_real: pd.DataFrame,
) -> tuple[PgenBinPool, VDJBetOverlapAnalysis, pd.DataFrame, float, pd.DataFrame]:
    """Build synthetic null, score all samples, and compute scale-factor X."""
    pool_synth = PgenBinPool(
        "TRB",
        n=pool_size,
        n_jobs=n_jobs,
        seed=seed,
        pgen_adjustment=pgen_adj_olga,
    )
    analysis_synth = VDJBetOverlapAnalysis(
        reference,
        pool=pool_synth,
        n_mocks=n_mocks,
        n_jobs=n_jobs,
        seed=seed,
    )

    df_res_synth = score_samples_dataframe(analysis_synth, samples)
    x_scale = float(
        df_res_real["matched_n_mock_mean"].mean()
        / max(df_res_synth["matched_n_mock_mean"].mean(), 1e-12)
    )

    df_res_synth_scaled = df_res_synth.copy()
    df_res_synth_scaled["matched_n_mock_mean"] = df_res_synth_scaled["matched_n_mock_mean"] * x_scale
    df_res_synth_scaled["matched_n_mock_sd"] = df_res_synth_scaled["matched_n_mock_sd"] * x_scale
    df_res_synth_scaled["mock_n"] = df_res_synth_scaled["mock_n"].apply(
        lambda xs: [float(x) * x_scale for x in xs]
    )

    return pool_synth, analysis_synth, df_res_synth, x_scale, df_res_synth_scaled
