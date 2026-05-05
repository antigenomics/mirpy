"""TCRNET-like neighborhood enrichment for clonotype repertoires.

This implementation is metadata-first:
- neighborhood counts and p-values are written directly into clonotype metadata,
- table output is optional and derived from the annotated repertoire.
"""

from __future__ import annotations

import math
import typing as t
from dataclasses import dataclass

import pandas as pd
from scipy.stats import betabinom, binom

from mir.basic.gene_usage import GeneUsage
from mir.biomarkers._shared import MatchMode, iter_loci, match_flags, normalize_match_mode
from mir.common.clonotype import Clonotype
from mir.common.alleles import allele_to_major
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.sampling import resample_to_gene_usage
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats_by_locus

PValueMode = t.Literal["binomial", "beta-binomial"]


@dataclass(frozen=True)
class TcrnetParams:
    """TCRNET parameter bundle."""

    metric: t.Literal["hamming", "levenshtein"] = "hamming"
    threshold: int = 1
    match_mode: MatchMode = "none"
    pvalue_mode: PValueMode = "binomial"

    def validate(self) -> None:
        if self.metric not in {"hamming", "levenshtein"}:
            raise ValueError("metric must be 'hamming' or 'levenshtein'")
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("threshold must be in [0, 1] for TCRNET")
        if self.match_mode not in {"none", "v", "j", "vj"}:
            raise ValueError("match_mode must be one of: none, v, j, vj")
        if self.pvalue_mode not in {"binomial", "beta-binomial"}:
            raise ValueError("pvalue_mode must be 'binomial' or 'beta-binomial'")


@dataclass(frozen=True)
class TcrnetResult:
    """In-memory TCRNET output."""

    table: pd.DataFrame
    params: TcrnetParams

def _empty_locus(locus: str) -> LocusRepertoire:
    return LocusRepertoire(clonotypes=[], locus=locus)


def _df_to_locus_repertoire(df: pd.DataFrame, locus: str) -> LocusRepertoire:
    rows: list[Clonotype] = []
    for idx, rec in enumerate(df.to_dict(orient="records")):
        rows.append(
            Clonotype(
                sequence_id=str(idx),
                locus=locus,
                duplicate_count=max(1, int(rec.get("duplicate_count", 1) or 1)),
                junction=str(rec.get("junction", "")),
                junction_aa=str(rec.get("junction_aa", "")),
                v_gene=allele_to_major(str(rec.get("v_gene", ""))),
                j_gene=allele_to_major(str(rec.get("j_gene", ""))),
                _validate=False,
            )
        )
    return LocusRepertoire(clonotypes=rows, locus=locus)


def _resolve_control_loci(
    *,
    target_loci: dict[str, LocusRepertoire],
    control: LocusRepertoire | SampleRepertoire | None,
    control_type: str | None,
    species: str,
    control_manager: ControlManager | None,
    control_kwargs: dict | None,
) -> dict[str, LocusRepertoire]:
    if control is not None:
        control_loci = iter_loci(control)
        out: dict[str, LocusRepertoire] = {}
        for locus in target_loci:
            out[locus] = control_loci.get(locus, _empty_locus(locus))
        return out

    if control_type is None:
        raise ValueError("Provide either control repertoire or control_type")

    mgr = control_manager or ControlManager()
    kwargs = dict(control_kwargs or {})

    out: dict[str, LocusRepertoire] = {}
    for locus in target_loci:
        rec_df = mgr.ensure_and_load_control_df(control_type, species, locus, **kwargs)
        out[locus] = _df_to_locus_repertoire(rec_df, locus=locus)
    return out


def _normalize_control_vj(
    *,
    target_locus: LocusRepertoire,
    control_locus: LocusRepertoire,
    random_seed: int | None,
) -> LocusRepertoire:
    if not control_locus.clonotypes:
        return control_locus

    gu = GeneUsage.from_repertoire(target_locus)
    target_vj = gu.vj_usage(target_locus.locus, count="duplicates")
    if not target_vj:
        return control_locus

    return t.cast(
        LocusRepertoire,
        resample_to_gene_usage(
            control_locus,
            target_vj,
            scope="vj",
            weighted=True,
            random_seed=random_seed,
        ),
    )


def _p_value(n: int, N: int, m: int, M: int, mode: PValueMode) -> float:
    if N <= 0:
        return 1.0
    if M <= 0:
        return 1.0

    if mode == "binomial":
        p = min(1.0, max(0.0, m / M))
        return float(binom.sf(n - 1, N, p))

    alpha = float(m + 1)
    beta = float((M - m) + 1)
    return float(betabinom.sf(n - 1, N, alpha, beta))


def _fold_enrichment(n: int, N: int, m: int, M: int) -> float:
    if N <= 0 or M <= 0:
        return 0.0
    if m <= 0:
        return math.inf if n > 0 else 0.0
    return float((n / N) * (M / m))


def tcrnet_table(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    metadata_prefix: str = "tcrnet",
    sort: bool = True,
) -> pd.DataFrame:
    """Build a TCRNET result table from clonotype metadata."""
    rows: list[dict[str, t.Any]] = []
    for locus, lrep in iter_loci(repertoire).items():
        for clonotype in lrep.clonotypes:
            md = clonotype.clone_metadata
            n = int(md.get(f"{metadata_prefix}_n", 0))
            N = int(md.get(f"{metadata_prefix}_N", 0))
            m = int(md.get(f"{metadata_prefix}_m", 0))
            M = int(md.get(f"{metadata_prefix}_M", 0))
            rows.append(
                {
                    "sequence_id": clonotype.sequence_id,
                    "locus": locus,
                    "junction_aa": clonotype.junction_aa,
                    "v_gene": clonotype.v_gene,
                    "j_gene": clonotype.j_gene,
                    "n_neighbors": n,
                    "N_possible": N,
                    "m_control_neighbors": m,
                    "M_control_possible": M,
                    "sample_density": float(n / N) if N > 0 else 0.0,
                    "control_density": float(m / M) if M > 0 else 0.0,
                    "fold_enrichment": float(md.get(f"{metadata_prefix}_fold", 0.0)),
                    "p_value": float(md.get(f"{metadata_prefix}_p_value", 1.0)),
                }
            )
    table = pd.DataFrame.from_records(rows)
    if sort and not table.empty:
        table = table.sort_values(["p_value", "fold_enrichment"], ascending=[True, False]).reset_index(drop=True)
    return table


def compute_tcrnet(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    control: LocusRepertoire | SampleRepertoire | None = None,
    control_type: str | None = None,
    species: str = "human",
    control_manager: ControlManager | None = None,
    control_kwargs: dict | None = None,
    normalize_control_vj_usage: bool = False,
    metric: t.Literal["hamming", "levenshtein"] = "hamming",
    threshold: int = 1,
    match_mode: str = "none",
    pvalue_mode: PValueMode = "binomial",
    random_seed: int | None = None,
    metadata_prefix: str = "tcrnet",
    as_table: bool = True,
    n_jobs: int = 4,
) -> TcrnetResult | LocusRepertoire | SampleRepertoire:
    """Compute TCRNET-like enrichment, write clonotype metadata, and optionally return a table.

    Either pass an explicit control repertoire (``control=...``) or request a
    managed control via ``control_type`` with :class:`ControlManager`.
    """
    norm_match_mode = normalize_match_mode(match_mode)
    params = TcrnetParams(
        metric=metric,
        threshold=threshold,
        match_mode=norm_match_mode,
        pvalue_mode=pvalue_mode,
    )
    params.validate()
    match_v, match_j = match_flags(norm_match_mode)

    query_loci = iter_loci(repertoire)
    control_loci = _resolve_control_loci(
        target_loci=query_loci,
        control=control,
        control_type=control_type,
        species=species,
        control_manager=control_manager,
        control_kwargs=control_kwargs,
    )
    for locus, qrep in query_loci.items():
        crep = control_loci.get(locus, _empty_locus(locus))
        if normalize_control_vj_usage:
            crep = _normalize_control_vj(
                target_locus=qrep,
                control_locus=crep,
                random_seed=random_seed,
            )

        self_stats = compute_neighborhood_stats_by_locus(
            qrep,
            background=None,
            metric=metric,
            threshold=threshold,
            match_v_gene=match_v,
            match_j_gene=match_j,
            add_background_pseudocount=False,
            n_jobs=n_jobs,
        )
        control_stats = compute_neighborhood_stats_by_locus(
            qrep,
            background=crep,
            metric=metric,
            threshold=threshold,
            match_v_gene=match_v,
            match_j_gene=match_j,
            add_background_pseudocount=False,
            n_jobs=n_jobs,
        )

        locus_self = self_stats.get(locus, {})
        locus_ctrl = control_stats.get(locus, {})

        for clonotype in qrep.clonotypes:
            sid = clonotype.sequence_id
            s_stat = locus_self.get(sid, {"neighbor_count": 0, "potential_neighbors": 0})
            c_stat = locus_ctrl.get(sid, {"neighbor_count": 0, "potential_neighbors": 0})
            n = int(s_stat["neighbor_count"])
            N = int(s_stat["potential_neighbors"])
            # Add pseudocount of 1 to control: virtual clonotype-of-interest inserted into control
            m = int(c_stat["neighbor_count"]) + 1
            M = int(c_stat["potential_neighbors"]) + 1
            p = _p_value(n, N, m, M, pvalue_mode)
            fe = _fold_enrichment(n, N, m, M)
            clonotype.clone_metadata[f"{metadata_prefix}_n"] = n
            clonotype.clone_metadata[f"{metadata_prefix}_N"] = N
            clonotype.clone_metadata[f"{metadata_prefix}_m"] = m
            clonotype.clone_metadata[f"{metadata_prefix}_M"] = M
            clonotype.clone_metadata[f"{metadata_prefix}_sample_density"] = float(n / N) if N > 0 else 0.0
            clonotype.clone_metadata[f"{metadata_prefix}_control_density"] = float(m / M) if M > 0 else 0.0
            clonotype.clone_metadata[f"{metadata_prefix}_fold"] = fe
            clonotype.clone_metadata[f"{metadata_prefix}_p_value"] = p

    if as_table:
        return TcrnetResult(table=tcrnet_table(repertoire, metadata_prefix=metadata_prefix), params=params)
    return repertoire


def add_tcrnet_metadata(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    control: LocusRepertoire | SampleRepertoire | None = None,
    control_type: str | None = None,
    species: str = "human",
    control_manager: ControlManager | None = None,
    control_kwargs: dict | None = None,
    normalize_control_vj_usage: bool = False,
    metric: t.Literal["hamming", "levenshtein"] = "hamming",
    threshold: int = 1,
    match_mode: str = "none",
    pvalue_mode: PValueMode = "binomial",
    random_seed: int | None = None,
    metadata_prefix: str = "tcrnet",
    n_jobs: int = 4,
) -> LocusRepertoire | SampleRepertoire:
    """Compute TCRNET stats and write them into clonotype metadata in-place."""
    return t.cast(
        LocusRepertoire | SampleRepertoire,
        compute_tcrnet(
            repertoire,
            control=control,
            control_type=control_type,
            species=species,
            control_manager=control_manager,
            control_kwargs=control_kwargs,
            normalize_control_vj_usage=normalize_control_vj_usage,
            metric=metric,
            threshold=threshold,
            match_mode=match_mode,
            pvalue_mode=pvalue_mode,
            random_seed=random_seed,
            metadata_prefix=metadata_prefix,
            as_table=False,
            n_jobs=n_jobs,
        )
    )
