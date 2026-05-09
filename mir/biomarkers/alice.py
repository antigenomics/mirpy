"""ALICE neighborhood enrichment using OLGA generation probabilities.

ALICE computes neighborhood counts in a repertoire (self-background) and
compares observed counts to the expectation from sequence generation
probability (Pgen):

- Neighborhood fold enrichment: ``n / (N * pgen)``
- P-value: ``P(X >= n)`` where ``X ~ Poisson(N * pgen)``

For V/J-constrained analyses, Pgen can be conditioned by dividing by the
estimated OLGA gene usage probability from a synthetic control repertoire:

- ``pgen / (P(v) + 1e-6)``
- ``pgen / (P(j) + 1e-6)``
- ``pgen / (P(v,j) + 1e-6)``

Synthetic control gene-usage estimates are cached in-process and loaded via
:class:`mir.common.control.ControlManager`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import ceil
import math
import typing as t
from dataclasses import dataclass

import pandas as pd
from scipy.stats import poisson

from mir.basic.pgen import OlgaModel, get_olga_gene_usage_probabilities
from mir.common.alleles import allele_to_major
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.biomarkers._shared import MatchMode, iter_loci, match_flags, normalize_match_mode
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats_by_locus

PgenMode = t.Literal["exact", "1mm"]
_PGEN_PARALLEL_MIN_UNIQUE = 256
_PVALUE_PARALLEL_MIN_CLONOTYPES = 256
_OLGA_MODEL_CACHE: dict[tuple[str, str, int | None, type], OlgaModel] = {}


@dataclass(frozen=True)
class AliceParams:
    """ALICE parameter bundle."""

    threshold: int = 1
    match_mode: MatchMode = "none"
    pgen_mode: PgenMode = "exact"

    def validate(self) -> None:
        if self.threshold not in {0, 1}:
            raise ValueError("threshold must be 0 or 1 for ALICE")
        if self.match_mode not in {"none", "v", "j", "vj"}:
            raise ValueError("match_mode must be one of: none, v, j, vj")
        if self.pgen_mode not in {"exact", "1mm"}:
            raise ValueError("pgen_mode must be 'exact' or '1mm'")


@dataclass(frozen=True)
class AliceResult:
    """In-memory ALICE output."""

    table: pd.DataFrame
    params: AliceParams


def _get_cached_olga_model(*, locus: str, species: str, random_seed: int | None) -> OlgaModel:
    key = (locus, species, random_seed, OlgaModel)
    model = _OLGA_MODEL_CACHE.get(key)
    if model is None:
        model = OlgaModel(locus=locus, species=species, seed=random_seed)
        _OLGA_MODEL_CACHE[key] = model
    return model


def _compute_pgen_raw_by_junction_aa(
    clonotypes: list[Clonotype],
    *,
    locus: str,
    species: str,
    random_seed: int | None,
    pgen_mode: PgenMode,
    n_jobs: int,
) -> dict[str, float]:
    if not clonotypes:
        return {}

    unique_aas = list(dict.fromkeys(c.junction_aa for c in clonotypes))
    model = _get_cached_olga_model(locus=locus, species=species, random_seed=random_seed)
    mismatch_budget = 1 if pgen_mode == "1mm" else 0
    if hasattr(model, "compute_pgen_junction_aa_bulk"):
        pgens = model.compute_pgen_junction_aa_bulk(
            unique_aas,
            max_mismatches=mismatch_budget,
            # Benchmarks show threaded Pgen is typically slower than serial
            # due per-worker OLGA model construction overhead.
            n_jobs=1,
        )
    elif mismatch_budget == 1:
        pgens = [float(model.compute_pgen_junction_aa_1mm(seq)) for seq in unique_aas]
    else:
        pgens = [float(model.compute_pgen_junction_aa(seq)) for seq in unique_aas]
    return {junction_aa: float(pgen) for junction_aa, pgen in zip(unique_aas, pgens)}


def _fold_enrichment(n: int, N: int, pgen: float) -> float:
    denom = float(N) * float(pgen)
    if denom <= 0.0:
        return math.inf if n > 0 else 0.0
    return float(n) / denom


def _poisson_pvalue(n: int, N: int, pgen: float) -> float:
    expected = float(N) * float(pgen)
    if expected <= 0.0:
        return 0.0 if n > 0 else 1.0
    return float(poisson.sf(n - 1, expected))


def _compute_alice_metrics_batch(
    clonotypes: list[Clonotype],
    *,
    locus_stats: dict[str, dict[str, int]],
    pgen_raw_by_aa: dict[str, float],
    match_mode: MatchMode,
    probs: dict[str, dict[t.Any, float]] | None,
    gene_usage_epsilon: float,
) -> list[tuple[int, int, float, float, float, float, float]]:
    out: list[tuple[int, int, float, float, float, float, float]] = []
    for clonotype in clonotypes:
        sid = clonotype.sequence_id
        stat = locus_stats.get(sid, {"neighbor_count": 0, "potential_neighbors": 0})
        n = int(stat["neighbor_count"])
        N = int(stat["potential_neighbors"])
        pgen_raw = float(pgen_raw_by_aa.get(clonotype.junction_aa, 0.0))

        divisor = _gene_usage_divisor(
            clonotype,
            match_mode=match_mode,
            probs=probs,
            epsilon=gene_usage_epsilon,
        )
        pgen = pgen_raw / divisor if divisor > 0 else 0.0

        expected = float(N) * pgen
        p_value = _poisson_pvalue(n, N, pgen)
        fold = _fold_enrichment(n, N, pgen)
        out.append((n, N, pgen_raw, pgen, expected, fold, p_value))
    return out


def _apply_alice_metrics_batch(
    clonotypes: list[Clonotype],
    metrics: list[tuple[int, int, float, float, float, float, float]],
    *,
    metadata_prefix: str,
) -> None:
    for clonotype, (n, N, pgen_raw, pgen, expected, fold, p_value) in zip(clonotypes, metrics):
        clonotype.clone_metadata[f"{metadata_prefix}_n"] = n
        clonotype.clone_metadata[f"{metadata_prefix}_N"] = N
        clonotype.clone_metadata[f"{metadata_prefix}_pgen_raw"] = pgen_raw
        clonotype.clone_metadata[f"{metadata_prefix}_pgen"] = pgen
        clonotype.clone_metadata[f"{metadata_prefix}_expected"] = expected
        clonotype.clone_metadata[f"{metadata_prefix}_fold"] = fold
        clonotype.clone_metadata[f"{metadata_prefix}_p_value"] = p_value


def _gene_usage_divisor(
    clonotype: Clonotype,
    *,
    match_mode: MatchMode,
    probs: dict[str, dict[t.Any, float]] | None,
    epsilon: float,
) -> float:
    if match_mode == "none" or probs is None:
        return 1.0

    v_gene = allele_to_major(clonotype.v_gene or "")
    j_gene = allele_to_major(clonotype.j_gene or "")

    if match_mode == "v":
        return float(probs["v"].get(v_gene, 0.0)) + epsilon
    if match_mode == "j":
        return float(probs["j"].get(j_gene, 0.0)) + epsilon
    return float(probs["vj"].get((v_gene, j_gene), 0.0)) + epsilon


def alice_table(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    metadata_prefix: str = "alice",
    sort: bool = True,
) -> pd.DataFrame:
    """Build an ALICE result table from clonotype metadata."""
    rows: list[dict[str, t.Any]] = []
    for locus, lrep in iter_loci(repertoire).items():
        for clonotype in lrep.clonotypes:
            md = clonotype.clone_metadata
            n = int(md.get(f"{metadata_prefix}_n", 0))
            N = int(md.get(f"{metadata_prefix}_N", 0))
            rows.append(
                {
                    "sequence_id": clonotype.sequence_id,
                    "locus": locus,
                    "junction_aa": clonotype.junction_aa,
                    "v_gene": clonotype.v_gene,
                    "j_gene": clonotype.j_gene,
                    "n_neighbors": n,
                    "N_possible": N,
                    "pgen_raw": float(md.get(f"{metadata_prefix}_pgen_raw", 0.0)),
                    "pgen": float(md.get(f"{metadata_prefix}_pgen", 0.0)),
                    "expected_neighbors": float(md.get(f"{metadata_prefix}_expected", 0.0)),
                    "fold_enrichment": float(md.get(f"{metadata_prefix}_fold", 0.0)),
                    "p_value": float(md.get(f"{metadata_prefix}_p_value", 1.0)),
                }
            )

    table = pd.DataFrame.from_records(rows)
    if sort and not table.empty:
        table = table.sort_values(["p_value", "fold_enrichment"], ascending=[True, False]).reset_index(drop=True)
    return table


def compute_alice(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    species: str = "human",
    threshold: int = 1,
    match_mode: str = "none",
    pgen_mode: PgenMode = "exact",
    metric: t.Literal["hamming"] = "hamming",
    gene_usage_synthetic_n: int = 1_000_000,
    gene_usage_epsilon: float = 1e-6,
    control_manager: ControlManager | None = None,
    control_kwargs: dict | None = None,
    random_seed: int | None = 42,
    metadata_prefix: str = "alice",
    as_table: bool = True,
    n_jobs: int = 4,
) -> AliceResult | LocusRepertoire | SampleRepertoire:
    """Compute ALICE enrichment, write clonotype metadata, and optionally return a table.

    Execution order per locus is two-phase to reduce thread contention:
    1. Compute neighborhood stats (trie search) with ``n_jobs``.
    2. Compute OLGA Pgen values with ``n_jobs``.
    """
    norm_match_mode = normalize_match_mode(match_mode)
    params = AliceParams(
        threshold=threshold,
        match_mode=norm_match_mode,
        pgen_mode=pgen_mode,
    )
    params.validate()

    if metric != "hamming":
        raise ValueError("ALICE currently supports only metric='hamming'")

    match_v, match_j = match_flags(norm_match_mode)
    query_loci = iter_loci(repertoire)

    for locus, qrep in query_loci.items():
        probs = None
        if norm_match_mode != "none":
            probs = get_olga_gene_usage_probabilities(
                species=species,
                locus=locus,
                synthetic_n=gene_usage_synthetic_n,
                control_manager=control_manager,
                control_kwargs=control_kwargs,
            )

        # Run phases sequentially to avoid trie-search and Pgen thread contention.
        self_stats = compute_neighborhood_stats_by_locus(
            qrep,
            background=None,
            metric="hamming",
            threshold=threshold,
            match_v_gene=match_v,
            match_j_gene=match_j,
            add_background_pseudocount=False,
            n_jobs=n_jobs,
        )
        pgen_raw_by_aa = _compute_pgen_raw_by_junction_aa(
            qrep.clonotypes,
            locus=locus,
            species=species,
            random_seed=random_seed,
            pgen_mode=pgen_mode,
            n_jobs=n_jobs,
        )

        locus_stats = self_stats.get(locus, {})
        if n_jobs > 1 and len(qrep.clonotypes) >= _PVALUE_PARALLEL_MIN_CLONOTYPES:
            batch_size = max(1, ceil(len(qrep.clonotypes) / n_jobs))
            batches = [
                qrep.clonotypes[start : start + batch_size]
                for start in range(0, len(qrep.clonotypes), batch_size)
            ]
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(
                        _compute_alice_metrics_batch,
                        batch,
                        locus_stats=locus_stats,
                        pgen_raw_by_aa=pgen_raw_by_aa,
                        match_mode=norm_match_mode,
                        probs=probs,
                        gene_usage_epsilon=gene_usage_epsilon,
                    )
                    for batch in batches
                ]
                for batch, future in zip(batches, futures):
                    _apply_alice_metrics_batch(
                        batch,
                        future.result(),
                        metadata_prefix=metadata_prefix,
                    )
        else:
            _apply_alice_metrics_batch(
                qrep.clonotypes,
                _compute_alice_metrics_batch(
                    qrep.clonotypes,
                    locus_stats=locus_stats,
                    pgen_raw_by_aa=pgen_raw_by_aa,
                    match_mode=norm_match_mode,
                    probs=probs,
                    gene_usage_epsilon=gene_usage_epsilon,
                ),
                metadata_prefix=metadata_prefix,
            )

    if as_table:
        return AliceResult(table=alice_table(repertoire, metadata_prefix=metadata_prefix), params=params)
    return repertoire


def add_alice_metadata(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    species: str = "human",
    threshold: int = 1,
    match_mode: str = "none",
    pgen_mode: PgenMode = "exact",
    metric: t.Literal["hamming"] = "hamming",
    gene_usage_synthetic_n: int = 1_000_000,
    gene_usage_epsilon: float = 1e-6,
    control_manager: ControlManager | None = None,
    control_kwargs: dict | None = None,
    random_seed: int | None = 42,
    metadata_prefix: str = "alice",
    n_jobs: int = 4,
) -> LocusRepertoire | SampleRepertoire:
    """Compute ALICE stats and write them into clonotype metadata in-place."""
    return t.cast(
        LocusRepertoire | SampleRepertoire,
        compute_alice(
            repertoire,
            species=species,
            threshold=threshold,
            match_mode=match_mode,
            pgen_mode=pgen_mode,
            metric=metric,
            gene_usage_synthetic_n=gene_usage_synthetic_n,
            gene_usage_epsilon=gene_usage_epsilon,
            control_manager=control_manager,
            control_kwargs=control_kwargs,
            random_seed=random_seed,
            metadata_prefix=metadata_prefix,
            as_table=False,
            n_jobs=n_jobs,
        ),
    )
