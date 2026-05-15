"""ALICE-like neighborhood enrichment using OLGA generation probabilities.

ALICE computes neighborhood counts in a repertoire (self-background) and
compares observed counts to the expectation from sequence generation
probability (Pgen):

- Neighborhood fold enrichment: ``n / (N * pgen)``
- P-value: ``P(X >= n)`` where ``X ~ Poisson(N * pgen)``

``match_mode`` controls which sequences are eligible neighbors:
``"v"`` restricts to the same V gene, ``"j"`` to the same J gene,
``"vj"`` to both.  Raw OLGA Pgen is used directly without synthetic-control
gene-usage conditioning.

This module is a highly customized MIR implementation inspired by ideas
described in the ALICE paper, not a literal line-by-line reimplementation of
the original code.

Reference
---------
Pogorelyy MV, Minervina AA, Touzel MP, et al. Detecting T cell receptors
involved in immune responses from single repertoire snapshots. PLoS Biol.
2019;17(6):e3000314. doi:10.1371/journal.pbio.3000314. PMID:31194732.
PubMed: https://pubmed.ncbi.nlm.nih.gov/31194732/
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from math import ceil
import math
import multiprocessing
import os
import typing as t

_MP_CTX = multiprocessing.get_context("spawn")
from dataclasses import dataclass

import polars as pl
from scipy.stats import nbinom, poisson

from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.biomarkers._shared import (
    MatchMode,
    apply_bh_qvalues_to_metadata,
    iter_loci,
    match_flags,
    normalize_match_mode,
)
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats_by_locus

PgenMode = t.Literal["exact", "1mm"]
_PGEN_PARALLEL_MIN_UNIQUE = 256
_PVALUE_PARALLEL_MIN_CLONOTYPES = 256
_OLGA_MODEL_CACHE: dict[tuple[str, str, int | None, type], OlgaModel] = {}


AlicePValueMode = t.Literal["poisson", "negative-binomial"]


@dataclass(frozen=True)
class AliceParams:
    """ALICE parameter bundle."""

    match_mode: MatchMode = "none"
    pgen_mode: PgenMode = "exact"
    pvalue_mode: AlicePValueMode = "poisson"
    pseudocount: float = 0.0

    def validate(self) -> None:
        if self.match_mode not in {"none", "v", "j", "vj"}:
            raise ValueError("match_mode must be one of: none, v, j, vj")
        if self.pgen_mode not in {"exact", "1mm"}:
            raise ValueError("pgen_mode must be 'exact' or '1mm'")
        if self.pvalue_mode not in {"poisson", "negative-binomial"}:
            raise ValueError("pvalue_mode must be 'poisson' or 'negative-binomial'")
        if self.pseudocount < 0:
            raise ValueError("pseudocount must be >= 0")


@dataclass(frozen=True)
class AliceResult:
    """In-memory ALICE output."""

    table: pl.DataFrame
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
    pgen_jobs = n_jobs if (n_jobs > 1 and len(unique_aas) >= _PGEN_PARALLEL_MIN_UNIQUE) else 1
    pgens = model.compute_pgen_junction_aa_bulk(
        unique_aas,
        max_mismatches=mismatch_budget,
        n_jobs=pgen_jobs,
    )
    return {aa: float(p) for aa, p in zip(unique_aas, pgens)}


def _fold_enrichment(n: int, N: int, pgen: float) -> float:
    denom = float(N) * float(pgen)
    if denom <= 0.0:
        return math.inf if n > 0 else 0.0
    return float(n) / denom


def _poisson_pvalue(n: float, N: float, pgen: float) -> float:
    expected = float(N) * float(pgen)
    if expected <= 0.0:
        return 0.0 if n > 0 else 1.0
    return float(poisson.sf(n - 1, expected))


def _negbinom_pvalue(n: float, N: float, pgen: float, dispersion: float = 1.0) -> float:
    """Negative-Binomial p-value: P(X >= n) where X ~ NB(mu=N*pgen, r=dispersion).

    As ``dispersion`` → ∞ the distribution converges to Poisson.
    Default dispersion=1 gives geometric (maximum overdispersion for a given mean).
    """
    mu = float(N) * float(pgen)
    if mu <= 0.0:
        return 0.0 if n > 0 else 1.0
    r = float(dispersion)
    p_success = r / (r + mu)
    return float(nbinom.sf(n - 1, r, p_success))


def _compute_alice_metrics_batch(
    clonotypes: list[Clonotype],
    *,
    locus_stats: dict[str, dict[str, int]],
    pgen_raw_by_aa: dict[str, float],
    pvalue_mode: AlicePValueMode = "poisson",
    pseudocount: float = 0.0,
) -> list[tuple[int, int, float, float, float, float, float]]:
    out: list[tuple[int, int, float, float, float, float, float]] = []
    for clonotype in clonotypes:
        sid = clonotype.sequence_id
        stat = locus_stats.get(sid, {"neighbor_count": 0, "potential_neighbors": 0})
        n = int(stat["neighbor_count"])
        N = int(stat["potential_neighbors"])
        pgen = float(pgen_raw_by_aa.get(clonotype.junction_aa, 0.0))

        n_eff = float(n) + pseudocount
        N_eff = float(N) + pseudocount
        expected = N_eff * pgen
        if pvalue_mode == "negative-binomial":
            p_value = _negbinom_pvalue(n_eff, N_eff, pgen)
        else:
            p_value = _poisson_pvalue(n_eff, N_eff, pgen)
        fold = _fold_enrichment(n, N, pgen)
        out.append((n, N, pgen, pgen, expected, fold, p_value))
    return out


def _compute_alice_metrics_batch_from_args(
    args: tuple[
        list[Clonotype],
        dict[str, dict[str, int]],
        dict[str, float],
        AlicePValueMode,
        float,
    ],
) -> list[tuple[int, int, float, float, float, float, float]]:
    """Pickle-friendly wrapper for process-pool batch execution."""
    clonotypes, locus_stats, pgen_raw_by_aa, pvalue_mode, pseudocount = args
    return _compute_alice_metrics_batch(
        clonotypes,
        locus_stats=locus_stats,
        pgen_raw_by_aa=pgen_raw_by_aa,
        pvalue_mode=pvalue_mode,
        pseudocount=pseudocount,
    )


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



_ALICE_TABLE_SCHEMA: dict[str, type] = {
    "sequence_id": pl.Utf8,
    "locus": pl.Utf8,
    "junction_aa": pl.Utf8,
    "v_gene": pl.Utf8,
    "j_gene": pl.Utf8,
    "n_neighbors": pl.Int64,
    "N_possible": pl.Int64,
    "pgen_raw": pl.Float64,
    "pgen": pl.Float64,
    "expected_neighbors": pl.Float64,
    "fold_enrichment": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
}


def alice_table(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    metadata_prefix: str = "alice",
    sort: bool = True,
) -> pl.DataFrame:
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
                    "q_value": float(md.get(f"{metadata_prefix}_q_value", 1.0)),
                }
            )

    if not rows:
        return pl.DataFrame(schema=_ALICE_TABLE_SCHEMA)
    table = pl.from_dicts(rows)
    if sort:
        table = table.sort(["p_value", "fold_enrichment"], descending=[False, True])
    return table


def compute_alice(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    species: str = "human",
    match_mode: str = "none",
    pgen_mode: PgenMode = "exact",
    metric: t.Literal["hamming"] = "hamming",
    random_seed: int | None = None,
    metadata_prefix: str = "alice",
    as_table: bool = True,
    pseudocount: float = 0.0,
    pvalue_mode: AlicePValueMode = "poisson",
    n_jobs: int = 4,
) -> AliceResult | LocusRepertoire | SampleRepertoire:
    """Compute ALICE enrichment, write clonotype metadata, and optionally return a table.

    Execution order per locus is two-phase:
    1. Compute neighborhood stats (trie search) with ``n_jobs``.
    2. Compute OLGA Pgen values with ``n_jobs``.

    Raw OLGA Pgen is used directly; ``match_mode`` controls which sequences
    are eligible neighbors (same V, same J, or both) but does not trigger
    any synthetic-control gene-usage conditioning.
    """
    norm_match_mode = normalize_match_mode(match_mode)
    params = AliceParams(
        match_mode=norm_match_mode,
        pgen_mode=pgen_mode,
        pvalue_mode=pvalue_mode,
        pseudocount=pseudocount,
    )
    params.validate()

    if metric != "hamming":
        raise ValueError("ALICE currently supports only metric='hamming'")

    match_v, match_j = match_flags(norm_match_mode)
    # ALICE always counts Hamming-1 neighborhoods (CDR3s differing by ≤1 AA).
    # pgen_mode only controls how Pgen is computed: "exact" uses OLGA on the
    # CDR3 directly; "1mm" sums Pgen over the CDR3 and all 1-mismatch variants.
    neighborhood_threshold = 1
    query_loci = iter_loci(repertoire)

    for locus, qrep in query_loci.items():
        # Run phases sequentially to avoid trie-search and Pgen thread contention.
        self_stats = compute_neighborhood_stats_by_locus(
            qrep,
            background=None,
            metric="hamming",
            threshold=neighborhood_threshold,
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
            batch_args = [
                (batch, locus_stats, pgen_raw_by_aa, pvalue_mode, pseudocount)
                for batch in batches
            ]
            executor_mode = os.getenv("MIRPY_ALICE_PVALUE_EXECUTOR", "process").strip().lower()
            if executor_mode == "thread":
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    metric_chunks = list(executor.map(_compute_alice_metrics_batch_from_args, batch_args))
            else:
                with ProcessPoolExecutor(max_workers=n_jobs, mp_context=_MP_CTX) as executor:
                    metric_chunks = list(executor.map(_compute_alice_metrics_batch_from_args, batch_args))

            for batch, metrics in zip(batches, metric_chunks):
                _apply_alice_metrics_batch(batch, metrics, metadata_prefix=metadata_prefix)
        else:
            _apply_alice_metrics_batch(
                qrep.clonotypes,
                _compute_alice_metrics_batch(
                    qrep.clonotypes,
                    locus_stats=locus_stats,
                    pgen_raw_by_aa=pgen_raw_by_aa,
                    pvalue_mode=pvalue_mode,
                    pseudocount=pseudocount,
                ),
                metadata_prefix=metadata_prefix,
            )

    apply_bh_qvalues_to_metadata(repertoire, metadata_prefix=metadata_prefix)

    if as_table:
        return AliceResult(table=alice_table(repertoire, metadata_prefix=metadata_prefix), params=params)
    return repertoire


def add_alice_metadata(
    repertoire: LocusRepertoire | SampleRepertoire,
    *,
    species: str = "human",
    match_mode: str = "none",
    pgen_mode: PgenMode = "exact",
    pvalue_mode: AlicePValueMode = "poisson",
    pseudocount: float = 0.0,
    metric: t.Literal["hamming"] = "hamming",
    random_seed: int | None = None,
    metadata_prefix: str = "alice",
    n_jobs: int = 4,
) -> LocusRepertoire | SampleRepertoire:
    """Compute ALICE stats and write them into clonotype metadata in-place."""
    return t.cast(
        LocusRepertoire | SampleRepertoire,
        compute_alice(
            repertoire,
            species=species,
            match_mode=match_mode,
            pgen_mode=pgen_mode,
            pvalue_mode=pvalue_mode,
            pseudocount=pseudocount,
            metric=metric,
            random_seed=random_seed,
            metadata_prefix=metadata_prefix,
            as_table=False,
            n_jobs=n_jobs,
        ),
    )
