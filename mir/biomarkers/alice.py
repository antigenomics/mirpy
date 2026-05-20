"""ALICE-like neighborhood enrichment using OLGA generation probabilities.

ALICE computes neighborhood counts in a repertoire (self-background) and
compares observed counts to the expectation from sequence generation
probability (Pgen):

- Neighborhood fold enrichment: ``n / (N × pgen_1mm)``
- P-value: ``P(X >= n)`` where ``X ~ Poisson(N × pgen_1mm)``

``match_mode`` controls which sequences are eligible neighbors:
``"v"`` restricts to the same V gene, ``"j"`` to the same J gene,
``"vj"`` (default) to both.  When ``match_mode`` is not ``"none"``, the
background size *N* and Pgen are both conditioned on the V/J gene:
``N_adj = P_ctrl(gene) × N_total`` and ``pgen_adj = pgen / P_ctrl(gene)``,
where P_ctrl is derived analytically from the OLGA model.  These two
adjustments cancel in the expected count λ = N_total × pgen (same as
``match_mode="none"``), but the *observed* k counts only V/J-matching
neighbours, making the test more specific.

Differences from the original ALICE paper
------------------------------------------
The original paper (Pogorelyy et al. *PLoS Biol.* 2019) uses:

* A **100-million** sequence synthetic MC pool to estimate ``pgen_1mm``.
  This implementation uses a **10-million** pool by default
  (``mc_n_pool=10_000_000``) and falls back to OLGA analytical 1mm Pgen for
  sequences with fewer than ``mc_min_count=2`` pool matches, so rare sequences
  use the same λ scale as pool-covered ones.
* **V+J gene matching** (``match_mode="vj"``) as the default neighbor
  restriction.  This implementation now defaults to ``match_mode="vj"``
  to match the paper's V+J restriction.

Relationship to TCRNET
-----------------------
ALICE (Poisson, ``λ = N × pgen_1mm``) and TCRNET (Binomial, ``p = m/M``)
converge when the MC pool is large.  TCRNET is the more general formulation:

* TCRNET uses **any** MC control (real repertoire or synthetic pool) and makes
  no OLGA Pgen calls at all.  When a real control is used it naturally captures
  V/J gene usage bias.
* ALICE uses the synthetic pool only as a Pgen estimator, falling back to OLGA
  for sparse sequences.

To reproduce original ALICE behavior using TCRNET (V+J matching, 100M pool,
selection-corrected)::

    compute_tcrnet(
        rep,
        control=McPgenPool.build_synthetic(100_000_000, locus="TRB").as_locus_repertoire(),
        match_mode="vj",
        pvalue_mode="binomial",
        q_factor=3.0,   # thymic-selection correction; estimate from real vs OLGA pgen ratio
    )

This module is a highly customized MIR implementation inspired by ideas
described in the ALICE paper, not a literal line-by-line reimplementation of
the original code.

Reference
---------
Pogorelyy MV, Minervina AA, Shugay M, et al. Detecting T cell receptors
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

from mir.basic.gene_usage import get_gene_usage_from_olga_model
from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.biomarkers._shared import (
    MatchMode,
    apply_bh_qvalues_to_metadata,
    iter_loci,
    lookup_gene_frac,
    match_flags,
    normalize_match_mode,
)
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats_by_locus

PgenMode = t.Literal["exact", "1mm", "mc"]
_PGEN_PARALLEL_MIN_UNIQUE = 256
_PVALUE_PARALLEL_MIN_CLONOTYPES = 256
_OLGA_MODEL_CACHE: dict[tuple[str, str, int | None, type], OlgaModel] = {}
_OLGA_MODEL_CACHE_MAX = max(0, int(os.getenv("MIRPY_ALICE_MODEL_CACHE_SIZE", "4")))

# Cross-call Pgen result cache: (locus, species, junction_aa, pgen_mode) → float.
# Avoids recomputing the same CDR3 across samples in multi-sample workflows.
_PGEN_RESULT_CACHE: dict[tuple[str, str, str, str], float] = {}
_PGEN_RESULT_CACHE_MAX = max(0, int(os.getenv("MIRPY_ALICE_PGEN_RESULT_CACHE_SIZE", "1000000")))


AlicePValueMode = t.Literal["poisson", "negative-binomial"]


@dataclass(frozen=True)
class AliceParams:
    """ALICE parameter bundle.

    When ``pgen_mode="mc"``, a Monte-Carlo pool of *mc_n_pool* productive
    sequences is generated once and cached.  Pgen is estimated by counting
    exact + inner-1mm matches in that pool.  Sequences with fewer than
    *mc_min_count* matches fall back to OLGA analytical 1mm Pgen so that
    rare sequences are scored on the same scale as the rest.
    """

    match_mode: MatchMode = "vj"
    pgen_mode: PgenMode = "exact"
    pvalue_mode: AlicePValueMode = "poisson"
    pseudocount: float = 0.0
    min_neighbors: int = 2
    q_factor: float = 1.0
    mc_n_pool: int = 10_000_000
    mc_seed: int = 42
    mc_min_count: int = 2

    def validate(self) -> None:
        if self.match_mode not in {"none", "v", "j", "vj"}:
            raise ValueError("match_mode must be one of: none, v, j, vj")
        if self.pgen_mode not in {"exact", "1mm", "mc"}:
            raise ValueError("pgen_mode must be 'exact', '1mm', or 'mc'")
        if self.pvalue_mode not in {"poisson", "negative-binomial"}:
            raise ValueError("pvalue_mode must be 'poisson' or 'negative-binomial'")
        if self.pseudocount < 0:
            raise ValueError("pseudocount must be >= 0")
        if self.min_neighbors < 0:
            raise ValueError("min_neighbors must be >= 0")
        if self.q_factor <= 0:
            raise ValueError("q_factor must be > 0")
        if self.mc_n_pool < 100_000:
            raise ValueError("mc_n_pool must be >= 100,000")
        if self.mc_min_count < 0:
            raise ValueError("mc_min_count must be >= 0")


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
        if _OLGA_MODEL_CACHE_MAX > 0:
            while len(_OLGA_MODEL_CACHE) >= _OLGA_MODEL_CACHE_MAX:
                _OLGA_MODEL_CACHE.pop(next(iter(_OLGA_MODEL_CACHE)))
            _OLGA_MODEL_CACHE[key] = model
    return model


def clear_olga_model_cache() -> None:
    """Clear cached OLGA models used by ALICE helpers."""
    _OLGA_MODEL_CACHE.clear()


def clear_pgen_result_cache() -> None:
    """Clear the cross-call Pgen result cache."""
    _PGEN_RESULT_CACHE.clear()


def _bulk_pgen(
    aas: list[str],
    *,
    locus: str,
    species: str,
    random_seed: int | None,
    pgen_mode: PgenMode,
    n_jobs: int,
) -> dict[str, float]:
    """Compute pgen for *aas*, consulting and populating the cross-call cache."""
    result: dict[str, float] = {}
    uncached: list[str] = []
    for aa in aas:
        v = _PGEN_RESULT_CACHE.get((locus, species, aa, pgen_mode))
        if v is not None:
            result[aa] = v
        else:
            uncached.append(aa)
    if uncached:
        model = _get_cached_olga_model(locus=locus, species=species, random_seed=random_seed)
        mismatch_budget = 1 if pgen_mode == "1mm" else 0
        pgen_jobs = n_jobs if (n_jobs > 1 and len(uncached) >= _PGEN_PARALLEL_MIN_UNIQUE) else 1
        pgens = model.compute_pgen_junction_aa_bulk(
            uncached, max_mismatches=mismatch_budget, n_jobs=pgen_jobs,
        )
        for aa, p in zip(uncached, pgens):
            fp = float(p)
            result[aa] = fp
            if _PGEN_RESULT_CACHE_MAX > 0 and len(_PGEN_RESULT_CACHE) < _PGEN_RESULT_CACHE_MAX:
                _PGEN_RESULT_CACHE[(locus, species, aa, pgen_mode)] = fp
    return result


def _compute_pgen_raw_by_junction_aa(
    clonotypes: list[Clonotype],
    *,
    locus: str,
    species: str,
    random_seed: int | None,
    pgen_mode: PgenMode,
    n_jobs: int,
    locus_stats: dict[str, dict[str, int]] | None = None,
    min_neighbors: int = 0,
    mc_n_pool: int = 10_000_000,
    mc_seed: int = 42,
    mc_min_count: int = 2,
) -> dict[str, float]:
    """Return ``{junction_aa: pgen}`` for clonotypes eligible for ALICE scoring.

    **Filtering by min_neighbors**: When ``locus_stats`` is provided and
    ``min_neighbors >= 1``, pgen is only computed for sequences whose
    ``neighbor_count`` (self included) is at least ``min_neighbors``.  Others
    are absent from the returned dict, causing ``_compute_alice_metrics_batch``
    to assign ``p_value = 1.0`` without calling OLGA.

    **MC mode**: Queries the synthetic pool for 1mm neighbourhood pgen.
    Sequences with fewer than *mc_min_count* pool matches fall back to OLGA
    analytical 1mm Pgen so that sparse sequences use the same λ scale.

    **1mm mode**: Calls OLGA 1mm Pgen directly for all eligible sequences.

    **Cross-call cache**: Computed pgen values are stored in the module-level
    ``_PGEN_RESULT_CACHE`` keyed by ``(locus, species, junction_aa, pgen_mode)``
    so that repeated calls across multiple samples in the same session avoid
    redundant OLGA computation.

    Note:
        ``neighbor_count`` includes the sequence itself (self is always a
        Hamming-0 neighbour).  ``min_neighbors=2`` therefore requires the sequence
        plus at least one additional Hamming-1 neighbour in the repertoire.
    """
    if not clonotypes:
        return {}

    # ── Step 1: filter by min_neighbors ──────────────────────────────────────
    if locus_stats is not None and min_neighbors >= 1:
        eligible_aas: set[str] = {
            c.junction_aa
            for c in clonotypes
            if locus_stats.get(c.sequence_id, {}).get("neighbor_count", 0) >= min_neighbors
        }
        unique_aas = [
            aa for aa in dict.fromkeys(c.junction_aa for c in clonotypes)
            if aa in eligible_aas
        ]
    else:
        unique_aas = list(dict.fromkeys(c.junction_aa for c in clonotypes))

    if not unique_aas:
        return {}

    if pgen_mode == "exact":
        return _bulk_pgen(
            unique_aas,
            locus=locus, species=species, random_seed=random_seed,
            pgen_mode="exact", n_jobs=n_jobs,
        )

    # ── MC mode ──────────────────────────────────────────────────────────────
    # Estimate 1mm neighbourhood pgen via match counting in a synthetic pool.
    # Sequences with < mc_min_count pool matches fall back to OLGA 1mm Pgen
    # so that sparse sequences use the same λ scale as pool-covered ones.
    if pgen_mode == "mc":
        from mir.basic.pgen import get_or_build_mc_pool
        from mir.basic.pgen import _PGEN_1MM_SKIP_ENDS as _skip_ends
        pool = get_or_build_mc_pool(
            locus=locus,
            species=species,
            n=mc_n_pool,
            seed=mc_seed,
            skip_ends=_skip_ends,
            n_jobs=n_jobs,
        )
        mc_pgens = pool.pgen_1mm_bulk(unique_aas, n_jobs=n_jobs)

        needs_olga: list[str] = []
        mc_map: dict[str, float] = {}
        for aa, mp in zip(unique_aas, mc_pgens):
            if int(round(mp * pool.n_total)) >= mc_min_count:
                mc_map[aa] = mp
            else:
                needs_olga.append(aa)

        if needs_olga:
            mc_map.update(_bulk_pgen(
                needs_olga,
                locus=locus, species=species, random_seed=random_seed,
                pgen_mode="1mm", n_jobs=n_jobs,
            ))

        return mc_map

    # ── 1mm mode ──────────────────────────────────────────────────────────────
    return _bulk_pgen(
        unique_aas,
        locus=locus, species=species, random_seed=random_seed,
        pgen_mode="1mm", n_jobs=n_jobs,
    )


def warm_pgen_cache(
    clonotypes: list[Clonotype],
    *,
    locus: str = "TRB",
    species: str = "human",
    pgen_mode: PgenMode = "1mm",
    n_jobs: int = 4,
    random_seed: int | None = None,
) -> int:
    """Pre-compute and cache Pgen for all unique CDR3s in one parallel batch.

    Call this before a multi-sample ALICE loop with the concatenated clonotypes
    from ALL samples.  All OLGA computation runs once in a single
    ``pool.map`` call, maximising worker utilisation and eliminating
    redundant per-sample recomputation for shared CDR3 sequences.

    Args:
        clonotypes: Flat list of clonotypes from all samples to be analysed.
        locus: OLGA locus (e.g. ``"TRB"``).
        species: ``"human"`` or ``"mouse"``.
        pgen_mode: ``"exact"`` or ``"1mm"``.
        n_jobs: Worker processes for OLGA computation.
        random_seed: OLGA model seed.

    Returns:
        Number of newly computed Pgen values stored in the cache.
    """
    if not clonotypes:
        return 0
    unique_aas = list(dict.fromkeys(c.junction_aa for c in clonotypes))
    to_compute = [
        aa for aa in unique_aas
        if _PGEN_RESULT_CACHE.get((locus, species, aa, pgen_mode)) is None
    ]
    if not to_compute:
        return 0
    model = _get_cached_olga_model(locus=locus, species=species, random_seed=random_seed)
    mismatch_budget = 1 if pgen_mode == "1mm" else 0
    pgen_jobs = n_jobs if (n_jobs > 1 and len(to_compute) >= _PGEN_PARALLEL_MIN_UNIQUE) else 1
    new_pgens = model.compute_pgen_junction_aa_bulk(
        to_compute,
        max_mismatches=mismatch_budget,
        n_jobs=pgen_jobs,
    )
    stored = 0
    for aa, p in zip(to_compute, new_pgens):
        if _PGEN_RESULT_CACHE_MAX > 0 and len(_PGEN_RESULT_CACHE) < _PGEN_RESULT_CACHE_MAX:
            _PGEN_RESULT_CACHE[(locus, species, aa, pgen_mode)] = float(p)
            stored += 1
    return stored


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
    q_factor: float = 1.0,
    match_mode: str = "none",
    gene_usage_fracs: "dict | None" = None,
    n_background_total: int = 0,
) -> list[tuple[int, int, float, float, float, float, float]]:
    out: list[tuple[int, int, float, float, float, float, float]] = []
    use_gene_scaling = (
        match_mode != "none"
        and gene_usage_fracs is not None
        and n_background_total > 0
    )
    for clonotype in clonotypes:
        sid = clonotype.sequence_id
        stat = locus_stats.get(sid, {"neighbor_count": 0, "potential_neighbors": 0})
        n = int(stat["neighbor_count"])
        N = int(stat["potential_neighbors"])
        pgen_opt = pgen_raw_by_aa.get(clonotype.junction_aa)
        if pgen_opt is None:
            # Pgen was not computed (sequence below min_neighbors); assign p_value=1.0
            # so these sequences are never called ALICE hits after BH correction.
            out.append((n, N, 0.0, 0.0, 0.0, 0.0, 1.0))
            continue
        pgen = float(pgen_opt)
        if use_gene_scaling:
            # Scale N and pgen by OLGA V/J gene usage so λ = N_total × pgen
            # (gene-usage-conditioned N and pgen cancel in λ, but k is V/J-filtered).
            p_gene = lookup_gene_frac(
                match_mode, clonotype.v_gene or "", clonotype.j_gene or "",
                gene_usage_fracs,
            )
            N = int(round(p_gene * n_background_total))
            pgen_adj = pgen * q_factor / p_gene
        else:
            pgen_adj = pgen * q_factor

        n_eff = float(n) + pseudocount
        N_eff = float(N) + pseudocount
        expected = N_eff * pgen_adj
        if pvalue_mode == "negative-binomial":
            p_value = _negbinom_pvalue(n_eff, N_eff, pgen_adj)
        else:
            p_value = _poisson_pvalue(n_eff, N_eff, pgen_adj)
        fold = _fold_enrichment(n, N, pgen_adj)
        out.append((n, N, pgen, pgen_adj, expected, fold, p_value))
    return out


def _compute_alice_metrics_batch_from_args(
    args: tuple,
) -> list[tuple[int, int, float, float, float, float, float]]:
    """Pickle-friendly wrapper for process-pool batch execution."""
    (clonotypes, locus_stats, pgen_raw_by_aa, pvalue_mode, pseudocount, q_factor,
     match_mode, gene_usage_fracs, n_background_total) = args
    return _compute_alice_metrics_batch(
        clonotypes,
        locus_stats=locus_stats,
        pgen_raw_by_aa=pgen_raw_by_aa,
        pvalue_mode=pvalue_mode,
        pseudocount=pseudocount,
        q_factor=q_factor,
        match_mode=match_mode,
        gene_usage_fracs=gene_usage_fracs,
        n_background_total=n_background_total,
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


def alice_hit_clusters(
    hits_df: "pd.DataFrame",
    full_df: "pd.DataFrame | None" = None,
    *,
    non_enriched_neighbors: bool = False,
) -> "pd.DataFrame":
    """Cluster ALICE hits into connected components via V-gene-restricted 1mm CDR3 edges.

    Nodes are ALICE hits (rows from an ALICE result table filtered by q_value and
    n_neighbors).  An edge exists between two nodes when their CDR3 amino acid
    sequences differ by exactly one position **and** they share the same V-gene
    family (e.g. ``TRBV9*01`` → ``TRBV9``).  V-gene restriction prevents spurious
    transitive merges across unrelated gene families.

    Connected components are labelled with an integer ``cluster_id`` column.
    Singletons (no 1mm same-V neighbour among other hits) each receive a unique id.

    Args:
        hits_df: DataFrame of ALICE hits — must contain ``junction_aa`` and
            ``v_gene`` columns.
        full_df: Full ALICE result table for the same sample(s), required when
            ``non_enriched_neighbors=True``.  May contain non-hit sequences.
        non_enriched_neighbors: When ``True``, every non-enriched sequence in
            ``full_df`` that is 1mm (same V-gene) from any hit is added to that
            hit's cluster.  ``full_df`` must be provided.

    Returns:
        DataFrame equal to ``hits_df`` (plus non-enriched neighbors when
        ``non_enriched_neighbors=True``) with an added integer ``cluster_id``
        column.  Rows from ``hits_df`` carry ``is_hit=True``; added neighbors
        carry ``is_hit=False``.

    Raises:
        ValueError: If ``non_enriched_neighbors=True`` but ``full_df`` is ``None``.
    """
    import pandas as pd

    if non_enriched_neighbors and full_df is None:
        raise ValueError("full_df must be provided when non_enriched_neighbors=True")

    if hits_df.empty:
        result = hits_df.copy()
        if non_enriched_neighbors:
            result = result.assign(is_hit=pd.array([], dtype=bool))
        return result.assign(cluster_id=pd.array([], dtype="int64"))

    def _vfam(v: str) -> str:
        return v.split("*")[0] if v else ""

    has_v = "v_gene" in hits_df.columns

    def _uf(rows: list[tuple[str, str]]) -> list[int]:
        n = len(rows)
        idx_by: dict[tuple[str, str], list[int]] = {}
        for i, (c, v) in enumerate(rows):
            idx_by.setdefault((c, _vfam(v)), []).append(i)

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, (cdr3, v) in enumerate(rows):
            vf = _vfam(v)
            for pos in range(len(cdr3)):
                o = cdr3[pos]
                for aa in "ACDEFGHIKLMNPQRSTVWY":
                    if aa != o:
                        for j in idx_by.get((cdr3[:pos] + aa + cdr3[pos + 1:], vf), []):
                            if j > i:
                                parent[find(i)] = find(j)

        rm: dict[int, int] = {}
        ids = []
        for i in range(n):
            r = find(i)
            if r not in rm:
                rm[r] = len(rm)
            ids.append(rm[r])
        return ids

    if non_enriched_neighbors:
        assert full_df is not None
        if has_v:
            hit_keys: set = set(
                zip(hits_df["junction_aa"], hits_df["v_gene"].map(_vfam))
            )
        else:
            hit_keys = set(hits_df["junction_aa"])

        neighbor_rows = []
        for _, row in full_df.iterrows():
            cdr3 = str(row.get("junction_aa", ""))
            vf = _vfam(str(row.get("v_gene", ""))) if has_v else ""
            key: tuple | str = (cdr3, vf) if has_v else cdr3
            if key in hit_keys:
                continue
            found = False
            for pos in range(len(cdr3)):
                o = cdr3[pos]
                for aa in "ACDEFGHIKLMNPQRSTVWY":
                    if aa != o:
                        nbr = cdr3[:pos] + aa + cdr3[pos + 1:]
                        nbr_key: tuple | str = (nbr, vf) if has_v else nbr
                        if nbr_key in hit_keys:
                            neighbor_rows.append(row.to_dict())
                            found = True
                            break
                if found:
                    break

        if neighbor_rows:
            neighbor_df = pd.DataFrame(neighbor_rows).assign(is_hit=False)
            combined = pd.concat(
                [hits_df.assign(is_hit=True), neighbor_df], ignore_index=True
            )
        else:
            combined = hits_df.assign(is_hit=True)
    else:
        combined = hits_df

    if has_v:
        rows_for_uf = list(zip(combined["junction_aa"], combined["v_gene"]))
    else:
        rows_for_uf = [(c, "") for c in combined["junction_aa"]]

    return combined.assign(cluster_id=_uf(rows_for_uf))


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
    match_mode: str = "vj",
    pgen_mode: PgenMode = "exact",
    metric: t.Literal["hamming"] = "hamming",
    random_seed: int | None = None,
    metadata_prefix: str = "alice",
    as_table: bool = True,
    pseudocount: float = 0.0,
    pvalue_mode: AlicePValueMode = "poisson",
    min_neighbors: int = 2,
    q_factor: float = 1.0,
    mc_n_pool: int = 10_000_000,
    mc_seed: int = 42,
    mc_min_count: int = 2,
    n_jobs: int = 4,
) -> AliceResult | LocusRepertoire | SampleRepertoire:
    """Compute ALICE enrichment, write clonotype metadata, and optionally return a table.

    Execution order per locus is two-phase:
    1. Compute neighborhood stats (trie search) with ``n_jobs``.
    2. Compute Pgen values with ``n_jobs``.

    Args:
        min_neighbors: Minimum ``neighbor_count`` (self + Hamming-1 neighbours)
            a sequence must have before pgen is computed.  The default of 2
            requires self plus at least one additional Hamming-1 neighbour.
            Sequences below this threshold get ``p_value = 1.0`` without any
            OLGA calls.
        pgen_mode: ``"exact"`` uses OLGA analytical Pgen.  ``"1mm"`` sums Pgen
            over the Hamming-1 neighbourhood.  ``"mc"`` estimates the same 1mm
            pgen via a large synthetic pool, falling back to OLGA 1mm Pgen for
            sequences with fewer than *mc_min_count* pool matches.
        mc_n_pool: Synthetic pool size for ``pgen_mode="mc"`` (default 10M).
        mc_seed: OLGA seed for pool generation.
        mc_min_count: Minimum pool-match count for MC Pgen; sequences below
            this threshold fall back to OLGA 1mm Pgen.

    When ``match_mode != "none"``, gene-usage probabilities are read
    analytically from the OLGA model and used to scale *N* and *pgen* so that
    ``λ = N_total × pgen`` (independent of gene restriction).  The observed *k*
    counts only gene-matching neighbours, making the test more specific without
    changing the null expectation.
    """
    norm_match_mode = normalize_match_mode(match_mode)
    params = AliceParams(
        match_mode=norm_match_mode,
        pgen_mode=pgen_mode,
        pvalue_mode=pvalue_mode,
        pseudocount=pseudocount,
        min_neighbors=min_neighbors,
        q_factor=q_factor,
        mc_n_pool=mc_n_pool,
        mc_seed=mc_seed,
        mc_min_count=mc_min_count,
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
        locus_stats = self_stats.get(locus, {})
        pgen_raw_by_aa = _compute_pgen_raw_by_junction_aa(
            qrep.clonotypes,
            locus=locus,
            species=species,
            random_seed=random_seed,
            pgen_mode=pgen_mode,
            n_jobs=n_jobs,
            locus_stats=locus_stats,
            min_neighbors=min_neighbors,
            mc_n_pool=mc_n_pool,
            mc_seed=mc_seed,
            mc_min_count=mc_min_count,
        )

        # Gene-usage conditioning: scale N and pgen by OLGA V/J frequencies so
        # that λ = N_total × pgen regardless of gene restriction.
        gene_usage_fracs: dict | None = None
        n_background_total = 0
        if norm_match_mode != "none":
            olga_model = _get_cached_olga_model(
                locus=locus, species=species, random_seed=random_seed
            )
            gene_usage_fracs = get_gene_usage_from_olga_model(olga_model)
            n_background_total = len(qrep.clonotypes)

        if n_jobs > 1 and len(qrep.clonotypes) >= _PVALUE_PARALLEL_MIN_CLONOTYPES:
            batch_size = max(1, ceil(len(qrep.clonotypes) / n_jobs))
            batches = [
                qrep.clonotypes[start : start + batch_size]
                for start in range(0, len(qrep.clonotypes), batch_size)
            ]
            batch_args = [
                (batch, locus_stats, pgen_raw_by_aa, pvalue_mode, pseudocount, q_factor,
                 norm_match_mode, gene_usage_fracs, n_background_total)
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
                    q_factor=q_factor,
                    match_mode=norm_match_mode,
                    gene_usage_fracs=gene_usage_fracs,
                    n_background_total=n_background_total,
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
    match_mode: str = "vj",
    pgen_mode: PgenMode = "exact",
    pvalue_mode: AlicePValueMode = "poisson",
    pseudocount: float = 0.0,
    metric: t.Literal["hamming"] = "hamming",
    random_seed: int | None = None,
    metadata_prefix: str = "alice",
    min_neighbors: int = 2,
    q_factor: float = 1.0,
    mc_n_pool: int = 10_000_000,
    mc_seed: int = 42,
    mc_min_count: int = 2,
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
            min_neighbors=min_neighbors,
            q_factor=q_factor,
            mc_n_pool=mc_n_pool,
            mc_seed=mc_seed,
            mc_min_count=mc_min_count,
            n_jobs=n_jobs,
        ),
    )
