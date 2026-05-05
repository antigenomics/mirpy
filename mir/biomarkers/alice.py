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

import math
import typing as t
from dataclasses import dataclass

import pandas as pd
from scipy.stats import poisson

from mir.basic.pgen import OlgaModel
from mir.common.alleles import allele_to_major
from mir.common.clonotype import Clonotype
from mir.common.control import ControlManager
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.graph.neighborhood_enrichment import compute_neighborhood_stats_by_locus

MatchMode = t.Literal["none", "v", "j", "vj"]
PgenMode = t.Literal["exact", "1mm"]

_GENE_USAGE_CACHE: dict[tuple[str, str, int], dict[str, dict[t.Any, float]]] = {}


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


def _normalize_match_mode(match_mode: str) -> MatchMode:
    mode = match_mode.strip().lower().replace("_", "")
    if mode == "vj":
        return "vj"
    if mode in {"none", "v", "j"}:
        return t.cast(MatchMode, mode)
    raise ValueError("match_mode must be one of: none, v, j, vj (or v_j)")


def _match_flags(match_mode: MatchMode) -> tuple[bool, bool]:
    return match_mode in {"v", "vj"}, match_mode in {"j", "vj"}


def _iter_loci(
    repertoire: LocusRepertoire | SampleRepertoire,
) -> dict[str, LocusRepertoire]:
    if isinstance(repertoire, SampleRepertoire):
        return dict(repertoire.loci)
    if isinstance(repertoire, LocusRepertoire):
        return {repertoire.locus: repertoire}
    raise TypeError("repertoire must be LocusRepertoire or SampleRepertoire")


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


def _compute_gene_usage_probabilities(
    *,
    control_df: pd.DataFrame,
) -> dict[str, dict[t.Any, float]]:
    """Compute OLGA gene usage probabilities from synthetic controls."""
    required = {"v_gene", "j_gene"}
    missing = required.difference(control_df.columns)
    if missing:
        raise ValueError(f"control_df missing required columns: {sorted(missing)}")

    df = control_df.loc[:, ["v_gene", "j_gene"]].copy()
    df["v_gene"] = df["v_gene"].map(lambda x: allele_to_major(str(x or "")))
    df["j_gene"] = df["j_gene"].map(lambda x: allele_to_major(str(x or "")))
    df = df[(df["v_gene"] != "") & (df["j_gene"] != "")]
    total = len(df)
    if total == 0:
        return {"v": {}, "j": {}, "vj": {}}

    p_v = (df["v_gene"].value_counts(sort=False) / total).to_dict()
    p_j = (df["j_gene"].value_counts(sort=False) / total).to_dict()
    p_vj = (
        df.groupby(["v_gene", "j_gene"], sort=False).size() / total
    ).to_dict()

    return {
        "v": {k: float(v) for k, v in p_v.items()},
        "j": {k: float(v) for k, v in p_j.items()},
        "vj": {k: float(v) for k, v in p_vj.items()},
    }


def _get_olga_gene_usage_probs(
    *,
    species: str,
    locus: str,
    synthetic_n: int,
    control_manager: ControlManager | None,
    control_kwargs: dict | None,
) -> dict[str, dict[t.Any, float]]:
    cache_key = (species.lower().strip(), locus, int(synthetic_n))
    if cache_key in _GENE_USAGE_CACHE:
        return _GENE_USAGE_CACHE[cache_key]

    manager = control_manager or ControlManager()
    kwargs = dict(control_kwargs or {})
    kwargs.setdefault("n", int(synthetic_n))
    kwargs.setdefault("progress", False)
    control_df = manager.ensure_and_load_control_df(
        "synthetic",
        species,
        locus,
        **kwargs,
    )
    probs = _compute_gene_usage_probabilities(control_df=control_df)
    _GENE_USAGE_CACHE[cache_key] = probs
    return probs


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
    for locus, lrep in _iter_loci(repertoire).items():
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
    """Compute ALICE enrichment, write clonotype metadata, and optionally return a table."""
    norm_match_mode = _normalize_match_mode(match_mode)
    params = AliceParams(
        threshold=threshold,
        match_mode=norm_match_mode,
        pgen_mode=pgen_mode,
    )
    params.validate()

    if metric != "hamming":
        raise ValueError("ALICE currently supports only metric='hamming'")

    match_v, match_j = _match_flags(norm_match_mode)
    query_loci = _iter_loci(repertoire)

    for locus, qrep in query_loci.items():
        olga_model = OlgaModel(locus=locus, species=species, seed=random_seed)
        probs = None
        if norm_match_mode != "none":
            probs = _get_olga_gene_usage_probs(
                species=species,
                locus=locus,
                synthetic_n=gene_usage_synthetic_n,
                control_manager=control_manager,
                control_kwargs=control_kwargs,
            )

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
        locus_stats = self_stats.get(locus, {})

        for clonotype in qrep.clonotypes:
            sid = clonotype.sequence_id
            stat = locus_stats.get(sid, {"neighbor_count": 0, "potential_neighbors": 0})
            n = int(stat["neighbor_count"])
            N = int(stat["potential_neighbors"])

            if pgen_mode == "1mm":
                pgen_raw = float(olga_model.compute_pgen_junction_aa_1mm(clonotype.junction_aa))
            else:
                pgen_raw = float(olga_model.compute_pgen_junction_aa(clonotype.junction_aa))

            divisor = _gene_usage_divisor(
                clonotype,
                match_mode=norm_match_mode,
                probs=probs,
                epsilon=gene_usage_epsilon,
            )
            pgen = pgen_raw / divisor if divisor > 0 else 0.0

            expected = float(N) * pgen
            p_value = _poisson_pvalue(n, N, pgen)
            fold = _fold_enrichment(n, N, pgen)

            clonotype.clone_metadata[f"{metadata_prefix}_n"] = n
            clonotype.clone_metadata[f"{metadata_prefix}_N"] = N
            clonotype.clone_metadata[f"{metadata_prefix}_pgen_raw"] = pgen_raw
            clonotype.clone_metadata[f"{metadata_prefix}_pgen"] = pgen
            clonotype.clone_metadata[f"{metadata_prefix}_expected"] = expected
            clonotype.clone_metadata[f"{metadata_prefix}_fold"] = fold
            clonotype.clone_metadata[f"{metadata_prefix}_p_value"] = p_value

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
