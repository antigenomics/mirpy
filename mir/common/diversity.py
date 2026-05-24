"""Diversity metrics and curve utilities for repertoire analysis.

Implements VDJtools-style summary indices and iNEXT-style rarefaction/coverage
estimators from abundance tables. The rarefaction math uses NumPy/SciPy vector
kernels for fast execution on large abundance vectors.

Summary statistics follow the VDJtools convention (Shugay *et al.* 2015):
richness (observed and Chao1), Shannon entropy, Gini-Simpson index, singleton/
doubleton counts, and expanded/hyperexpanded clone fractions.

Rarefaction and extrapolation curves implement the iNEXT framework (Chao *et al.*
2014; Hsieh *et al.* 2016), which unifies interpolation (m ≤ n) and extrapolation
(m > n) under a single sample-coverage estimator based on the Chao1 formula.

Hill diversity profiles follow Hill (1973): the order-q family
``D_q = (Σ p_i^q)^{1/(1-q)}`` with q=0 (richness), q→1 (exp(Shannon)),
and q=2 (inverse Simpson) as special cases.

References
----------
Shugay M, Bagaev DV, Turchaninova MA, Bolotin DA, Zvyagin IV, Putintseva EV,
Pogorelyy MV, Radko SP, Lebedev YB, Chudakov DM. VDJtools: unifying
post-analysis of T cell receptor repertoires. *PLoS Comput Biol.*
2015;11(11):e1004503. PMID:26606115. https://pubmed.ncbi.nlm.nih.gov/26606115/

Chao A. Nonparametric estimation of the number of classes in a population.
*Scand J Stat.* 1984;11(4):265-270.

Hsieh TC, Ma KH, Chao A. iNEXT: an R package for rarefaction and
extrapolation of species diversity (Hill numbers). *Methods Ecol Evol.*
2016;7(12):1451-1456. doi:10.1111/2041-210X.12613.

Hill MO. Diversity and evenness: a unifying notation and its consequences.
*Ecology.* 1973;54(2):427-432. doi:10.2307/1934352.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np
import polars as pl
from scipy.special import gammaln, ndtri

CountField = Literal["duplicate_count", "umi_count", "barcode_count"]


@dataclass(frozen=True, slots=True)
class DiversitySummary:
    """Compact diversity summary for one repertoire partition."""

    abundance: int
    diversity: int
    singletons: int
    doubletons: int
    expanded: int
    hyperexpanded: int
    chao1: float
    gini_simpson: float
    shannon: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RarefactionResult:
    """One rarefaction/coverage estimate point."""

    m: int
    n: int
    f0: float
    f1: float
    f2: float
    s_obs: float
    s_est: float
    s_sd: float
    s_lwr: float
    s_upr: float
    coverage: float
    coverage_sd: float
    coverage_lwr: float
    coverage_upr: float
    extrapolation: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _as_positive_counts(counts: Sequence[int] | np.ndarray | Iterable[int]) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.int64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    return arr[arr > 0]


def build_abundance_table(counts: Sequence[int] | np.ndarray) -> pl.DataFrame:
    """Convert clonotype counts to a sparse abundance table with columns (k, f_k)."""
    arr = _as_positive_counts(counts)
    if arr.size == 0:
        return pl.DataFrame(schema={"k": pl.Int64, "f_k": pl.Int64})
    k, f_k = np.unique(arr, return_counts=True)
    return pl.DataFrame({"k": k.astype(np.int64), "f_k": f_k.astype(np.int64)})


def summarize_counts(
    counts: Sequence[int] | np.ndarray,
    *,
    expanded_threshold: float = 1e-3,
    hyperexpanded_threshold: float = 1e-2,
) -> DiversitySummary:
    """Compute VDJtools-like repertoire diversity summary metrics."""
    arr = _as_positive_counts(counts)
    abundance = int(arr.sum()) if arr.size else 0
    diversity = int(arr.size)
    if abundance == 0:
        return DiversitySummary(
            abundance=0,
            diversity=0,
            singletons=0,
            doubletons=0,
            expanded=0,
            hyperexpanded=0,
            chao1=0.0,
            gini_simpson=0.0,
            shannon=0.0,
        )

    singletons = int(np.sum(arr == 1))
    doubletons = int(np.sum(arr == 2))
    freqs = arr / float(abundance)
    expanded = int(np.sum(freqs > expanded_threshold))
    hyperexpanded = int(np.sum(freqs > hyperexpanded_threshold))
    chao1 = float(diversity + (singletons * (singletons - 1)) / (2.0 * (doubletons + 1)))
    gini_simpson = float(1.0 - np.sum(freqs * freqs))
    nonzero = freqs[freqs > 0]
    shannon = float(-np.sum(nonzero * np.log(nonzero)))

    return DiversitySummary(
        abundance=abundance,
        diversity=diversity,
        singletons=singletons,
        doubletons=doubletons,
        expanded=expanded,
        hyperexpanded=hyperexpanded,
        chao1=chao1,
        gini_simpson=gini_simpson,
        shannon=shannon,
    )


def hill_curve(
    counts: Sequence[int] | np.ndarray,
    q_values: Sequence[float] | np.ndarray | None = None,
) -> pl.DataFrame:
    """Compute the Hill diversity profile for q values."""
    arr = _as_positive_counts(counts)
    if q_values is None:
        q_values = np.linspace(0.0, 4.0, 41)
    q_arr = np.asarray(list(q_values), dtype=np.float64)
    if arr.size == 0:
        return pl.DataFrame({"q": q_arr, "hill": np.zeros_like(q_arr)})

    total = float(arr.sum())
    p = arr / total
    richness = float(arr.size)
    shannon = float(-np.sum(p * np.log(p)))

    # Vectorised Hill computation: avoid Python loop over q values.
    # Handle q==0 (richness) and q==1 (exp(H)) as special cases via masking.
    is_q0 = np.isclose(q_arr, 0.0)
    is_q1 = np.isclose(q_arr, 1.0)
    is_generic = ~(is_q0 | is_q1)

    hills = np.empty_like(q_arr)
    hills[is_q0] = richness
    hills[is_q1] = np.exp(shannon)

    if np.any(is_generic):
        q_generic = q_arr[is_generic]
        # p shape: (n_species,); q shape: (n_q,) — broadcast via outer product.
        log_sum_pq = np.log(np.sum(p[:, np.newaxis] ** q_generic[np.newaxis, :], axis=0))
        hills[is_generic] = np.exp(log_sum_pq / (1.0 - q_generic))

    return pl.DataFrame({"q": q_arr, "hill": hills})


def _lchoose(n: float, k: float) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return float(gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0))


def _safe_sqrt(x: float) -> float:
    return float(np.sqrt(max(0.0, x)))


def _confidence_z(confidence: float) -> float:
    alpha = 1.0 - confidence
    return float(ndtri(1.0 - alpha / 2.0))


def _ci_bounds(value: float, sd: float, z: float) -> tuple[float, float]:
    return max(0.0, value - z * sd), value + z * sd


def rarefaction_point(
    abundance_table: pl.DataFrame,
    m: int,
    *,
    confidence: float = 0.95,
) -> RarefactionResult:
    """Compute one rarefaction or extrapolation point from abundance table."""
    if abundance_table.height == 0:
        return RarefactionResult(
            m=m,
            n=0,
            f0=0.0,
            f1=0.0,
            f2=0.0,
            s_obs=0.0,
            s_est=0.0,
            s_sd=0.0,
            s_lwr=0.0,
            s_upr=0.0,
            coverage=0.0,
            coverage_sd=0.0,
            coverage_lwr=0.0,
            coverage_upr=0.0,
            extrapolation=False,
        )

    at = abundance_table.select(pl.col("k").cast(pl.Float64), pl.col("f_k").cast(pl.Float64))
    k = at["k"].to_numpy()
    f_k = at["f_k"].to_numpy()

    n = int(np.sum(k * f_k))
    m = int(m)
    if n <= 0 or m <= 0:
        return RarefactionResult(
            m=m,
            n=n,
            f0=0.0,
            f1=0.0,
            f2=0.0,
            s_obs=float(np.sum(f_k)),
            s_est=0.0,
            s_sd=0.0,
            s_lwr=0.0,
            s_upr=0.0,
            coverage=0.0,
            coverage_sd=0.0,
            coverage_lwr=0.0,
            coverage_upr=0.0,
            extrapolation=m > n,
        )

    s_obs = float(np.sum(f_k))
    f1 = float(f_k[k == 1.0][0]) if np.any(k == 1.0) else 0.0
    f2 = float(f_k[k == 2.0][0]) if np.any(k == 2.0) else 0.0
    f0 = float(f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0)))
    z = _confidence_z(confidence)

    if m <= n:
        ldenom = _lchoose(float(n), float(m))
        alpha = np.where(
            k <= (n - m),
            np.exp(np.vectorize(_lchoose)(float(n) - k, float(m)) - ldenom),
            0.0,
        )
        sum1 = float(np.sum(f_k * alpha))
        sum2 = float(np.sum(f_k * (1.0 - alpha) * (1.0 - alpha)))
        if n == m:
            sum3 = 0.0
        else:
            sum3 = float(np.sum(f_k * k / float(n - m) * alpha))

        s_est = float(s_obs - sum1)
        s_sd = _safe_sqrt(sum2 - s_est * s_est / max(s_obs + f0, 1e-12))

        if m == n:
            denom = ((n - 1.0) * f1 + 2.0 * f2)
            if n <= 1 or denom <= 0:
                coverage = 1.0
            else:
                coverage = float(1.0 - f1 * (f1 - 1.0) / n * (n - 1.0) / denom)
        else:
            coverage = float(1.0 - sum3)
    else:
        dm = float(m - n)
        if f0 <= 0:
            s_est = s_obs
            s_sd = 0.0
            coverage = 1.0
        else:
            s_est = float(s_obs + f0 * (1.0 - (1.0 - f1 / (n * f0 + f1)) ** dm))

            d_f0_d_f1 = (2.0 * f1 - 1.0) / (2.0 * (f2 + 1.0))
            d_f0_d_f2 = -f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0) * (f2 + 1.0))
            inner = 1.0 - f1 / n / f0
            brackets = 1.0 - inner**dm
            d_brackets = -dm * inner ** (dm - 1.0)
            d_brackets_d_f1 = d_brackets * (-1.0 / n / f0 + f1 / n / (f0 * f0) * d_f0_d_f1)
            d_brackets_d_f2 = d_brackets * (f1 / n / (f0 * f0) * d_f0_d_f2)
            d_s_d_f1 = d_f0_d_f1 * brackets + f0 * d_brackets_d_f1
            d_s_d_f2 = d_f0_d_f2 * brackets + f0 * d_brackets_d_f2

            cov11 = f1 * (1.0 - f1 / (s_obs + f0))
            cov22 = f2 * (1.0 - f2 / (s_obs + f0))
            cov12 = -f1 * f2 / (s_obs + f0)
            var_s = (
                s_obs * (1.0 - s_obs / (s_obs + f0))
                + d_s_d_f1 * d_s_d_f1 * cov11
                + d_s_d_f2 * d_s_d_f2 * cov22
                + 2.0 * d_s_d_f1 * d_s_d_f2 * cov12
            )
            s_sd = _safe_sqrt(var_s)

            denom = (f1 + 2.0 * f2 / max(n - 1.0, 1.0))
            if f1 <= 0 or denom <= 0:
                coverage = 1.0
            else:
                coverage = float(1.0 - f1 / n * (f1 / denom) ** (dm + 1.0))

    coverage = float(np.clip(coverage, 0.0, 1.0))
    s_lwr, s_upr = _ci_bounds(s_est, s_sd, z)

    # Approximate coverage uncertainty for plotting CIs.
    coverage_sd = float(np.sqrt(max(coverage * (1.0 - coverage), 0.0) / max(float(m), 1.0)))
    cov_lwr = max(0.0, coverage - z * coverage_sd)
    cov_upr = min(1.0, coverage + z * coverage_sd)

    return RarefactionResult(
        m=m,
        n=n,
        f0=f0,
        f1=f1,
        f2=f2,
        s_obs=s_obs,
        s_est=s_est,
        s_sd=s_sd,
        s_lwr=s_lwr,
        s_upr=s_upr,
        coverage=coverage,
        coverage_sd=coverage_sd,
        coverage_lwr=cov_lwr,
        coverage_upr=cov_upr,
        extrapolation=m > n,
    )


def rarefaction_curve(
    counts: Sequence[int] | np.ndarray,
    *,
    m_steps: Sequence[int] | None = None,
    include_exact: bool = True,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Compute rarefaction/extrapolation richness and sample coverage curves."""
    arr = _as_positive_counts(counts)
    if arr.size == 0:
        return pl.DataFrame(
            schema={
                "m": pl.Int64,
                "n": pl.Int64,
                "f0": pl.Float64,
                "f1": pl.Float64,
                "f2": pl.Float64,
                "s_obs": pl.Float64,
                "s_est": pl.Float64,
                "s_sd": pl.Float64,
                "s_lwr": pl.Float64,
                "s_upr": pl.Float64,
                "coverage": pl.Float64,
                "coverage_sd": pl.Float64,
                "coverage_lwr": pl.Float64,
                "coverage_upr": pl.Float64,
                "extrapolation": pl.Boolean,
            }
        )

    n = int(arr.sum())
    if m_steps is None:
        base = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
        m_steps = [m for m in base if m <= 2 * n]
        if not m_steps:
            m_steps = [max(1, min(n, 10)), n]
    m_unique = sorted({int(m) for m in m_steps if int(m) > 0})

    at = build_abundance_table(arr)
    rows = [rarefaction_point(at, m, confidence=confidence).to_dict() for m in m_unique]
    if include_exact and n not in m_unique:
        rows.append(rarefaction_point(at, n, confidence=confidence).to_dict())

    return pl.DataFrame(rows).sort("m")


def counts_from_clonotypes(clonotypes: Sequence[object], count_field: CountField = "duplicate_count") -> np.ndarray:
    """Extract positive integer counts from a clonotype-like sequence."""
    if count_field == "barcode_count":
        raise ValueError("barcode_count requires explicit barcode aggregation from single-cell objects")
    values = [int(getattr(c, count_field, 0) or 0) for c in clonotypes]
    return _as_positive_counts(values)


def summarize_clonotypes(
    clonotypes: Sequence[object],
    *,
    count_field: CountField = "duplicate_count",
    expanded_threshold: float = 1e-3,
    hyperexpanded_threshold: float = 1e-2,
) -> DiversitySummary:
    """Summarize diversity metrics directly from clonotype-like objects."""
    counts = counts_from_clonotypes(clonotypes, count_field=count_field)
    return summarize_counts(
        counts,
        expanded_threshold=expanded_threshold,
        hyperexpanded_threshold=hyperexpanded_threshold,
    )


def hill_curve_clonotypes(
    clonotypes: Sequence[object],
    *,
    count_field: CountField = "duplicate_count",
    q_values: Sequence[float] | np.ndarray | None = None,
) -> pl.DataFrame:
    """Compute Hill profile directly from clonotype-like objects."""
    counts = counts_from_clonotypes(clonotypes, count_field=count_field)
    return hill_curve(counts, q_values=q_values)


def rarefaction_curve_clonotypes(
    clonotypes: Sequence[object],
    *,
    count_field: CountField = "duplicate_count",
    m_steps: Sequence[int] | None = None,
    include_exact: bool = True,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Compute rarefaction/coverage curve directly from clonotype-like objects."""
    counts = counts_from_clonotypes(clonotypes, count_field=count_field)
    return rarefaction_curve(
        counts,
        m_steps=m_steps,
        include_exact=include_exact,
        confidence=confidence,
    )


def summarize_count_groups(
    count_groups: Mapping[str, Sequence[int] | np.ndarray],
    *,
    expanded_threshold: float = 1e-3,
    hyperexpanded_threshold: float = 1e-2,
) -> dict[str, DiversitySummary]:
    """Summarize diversity metrics for each named count group."""
    return {
        group: summarize_counts(
            counts,
            expanded_threshold=expanded_threshold,
            hyperexpanded_threshold=hyperexpanded_threshold,
        )
        for group, counts in count_groups.items()
    }


def hill_curve_count_groups(
    count_groups: Mapping[str, Sequence[int] | np.ndarray],
    *,
    q_values: Sequence[float] | np.ndarray | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute Hill profiles for each named count group."""
    return {
        group: hill_curve(counts, q_values=q_values)
        for group, counts in count_groups.items()
    }


def rarefaction_curve_count_groups(
    count_groups: Mapping[str, Sequence[int] | np.ndarray],
    *,
    m_steps: Sequence[int] | None = None,
    include_exact: bool = True,
    confidence: float = 0.95,
) -> dict[str, pl.DataFrame]:
    """Compute rarefaction/coverage curves for each named count group."""
    return {
        group: rarefaction_curve(
            counts,
            m_steps=m_steps,
            include_exact=include_exact,
            confidence=confidence,
        )
        for group, counts in count_groups.items()
    }


def pooled_count_values(count_groups: Mapping[str, Sequence[int] | np.ndarray]) -> list[int]:
    """Flatten grouped counts into one pooled positive-count vector."""
    pooled: list[int] = []
    for counts in count_groups.values():
        pooled.extend(int(x) for x in np.asarray(list(counts), dtype=np.int64) if int(x) > 0)
    return pooled


def summarize_loci_clonotypes(
    loci: Mapping[str, Sequence[object]],
    *,
    count_field: CountField = "duplicate_count",
    expanded_threshold: float = 1e-3,
    hyperexpanded_threshold: float = 1e-2,
) -> dict[str, DiversitySummary]:
    """Summarize diversity metrics for each locus from clonotype groups."""
    return {
        locus: summarize_clonotypes(
            clonotypes,
            count_field=count_field,
            expanded_threshold=expanded_threshold,
            hyperexpanded_threshold=hyperexpanded_threshold,
        )
        for locus, clonotypes in loci.items()
    }


def hill_curve_loci_clonotypes(
    loci: Mapping[str, Sequence[object]],
    *,
    count_field: CountField = "duplicate_count",
    q_values: Sequence[float] | np.ndarray | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute Hill profiles for each locus from clonotype groups."""
    return {
        locus: hill_curve_clonotypes(
            clonotypes,
            count_field=count_field,
            q_values=q_values,
        )
        for locus, clonotypes in loci.items()
    }


def rarefaction_curve_loci_clonotypes(
    loci: Mapping[str, Sequence[object]],
    *,
    count_field: CountField = "duplicate_count",
    m_steps: Sequence[int] | None = None,
    include_exact: bool = True,
    confidence: float = 0.95,
) -> dict[str, pl.DataFrame]:
    """Compute rarefaction/coverage curves for each locus from clonotype groups."""
    return {
        locus: rarefaction_curve_clonotypes(
            clonotypes,
            count_field=count_field,
            m_steps=m_steps,
            include_exact=include_exact,
            confidence=confidence,
        )
        for locus, clonotypes in loci.items()
    }


def summaries_to_polars(summaries: dict[str, DiversitySummary]) -> pl.DataFrame:
    """Serialize a dictionary of diversity summaries to a Polars table."""
    if not summaries:
        return pl.DataFrame(
            schema={
                "group": pl.Utf8,
                "abundance": pl.Int64,
                "diversity": pl.Int64,
                "singletons": pl.Int64,
                "doubletons": pl.Int64,
                "expanded": pl.Int64,
                "hyperexpanded": pl.Int64,
                "chao1": pl.Float64,
                "gini_simpson": pl.Float64,
                "shannon": pl.Float64,
            }
        )
    rows: list[dict[str, int | float | str]] = []
    for group, summary in summaries.items():
        row: dict[str, int | float | str] = {"group": group}
        row.update(summary.to_dict())
        rows.append(row)
    return pl.DataFrame(rows)
