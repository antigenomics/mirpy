"""Association analysis for clonotype panels against sample metadata.

The initial implementation focuses on exact contingency-table statistics for
single-chain sample repertoires, co-occurrence of two single-chain targets,
and paired-chain sample repertoires when metadata are supplied separately.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp
from math import log2
import re
from typing import Iterable, Literal, Sequence

import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from mir.biomarkers._shared import MatchMode, match_flags, normalize_match_mode
from mir.common.alleles import genes_match, strip_allele
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.single_cell import PairedClonotype, PairedRepertoire
from mir.graph.distance_utils import is_within_threshold

AssociationCountMode = Literal["sample", "rearrangement"]
AssociationTest = Literal["auto", "fisher", "chi2", "depth_glm"]

_METADATA_SPLIT_RE = re.compile(r"[,;|]")


@dataclass(frozen=True)
class AssociationParams:
    """Parameters controlling clonotype association scans.

    Args:
        match_mode: V/J restriction mode.
        metric: Sequence distance metric.
        max_distance: Maximum allowed distance between target and hit.
        count_mode: Use sample presence or rearrangement-row counts.
        test: Statistical test mode.
        p_adj_method: Multiple-testing correction passed to statsmodels.
    """

    match_mode: MatchMode = "vj"
    metric: Literal["hamming", "levenshtein"] = "hamming"
    max_distance: int = 0
    count_mode: AssociationCountMode = "sample"
    test: AssociationTest = "auto"
    p_adj_method: str = "fdr_bh"

    def __post_init__(self) -> None:
        normalize_match_mode(self.match_mode)
        if self.metric not in {"hamming", "levenshtein"}:
            raise ValueError("metric must be 'hamming' or 'levenshtein'")
        if self.max_distance < 0:
            raise ValueError("max_distance must be >= 0")
        if self.count_mode not in {"sample", "rearrangement"}:
            raise ValueError("count_mode must be 'sample' or 'rearrangement'")
        if self.test not in {"auto", "fisher", "chi2", "depth_glm"}:
            raise ValueError("test must be 'auto', 'fisher', 'chi2', or 'depth_glm'")


@dataclass(frozen=True)
class AssociationResult:
    """Association outputs.

    Attributes:
        table: One row per target with the global association test.
        contrast_table: One-vs-rest binary contrasts per target and metadata level.
        params: Effective parameter bundle.
    """

    table: pl.DataFrame
    contrast_table: pl.DataFrame
    params: AssociationParams


@dataclass(frozen=True)
class CooccurrenceResult:
    """Co-occurrence results for pairs of targets."""

    table: pl.DataFrame
    params: AssociationParams


def build_public_clonotype_panel(
    samples: Sequence[SampleRepertoire],
    *,
    locus: str,
    min_sample_fraction: float = 0.05,
    min_sample_count: int | None = None,
) -> list[Clonotype]:
    """Build an exact public clonotype panel from a sample cohort.

    Public clonotypes are defined by exact ``(junction_aa, v_call, j_call)``
    identity and retained when present in at least ``min_sample_fraction`` of
    samples or ``min_sample_count`` samples, whichever is larger.
    """
    if not samples:
        return []
    if min_sample_fraction <= 0 or min_sample_fraction > 1:
        raise ValueError("min_sample_fraction must be within (0, 1]")

    threshold = max(1, int(len(samples) * min_sample_fraction + 0.999999))
    if min_sample_count is not None:
        threshold = max(threshold, int(min_sample_count))

    sample_hits: Counter[tuple[str, str, str]] = Counter()
    exemplar: dict[tuple[str, str, str], Clonotype] = {}
    for sample in samples:
        repertoire = sample.loci.get(locus)
        if repertoire is None:
            continue
        seen: set[tuple[str, str, str]] = set()
        for clonotype in repertoire.clonotypes:
            key = (
                clonotype.junction_aa,
                strip_allele(clonotype.v_call),
                strip_allele(clonotype.j_call),
            )
            seen.add(key)
            exemplar.setdefault(key, clonotype)
        sample_hits.update(seen)

    panel: list[Clonotype] = []
    for key, count in sample_hits.items():
        if count >= threshold:
            clone = exemplar[key]
            panel.append(
                Clonotype(
                    sequence_id=clone.sequence_id,
                    locus=locus,
                    junction_aa=clone.junction_aa,
                    v_call=clone.v_call,
                    j_call=clone.j_call,
                    duplicate_count=clone.duplicate_count,
                    _validate=False,
                )
            )
    return panel


def associate_clonotype_metadata(
    samples: Sequence[SampleRepertoire],
    targets: Sequence[Clonotype],
    *,
    metadata_field: str,
    metadata_value: str | None = None,
    params: AssociationParams | None = None,
) -> AssociationResult:
    """Test single-chain clonotype associations against sample metadata.

    When ``metadata_value`` is provided, the metadata field is treated as a
    flexible label container and converted to a binary present/absent phenotype.
    Otherwise the metadata field is treated as categorical, with a global test
    per target and one-vs-rest contrasts for effect-size inspection.
    """
    effective = params or AssociationParams()
    summaries: list[dict] = []
    contrasts: list[dict] = []

    # For exact matching (max_distance == 0) a target hit reduces to junction_aa
    # equality, so we index each repertoire by junction_aa once and reuse it
    # across all targets instead of rescanning every clonotype per target.
    match_index_cache: dict[int, dict[str, list[Clonotype]]] = {}

    for target in targets:
        categories, detected, background, sample_rows = _counts_for_single_chain_target(
            samples=samples,
            target=target,
            metadata_field=metadata_field,
            metadata_value=metadata_value,
            params=effective,
            match_index_cache=match_index_cache,
        )
        if not categories:
            continue

        table = [[int(detected[i]), int(background[i])] for i in range(len(categories))]
        p_value, test_name, depth_or = _run_table_test(
            table,
            effective.test,
            categories=categories,
            sample_rows=sample_rows,
            count_mode=effective.count_mode,
        )
        odds_ratio, log2_odds_ratio = (None, None)
        if len(categories) == 2:
            odds_ratio, log2_odds_ratio = _compute_or(table[0][0], table[0][1], table[1][0], table[1][1])
        if depth_or is not None:
            odds_ratio = depth_or
            log2_odds_ratio = float(log2(depth_or)) if depth_or > 0 else None

        summaries.append(
            {
                "target_id": _target_id(target),
                "locus": target.locus,
                "junction_aa": target.junction_aa,
                "v_call": target.v_call,
                "j_call": target.j_call,
                "metadata_field": metadata_field,
                "metadata_value": metadata_value,
                "levels": categories,
                "detected_counts": detected,
                "background_counts": background,
                "p_value": p_value,
                "test": test_name,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log2_odds_ratio,
            }
        )
        contrasts.extend(
            _build_contrast_rows(
                target=target,
                metadata_field=metadata_field,
                categories=categories,
                detected=detected,
                background=background,
            )
        )

    summary_df = _apply_p_adjustment(
        pl.DataFrame(summaries), p_col="p_value", q_col="p_value_adj", method=effective.p_adj_method
    )
    contrast_df = _apply_p_adjustment(
        pl.DataFrame(contrasts), p_col="p_value", q_col="p_value_adj", method=effective.p_adj_method
    )
    return AssociationResult(table=summary_df, contrast_table=contrast_df, params=effective)


def associate_paired_clonotype_metadata(
    samples: Sequence[PairedRepertoire],
    targets: Sequence[PairedClonotype | tuple[Clonotype, Clonotype]],
    *,
    sample_metadata: Sequence[dict],
    metadata_field: str,
    metadata_value: str | None = None,
    params: AssociationParams | None = None,
) -> AssociationResult:
    """Test paired-chain clonotype associations against sample metadata."""
    if len(samples) != len(sample_metadata):
        raise ValueError("samples and sample_metadata must have the same length")

    effective = params or AssociationParams()
    summaries: list[dict] = []
    contrasts: list[dict] = []

    normalized_targets = [_coerce_paired_target(target) for target in targets]
    for paired_target in normalized_targets:
        categories, detected, background, sample_rows = _counts_for_paired_target(
            samples=samples,
            sample_metadata=sample_metadata,
            target=paired_target,
            metadata_field=metadata_field,
            metadata_value=metadata_value,
            params=effective,
        )
        if not categories:
            continue

        table = [[int(detected[i]), int(background[i])] for i in range(len(categories))]
        p_value, test_name, depth_or = _run_table_test(
            table,
            effective.test,
            categories=categories,
            sample_rows=sample_rows,
            count_mode=effective.count_mode,
        )
        odds_ratio, log2_odds_ratio = (None, None)
        if len(categories) == 2:
            odds_ratio, log2_odds_ratio = _compute_or(table[0][0], table[0][1], table[1][0], table[1][1])
        if depth_or is not None:
            odds_ratio = depth_or
            log2_odds_ratio = float(log2(depth_or)) if depth_or > 0 else None

        summaries.append(
            {
                "target_id": _paired_target_id(paired_target),
                "locus_pair": f"{paired_target.clonotype1.locus}_{paired_target.clonotype2.locus}",
                "junction_aa_1": paired_target.clonotype1.junction_aa,
                "junction_aa_2": paired_target.clonotype2.junction_aa,
                "v_gene_1": paired_target.clonotype1.v_call,
                "v_gene_2": paired_target.clonotype2.v_call,
                "j_gene_1": paired_target.clonotype1.j_call,
                "j_gene_2": paired_target.clonotype2.j_call,
                "metadata_field": metadata_field,
                "metadata_value": metadata_value,
                "levels": categories,
                "detected_counts": detected,
                "background_counts": background,
                "p_value": p_value,
                "test": test_name,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log2_odds_ratio,
            }
        )
        contrasts.extend(
            _build_paired_contrast_rows(
                target=paired_target,
                metadata_field=metadata_field,
                categories=categories,
                detected=detected,
                background=background,
            )
        )

    summary_df = _apply_p_adjustment(
        pl.DataFrame(summaries), p_col="p_value", q_col="p_value_adj", method=effective.p_adj_method
    )
    contrast_df = _apply_p_adjustment(
        pl.DataFrame(contrasts), p_col="p_value", q_col="p_value_adj", method=effective.p_adj_method
    )
    return AssociationResult(table=summary_df, contrast_table=contrast_df, params=effective)


def associate_clonotype_cooccurrence(
    samples: Sequence[SampleRepertoire],
    left_targets: Sequence[Clonotype],
    right_targets: Sequence[Clonotype],
    *,
    params: AssociationParams | None = None,
) -> CooccurrenceResult:
    """Measure sample-level co-occurrence for two sets of single-chain targets."""
    effective = params or AssociationParams(count_mode="sample")
    rows: list[dict] = []
    for left in left_targets:
        left_hits = [_single_target_presence(sample, left, effective) for sample in samples]
        for right in right_targets:
            right_hits = [_single_target_presence(sample, right, effective) for sample in samples]
            both = sum(1 for lh, rh in zip(left_hits, right_hits) if lh and rh)
            left_only = sum(1 for lh, rh in zip(left_hits, right_hits) if lh and not rh)
            right_only = sum(1 for lh, rh in zip(left_hits, right_hits) if not lh and rh)
            neither = sum(1 for lh, rh in zip(left_hits, right_hits) if not lh and not rh)
            odds_ratio, log2_odds_ratio = _compute_or(both, left_only, right_only, neither)
            p_value = float(fisher_exact([[both, left_only], [right_only, neither]], alternative="two-sided")[1])
            rows.append(
                {
                    "left_target_id": _target_id(left),
                    "right_target_id": _target_id(right),
                    "both": both,
                    "left_only": left_only,
                    "right_only": right_only,
                    "neither": neither,
                    "odds_ratio": odds_ratio,
                    "log2_odds_ratio": log2_odds_ratio,
                    "p_value": p_value,
                }
            )
    table = _apply_p_adjustment(pl.DataFrame(rows), p_col="p_value", q_col="p_value_adj", method=effective.p_adj_method)
    return CooccurrenceResult(table=table, params=effective)


def _counts_for_single_chain_target(
    *,
    samples: Sequence[SampleRepertoire],
    target: Clonotype,
    metadata_field: str,
    metadata_value: str | None,
    params: AssociationParams,
    match_index_cache: dict[int, dict[str, list[Clonotype]]] | None = None,
) -> tuple[list[str], list[int], list[int], list[dict]]:
    by_category_detected: Counter[str] = Counter()
    by_category_total: Counter[str] = Counter()
    sample_rows: list[dict] = []

    for sample in samples:
        raw_category = sample.sample_metadata.get(metadata_field)
        category = _categorize_metadata(raw_category, metadata_value)
        if category is None:
            continue

        repertoire = sample.loci.get(target.locus)
        if repertoire is None:
            by_category_total[category] += 0
            continue

        matched = _count_single_chain_matches(
            repertoire, target, params, match_index_cache=match_index_cache
        )
        total = 1 if params.count_mode == "sample" else repertoire.clonotype_count
        sample_rows.append({"category": category, "matched": int(matched), "total": int(total)})

        by_category_total[category] += total
        if params.count_mode == "sample":
            by_category_detected[category] += int(matched > 0)
        else:
            by_category_detected[category] += matched

    categories = sorted(by_category_total)
    detected = [int(by_category_detected.get(category, 0)) for category in categories]
    background = [int(by_category_total[category] - by_category_detected.get(category, 0)) for category in categories]
    return categories, detected, background, sample_rows


def _counts_for_paired_target(
    *,
    samples: Sequence[PairedRepertoire],
    sample_metadata: Sequence[dict],
    target: PairedClonotype,
    metadata_field: str,
    metadata_value: str | None,
    params: AssociationParams,
) -> tuple[list[str], list[int], list[int], list[dict]]:
    by_category_detected: Counter[str] = Counter()
    by_category_total: Counter[str] = Counter()
    sample_rows: list[dict] = []
    locus_pair = f"{target.clonotype1.locus}_{target.clonotype2.locus}"

    for sample, metadata in zip(samples, sample_metadata):
        raw_category = metadata.get(metadata_field)
        category = _categorize_metadata(raw_category, metadata_value)
        if category is None:
            continue
        repertoire = sample.paired_locus_repertoires.get(locus_pair)
        if repertoire is None:
            continue

        matched = _count_paired_matches(repertoire.paired_clonotypes, target, params)
        total = 1 if params.count_mode == "sample" else repertoire.clonotype_count
        sample_rows.append({"category": category, "matched": int(matched), "total": int(total)})
        by_category_total[category] += total
        if params.count_mode == "sample":
            by_category_detected[category] += int(matched > 0)
        else:
            by_category_detected[category] += matched

    categories = sorted(by_category_total)
    detected = [int(by_category_detected.get(category, 0)) for category in categories]
    background = [int(by_category_total[category] - by_category_detected.get(category, 0)) for category in categories]
    return categories, detected, background, sample_rows


def _single_target_presence(sample: SampleRepertoire, target: Clonotype, params: AssociationParams) -> bool:
    repertoire = sample.loci.get(target.locus)
    if repertoire is None:
        return False
    return _count_single_chain_matches(repertoire, target, params) > 0


def _count_single_chain_matches(
    repertoire: LocusRepertoire,
    target: Clonotype,
    params: AssociationParams,
    *,
    match_index_cache: dict[int, dict[str, list[Clonotype]]] | None = None,
) -> int:
    match_v, match_j = match_flags(normalize_match_mode(params.match_mode))

    # Fast path: exact matching means a hit requires identical junction_aa, so
    # we restrict to clonotypes sharing the target junction (indexed once per
    # repertoire) and only check the V/J flags on that short candidate list.
    if match_index_cache is not None and params.max_distance == 0:
        index = match_index_cache.get(id(repertoire))
        if index is None:
            index = {}
            for clonotype in repertoire.clonotypes:
                index.setdefault(clonotype.junction_aa, []).append(clonotype)
            match_index_cache[id(repertoire)] = index
        candidates = index.get(target.junction_aa)
        if not candidates:
            return 0
        if not match_v and not match_j:
            return len(candidates)
        return sum(
            1
            for clonotype in candidates
            if (not match_v or genes_match(clonotype.v_call, target.v_call))
            and (not match_j or genes_match(clonotype.j_call, target.j_call))
        )

    matched = 0
    for clonotype in repertoire.clonotypes:
        if match_v and not genes_match(clonotype.v_call, target.v_call):
            continue
        if match_j and not genes_match(clonotype.j_call, target.j_call):
            continue
        if is_within_threshold(clonotype.junction_aa, target.junction_aa, params.metric, params.max_distance):
            matched += 1
    return matched


def _count_paired_matches(
    paired_clonotypes: Sequence[PairedClonotype],
    target: PairedClonotype,
    params: AssociationParams,
) -> int:
    matched = 0
    for paired in paired_clonotypes:
        if _pair_matches(paired, target, params):
            matched += 1
    return matched


def _pair_matches(observed: PairedClonotype, target: PairedClonotype, params: AssociationParams) -> bool:
    return _single_chain_matches(observed.clonotype1, target.clonotype1, params) and _single_chain_matches(
        observed.clonotype2,
        target.clonotype2,
        params,
    )


def _single_chain_matches(observed: Clonotype, target: Clonotype, params: AssociationParams) -> bool:
    match_v, match_j = match_flags(normalize_match_mode(params.match_mode))
    if match_v and not genes_match(observed.v_call, target.v_call):
        return False
    if match_j and not genes_match(observed.j_call, target.j_call):
        return False
    return is_within_threshold(observed.junction_aa, target.junction_aa, params.metric, params.max_distance)


def _coerce_paired_target(target: PairedClonotype | tuple[Clonotype, Clonotype]) -> PairedClonotype:
    if isinstance(target, PairedClonotype):
        return target
    return PairedClonotype(
        pair_id=f"{_target_id(target[0])}__{_target_id(target[1])}", clonotype1=target[0], clonotype2=target[1]
    )


def _categorize_metadata(raw_value, metadata_value: str | None) -> str | None:
    if metadata_value is not None:
        return str(metadata_value) if _metadata_contains_label(raw_value, metadata_value) else f"not_{metadata_value}"
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return value or None
    if isinstance(raw_value, Iterable) and not isinstance(raw_value, (bytes, bytearray, dict)):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
        if not values:
            return None
        return ",".join(sorted(dict.fromkeys(values)))
    return str(raw_value)


def _metadata_contains_label(raw_value, metadata_value: str) -> bool:
    expected = str(metadata_value).strip().casefold()
    if raw_value is None:
        return False
    if isinstance(raw_value, str):
        tokens = [tok.strip() for tok in _METADATA_SPLIT_RE.split(raw_value) if tok.strip()]
        if not tokens:
            tokens = [raw_value.strip()]
        return any(token.casefold() == expected for token in tokens if token)
    if isinstance(raw_value, Iterable) and not isinstance(raw_value, (bytes, bytearray, dict)):
        return any(str(item).strip().casefold() == expected for item in raw_value)
    return str(raw_value).strip().casefold() == expected


def _run_table_test(
    table: list[list[int]],
    requested: AssociationTest,
    *,
    categories: Sequence[str],
    sample_rows: Sequence[dict],
    count_mode: AssociationCountMode,
) -> tuple[float, str, float | None]:
    if requested == "depth_glm":
        p_depth, test_name, depth_or = _run_depth_glm_test(
            categories=categories,
            sample_rows=sample_rows,
            count_mode=count_mode,
        )
        if p_depth is not None:
            return p_depth, test_name, depth_or

    if len(table) == 2:
        if requested in {"auto", "fisher"}:
            p_value = float(fisher_exact(table, alternative="two-sided")[1])
            return p_value, "fisher", None
        p_value = float(chi2_contingency(table)[1])
        return p_value, "chi2", None

    p_value = float(chi2_contingency(table)[1])
    return p_value, "chi2", None


def _run_depth_glm_test(
    *,
    categories: Sequence[str],
    sample_rows: Sequence[dict],
    count_mode: AssociationCountMode,
) -> tuple[float | None, str, float | None]:
    if len(categories) != 2 or not sample_rows:
        return None, "depth_glm_unavailable", None

    category_to_idx = {cat: i for i, cat in enumerate(categories)}
    y: list[float] = []
    grp: list[float] = []
    depth: list[float] = []
    freq_weights: list[float] | None = None

    if count_mode == "rearrangement":
        freq_weights = []

    for row in sample_rows:
        category = row["category"]
        if category not in category_to_idx:
            continue
        matched = int(row["matched"])
        total = max(1, int(row["total"]))
        grp.append(float(category_to_idx[category]))
        depth.append(float(np.log1p(total)))
        if count_mode == "sample":
            y.append(float(matched > 0))
        else:
            y.append(float(matched / total))
            freq_weights.append(float(total))

    if len(y) < 4 or len(set(grp)) < 2:
        return None, "depth_glm_unavailable", None

    exog = sm.add_constant(np.column_stack([np.array(grp), np.array(depth)]), has_constant="add")

    try:
        if count_mode == "sample":
            fit = sm.GLM(np.array(y), exog, family=sm.families.Binomial()).fit()
        else:
            fit = sm.GLM(
                np.array(y),
                exog,
                family=sm.families.Binomial(),
                freq_weights=np.array(freq_weights),
            ).fit()
    except Exception:
        return None, "depth_glm_failed", None

    p_value = float(fit.pvalues[1])
    odds_ratio = float(exp(float(fit.params[1])))
    return p_value, "depth_glm", odds_ratio


def _compute_or(a: int, b: int, c: int, d: int) -> tuple[float | None, float | None]:
    odds_ratio = float(fisher_exact([[a, b], [c, d]], alternative="two-sided")[0])
    if odds_ratio <= 0:
        return odds_ratio, None
    return odds_ratio, float(log2(odds_ratio))


def _build_contrast_rows(
    *,
    target: Clonotype,
    metadata_field: str,
    categories: Sequence[str],
    detected: Sequence[int],
    background: Sequence[int],
) -> list[dict]:
    rows: list[dict] = []
    total_detected = int(sum(detected))
    total_background = int(sum(background))
    for idx, level in enumerate(categories):
        detected_in = int(detected[idx])
        background_in = int(background[idx])
        detected_out = total_detected - detected_in
        background_out = total_background - background_in
        odds_ratio, log2_odds_ratio = _compute_or(detected_in, background_in, detected_out, background_out)
        p_value = float(fisher_exact([[detected_in, background_in], [detected_out, background_out]], alternative="two-sided")[1])
        rows.append(
            {
                "target_id": _target_id(target),
                "locus": target.locus,
                "junction_aa": target.junction_aa,
                "v_call": target.v_call,
                "j_call": target.j_call,
                "metadata_field": metadata_field,
                "level": level,
                "detected_in_level": detected_in,
                "background_in_level": background_in,
                "detected_outside_level": detected_out,
                "background_outside_level": background_out,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log2_odds_ratio,
                "p_value": p_value,
            }
        )
    return rows


def _build_paired_contrast_rows(
    *,
    target: PairedClonotype,
    metadata_field: str,
    categories: Sequence[str],
    detected: Sequence[int],
    background: Sequence[int],
) -> list[dict]:
    rows: list[dict] = []
    total_detected = int(sum(detected))
    total_background = int(sum(background))
    for idx, level in enumerate(categories):
        detected_in = int(detected[idx])
        background_in = int(background[idx])
        detected_out = total_detected - detected_in
        background_out = total_background - background_in
        odds_ratio, log2_odds_ratio = _compute_or(detected_in, background_in, detected_out, background_out)
        p_value = float(fisher_exact([[detected_in, background_in], [detected_out, background_out]], alternative="two-sided")[1])
        rows.append(
            {
                "target_id": _paired_target_id(target),
                "locus_pair": f"{target.clonotype1.locus}_{target.clonotype2.locus}",
                "junction_aa_1": target.clonotype1.junction_aa,
                "junction_aa_2": target.clonotype2.junction_aa,
                "metadata_field": metadata_field,
                "level": level,
                "detected_in_level": detected_in,
                "background_in_level": background_in,
                "detected_outside_level": detected_out,
                "background_outside_level": background_out,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log2_odds_ratio,
                "p_value": p_value,
            }
        )
    return rows


def _apply_p_adjustment(df: pl.DataFrame, *, p_col: str, q_col: str, method: str) -> pl.DataFrame:
    if df.is_empty() or p_col not in df.columns:
        return df
    qvals = multipletests(df[p_col].to_list(), method=method)[1].tolist()
    return df.with_columns(pl.Series(name=q_col, values=qvals))


def _target_id(target: Clonotype) -> str:
    if target.sequence_id:
        return target.sequence_id
    return f"{target.locus}:{target.v_call}:{target.j_call}:{target.junction_aa}"


def _paired_target_id(target: PairedClonotype) -> str:
    return target.pair_id or f"{_target_id(target.clonotype1)}__{_target_id(target.clonotype2)}"


__all__ = [
    "AssociationParams",
    "AssociationResult",
    "CooccurrenceResult",
    "associate_clonotype_cooccurrence",
    "associate_clonotype_metadata",
    "associate_paired_clonotype_metadata",
    "build_public_clonotype_panel",
]
