from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mir.basic.gene_usage import (
    GeneUsage,
    compute_batch_corrected_gene_usage,
    marginalize_batch_corrected_gene_usage,
)
from mir.basic.pgen import OlgaModel
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset
from mir.common.sampling import resample_to_gene_usage


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_simple_airr(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="	", index=False)


# ---------------------------------------------------------------------------
# Statistical validation helpers  (independent from library under test)
# ---------------------------------------------------------------------------

def _jsd_and_chi2(
    p_map: dict[object, float],
    q_map: dict[object, float],
) -> tuple[float, float, float]:
    keys = sorted(set(p_map) | set(q_map))
    if not keys:
        return 0.0, 0.0, 1.0
    p = np.array([max(float(p_map.get(k, 0.0)), 1e-12) for k in keys], dtype=float)
    q = np.array([max(float(q_map.get(k, 0.0)), 1e-12) for k in keys], dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    jsd = float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))
    observed = p * 1000.0
    expected = q * observed.sum()
    chi2_stat, chi2_p = stats.chisquare(observed, expected)
    return jsd, float(chi2_stat), float(chi2_p)


def _avg_freq(freq_list: list[dict[object, float]]) -> dict[object, float]:
    all_genes: set[object] = set().union(*freq_list)
    return {g: float(np.mean([f.get(g, 0.0) for f in freq_list])) for g in all_genes}


def _distribution_by_batch(
    df: pd.DataFrame,
    *,
    locus: str,
    batch_id: str,
    value_col: str,
) -> dict[object, float]:
    subset = df[(df["locus"] == locus) & (df["batch_id"] == batch_id)]
    if subset.empty:
        return {}
    vec = {k: max(float(v), 1e-12) for k, v in subset.groupby("gene")[value_col].mean().items()}
    total = sum(vec.values())
    return {k: v / total for k, v in vec.items()} if total > 0 else {k: 1.0 / len(vec) for k in vec}


# ---------------------------------------------------------------------------
# Mock repertoire builders
# ---------------------------------------------------------------------------

_TRB_BASE: dict[tuple[str, str], int] = {
    ("TRBV1", "TRBJ1-1"): 60,
    ("TRBV2", "TRBJ1-2"): 35,
    ("TRBV3", "TRBJ2-1"): 20,
}
_TRA_BASE: dict[tuple[str, str], int] = {
    ("TRAV1-2",  "TRAJ33"): 45,
    ("TRAV12-2", "TRAJ24"): 25,
    ("TRAV8-3",  "TRAJ31"): 15,
}
_AA = "ACDEFGHIKLMNPQRSTVWY"


def _make_clones_for_vj(
    locus: str,
    profile: dict[tuple[str, str], int],
    sample_id: str,
    rng: np.random.Generator,
    offset: int = 0,
) -> list[Clonotype]:
    clones: list[Clonotype] = []
    i = offset
    for (v, j), n in profile.items():
        for _ in range(max(int(n), 0)):
            dc = int(min(rng.zipf(1.5), 10_000))
            clones.append(Clonotype(
                sequence_id=f"{sample_id}_{locus}_{i}",
                locus=locus,
                junction_aa=f"CASS{_AA[i % len(_AA)]}F",
                v_gene=v,
                j_gene=j,
                duplicate_count=dc,
            ))
            i += 1
    return clones


def _make_locus_rep(
    locus: str,
    profile: dict[tuple[str, str], int],
    sample_id: str,
    rng: np.random.Generator,
    offset: int = 0,
) -> LocusRepertoire:
    return LocusRepertoire(
        clonotypes=_make_clones_for_vj(locus, profile, sample_id, rng, offset=offset),
        locus=locus,
        repertoire_id=sample_id,
    )


def _build_mock_dataset(
    *,
    n_per_batch: int = 6,
    apply_shift: bool = True,
    include_tra: bool = False,
    include_mismatch_loci: bool = False,
    include_empty_locus: bool = False,
    outlier: bool = False,
    seed: int = 7,
) -> RepertoireDataset:
    rng = np.random.default_rng(seed)
    samples: dict[str, SampleRepertoire] = {}
    for batch_idx, batch_id in enumerate(["batch_1", "batch_2"]):
        for i in range(n_per_batch):
            sample_id = f"{batch_id}_S{i:02d}"
            trb_profile = {k: max(1, int(v + rng.integers(-4, 5))) for k, v in _TRB_BASE.items()}
            if apply_shift and batch_id == "batch_2":
                trb_profile[("TRBV1", "TRBJ1-1")] = int(trb_profile[("TRBV1", "TRBJ1-1")] * 2)
            if outlier and batch_id == "batch_2" and i == 0:
                trb_profile[("TRBV1", "TRBJ1-1")] = 4000
            loci: dict[str, LocusRepertoire] = {
                "TRB": _make_locus_rep("TRB", trb_profile, sample_id, rng),
            }
            if include_tra:
                tra_profile = {k: max(1, int(v + rng.integers(-3, 4))) for k, v in _TRA_BASE.items()}
                loci["TRA"] = _make_locus_rep("TRA", tra_profile, sample_id, rng, offset=10_000)
            if include_mismatch_loci and i % 2 == 0:
                loci.pop("TRA", None)
            if include_empty_locus and i % 3 == 0:
                loci["TRG"] = LocusRepertoire(clonotypes=[], locus="TRG", repertoire_id=sample_id)
            samples[sample_id] = SampleRepertoire(
                loci=loci,
                sample_id=sample_id,
                sample_metadata={"sample_id": sample_id, "batch_id": batch_id, "batch_idx": batch_idx},
            )
    return RepertoireDataset(samples=samples)


# ---------------------------------------------------------------------------
# RepertoireDataset tests
# ---------------------------------------------------------------------------

def test_repertoire_dataset_from_folder_with_file_name(tmp_path: Path) -> None:
    _write_simple_airr(tmp_path / "s1.tsv", [
        {"junction_aa": "CASSIRSSYEQYF", "v_gene": "TRBV1", "j_gene": "TRBJ1-2", "duplicate_count": 3},
        {"junction_aa": "CASSLGQDTQYF",  "v_gene": "TRBV2", "j_gene": "TRBJ2-3", "duplicate_count": 2},
    ])
    _write_simple_airr(tmp_path / "sub" / "s2.tsv", [
        {"junction_aa": "CASSQETQYF", "v_gene": "TRBV3", "j_gene": "TRBJ1-1", "duplicate_count": 5},
    ])
    pd.DataFrame([
        {"sample_id": "S1", "file_name": "s1.tsv",     "batch_id": "batch_1"},
        {"sample_id": "S2", "file_name": "sub/s2.tsv", "batch_id": "batch_2"},
    ]).to_csv(tmp_path / "metadata.tsv", sep="	", index=False)
    ds = RepertoireDataset.from_folder(tmp_path)
    assert set(ds.samples.keys()) == {"S1", "S2"}
    assert ds["S1"].sample_id == "S1"
    assert ds["S2"].sample_id == "S2"
    assert ds.metadata["S1"]["batch_id"] == "batch_1"
    assert "TRB" in ds["S1"].loci


def test_repertoire_dataset_from_folder_with_mapping_function(tmp_path: Path) -> None:
    _write_simple_airr(tmp_path / "files" / "A.tsv", [
        {"junction_aa": "CASSIRSSYEQYF", "v_gene": "TRBV1", "j_gene": "TRBJ1-2", "duplicate_count": 1},
    ])
    _write_simple_airr(tmp_path / "files" / "B.tsv", [
        {"junction_aa": "CASSLGQDTQYF", "v_gene": "TRBV2", "j_gene": "TRBJ2-3", "duplicate_count": 1},
    ])
    pd.DataFrame([
        {"sample_id": "A", "batch_id": "batch_1"},
        {"sample_id": "B", "batch_id": "batch_2"},
    ]).to_csv(tmp_path / "metadata.tsv", sep="	", index=False)
    ds = RepertoireDataset.from_folder(
        tmp_path,
        file_name_to_sample_id=lambda sample_id: f"files/{sample_id}.tsv",
    )
    assert set(ds.samples.keys()) == {"A", "B"}
    assert ds["A"].sample_id == "A"
    assert ds.metadata["B"]["file_name"] == "files/B.tsv"


def test_sample_metadata_aggregated_from_sample_repertoires() -> None:
    locus_rep = LocusRepertoire(
        clonotypes=[Clonotype(
            sequence_id="c1", locus="TRB", junction_aa="CASSF",
            v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1,
        )],
        locus="TRB",
        repertoire_id="donor_1",
    )
    sample = SampleRepertoire(
        loci={"TRB": locus_rep},
        sample_id="donor_1",
        sample_metadata={"donor": "D1", "cohort": "A"},
    )
    ds = RepertoireDataset(samples={"donor_1": sample})
    assert ds.metadata["donor_1"]["donor"] == "D1"
    assert ds.metadata["donor_1"]["cohort"] == "A"
    assert ds.metadata["donor_1"]["sample_id"] == "donor_1"
    assert ds["donor_1"].sample_metadata["donor"] == "D1"
    assert ds["donor_1"].sample_id == "donor_1"


def test_caller_metadata_overrides_sample_metadata() -> None:
    locus_rep = LocusRepertoire(clonotypes=[], locus="TRB", repertoire_id="s1")
    sample = SampleRepertoire(
        loci={"TRB": locus_rep},
        sample_id="s1",
        sample_metadata={"batch_id": "old_batch", "extra": "keep"},
    )
    ds = RepertoireDataset(
        samples={"s1": sample},
        metadata={"s1": {"batch_id": "new_batch"}},
    )
    assert ds.metadata["s1"]["batch_id"] == "new_batch"
    assert ds.metadata["s1"]["extra"] == "keep"


# ---------------------------------------------------------------------------
# Batch correction tests
# ---------------------------------------------------------------------------

def _assert_correction_reduces_batch_effect(
    ds: RepertoireDataset,
    locus: str,
    scope: str,
    weighted: bool,
) -> None:
    corrected = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope=scope, weighted=weighted
    )
    raw_b1  = _distribution_by_batch(corrected, locus=locus, batch_id="batch_1", value_col="p")
    raw_b2  = _distribution_by_batch(corrected, locus=locus, batch_id="batch_2", value_col="p")
    corr_b1 = _distribution_by_batch(corrected, locus=locus, batch_id="batch_1", value_col="pfinal")
    corr_b2 = _distribution_by_batch(corrected, locus=locus, batch_id="batch_2", value_col="pfinal")
    raw_jsd,  _, raw_chi2_p  = _jsd_and_chi2(raw_b1,  raw_b2)
    corr_jsd, _, corr_chi2_p = _jsd_and_chi2(corr_b1, corr_b2)
    assert corr_jsd < raw_jsd, (
        f"scope={scope!r} weighted={weighted}: JSD raw={raw_jsd:.6f} corrected={corr_jsd:.6f}"
    )
    assert corr_chi2_p >= raw_chi2_p, (
        f"scope={scope!r} weighted={weighted}: chi2 p raw={raw_chi2_p:.4g} corrected={corr_chi2_p:.4g}"
    )


@pytest.mark.integration
def test_batch_correction_vj_scope_duplicates() -> None:
    _assert_correction_reduces_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="vj", weighted=True
    )


@pytest.mark.integration
def test_batch_correction_v_scope_duplicates() -> None:
    _assert_correction_reduces_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="v", weighted=True
    )


@pytest.mark.integration
def test_batch_correction_j_scope_duplicates() -> None:
    _assert_correction_reduces_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="j", weighted=True
    )


@pytest.mark.integration
def test_batch_correction_vj_scope_clonotypes() -> None:
    _assert_correction_reduces_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="vj", weighted=False
    )


@pytest.mark.integration
def test_batch_correction_v_scope_clonotypes() -> None:
    _assert_correction_reduces_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="v", weighted=False
    )


@pytest.mark.integration
def test_batch_correction_j_scope_clonotypes() -> None:
    _assert_correction_reduces_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="j", weighted=False
    )


@pytest.mark.integration
def test_no_artificial_batch_difference_shows_similarity() -> None:
    ds = _build_mock_dataset(apply_shift=False)
    corrected = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope="vj", weighted=True
    )
    b1 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_1", value_col="p")
    b2 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_2", value_col="p")
    jsd, _, _ = _jsd_and_chi2(b1, b2)
    # JSD only: Zipf-distributed duplicate_counts introduce read-level variance
    # between batches even without a profile shift, making chi2 uninformative here.
    assert jsd < 0.08, f"Unexpectedly high no-shift JSD={jsd:.6f}"


@pytest.mark.integration
def test_multi_locus_mismatch_and_empty_locus_are_skipped() -> None:
    ds = _build_mock_dataset(
        apply_shift=True, include_tra=True,
        include_mismatch_loci=True, include_empty_locus=True,
    )
    out = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope="vj", weighted=True
    )
    assert not out.empty
    assert "TRB" in set(out["locus"])
    assert "TRA" in set(out["locus"])
    assert "TRG" not in set(out["locus"]), "Empty locus repertoires should be skipped"
    missing_tra = [sid for sid, s in ds.samples.items() if "TRA" not in s.loci]
    assert missing_tra
    for sid in missing_tra:
        assert set(out[out["sample_id"] == sid]["locus"]) == {"TRB"}


@pytest.mark.integration
def test_outlier_stability_winsorized_and_z_capped() -> None:
    ds = _build_mock_dataset(apply_shift=True, outlier=True)
    out = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope="vj", weighted=True, pseudocount=1.0, z_cap=6.0,
    )
    assert not out.empty
    assert np.isfinite(out["mu"]).all()
    assert np.isfinite(out["sigma"]).all()
    assert np.isfinite(out["z"]).all()
    assert out["z"].abs().max() <= 6.0 + 1e-12


@pytest.mark.integration
def test_batch_correction_fails_for_non_overlapping_olga_vj_support() -> None:
    """When two batches have disjoint VJ support, correction should remain poor.

    We generate OLGA TRB clonotypes, split unique VJ pairs into two disjoint
    partitions, and build 10 samples for each partition/batch.
    """
    rng = np.random.default_rng(123)
    model = OlgaModel(locus="TRB", species="human", seed=123)

    # Build a pool with many unique VJ pairs, then split those pairs into two
    # disjoint partitions to enforce a non-overlapping support between batches.
    records = model.generate_sequences_with_meta(12_000, pgens=False, seed=123)
    by_vj: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        key = (str(rec["v_gene"]), str(rec["j_gene"]))
        by_vj.setdefault(key, []).append(rec)

    vj_keys = list(by_vj.keys())
    assert len(vj_keys) >= 8, "Not enough distinct VJ pairs from OLGA generation"
    rng.shuffle(vj_keys)
    mid = len(vj_keys) // 2
    left_vj = set(vj_keys[:mid])
    right_vj = set(vj_keys[mid:])

    samples: dict[str, SampleRepertoire] = {}
    for batch_id, partition in (("batch_1", left_vj), ("batch_2", right_vj)):
        pool = [rec for key in partition for rec in by_vj[key]]
        assert pool, f"Empty OLGA pool for {batch_id}"

        for i in range(10):
            sample_id = f"{batch_id}_S{i:02d}"
            chosen_idx = rng.integers(0, len(pool), size=700)
            clonotypes: list[Clonotype] = []
            for k, idx in enumerate(chosen_idx):
                rec = pool[int(idx)]
                clonotypes.append(
                    Clonotype(
                        sequence_id=f"{sample_id}_{k}",
                        locus="TRB",
                        junction_aa=str(rec["junction_aa"]),
                        v_gene=str(rec["v_gene"]),
                        j_gene=str(rec["j_gene"]),
                        duplicate_count=int(min(rng.zipf(1.5), 10_000)),
                    )
                )

            locus_rep = LocusRepertoire(clonotypes=clonotypes, locus="TRB", repertoire_id=sample_id)
            samples[sample_id] = SampleRepertoire(
                loci={"TRB": locus_rep},
                sample_id=sample_id,
                sample_metadata={"sample_id": sample_id, "batch_id": batch_id},
            )

    ds = RepertoireDataset(samples=samples)
    corrected = compute_batch_corrected_gene_usage(ds, batch_field="batch_id", scope="vj", weighted=True)

    raw_b1 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_1", value_col="p")
    raw_b2 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_2", value_col="p")
    pfinal_b1 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_1", value_col="pfinal")
    pfinal_b2 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_2", value_col="pfinal")

    raw_jsd, _, _ = _jsd_and_chi2(raw_b1, raw_b2)
    pfinal_jsd, _, _ = _jsd_and_chi2(pfinal_b1, pfinal_b2)

    # Non-overlapping support should remain strongly separated in the final
    # corrected probabilities.
    assert raw_jsd > 0.20, f"Expected strong raw batch separation, got JSD={raw_jsd:.6f}"
    assert pfinal_jsd > 0.40, (
        f"Expected large residual separation in pfinal for disjoint supports, got JSD={pfinal_jsd:.6f}"
    )

    # Even with correction, resampling to the pooled target cannot fully
    # remove separation when batches have disjoint VJ supports.
    scale = 10_000
    target_gene_usage = {
        row["gene"]: max(1, int(round(row["pavg"] * scale)))
        for _, row in (
            corrected[corrected["locus"] == "TRB"]
            .drop_duplicates("gene")[["gene", "pavg"]]
            .iterrows()
        )
    }
    resampled = _resample_batch_to_target(ds, target_gene_usage, "TRB", "vj", True)
    avg_resampled_b1 = _avg_freq(resampled["batch_1"])
    avg_resampled_b2 = _avg_freq(resampled["batch_2"])
    resampled_jsd, _, _ = _jsd_and_chi2(avg_resampled_b1, avg_resampled_b2)
    assert resampled_jsd > 0.50, (
        f"Expected high residual divergence after resampling disjoint supports, got JSD={resampled_jsd:.6f}"
    )


@pytest.mark.integration
def test_pfinal_is_normalized_per_sample_locus() -> None:
    ds = _build_mock_dataset(apply_shift=True, include_tra=True)
    out = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope="vj", weighted=True
    )
    grouped = out.groupby(["sample_id", "locus"], dropna=False)["pfinal"].sum()
    assert np.allclose(grouped.values, 1.0, atol=1e-9)


def test_marginalize_batch_corrected_gene_usage_v_matches_manual_groupby() -> None:
    ds = _build_mock_dataset(apply_shift=True, include_tra=True)
    out_vj = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope="vj", weighted=True
    )

    got = marginalize_batch_corrected_gene_usage(out_vj, scope="v")

    expected = out_vj.copy()
    expected["gene"] = expected["gene"].map(lambda g: str(g[0]))
    expected = (
        expected.groupby(["sample_id", "batch_id", "locus", "gene"], as_index=False)
        [["p", "pfinal", "pavg"]]
        .sum()
        .sort_values(["sample_id", "locus", "gene"])
        .reset_index(drop=True)
    )

    got = got.sort_values(["sample_id", "locus", "gene"]).reset_index(drop=True)
    assert list(got[["sample_id", "batch_id", "locus", "gene"]].itertuples(index=False, name=None)) == \
        list(expected[["sample_id", "batch_id", "locus", "gene"]].itertuples(index=False, name=None))
    assert np.allclose(got["p"].to_numpy(dtype=float), expected["p"].to_numpy(dtype=float), atol=1e-12)
    assert np.allclose(got["pfinal"].to_numpy(dtype=float), expected["pfinal"].to_numpy(dtype=float), atol=1e-12)


def test_marginalize_batch_corrected_gene_usage_j_matches_manual_groupby() -> None:
    ds = _build_mock_dataset(apply_shift=True, include_tra=True)
    out_vj = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope="vj", weighted=True
    )

    got = marginalize_batch_corrected_gene_usage(out_vj, scope="j")

    expected = out_vj.copy()
    expected["gene"] = expected["gene"].map(lambda g: str(g[1]))
    expected = (
        expected.groupby(["sample_id", "batch_id", "locus", "gene"], as_index=False)
        [["p", "pfinal", "pavg"]]
        .sum()
        .sort_values(["sample_id", "locus", "gene"])
        .reset_index(drop=True)
    )

    got = got.sort_values(["sample_id", "locus", "gene"]).reset_index(drop=True)
    assert list(got[["sample_id", "batch_id", "locus", "gene"]].itertuples(index=False, name=None)) == \
        list(expected[["sample_id", "batch_id", "locus", "gene"]].itertuples(index=False, name=None))
    assert np.allclose(got["p"].to_numpy(dtype=float), expected["p"].to_numpy(dtype=float), atol=1e-12)
    assert np.allclose(got["pfinal"].to_numpy(dtype=float), expected["pfinal"].to_numpy(dtype=float), atol=1e-12)


# ---------------------------------------------------------------------------
# Resampling tests
# ---------------------------------------------------------------------------

def _resample_batch_to_target(
    ds: RepertoireDataset,
    target_gene_usage: dict,
    locus: str,
    scope: str,
    weighted: bool,
) -> dict[str, list[dict[object, float]]]:
    count = "duplicates" if weighted else "clonotypes"
    resampled_by_batch: dict[str, list[dict[object, float]]] = {}
    for sample_id, sample in ds.samples.items():
        if locus not in sample.loci:
            continue
        resampled = resample_to_gene_usage(
            sample.loci[locus],
            target_gene_usage,
            scope=scope,
            weighted=weighted,
            random_seed=42,
        )
        gu = GeneUsage.from_repertoire(resampled)
        if scope == "vj":
            usage = gu.vj_usage(locus, count=count)
        elif scope == "v":
            usage = gu.v_usage(locus, count=count)
        else:
            usage = gu.j_usage(locus, count=count)
        total = sum(usage.values())
        if total == 0:
            continue
        freq = {gene: cnt / total for gene, cnt in usage.items()}
        resampled_by_batch.setdefault(ds.metadata[sample_id]["batch_id"], []).append(freq)
    return resampled_by_batch


def _assert_resampling_removes_batch_effect(
    ds: RepertoireDataset,
    locus: str,
    scope: str,
    weighted: bool,
) -> None:
    corrected = compute_batch_corrected_gene_usage(
        ds, batch_field="batch_id", scope=scope, weighted=weighted
    )
    scale = 10_000
    target_gene_usage: dict = {
        row["gene"]: max(1, int(round(row["pavg"] * scale)))
        for _, row in (
            corrected[corrected["locus"] == locus]
            .drop_duplicates("gene")[["gene", "pavg"]]
            .iterrows()
        )
    }
    resampled = _resample_batch_to_target(ds, target_gene_usage, locus, scope, weighted)
    avg_b1 = _avg_freq(resampled["batch_1"])
    avg_b2 = _avg_freq(resampled["batch_2"])
    target_total = sum(target_gene_usage.values())
    target_freq = {g: c / target_total for g, c in target_gene_usage.items()}
    jsd_batches,   _, p_batches   = _jsd_and_chi2(avg_b1, avg_b2)
    jsd_b1_target, _, p_b1_target = _jsd_and_chi2(avg_b1, target_freq)
    jsd_b2_target, _, p_b2_target = _jsd_and_chi2(avg_b2, target_freq)
    assert jsd_batches   < 0.005, f"scope={scope!r}: batch JSD={jsd_batches:.6f}"
    assert p_batches     > 0.05,  f"scope={scope!r}: batches still differ (p={p_batches:.4g})"
    assert jsd_b1_target < 0.01,  f"scope={scope!r}: batch_1 vs target JSD={jsd_b1_target:.6f}"
    assert jsd_b2_target < 0.01,  f"scope={scope!r}: batch_2 vs target JSD={jsd_b2_target:.6f}"
    assert p_b1_target   > 0.01,  f"scope={scope!r}: batch_1 vs target p={p_b1_target:.4g}"
    assert p_b2_target   > 0.01,  f"scope={scope!r}: batch_2 vs target p={p_b2_target:.4g}"


@pytest.mark.integration
def test_resample_vj_scope_to_pavg_removes_batch_effect() -> None:
    _assert_resampling_removes_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="vj", weighted=True
    )


@pytest.mark.integration
def test_resample_v_scope_to_pavg_removes_batch_effect() -> None:
    _assert_resampling_removes_batch_effect(
        _build_mock_dataset(apply_shift=True), locus="TRB", scope="v", weighted=True
    )
