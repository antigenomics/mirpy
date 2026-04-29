from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mir.basic.gene_usage import compute_batch_corrected_gene_usage
from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset


def _write_simple_airr(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def test_repertoire_dataset_from_folder_with_file_name(tmp_path: Path) -> None:
    _write_simple_airr(
        tmp_path / "s1.tsv",
        [
            {"junction_aa": "CASSIRSSYEQYF", "v_gene": "TRBV1", "j_gene": "TRBJ1-2", "duplicate_count": 3},
            {"junction_aa": "CASSLGQDTQYF", "v_gene": "TRBV2", "j_gene": "TRBJ2-3", "duplicate_count": 2},
        ],
    )
    _write_simple_airr(
        tmp_path / "sub" / "s2.tsv",
        [
            {"junction_aa": "CASSQETQYF", "v_gene": "TRBV3", "j_gene": "TRBJ1-1", "duplicate_count": 5},
        ],
    )

    metadata = pd.DataFrame(
        [
            {"sample_id": "S1", "file_name": "s1.tsv", "batch_id": "batch_1"},
            {"sample_id": "S2", "file_name": "sub/s2.tsv", "batch_id": "batch_2"},
        ]
    )
    metadata.to_csv(tmp_path / "metadata.tsv", sep="\t", index=False)

    ds = RepertoireDataset.from_folder(tmp_path)

    assert set(ds.samples.keys()) == {"S1", "S2"}
    assert ds["S1"].sample_id == "S1"
    assert ds["S2"].sample_id == "S2"
    assert ds.metadata["S1"]["batch_id"] == "batch_1"
    assert "TRB" in ds["S1"].loci


def test_repertoire_dataset_from_folder_with_mapping_function(tmp_path: Path) -> None:
    _write_simple_airr(
        tmp_path / "files" / "A.tsv",
        [{"junction_aa": "CASSIRSSYEQYF", "v_gene": "TRBV1", "j_gene": "TRBJ1-2", "duplicate_count": 1}],
    )
    _write_simple_airr(
        tmp_path / "files" / "B.tsv",
        [{"junction_aa": "CASSLGQDTQYF", "v_gene": "TRBV2", "j_gene": "TRBJ2-3", "duplicate_count": 1}],
    )

    metadata = pd.DataFrame(
        [
            {"sample_id": "A", "batch_id": "batch_1"},
            {"sample_id": "B", "batch_id": "batch_2"},
        ]
    )
    metadata.to_csv(tmp_path / "metadata.tsv", sep="\t", index=False)

    ds = RepertoireDataset.from_folder(
        tmp_path,
        file_name_to_sample_id=lambda sample_id: f"files/{sample_id}.tsv",
    )

    assert set(ds.samples.keys()) == {"A", "B"}
    assert ds["A"].sample_id == "A"
    assert ds.metadata["B"]["file_name"] == "files/B.tsv"


def test_sample_metadata_aggregated_from_sample_repertoires() -> None:
    """sample_id and sample_metadata from SampleRepertoire are reflected in the dataset."""
    locus_rep = LocusRepertoire(
        clonotypes=[
            Clonotype(
                sequence_id="c1", locus="TRB", junction_aa="CASSF",
                v_gene="TRBV1", j_gene="TRBJ1-1", duplicate_count=1,
            )
        ],
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
    # sample_metadata must be synced back
    assert ds["donor_1"].sample_metadata["donor"] == "D1"
    assert ds["donor_1"].sample_id == "donor_1"


def test_caller_metadata_overrides_sample_metadata() -> None:
    """Caller-supplied metadata takes precedence over sample.sample_metadata."""
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

    assert ds.metadata["s1"]["batch_id"] == "new_batch"   # override wins
    assert ds.metadata["s1"]["extra"] == "keep"            # sample field preserved


def _make_clones_for_vj(locus: str, profile: dict[tuple[str, str], int], sample_id: str, offset: int = 0) -> list[Clonotype]:
    clones: list[Clonotype] = []
    aa = "ACDEFGHIKLMNPQRSTVWY"
    i = offset
    for (v, j), n in profile.items():
        for _ in range(max(int(n), 0)):
            aa_char = aa[i % len(aa)]
            clones.append(
                Clonotype(
                    sequence_id=f"{sample_id}_{locus}_{i}",
                    locus=locus,
                    junction_aa=f"CASS{aa_char}F",
                    v_gene=v,
                    j_gene=j,
                    duplicate_count=1,
                )
            )
            i += 1
    return clones


def _make_locus_rep(locus: str, profile: dict[tuple[str, str], int], sample_id: str, offset: int = 0) -> LocusRepertoire:
    return LocusRepertoire(
        clonotypes=_make_clones_for_vj(locus, profile, sample_id, offset=offset),
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
) -> RepertoireDataset:
    rng = np.random.default_rng(7)

    trb_base = {
        ("TRBV1", "TRBJ1-1"): 60,
        ("TRBV2", "TRBJ1-2"): 35,
        ("TRBV3", "TRBJ2-1"): 20,
    }
    tra_base = {
        ("TRAV1-2", "TRAJ33"): 45,
        ("TRAV12-2", "TRAJ24"): 25,
        ("TRAV8-3", "TRAJ31"): 15,
    }

    samples: dict[str, SampleRepertoire] = {}

    for batch_idx, batch_id in enumerate(["batch_1", "batch_2"]):
        for i in range(n_per_batch):
            sample_id = f"{batch_id}_S{i:02d}"

            trb_profile = dict(trb_base)
            for k in trb_profile:
                trb_profile[k] = max(1, int(trb_profile[k] + rng.integers(-4, 5)))

            if apply_shift and batch_id == "batch_2":
                trb_profile[("TRBV1", "TRBJ1-1")] = int(trb_profile[("TRBV1", "TRBJ1-1")] * 2)

            if outlier and batch_id == "batch_2" and i == 0:
                trb_profile[("TRBV1", "TRBJ1-1")] = 4000

            loci: dict[str, LocusRepertoire] = {"TRB": _make_locus_rep("TRB", trb_profile, sample_id)}

            if include_tra:
                tra_profile = dict(tra_base)
                for k in tra_profile:
                    tra_profile[k] = max(1, int(tra_profile[k] + rng.integers(-3, 4)))
                loci["TRA"] = _make_locus_rep("TRA", tra_profile, sample_id, offset=10000)

            if include_mismatch_loci and i % 2 == 0:
                loci.pop("TRA", None)

            if include_empty_locus and i % 3 == 0:
                loci["TRG"] = LocusRepertoire(clonotypes=[], locus="TRG", repertoire_id=sample_id)

            samples[sample_id] = SampleRepertoire(
                loci=loci,
                sample_id=sample_id,
                sample_metadata={"sample_id": sample_id, "batch_id": batch_id, "batch_idx": batch_idx},
            )

    # Metadata is aggregated from sample.sample_metadata; no separate dict needed.
    return RepertoireDataset(samples=samples)


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
    vec = subset.groupby("gene")[value_col].mean().to_dict()
    vec = {k: max(float(v), 1e-12) for k, v in vec.items()}
    s = sum(vec.values())
    if s <= 0:
        n = max(len(vec), 1)
        return {k: 1.0 / n for k in vec}
    return {k: v / s for k, v in vec.items()}


def _jsd_and_chi2(p_map: dict[object, float], q_map: dict[object, float]) -> tuple[float, float, float]:
    keys = sorted(set(p_map) | set(q_map))
    if not keys:
        return 0.0, 0.0, 1.0

    p = np.array([max(float(p_map.get(k, 0.0)), 1e-12) for k in keys], dtype=float)
    q = np.array([max(float(q_map.get(k, 0.0)), 1e-12) for k in keys], dtype=float)
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    jsd = float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))

    observed = p * 1000.0
    expected = q * observed.sum()
    chi2_stat, chi2_p = stats.chisquare(observed, expected)
    return jsd, float(chi2_stat), float(chi2_p)


@pytest.mark.integration
def test_batch_correction_reduces_jsd_and_improves_chi2_similarity() -> None:
    ds = _build_mock_dataset(apply_shift=True, include_tra=False)
    corrected = compute_batch_corrected_gene_usage(ds, batch_field="batch_id", scope="vj", weighted=True)
    corrected["zsig"] = 1.0 / (1.0 + np.exp(-corrected["z"]))

    raw_b1 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_1", value_col="p")
    raw_b2 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_2", value_col="p")
    corr_b1 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_1", value_col="zsig")
    corr_b2 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_2", value_col="zsig")

    raw_jsd, raw_chi2, raw_chi2_p = _jsd_and_chi2(raw_b1, raw_b2)
    corr_jsd, corr_chi2, corr_chi2_p = _jsd_and_chi2(corr_b1, corr_b2)

    assert corr_jsd < raw_jsd, (
        f"Expected lower JSD after correction, got raw={raw_jsd:.6f}, corrected={corr_jsd:.6f}"
    )
    assert corr_chi2_p >= raw_chi2_p, (
        f"Expected >= chi2 p-value after correction, got raw={raw_chi2_p:.6g}, corrected={corr_chi2_p:.6g}"
    )
    assert np.isfinite(raw_chi2) and np.isfinite(corr_chi2)


@pytest.mark.integration
def test_no_artificial_batch_difference_shows_similarity() -> None:
    ds = _build_mock_dataset(apply_shift=False, include_tra=False)
    corrected = compute_batch_corrected_gene_usage(ds, batch_field="batch_id", scope="vj", weighted=True)

    b1 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_1", value_col="p")
    b2 = _distribution_by_batch(corrected, locus="TRB", batch_id="batch_2", value_col="p")

    jsd, _, chi2_p = _jsd_and_chi2(b1, b2)
    assert jsd < 0.08, f"Unexpectedly high no-shift JSD={jsd:.6f}"
    assert chi2_p > 1e-3, f"Unexpectedly low no-shift chi2 p-value={chi2_p:.6g}"


@pytest.mark.integration
def test_multi_locus_mismatch_and_empty_locus_are_skipped_without_error() -> None:
    ds = _build_mock_dataset(
        apply_shift=True,
        include_tra=True,
        include_mismatch_loci=True,
        include_empty_locus=True,
    )
    out = compute_batch_corrected_gene_usage(ds, batch_field="batch_id", scope="vj", weighted=True)

    assert not out.empty
    assert "TRB" in set(out["locus"])
    assert "TRA" in set(out["locus"])
    assert "TRG" not in set(out["locus"]), "Empty locus repertoires should be skipped"

    missing_tra_samples = [sid for sid, s in ds.samples.items() if "TRA" not in s.loci]
    assert missing_tra_samples
    for sid in missing_tra_samples:
        sample_rows = out[out["sample_id"] == sid]
        assert set(sample_rows["locus"]) == {"TRB"}


@pytest.mark.integration
def test_outlier_stability_winsorized_and_z_capped() -> None:
    ds = _build_mock_dataset(apply_shift=True, include_tra=False, outlier=True)
    out = compute_batch_corrected_gene_usage(
        ds,
        batch_field="batch_id",
        scope="vj",
        weighted=True,
        pseudocount=1.0,
        z_cap=6.0,
    )

    assert not out.empty
    assert np.isfinite(out["mu"]).all()
    assert np.isfinite(out["sigma"]).all()
    assert np.isfinite(out["z"]).all()
    assert out["z"].abs().max() <= 6.0 + 1e-12
