"""Tests for SampleRepertoire and LocusRepertoire using SRX-format data.

Samples are read directly from the tarball (tests/srx_repertoires/samples.tar.gz)
so no extracted files need to be kept in the repository.
"""

from __future__ import annotations

import io
import tarfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from mir.common.clonotype import Clonotype
from mir.common.repertoire import LocusRepertoire, SampleRepertoire, infer_locus

SRX_DIR = Path(__file__).parent / "srx_repertoires"
TARBALL = SRX_DIR / "samples.tar.gz"
META_PATH = SRX_DIR / "meta.tsv"

_AIRR_CALL_RENAMES = {"v_call": "v_gene", "j_call": "j_gene", "c_call": "c_gene"}

_ALL_LOCI = {"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tsv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), sep="\t").rename(columns=_AIRR_CALL_RENAMES)


def _clonotypes_from_df(df: pd.DataFrame) -> list[Clonotype]:
    result = []
    for _, row in df.iterrows():
        v_gene = str(row.get("v_gene", ""))
        locus = infer_locus(v_gene)
        result.append(Clonotype(
            duplicate_count=int(row.get("duplicate_count", 1)),
            junction=str(row.get("junction", "")),
            junction_aa=str(row.get("junction_aa", "")),
            v_gene=v_gene,
            j_gene=str(row.get("j_gene", "")),
            c_gene=str(row.get("c_gene", "")),
            locus=locus,
        ))
    return result


def load_srx_sample(tar: tarfile.TarFile, run_id: str) -> SampleRepertoire:
    """Read one run TSV from an open tarball into a :class:`SampleRepertoire`."""
    member_name = f"./{run_id}.tsv"
    try:
        raw = tar.extractfile(member_name).read()
    except KeyError:
        raise FileNotFoundError(f"{run_id}.tsv not found in tarball")
    df = _parse_tsv_bytes(raw)
    clonotypes = _clonotypes_from_df(df)
    return SampleRepertoire.from_clonotypes(clonotypes, sample_id=run_id)


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def meta() -> pd.DataFrame:
    return pd.read_csv(META_PATH, sep="\t")


@pytest.fixture(scope="module")
def samples_with_ids(meta) -> tuple[list[SampleRepertoire], list[str]]:
    """Load the first 15 runs that exist in the tarball."""
    with tarfile.open(TARBALL, "r:gz") as tar:
        members = {Path(m.name).stem for m in tar.getmembers() if m.name.endswith(".tsv")}
        run_ids = [r for r in meta["Run"] if r in members][:15]
        assert run_ids, "No matching run TSVs found in tarball"
        samples = [load_srx_sample(tar, rid) for rid in run_ids]
    return samples, run_ids


@pytest.fixture(scope="module")
def samples(samples_with_ids) -> list[SampleRepertoire]:
    return samples_with_ids[0]


@pytest.fixture(scope="module")
def run_ids(samples_with_ids) -> list[str]:
    return samples_with_ids[1]


@pytest.fixture(scope="module")
def first(samples) -> SampleRepertoire:
    return samples[0]


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestSampleRepertoireBasic:
    def test_non_empty(self, first):
        assert len(first.clonotypes) > 0

    def test_has_loci(self, first):
        assert len(first.loci) > 0

    def test_sample_id_set(self, samples, run_ids):
        for sr, rid in zip(samples, run_ids):
            assert sr.sample_id == rid

    def test_locus_keys_are_valid(self, samples):
        for sr in samples:
            for locus in sr.loci:
                assert locus in _ALL_LOCI or locus == "", f"Unexpected locus: {locus!r}"

    def test_all_samples_loaded(self, samples, run_ids):
        assert len(samples) == len(run_ids)


# ---------------------------------------------------------------------------
# Locus coverage analysis
# ---------------------------------------------------------------------------

class TestLocusCoverage:
    def test_loci_present_report(self, samples):
        """Summarise which loci each sample has and warn about missing ones."""
        coverage: dict[str, set[str]] = {sr.sample_id: set(sr.loci.keys()) for sr in samples}
        all_seen = set().union(*coverage.values())

        missing: dict[str, set[str]] = {}
        for sample_id, present in coverage.items():
            absent = all_seen - present
            if absent:
                missing[sample_id] = absent

        if missing:
            lines = [f"  {sid}: missing {sorted(loci)}" for sid, loci in sorted(missing.items())]
            warnings.warn(
                f"Locus coverage is uneven across {len(missing)}/{len(samples)} samples:\n"
                + "\n".join(lines),
                stacklevel=2,
            )
        # Informational — always passes; coverage printed above
        assert len(coverage) == len(samples)

    def test_at_least_two_loci_in_dataset(self, samples):
        """Paired (TRA+TRB or IGH+IGK/IGL) samples should have multiple loci."""
        all_loci = set().union(*(set(sr.loci.keys()) for sr in samples))
        assert len(all_loci) >= 2, f"Expected multiple loci across samples, got {all_loci!r}"

    def test_each_locus_has_clonotypes(self, samples):
        for sr in samples:
            for locus, lr in sr.loci.items():
                assert lr.clonotype_count > 0, (
                    f"Sample {sr.sample_id!r} has empty LocusRepertoire for locus {locus!r}"
                )

    def test_duplicate_count_positive_per_locus(self, samples):
        for sr in samples:
            for locus, lr in sr.loci.items():
                assert lr.duplicate_count > 0, (
                    f"Sample {sr.sample_id!r} locus {locus!r} has zero total reads"
                )


# ---------------------------------------------------------------------------
# LocusRepertoire per-sample checks
# ---------------------------------------------------------------------------

class TestLocusRepertoire:
    def test_clonotype_count_matches_len(self, first):
        for lr in first:
            assert lr.clonotype_count == len(lr.clonotypes)

    def test_locus_field_consistent(self, first):
        for locus, lr in first.loci.items():
            for c in lr.clonotypes:
                if c.locus:
                    assert c.locus == locus

    def test_sort_orders_by_duplicate_count(self, first):
        for lr in first:
            if lr.clonotype_count < 2:
                continue
            lr.sort()
            assert lr.is_sorted
            counts = [c.duplicate_count for c in lr.clonotypes]
            assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# Sorting — SampleRepertoire level
# ---------------------------------------------------------------------------

class TestSampleRepertoireSorting:
    def test_sort_all_loci(self, samples):
        for sr in samples:
            sr.sort()
            assert sr.is_sorted, f"Sample {sr.sample_id!r} not sorted after .sort()"


# ---------------------------------------------------------------------------
# Polars round-trip
# ---------------------------------------------------------------------------

class TestPolarsRoundTrip:
    def test_to_polars_row_count(self, first):
        df = first.to_polars()
        assert df.height == len(first.clonotypes)

    def test_to_polars_has_airr_columns(self, first):
        df = first.to_polars()
        for col in ("junction_aa", "v_gene", "j_gene", "locus"):
            assert col in df.columns

    def test_from_polars_roundtrip(self, first):
        df = first.to_polars()
        restored = SampleRepertoire.from_polars(df, locus_column="locus",
                                                sample_id=first.sample_id)
        assert len(restored.clonotypes) == len(first.clonotypes)
        assert set(restored.loci.keys()) == set(first.loci.keys())

    def test_locus_repertoire_roundtrip(self, first):
        for locus, lr in first.loci.items():
            df = lr.to_polars()
            restored = LocusRepertoire.from_polars(df, locus=locus)
            assert restored.clonotype_count == lr.clonotype_count
            assert restored.locus == locus


# ---------------------------------------------------------------------------
# from_clonotypes grouping
# ---------------------------------------------------------------------------

class TestFromClonotypes:
    def test_groups_by_locus(self, first):
        flat = first.clonotypes
        restored = SampleRepertoire.from_clonotypes(flat, sample_id="roundtrip")
        for locus, lr in first.loci.items():
            assert locus in restored.loci
            assert restored[locus].clonotype_count == lr.clonotype_count

    def test_empty_clonotype_list(self):
        sr = SampleRepertoire.from_clonotypes([])
        assert len(sr.loci) == 0
        assert sr.clonotypes == []


# ---------------------------------------------------------------------------
# Stand-alone LocusRepertoire construction / validation
# ---------------------------------------------------------------------------

class TestLocusRepertoireConstruction:
    def test_locus_mismatch_raises(self):
        c1 = Clonotype(junction_aa="CASSEGF", locus="TRB")
        c2 = Clonotype(junction_aa="CATSEGF", locus="TRA")
        with pytest.raises(ValueError, match="locus"):
            LocusRepertoire([c1, c2], locus="TRB")

    def test_empty_clonotype_locus_is_accepted(self):
        c = Clonotype(junction_aa="CASSEGF", locus="")
        lr = LocusRepertoire([c], locus="TRB")
        assert lr.clonotype_count == 1

    def test_sort_and_is_sorted(self):
        clonotypes = [
            Clonotype(junction_aa="CASSEGF", duplicate_count=1),
            Clonotype(junction_aa="CASSEF",  duplicate_count=5),
            Clonotype(junction_aa="CASSEFG", duplicate_count=3),
        ]
        lr = LocusRepertoire(clonotypes)
        assert not lr.is_sorted
        lr.sort()
        assert lr.is_sorted
        assert [c.duplicate_count for c in lr.clonotypes] == [5, 3, 1]

    def test_counts(self):
        clonotypes = [
            Clonotype(junction_aa="CASSEGF", duplicate_count=10),
            Clonotype(junction_aa="CASSEF",  duplicate_count=5),
        ]
        lr = LocusRepertoire(clonotypes)
        assert lr.duplicate_count == 15
        assert lr.clonotype_count == 2

    def test_backward_compat_properties(self):
        lr = LocusRepertoire(
            [Clonotype(junction_aa="CASSEGF", duplicate_count=7)],
            metadata={"key": "val"},
            gene="TRB",
        )
        assert lr.locus == "TRB"
        assert lr.gene == "TRB"
        assert lr.metadata == {"key": "val"}
        assert lr.number_of_clones == 1
        assert lr.number_of_reads == 7
