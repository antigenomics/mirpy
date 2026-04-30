"""Tests for repertoire serialization (pickle, polars) and RepertoireDataset I/O.

All file writes go to pytest tmp_path; nothing persists on disk.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from mir.common.parser import VDJtoolsParser
from mir.common.repertoire import LocusRepertoire, SampleRepertoire
from mir.common.repertoire_dataset import RepertoireDataset

ASSETS = Path(__file__).parent / "assets"
_METADATA = "toy_dataset_metadata.tsv"


def _dataset(**kw) -> RepertoireDataset:
    return RepertoireDataset.from_folder_polars(
        ASSETS, parser=VDJtoolsParser(), metadata_file=_METADATA,
        file_name_column="file_name", sample_id_column="sample_id",
        metadata_sep="\t", skip_missing_files=False, n_workers=2, progress=False, **kw,
    )


@pytest.fixture(scope="module")
def dataset() -> RepertoireDataset:
    return _dataset()


@pytest.fixture(scope="module")
def locus_rep(dataset) -> LocusRepertoire:
    return dataset.samples["s1"].loci["TRB"]


def test_load_samples_and_metadata(dataset):
    assert set(dataset.samples) == {"s1", "s2"}
    assert dataset.metadata["s1"]["batch_id"] == "batch_A"
    assert dataset.metadata["s2"]["batch_id"] == "batch_B"


def test_load_min_duplicate_count():
    ds = _dataset(min_duplicate_count=4_000)
    assert "s2" in ds.samples and "s1" not in ds.samples


def test_load_d_gene_normalised(dataset):
    dots = [c for srep in dataset.samples.values() for lr in srep.loci.values()
            for c in lr.clonotypes if c.d_gene == "."]
    assert not dots, "d_gene not normalised during dataset load"


def test_locus_pickle_roundtrip(tmp_path, locus_rep):
    p = tmp_path / "lr.pkl"
    locus_rep.to_pickle(p)
    rt = LocusRepertoire.from_pickle(p)
    assert rt.locus == locus_rep.locus
    assert len(rt.clonotypes) == len(locus_rep.clonotypes)
    assert (sum(c.duplicate_count for c in rt.clonotypes) ==
            sum(c.duplicate_count for c in locus_rep.clonotypes))


def test_locus_parquet_roundtrip(tmp_path, locus_rep):
    p = tmp_path / "lr.parquet"
    locus_rep.write_polars(p, format="parquet")
    rt = LocusRepertoire.read_polars(p, locus="TRB", repertoire_id="s1")
    assert sorted(c.v_gene for c in rt.clonotypes) == sorted(c.v_gene for c in locus_rep.clonotypes)


def test_sample_pickle_roundtrip(tmp_path, dataset):
    srep = dataset.samples["s1"]
    p = tmp_path / "srep.pkl"
    srep.to_pickle(p)
    rt = SampleRepertoire.from_pickle(p)
    assert set(rt.loci) == set(srep.loci)
    assert len(rt.loci["TRB"].clonotypes) == len(srep.loci["TRB"].clonotypes)


def test_dataset_pickle_roundtrip(tmp_path, dataset):
    p = tmp_path / "ds.pkl"
    dataset.to_pickle(p)
    rt = RepertoireDataset.from_pickle(p)
    assert set(rt.samples) == set(dataset.samples)
    for sid in dataset.samples:
        assert rt.metadata[sid]["batch_id"] == dataset.metadata[sid]["batch_id"]


def test_dataset_write_folder_tsv(tmp_path, dataset):
    meta_path = dataset.write_folder(tmp_path / "out", format="tsv")
    df = pd.read_csv(meta_path, sep="\t")
    assert {"file_name", "sample_id"}.issubset(df.columns)


def test_dataset_write_folder_parquet(tmp_path, dataset):
    out = tmp_path / "parquet"
    meta_path = dataset.write_folder(out, format="parquet")
    df = pd.read_csv(meta_path, sep="\t")
    for _, row in df.iterrows():
        assert (out / row["file_name"]).exists()
