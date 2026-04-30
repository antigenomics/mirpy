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


def _clone_key(c):
    return (
        c.sequence_id,
        c.locus,
        c.duplicate_count,
        c.junction,
        c.junction_aa,
        c.v_gene,
        c.d_gene,
        c.j_gene,
        c.c_gene,
        c.v_sequence_end,
        c.d_sequence_start,
        c.d_sequence_end,
        c.j_sequence_start,
    )


def _assert_locus_equal(a: LocusRepertoire, b: LocusRepertoire):
    assert a.locus == b.locus
    assert sorted(_clone_key(c) for c in a.clonotypes) == sorted(_clone_key(c) for c in b.clonotypes)


def _assert_sample_equal(a: SampleRepertoire, b: SampleRepertoire):
    assert set(a.loci) == set(b.loci)
    for locus in a.loci:
        _assert_locus_equal(a.loci[locus], b.loci[locus])


def _assert_dataset_equal(a: RepertoireDataset, b: RepertoireDataset):
    assert set(a.samples) == set(b.samples)
    for sid in a.samples:
        _assert_sample_equal(a.samples[sid], b.samples[sid])


def test_locus_tsv_roundtrip(tmp_path, locus_rep):
    p = tmp_path / "lr.tsv"
    locus_rep.to_tsv(p)
    rt = LocusRepertoire.from_tsv(p, locus=locus_rep.locus)
    _assert_locus_equal(locus_rep, rt)


def test_locus_tsvgz_roundtrip(tmp_path, locus_rep):
    p = tmp_path / "lr.tsv.gz"
    locus_rep.to_tsv(p, gzip_output=True)
    rt = LocusRepertoire.from_tsv(p, locus=locus_rep.locus)
    _assert_locus_equal(locus_rep, rt)


def test_locus_parquet_helper_roundtrip(tmp_path, locus_rep):
    p = tmp_path / "lr_helper.parquet"
    locus_rep.to_parquet(p)
    rt = LocusRepertoire.from_parquet(p, locus=locus_rep.locus)
    _assert_locus_equal(locus_rep, rt)


def test_sample_tsv_single_roundtrip(tmp_path, dataset):
    srep = dataset.samples["s1"]
    p = tmp_path / "s1.tsv"
    srep.to_tsv(p)
    rt = SampleRepertoire.from_tsv(p, sample_id="s1")
    _assert_sample_equal(srep, rt)


def test_sample_tsv_split_roundtrip(tmp_path, dataset):
    srep = dataset.samples["s1"]
    out = tmp_path / "s1_split_tsv"
    srep.to_tsv(out, split_loci=True)
    rt = SampleRepertoire.from_tsv(out, split_loci=True, sample_id="s1")
    _assert_sample_equal(srep, rt)


def test_sample_parquet_single_roundtrip(tmp_path, dataset):
    srep = dataset.samples["s1"]
    p = tmp_path / "s1.parquet"
    srep.to_parquet(p)
    rt = SampleRepertoire.from_parquet(p, sample_id="s1")
    _assert_sample_equal(srep, rt)


def test_sample_parquet_split_roundtrip(tmp_path, dataset):
    srep = dataset.samples["s1"]
    out = tmp_path / "s1_split_parquet"
    srep.to_parquet(out, split_loci=True)
    rt = SampleRepertoire.from_parquet(out, split_loci=True, sample_id="s1")
    _assert_sample_equal(srep, rt)


def test_dataset_tsv_per_sample_locus_roundtrip(tmp_path, dataset):
    out = tmp_path / "dataset_tsv_split"
    dataset.to_tsv(out, layout="per_sample_locus")
    rt = RepertoireDataset.from_tsv(out, layout="per_sample_locus", n_workers=2)
    _assert_dataset_equal(dataset, rt)


def test_dataset_parquet_per_sample_locus_roundtrip(tmp_path, dataset):
    out = tmp_path / "dataset_pq_split"
    dataset.to_parquet(out, layout="per_sample_locus")
    rt = RepertoireDataset.from_parquet(out, layout="per_sample_locus", n_workers=2)
    _assert_dataset_equal(dataset, rt)


def test_dataset_parquet_single_file_roundtrip(tmp_path, dataset):
    out = tmp_path / "dataset_pq_single"
    dataset.to_parquet(out, layout="single_file", data_file="all.parquet")
    rt = RepertoireDataset.from_parquet(
        out,
        layout="single_file",
        data_file="all.parquet",
        n_workers=2,
    )
    _assert_dataset_equal(dataset, rt)


def test_dataset_tsv_single_file_roundtrip(tmp_path, dataset):
    out = tmp_path / "dataset_tsv_single"
    dataset.to_tsv(out, layout="single_file", data_file="all.tsv")
    rt = RepertoireDataset.from_tsv(
        out,
        layout="single_file",
        data_file="all.tsv",
        n_workers=2,
    )
    _assert_dataset_equal(dataset, rt)


def _write_min_vdjtools(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)


def test_from_folder_polars_merges_multi_file_sample_and_keeps_metadata(tmp_path):
    # Same sample_id appears in two metadata rows (TRA/TRB files).
    _write_min_vdjtools(
        tmp_path / "s_multi_tra.tsv",
        [
            {
                "count": 120,
                "cdr3nt": "TGTGCCAGCAGC",
                "cdr3aa": "CASS",
                "v": "TRAV1-2*01",
                "d": ".",
                "j": "TRAJ33*01",
                "locus": "TRA",
            }
        ],
    )
    _write_min_vdjtools(
        tmp_path / "s_multi_trb.tsv",
        [
            {
                "count": 180,
                "cdr3nt": "TGTGCCAGCAGCTAG",
                "cdr3aa": "CASSL",
                "v": "TRBV5-1*01",
                "d": "TRBD1*01",
                "j": "TRBJ2-7*01",
                "locus": "TRB",
            }
        ],
    )
    _write_min_vdjtools(
        tmp_path / "s_single_trb.tsv",
        [
            {
                "count": 200,
                "cdr3nt": "TGTGCCAGCAGCG",
                "cdr3aa": "CASSG",
                "v": "TRBV6-1*01",
                "d": "TRBD2*01",
                "j": "TRBJ1-2*01",
                "locus": "TRB",
            }
        ],
    )

    meta = pd.DataFrame(
        [
            {"sample_id": "s_multi", "file_name": "s_multi_tra.tsv", "batch_id": "batch_X", "locus": "TRA"},
            {"sample_id": "s_multi", "file_name": "s_multi_trb.tsv", "batch_id": "batch_X", "locus": "TRB"},
            {"sample_id": "s_single", "file_name": "s_single_trb.tsv", "batch_id": "batch_Y", "locus": "TRB"},
        ]
    )
    meta.to_csv(tmp_path / "metadata.tsv", sep="\t", index=False)

    ds = RepertoireDataset.from_folder_polars(
        tmp_path,
        parser=VDJtoolsParser(),
        metadata_file="metadata.tsv",
        file_name_column="file_name",
        sample_id_column="sample_id",
        metadata_sep="\t",
        skip_missing_files=False,
        progress=False,
        n_workers=2,
    )

    assert set(ds.samples) == {"s_multi", "s_single"}
    assert set(ds.samples["s_multi"].loci) == {"TRA", "TRB"}
    assert ds.metadata["s_multi"]["batch_id"] == "batch_X"
    assert ds.samples["s_multi"].sample_metadata["batch_id"] == "batch_X"
    assert ds.metadata["s_single"]["batch_id"] == "batch_Y"


def test_from_folder_polars_min_duplicate_count_uses_combined_sample_files(tmp_path):
    # Combined duplicates across TRA+TRB should be used for sample filtering.
    _write_min_vdjtools(
        tmp_path / "a_tra.tsv",
        [{"count": 60, "cdr3nt": "AAA", "cdr3aa": "CAA", "v": "TRAV1-2*01", "d": ".", "j": "TRAJ33*01", "locus": "TRA"}],
    )
    _write_min_vdjtools(
        tmp_path / "a_trb.tsv",
        [{"count": 70, "cdr3nt": "BBB", "cdr3aa": "CBB", "v": "TRBV5-1*01", "d": "TRBD1*01", "j": "TRBJ2-7*01", "locus": "TRB"}],
    )
    _write_min_vdjtools(
        tmp_path / "b_trb.tsv",
        [{"count": 100, "cdr3nt": "CCC", "cdr3aa": "CCC", "v": "TRBV6-1*01", "d": "TRBD2*01", "j": "TRBJ1-2*01", "locus": "TRB"}],
    )

    pd.DataFrame(
        [
            {"sample_id": "a", "file_name": "a_tra.tsv", "batch_id": "b1", "locus": "TRA"},
            {"sample_id": "a", "file_name": "a_trb.tsv", "batch_id": "b1", "locus": "TRB"},
            {"sample_id": "b", "file_name": "b_trb.tsv", "batch_id": "b2", "locus": "TRB"},
        ]
    ).to_csv(tmp_path / "metadata.tsv", sep="\t", index=False)

    ds = RepertoireDataset.from_folder_polars(
        tmp_path,
        parser=VDJtoolsParser(),
        metadata_file="metadata.tsv",
        file_name_column="file_name",
        sample_id_column="sample_id",
        metadata_sep="\t",
        skip_missing_files=False,
        min_duplicate_count=120,
        progress=False,
        n_workers=2,
    )

    # sample 'a' must be retained because 60 + 70 >= 120.
    assert "a" in ds.samples
    assert "b" not in ds.samples
