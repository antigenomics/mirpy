"""CLI smoke tests — the two `mir embed` commands on tiny bundled-geometry frames."""

import polars as pl
import pytest

from mir.cli import main


def _write(path, rows, loci=None):
    """Write a tiny AIRR TSV. `rows` = list of (v, j, junction, count)."""
    df = pl.DataFrame(
        {"v_call": [r[0] for r in rows], "j_call": [r[1] for r in rows],
         "junction_aa": [r[2] for r in rows], "duplicate_count": [r[3] for r in rows]}
    )
    df.write_csv(path, separator="\t")


TRB = [
    ("TRBV10-3*01", "TRBJ2-7*01", "CASSIRSSYEQYF", 120),
    ("TRBV20-1*01", "TRBJ1-2*01", "CSARVSGYYGYTF", 40),
    ("TRBV28*01", "TRBJ2-1*01", "CASSLGQAYEQFF", 12),
    ("TRBV19*01", "TRBJ2-3*01", "CASSISGGADTQYF", 7),
]


def test_embed_clonotypes_writes_embedding_table(tmp_path):
    src = tmp_path / "S.tsv"
    out = tmp_path / "emb.tsv"
    _write(src, TRB)
    main(["embed", "clonotypes", str(src), "--n-prototypes", "300", "--pca", "3", "-o", str(out)])

    got = pl.read_csv(out, separator="\t")
    assert got.height == 4                                  # one row per clonotype
    assert {"junction_aa", "v_call", "j_call", "e0", "e1", "e2"} <= set(got.columns)
    assert got.select(pl.col("e0")).dtypes[0].is_numeric()


def test_embed_repertoires_one_row_per_sample(tmp_path):
    s1, s2 = tmp_path / "P1.tsv", tmp_path / "P2.tsv"
    out = tmp_path / "phi.tsv"
    _write(s1, TRB)
    _write(s2, TRB[:3])
    main(["embed", "repertoires", str(s1), str(s2), "--n-prototypes", "300",
          "--n-rff", "32", "-o", str(out)])

    got = pl.read_csv(out, separator="\t")
    assert got.height == 2                                  # one Φ(S) per sample
    assert got["sample_id"].to_list() == ["P1", "P2"]       # id = filename stem
    assert got["locus"].unique().to_list() == ["TRB"]
    assert any(c.startswith("phi") for c in got.columns)


def test_multiple_loci_without_flag_errors(tmp_path):
    src = tmp_path / "mixed.tsv"
    _write(src, [TRB[0], ("TRAV1-2*01", "TRAJ33*01", "CAVMDSNYQLIW", 5)])
    with pytest.raises(SystemExit):
        main(["embed", "clonotypes", str(src), "--n-prototypes", "300"])
