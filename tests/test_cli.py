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


def test_embed_clonotypes_filters_non_coding_by_default(tmp_path):
    src = tmp_path / "S.tsv"
    out = tmp_path / "emb.tsv"
    noncoding = [("TRBV10-3*01", "TRBJ2-7*01", "CASSIRS_YEQYF", 3),   # out-of-frame '_'
                 ("TRBV20-1*01", "TRBJ1-2*01", "CSARVSG*YGYTF", 2)]   # stop codon '*'
    _write(src, TRB + noncoding)
    main(["embed", "clonotypes", str(src), "--n-prototypes", "300", "-o", str(out)])

    got = pl.read_csv(out, separator="\t")
    assert got.height == len(TRB)          # both non-coding rows dropped, no crash


def test_embed_clonotypes_no_filter_functional_still_raises_on_underscore(tmp_path):
    # --no-filter-functional is the "I want non-functional receptors" opt-in and reaches the
    # embedder with allow_nonstandard=True -- but '_' crashes seqtree, so it raises regardless.
    src = tmp_path / "S.tsv"
    _write(src, TRB + [("TRBV10-3*01", "TRBJ2-7*01", "CASSIRS_YEQYF", 3)])
    with pytest.raises(ValueError, match="out-of-frame"):
        main(["embed", "clonotypes", str(src), "--n-prototypes", "300", "--no-filter-functional"])


def test_embed_clonotypes_no_filter_functional_embeds_stop_codons(tmp_path):
    # the opt-in does what it says for the one case that is a real receptor category
    src = tmp_path / "S.tsv"
    out = tmp_path / "emb.tsv"
    _write(src, TRB + [("TRBV20-1*01", "TRBJ1-2*01", "CSARVSG*YGYTF", 2)])
    main(["embed", "clonotypes", str(src), "--n-prototypes", "300", "-o", str(out),
          "--no-filter-functional"])
    assert pl.read_csv(out, separator="\t").height == len(TRB) + 1


def test_embed_clonotypes_rejects_a_corrupt_table(tmp_path):
    # an ambiguity code is a damaged file, not a kind of receptor -- the opt-in must not hide it
    src = tmp_path / "S.tsv"
    _write(src, TRB + [("TRBV20-1*01", "TRBJ1-2*01", "CSARVSGXYGYTF", 2)])
    with pytest.raises(ValueError, match="CORRUPT"):
        main(["embed", "clonotypes", str(src), "--n-prototypes", "300",
              "--no-filter-functional"])


def test_embed_repertoires_skips_sample_left_empty_by_filter(tmp_path):
    s1, s2 = tmp_path / "P1.tsv", tmp_path / "P2.tsv"
    out = tmp_path / "phi.tsv"
    _write(s1, TRB)
    _write(s2, [("TRBV10-3*01", "TRBJ2-7*01", "CASSIRS_YEQYF", 3)])   # all non-coding
    main(["embed", "repertoires", str(s1), str(s2), "--n-prototypes", "300",
          "--n-rff", "32", "-o", str(out)])

    got = pl.read_csv(out, separator="\t")
    assert got["sample_id"].to_list() == ["P1"]      # P2 skipped, P1 still embedded


def test_per_locus_mmd_path_splits_on_the_extension(tmp_path):
    """Regression: `--mmd` inserted the locus at the first dot ANYWHERE in the path.

    `str.replace(".", ".TRB.", 1)` turned `./mmd.tsv` into `.TRB./mmd.tsv` -> FileNotFoundError,
    raised only after both loci had been embedded and before `-o` was written, losing the run.
    An extensionless path was worse: a silent no-op, so one locus overwrote the other's matrix
    and the single file claimed to be both.
    """
    from mir.cli import _per_locus_path

    assert _per_locus_path("./mmd.tsv", "TRB") == "mmd.TRB.tsv"
    assert _per_locus_path("out/mmd.tsv", "TRA") == "out/mmd.TRA.tsv"
    assert _per_locus_path("mmdout", "TRB") == "mmdout.TRB"          # no longer a silent no-op
    assert _per_locus_path("../a.b/mmd.parquet", "IGH") == "../a.b/mmd.IGH.parquet"


def test_embed_repertoires_writes_one_mmd_matrix_per_locus(tmp_path):
    """End-to-end: two loci in a dot-containing directory each get their own MMD file."""
    d = tmp_path / "v1.0"
    d.mkdir()
    s1, s2 = d / "P1.tsv", d / "P2.tsv"
    mixed = [TRB[0], TRB[1], ("TRAV1-2*01", "TRAJ33*01", "CAVMDSNYQLIW", 5),
             ("TRAV12-2*01", "TRAJ42*01", "CAVNGGSQGNLIF", 9)]
    _write(s1, mixed)
    _write(s2, mixed)
    main(["embed", "repertoires", str(s1), str(s2), "--n-prototypes", "300",
          "--n-rff", "32", "-o", str(d / "phi.tsv"), "--mmd", str(d / "mmd.tsv")])

    for locus in ("TRB", "TRA"):
        got = pl.read_csv(d / f"mmd.{locus}.tsv", separator="\t")
        assert got.height == 2 and got["sample_id"].to_list() == ["P1", "P2"]


def test_locus_flag_accepts_aliases(tmp_path):
    """Regression: `--locus beta` bypassed normalize_locus_alias and matched zero rows."""
    src = tmp_path / "S.tsv"
    _write(src, TRB)
    out = tmp_path / "emb.tsv"
    main(["embed", "clonotypes", str(src), "--locus", "beta", "--n-prototypes", "300",
          "-o", str(out)])
    assert pl.read_csv(out, separator="\t").height == 4

    with pytest.raises(SystemExit, match="Unknown locus"):
        main(["embed", "clonotypes", str(src), "--locus", "nonsense", "--n-prototypes", "300"])
