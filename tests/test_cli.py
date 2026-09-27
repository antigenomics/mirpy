"""CLI smoke tests — `mir embed`, and the input-free `mir signature` introspection flags."""

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


# `--no-filter-functional` was removed in 3.12.0: a guard you can switch off is not a guard
# against something that fails silently, and a stop codon is IN seqtree's alphabet, so an
# unfiltered frame embeds to a finite, meaningless distance rather than crashing. The three tests
# that used to assert the opt-in's behaviour now assert that it is refused and says where to go.


@pytest.mark.parametrize("extra", [[], ["--n-prototypes", "300"]])
def test_no_filter_functional_is_refused_and_points_at_vdjtools(tmp_path, extra):
    src = tmp_path / "S.tsv"
    _write(src, TRB + [("TRBV20-1*01", "TRBJ1-2*01", "CSARVSG*YGYTF", 2)])
    with pytest.raises(SystemExit, match="vdjtools filter"):
        main(["embed", "clonotypes", str(src), *extra, "--no-filter-functional"])


def test_a_corrupt_table_raises_on_the_default_path(tmp_path):
    """An ambiguity code is a damaged file, not a kind of receptor.

    It has to raise on the path people actually use, which is now the only path -- previously
    this was only asserted behind the removed opt-in, so nothing covered the default.
    """
    src = tmp_path / "S.tsv"
    _write(src, TRB + [("TRBV20-1*01", "TRBJ1-2*01", "CSARVSGXYGYTF", 2)])
    with pytest.raises(ValueError, match="unparseable value"):
        main(["embed", "clonotypes", str(src), "--n-prototypes", "300"])


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


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    """One small artifact, built once: these tests are about the command, not the fit."""
    from mir.signature import synthesize

    art, _ = synthesize("memory", loci=("TRG",), n_samples=24, size=80, seed=4, n_components=4)
    return str(art.save(tmp_path_factory.mktemp("corpus") / "tiny"))


def test_signature_requires_a_corpus():
    """No default: a signature is comparable only to one rotated through the same corpus."""
    with pytest.raises(SystemExit):
        main(["signature", "nope.tsv"])


def test_describe_names_exactly_the_columns_this_invocation_emits(tmp_path, tiny_corpus):
    """The one output whose entire job is to be right about the width.

    It used to print all 688 columns while the command emitted 528 -- naming columns you would not
    get. So it is resolved against --corpus and --components, and compared here against the real
    emitted header rather than against a layout constant.
    """
    from mir.signature import Corpus

    art = Corpus.load(tiny_corpus)
    for extra, want_k in (([], art.k["TRG"]), (["--components", "2"], 2)):
        out = tmp_path / f"d{len(extra)}.tsv"
        main(["signature", "--corpus", tiny_corpus, "--describe", "-o", str(out), *extra])
        rows = out.read_text().strip().split("\n")
        assert rows[0].split("\t")[:2] == ["column", "block"]
        cols = [r.split("\t")[0] for r in rows[1:]]
        assert cols == art.columns(want_k), extra
        assert not [c for c in cols if c.startswith("vsig:")], "vsig is vdjtools' half"
        assert sum(":pc:" in c for c in cols) == want_k


def test_describe_and_emit_agree_on_the_header(tmp_path, tiny_corpus):
    src = tmp_path / "S1.TRG.tsv"
    src.write_text("junction_aa\tv_call\tj_call\tduplicate_count\n"
                   + "".join(f"CASS{'ACDEFGHIKLMNPQRSTVWY'[i % 20] * 3}YW\tTRGV9\tTRGJ1\t{i + 1}\n"
                             for i in range(40)))
    desc, emit = tmp_path / "d.tsv", tmp_path / "e.tsv"
    main(["signature", "--corpus", tiny_corpus, "--describe", "-o", str(desc)])
    main(["signature", "--corpus", tiny_corpus, str(src), "-o", str(emit)])
    described = [r.split("\t")[0] for r in desc.read_text().strip().split("\n")[1:]]
    emitted = [c for c in emit.read_text().split("\n")[0].split("\t") if c != "sample_id"]
    assert described == emitted


@pytest.mark.parametrize("flag", [
    ["--tier", "core"], ["--preset", "classify"], ["--channels"], ["--standardize", "none"],
    ["--scale", "blood"], ["--clip", "8"], ["--squash", "soft"], ["--on-unscaled", "hole"],
    ["--threads", "4"],
])
def test_every_removed_flag_is_rejected(tiny_corpus, flag):
    with pytest.raises(SystemExit):
        main(["signature", "--corpus", tiny_corpus, "x.tsv", *flag])


def test_the_presets_command_is_gone():
    with pytest.raises(SystemExit):
        main(["presets"])
