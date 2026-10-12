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
    main(["embed", "clonotypes", str(src), "--n-prototypes", "300", "-o", str(out)])

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
          "-o", str(out)])

    got = pl.read_csv(out, separator="\t")
    assert got.height == 2                                  # one Φ(S) per sample
    assert got["sample_id"].to_list() == ["P1", "P2"]       # id = filename stem
    assert got["present:TRB"].to_list() == [1, 1]
    assert any(c.startswith("phi") for c in got.columns)


def test_clonotype_list_and_multiple_loci_preserve_both_ids(tmp_path):
    s1, s2 = tmp_path/'A.tsv', tmp_path/'B.tsv'
    _write(s1, [TRB[0], ('TRAV1-2*01','TRAJ33*01','CAVMDSNYQLIW',5)])
    _write(s2, TRB)
    out = tmp_path/'clones.tsv'
    main(['embed','clonotypes',str(s1),str(s2),'--n-prototypes','32','--threads','1','-o',str(out)])
    got = pl.read_csv(out, separator='\t')
    assert got.height == 6
    assert set(got['sample_id']) == {'A','B'}
    assert got.select('sample_id','clonotype_id').unique().height == 6
    assert set(got['locus']) == {'TRA','TRB'}


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
          "-o", str(out)])

    got = pl.read_csv(out, separator="\t")
    assert got["sample_id"].to_list() == ["P1", "P2"]
    assert got["present:TRB"].to_list() == [1, 0]  # immune desert retained as missing


def test_rff_sample_table_combines_loci_and_pca_is_separate(tmp_path):
    s1, s2 = tmp_path/'A.tsv', tmp_path/'B.tsv'
    _write(s1, [TRB[0], ('TRAV1-2*01','TRAJ33*01','CAVMDSNYQLIW',5),
                ('TRAV1-2*01','TRAJ33*01','CAVADSNYQLIW',3)])
    _write(s2, TRB)
    out, pcs, again, rot = (tmp_path/p for p in ('raw.tsv','pcs.tsv','again.tsv','rotation.npz'))
    main(['embed','repertoires',str(s1),str(s2),'--n-prototypes','32','--threads','1','-o',str(out)])
    got = pl.read_csv(out,separator='\t')
    assert got.height == 2 and got['sample_id'].to_list() == ['A','B']
    assert got['present:TRA'].to_list() == [1,0]
    main(['pca',str(out),'--fit',str(rot),'--n-components','2','-o',str(pcs)])
    main(['pca',str(out),'--rotation',str(rot),'-o',str(again)])
    assert pcs.read_text() == again.read_text()


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


def test_cli_preserves_numeric_looking_airr_clonotype_ids(tmp_path):
    src = tmp_path / 'S.tsv'
    _write(src, TRB[:2])
    raw = pl.read_csv(src, separator='\t').with_columns(pl.Series('clonotype_id',['001','002']))
    raw.write_csv(src, separator='\t')
    out = tmp_path / 'clones.tsv'
    main(['embed','clonotypes',str(src),'--n-prototypes','8','--threads','1','-o',str(out)])
    got = pl.read_csv(out, separator='\t', schema_overrides={'clonotype_id':pl.String})
    assert got['clonotype_id'].to_list() == ['001','002']


def test_inline_and_control_cohort_pca_reuse_same_rff_basis(tmp_path):
    import numpy as np
    inputs = [tmp_path/f'{i:03}.tsv' for i in range(4)]
    for i,path in enumerate(inputs):
        _write(path,[(v,j,seq,n*(i+1) if k == i else n) for k,(v,j,seq,n) in enumerate(TRB)])
    raw, inline, rot = (tmp_path/n for n in ('rff.tsv','inline.tsv','pca.npz'))
    basis = tmp_path/'basis'
    args = ['embed','repertoires',*(str(p) for p in inputs),'--locus','TRB',
            '--n-prototypes','32','--n-rff','16','--n-components','3','--threads','1']
    main([*args,'--save-basis',str(basis),'--rff-output',str(raw),'--pca','2',
          '--save-rotation',str(rot),'-o',str(inline)])
    saved, control = tmp_path/'saved.tsv',tmp_path/'control.tsv'
    main([*args,'--basis',str(basis),'--rotation',str(rot),'-o',str(saved)])
    main([*args,'--basis',str(basis),'--pca-cohort',str(raw),'--pca','2','-o',str(control)])
    assert inline.read_bytes() == saved.read_bytes()
    a = pl.read_csv(inline,separator='\t',schema_overrides={'sample_id':pl.String})
    b = pl.read_csv(control,separator='\t',schema_overrides={'sample_id':pl.String})
    np.testing.assert_allclose(a.select('pc0','pc1'),b.select('pc0','pc1'),atol=1e-12)
    assert a['sample_id'].to_list() == ['000','001','002','003']
    assert any(c.startswith('diversity:') for c in a.columns)


def test_clonotype_pca_and_mmd_options_remain_available(tmp_path):
    from mir.cli import main
    from mir.embedding.tcremp import TCREmp
    m = TCREmp.from_defaults('human','TRB',n_prototypes=16,threads=1)
    for i in range(2):
        pl.DataFrame({'v_call':m._proto_v,'j_call':m._proto_j,
                       'junction_aa':m._proto_junction,
                       'duplicate_count':[i+1]*len(m._proto_v)}).write_csv(
            tmp_path/f's{i}.tsv',separator='\t')
    cl = tmp_path/'cl.tsv'
    main(['embed','clonotypes',str(tmp_path/'s0.tsv'),'--locus','TRB',
          '--n-prototypes','16','--pca','2','--threads','1','-o',str(cl)])
    assert {'e0','e1','sample_id','clonotype_id'} <= set(pl.read_csv(cl,separator='\t').columns)
    mmd = tmp_path/'mmd.tsv'
    main(['embed','repertoires',str(tmp_path/'s0.tsv'),str(tmp_path/'s1.tsv'),
          '--locus','TRB','--n-prototypes','16','--n-components','8',
          '--n-rff','16','--threads','1','--mmd',str(mmd),'-o',str(tmp_path/'rff.tsv')])
    assert pl.read_csv(mmd,separator='\t').height == 2


def test_embedding_reader_trims_columns_without_collapsing_nt_clones_or_shm(tmp_path):
    from mir.cli import _read_sample
    from mir.embedding.tcremp import TCREmp
    m = TCREmp.from_defaults('human','IGH',n_prototypes=16,threads=1)
    df = pl.DataFrame({'v_call':[m._proto_v[0]]*2,'j_call':[m._proto_j[0]]*2,
                       'junction_aa':[m._proto_junction[0]]*2,
                       'junction_nt':['TGTGCC','TGCGCC'],'duplicate_count':[3,7],
                       'v_mutations':['A23V','S31N'],'locus':['IGH']*2,
                       'discarded_note':['x'*1000]*2})
    path = tmp_path/'sample.tsv'
    df.write_csv(path,separator='\t')
    full = _read_sample([str(path)])['IGH']
    trimmed = _read_sample([str(path)],embedding_only=True)['IGH']
    assert 'frequency' not in trimmed.columns
    assert trimmed.height == 2
    assert trimmed['junction_nt'].n_unique() == 2
    assert trimmed['duplicate_count'].to_list() == [3,7]
    assert trimmed['v_mutations'].to_list() == ['A23V','S31N']
    import numpy as np
    np.testing.assert_array_equal(m.embed(trimmed),m.embed(full))


def test_repertoire_basis_preparation_obeys_serial_cli_budget(tmp_path,monkeypatch):
    import mir.embedding.table as tables
    from vdjtools.cores import kernel_threads
    original = tables.fit_sample_spaces
    def checked(*args,**kwargs):
        assert kernel_threads(0) == 1
        return original(*args,**kwargs)
    monkeypatch.setattr(tables,'fit_sample_spaces',checked)
    path = tmp_path/'sample.tsv'
    _write(path,TRB)
    main(['embed','repertoires',str(path),'--locus','TRB','--n-prototypes','16',
          '--n-rff','16','--threads','1','-o',str(tmp_path/'rff.tsv')])
