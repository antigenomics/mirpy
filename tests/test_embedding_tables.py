"""Separate raw embedding tables from cohort-level fitted rotations."""
import numpy as np
import polars as pl
import pytest


def frame():
    return pl.DataFrame({'v_call':['TRBV10-3*01','TRBV20-1*01'],
                         'j_call':['TRBJ2-7*01','TRBJ1-2*01'],
                         'junction_aa':['CASSIRSSYEQYF','CSARVSGYYGYTF'],
                         'duplicate_count':[12,4]})


def test_sample_rows_are_raw_weighted_means_without_any_pca(monkeypatch):
    from mir.embedding.table import sample_embeddings
    from mir.embedding.tcremp import TCREmp
    from mir.signature.features import weights
    from sklearn.decomposition import PCA

    def forbidden(*args, **kwargs):
        raise AssertionError('embedding must not fit PCA')

    monkeypatch.setattr(PCA, 'fit', forbidden)
    model = TCREmp.from_defaults('human','TRB',n_prototypes=32,threads=0)
    raw = frame()
    expected = weights(raw['duplicate_count'].to_numpy()) @ model.embed(raw).astype(float)
    calls = []
    def read():
        calls.append(1)
        return {'TRB': raw.clone()}
    out = sample_embeddings([('A',read),('B',read)], {'TRB':model}, n_jobs=1)
    assert calls == [1,1]
    assert out['sample_id'].to_list() == ['A','B']
    np.testing.assert_allclose(out.select(pl.col('^phi:.*$')).to_numpy(),
                               np.vstack([expected,expected]), rtol=0,atol=1e-12)


def test_table_pca_roundtrip_preserves_ids_and_refuses_other_space(tmp_path):
    from mir.embedding.table import TablePCA
    rng = np.random.default_rng(9)
    table = pl.DataFrame({'sample_id':[str(i) for i in range(30)],
                          'embedding_space':['panel-a']*30,
                          **{f'phi:TRB:{i}':rng.normal(size=30) for i in range(12)}})
    table = table.with_columns(pl.when(pl.col('sample_id')=='0').then(None)
                              .otherwise(pl.col('phi:TRB:0')).alias('phi:TRB:0'))
    rotation = TablePCA.fit(table,n_components=5,seed=3)
    expected = rotation.transform(table)
    rotation.save(tmp_path/'rotation.npz')
    got = TablePCA.load(tmp_path/'rotation.npz').transform(table)
    assert got['sample_id'].to_list() == table['sample_id'].to_list()
    np.testing.assert_array_equal(got.select(pl.col('^pc.*$')).to_numpy(),
                                  expected.select(pl.col('^pc.*$')).to_numpy())
    assert np.isfinite(got.select(pl.col('^pc.*$')).to_numpy()).all()
    with pytest.raises(ValueError,match='space'):
        rotation.transform(table.with_columns(pl.lit('panel-b').alias('embedding_space')))
    with pytest.raises(ValueError,match='columns'):
        rotation.transform(table.drop('phi:TRB:1'))


def test_parallel_sample_rows_equal_serial_with_missing_loci():
    from mir.embedding.table import sample_embeddings
    from mir.embedding.tcremp import TCREmp
    models = {loc: TCREmp.from_defaults('human',loc,n_prototypes=8,threads=0)
              for loc in ('TRB','TRA')}
    items = [(str(i), lambda: {'TRB':frame()}) for i in range(5)] + [('empty',lambda:{})]
    serial = sample_embeddings(items,models,n_jobs=1)
    parallel = sample_embeddings(items,models,n_jobs=2)
    assert serial.equals(parallel)
    assert parallel['sample_id'].to_list() == [str(i) for i in range(5)] + ['empty']
    assert parallel['present:TRA'].sum() == 0
    assert parallel['present:TRB'].to_list() == [1]*5 + [0]
    assert np.isnan(parallel.select(pl.col('^phi:TRA:.*$')).to_numpy()).all()


def test_clonotype_stream_bounds_ids_and_atomic_output(tmp_path):
    from mir.embedding.table import clonotype_embeddings
    from mir.embedding.tcremp import TCREmp
    from mir.cli import _write_clonotype_batches
    model = TCREmp.from_defaults('human','TRB',n_prototypes=8,threads=1)
    raw = frame().with_columns(pl.Series('clonotype_id',['001','supplied']))
    batches = list(clonotype_embeddings([('A',{'TRB':raw})],{'TRB':model},chunk=1))
    assert [b.height for b in batches] == [1,1]
    assert pl.concat(batches)['clonotype_id'].to_list() == ['001','supplied']
    assert pl.concat(batches)['sample_id'].to_list() == ['A','A']
    path = tmp_path/'clones.tsv'
    path.write_text('previous result')
    def fail():
        yield batches[0]
        raise RuntimeError('input failure')
    with pytest.raises(RuntimeError,match='failure'):
        _write_clonotype_batches(fail(),str(path))
    assert path.read_text() == 'previous result'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['clones.tsv']
    pytest.importorskip('pyarrow')
    _write_clonotype_batches(iter(batches),str(tmp_path/'clones.parquet'))
    assert pl.read_parquet(tmp_path/'clones.parquet').equals(pl.concat(batches))


def test_rotation_uses_training_fill_for_new_rows_and_handles_infinity():
    from mir.embedding.table import TablePCA
    table = pl.DataFrame({'sample_id':['a','b','c'], 'embedding_space':['fixed']*3,
                          'phi:TRB:0':[1.,3.,float('inf')],
                          'phi:TRA:0':[float('nan')]*3, 'present:TRA':[0]*3})
    rotation = TablePCA.fit(table,n_components=2)
    np.testing.assert_array_equal(rotation.fill,[2.,0.,0.])
    new = table.with_columns(pl.lit(float('nan')).alias('phi:TRB:0'))
    assert np.isfinite(rotation.transform(new).select(pl.col('^pc.*$')).to_numpy()).all()
