"""The reference recipe exports raw-feature/RFF rotations with their exact basis."""
import json
import hashlib
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from mir.embedding.prototypes import load_prototypes
from mir.embedding.reference import build_reference_rotations, _check
from vdjtools.signature.corpus import TablePCA


def test_reference_bundle_is_separate_and_reusable(tmp_path, monkeypatch):
    pool = load_prototypes('human', 'TRB').slice(5000, 80)
    readers = [(f'S{i}', lambda i=i: {'TRB': pool.slice(16*i, 32).with_columns(
        pl.lit('TRB').alias('locus'), pl.Series('duplicate_count', range(1, 33)))})
        for i in range(3)]
    output = tmp_path / 'bundle'
    build_reference_rotations(readers, ['TRB'], output, components=2, n_rff=16, fit_cap=32)
    meta = json.loads((output / 'bundle.json').read_text())
    assert meta['representations'] == ['vdjtools-features', 'mirpy-rff']
    assert all(not p.endswith(('.tsv', '.parquet')) for p in meta['files'])
    assert all(hashlib.sha256((output / p).read_bytes()).hexdigest() == digest
               for p, digest in meta['files'].items())
    assert set(meta['versions']) == {'mirpy', 'vdjtools', 'numpy', 'sklearn'}
    assert TablePCA.load(output / 'rff-pca.npz').components.shape == (2, 16)
    assert TablePCA.load(output / 'features-pca.npz').space.startswith('vsig:')
    assert (output / 'basis/TRB.pkl').is_file()
    from vdjtools.signature import raw_features_cohort
    vocab = json.loads((output / 'feature-vocabulary.json').read_text())
    frozen = raw_features_cohort(readers, vocab=vocab, n_jobs=1, cstar_target=0.8)
    rotation = TablePCA.load(output / 'features-pca.npz')
    assert frozen['embedding_space'][0] == rotation.space
    assert rotation.transform(frozen).height == len(readers)
    monkeypatch.setattr('mir.embedding.reference.fit_sample_spaces',
                        lambda *a, **kw: pytest.fail('PCA refit must reuse its basis'))
    refit = tmp_path / 'refit'
    build_reference_rotations(readers, ['TRB'], refit, n_rff=16,
                              reuse_basis=output / 'basis')
    assert (refit/'basis/TRB.pkl').read_bytes() == (output/'basis/TRB.pkl').read_bytes()
    report = json.loads((refit/'bundle.json').read_text())
    assert report['requested_components'] == 1024
    for name in ('features','rff'):
        saved = TablePCA.load(refit / f'{name}-pca.npz')
        assert saved.components.shape[0] == 3
        assert saved.explained_variance_ratio.shape == (3,)
        assert np.isfinite(saved.explained_variance_ratio).all()
        for k in (256,512,1024):
            assert report['variance'][name]['cumulative'][str(k)] == float(
                saved.explained_variance_ratio[:k].sum())
    with pytest.raises(FileExistsError):
        build_reference_rotations(readers, ['TRB'], output)
    with pytest.raises(ValueError, match='unique samples'):
        build_reference_rotations([readers[0], readers[0]], ['TRB'], tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()

    rotation = TablePCA.load(output / 'rff-pca.npz')
    rows = pl.DataFrame({'sample_id': ['CHECK1', 'CHECK2'],
                         'embedding_space': [rotation.space] * 2,
                         **{c: [0.0, 1.0] for c in rotation.columns}})
    changed = rotation.components.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    replace(rotation, components=changed).save(tmp_path / 'changed.npz')
    with pytest.raises(AssertionError):
        _check(rotation, rows, tmp_path / 'changed.npz')


def test_configured_statistical_reference_carries_its_definition(tmp_path, monkeypatch):
    from vdjtools.signature import feature_definition
    pool = load_prototypes('human', 'TRB').slice(5000, 80)
    readers = [(f'S{i}', lambda i=i: {'TRB': pool.slice(16*i, 32).with_columns(
        pl.lit('TRB').alias('locus'), pl.Series('duplicate_count', range(1, 33)))}) for i in range(3)]
    from vdjtools.cores import available_cores
    budget = min(2, available_cores())
    definition = feature_definition('basic', motifs='imgt11')
    out = tmp_path / 'configured'
    original = TablePCA.fit_features.__func__
    calls = []
    def bounded(cls, samples, **kw):
        calls.append(kw)
        return original(cls, samples, **kw)
    monkeypatch.setattr(TablePCA, 'fit_features', classmethod(bounded))
    import mir.embedding.reference as reference
    dense = reference.raw_features_cohort
    def small_only(items, **kw):
        assert len(items) <= 2, 'configured reference materialized the whole dense cohort'
        return dense(items, **kw)
    monkeypatch.setattr(reference, 'raw_features_cohort', small_only)
    build_reference_rotations(readers, ['TRB'], out, components=2, n_rff=16, fit_cap=32,
                              feature_definition=definition, threads=budget)
    assert len(calls) == 1 and calls[0]['threads'] == budget
    assert json.loads((out / 'bundle.json').read_text())['threads'] == budget
    assert json.loads((out / 'feature-definition.json').read_text()) == definition
    columns = (out / 'feature-columns.txt').read_text().splitlines()
    assert any(':motif_XX_X:' in c for c in columns)
    assert not any(':pchem:' in c for c in columns)
    assert TablePCA.load(out / 'features-pca.npz').space.startswith('vsig:')
