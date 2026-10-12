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
