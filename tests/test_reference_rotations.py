"""The reference recipe exports raw-feature/RFF rotations with their exact basis."""
import json
import hashlib

import polars as pl
import pytest
from mir.embedding.prototypes import load_prototypes
from mir.embedding.reference import build_reference_rotations
from vdjtools.signature.corpus import TablePCA


def test_reference_bundle_is_separate_and_reusable(tmp_path):
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
    with pytest.raises(FileExistsError):
        build_reference_rotations(readers, ['TRB'], output)
    with pytest.raises(ValueError, match='unique samples'):
        build_reference_rotations([readers[0], readers[0]], ['TRB'], tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
