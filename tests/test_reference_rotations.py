"""The reference recipe exports raw-feature/RFF rotations with their exact basis."""
from pathlib import Path
import importlib.util
import json

import polars as pl
import pytest
from mir.embedding.prototypes import load_prototypes
from vdjtools.signature.corpus import TablePCA


def test_reference_bundle_is_separate_and_reusable(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'reference_rotations', Path(__file__).parents[1] / 'examples/reference_rotations.py')
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    pool = load_prototypes('human', 'TRB').slice(5000, 80)
    readers = [(f'S{i}', lambda i=i: {'TRB': pool.slice(16*i, 32).with_columns(
        pl.lit('TRB').alias('locus'), pl.Series('duplicate_count', range(1, 33)))})
        for i in range(3)]
    output = tmp_path / 'bundle'
    recipe.build(readers, ['TRB'], output, components=2, n_rff=16, fit_cap=32)
    meta = json.loads((output / 'bundle.json').read_text())
    assert meta['representations'] == ['vdjtools-features', 'mirpy-rff']
    assert all(not p.endswith(('.tsv', '.parquet')) for p in meta['files'])
    assert TablePCA.load(output / 'rff-pca.npz').components.shape == (2, 16)
    assert TablePCA.load(output / 'features-pca.npz').space.startswith('vsig:')
    assert (output / 'basis/TRB.pkl').is_file()
    with pytest.raises(FileExistsError):
        recipe.build(readers, ['TRB'], output)
    with pytest.raises(ValueError, match='unique samples'):
        recipe.build([readers[0], readers[0]], ['TRB'], tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
