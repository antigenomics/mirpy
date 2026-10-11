"""Mirpy native entry points share the allocation budget with vdjtools."""
import numpy as np
import pytest


def test_embedding_and_distances_respect_restricted_allocation(monkeypatch):
    from mir.embedding import TCREmp
    from mir.distances.junction import junction_distance_matrix
    monkeypatch.setattr('vdjtools.cores.available_cores', lambda: 2)
    model = TCREmp.from_defaults('human', 'TRB', threads=0)
    assert model.threads == 0
    from vdjtools.cores import serial_kernels
    import polars as pl
    frame = pl.DataFrame({"v_call":["TRBV5-1*01"], "j_call":["TRBJ2-7*01"],
                          "junction_aa":["CASSLGQETQYF"]})
    expected = model.embed(frame)
    with serial_kernels():
        np.testing.assert_array_equal(model.embed(frame), expected)
    with pytest.raises(ValueError):
        TCREmp.from_defaults('human', 'TRB', threads=3)
    seqs = ['CASSLGQETQYF', 'CASSLGQETQFF']
    np.testing.assert_array_equal(junction_distance_matrix(seqs, seqs, threads=0),
                                  junction_distance_matrix(seqs, seqs, threads=1))
    with pytest.raises(ValueError):
        junction_distance_matrix(seqs, seqs, threads=3)


def test_density_rejects_excessive_threads(monkeypatch):
    from mir.density import neighbor_enrichment
    monkeypatch.setattr('vdjtools.cores.available_cores', lambda: 2)
    x = np.arange(60, dtype=float).reshape(20, 3)
    with pytest.raises(ValueError):
        neighbor_enrichment(x, x, radius=1, threads=3)


def test_ann_fresh_runtime_owns_requested_budget():
    pytest.importorskip('pynndescent')
    import subprocess
    import sys
    code = """import mir
import vdjtools.cores as cores
cores.available_cores = lambda: 2
import numpy as np
from mir.density import neighbor_enrichment
rng = np.random.default_rng(42)
x = rng.normal(size=(80, 4))
bg = rng.normal(size=(200, 4))
result = neighbor_enrichment(x, bg, radius=0.3, backend='ann', threads=0)
from numba import config
assert config.NUMBA_NUM_THREADS == 2
assert result.n_bg.shape == (80,)
try:
    neighbor_enrichment(x, bg, radius=0.3, backend='ann', threads=1)
except RuntimeError as e:
    assert 'fresh interpreter' in str(e)
else:
    raise AssertionError('conflicting persistent ANN budget accepted')
"""
    subprocess.run([sys.executable, '-c', code], check=True, timeout=120)


def test_cluster_accepts_sklearn_none_budget():
    from mir.bench.metrics import cluster
    labels = cluster(np.arange(20, dtype=float).reshape(10, 2), eps=4, n_jobs=None)
    assert labels.shape == (10,)


def test_torch_initialized_interop_pool_has_actionable_error():
    pytest.importorskip('torch')
    import subprocess
    import sys
    code = """import torch
torch.set_num_interop_threads(2)
from mir.ml.train import pick_device
try:
    pick_device('cpu')
except RuntimeError as e:
    assert 'fresh interpreter' in str(e)
else:
    raise AssertionError('conflicting one-shot Torch pool accepted')
"""
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


@pytest.mark.parametrize('jobs,expected', [(-1, 4), (-2, 3), (-4, 1), (-10, 1), (None, 1)])
def test_cluster_relative_worker_budget(monkeypatch, jobs, expected):
    import mir.bench.metrics as metrics
    monkeypatch.setattr('vdjtools.cores.available_cores', lambda: 4)
    class Estimator:
        def __init__(self, **kw):
            assert kw['n_jobs'] == expected
        def fit_predict(self, X):
            return np.zeros(len(X), dtype=int)
    monkeypatch.setattr(metrics, 'DBSCAN', Estimator)
    assert metrics.cluster(np.zeros((5, 3)), eps=1, n_jobs=jobs).shape == (5,)
