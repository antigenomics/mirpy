"""Candidate runtime contract; published vdjtools5.9.0 lacks phase inheritance.

Small synthetic phases verify the coordinated upstream runtime candidate.
"""
import mir  # noqa: F401
import numpy as np
import polars as pl
import pytest
from types import SimpleNamespace
from vdjtools.cores import numerical_kernels, numerical_threads, serial_kernels


@pytest.mark.parametrize("serial", [False, True])
def test_pca_phases_inherit_budget(monkeypatch, serial):
    from sklearn.decomposition import PCA
    from mir.embedding.pca import pca_denoise
    from mir.repertoire import _fit_repertoire_coordinates
    from mir.cohort import _identity_matrix
    from mir import density
    from contextlib import nullcontext
    expected = 1 if serial else numerical_threads(0)
    calls = []
    original = PCA.fit
    def fit(self, *args, **kwargs):
        calls.append(numerical_threads(0))
        assert calls[-1] == expected
        return original(self, *args, **kwargs)
    monkeypatch.setattr(PCA, 'fit', fit)
    for method in ('fit_transform', 'transform'):
        original_method = getattr(PCA, method)
        def checked(self, *args, _method=original_method, **kwargs):
            assert numerical_threads(0) == expected
            return _method(self, *args, **kwargs)
        monkeypatch.setattr(PCA, method, checked)
    X = np.random.default_rng(2).normal(size=(12, 6)).astype(np.float32)
    model = SimpleNamespace(n_features=6)
    frame = pl.DataFrame({'x': range(6)})
    monkeypatch.setattr(density, '_embed', lambda model, df, space: X[df['x'].to_numpy()].copy())
    before = numerical_threads(0)
    with numerical_kernels(expected):
        with serial_kernels() if serial else nullcontext():
            assert pca_denoise(X.copy(), 2).shape == (12, 2)
            clono = _fit_repertoire_coordinates(model, X.copy(), n_components=2)
            assert clono.pca.n_components_ == 2
            means = list(enumerate(X))
            out, fitted = _identity_matrix(means, 12, 2)
            assert out.shape == (12, 2)
            np.testing.assert_allclose(_identity_matrix(means, 12, 2, pca=fitted)[0], out)
            for chunk in (None, 2):
                ds, obs, bg = density.fit_density_space(model, frame, frame, n_components=2, chunk_size=chunk)
                assert obs.shape == bg.shape == (6, 2)
                np.testing.assert_allclose(ds.transform(frame), obs, rtol=1e-5, atol=1e-6)
    assert len(calls) == 4
    assert numerical_threads(0) == before
