"""Tests for mir.ml.diffusion (conditional DDPM/DDIM generator). Needs the ``[ml]`` extra (torch)."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip("torch")  # skip, don't fail, on a torch-free install ([ml] extra)

from mir.ml.diffusion import DiffusionModel, cosine_beta_schedule, train_diffusion  # noqa: E402


def _two_clusters(n=400, dim=4, seed=0, sep=5.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, dim)).astype(np.float32)
    labels = np.array(["a"] * (n // 2) + ["b"] * (n - n // 2))
    z[labels == "b"] += sep
    return z, labels


def test_cosine_beta_schedule_shape_and_range():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert (betas > 0).all() and (betas < 1).all()


def test_train_raises_on_too_few_samples():
    with pytest.raises(ValueError, match=">= 4"):
        train_diffusion(np.zeros((3, 4)), T=10, epochs=1, verbose=False)


def test_unconditional_sample_matches_data_moments():
    z, _ = _two_clusters(seed=1)
    model, metrics = train_diffusion(z, T=100, epochs=100, seed=0, verbose=False)
    assert metrics["n_classes"] == 0
    synth = model.sample(300, steps=30, seed=2)
    assert synth.shape == (300, z.shape[1])
    # a loose moment check (this is a stochastic generator): should land near the pooled mean, not
    # diverge or collapse onto just one of the two synthetic clusters (which are 5.0 apart).
    assert abs(float(synth.mean()) - float(z.mean())) < 2.5


def test_conditional_sample_separates_classes():
    z, labels = _two_clusters(seed=2, sep=6.0)
    model, metrics = train_diffusion(z, labels, T=100, epochs=80, seed=0, verbose=False)
    assert metrics["n_classes"] == 2
    a = model.sample(200, condition="a", steps=30, guidance_scale=2.0, seed=3)
    b = model.sample(200, condition="b", steps=30, guidance_scale=2.0, seed=3)
    assert b.mean() - a.mean() > 3.0


def test_unknown_condition_raises():
    z, labels = _two_clusters(seed=3)
    model, _ = train_diffusion(z, labels, T=50, epochs=20, seed=0, verbose=False)
    with pytest.raises(ValueError, match="not in fitted classes"):
        model.sample(1, condition="nope")


def test_condition_without_training_labels_raises():
    z, _ = _two_clusters(seed=4)
    model, _ = train_diffusion(z, T=50, epochs=20, seed=0, verbose=False)   # unconditional
    with pytest.raises(ValueError, match="trained unconditionally"):
        model.sample(1, condition="a")


def test_guidance_without_condition_raises():
    z, labels = _two_clusters(seed=5)
    model, _ = train_diffusion(z, labels, T=50, epochs=20, seed=0, verbose=False)
    with pytest.raises(ValueError, match="guidance_scale"):
        model.sample(1, guidance_scale=2.0)


def test_save_load_roundtrip(tmp_path):
    z, _ = _two_clusters(n=200, seed=6)
    model, _ = train_diffusion(z, T=50, epochs=20, seed=0, verbose=False)
    path = tmp_path / "diffusion.pt"
    model.save(path)
    loaded = DiffusionModel.load(path)
    a = model.sample(10, steps=15, seed=7)
    b = loaded.sample(10, steps=15, seed=7)
    assert np.allclose(a, b, atol=1e-4)


def test_matches_prototype_hash():
    z, _ = _two_clusters(n=100, seed=7)
    model, _ = train_diffusion(z, T=50, epochs=10, seed=0, verbose=False, meta={"prototype_hash": "abc123"})
    assert model.matches_prototype_hash("abc123")
    assert not model.matches_prototype_hash("different")
