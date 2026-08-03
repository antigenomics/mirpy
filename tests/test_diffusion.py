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


def test_save_load_roundtrip_with_a_non_default_architecture(tmp_path):
    """Regression: `load` rebuilt DiffusionMLP at constructor defaults.

    hidden / time_dim / class_dim all change state-dict shapes but were never recorded in meta,
    so anything trained off-defaults raised "size mismatch" on load. The save/load contract held
    only for the architecture the tests happened to use.
    """
    z, labels = _two_clusters(n=200, seed=6)
    model, _ = train_diffusion(z, labels, T=50, epochs=10, seed=0, verbose=False,
                               hidden=64, time_dim=16, class_dim=8)
    assert (model.meta["hidden"], model.meta["time_dim"], model.meta["class_dim"]) == (64, 16, 8)

    path = tmp_path / "diffusion.pt"
    model.save(path)
    loaded = DiffusionModel.load(path)          # raised RuntimeError: size mismatch before the fix
    assert np.allclose(model.sample(10, condition="a", steps=15, seed=7),
                       loaded.sample(10, condition="a", steps=15, seed=7), atol=1e-4)


def test_load_defaults_the_architecture_for_bundles_saved_before_it_was_recorded(tmp_path):
    """Older bundles carry no architecture keys; they were all trained at the defaults."""
    import torch

    z, _ = _two_clusters(n=120, seed=3)
    model, _ = train_diffusion(z, T=30, epochs=5, seed=0, verbose=False)
    legacy = dict(model.meta)
    for k in ("hidden", "time_dim", "class_dim"):
        legacy.pop(k)
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": model.model.state_dict(), "betas": model.betas, "meta": legacy}, path)

    loaded = DiffusionModel.load(path)
    assert np.allclose(model.sample(5, steps=10, seed=1),
                       loaded.sample(5, steps=10, seed=1), atol=1e-4)


def test_ddim_x0_clamp_recomputes_eps():
    """Regression: the x0 clamp left `eps` at the raw network output, so the step ignored it.

    At the timesteps the clamp exists for (alphabar_t ~ 0, sqrt(1-ab_prev) ~ 1) the update
    degenerates to x <- eps, so an unbounded eps propagated anyway.

    The *returned* sample cannot show this -- the last step has ab_prev == 1, so eps is multiplied
    by zero and the output is the clamped x0 either way. The corruption lives in the intermediate
    trajectory, which is what a real model then sees as out-of-distribution input. So probe the
    trajectory: record what the model is handed at each step.
    """
    import torch

    z, _ = _two_clusters(n=120, dim=3, seed=4)
    model, _ = train_diffusion(z, T=40, epochs=5, seed=0, verbose=False)

    seen = []

    class Exploding(torch.nn.Module):
        """Stands in for a badly-fit epsilon predictor at high noise."""

        n_classes = 0

        def __init__(self):
            super().__init__()
            self.unused = torch.nn.Parameter(torch.zeros(1))  # so `sample` can read the device

        def forward(self, x, t, y=None):
            seen.append(float(x.abs().max()))
            return torch.full_like(x, 500.0)

    model.model = Exploding()
    out = model.sample(8, steps=12, seed=0)

    assert np.isfinite(out).all()
    # x0 is clamped to +-6 standardized units and eps is re-derived from THAT, so the state stays
    # on the order of the clamp. Leaving eps at the raw 500 pushed the trajectory to ~500.
    assert max(seen) < 25.0, f"trajectory left the clamp scale: max|x| = {max(seen):.1f}"


def test_validation_yardstick_is_fixed_and_independent_of_training_randomness():
    """Regression: val_loss redrew (t, eps) every epoch, so best-epoch selection was noise.

    Eps-prediction MSE varies systematically with the timestep, so consecutive epochs measured
    *different problems* and `best_state` latched onto whichever epoch drew easy timesteps.

    The fixed noising is drawn once from a seeded generator BEFORE training, so it no longer
    depends on how much global RNG the training loop happens to consume. Changing only the batch
    size changes that consumption, and under the old code moved the yardstick with it; now the
    two runs are scored against the identical validation problem.
    """
    z, _ = _two_clusters(n=400, seed=9)
    _, m1 = train_diffusion(z, T=50, epochs=20, seed=0, verbose=False)
    _, m2 = train_diffusion(z, T=50, epochs=20, seed=0, verbose=False)
    assert m1["val_loss"] == pytest.approx(m2["val_loss"], rel=1e-6)   # reproducible

    # An eps-predictor scored on a FIXED noising must beat an untrained one by a wide margin;
    # against a lottery of per-epoch redraws this comparison had no fixed meaning at all.
    _, m_trained = train_diffusion(z, T=50, epochs=60, seed=0, verbose=False)
    _, m_barely = train_diffusion(z, T=50, epochs=1, seed=0, verbose=False)
    assert m_trained["val_loss"] < m_barely["val_loss"]
