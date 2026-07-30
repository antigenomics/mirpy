"""Tests for mir.generate (DescriptorDensity: sample / evolve over RepertoireDescriptor)."""

import numpy as np
import pytest

from mir.generate import DescriptorDensity, evolve, fit_descriptor_density
from mir.repertoire import RepertoireDescriptor


def _descriptors(n=200, dim=5, seed=0, shift_second_half=None):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim))
    base[:, 1] = 0.8 * base[:, 0] + 0.2 * rng.standard_normal(n)   # coord 1 tracks coord 0
    if shift_second_half is not None:
        base[n // 2:, 0] += shift_second_half
    return [RepertoireDescriptor(log_mass=float(r[0]), log_neff=float(r[1]), simpson=float(r[2]), mean=r[3:])
            for r in base], base


def test_fit_raises_on_too_few_samples():
    with pytest.raises(ValueError, match=">= 2"):
        DescriptorDensity.fit(np.zeros((1, 3)))


def test_sample_moments_match_fit_data():
    descriptors, base = _descriptors(n=500, seed=0)
    density = fit_descriptor_density(descriptors)
    synth = density.sample(5000, seed=1)
    assert synth.shape == (5000, base.shape[1])
    assert np.allclose(synth.mean(axis=0), base.mean(axis=0), atol=0.15)
    assert np.allclose(np.cov(synth, rowvar=False), np.cov(base, rowvar=False), atol=0.3)


def test_evolve_moves_target_coordinate_exactly_and_couples_correlated_one():
    descriptors, _ = _descriptors(n=300, seed=2)
    density = fit_descriptor_density(descriptors)
    d0 = descriptors[0]

    moved = evolve(density, d0, coordinate="infiltration", delta=2.0)
    assert abs(moved.log_mass - (d0.log_mass + 2.0)) < 1e-9
    assert moved.log_neff > d0.log_neff          # positive coupling (coord 1 tracks coord 0)
    assert abs(moved.simpson - d0.simpson) < 0.5  # simpson has no planted correlation -> only moves a little


def test_evolve_by_integer_index_matches_named_coordinate():
    descriptors, _ = _descriptors(n=200, seed=3)
    density = fit_descriptor_density(descriptors)
    d0 = descriptors[0]
    by_name = evolve(density, d0, coordinate="infiltration", delta=1.5)
    by_index = evolve(density, d0, coordinate=0, delta=1.5)
    assert np.allclose(by_name.vector, by_index.vector)


def test_conditional_density_separates_groups():
    descriptors, base = _descriptors(n=400, seed=4, shift_second_half=6.0)
    labels = ["a"] * (len(descriptors) // 2) + ["b"] * (len(descriptors) - len(descriptors) // 2)
    density = fit_descriptor_density(descriptors, labels=labels)
    a = density.sample(500, condition="a", seed=5)
    b = density.sample(500, condition="b", seed=5)
    assert b[:, 0].mean() - a[:, 0].mean() > 4.0


def test_unknown_condition_raises():
    descriptors, _ = _descriptors(n=50, seed=6)
    density = fit_descriptor_density(descriptors, labels=["a"] * 25 + ["b"] * 25)
    with pytest.raises(ValueError, match="not in fitted labels"):
        density.sample(1, condition="nope")
