"""Tests for mir.twin (DonorTwin: perturb / simulate glue over descriptor + generator)."""

import numpy as np
import pytest

from mir.generate import fit_descriptor_density
from mir.repertoire import RepertoireDescriptor
from mir.twin import DonorTwin, make_twins


def _cohort(n=200, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim))
    base[:, 1] = 0.7 * base[:, 0] + 0.3 * rng.standard_normal(n)
    labels = ["hot"] * (n // 2) + ["cold"] * (n - n // 2)
    base[n // 2:, 0] -= 4.0
    descriptors = [RepertoireDescriptor(log_mass=float(r[0]), log_neff=float(r[1]), simpson=float(r[2]), mean=r[3:])
                  for r in base]
    return descriptors, labels


def test_make_twins_zips_fields_and_defaults():
    descriptors, labels = _cohort(n=10, seed=1)
    twins = make_twins(descriptors, conditions=labels, donor_ids=[f"D{i}" for i in range(10)])
    assert len(twins) == 10
    assert twins[0].condition == "hot" and twins[0].donor_id == "D0" and twins[0].tau is None
    assert twins[0].descriptor is descriptors[0]

    bare = make_twins(descriptors)
    assert all(t.condition is None and t.tau is None and t.donor_id is None for t in bare)


def test_make_twins_length_mismatch_raises():
    descriptors, _ = _cohort(n=5, seed=2)
    with pytest.raises(ValueError, match="conditions"):
        make_twins(descriptors, conditions=["a", "b"])


def test_metrics_delegates_to_descriptor():
    descriptors, _ = _cohort(n=5, seed=3)
    twin = DonorTwin(descriptor=descriptors[0])
    assert twin.metrics() == descriptors[0].metrics()


def test_perturb_moves_exact_coordinate_and_preserves_identity():
    descriptors, labels = _cohort(seed=4)
    density = fit_descriptor_density(descriptors, labels=labels)
    twin = DonorTwin(descriptor=descriptors[0], condition="hot", donor_id="D0", tau=1.23)

    moved = twin.perturb(density, coordinate="infiltration", delta=2.0)
    assert abs(moved.descriptor.log_mass - (twin.descriptor.log_mass + 2.0)) < 1e-9
    assert moved.condition == "hot" and moved.donor_id == "D0" and moved.tau == 1.23
    assert moved.descriptor is not twin.descriptor    # a new object, not mutated in place


def test_simulate_defaults_to_own_condition():
    descriptors, labels = _cohort(seed=5)
    density = fit_descriptor_density(descriptors, labels=labels)
    hot_twin = DonorTwin(descriptor=descriptors[0], condition="hot")
    cold_twin = DonorTwin(descriptor=descriptors[-1], condition="cold")

    hot_synth = hot_twin.simulate(density, 300, seed=1)
    cold_synth = cold_twin.simulate(density, 300, seed=1)
    hot_mean = np.mean([m["infiltration"] for m in hot_synth])
    cold_mean = np.mean([m["infiltration"] for m in cold_synth])
    assert hot_mean - cold_mean > 3.0


def test_simulate_condition_override():
    descriptors, labels = _cohort(seed=6)
    density = fit_descriptor_density(descriptors, labels=labels)
    twin = DonorTwin(descriptor=descriptors[0], condition="hot")
    synth = twin.simulate(density, 300, condition="cold", seed=1)
    assert np.mean([m["infiltration"] for m in synth]) < 0.0   # overridden to the "cold" group


def test_simulate_returns_decoded_metric_dicts():
    descriptors, labels = _cohort(seed=7)
    density = fit_descriptor_density(descriptors, labels=labels)
    twin = DonorTwin(descriptor=descriptors[0], condition="hot")
    synth = twin.simulate(density, 3, seed=1)
    assert len(synth) == 3
    assert all(set(m) == {"infiltration", "log_neff", "diversity", "clonality"} for m in synth)
