"""The digital twin: one donor's compact, perturbable, simulatable state.

Where :class:`~mir.cohort.DonorCohort` fuses a *cohort's* measured state into one comparable matrix
for scoring, this module wraps **one** donor's state as an object that can be perturbed
(:func:`mir.generate.evolve`, a "what if" move) or used to seed brand-new synthetic realizations from
a fitted generator — :class:`mir.generate.DescriptorDensity` (linear-Gaussian) or
:class:`mir.ml.diffusion.DiffusionModel` (non-linear, needs ``[ml]``) — both of which share the same
``sample(n, *, condition=None, seed=0)`` call shape, so either drops in unchanged. A twin optionally
carries this donor's position on a fitted :class:`~mir.track.TrajectoryFit` (where along an inferred
exposure/progression axis it sits) and its known covariate, so :meth:`~DonorTwin.simulate` can
resample condition-matched to the donor's own group by default.

This closes the loop the rest of the library already opened: :mod:`mir.cohort` *measures* a donor,
:mod:`mir.track` locates it on a *trajectory*, :mod:`mir.generate` / :mod:`mir.ml.diffusion`
*generate* new states — :class:`DonorTwin` is the one object a caller perturbs or resamples through,
instead of threading three APIs together by hand for every donor.

Torch-free itself; the diffusion generator (only if actually passed to :meth:`~DonorTwin.simulate`)
is imported lazily by the caller, not by this module.

Typical usage::

    from mir.twin import make_twins
    from mir.generate import fit_descriptor_density

    density = fit_descriptor_density(descriptors, labels=tumor_type)
    twins = make_twins(descriptors, conditions=tumor_type, donor_ids=sample_ids)

    hotter = twins[0].perturb(density, coordinate="infiltration", delta=2.0)  # this donor, "what if hotter"
    synthetic = twins[0].simulate(density, n=20)                              # 20 new synthetic peers
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mir.repertoire import RepertoireDescriptor, decode_metrics


@dataclass
class DonorTwin:
    """One donor's digital twin: measured state + optional trajectory position + covariate.

    Attributes:
        descriptor: The donor's measured :class:`~mir.repertoire.RepertoireDescriptor`.
        tau: This donor's position on a fitted :class:`~mir.track.TrajectoryFit`, if known
            (typically ``fit.tau[donor_index]``).
        condition: The donor's known covariate/group label (tumor type, HLA, batch, …) — the default
            conditioning value for :meth:`perturb` / :meth:`simulate`.
        donor_id: Optional identifier, carried through unchanged by every method here.
    """

    descriptor: RepertoireDescriptor
    tau: float | None = None
    condition: object | None = None
    donor_id: str | None = None

    def metrics(self) -> dict:
        """Named metrics of the current descriptor (:meth:`~mir.repertoire.RepertoireDescriptor.metrics`)."""
        return self.descriptor.metrics()

    def perturb(self, density, *, coordinate, delta: float, condition=None) -> "DonorTwin":
        """Move this twin along one descriptor coordinate (:func:`mir.generate.evolve`).

        Args:
            density: A fitted :class:`mir.generate.DescriptorDensity` — the coupling between
                coordinates comes from its covariance.
            coordinate: Vector index, or ``"infiltration"`` / ``"log_neff"`` / ``"clonality"``.
            delta: Amount to shift ``coordinate`` by (in the descriptor's own units).
            condition: Which of ``density``'s fitted groups defines the coupling; defaults to this
                twin's own :attr:`condition`.

        Returns:
            A new :class:`DonorTwin` with the perturbed descriptor — :attr:`tau`, :attr:`condition`
            and :attr:`donor_id` carry over unchanged (a perturbation is a "what if", not a
            re-measurement of a different donor).
        """
        from mir.generate import evolve

        new_descriptor = evolve(
            density, self.descriptor, coordinate=coordinate, delta=delta,
            condition=condition if condition is not None else self.condition,
        )
        return DonorTwin(descriptor=new_descriptor, tau=self.tau, condition=self.condition,
                         donor_id=self.donor_id)

    def simulate(self, generator, n: int = 1, *, condition=None, seed: int = 0, **kwargs) -> list[dict]:
        """Draw ``n`` new synthetic donor states from a fitted generator, decoded to named metrics.

        Args:
            generator: A fitted :class:`mir.generate.DescriptorDensity` or
                :class:`mir.ml.diffusion.DiffusionModel` — both share the ``sample(n, *, condition,
                seed)`` call shape; either drops in unchanged.
            n: Number of synthetic states to draw.
            condition: Which fitted group to sample from; defaults to this twin's own
                :attr:`condition`.
            seed: RNG seed.
            **kwargs: Forwarded to ``generator.sample`` (e.g. a :class:`~mir.ml.diffusion.DiffusionModel`'s
                ``steps=``/``guidance_scale=``; unused by :class:`~mir.generate.DescriptorDensity`).

        Returns:
            ``n`` decoded metric dicts (:func:`mir.repertoire.decode_metrics`) — synthetic "what a
            donor like this typically looks like" states, not perturbations of this specific twin
            (see :meth:`perturb` for that).
        """
        cond = condition if condition is not None else self.condition
        vectors = generator.sample(n, condition=cond, seed=seed, **kwargs)
        return [decode_metrics(v) for v in vectors]


def make_twins(
    descriptors: list[RepertoireDescriptor], *,
    tau: np.ndarray | None = None, conditions: list | None = None, donor_ids: list[str] | None = None,
) -> list[DonorTwin]:
    """Zip a cohort's descriptors (+ optional trajectory/condition/id) into one :class:`DonorTwin` each.

    Args:
        descriptors: One :class:`~mir.repertoire.RepertoireDescriptor` per donor.
        tau: Optional ``(n_donors,)`` trajectory position per donor (a fitted
            :class:`~mir.track.TrajectoryFit`'s ``.tau``), row-aligned to ``descriptors``.
        conditions: Optional per-donor covariate/group labels, row-aligned.
        donor_ids: Optional per-donor identifiers, row-aligned.

    Returns:
        One :class:`DonorTwin` per donor, in the same order as ``descriptors``.

    Raises:
        ValueError: If a supplied ``tau`` / ``conditions`` / ``donor_ids`` length disagrees with
            ``descriptors``.
    """
    n = len(descriptors)
    for name, seq in (("tau", tau), ("conditions", conditions), ("donor_ids", donor_ids)):
        if seq is not None and len(seq) != n:
            raise ValueError(f"{name} has {len(seq)} entries, descriptors has {n}")
    tau = [None] * n if tau is None else list(tau)
    conditions = [None] * n if conditions is None else conditions
    donor_ids = [None] * n if donor_ids is None else donor_ids
    return [DonorTwin(descriptor=d, tau=t, condition=c, donor_id=i)
            for d, t, c, i in zip(descriptors, tau, conditions, donor_ids)]


def _demo() -> None:
    """Self-check: perturb agrees with mir.generate.evolve directly; simulate returns
    condition-matched synthetic states; make_twins zips a cohort correctly."""
    from mir.generate import fit_descriptor_density

    rng = np.random.default_rng(0)
    n, dim = 200, 5
    base = rng.standard_normal((n, dim))
    base[:, 1] = 0.7 * base[:, 0] + 0.3 * rng.standard_normal(n)   # coord 1 tracks coord 0
    labels = ["hot"] * (n // 2) + ["cold"] * (n - n // 2)
    base[n // 2:, 0] -= 4.0                                        # separate the two groups
    descriptors = [
        RepertoireDescriptor(log_mass=float(r[0]), log_neff=float(r[1]), simpson=float(r[2]), mean=r[3:])
        for r in base
    ]
    density = fit_descriptor_density(descriptors, labels=labels)

    twins = make_twins(descriptors, conditions=labels, donor_ids=[f"D{i}" for i in range(n)])
    assert len(twins) == n and twins[0].donor_id == "D0" and twins[0].condition == "hot"

    moved = twins[0].perturb(density, coordinate="infiltration", delta=2.0)
    assert abs(moved.descriptor.log_mass - (twins[0].descriptor.log_mass + 2.0)) < 1e-9
    assert moved.condition == twins[0].condition and moved.donor_id == twins[0].donor_id

    synth_hot = twins[0].simulate(density, 200, seed=1)     # defaults to its own "hot" condition
    synth_cold = twins[-1].simulate(density, 200, seed=1)   # defaults to its own "cold" condition
    hot_mean = np.mean([m["infiltration"] for m in synth_hot])
    cold_mean = np.mean([m["infiltration"] for m in synth_cold])
    assert hot_mean - cold_mean > 3.0, f"twin-conditioned simulation not separated: {hot_mean - cold_mean:.2f}"

    print(f"[ok] perturb exact on target coord; twin.simulate separates hot/cold by "
          f"{hot_mean - cold_mean:.2f}; make_twins zipped {len(twins)} donors")


if __name__ == "__main__":
    _demo()
