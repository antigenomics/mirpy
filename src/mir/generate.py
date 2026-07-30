"""The generative loop: a fitted density over repertoire descriptors, sample from it, evolve one.

Promotes the ad-hoc "in-silico evolution" pattern (a bare ``np.cov`` + a hardcoded conditional-mean
slope over a hand-picked metric list, prototyped analysis-side over TCGA) into a real, reusable
library object built on :class:`mir.repertoire.RepertoireDescriptor` — the mass-preserving,
decodable, smooth coordinate system :mod:`mir.repertoire` already ships (``[infiltration, log n_eff,
clonality] ‖ identity mean``). :class:`DescriptorDensity` is the (optionally class-conditional)
Gaussian fit over that coordinate; :meth:`~DescriptorDensity.sample` draws brand-new synthetic donor
states from it (a *digital twin* generator — see :mod:`mir.track` for the complementary trajectory
axis), and :meth:`~DescriptorDensity.evolve` perturbs one coordinate of a real descriptor and
propagates the coupled shift through every other coordinate via the fitted covariance's multivariate-
normal conditional mean — the same move as "hotter ⇒ diversity ↑ / switch ↑ / T-vs-B ↓" but as a
library primitive instead of a one-off script.

This is the *mechanical* half of the generative loop (ROADMAP Phase 2): a linear-Gaussian density is
what the shipped in-silico-evolution numbers were fit with, so it stays the default here. The
*research* half — a genuinely non-linear, learned generative model over the full embedding (not just
the compact descriptor) — is :mod:`mir.ml.diffusion`, which this module's ``sample``/``evolve`` API
deliberately mirrors so the two are drop-in alternatives.

What stays analysis-local (ROADMAP non-goals): fitting a survival/classification model on the
perturbed metrics (e.g. "does hotter predict better survival") is the study's scorer, not this
module's job — :mod:`mir.explain`'s ``channel_report``/the analysis's own CoxPH fit consume the
*output* of :meth:`~DescriptorDensity.evolve`, this module never sees an outcome ``y``.

Torch-free (numpy / sklearn).

Typical usage::

    from mir.repertoire import sample_descriptor
    from mir.generate import fit_descriptor_density, evolve

    descriptors = [sample_descriptor(space, s) for s in samples]
    density = fit_descriptor_density(descriptors, labels=tumor_type)   # one Gaussian per label

    synthetic = density.sample(50, condition="SKCM")                   # 50 new synthetic donor states
    hotter = evolve(density, descriptors[0], coordinate="infiltration", delta=2.0, condition="SKCM")
    hotter.metrics()                                                   # named metrics after the move
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mir.repertoire import RepertoireDescriptor

_COORD_INDEX = {"infiltration": 0, "log_neff": 1, "clonality": 2}


def _shrunk_cov(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance — stable with modest cohort sizes vs. the raw sample covariance."""
    from sklearn.covariance import LedoitWolf

    if X.shape[0] < 2:
        return np.eye(X.shape[1])
    return LedoitWolf().fit(X).covariance_


@dataclass
class DescriptorDensity:
    """A fitted (optionally class-conditional) Gaussian density over descriptor vectors.

    Attributes:
        mean: ``{label: (dim,) mean}`` — a single key ``None`` when unconditional.
        cov: ``{label: (dim, dim) shrinkage covariance}``.
        dim: Vector width (``len(RepertoireDescriptor.vector)``).
    """

    mean: dict = field(default_factory=dict)
    cov: dict = field(default_factory=dict)
    dim: int = 0

    @classmethod
    def fit(cls, X: np.ndarray, labels: list | None = None) -> "DescriptorDensity":
        """Fit a Gaussian (or, with ``labels``, one Gaussian per label) over descriptor vectors.

        Args:
            X: ``(n_samples, dim)`` stacked descriptor vectors (e.g.
                ``np.stack([d.vector for d in descriptors])``).
            labels: Optional length-``n_samples`` group labels (tumor type, batch, condition, …) —
                fits one mean/covariance per distinct label instead of a single pooled Gaussian.

        Returns:
            A fitted :class:`DescriptorDensity`.

        Raises:
            ValueError: If ``X`` has fewer than 2 rows.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] < 2:
            raise ValueError(f"need >= 2 samples to fit a density, got {X.shape[0]}")
        groups = {None: np.arange(X.shape[0])} if labels is None else {
            lab: np.where(np.asarray(labels) == lab)[0] for lab in sorted(set(labels), key=str)
        }
        mean, cov = {}, {}
        for lab, idx in groups.items():
            Xg = X[idx]
            mean[lab] = Xg.mean(axis=0)
            cov[lab] = _shrunk_cov(Xg) if Xg.shape[0] >= 2 else np.eye(X.shape[1])
        return cls(mean=mean, cov=cov, dim=X.shape[1])

    def _resolve(self, condition):
        if condition in self.mean:
            return self.mean[condition], self.cov[condition]
        if condition is None and len(self.mean) == 1:
            return next(iter(self.mean.values())), next(iter(self.cov.values()))
        raise ValueError(f"condition {condition!r} not in fitted labels {sorted(self.mean, key=str)}")

    def sample(self, n: int = 1, *, condition=None, seed: int = 0) -> np.ndarray:
        """Draw ``n`` synthetic descriptor vectors from the fitted (conditional) Gaussian.

        Args:
            n: Number of synthetic vectors to draw.
            condition: Label to condition on (must be one of the labels ``fit`` saw); ``None`` for an
                unconditional density, or the sole label when only one was fit.
            seed: RNG seed.

        Returns:
            ``(n, dim)`` array — decode with :func:`mir.repertoire.decode_metrics` per row.
        """
        mu, sigma = self._resolve(condition)
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(mu, sigma, size=n)

    def evolve(self, vector: np.ndarray, *, coordinate: int | str, delta: float, condition=None) -> np.ndarray:
        """Shift ``vector`` by ``delta`` along one coordinate, propagating the coupled response.

        The multivariate-normal conditional mean: fixing coordinate ``d`` to move by ``delta``
        shifts every other coordinate ``j`` by ``Sigma[j,d]/Sigma[d,d] * delta`` in expectation —
        this is the "hotter -> diversity up / switch up / T-vs-B down" move, generalized from a
        hand-picked metric list to the full descriptor (and, via :attr:`RepertoireDescriptor.mean`,
        the identity block too).

        Args:
            vector: A ``(dim,)`` descriptor vector to perturb (e.g. one real donor's).
            coordinate: Vector index, or one of ``"infiltration"`` / ``"log_neff"`` / ``"clonality"``
                for the named scalar coordinates.
            delta: Amount to shift ``coordinate`` by (in the descriptor's own units — e.g. log-mass).
            condition: Which fitted Gaussian's covariance defines the coupling (see :meth:`sample`).

        Returns:
            The perturbed ``(dim,)`` vector — decode with :func:`mir.repertoire.decode_metrics`.
        """
        d = _COORD_INDEX.get(coordinate, coordinate) if isinstance(coordinate, str) else coordinate
        _, sigma = self._resolve(condition)
        slope = sigma[:, d] / sigma[d, d]
        out = np.asarray(vector, dtype=np.float64) + delta * slope
        out[d] = vector[d] + delta   # the perturbed coordinate itself moves by exactly delta
        return out


def fit_descriptor_density(descriptors: list[RepertoireDescriptor], labels: list | None = None) -> DescriptorDensity:
    """Fit a :class:`DescriptorDensity` directly from :class:`~mir.repertoire.RepertoireDescriptor` objects.

    Args:
        descriptors: One :class:`~mir.repertoire.RepertoireDescriptor` per sample
            (:func:`mir.repertoire.sample_descriptor`).
        labels: Optional per-sample group labels — see :meth:`DescriptorDensity.fit`.

    Returns:
        A fitted :class:`DescriptorDensity`.
    """
    X = np.stack([d.vector for d in descriptors])
    return DescriptorDensity.fit(X, labels=labels)


def evolve(
    density: DescriptorDensity, descriptor: RepertoireDescriptor, *,
    coordinate: int | str, delta: float, condition=None,
) -> RepertoireDescriptor:
    """Perturb one :class:`~mir.repertoire.RepertoireDescriptor` and rebuild the result as one.

    Thin wrapper around :meth:`DescriptorDensity.evolve` that unpacks/repacks
    :class:`~mir.repertoire.RepertoireDescriptor` instead of a bare vector, so the result's
    ``.metrics()`` reads off directly.

    Args:
        density: A fitted :class:`DescriptorDensity` (the coupling comes from its covariance).
        descriptor: The descriptor to perturb.
        coordinate, delta, condition: See :meth:`DescriptorDensity.evolve`.

    Returns:
        The perturbed :class:`~mir.repertoire.RepertoireDescriptor`.
    """
    v = density.evolve(descriptor.vector, coordinate=coordinate, delta=delta, condition=condition)
    return RepertoireDescriptor(log_mass=float(v[0]), log_neff=float(v[1]), simpson=float(v[2]), mean=v[3:])


def _demo() -> None:
    """Self-check on synthetic descriptors: sampling matches the fit moments; evolve moves the target
    coordinate by exactly delta and couples a correlated coordinate in the expected direction."""
    rng = np.random.default_rng(0)
    n, dim = 300, 6
    # a correlated synthetic cohort: coordinate 0 ("infiltration") positively drives coordinate 1
    base = rng.standard_normal((n, dim))
    base[:, 1] = 0.8 * base[:, 0] + 0.2 * rng.standard_normal(n)   # coord 1 tracks coord 0
    descriptors = [
        RepertoireDescriptor(log_mass=float(r[0]), log_neff=float(r[1]), simpson=float(r[2]), mean=r[3:])
        for r in base
    ]
    density = fit_descriptor_density(descriptors)

    synth = density.sample(2000, seed=1)
    assert synth.shape == (2000, dim)
    assert np.allclose(synth.mean(axis=0), base.mean(axis=0), atol=0.15)

    d0 = descriptors[0]
    moved = evolve(density, d0, coordinate="infiltration", delta=2.0)
    assert abs(moved.log_mass - (d0.log_mass + 2.0)) < 1e-9          # exact move on the target coord
    assert moved.log_neff > d0.log_neff                                # positive coupling recovered

    # class-conditional fit: two well-separated labels sample from their own moments
    labels = ["hot"] * (n // 2) + ["cold"] * (n - n // 2)
    base2 = base.copy()
    base2[n // 2:, 0] -= 5.0
    descriptors2 = [
        RepertoireDescriptor(log_mass=float(r[0]), log_neff=float(r[1]), simpson=float(r[2]), mean=r[3:])
        for r in base2
    ]
    cond_density = fit_descriptor_density(descriptors2, labels=labels)
    hot_synth = cond_density.sample(500, condition="hot", seed=2)
    cold_synth = cond_density.sample(500, condition="cold", seed=2)
    assert hot_synth[:, 0].mean() - cold_synth[:, 0].mean() > 3.0

    print(f"[ok] sample moments match (atol=0.15); evolve moved infiltration by exactly 2.0, "
          f"coupled log_neff {d0.log_neff:.3f}->{moved.log_neff:.3f}; conditional sampling separates "
          f"hot/cold by {hot_synth[:,0].mean()-cold_synth[:,0].mean():.2f}")


if __name__ == "__main__":
    _demo()
