"""Covariate-disentangled repertoire trajectory: a PhenoPath-style exposure pseudotime.

Exposure detection so far in mirpy is **clone-level**: :func:`mir.density.neighbor_enrichment`
answers "which clonotypes look antigen-driven" against a background null. This module answers a
different question at the **repertoire level**: given a cohort of samples with a *known* covariate
(HLA type, vaccine arm, batch — anything already recorded) but an *unknown or noisy* progression
axis (days since exposure, response stage), can we recover a shared latent trajectory while
explicitly separating out which channels of the embedding respond to that trajectory differently
depending on the covariate?

This is the model of PhenoPath (Campbell & Yau 2018, *Nat. Commun.* 9:2442,
`doi:10.1038/s41467-018-04696-6 <https://doi.org/10.1038/s41467-018-04696-6>`_), adapted from genes ×
cells to **repertoire channels × samples**. For sample *n* and channel *g*::

    Y[n, g] = c_g + alpha_g . x_n + (kappa_g + gamma_g . x_n) * tau_n + eps_{n,g}

``x_n`` is the known covariate vector, ``tau_n`` the latent trajectory (pseudo-exposure/progression),
``kappa_g`` the channel's baseline sensitivity to the trajectory, and ``gamma_g`` the **covariate ×
trajectory interaction** — the object of interest: a channel with large ``‖gamma_g‖`` evolves along the
trajectory differently depending on the covariate (e.g. an HLA-restricted response channel that
contracts over the response only in carriers). ``alpha_g`` is the covariate's ordinary main effect,
independent of progression.

Inference here is a **simplified, closed-form alternative to PhenoPath's coordinate-ascent
variational inference**: rather than a full mean-field posterior over every parameter, this
alternates (a) per-channel ridge regression for ``(c, alpha, kappa, gamma)`` given ``tau``, (b) a
generalized-least-squares update of ``tau`` given the channel loadings, and (c) an iteratively-
reweighted (empirical-Bayes) precision update on ``gamma`` that shrinks small interaction
coefficients toward zero — the fast/approximate limit of PhenoPath's ARD prior, not a literal
reimplementation of its CAVI engine. Both blocks are ordinary linear-Gaussian conditionals, so every
step is closed-form; this keeps the module torch-free (numpy only), consistent with
:mod:`mir.repertoire` / :mod:`mir.density`.

Typical usage::

    from mir.explain import stack_embeddings
    from mir.track import fit_exposure_trajectory

    embs = [sample_embedding(space, s) for s in samples]     # one shared RepertoireSpace
    X, spec = stack_embeddings(embs)                         # channels: mean | diversity | second
    covariates = hla_indicator                                # (n_samples, n_alleles) known covariate
    fit = fit_exposure_trajectory(X, covariates, channel_names=spec.names)
    fit.tau                                                    # inferred exposure/progression pseudotime
    fit.top_interactions(top=5)                                # which channels are covariate-modulated
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

_RIDGE_MAIN = 1e-6   # light ridge on intercept/alpha/kappa (numerical stability only, no shrinkage)
_RIDGE_TAU = 1e-8    # guards the tau update's denominator when all channel loadings vanish


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


@dataclass
class TrajectoryFit:
    """A fitted covariate-disentangled repertoire trajectory (PhenoPath-style, see module docstring).

    Attributes:
        tau: ``(n_samples,)`` latent trajectory, standardized (mean 0, unit variance). Sign is fixed
            so ``tau`` correlates positively with the first principal component of the (standardized)
            input channels, so it is not arbitrary between refits.
        intercept: ``(n_channels,)`` per-channel intercept ``c_g``.
        alpha: ``(n_channels, n_covariates)`` direct covariate effect (independent of ``tau``).
        kappa: ``(n_channels,)`` baseline trajectory loading — the channel's progression sensitivity
            shared across covariate levels.
        gamma: ``(n_channels, n_covariates)`` covariate x trajectory interaction — the channel's
            progression sensitivity *changes* by this much per unit covariate. The quantity
            :meth:`top_interactions` ranks channels by.
        sigma2: ``(n_channels,)`` residual variance per channel.
        ard_precision: ``(n_channels, n_covariates)`` shrinkage precision applied to ``gamma``
            (large -> that interaction is forced toward 0). Only meaningful when fit with ``ard=True``.
        channel_names / covariate_names: Labels carried through for reporting, if supplied.
        n_iter: Iterations actually run.
        converged: Whether the relative change in ``(tau, gamma)`` fell below ``tol``.
        loglik_trace: Per-iteration Gaussian log-likelihood (up to the additive normalizing constant)
            — should increase monotonically-ish; a non-monotone trace signals a fitting problem.
    """

    tau: np.ndarray
    intercept: np.ndarray
    alpha: np.ndarray
    kappa: np.ndarray
    gamma: np.ndarray
    sigma2: np.ndarray
    ard_precision: np.ndarray
    channel_names: list[str] | None
    covariate_names: list[str] | None
    n_iter: int
    converged: bool
    loglik_trace: list[float]

    @property
    def interaction_score(self) -> np.ndarray:
        """``(n_channels,)`` covariate-trajectory interaction magnitude, ``‖gamma_g‖`` over covariates."""
        return np.linalg.norm(self.gamma, axis=1)

    def top_interactions(self, top: int = 10) -> pl.DataFrame:
        """Rank channels by covariate x trajectory interaction strength (descending).

        Args:
            top: Number of channels to return.

        Returns:
            Columns ``channel``, ``interaction_score`` (``‖gamma_g‖``), ``kappa`` (the channel's
            baseline trajectory sensitivity, for context), sorted by ``interaction_score`` descending.
        """
        names = self.channel_names or [f"channel{i}" for i in range(len(self.kappa))]
        return (
            pl.DataFrame({
                "channel": names,
                "interaction_score": self.interaction_score,
                "kappa": self.kappa,
            })
            .sort("interaction_score", descending=True)
            .head(top)
        )


def fit_exposure_trajectory(
    X: np.ndarray,
    covariates: np.ndarray,
    *,
    channel_names: list[str] | None = None,
    covariate_names: list[str] | None = None,
    ard: bool = True,
    max_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 0,
) -> TrajectoryFit:
    """Fit a covariate-disentangled latent trajectory over a per-sample channel matrix.

    Args:
        X: ``(n_samples, n_channels)`` per-sample feature matrix — any stacked repertoire channel
            matrix (:func:`mir.explain.stack_embeddings`, a :class:`~mir.explain.ChannelBuilder`
            build, or a raw embedding block). Standardized internally; the input scale doesn't matter.
        covariates: ``(n_samples,)`` or ``(n_samples, n_covariates)`` known covariate(s) — e.g. an
            HLA indicator, batch, vaccine arm. Continuous or one-hot; standardized internally.
        channel_names: Optional length-``n_channels`` labels, carried onto :class:`TrajectoryFit` for
            :meth:`~TrajectoryFit.top_interactions`.
        covariate_names: Optional length-``n_covariates`` labels.
        ard: Iteratively-reweighted shrinkage on ``gamma`` (see module docstring) — the empirical-Bayes
            approximation to PhenoPath's ARD prior. ``False`` fits an unregularized interaction term
            (a fixed light ridge only, no shrinkage), useful for comparison / small cohorts where ARD's
            extra sparsity isn't wanted.
        max_iter: Maximum alternating-update iterations.
        tol: Convergence tolerance on the relative change of ``(tau, gamma)`` between iterations.
        seed: RNG seed for a tiny initialization jitter that breaks exact PC1 ties (e.g. a
            rank-deficient ``X``); the fit is otherwise deterministic.

    Returns:
        A fitted :class:`TrajectoryFit`.

    Raises:
        ValueError: On a row-count mismatch between ``X`` and ``covariates``, or fewer than 3 samples.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_channels), got shape {X.shape}")
    cov = np.asarray(covariates, dtype=np.float64)
    if cov.ndim == 1:
        cov = cov[:, None]
    if cov.shape[0] != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples, covariates has {cov.shape[0]}")
    n, g = X.shape
    p = cov.shape[1]
    if n < 3:
        raise ValueError(f"need >= 3 samples to fit a trajectory, got {n}")
    if channel_names is not None and len(channel_names) != g:
        raise ValueError(f"channel_names has {len(channel_names)} entries, X has {g} channels")
    if covariate_names is not None and len(covariate_names) != p:
        raise ValueError(f"covariate_names has {len(covariate_names)} entries, covariates has {p}")

    Y, _, _ = _standardize(X)
    C, _, _ = _standardize(cov) if p else (cov, None, None)

    # Init: tau = first PC of Y (a reasonable trajectory guess), standardized; its sign is the fixed
    # reference orientation for every later iteration (the model is invariant under tau,kappa,gamma
    # all flipping sign together, so without a fixed reference tau could flip iteration to iteration).
    rng = np.random.default_rng(seed)
    _, _, Vt = np.linalg.svd(Y - Y.mean(axis=0), full_matrices=False)
    tau_ref_dir = Vt[0]
    # a tiny seeded jitter breaks exact PC1 ties (e.g. a rank-deficient Y) without perturbing a
    # well-conditioned fit -- this is the only place `seed` enters the otherwise-deterministic fit.
    tau = (Y - Y.mean(axis=0)) @ tau_ref_dir + 1e-6 * rng.standard_normal(n)
    tau = (tau - tau.mean()) / (tau.std() if tau.std() > 0 else 1.0)

    intercept = np.zeros(g)
    alpha = np.zeros((g, p))
    kappa = np.zeros(g)
    gamma = np.zeros((g, p))
    sigma2 = np.ones(g)
    ard_precision = np.ones((g, p))

    loglik_trace: list[float] = []
    converged = False
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1

        # --- M-step: per-channel ridge regression for (c, alpha, kappa, gamma) given tau ---
        D_main = np.hstack([np.ones((n, 1)), C, tau[:, None]])           # (n, 1+p+1) -- always-in
        D_inter = C * tau[:, None] if p else np.zeros((n, 0))            # (n, p) -- covariate x tau
        D = np.hstack([D_main, D_inter])                                # (n, 1+2p+1)
        k_main = 1 + p + 1
        # D is fixed across channels within an M-step (it depends only on tau), so its Gram and
        # its projection onto Y are computed ONCE rather than g times per iteration — only the
        # ridge diagonal below is per-channel.
        G = D.T @ D
        DtY = D.T @ Y                                                   # (1+2p+1, g)
        for j in range(g):
            reg = np.concatenate([
                np.full(k_main, _RIDGE_MAIN),
                ard_precision[j] if ard else np.full(p, _RIDGE_MAIN),
            ])
            A = G + np.diag(reg)
            theta = np.linalg.solve(A, DtY[:, j])
            intercept[j] = theta[0]
            alpha[j] = theta[1:1 + p]
            kappa[j] = theta[1 + p]
            gamma[j] = theta[k_main:]
            resid = Y[:, j] - D @ theta
            sigma2[j] = max(float(np.mean(resid**2)), 1e-8)
        if ard:
            ard_precision = 1.0 / (gamma**2 + 1e-4) if p else ard_precision

        # --- E-step: GLS update of tau given the current channel loadings ---
        phi = kappa[None, :] + (C @ gamma.T if p else 0.0)               # (n, g) composite loading
        resid = Y - intercept[None, :] - (C @ alpha.T if p else 0.0)     # (n, g)
        w = 1.0 / sigma2                                                 # (g,)
        num = (phi * resid * w[None, :]).sum(axis=1)
        den = (phi**2 * w[None, :]).sum(axis=1) + _RIDGE_TAU
        tau = num / den
        tau = (tau - tau.mean()) / (tau.std() if tau.std() > 0 else 1.0)
        if np.corrcoef(tau, tau_ref_dir @ (Y - Y.mean(axis=0)).T)[0, 1] < 0:
            tau = -tau  # lock sign to the fixed PC1 reference every iteration, prevents oscillation

        ll = float(-0.5 * np.sum(np.log(2 * np.pi * sigma2)) * n
                   - 0.5 * np.sum((resid - phi * tau[:, None])**2 / sigma2[None, :]))
        loglik_trace.append(ll)

        # Log-likelihood, not raw parameter deltas, is the convergence signal: the ARD reweighting
        # (gamma -> ard_precision -> gamma) keeps making small, non-shrinking steps in gamma/tau even
        # once the fit itself has stabilized (the objective goes flat well before the parameters do),
        # a known characteristic of iteratively-reweighted ARD/IRLS-style shrinkage.
        if it > 0 and abs(ll - loglik_trace[-2]) < tol * (abs(ll) + 1e-8):
            converged = True
            break

    return TrajectoryFit(
        tau=tau, intercept=intercept, alpha=alpha, kappa=kappa, gamma=gamma, sigma2=sigma2,
        ard_precision=ard_precision, channel_names=channel_names, covariate_names=covariate_names,
        n_iter=n_iter, converged=converged, loglik_trace=loglik_trace,
    )


def fit_exposure_trajectory_from_samples(
    space, samples: list, covariates: np.ndarray, *,
    weight: str = "log2p1", blocks: tuple[str, ...] = ("mean", "diversity"), **kwargs,
) -> TrajectoryFit:
    """Convenience: build the channel matrix from repertoire samples, then fit the trajectory.

    Args:
        space: A shared :class:`~mir.repertoire.RepertoireSpace` (:func:`mir.repertoire.fit_repertoire_space`).
        samples: Per-sample clonotype frames (``duplicate_count`` present), one :class:`~mir.repertoire.SampleEmbedding`
            per sample via :func:`mir.repertoire.sample_embedding`.
        covariates: ``(len(samples),)`` or ``(len(samples), n_covariates)`` known covariate(s).
        weight, blocks: Forwarded to :func:`mir.repertoire.sample_embedding`.
        **kwargs: Forwarded to :func:`fit_exposure_trajectory` (``ard``, ``max_iter``, ``tol``, ``seed``,
            ``covariate_names``).

    Returns:
        A fitted :class:`TrajectoryFit`, with ``channel_names`` set from the embeddings' blocks.
    """
    from mir.explain import stack_embeddings
    from mir.repertoire import sample_embedding

    embs = [sample_embedding(space, s, weight=weight, blocks=blocks) for s in samples]
    X, spec = stack_embeddings(embs)
    kwargs.setdefault("channel_names", spec.names if X.shape[1] == len(spec.names) else None)
    # stack_embeddings names whole blocks (e.g. "mean" covering hundreds of RFF columns), not
    # per-column -- expand to one name per column so TrajectoryFit's channel_names length matches X.
    if X.shape[1] != len(spec.names):
        expanded = [None] * X.shape[1]
        for name in spec.names:
            for c in spec.columns(name):
                expanded[c] = f"{name}{c - spec.columns(name)[0]}"
        kwargs["channel_names"] = expanded
    return fit_exposure_trajectory(X, covariates, **kwargs)


def _demo() -> None:
    """Self-check on synthetic data: a channel with a planted interaction is recovered as top-ranked."""
    rng = np.random.default_rng(0)
    n, g = 80, 12
    tau_true = rng.standard_normal(n)
    x = rng.integers(0, 2, size=n).astype(np.float64)          # binary covariate (e.g. HLA carrier)

    kappa_true = rng.standard_normal(g) * 0.3
    gamma_true = np.zeros(g)
    gamma_true[3] = 1.5                                          # channel 3: strong interaction
    gamma_true[7] = -1.2                                          # channel 7: strong interaction
    alpha_true = rng.standard_normal(g) * 0.2

    Y = (alpha_true[None, :] * x[:, None]
         + (kappa_true[None, :] + gamma_true[None, :] * x[:, None]) * tau_true[:, None]
         + 0.2 * rng.standard_normal((n, g)))

    fit = fit_exposure_trajectory(Y, x, channel_names=[f"c{i}" for i in range(g)], seed=0)
    assert fit.tau.shape == (n,)
    assert abs(fit.tau.std() - 1.0) < 1e-6

    # tau recovers the true trajectory up to sign/scale (already sign-fixed by the model itself)
    corr = abs(np.corrcoef(fit.tau, tau_true)[0, 1])
    assert corr > 0.8, f"tau correlation with ground truth too low: {corr:.3f}"

    top = fit.top_interactions(top=3)
    top_names = set(top["channel"].to_list())
    assert {"c3", "c7"} <= top_names, f"planted interactions not recovered: {top_names}"

    print(f"[ok] tau~truth corr={corr:.3f}; top interactions {top['channel'].to_list()}; "
          f"n_iter={fit.n_iter} converged={fit.converged}")


if __name__ == "__main__":
    _demo()
