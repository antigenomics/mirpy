"""Sample-level (repertoire) embedding: one fixed vector per repertoire (Theory §T.7).

A repertoire is an order-invariant multiset of clonotypes with clone counts,
``S = {(σ, a_σ)}``. We embed it as the *empirical measure* ``ρ_S = Σ_σ w_σ δ_{z_σ}`` on
the clonotype embedding space (``z_σ = φ(σ)`` from :class:`~mir.embedding.tcremp.TCREmp`),
with concave frequency weights ``w_σ = g(a_σ) / Σ_τ g(a_τ)`` (``g`` = ``log1p`` by default,
so one hyperexpanded clone can't dominate). ``Φ(S)`` is a sketch of ``ρ_S`` in three blocks,
each owning one requirement (appendix §T.7):

* **mean** ``Φ₁ = Σ_σ w_σ ψ(z_σ)`` — the random-Fourier-feature **kernel mean embedding**
  (order-invariant + depth-robust: converges to the population mean map at rate
  ``n_eff^{-1/2}``, ``n_eff = (Σ w²)⁻¹``). Distance ``‖Φ₁(S)−Φ₁(S')‖ ≈ MMD``. Codebook-free —
  no ``K``, no clustering (Prop. ``prop:kme``/``prop:codebook``).
* **diversity** ``Φ₂ = {⁰D, ¹D, ²D}`` (Hill numbers), optionally coverage-standardized to a
  common Good–Turing coverage ``Ĉ*`` via :mod:`vdjtools.stats.inext`; ``n_eff`` is itself a
  Hill number, ``n_eff ∈ [²D, ⁰D]`` (Prop. ``prop:antag``).
* **second** ``Σ_S = Σ_σ w_σ ψ₂(z_σ) ψ₂(z_σ)ᵀ`` — the codebook-free Fisher vector carrying
  clonotype co-occurrence / HLA-linked public structure (Prop. ``prop:interact``); computed on
  a small RFF so its upper triangle stays cheap.

The comparability invariant (as in :class:`mir.ml.bundle.CodecBundle` and
:func:`mir.density.fit_density_space`): every sample in a cohort must be embedded through
**one** prototype set and **one** PCA + RFF basis, or the measures are incomparable.
:func:`fit_repertoire_space` fits that basis once on the pooled clonotype cloud;
:class:`RepertoireSpace` serializes it and refuses a prototype-hash mismatch on load.

Torch-free (numpy / sklearn; vdjtools + the learned set-network in :mod:`mir.ml` are lazy).

Typical usage::

    from mir.repertoire import fit_repertoire_space, sample_embedding, mmd_distance
    from mir.embedding.tcremp import TCREmp
    import polars as pl

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
    pooled = pl.concat(samples)                    # all clonotypes across the cohort
    space = fit_repertoire_space(model, pooled)    # ONE basis for the cohort
    embs = [sample_embedding(space, s) for s in samples]
    d = mmd_distance(embs[0], embs[1])             # ≈ MMD between the two repertoires
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from mir.density import DensitySpace, _WEIGHTS, _embed, calibrate_radius

_COUNT = "duplicate_count"


# --------------------------------------------------------------------------- RFF


@dataclass
class RandomFourierFeatures:
    """Random Fourier features for a Gaussian kernel ``k(z,z')=exp(−‖z−z'‖²/2ℓ²)``.

    ``ψ(z) = √(2/D)·cos(Ωzᵀ + b)`` with ``Ω ~ N(0, 1/ℓ²)``, ``b ~ U[0, 2π]``, so that
    ``E[ψ(z)·ψ(z')] = k(z,z')`` (Rahimi–Recht 2007). Bandwidth ``ℓ`` is set to the
    one-substitution embedding scale ``r₁`` so the kernel resolves ~one CDR3 mutation.
    """

    omega: np.ndarray       # (p, D)
    b: np.ndarray           # (D,)
    length_scale: float

    @property
    def dim(self) -> int:
        return self.omega.shape[1]

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Map ``(n, p)`` embedding rows to ``(n, D)`` random features."""
        proj = np.asarray(Z, dtype=np.float64) @ self.omega + self.b
        return np.sqrt(2.0 / self.dim) * np.cos(proj)


def _make_rff(dim: int, n_rff: int, length_scale: float, seed: int) -> RandomFourierFeatures:
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((dim, n_rff)) / length_scale
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_rff)
    return RandomFourierFeatures(omega, b, float(length_scale))


# ------------------------------------------------------------------ RepertoireSpace


@dataclass
class RepertoireSpace:
    """One fitted clonotype→PCA→RFF basis shared by a whole cohort.

    ``clono`` projects a clonotype frame into the shared PCA coordinates; ``rff`` lifts those
    into the kernel-mean feature space (mean block); ``rff2`` is a smaller RFF for the
    second-moment block. ``meta`` records the embedding-space identity (prototype hash + knobs)
    so :meth:`load` can refuse an incomparable basis.
    """

    clono: DensitySpace                     # clonotype transform (model + scaler + PCA)
    rff: RandomFourierFeatures              # mean-block features
    rff2: RandomFourierFeatures | None      # second-moment features (smaller)
    meta: dict

    def transform_clonotypes(self, df: pl.DataFrame) -> np.ndarray:
        """Project a clonotype frame into the shared PCA coordinate system."""
        return self.clono.transform(df)

    def sample_cloud(self, df: pl.DataFrame, *, weight: str = "log1p"):
        """A sample as ``(Z, w)``: PCA-coord clonotypes ``Z`` and normalized weights ``w=g(a)/Σg``.

        The raw material for both the fixed kernel mean (:func:`sample_embedding`) and the learned
        set network (:mod:`mir.ml.set_encoder`) — same basis, so their embeddings are comparable.
        """
        a = df[_COUNT].to_numpy().astype(np.float64)
        g = _WEIGHTS[weight](a)
        return self.transform_clonotypes(df), g / g.sum()

    def save(self, path) -> None:
        """Pickle the basis (scaler + PCA + RFF + meta); the model is reconstructed on load."""
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(
                {"meta": self.meta, "space": self.clono.space, "scaler": self.clono.scaler,
                 "pca": self.clono.pca, "rff": self.rff, "rff2": self.rff2}, fh)

    @classmethod
    def load(cls, path, *, verify: bool = True) -> "RepertoireSpace":
        """Load a basis, rebuilding the model and verifying prototype comparability."""
        import pickle

        from mir.embedding.tcremp import TCREmp
        from mir.ml.bundle import prototype_hash

        with open(path, "rb") as fh:
            d = pickle.load(fh)
        m = d["meta"]
        if verify:
            cur = prototype_hash(m["species"], m["locus"], m["n_prototypes"])
            if cur != m["prototype_hash"]:
                raise ValueError(
                    f"prototype hash mismatch for {m['species']}_{m['locus']}: this space was "
                    "fit on a different prototype set — its embeddings are NOT comparable to the "
                    "current prototypes. Pass verify=False only if the difference is intentional."
                )
        model = TCREmp.from_defaults(
            m["species"], m["locus"], m["n_prototypes"], mode=m["mode"],
            metric=m["metric"], gap_positions=tuple(m["gap_positions"]))
        clono = DensitySpace(model=model, space=d["space"], scaler=d["scaler"], pca=d["pca"])
        return cls(clono, d["rff"], d["rff2"], m)


def fit_repertoire_space(
    model,
    cohort_df: pl.DataFrame,
    *,
    n_rff: int = 2048,
    n_rff_second: int = 128,
    n_eigs: int | None = None,
    length_scale: float | None = None,
    n_components: int | None = None,
    space: str = "full",
    pca_fit_cap: int | None = 200_000,
    seed: int = 0,
) -> RepertoireSpace:
    """Fit ONE PCA + RFF basis on the pooled clonotype cloud of a cohort.

    Args:
        model: A fitted single-chain :class:`~mir.embedding.tcremp.TCREmp`.
        cohort_df: Pooled clonotypes across all samples (``v_call``/``j_call``/``junction_aa``);
            counts are ignored here — only the geometry of the cloud sets the basis.
        n_rff: Mean-block RFF dimension ``D`` (~1–4k; §T.7 ``tab:sample``).
        n_rff_second: Second-moment-block RFF dimension ``D₂`` (kept small — the block stores
            ``D₂(D₂+1)/2`` upper-triangle entries, or its top-``n_eigs`` eigenvalues).
        n_eigs: If set, the second-moment block keeps the **top-``n_eigs`` eigenvalues** of the
            weighted covariance ``Σ_σ w_σ ψ₂ψ₂ᵀ`` (a compact, rotation-invariant spectral signature)
            instead of its full ``D₂(D₂+1)/2`` upper triangle. ``None`` (default) keeps the upper
            triangle — unchanged behaviour. Must satisfy ``0 < n_eigs ≤ n_rff_second``.
        length_scale: Gaussian-kernel bandwidth ``ℓ``. ``None`` → :func:`mir.density.calibrate_radius`
            (the one-substitution scale ``r₁``), so the kernel resolves ~one CDR3 mutation.
        n_components: PCA dimensionality; ``None`` → ``get_preset(species, locus).n_components``.
        space: ``"full"`` (V+J+junction) or ``"junction"`` (CDR3 sub-block only).
        pca_fit_cap: Fit the scaler + PCA on at most this many randomly-sampled pooled rows.
        seed: RNG seed (PCA solver + RFF draws).

    Returns:
        A :class:`RepertoireSpace` ready for :func:`sample_embedding`.
    """
    from mir.embedding.presets import get_preset
    from mir.ml.bundle import prototype_hash
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if n_eigs is not None and not (0 < n_eigs <= n_rff_second):
        raise ValueError(f"n_eigs must be in (0, n_rff_second={n_rff_second}], got {n_eigs}")

    if n_components is None:
        n_components = get_preset(model.species, model.locus).n_components

    X = _embed(model, cohort_df, space)
    if pca_fit_cap is not None and X.shape[0] > pca_fit_cap:
        rng = np.random.default_rng(seed)
        X_fit = X[rng.choice(X.shape[0], pca_fit_cap, replace=False)]
    else:
        X_fit = X
    scaler = StandardScaler().fit(X_fit)
    k = min(n_components, X_fit.shape[0], X.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(scaler.transform(X_fit))
    clono = DensitySpace(model=model, space=space, scaler=scaler, pca=pca)

    if length_scale is None:
        length_scale = calibrate_radius(clono, seed=seed)
    rff = _make_rff(k, n_rff, length_scale, seed)
    rff2 = _make_rff(k, n_rff_second, length_scale, seed + 1) if n_rff_second else None

    meta = {
        "species": model.species, "locus": model.locus, "n_prototypes": model.n_prototypes,
        "mode": model.mode, "metric": model.metric, "gap_positions": list(model._gap_positions),
        "prototype_hash": prototype_hash(model.species, model.locus, model.n_prototypes),
        "space": space, "n_components": k, "n_rff": n_rff, "n_rff_second": n_rff_second,
        "n_eigs": n_eigs, "length_scale": float(length_scale), "seed": seed,
    }
    return RepertoireSpace(clono, rff, rff2, meta)


# --------------------------------------------------------------------- embedding


@dataclass
class SampleEmbedding:
    """One repertoire's fixed-width embedding, kept block-wise so MMD uses only the mean block."""

    mean: np.ndarray                    # Φ₁, the kernel mean (n_rff,)
    diversity: np.ndarray | None        # Φ₂, [log ⁰D, log ¹D, log ²D, Ĉ]
    second: np.ndarray | None           # Σ_S: upper-tri (D₂·(D₂+1)/2,) or top-n_eigs eigvals (n_eigs,)
    n_eff: float                        # (Σ w²)⁻¹ — a Hill number in [²D, ⁰D]

    @property
    def vector(self) -> np.ndarray:
        """The concatenated feature vector (the multimodal-fusion tensor)."""
        parts = [self.mean]
        if self.diversity is not None:
            parts.append(self.diversity)
        if self.second is not None:
            parts.append(self.second)
        return np.concatenate(parts).astype(np.float64)


def _hill(f: np.ndarray) -> tuple[float, float, float]:
    """Observed Hill numbers ``(⁰D, ¹D, ²D)`` from a frequency vector."""
    f = f[f > 0]
    d0 = float(f.size)                                   # richness
    d1 = float(np.exp(-np.sum(f * np.log(f))))           # exp(Shannon)
    d2 = float(1.0 / np.sum(f * f))                      # inverse Simpson
    return d0, d1, d2


def _diversity_block(counts: np.ndarray, coverage: float | None) -> np.ndarray:
    """``[log ⁰D, log ¹D, log ²D, Ĉ]``; coverage-standardized to ``Ĉ*`` when given (vdjtools)."""
    f = counts / counts.sum()
    if coverage is None:
        d0, d1, d2 = _hill(f)
        chat = 1.0
    else:
        from vdjtools.stats.inext import estimate_d, sample_coverage

        est = estimate_d(counts.astype(np.int64), base="coverage", level=coverage,
                         q=(0, 1, 2), se=False)
        qd = {int(r): float(v) for r, v in zip(est["order_q"], est["qD"])}
        d0, d1, d2 = qd[0], qd[1], qd[2]
        chat = float(sample_coverage(counts.astype(np.int64)))
    return np.array([np.log(d0), np.log(d1), np.log(d2), chat], dtype=np.float64)


def sample_embedding(
    space: RepertoireSpace,
    sample_df: pl.DataFrame,
    *,
    weight: str = "log1p",
    blocks: tuple[str, ...] = ("mean", "diversity", "second"),
    coverage: float | None = None,
) -> SampleEmbedding:
    """Embed one repertoire into ``Φ(S)`` (mean ‖ diversity ‖ second).

    Args:
        space: A :class:`RepertoireSpace` from :func:`fit_repertoire_space`.
        sample_df: One sample's clonotypes with ``duplicate_count`` (counts drive the weights).
        weight: Concave clone-size weight ``g`` — ``"log1p"`` (default) / ``"anscombe"`` /
            ``"distinct"`` (``g≡1``, presence). Frequencies are ``w = g(a)/Σg`` (scale-free).
        blocks: Which blocks to compute/return.
        coverage: Common Good–Turing coverage ``Ĉ*`` for the diversity block; ``None`` uses the
            sample's observed Hill numbers.

    Returns:
        A :class:`SampleEmbedding`; ``.vector`` is the concatenated fixed-width tensor.
    """
    a = sample_df[_COUNT].to_numpy().astype(np.float64)
    g = _WEIGHTS[weight](a)
    w = g / g.sum()
    n_eff = float(1.0 / np.sum(w * w))

    mean = div = sec = None
    if "mean" in blocks or "second" in blocks:
        Z = space.transform_clonotypes(sample_df)          # (n, p) shared PCA coords
    if "mean" in blocks:
        mean = w @ space.rff.transform(Z)                  # Φ₁ = Σ w ψ(z)
    if "diversity" in blocks:
        div = _diversity_block(a, coverage)
    if "second" in blocks and space.rff2 is not None:
        psi2 = space.rff2.transform(Z)                     # (n, D₂)
        sigma = (psi2 * w[:, None]).T @ psi2               # Σ w ψ₂ψ₂ᵀ  (D₂, D₂), symmetric PSD
        n_eigs = space.meta.get("n_eigs")
        if n_eigs:
            # top-r eigenvalues (energy spectrum) — compact, rotation-invariant.
            # eigvalsh not eigh: we need the spectrum, not the eigenvectors.
            ev = np.linalg.eigvalsh(sigma)                 # ascending, D₂ values
            sec = ev[::-1][:n_eigs].copy()                 # top-r, descending
        else:
            iu = np.triu_indices(sigma.shape[0])
            sec = sigma[iu]
    return SampleEmbedding(mean=mean, diversity=div, second=sec, n_eff=n_eff)


# ---------------------------------------------------------------------- distance


def mmd_distance(a: SampleEmbedding, b: SampleEmbedding, *, unbiased: bool = False) -> float:
    """MMD between two repertoires, ``‖Φ₁(a) − Φ₁(b)‖`` (Eq. ``eq:kme``).

    The default (``unbiased=False``) is the biased V-statistic ``‖μ̂_a − μ̂_b‖`` — simple, but its
    self-terms carry a positive bias ``≈ 1/n_eff`` (from the ``k(z,z)`` diagonal), so a low-diversity
    (small ``n_eff``) sample gets its distances **inflated by construction**. When diversity is itself
    the variable of interest (e.g. divergence-vs-age), that bias masquerades as signal with the *wrong*
    sign (distance tracks low diversity, not high). ``unbiased=True`` removes the diagonal analytically
    using the stored ``n_eff`` (Gretton et al. 2012, unbiased MMD²) — the estimator to trust when
    comparing samples of unequal depth/diversity.
    """
    if not unbiased:
        return float(np.linalg.norm(a.mean - b.mean))
    sa, sb = 1.0 / a.n_eff, 1.0 / b.n_eff                   # Σwᵢ² ; RFF self-similarity k(z,z)≈1
    haa = (float(a.mean @ a.mean) - sa) / (1.0 - sa)        # diagonal-removed ‖μ‖²
    hbb = (float(b.mean @ b.mean) - sb) / (1.0 - sb)
    return float(np.sqrt(max(haa + hbb - 2.0 * float(a.mean @ b.mean), 0.0)))


def mmd_matrix(embs: list[SampleEmbedding], *, unbiased: bool = False) -> np.ndarray:
    """Symmetric sample×sample MMD matrix (feeds a regressor / ``cluster_samples``).

    ``unbiased=True`` uses the diagonal-removed MMD² (see :func:`mmd_distance`) — necessary whenever
    samples differ in depth/``n_eff``, else the ``1/n_eff`` self-bias confounds the comparison.
    """
    M = np.stack([e.mean for e in embs])                   # (S, D)
    G = M @ M.T
    sq = np.diag(G).copy()
    if unbiased:
        s = np.array([1.0 / e.n_eff for e in embs])        # per-sample Σwᵢ²
        h = (sq - s) / (1.0 - s)                           # diagonal-removed self-inner-products
        d2 = h[:, None] + h[None, :] - 2.0 * G
    else:
        d2 = sq[:, None] + sq[None, :] - 2.0 * G
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def class_witness(
    space: RepertoireSpace,
    pos: list[pl.DataFrame],
    neg: list[pl.DataFrame],
    candidates: pl.DataFrame,
    *,
    weight: str = "log1p",
    top: int = 30,
) -> pl.DataFrame:
    """Rank clonotypes by how much they drive the ``pos`` vs ``neg`` group difference (Prop. ``prop:witness``).

    The MMD witness is the mean-embedding difference ``w = μ_pos − μ_neg`` in RFF feature space; a
    clonotype ``σ`` scores ``s(σ) = ⟨w, ψ(φ(σ))⟩``, so the top-scoring candidates are the discriminative
    **public clones / motifs** separating the two groups — the supervised way to *find motifs* that the
    unsupervised bulk kernel mean cannot surface (it is swamped by the naive background).

    Args:
        space: The shared :class:`RepertoireSpace`.
        pos, neg: Per-group lists of sample clonotype frames (with ``duplicate_count``).
        candidates: Clonotype frame to score (e.g. all clonotypes seen in ``pos``).
        weight: Clone-size weight for the per-sample kernel means.
        top: Number of top motifs to return.

    Returns:
        ``candidates`` with a ``witness_score`` column, sorted descending, truncated to ``top``.
    """
    def group_mean(frames):
        return np.mean([w @ space.rff.transform(Z)
                        for Z, w in (space.sample_cloud(f, weight=weight) for f in frames)], axis=0)

    w = group_mean(pos) - group_mean(neg)
    psi = space.rff.transform(space.transform_clonotypes(candidates))     # (n, D)
    scores = psi @ w
    return (candidates.with_columns(pl.Series("witness_score", scores))
            .sort("witness_score", descending=True).head(top))


def hla_stratified_mmd(embs: list[SampleEmbedding], hla: list[set]) -> np.ndarray:
    """MMD restricted to HLA-matched sample pairs; ``nan`` where two samples share no allele.

    Response ridges are presented only by a specific HLA type, so cross-sample overlap is
    informative *only* within an HLA-matched stratum (Prop. ``prop:hla``). Pairs that share no
    HLA allele are masked to ``nan`` rather than compared.
    """
    D = mmd_matrix(embs)
    out = np.full_like(D, np.nan)
    for i in range(len(embs)):
        for j in range(len(embs)):
            if hla[i] & hla[j]:
                out[i, j] = D[i, j]
    return out


# --------------------------------------------------------------------- self-check


def _demo() -> None:
    """Self-check on bundled prototypes: two injected cohorts separate; ``n_eff ∈ [²D, ⁰D]``."""
    from mir.embedding.tcremp import TCREmp

    # RFF kernel approximation
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((200, 8))
    rff = _make_rff(8, 20000, length_scale=1.5, seed=0)
    P = rff.transform(Z)
    approx = P[0] @ P[5]
    exact = np.exp(-np.sum((Z[0] - Z[5]) ** 2) / (2 * 1.5 ** 2))
    assert abs(approx - exact) < 0.05, f"RFF kernel approx off: {approx:.3f} vs {exact:.3f}"

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=300)
    protos = pl.DataFrame({"v_call": model._proto_v, "j_call": model._proto_j,
                           "junction_aa": model._proto_junction})
    space = fit_repertoire_space(model, protos, n_rff=1024, n_components=20, seed=0)

    # two cohorts: A = random draws from the pool; B = A enriched with a repeated clonotype block
    base = protos.sample(150, seed=1).with_columns(pl.lit(1.0).alias(_COUNT))
    spike = protos.slice(0, 5).with_columns(pl.lit(500.0).alias(_COUNT))   # a public expansion
    A = [sample_embedding(space, base.sample(120, seed=s)) for s in range(4)]
    B = [sample_embedding(space, pl.concat([base.sample(120, seed=s), spike])) for s in range(4)]

    within = np.mean([mmd_distance(A[i], A[j]) for i in range(4) for j in range(i + 1, 4)])
    between = np.mean([mmd_distance(a, b) for a in A for b in B])
    assert between > within, f"cohorts not separated: between={between:.3f} within={within:.3f}"

    for e in A + B:
        emb = sample_embedding(space, base.sample(120, seed=7))
        d0 = np.exp(emb.diversity[0])
        d2 = np.exp(emb.diversity[2])
        assert d2 - 1e-6 <= emb.n_eff <= d0 + 1e-6, f"n_eff {emb.n_eff} not in [{d2}, {d0}]"

    # opt-in spectral (top-r eigval) interaction block: exactly r values, non-negative, descending
    space_eig = fit_repertoire_space(model, protos, n_rff=1024, n_rff_second=64,
                                     n_eigs=8, n_components=20, seed=0)
    sec = sample_embedding(space_eig, base.sample(120, seed=1)).second
    assert sec.shape == (8,), f"top-r block shape {sec.shape} != (8,)"
    assert np.all(sec >= -1e-9) and np.all(np.diff(sec) <= 1e-9), "eigvals not non-neg descending"

    print(f"[ok] RFF kernel approx {approx:.3f}≈{exact:.3f}; "
          f"cohort MMD between={between:.3f} > within={within:.3f}; n_eff∈[²D,⁰D] holds; "
          f"top-8 eigval block {sec.shape}")


if __name__ == "__main__":
    _demo()
