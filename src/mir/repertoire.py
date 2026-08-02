"""Sample-level (repertoire) embedding: one fixed vector per repertoire (Theory §T.7).

A repertoire is an order-invariant multiset of clonotypes with clone counts,
``S = {(σ, a_σ)}``. We embed it as the *empirical measure* ``ρ_S = Σ_σ w_σ δ_{z_σ}`` on
the clonotype embedding space (``z_σ = φ(σ)`` from :class:`~mir.embedding.tcremp.TCREmp`),
with clone-size weights ``w_σ = g(a_σ) / Σ_τ g(a_τ)`` (``g`` = ``log2p1`` — ``log2(1+a)`` — by
default, so one hyperexpanded clone can't dominate; ``"duplicate_count"`` weights linearly by clone
size and ``"distinct"`` ignores size entirely, ``g≡1``). ``Φ(S)`` is a sketch of ``ρ_S`` in three blocks,
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

**Sub-probability embeddings.** ``Φ₁`` above is the kernel mean of a *probability* measure — the
weights are normalised to sum to 1, so every sample asserts one full unit of confidence. At RNA-seq
depth that premise fails: a tissue sample holding 21 unique clonotypes has ``w_σ = 1/n`` for a
technical draw size, not a clonal frequency, and normalising **forces** a 5-clonotype tumour to land
somewhere arbitrary on the unit sphere. ``sample_embedding(missing_mass="turing"|"chao")`` instead
estimates the probability mass of the clonotypes that were *never drawn* (:func:`missing_mass`) and
records the retained mass on :attr:`SampleEmbedding.mass` ≤ 1 — a **sub-probability** measure, which
keeps everything that makes ``Φ`` useful (``‖Φ_P − Φ_Q‖ = MMD``, convex combinations are real pooled
repertoires) while a signed measure would not. The unseen block gets a principled location from
:func:`naive_reference` (the germline recombination model, *not* the corpus centroid — measured
worse), and :func:`contrast_embedding` composes the two into ``Ψ_S = mass·(Φ_S − naive)``, where
magnitude = confidence × deviation-from-naive and an immune desert lands at the origin instead of
being filtered out. See :func:`mir.bench.recovery_report` for the "is this object honest" check.

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

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from mir.density import DensitySpace, _WEIGHTS, _embed, calibrate_radius

_COUNT = "duplicate_count"
_MISSING_MASS = ("none", "turing", "chao")


def missing_mass(counts, method: str = "chao") -> float:
    """Estimated probability mass ``M₀`` of the clonotypes that were **never drawn**.

    Both estimators feed the same identity with a different ``M₀``; they rank samples nearly
    identically (measured ``r`` = 0.93 blood TRB, 0.76 tissue IGH) but differ in level — Chao runs
    above Turing when singletons dominate doubletons (``f₁ ≫ 2f₂``) and below it when doubletons are
    abundant. Measured means: blood TRB 0.552 (Turing) vs 0.649 (Chao), tissue IGH 0.189 vs 0.458::

        none    M0 = 0                        the sample is treated as complete
        turing  M0 = f1 / N                   Good-Turing, read-weighted
        chao    M0 = S_u / (N + S_u)          S_u = f1(f1-1) / (2(f2+1))

    ``f₁``/``f₂`` are the singleton / doubleton clonotype counts and ``N`` the total reads. The
    ``chao`` form rests on one modelling assumption: *an unseen clonotype is rare, so had it been
    drawn it would carry at most one read, and unseen clonotypes do not overlap.* That pins each
    unseen clone's frequency at the detection boundary ``1/(N + S_u)`` and is what makes the units
    legal — ``N`` counts reads and ``S_u`` counts clonotypes, and adding them is only defensible
    because each unseen clone contributes exactly one read. The observed term then collapses to
    exactly ``a_σ/(N + S_u)``.

    ``S_u`` is the **bias-corrected** Chao1 ``f1(f1-1)/(2(f2+1))``, never the classical
    ``f1²/(2f2)``: at these depths a sample with no clonotype seen exactly twice is common, and the
    classical form is undefined there.

    Args:
        counts: Per-clonotype read counts (``duplicate_count``); rounded to integers to count
            singletons and doubletons.
        method: ``"none"`` | ``"turing"`` | ``"chao"``.

    Returns:
        ``M₀ ∈ [0, 1]`` — the missing (unobserved) mass. The retained mass is ``1 − M₀``.

    Raises:
        ValueError: If ``method`` is not recognised.

    Example:
        >>> missing_mass([5000, 3000, 2000], "chao")     # deep, no singletons
        0.0
    """
    if method not in _MISSING_MASS:
        raise ValueError(f"missing_mass must be one of {_MISSING_MASS}, got {method!r}")
    if method == "none":
        return 0.0
    a = np.rint(np.asarray(counts, dtype=np.float64))
    n_reads = float(a.sum())
    f1 = float(np.count_nonzero(a == 1))
    if method == "turing":
        return 0.0 if n_reads <= 0 else min(f1 / n_reads, 1.0)
    f2 = float(np.count_nonzero(a == 2))
    s_u = f1 * (f1 - 1.0) / (2.0 * (f2 + 1.0))          # +1: never divides by zero
    total = n_reads + s_u
    return 0.0 if total <= 0 else float(s_u / total)


def sample_statistics(sample_df: pl.DataFrame) -> dict:
    """The **sampling fingerprint** of one repertoire — the basic statistics, from counts alone.

    Two uses. (a) The targets :func:`mir.bench.recovery_report` asks the embedding to *carry*: if a
    statistic is recoverable from ``Φ``, nothing has to be bolted on beside it. (b) Features in their
    own right — the abundance classes ``f1``/``f2``/``f3plus`` and the top-clone fraction are
    clonality and expansion, i.e. candidate biology, not only depth nuisance.

    Args:
        sample_df: One sample's clonotypes with ``duplicate_count``.

    Returns:
        ``{name: float}`` with ``n_reads``, ``log_reads``, ``richness``, ``log_richness``, ``f1``,
        ``f2``, ``f3plus``, ``singleton_fraction``, ``top_clone_fraction``, ``shannon``,
        ``missing_mass_turing``, ``missing_mass_chao``.

    Raises:
        ValueError: On an empty or all-zero-count repertoire (via :func:`_sample_weights`).
    """
    a, _, _ = _sample_weights(sample_df, "duplicate_count")
    ai = np.rint(a)
    p = a / a.sum()
    return {
        "n_reads": float(a.sum()),
        "log_reads": float(np.log1p(a.sum())),
        "richness": float(a.size),
        "log_richness": float(np.log(a.size)),
        "f1": float(np.count_nonzero(ai == 1)),
        "f2": float(np.count_nonzero(ai == 2)),
        "f3plus": float(np.count_nonzero(ai >= 3)),
        "singleton_fraction": float(np.count_nonzero(ai == 1) / a.size),
        "top_clone_fraction": float(p.max()),
        "shannon": float(-np.sum(p * np.log(p))),
        "missing_mass_turing": missing_mass(a, "turing"),
        "missing_mass_chao": missing_mass(a, "chao"),
    }


def cohort_statistics(frames) -> dict:
    """Per-statistic arrays over a cohort — :func:`sample_statistics` transposed.

    Args:
        frames: One clonotype frame per sample, in the same order as the embedding rows.

    Returns:
        ``{name: (n_samples,) array}``, ready as :func:`mir.bench.recovery_report`'s ``stats``.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not len(frames):
        raise ValueError("frames is empty")
    rows = [sample_statistics(f) for f in frames]
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}


def _sample_weights(sample_df: pl.DataFrame, weight: str):
    """Validated ``(counts a, concave g, normalized weights w=g/Σg)`` for one sample.

    Raises on the degenerate inputs that would otherwise propagate a silent ``NaN``/``inf`` through
    every block: an empty repertoire (no clonotypes) or all-zero ``duplicate_count`` (``g.sum()==0``
    ⇒ ``w`` all-``NaN``, ``n_eff=inf``).
    """
    if sample_df.height == 0:
        raise ValueError("empty repertoire: sample_df has no clonotypes")
    a = sample_df[_COUNT].to_numpy().astype(np.float64)
    if not (a.sum() > 0):
        raise ValueError("degenerate repertoire: duplicate_count must sum to a positive value")
    g = _WEIGHTS[weight](a)
    return a, g, g / g.sum()


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
        # in-place after the matmul: the (n, D) block is the memory bottleneck of a deep
        # sample (n=2e5, D=2048 → 3 GB), and the naive expression peaks at ~3× that.
        proj = np.asarray(Z, dtype=np.float64) @ self.omega
        proj += self.b
        np.cos(proj, out=proj)
        proj *= np.sqrt(2.0 / self.dim)
        return proj


def _make_rff(dim: int, n_rff: int, length_scale: float, seed: int) -> RandomFourierFeatures:
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((dim, n_rff)) / length_scale
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_rff)
    return RandomFourierFeatures(omega, b, float(length_scale))


# ------------------------------------------------------------------ RepertoireSpace


def _check_rebuildable(model) -> None:
    """Refuse to serialize a basis whose model ``load`` cannot rebuild identically.

    ``load`` reconstructs the ``TCREmp`` from ``meta`` (species/locus/n_prototypes/mode/metric/
    gap_positions). A custom ``matrix=`` or ``alignment="sw"`` is *not* in ``meta`` and is *not*
    covered by the prototype hash, so it would come back as the default gapblock/BLOSUM62 space —
    a different coordinate system that passes verification. Fail at save time instead.
    """
    if getattr(model, "_matrix", None) is not None or getattr(model, "_alignment", "gapblock") != "gapblock":
        raise ValueError(
            "cannot save a basis fit with a custom matrix= or alignment=: those knobs are not "
            "recorded, so loading would silently rebuild the default gapblock/BLOSUM62 space. "
            "Refit with the default coordinate system, or keep the fitted object in-process."
        )


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
    _naive: dict = field(default_factory=dict, repr=False, compare=False)  # naive_reference cache

    def transform_clonotypes(self, df: pl.DataFrame) -> np.ndarray:
        """Project a clonotype frame into the shared PCA coordinate system."""
        return self.clono.transform(df)

    def sample_cloud(self, df: pl.DataFrame, *, weight: str = "log2p1"):
        """A sample as ``(Z, w)``: PCA-coord clonotypes ``Z`` and normalized weights ``w=g(a)/Σg``.

        The raw material for both the fixed kernel mean (:func:`sample_embedding`) and the learned
        set network (:mod:`mir.ml.set_encoder`) — same basis, so their embeddings are comparable.
        """
        _, _, w = _sample_weights(df, weight)
        return self.transform_clonotypes(df), w

    def save(self, path) -> None:
        """Pickle the basis (scaler + PCA + RFF + meta); the model is reconstructed on load.

        Raises:
            ValueError: If the model uses a non-default ``matrix`` / ``alignment``. Neither is
                recorded in ``meta``, so :meth:`load` would silently rebuild a *different*
                coordinate system while the prototype-hash check still passed.
        """
        import pickle

        _check_rebuildable(self.clono.model)
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
        replicate = m.get("replicate", 0)      # .get: spaces predating replicates are draw 0
        if verify:
            cur = prototype_hash(m["species"], m["locus"], m["n_prototypes"], replicate)
            if cur != m["prototype_hash"]:
                raise ValueError(
                    f"prototype hash mismatch for {m['species']}_{m['locus']}: this space was "
                    "fit on a different prototype set — its embeddings are NOT comparable to the "
                    "current prototypes. Pass verify=False only if the difference is intentional."
                )
        model = TCREmp.from_defaults(
            m["species"], m["locus"], m["n_prototypes"], mode=m["mode"], replicate=replicate,
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

    # Fit the scaler+PCA on at most pca_fit_cap pooled rows. Sample the FRAME rows and embed only
    # those — bit-identical to embedding all then subsetting (embedding is row-wise, same seed picks
    # the same indices) but never materializes the full raw matrix for a huge cohort.
    if pca_fit_cap is not None and cohort_df.height > pca_fit_cap:
        rng = np.random.default_rng(seed)
        rows = rng.choice(cohort_df.height, pca_fit_cap, replace=False)
        X_fit = _embed(model, cohort_df[rows], space)
    else:
        X_fit = _embed(model, cohort_df, space)
    scaler = StandardScaler().fit(X_fit)
    k = min(n_components, X_fit.shape[0], X_fit.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(scaler.transform(X_fit))
    clono = DensitySpace(model=model, space=space, scaler=scaler, pca=pca)

    if length_scale is None:
        length_scale = calibrate_radius(clono, seed=seed)
    rff = _make_rff(k, n_rff, length_scale, seed)
    rff2 = _make_rff(k, n_rff_second, length_scale, seed + 1) if n_rff_second else None

    meta = {
        "species": model.species, "locus": model.locus, "n_prototypes": model.n_prototypes,
        "mode": model.mode, "metric": model.metric, "gap_positions": list(model._gap_positions),
        "replicate": getattr(model, "replicate", 0),
        "prototype_hash": prototype_hash(model.species, model.locus, model.n_prototypes,
                                         getattr(model, "replicate", 0)),
        "space": space, "n_components": k, "n_rff": n_rff, "n_rff_second": n_rff_second,
        "n_eigs": n_eigs, "length_scale": float(length_scale), "seed": seed,
    }
    return RepertoireSpace(clono, rff, rff2, meta)


def fit_repertoire_spaces(
    models: dict,
    cohort_frames: dict,
    *,
    min_clonotypes: int = 50,
    **kwargs,
) -> dict:
    """Fit one :class:`RepertoireSpace` **per locus** — the multi-chain (digital-donor) basis.

    Each locus gets an independent basis (its own prototype set + PCA + RFF), so per-chain kernel
    means are only ever compared within their own chain, honouring the comparability contract.
    Loci whose pooled cloud is too small to fit a stable PCA are skipped (returned dict omits them).

    Args:
        models: ``{locus: TCREmp}`` — one fitted single-chain model per locus.
        cohort_frames: ``{locus: pooled clonotype frame}`` (``v_call``/``j_call``/``junction_aa``),
            the pooled cloud for that locus across the whole cohort.
        min_clonotypes: Skip a locus whose pooled frame has fewer rows (PCA would be degenerate).
        **kwargs: Forwarded to :func:`fit_repertoire_space` (``n_rff``, ``n_components``, ``seed`` …).

    Returns:
        ``{locus: RepertoireSpace}`` for the loci that could be fit.
    """
    spaces: dict = {}
    for loc, model in models.items():
        pool = cohort_frames.get(loc)
        if pool is None or pool.height < min_clonotypes:
            continue
        spaces[loc] = fit_repertoire_space(model, pool, **kwargs)
    return spaces


# --------------------------------------------------------------------- embedding


@dataclass
class SampleEmbedding:
    """One repertoire's fixed-width embedding, kept block-wise so MMD uses only the mean block.

    ``mass`` is the retained probability mass ``1 − M₀``: ``1.0`` (the default) means the sample is
    treated as a complete probability measure, ``< 1`` that :func:`missing_mass` estimated part of
    the repertoire was never drawn. The blocks themselves are **unchanged** by ``mass`` — the
    deficiency is applied where it is meaningful, in :func:`contrast_embedding`.
    """

    mean: np.ndarray                    # Φ₁, the kernel mean (n_rff,)
    diversity: np.ndarray | None        # Φ₂, [log ⁰D, log ¹D, log ²D, Ĉ]
    second: np.ndarray | None           # Σ_S: upper-tri (D₂·(D₂+1)/2,) or top-n_eigs eigvals (n_eigs,)
    n_eff: float                        # (Σ w²)⁻¹ — a Hill number in [²D, ⁰D]
    mass: float = 1.0                   # retained mass 1 − M₀ (≤ 1; sub-probability when < 1)

    @property
    def vector(self) -> np.ndarray:
        """The concatenated feature vector (the multimodal-fusion tensor).

        Only the blocks that were computed are concatenated — ``blocks=("diversity",)`` is a valid
        (mean-less) Φ, even though MMD then is not (see :func:`mmd_distance`).
        """
        parts = [b for b in (self.mean, self.diversity, self.second) if b is not None]
        if not parts:
            raise ValueError("SampleEmbedding has no blocks; sample_embedding(blocks=...) was empty")
        return np.concatenate(parts).astype(np.float64)

    def _require_mean(self) -> np.ndarray:
        if self.mean is None:
            raise ValueError(
                "this SampleEmbedding has no mean (kernel-mean) block, so MMD is undefined — "
                "MMD is the distance between kernel means. Re-embed with 'mean' in blocks=."
            )
        return self.mean


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


_missing_mass = missing_mass          # the sample_embedding kwarg shadows the public name


def sample_embedding(
    space: RepertoireSpace,
    sample_df: pl.DataFrame,
    *,
    weight: str = "log2p1",
    blocks: tuple[str, ...] = ("mean", "diversity", "second"),
    coverage: float | None = None,
    missing_mass: str = "none",
) -> SampleEmbedding:
    """Embed one repertoire into ``Φ(S)`` (mean ‖ diversity ‖ second).

    Args:
        space: A :class:`RepertoireSpace` from :func:`fit_repertoire_space`.
        sample_df: One sample's clonotypes with ``duplicate_count`` (counts drive the weights).
        weight: Clone-size weight ``g`` — ``"log2p1"`` (default, ``g=log2(1+a)``) /
            ``"duplicate_count"`` (``g=a``, linear) / ``"distinct"`` (``g≡1``, presence) /
            ``"log1p"`` (``g=ln(1+a)``) / ``"anscombe"`` (``g=√(a+3/8)``). Frequencies are
            ``w = g(a)/Σg`` (scale-free).
        blocks: Which blocks to compute/return.
        coverage: Common Good–Turing coverage ``Ĉ*`` for the diversity block; ``None`` uses the
            sample's observed Hill numbers.
        missing_mass: Estimator for the mass of the never-drawn clonotypes (:func:`missing_mass`):
            ``"none"`` (default — a complete probability measure, ``mass=1``, **bit-identical** to
            pre-3.8 output), ``"turing"`` or ``"chao"``. Only ``.mass`` changes; the blocks do not.

    Returns:
        A :class:`SampleEmbedding`; ``.vector`` is the concatenated fixed-width tensor.
    """
    a, g, w = _sample_weights(sample_df, weight)
    n_eff = float(1.0 / np.sum(w * w))
    mass = 1.0 - _missing_mass(a, missing_mass)      # aliased: the kwarg shadows the public name

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
    return SampleEmbedding(mean=mean, diversity=div, second=sec, n_eff=n_eff, mass=mass)


# ------------------------------------------------------ functional diversity from Φ


def rao_q(emb: SampleEmbedding) -> float:
    """Rao's quadratic entropy of the repertoire, read straight off ``Φ₁`` as ``1 − ‖Φ₁‖²``.

    Rao's ``Q = Σ_{σ,τ} w_σ w_τ d(σ,τ)`` with the kernel dissimilarity ``d = 1 − k`` collapses, for
    weights summing to 1 and a normalised kernel (``k(z,z)=1``), to exactly ``1 − ‖Φ₁‖²`` — so the
    **norm** of the kernel mean *is* a diversity statistic and no Gram matrix is needed. Verified
    against an explicit Gram to machine precision (max relative error ~1e-16).

    This is the diversity the Hill block cannot express. Every Hill number is a functional of the
    clone-size distribution alone, hence invariant to permuting *which* receptor carries which
    abundance — it describes the shape of the abundance distribution and is blind to composition.
    Rao's ``Q`` weights each pair by how *different the receptors are*, so it is a **functional**
    diversity: sequence-aware, and already carried inside ``Φ``. Measured: this one scalar recovers
    R² 0.74–0.85 of classical diversity, while embedding derivatives reach R² 0.974–0.994 for Shannon
    and 0.985–0.9999 for richness.

    Two caveats. The kernel is a random-Fourier *approximation*, so ``k(z,z)=1`` holds only to
    ``O(D^{-1/2})`` and a single-clone sample reads ``Q`` of order ``1e-2`` rather than exactly 0 —
    widen ``n_rff`` if the absolute level matters, or read differences, which are unaffected. And it
    is valid **only for the true, uncentred** ``Φ₁``: Centring or PCA-projecting ``Φ₁`` preserves
    *differences* (so MMD survives) but not norms, so ``Q`` cannot be read off a centred/reduced
    identity block — check which object you hold before quoting any norm-based quantity. With a
    deficient ``mass`` this is the ``Q`` of the observed (renormalised) part, which is the honest
    reading: it is the diversity you measured.

    Args:
        emb: A :class:`SampleEmbedding` with its mean block (uncentred, as returned).

    Returns:
        ``Q ∈ [0, 1]`` — 0 for a single clone, rising as the receptors present grow more dissimilar.

    Raises:
        ValueError: If ``emb`` has no mean block.
    """
    m = emb._require_mean()
    return float(1.0 - m @ m)


@dataclass
class DepthThreshold:
    """Where sampling noise stops being smaller than between-sample signal (see :func:`depth_threshold`)."""

    kappa: float             # σ²/τ² — the effective size at which noise == signal
    tau2: float              # between-sample (biological) variance
    sigma2: float            # within-sample sampling variance, per unit 1/n
    fraction_below: float    # share of the cohort below kappa
    sizes: np.ndarray        # the per-sample n used

    def __repr__(self) -> str:      # arrays make the default repr unreadable
        return (f"DepthThreshold(kappa={self.kappa:.1f}, tau2={self.tau2:.4g}, "
                f"sigma2={self.sigma2:.4g}, fraction_below={self.fraction_below:.3f}, "
                f"n_samples={self.sizes.size})")


def depth_threshold(embs, sizes=None) -> DepthThreshold:
    """Estimate ``κ``, the sample size below which a repertoire's ``Φ`` is mostly sampling noise.

    The damage depth does to a kernel mean is not bias — it is **variance ∝ 1/n**, over an ``n``
    spanning four orders of magnitude, and in a neighbour graph that heteroscedasticity *is* a depth
    axis. So regress each sample's squared distance from the cohort centroid on ``1/n``:

    ``E‖Φ_S − Φ̄‖² ≈ τ² + σ²/n``

    The intercept ``τ²`` is between-sample (biological) spread, the slope ``σ²`` is within-sample
    sampling noise, and **``κ = σ²/τ²`` is the size at which the two are equal** — below it a
    sample's ``Φ`` carries more noise than signal. Measured across four independent views, ``κ`` came
    out at **40–70 clonotypes** every time, with 23–69% of samples below it.

    This is the estimable replacement for a hand-picked clonotype floor: report ``κ`` and
    ``fraction_below`` for the cohort in front of you rather than importing someone else's cutoff.
    Note the library still does **not** apply a floor — in blood a low clonotype count is shallow
    sequencing, in a tumour it is low infiltration (the phenotype of interest). Pair ``κ`` with
    :func:`missing_mass` / :func:`contrast_embedding` and weight instead of filtering.

    Args:
        embs: Per-sample :class:`SampleEmbedding` (mean block present), one cohort, one basis.
        sizes: Per-sample size to use as ``n``. ``None`` → each embedding's ``n_eff``. Pass observed
            clonotype richness to read ``κ`` in clonotypes rather than effective clonotypes.

    Returns:
        A :class:`DepthThreshold`. ``kappa`` is ``inf`` when the fit finds no between-sample spread
        (``τ² ≤ 0``) and ``0.0`` when it finds no depth dependence (``σ² ≤ 0``).

    Raises:
        ValueError: If fewer than three samples are given, or ``sizes`` has the wrong length.
    """
    M = np.stack([e._require_mean() for e in embs])
    if M.shape[0] < 3:
        raise ValueError(f"need at least 3 samples to fit a depth threshold, got {M.shape[0]}")
    n = (np.array([e.n_eff for e in embs], dtype=np.float64) if sizes is None
         else np.asarray(sizes, dtype=np.float64))
    if n.shape != (M.shape[0],):
        raise ValueError(f"sizes has shape {n.shape}, expected ({M.shape[0]},)")

    y = np.sum((M - M.mean(axis=0)) ** 2, axis=1)
    A = np.column_stack([np.ones_like(n), 1.0 / n])
    tau2, sigma2 = np.linalg.lstsq(A, y, rcond=None)[0]
    if sigma2 <= 0:
        kappa = 0.0
    elif tau2 <= 0:
        kappa = float("inf")
    else:
        kappa = float(sigma2 / tau2)
    return DepthThreshold(kappa=kappa, tau2=float(tau2), sigma2=float(sigma2),
                          fraction_below=float(np.mean(n < kappa)), sizes=n)


# ------------------------------------------------------- the location of the unseen


def naive_reference(
    space: RepertoireSpace,
    *,
    n: int = 20_000,
    seed: int = 0,
    sequences: pl.DataFrame | None = None,
) -> np.ndarray:
    """Kernel mean of ``n`` naive V(D)J recombinations — where the **unseen** clonotypes live.

    Once a sample's mass is deficient (:func:`missing_mass`), something has to occupy the unseen
    block, and the choice decides whether the object is a shrinkage estimator toward a *meaningful*
    point. Three priors were measured:

    * the sample's own singletons — begs the question, adds no information;
    * the **corpus mean** — James–Stein toward the centroid, and it measurably **hurt**: it piles
      shallow samples into a dense ball that is itself depth-correlated;
    * the **germline draw** — the one that works. Filling the unseen block with it dropped
      ``R²(PC1, depth)`` from 0.259 to 0.001 (blood TRB) and 0.067 to 0.006 (tissue IGH), with kNN
      label entropy unchanged or better and PC1's explained variance *unchanged* (so PC1 is a
      different direction, not a collapsed one).

    The draw comes from :func:`vdjtools.model.generate` over the bundled germline model for this
    space's locus (~8 s for 20,000 sequences), and is cached per ``(n, seed)`` on the space. Pass
    ``sequences=`` to supply the reference repertoire yourself (an uncached path) if you do not want
    the vdjtools generative model in the loop.

    Args:
        space: The shared :class:`RepertoireSpace` — supplies the locus *and* the coordinate system,
            so the reference is automatically comparable to every ``Φ`` fit through it.
        n: Number of naive recombinations to draw.
        seed: Generation seed; the result is deterministic in ``(n, seed)``.
        sequences: Optional pre-built clonotype frame to use instead of generating (``v_call`` /
            ``j_call`` / ``junction_aa``). Not cached.

    Returns:
        ``(n_rff,)`` unweighted kernel mean of the naive draw, in ``space``'s RFF coordinates.

    Raises:
        ValueError: If the space is non-human (only human germline models are bundled).

    Example:
        >>> ref = naive_reference(space)                      # doctest: +SKIP
        >>> psi = contrast_embedding(emb, ref)                # doctest: +SKIP
    """
    if sequences is None:
        key = (n, seed)
        if key in space._naive:
            return space._naive[key]
        if space.meta["species"] != "human":
            raise ValueError(
                f"no bundled germline model for species {space.meta['species']!r}; pass "
                "sequences= with your own naive draw"
            )
        from vdjtools.model import load_bundled
        from vdjtools.model.generate import generate

        sequences = generate(load_bundled(space.meta["locus"]), n, seed=seed, productive_only=True)
        ref = space.rff.transform(space.transform_clonotypes(sequences)).mean(axis=0)
        space._naive[key] = ref
        return ref
    return space.rff.transform(space.transform_clonotypes(sequences)).mean(axis=0)


def contrast_embedding(emb: SampleEmbedding, reference: np.ndarray) -> np.ndarray:
    """``Ψ_S = mass · (Φ₁(S) − reference)`` — the deficient measure as a signed contrast.

    This is the answer to "can the missing mass be a *negative* probability". It cannot: ``Φ``'s
    value is that ``‖Φ_P − Φ_Q‖ = MMD(P,Q)`` and that a convex combination of two ``Φ``'s is the
    ``Φ`` of a real pooled repertoire (what makes :mod:`mir.twin` and trajectory interpolation
    mean anything), and a measure allowed to go negative on a set is not a probability measure.
    But a signed **difference of two** probability measures already gives signed coordinates —
    negative wherever the sample is depleted relative to unselected recombination — and stays an
    ordinary RKHS element with ``‖Ψ_S‖ = MMD(S, naive)`` intact.

    Note ``Ψ_S`` is exactly the reference-centred sub-probability embedding: with
    ``Φ_true = mass·Φ₁ + M₀·reference``, ``Φ_true − reference = mass·(Φ₁ − reference)``.

    Combined with a deficient mass, **magnitude = confidence × deviation-from-naive**. An immune
    desert has ``M₀ → 1`` and lands at the **origin**, which is the correct place for "no infiltrate
    detected"; a vague shallow blood sample lands there too and says so *by its norm* rather than by
    being dropped from the cohort. Scale such a block with **one global scalar** — see
    :meth:`mir.explain.ChannelBuilder.add`'s ``preserve_magnitude``.

    Args:
        emb: A :class:`SampleEmbedding` with a mean block (embed with ``missing_mass=`` for
            ``mass < 1``; at the ``"none"`` default this is just the centred kernel mean).
        reference: ``(n_rff,)`` unseen-block location, normally :func:`naive_reference`.

    Returns:
        ``(n_rff,)`` signed contrast vector.

    Raises:
        ValueError: If ``emb`` has no mean block.
    """
    ref = np.asarray(reference, dtype=np.float64)
    return float(emb.mass) * (emb._require_mean() - ref)


# ------------------------------------------------------------ compartments (bands)

_ISOTYPE_BANDS = {
    "igm": ("IGHM", "IGHD"),                          # unswitched: naive + marginal zone
    "igg": ("IGHG1", "IGHG2", "IGHG3", "IGHG4"),      # class-switched, T-dependent, affinity-matured
    "iga": ("IGHA1", "IGHA2"),                        # mucosal
}
_SIZE_BANDS = ("singleton", "expanded", "top")


def band_frames(
    sample_df: pl.DataFrame,
    *,
    bands: tuple[str, ...] = _SIZE_BANDS,
    top_fraction: float = 0.01,
    top_clip: tuple[int, int] = (10, 500),
    min_clonotypes: int = 5,
) -> dict:
    """Partition one sample's clonotypes into **compartments** — abundance bands or IGH isotypes.

    ``Φ₁`` is a clone-size-weighted *average*, which is the right operation for estimating a
    population mean and the wrong one for detecting a **minority** signal. Writing the repertoire as
    a mixture ``ρ_S = (1−π)ρ_naive + π ρ_expanded`` gives ``Φ₁(S) = (1−π)Φ₁(naive) + π Φ₁(expanded)``,
    so a difference confined to the expanded compartment reaches ``Φ₁`` attenuated to ``πΔ`` while the
    *noise* is supplied by the naive compartment that owns most of the clonotypes. Embedding each
    compartment separately occupies exactly the blind spot of the Hill block, which is invariant to
    which receptor carries which abundance.

    Bands (on ``duplicate_count``, never on a frequency column):

    ==============  ==========================================  ====================================
    band            definition                                  compartment
    ==============  ==========================================  ====================================
    ``singleton``   ``count == 1``                              naive-dominated background; also
                                                                where sequencing error concentrates
    ``expanded``    ``count >= 2``                              clones that divided at least once
    ``top``         top ``top_fraction`` by count, clipped to   the dominant clonal compartment
                    ``top_clip``                                (chronic exposure history)
    ``igm``         ``c_call`` in IGHM/IGHD                     unswitched (IGH only)
    ``igg``         ``c_call`` in IGHG1–4                       class-switched (IGH only)
    ``iga``         ``c_call`` in IGHA1–2                       mucosal (IGH only)
    ==============  ==========================================  ====================================

    ``top`` and ``expanded`` overlap deliberately — "what does the dominant compartment look like" and
    "what does everything that expanded look like" are different questions. The isotype cut partitions
    on an irreversible molecular event rather than an abundance threshold, so IgM/IgG/IgA are
    genuinely different cell populations sharing one locus. Rows with a **null** ``c_call`` are
    *excluded* from every isotype band, never defaulted to IgM: an uncalled constant region is missing
    data, and roughly 43% of IGH reads have none at RNA-seq coverage.

    Args:
        sample_df: One sample's clonotypes with ``duplicate_count`` (and ``c_call`` for isotypes).
        bands: Which bands to build. Any of ``singleton``/``expanded``/``top``/``igm``/``igg``/``iga``.
        top_fraction: Fraction of clonotypes in the ``top`` band before clipping.
        top_clip: ``(min, max)`` clonotype count for the ``top`` band.
        min_clonotypes: A band with fewer clonotypes is returned as ``None`` — recorded **absent**,
            never embedded. A 3-clone "repertoire" is not an estimate; ``None`` is the same
            hole convention :class:`~mir.explain.ChannelBuilder` and
            :func:`mir.cohort.align_loci` already understand.

    Returns:
        ``{band: frame or None}`` in the requested order.

    Raises:
        ValueError: On an unknown band name, or an isotype band without a ``c_call`` column.
    """
    unknown = [b for b in bands if b not in _SIZE_BANDS and b not in _ISOTYPE_BANDS]
    if unknown:
        raise ValueError(f"unknown bands {unknown}; known: {list(_SIZE_BANDS) + list(_ISOTYPE_BANDS)}")
    if any(b in _ISOTYPE_BANDS for b in bands) and "c_call" not in sample_df.columns:
        raise ValueError("isotype bands need a 'c_call' column; none present")

    out: dict = {}
    for b in bands:
        if b == "singleton":
            f = sample_df.filter(pl.col(_COUNT) == 1)
        elif b == "expanded":
            f = sample_df.filter(pl.col(_COUNT) >= 2)
        elif b == "top":
            k = int(np.clip(round(top_fraction * sample_df.height), *top_clip))
            f = sample_df.sort(_COUNT, descending=True).head(k)
        else:
            # strip the allele suffix so IGHG1*01 lands in igg; a null c_call is missing data
            f = sample_df.filter(
                pl.col("c_call").str.split("*").list.first().is_in(_ISOTYPE_BANDS[b]))
        out[b] = f if f.height >= min_clonotypes else None
    return out


def band_embeddings(space: RepertoireSpace, sample_df: pl.DataFrame, *,
                    bands: tuple[str, ...] = _SIZE_BANDS, min_clonotypes: int = 5,
                    **kwargs) -> dict:
    """Embed each compartment of one sample through the **same** basis (see :func:`band_frames`).

    A band is never a reason to refit: refitting would put each compartment in its own geometry and
    make band-to-band and cohort-to-cohort distances meaningless. One ``space``, every band.

    Args:
        space: The shared :class:`RepertoireSpace`.
        sample_df: One sample's clonotypes.
        bands: Which bands to embed (:func:`band_frames`).
        min_clonotypes: Bands smaller than this are ``None`` (absent, not embedded).
        **kwargs: Forwarded to :func:`sample_embedding` (``weight``, ``blocks``, ``missing_mass``…).

    Returns:
        ``{band: SampleEmbedding or None}``.

    Note:
        To compare two compartments' *directions*, centre the cohort first (``X - X.mean(0)``, or
        :func:`centroid_atypicality` for the per-group version) and cosine the residuals. Do not read
        a raw cosine between two TCREmp-derived vectors, and do not think subtracting each vector's
        own scalar mean fixes it: on mean distance profiles, measured raw cos 0.9993, self-centred
        0.9985, cohort-centred **−0.14**. Only cohort centring removes the shared profile. Measured
        compartment overlaps of order 0.01 (RFF kernel means, whose null is already ≈0) are real
        signal above that baseline.
    """
    frames = band_frames(sample_df, bands=bands, min_clonotypes=min_clonotypes)
    return {b: (None if f is None else sample_embedding(space, f, **kwargs))
            for b, f in frames.items()}


def mixture_weights(whole, parts: dict) -> dict:
    """Non-negative mixture weights ``π_c`` of compartments inside a whole-repertoire ``Φ₁``.

    ``Φ`` is linear in the clone-weight measure, so ``Φ(S) = Σ_c π_c Φ(c)`` **exactly** for any
    partition, which makes a non-negative least squares of the whole on its compartments well-posed
    rather than a heuristic fit. (Exact in exact arithmetic: the realised agreement is limited by the
    float32 clonotype embedding, whose per-row values depend slightly on the thread batching, so
    expect a relative residual around ``1e-5`` rather than machine epsilon.) The weights are the answer to "how much of
    what I am measuring does this compartment actually own" — the dilution factor ``π`` of
    :func:`band_frames`, measured instead of assumed.

    Two things it is for. **Interpretation**: measured on IGH, the class-switched IgG compartment —
    affinity-matured, T-dependent, the fraction most likely to carry antigen-specific signal —
    carries a median ``π`` of only **0.070** of ``Φ₁(IGH)``, with unswitched IgM at 0.230, IgA at
    0.176 and 0.520 unexplained (the uncalled-isotype share, agreeing in ballpark with the ~0.43
    counted from reads — different denominators, embedding *weight* vs *reads*). **Power**: ``π`` is a
    prior estimate of how far a subset can move ``Φ``, so a subset carrying ``π`` ≈ 0.001 will not be
    detectable by any aggregate distance on ``Φ``, and the per-clonotype route
    (:func:`class_witness`) is the sensitive one. Check ``π`` before spending compute on the aggregate.

    Args:
        whole: The whole repertoire's ``Φ₁`` — a :class:`SampleEmbedding` or a ``(D,)`` array.
        parts: ``{name: SampleEmbedding | array | None}``; ``None`` entries are skipped
            (an absent band is not a zero-weight band).

    Returns:
        ``{"weights": {name: π}, "residual": 1 − Σπ, "r2": reconstruction R²}``. ``residual`` is the
        share of the whole no compartment accounts for (an unmeasured or excluded compartment), and
        can be negative if the parts overlap (``top`` ⊂ ``expanded``).

    Raises:
        ValueError: If no usable parts were given, or a part's width differs from ``whole``'s.
    """
    from scipy.optimize import nnls

    def vec(x):
        # NB not getattr(x, "mean", x): an ndarray has a .mean *method*
        return np.asarray(x.mean if isinstance(x, SampleEmbedding) else x, dtype=np.float64)

    y = vec(whole)
    names = [k for k, v in parts.items() if v is not None]
    if not names:
        raise ValueError("no usable parts: every entry is None")
    cols = []
    for k in names:
        c = vec(parts[k])
        if c.shape != y.shape:
            raise ValueError(f"part {k!r} has shape {c.shape}, whole has {y.shape}")
        cols.append(c)
    A = np.column_stack(cols)
    pi, _ = nnls(A, y)
    resid = y - A @ pi
    denom = float(y @ y)
    return {"weights": {k: float(p) for k, p in zip(names, pi)},
            "residual": float(1.0 - pi.sum()),
            "r2": float(1.0 - (resid @ resid) / denom) if denom > 0 else float("nan")}


# ----------------------------------------------------------------------- rarefaction


@dataclass
class RarefyResult:
    """A depth-standardised embedding plus the replicate dispersion (see :func:`rarefy_embedding`)."""

    embedding: SampleEmbedding   # Φ̄, the mean over replicates — itself a valid kernel mean
    v_rep: float                 # mean_r ‖Φ_r − Φ̄‖² — this sample's sampling noise, measured
    depth: int
    n_replicates: int

    def rao_gap(self) -> float:
        """``v_rep`` again, under the name of the identity it satisfies.

        ``Rao(Φ̄) = mean_r Rao(Φ_r) + v_rep`` holds **exactly**: ``Φ̄`` embeds the mixture over
        subsamples, which is genuinely more diverse than any single subsample, so the excess
        diversity of the average *is* the replicate variance. Averaging commutes with ``Φ`` but not
        with nonlinear functionals of it — ``mean(Rao) ≠ Rao(mean)`` — and this is the exact gap.
        """
        return self.v_rep


def rarefy_embedding(
    space: RepertoireSpace,
    sample_df: pl.DataFrame,
    depth: int,
    *,
    n_replicates: int = 10,
    weight: str = "log2p1",
    seed: int = 0,
) -> RarefyResult:
    """Embed a repertoire **rarefied to a common read depth**, averaged over replicate draws.

    The only depth correction that preserves the kernel-mean semantics exactly. ``Φ`` is linear in
    the clone-weight measure, so the mean of ``Φ`` over independent multinomial subsamples is itself
    a kernel mean embedding — of the mixture distribution over subsamples — and MMD, Rao's ``Q`` and
    mixture linearity all survive (verified to ~1e-15). An orthogonal projection breaks Rao; a
    per-coordinate location-scale rescale breaks MMD and Rao both.

    Not a default, and not a substitute for carrying depth explicitly: rarefying a whole cohort to
    the depth of its shallowest useful samples throws away the deep samples' entire advantage, which
    is why :func:`depth_threshold` + :func:`missing_mass` are the first-line tools. Reach for this
    when two groups must be compared *at matched depth* and you need the comparison to stay an MMD.

    Only the mean block is averaged — see :meth:`RarefyResult.rao_gap`.

    Args:
        space: The shared :class:`RepertoireSpace`.
        sample_df: One sample's clonotypes with ``duplicate_count``.
        depth: Reads to draw per replicate. Must not exceed the sample's total.
        n_replicates: Independent multinomial draws to average (``v_rep`` needs ``≥ 2``).
        weight: Clone-size weight applied to the *rarefied* counts.
        seed: RNG seed.

    Returns:
        A :class:`RarefyResult`.

    Raises:
        ValueError: If ``depth`` exceeds the sample's read total, ``depth < 1``, or
            ``n_replicates < 1``.
    """
    a, _, _ = _sample_weights(sample_df, weight)
    total = a.sum()
    if depth < 1 or n_replicates < 1:
        raise ValueError(f"depth and n_replicates must be >= 1, got {depth}, {n_replicates}")
    if depth > total:
        raise ValueError(
            f"cannot rarefy to {depth} reads: the sample holds {total:.0f}. Rarefaction only ever "
            "removes reads — drop the sample, or lower the common depth."
        )
    # Embed once; every replicate is then a re-weighting of the same feature rows (rarefaction
    # changes counts, never coordinates), which is both faster and obviously the same object.
    psi = space.rff.transform(space.transform_clonotypes(sample_df))
    rng = np.random.default_rng(seed)
    p = a / total
    phis, neffs = [], []
    for _ in range(n_replicates):
        draw = rng.multinomial(depth, p).astype(np.float64)
        m = draw > 0
        g = _WEIGHTS[weight](draw[m])
        w = g / g.sum()
        phis.append(w @ psi[m])
        neffs.append(1.0 / float(np.sum(w * w)))
    P = np.stack(phis)
    bar = P.mean(axis=0)
    v_rep = float(np.mean(np.sum((P - bar) ** 2, axis=1)))
    emb = SampleEmbedding(mean=bar, diversity=None, second=None, n_eff=float(np.mean(neffs)))
    return RarefyResult(embedding=emb, v_rep=v_rep, depth=int(depth), n_replicates=int(n_replicates))


# --------------------------------------------------------------- derivable descriptor


@dataclass
class RepertoireDescriptor:
    """Smooth, **mass-preserving** repertoire descriptor — every summary metric is a derivable coordinate.

    :func:`sample_embedding`'s ``Φ`` frequency-normalises, which discards the total mass (= infiltration).
    The descriptor instead **keeps the mass as a coordinate alongside diversity and clonality**, so the whole
    object is:

    * **decodable** — :meth:`metrics` reads infiltration / diversity / clonality off analytically;
    * **smooth** — ``log`` mass, ``log`` n_eff and Simpson λ are continuous (no integer richness ``⁰D``),
      the "smoother form" of the Hill block;
    * **simulatable** — :attr:`vector` is a fixed-width continuous vector you can perturb and generate:
      fit a density over a cohort, move along the infiltration coordinate → in-silico "hotter / colder"
      (:func:`decode_metrics` turns any perturbed vector back into named metrics).

    ``mean`` is the normalised kernel mean ``μ/G`` (clonotype identity / composition); the scalar coordinates
    are the count-distribution summaries. Distances/generation live in the concatenated :attr:`vector`.
    """

    log_mass: float          # log Σ a — infiltration / coverage (the mass Φ normalises away)
    log_neff: float          # log (Σg)²/Σg² — effective diversity (smooth Hill number)
    simpson: float           # Σ w² — clonality / dominance
    mean: np.ndarray         # μ/G — normalised kernel mean (identity)

    @property
    def scalar(self) -> np.ndarray:
        """The derivable-metric coordinates ``[infiltration, log n_eff, clonality]``."""
        return np.array([self.log_mass, self.log_neff, self.simpson], dtype=np.float64)

    @property
    def vector(self) -> np.ndarray:
        """The full descriptor ``[scalar ‖ mean]`` — the continuous object to perturb / generate."""
        return np.concatenate([self.scalar, self.mean]).astype(np.float64)

    def metrics(self) -> dict:
        """Named metrics read off the coordinates (all analytic, no recomputation from counts)."""
        return {"infiltration": float(self.log_mass), "log_neff": float(self.log_neff),
                "diversity": float(np.exp(self.log_neff)), "clonality": float(self.simpson)}


def sample_descriptor(space: RepertoireSpace, sample_df: pl.DataFrame, *,
                      weight: str = "log2p1") -> RepertoireDescriptor:
    """Mass-preserving smooth descriptor of one repertoire (see :class:`RepertoireDescriptor`).

    The scale (``log_mass`` = infiltration) is retained rather than normalised away, so infiltration,
    diversity and clonality are all smooth coordinates of the *same* object — the representation the
    in-silico-evolution / embedding-simulation workflow perturbs.
    """
    a, g, w = _sample_weights(sample_df, weight)
    mean = w @ space.rff.transform(space.transform_clonotypes(sample_df))
    sw2 = float(np.sum(w * w))
    return RepertoireDescriptor(log_mass=float(np.log1p(a.sum())),
                                log_neff=float(-np.log(sw2)), simpson=sw2, mean=mean)


def decode_metrics(vector: np.ndarray) -> dict:
    """Read named metrics off a (possibly *perturbed* or *generated*) descriptor vector — the inverse
    used for in-silico evolution: perturb :attr:`RepertoireDescriptor.vector`, decode the new metrics."""
    return {"infiltration": float(vector[0]), "log_neff": float(vector[1]),
            "diversity": float(np.exp(vector[1])), "clonality": float(vector[2])}


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
    ma, mb = a._require_mean(), b._require_mean()
    if not unbiased:
        return float(np.linalg.norm(ma - mb))
    if a.n_eff <= 1.0 or b.n_eff <= 1.0:
        raise ValueError(
            "unbiased MMD is undefined for a single-clonotype sample (n_eff ≤ 1): the diagonal "
            "cannot be removed from a point mass. Use unbiased=False, or drop degenerate samples."
        )
    sa, sb = 1.0 / a.n_eff, 1.0 / b.n_eff                   # Σwᵢ² ; RFF self-similarity k(z,z)≈1
    haa = (float(ma @ ma) - sa) / (1.0 - sa)                # diagonal-removed ‖μ‖²
    hbb = (float(mb @ mb) - sb) / (1.0 - sb)
    return float(np.sqrt(max(haa + hbb - 2.0 * float(ma @ mb), 0.0)))


def mmd_matrix(embs: list[SampleEmbedding], *, unbiased: bool = False) -> np.ndarray:
    """Symmetric sample×sample MMD matrix (feeds a regressor / ``cluster_samples``).

    ``unbiased=True`` uses the diagonal-removed MMD² (see :func:`mmd_distance`) — necessary whenever
    samples differ in depth/``n_eff``, else the ``1/n_eff`` self-bias confounds the comparison.
    """
    M = np.stack([e._require_mean() for e in embs])         # (S, D)
    G = M @ M.T
    sq = np.diag(G).copy()
    if unbiased:
        s = np.array([1.0 / e.n_eff for e in embs])        # per-sample Σwᵢ²
        if np.any(s >= 1.0):
            raise ValueError(
                "unbiased MMD is undefined for single-clonotype samples (n_eff ≤ 1); "
                f"{int(np.sum(s >= 1.0))} of {len(embs)} samples are degenerate. "
                "Use unbiased=False, or drop them before building the matrix."
            )
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
    weight: str = "log2p1",
    top: int = 30,
    witness: np.ndarray | None = None,
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
        witness: Optional precomputed witness direction ``μ_pos − μ_neg`` (``(D,)``). When given,
            ``pos``/``neg`` are not re-embedded — reuse it to score several candidate sets cheaply.

    Returns:
        ``candidates`` with a ``witness_score`` column, sorted descending, truncated to ``top``.
    """
    if witness is None:
        def group_mean(frames):
            return np.mean([w @ space.rff.transform(Z)
                            for Z, w in (space.sample_cloud(f, weight=weight) for f in frames)], axis=0)

        witness = group_mean(pos) - group_mean(neg)
    psi = space.rff.transform(space.transform_clonotypes(candidates))     # (n, D)
    scores = psi @ witness
    return (candidates.with_columns(pl.Series("witness_score", scores))
            .sort("witness_score", descending=True).head(top))


def hla_stratified_mmd(embs: list[SampleEmbedding], hla: list[set]) -> np.ndarray:
    """MMD restricted to HLA-matched sample pairs; ``nan`` where two samples share no allele.

    Response ridges are presented only by a specific HLA type, so cross-sample overlap is
    informative *only* within an HLA-matched stratum (Prop. ``prop:hla``). Pairs that share no
    HLA allele are masked to ``nan`` rather than compared.
    """
    D = mmd_matrix(embs)
    alleles = sorted({al for s in hla for al in s})
    col = {al: k for k, al in enumerate(alleles)}
    A = np.zeros((len(hla), len(alleles)), dtype=np.int8)   # sample × allele indicator
    for i, s in enumerate(hla):
        for al in s:
            A[i, col[al]] = 1
    shared = A @ A.T                                        # #shared alleles per pair
    return np.where(shared > 0, D, np.nan)


def centroid_atypicality(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-sample cosine distance of its identity vector to its group's centroid — a Φ-geometry op.

    A sample far from its group's mean identity has an atypical clonotype composition (selection /
    divergence). Grouping is caller-supplied (tumour type, cohort, …); the geometry — centroid then
    ``1 − cos`` — is the library concern. Feeds an ``atypicality`` channel of a digital-donor embedding.

    Args:
        X: ``(n, d)`` identity block (a per-sample kernel-mean or its PCA reduction).
        groups: ``(n,)`` group label per row; centroids are computed within each group.

    Returns:
        ``(n,)`` atypicality in ``[0, 2]`` (0 = on the group centroid direction).
    """
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups)
    out = np.zeros(X.shape[0])
    for g in np.unique(groups):
        m = groups == g
        cen = X[m].mean(axis=0)
        cn = np.linalg.norm(cen) + 1e-9
        xn = np.linalg.norm(X[m], axis=1) + 1e-9
        out[m] = 1.0 - (X[m] @ cen) / (xn * cn)
    return out


def correct_batch(
    X: np.ndarray,
    batch: np.ndarray,
    *,
    covariates: np.ndarray | None = None,
    n_clusters: int = 8,
    theta: float = 1.0,
    sigma: float = 0.1,
    max_iter: int = 10,
    ridge: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Harmony-like cluster-aware batch correction on a stacked sample×feature ``Φ`` matrix.

    Removes a batch offset **per soft cluster** rather than globally, so a batch confounded with a
    biological cluster is corrected without erasing that biology — the failure mode of plain
    per-group mean subtraction (:func:`mir.cohort.residualize`). Follows Harmony (Korsunsky et al.
    2019, *Nat. Methods*): soft-cluster the samples with a batch-diversity penalty ``theta``
    (clusters pushed toward batch-balanced membership), then subtract each cluster's
    membership-weighted batch offset (covariates retained), and iterate. **Reduces exactly to**
    ``residualize`` at ``n_clusters=1`` or ``theta=0``.

    Args:
        X: Stacked ``(n_samples, n_features)`` matrix (e.g. :func:`mir.explain.stack_embeddings`).
        batch: Length-``n_samples`` batch label per row.
        covariates: Optional ``(n_samples, k)`` biological covariates to *retain* (never removed).
        n_clusters: Number of soft clusters (Harmony's ``K``); ``<=1`` disables clustering.
        theta: Batch-diversity penalty strength; ``0`` disables clustering (→ ``residualize``).
        sigma: Soft-assignment temperature.
        max_iter: Correction iterations (converges in a few — the batch fit vanishes once removed).
        ridge: L2 ridge on the per-cluster batch regression (stabilises small clusters).
        seed: KMeans seed for the soft-cluster initialisation.

    Returns:
        The corrected ``(n_samples, n_features)`` matrix — a new coordinate system (as with
        ``residualize``, never compare a corrected ``X`` to an uncorrected one).
    """
    # ponytail: Harmony-lite — fixed K, a single KMeans-seeded soft-cluster init, fixed theta. The
    # upgrade path is Harmony's full automatic-K + block-coordinate objective; this covers the
    # confounded-batch case and reduces exactly to residualize when K<=1 / theta==0.
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    batch = np.asarray(batch)
    _, binv = np.unique(batch, return_inverse=True)
    Phi = np.eye(int(binv.max()) + 1)[binv]                 # (n, nb) batch one-hot
    pr_b = Phi.mean(0)                                       # global batch proportions

    if n_clusters <= 1 or theta <= 0:                       # == residualize (exact)
        out = X.copy()
        for b in range(Phi.shape[1]):
            m = binv == b
            out[m] -= X[m].mean(0)
        return out

    from sklearn.cluster import KMeans

    # Cosine-geometry soft clustering (Harmony normalises to the unit sphere).
    Z = X - X.mean(0)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    K = min(n_clusters, n)
    centers = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Z).cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9)
    dist = 2.0 * (1.0 - Z @ centers.T)                      # (n, K) cosine distance

    # Batch-diversity-penalised soft assignment: up-weight clusters under-represented for a row's
    # batch, so clusters become batch-balanced (Harmony's key term) rather than batch-defined.
    R = np.exp(-dist / sigma)
    R = R / (R.sum(1, keepdims=True) + 1e-9)
    for _ in range(3):
        O = R.T @ Phi                                       # (K, nb) batch mass per cluster
        E = R.sum(0)[:, None] * pr_b[None, :] + 1e-9        # expected under global freq
        pen = (E / (O + 1e-9)) ** theta                     # (K, nb) diversity re-weight
        R = np.exp(-dist / sigma) * (Phi @ pen.T)           # (n, K)
        R = R / (R.sum(1, keepdims=True) + 1e-9)

    # Retain intercept + covariates, remove the batch design; drop one batch column for identifiability.
    Cov = np.zeros((n, 0)) if covariates is None else np.asarray(covariates, np.float64).reshape(n, -1)
    design = np.hstack([np.ones((n, 1)), Cov, Phi[:, 1:]])
    n_keep = 1 + Cov.shape[1]                               # columns kept; the rest (batch) removed
    Xc = X.copy()
    for _ in range(max_iter):
        acc = np.zeros_like(Xc)
        for k in range(K):
            w = R[:, k]
            Wd = design * w[:, None]
            A = Wd.T @ design + ridge * np.eye(design.shape[1])
            beta = np.linalg.solve(A, Wd.T @ Xc)            # (p, d) weighted ridge fit
            acc += w[:, None] * (Xc - design[:, n_keep:] @ beta[n_keep:])
        Xc = acc / (R.sum(1, keepdims=True) + 1e-9)
    return Xc


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
