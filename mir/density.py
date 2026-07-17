"""Continuous-density TCRNET / ALICE: enrichment in TCREMP embedding space (Theory T6).

TCRNET (Pogorelyy & Shugay, *Front Immunol* 2019) and ALICE (Pogorelyy et al.,
*PLoS Biol* 2019) flag antigen-driven / convergent-selected clonotypes by *neighbour
enrichment*: count a clonotype's near-identical (Hamming-1) neighbours and compare
against a background — OLGA generation probability for ALICE, a control repertoire for
TCRNET. Both are **graph** methods (a sequence trie).

This module reimplements the same enrichment test with neighbour counting in the
**TCREMP embedding space** instead of on a trie — the graph-free density ratio the
theory specifies (``THEORY.md`` T6, ``appendix/tcremp_theory.tex`` §Density):

    E(z) = f_obs(z) / f_gen(z),      f_gen = φ_# P_gen

For an observed clonotype embedded at ``z_i`` we count observed neighbours ``n_obs``
and background neighbours ``n_bg`` within a radius ``r`` (calibrated to one CDR3
substitution, so the continuous test approximates the discrete Hamming-1 one). Under
the null "the observed repertoire is a background sample", each of the other
``N_obs − 1`` observed clonotypes lands within ``r`` of ``z_i`` with probability
``p_bg = n_bg / N_bg``, so the expected count is ``(N_obs−1)·p_bg`` and

    p-value = poisson.sf(n_obs − 1, expected)        # ALICE-style   (test="poisson")
            = binom.sf(n_obs − 1, N_obs−1, p_bg)     # TCRNET-style  (test="binomial")

with Benjamini-Hochberg q-values. The fold enrichment ``fold = n_obs / expected`` is the
density ratio ``E(z)`` itself. The background is either **generated** from the vdjtools
P_gen model (:func:`generate_background`, the ALICE analog) or **any supplied control
repertoire** (the TCRNET analog) — both just become ``bg_df``.

Passing ``abundance=`` (clone sizes) to :func:`neighbor_enrichment` adds the **clonal-depth**
channel (appendix §T.6 ``sec:dens-abund``): the distinct in-ball count becomes a
variance-stabilised weighted mass ``S=Σ g(a_j)`` with concave ``g`` (default ``log(1+a)``) so a
hyperexpanded clone can't dominate, tested against a compound-Poisson Gamma tail, plus an orphan
size-test ``P(A≥a_j)`` combined with breadth by Fisher. Default (``abundance=None``) counts each
clonotype once — the shipped ``g≡1`` behaviour.

Observed and background embeddings are comparable *only* in one coordinate system (same
prototypes **and** same PCA rotation). :func:`fit_density_space` enforces that: it embeds
both through one :class:`~mir.embedding.tcremp.TCREmp` and fits one PCA on the pooled
matrix, returning a :class:`DensitySpace` that projects any further frame (a control, or
the radius-calibration mutants) into the *same* basis.

Typical usage::

    from mir.density import fit_density_space, calibrate_radius, neighbor_enrichment, enriched_mask
    from mir.embedding.tcremp import TCREmp
    from mir.embedding.presets import get_preset

    model = TCREmp.from_defaults("human", "TRB")
    nc = get_preset("human", "TRB").n_components
    space, obs_emb, bg_emb = fit_density_space(model, obs_df, bg_df, n_components=nc)
    r = calibrate_radius(space)                       # radius ≈ one CDR3 substitution
    res = neighbor_enrichment(obs_emb, bg_emb, r)     # E(z), p, q per observed clonotype
    hits = obs_df.filter(enriched_mask(res))          # significantly enriched clonotypes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from mir.distances.junction import junction_distance_matrix

_REQUIRED_COLS = ("v_call", "j_call", "junction_aa")
_AA = "ACDEFGHIKLMNPQRSTVWY"
_AUTO_CHUNK_ROWS = 500_000
"""Above this pooled row count, :func:`fit_density_space` auto-routes to the chunked path: the
single-shot path would materialize a >10 GB raw matrix (and ~2× again on the float64 upcast)."""


def _slice(emb: np.ndarray, space: str) -> np.ndarray:
    """Select the embedding coordinates that define a neighbourhood."""
    if space == "full":
        return emb
    if space == "junction":
        return emb[:, 2::3]  # junction is slot 2 of every [slot0, slot1, junction] triple
    raise ValueError(f"space must be 'full' or 'junction', got {space!r}")


def _embed(model, df: pl.DataFrame, space: str) -> np.ndarray:
    """Raw (pre-PCA) embedding of *df* in the requested space.

    ``space="junction"`` bypasses the full V+J+junction embedding and computes the
    junction sub-block directly (``junction_distance_matrix`` against the model's
    prototypes) — a third of the memory, which matters for whole-repertoire scale.
    ``space="full"`` slices the full single-chain ``TCREmp`` embedding.
    """
    if space == "junction" and hasattr(model, "_proto_junction"):
        return junction_distance_matrix(
            df["junction_aa"].to_list(), model._proto_junction,
            gap_positions=model._gap_positions, threads=model.threads,
            metric=getattr(model, "metric", "squared"),      # match the model's coordinate options
            matrix=getattr(model, "_matrix", None),
            alignment=getattr(model, "_alignment", "gapblock"),
        ).astype(np.float32)  # float32 halves the whole-repertoire memory footprint
    return _slice(model.embed(df), space).astype(np.float32, copy=False)


def _embed_transform_chunked(model, df: pl.DataFrame, space: str, scaler, pca,
                             chunk_size: int) -> np.ndarray:
    """Embed *df* and project it into ``scaler``+``pca``, ``chunk_size`` rows at a time.

    Only ``chunk_size × n_features`` of raw embedding is ever resident: each chunk is embedded,
    standardized, projected down to ``n_components``, and dropped. Peak memory is set by the chunk,
    not by ``len(df)`` — which is what makes whole-cohort clouds tractable (a 4.2M-clonotype arm is
    ~51 GB raw, and ~102 GB again once ``scaler.transform`` upcasts it to float64).
    """
    out = np.empty((df.height, pca.n_components_), dtype=np.float64)
    for start in range(0, df.height, chunk_size):
        stop = min(start + chunk_size, df.height)
        raw = _embed(model, df.slice(start, stop - start), space)
        out[start:stop] = pca.transform(scaler.transform(raw))
        del raw
    return out


@dataclass
class DensitySpace:
    """A single fitted TCREMP → PCA coordinate system for density comparison.

    Holds the embedding model plus the standardization + PCA fitted on the pooled
    observed+background matrix. :meth:`transform` projects any frame into this *same*
    basis so that observed, background, control and calibration points are all directly
    comparable — the invariant that makes ``E(z)`` meaningful.
    """

    model: object          # single-chain TCREmp
    space: str             # "full" | "junction"
    scaler: StandardScaler
    pca: PCA

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """Embed *df* and project it into this fitted coordinate system."""
        return self.pca.transform(self.scaler.transform(_embed(self.model, df, self.space)))


@dataclass
class EnrichmentResult:
    """Per-observed-clonotype neighbour enrichment (input order preserved)."""

    n_obs: np.ndarray    # observed neighbours within radius (self excluded)
    n_bg: np.ndarray     # background neighbours within radius
    expected: np.ndarray # expected observed count/mass under the background null
    fold: np.ndarray     # observed / expected == the density ratio E(z)
    pvalue: np.ndarray   # one-sided enrichment p-value (combined breadth×depth if abundance-aware)
    qvalue: np.ndarray   # Benjamini-Hochberg adjusted p-value
    radius: float
    # abundance-aware channels (None unless `abundance` was supplied); appendix §T.6 sec:dens-abund
    score: np.ndarray | None = None           # weighted in-ball mass S(z) = Σ g(a_j), self excluded
    pvalue_breadth: np.ndarray | None = None  # breadth channel: S vs the compound-Poisson null
    pvalue_size: np.ndarray | None = None      # depth/orphan channel: P(A ≥ a_j) under the size law


_WEIGHTS = {
    "distinct": lambda a: np.ones(a.shape, dtype=np.float64),  # g≡1: the shipped distinct count
    "log1p": lambda a: np.log1p(a),                            # concave, robust to the Zipf tail
    "anscombe": lambda a: np.sqrt(a + 0.375),                  # variance-stabilising root
}


def _emp_survival(a: np.ndarray) -> np.ndarray:
    """Empirical size-law survival ``P(A ≥ a_j)`` (the null clone-size law ν, read off the data)."""
    s = np.sort(a)
    return (a.size - np.searchsorted(s, a, side="left")) / a.size


def fit_density_space(
    model,
    obs_df: pl.DataFrame,
    bg_df: pl.DataFrame,
    *,
    n_components: int,
    space: str = "full",
    seed: int = 0,
    pca_fit_cap: int | None = None,
    chunk_size: int | None = None,
) -> tuple[DensitySpace, np.ndarray, np.ndarray]:
    """Embed observed + background frames into one shared PCA coordinate system.

    Both frames are embedded through the *same* ``model`` and a single PCA (fit on the
    pooled matrix) is applied to both, guaranteeing the two point clouds live in one
    comparable basis — the invariant that makes ``E(z)`` meaningful.

    Args:
        model: A fitted single-chain :class:`~mir.embedding.tcremp.TCREmp`.
        obs_df: Observed clonotypes (``v_call``, ``j_call``, ``junction_aa``).
        bg_df: Background clonotypes (generated P_gen sample or a control repertoire).
        n_components: PCA dimensionality (clamped to ``min(n_samples, n_features)``);
            use ``get_preset(species, locus).n_components``.
        space: ``"full"`` (whole V+J+junction vector — V/J restriction is soft/continuous)
            or ``"junction"`` (CDR3 sub-block only, mimicking pure-CDR3 TCRNET/ALICE and a
            third of the memory at whole-repertoire scale).
        seed: RNG / PCA solver seed.
        pca_fit_cap: Fit the ``StandardScaler`` + PCA on at most this many randomly-sampled
            pooled rows (then transform *all* rows). ``None`` fits on everything. NB on its own
            this caps the **fit**, not the memory: without ``chunk_size`` the full raw matrix for
            both frames is materialized before the PCA is fitted at all.
        chunk_size: Embed and project in batches of this many rows, so the full raw matrix is
            never resident (peak ≈ ``chunk_size × n_features`` instead of ``len(df) × n_features``).
            Required for whole-cohort clouds: 4.2M clonotypes × 1000 prototypes is ~51 GB raw and
            ~102 GB after ``scaler.transform`` upcasts to float64, versus ~2.4 GB at
            ``chunk_size=200_000``. When set and ``pca_fit_cap`` is ``None``, the fit is capped at
            200k pooled rows — fitting on everything would defeat the purpose. ``None`` (default)
            keeps the original single-shot path exactly.

    Returns:
        ``(density_space, obs_emb, bg_emb)`` — the fitted :class:`DensitySpace` and the
        two reduced ``float`` arrays, row-aligned to ``obs_df`` / ``bg_df``.
    """
    if chunk_size is None and obs_df.height + bg_df.height > _AUTO_CHUNK_ROWS:
        import warnings

        chunk_size = 200_000  # the single-shot path would materialize a >10 GB raw matrix
        warnings.warn(
            f"pooled rows ({obs_df.height + bg_df.height}) exceed {_AUTO_CHUNK_ROWS}: auto-enabling "
            f"the chunked fit (chunk_size={chunk_size}) to bound peak memory; the PCA is fit on a "
            "200k-row sample. Pass chunk_size explicitly to control this.", stacklevel=2)
    if chunk_size is not None:
        return _fit_density_space_chunked(
            model, obs_df, bg_df, n_components=n_components, space=space, seed=seed,
            pca_fit_cap=pca_fit_cap if pca_fit_cap is not None else 200_000,
            chunk_size=chunk_size,
        )
    obs = _embed(model, obs_df, space)
    bg = _embed(model, bg_df, space)
    n_total, n_feat = obs.shape[0] + bg.shape[0], obs.shape[1]

    if pca_fit_cap is not None and n_total > pca_fit_cap:
        rng = np.random.default_rng(seed)
        take_o = min(obs.shape[0], pca_fit_cap * obs.shape[0] // n_total)
        take_b = min(bg.shape[0], pca_fit_cap - take_o)
        fit_rows = np.vstack([obs[rng.choice(obs.shape[0], take_o, replace=False)],
                              bg[rng.choice(bg.shape[0], take_b, replace=False)]])
    else:
        fit_rows = np.vstack([obs, bg])

    scaler = StandardScaler().fit(fit_rows)
    k = min(n_components, fit_rows.shape[0], n_feat)
    pca = PCA(n_components=k, random_state=seed).fit(scaler.transform(fit_rows))
    ds = DensitySpace(model=model, space=space, scaler=scaler, pca=pca)
    return ds, pca.transform(scaler.transform(obs)), pca.transform(scaler.transform(bg))


def _fit_density_space_chunked(
    model, obs_df: pl.DataFrame, bg_df: pl.DataFrame, *, n_components: int, space: str,
    seed: int, pca_fit_cap: int, chunk_size: int,
) -> tuple[DensitySpace, np.ndarray, np.ndarray]:
    """``fit_density_space`` without ever holding a full raw matrix.

    Same coordinate system as the single-shot path — one scaler + one PCA fitted on a pooled
    sample, applied to both frames — but the fit sample is drawn as *frame rows* and embedded on
    its own, and each frame is then embedded/projected in chunks. Memory is set by
    ``max(pca_fit_cap, chunk_size)``, not by the cohort.
    """
    rng = np.random.default_rng(seed)
    n_o, n_b = obs_df.height, bg_df.height
    n_total = n_o + n_b

    if n_total > pca_fit_cap:
        take_o = min(n_o, pca_fit_cap * n_o // n_total)
        take_b = min(n_b, pca_fit_cap - take_o)
        fit_rows = np.vstack([
            _embed(model, obs_df[rng.choice(n_o, take_o, replace=False)], space),
            _embed(model, bg_df[rng.choice(n_b, take_b, replace=False)], space),
        ])
    else:
        fit_rows = np.vstack([_embed(model, obs_df, space), _embed(model, bg_df, space)])

    scaler = StandardScaler().fit(fit_rows)
    k = min(n_components, fit_rows.shape[0], fit_rows.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(scaler.transform(fit_rows))
    del fit_rows

    ds = DensitySpace(model=model, space=space, scaler=scaler, pca=pca)
    return (ds,
            _embed_transform_chunked(model, obs_df, space, scaler, pca, chunk_size),
            _embed_transform_chunked(model, bg_df, space, scaler, pca, chunk_size))


def _mutate1(seq: str, rng) -> str:
    """Apply one interior amino-acid substitution to a *different* residue."""
    if len(seq) <= 2:
        return seq
    p = int(rng.integers(1, len(seq) - 1))  # keep the conserved C…[FW] ends
    choices = _AA.replace(seq[p], "")
    return seq[:p] + choices[int(rng.integers(len(choices)))] + seq[p + 1:]


def calibrate_radius(
    space: DensitySpace,
    *,
    sample_df: pl.DataFrame | None = None,
    sample: int = 2000,
    seed: int = 0,
    quantile: float = 0.5,
) -> float:
    """Radius corresponding to one CDR3 substitution, in *space*'s coordinate system.

    Mutates one interior residue of each sampled junction and measures the embedding
    drift ``‖φ(seq) − φ(mutated)‖`` through the *same* fitted transform used for
    enrichment (reusing the Theory-T5 SHM-drift idea). The requested ``quantile`` of that
    drift is the neighbourhood radius, so the continuous test approximates the discrete
    Hamming-1 one. V/J are held fixed, isolating the junction contribution.

    Args:
        space: A :class:`DensitySpace` from :func:`fit_density_space`.
        sample_df: Frame to draw calibration sequences from; defaults to the model's own
            prototype set (the coordinate anchor).
        sample: Number of prototypes to use when ``sample_df`` is ``None``.
        seed: RNG seed for the substitutions.
        quantile: Drift quantile to return (``0.5`` = median).

    Returns:
        The calibrated radius (a positive float).
    """
    rng = np.random.default_rng(seed)
    if sample_df is None:
        m = space.model
        k = min(sample, len(m._proto_junction))
        sample_df = pl.DataFrame(
            {"v_call": m._proto_v[:k], "j_call": m._proto_j[:k],
             "junction_aa": m._proto_junction[:k]}
        )
    junc = sample_df["junction_aa"].to_list()
    mutated_df = sample_df.with_columns(
        pl.Series("junction_aa", [_mutate1(s, rng) for s in junc])
    )
    drift = np.linalg.norm(space.transform(sample_df) - space.transform(mutated_df), axis=1)
    r = float(np.quantile(drift, quantile))
    if not (r > 0):
        raise ValueError(
            "calibrate_radius produced a non-positive radius: the calibration junctions did not "
            "move under a single substitution (too short, or all identical), so every enrichment "
            "ball would be empty. Supply a sample_df with longer/real junctions, or pass an "
            "explicit radius to neighbor_enrichment."
        )
    return r


def _ann_neighbors(obs, bg, radius, lambda0, n_ref, *, k_max: int = 96, seed: int = 0):
    """Approximate neighbour queries (pynndescent) for whole-repertoire scale.

    Returns ``(rad, radius_out, n_bg, count_obs, lists_obs)`` matching the exact BallTree path:
    per-obs radius ``rad``, background occupancy ``n_bg``, and two closures giving the
    self-excluded observed-neighbour count and index lists within ``rad``. Neighbours come from a
    kNN graph (k=``k_max``) thresholded by radius — *approximate*: recall < 1 undercounts, biasing
    enrichment **down** (conservative). A saturated ball (all ``k_max`` neighbours inside ``rad``)
    is undercounted and warned. For large N where exact trees are slow; use ``backend='exact'`` for
    small or reproducibility-critical runs. Needs ``pynndescent`` (the ``[bench]`` extra).
    """
    from pynndescent import NNDescent

    obs = np.ascontiguousarray(obs, dtype=np.float32)  # pynndescent prefers float32
    bg = np.ascontiguousarray(bg, dtype=np.float32)
    n_obs_total, n_bg_total = len(obs), len(bg)
    bg_index = NNDescent(bg, metric="euclidean",
                         n_neighbors=min(max(k_max, 16), n_bg_total - 1), random_state=seed)
    obs_index = NNDescent(obs, metric="euclidean",
                          n_neighbors=min(k_max, n_obs_total - 1), random_state=seed)
    if radius is None:  # balloon: radius = k-th bg-neighbour distance, occupancy == k
        k = min(max(int(round(lambda0 * n_bg_total / n_ref)), 1), n_bg_total)
        rad = bg_index.query(obs, k=k)[1][:, -1].astype(np.float64)
        n_bg = np.full(n_obs_total, k, dtype=np.int64)
        radius_out = float(np.median(rad))
    else:  # fixed global radius
        rad = np.full(n_obs_total, float(radius), dtype=np.float64)
        bd = bg_index.query(obs, k=min(k_max, n_bg_total))[1]
        n_bg = (bd <= float(radius)).sum(1).astype(np.int64)
        radius_out = float(radius)
    oi, od = obs_index.query(obs, k=min(k_max, n_obs_total))  # self included (dist ~0)
    within = od <= rad[:, None]
    if bool(within.all(axis=1).any()):
        import warnings

        warnings.warn(f"ANN neighbour ball saturated at k_max={k_max} for some clonotypes; their "
                      "counts are undercounted — raise k_max or use backend='exact'.", stacklevel=2)

    def _count_obs():
        return (within.sum(1) - 1).astype(np.int64)

    def _lists_obs():
        return [oi[i][within[i]] for i in range(n_obs_total)]

    return rad, radius_out, n_bg, _count_obs, _lists_obs


def neighbor_enrichment(
    obs_emb: np.ndarray,
    bg_emb: np.ndarray,
    radius: float | None = None,
    *,
    lambda0: float = 3.0,
    test: str = "poisson",
    pseudocount: float = 1.0,
    calibrate: str | None = "median",
    abundance: np.ndarray | None = None,
    weight: str = "log1p",
    orphan: bool = True,
    backend: str = "kdtree",
) -> EnrichmentResult:
    """Continuous neighbour-enrichment test in one embedding coordinate system.

    Two neighbourhood-scale modes (appendix §T.6):

    * **balloon** (default, ``radius=None``) — *adaptive* bandwidth. Each observed point's
      radius is the distance to its ``k``-th background neighbour, with ``k`` chosen so the
      expected background occupancy equals ``lambda0``. The adaptive radius encodes the P_gen
      density *shape* (small where generation is dense, large where sparse); it is robust to
      the distance concentration that makes a single global radius fragile in high dimensions,
      and keeps the test well-powered everywhere (minimum detectable ``E ≳ 1 + c/√lambda0``).
    * **fixed** (``radius`` given) — one global radius (e.g. from :func:`calibrate_radius`).

    A real repertoire's background is systematically *denser* than a generative P_gen model
    (gene usage, sampling structure, thymic selection) — the "water level" ``π`` of §T.6.
    ``calibrate="median"`` rescales the null so the bulk of clones sits at ``fold ≈ 1`` and
    only clones exceeding the repertoire's own typical local density are called. Signal
    (< 5 % in naive repertoires) barely moves the median, so this is robust; pass
    ``calibrate=None`` to test against the raw P_gen null instead.

    **Clonal abundance** (appendix §T.6 ``sec:dens-abund``). By default each clonotype counts once
    (``abundance=None``), scoring only convergence *breadth*. Supplying per-clonotype clone sizes
    ``abundance`` adds the *depth* channel: the in-ball count is replaced by a variance-stabilised
    weighted mass ``S(z) = Σ g(a_j)`` over neighbours, with ``g`` non-decreasing and **concave**
    (``weight="log1p"`` ``g=log(1+a)`` or ``"anscombe"`` ``g=√(a+3/8)``) so a hyperexpanded clone
    contributes a bounded increment, not a thousand-fold one. Under H0 the mass is compound-Poisson
    with dispersion index ``φ=E[g²]/E[g]`` (Prop. *abund*), so ``S`` is tested against a
    moment-matched ``Gamma(μ_S/φ, φ)`` upper tail (which collapses to the Poisson count test when
    ``g≡1``). With ``orphan=True`` each clone additionally gets a size p-value
    ``P(A≥a_j)`` and the breadth/depth channels are combined by Fisher (``χ²₄``), recovering a
    hyperexpanded *orphan* (depth, no breadth) that the count test alone misses.

    Args:
        obs_emb: ``(N_obs, d)`` observed embedding.
        bg_emb: ``(N_bg, d)`` background embedding, **same basis** as ``obs_emb``
            (produce both with :func:`fit_density_space`). Use ``M ≥ 5N`` for a stable ratio.
        radius: Fixed neighbourhood radius; ``None`` selects the balloon estimator.
        lambda0: Target expected background occupancy for the balloon estimator (``[1, 5]``).
        test: ``"poisson"`` (ALICE analog) or ``"binomial"`` (TCRNET control analog). Applies to
            the distinct-count path; the abundance-weighted path always uses the Gamma tail.
        pseudocount: Added to the background count to stabilize ``p_bg``.
        calibrate: ``"median"`` empirical-null water-level calibration (default), or ``None``.
        abundance: Optional ``(N_obs,)`` clone sizes (e.g. ``duplicate_count``), row-aligned to
            ``obs_emb``; enables the weighted/orphan channels.
        weight: Concave size transform ``g`` — ``"log1p"`` (default), ``"anscombe"``, or
            ``"distinct"`` (``g≡1``, ignore sizes even if ``abundance`` is given).
        orphan: When abundance-aware, combine the size p-value with the breadth p-value by Fisher.
        backend: neighbour engine. ``"kdtree"`` (**default**, scipy cKDTree — exact, multithreaded,
            5–9× faster than BallTree), ``"exact"`` (BallTree — exact, single-threaded; the historical
            default, kept for bit-reproducing the BallTree baseline), or ``"ann"`` (approximate
            pynndescent kNN for whole-repertoire scale — ~30× faster at ≥40k but recall < 1 undercounts,
            biasing enrichment conservatively; see :func:`_ann_neighbors`). ``"kdtree"`` is bit-identical
            to ``"exact"`` at a *fixed* ``radius``; in balloon mode counts differ by at most ±1 at the
            ball boundary (float-epsilon in the computed k-th-neighbour distance), which is negligible —
            pass ``backend="exact"`` only to reproduce the BallTree baseline exactly.

    Returns:
        An :class:`EnrichmentResult`; ``fold`` is the (calibrated) density ratio ``E(z)``. In
        balloon mode ``radius`` is the median adaptive radius used. When abundance-aware, ``score``
        holds the weighted mass ``S`` and ``pvalue`` is the combined breadth×depth test.
    """
    obs = np.ascontiguousarray(obs_emb, dtype=np.float64)
    bg = np.ascontiguousarray(bg_emb, dtype=np.float64)
    n_obs_total, n_bg_total = obs.shape[0], bg.shape[0]
    n_ref = max(n_obs_total - 1, 1)
    if n_obs_total == 0 or n_bg_total == 0:
        raise ValueError(
            f"neighbor_enrichment requires non-empty obs and bg embeddings "
            f"(got N_obs={n_obs_total}, N_bg={n_bg_total})."
        )
    if pseudocount <= 0:
        raise ValueError(
            f"pseudocount must be > 0 (got {pseudocount}); it stabilizes p_bg so a clonotype in a "
            "zero-background ball cannot produce an infinite fold."
        )

    if backend == "exact":
        obs_tree, bg_tree = BallTree(obs), BallTree(bg)
        if radius is None:  # balloon: fix expected background occupancy at lambda0
            k = int(round(lambda0 * n_bg_total / n_ref))
            k = min(max(k, 1), n_bg_total)
            rad = bg_tree.query(obs, k=k)[0][:, -1]  # per-point radius = k-th bg-neighbour distance
            radius_out = float(np.median(rad))
        else:  # fixed global radius
            rad = radius
            radius_out = float(radius)
        # count the *actual* background occupancy in the ball (>= k under ties), not a hardcoded k.
        n_bg = bg_tree.query_radius(obs, rad, count_only=True).astype(np.int64)

        def _count_obs():
            return (obs_tree.query_radius(obs, rad, count_only=True) - 1).astype(np.int64)

        def _lists_obs():
            return obs_tree.query_radius(obs, rad, return_distance=False)
    elif backend == "kdtree":  # exact, multithreaded scipy cKDTree — 5-9x faster than BallTree
        from scipy.spatial import cKDTree

        obs_tree, bg_tree = cKDTree(obs), cKDTree(bg)
        if radius is None:
            k = int(round(lambda0 * n_bg_total / n_ref))
            k = min(max(k, 1), n_bg_total)
            d = bg_tree.query(obs, k=k, workers=-1)[0]
            rad = d[:, -1] if d.ndim == 2 else d  # cKDTree returns 1-D for k==1
            radius_out = float(np.median(rad))
        else:
            rad = np.full(n_obs_total, float(radius))
            radius_out = float(radius)
        n_bg = bg_tree.query_ball_point(obs, rad, return_length=True, workers=-1).astype(np.int64)

        def _count_obs():
            return (obs_tree.query_ball_point(obs, rad, return_length=True, workers=-1) - 1).astype(np.int64)

        def _lists_obs():
            return [np.asarray(x, dtype=np.intp)
                    for x in obs_tree.query_ball_point(obs, rad, workers=-1)]
    elif backend == "ann":  # approximate NN (pynndescent) for whole-repertoire scale
        rad, radius_out, n_bg, _count_obs, _lists_obs = _ann_neighbors(
            obs, bg, radius, lambda0, n_ref)
    else:
        raise ValueError(f"backend must be 'exact', 'kdtree', or 'ann', got {backend!r}")
    p_bg = (n_bg + pseudocount) / (n_bg_total + pseudocount)
    expected = n_ref * p_bg

    weighted = abundance is not None and weight != "distinct"
    if not weighted:
        n_obs = _count_obs()
        if calibrate == "median":
            pos = expected > 0
            c = np.median(n_obs[pos]) / np.median(expected[pos]) if pos.any() else 1.0
            expected = expected * max(float(c), 1.0)  # water level: centre the bulk at fold ~ 1
        elif calibrate is not None:
            raise ValueError(f"calibrate must be 'median' or None, got {calibrate!r}")
        if test == "poisson":
            pvalue = stats.poisson.sf(n_obs - 1, expected)
        elif test == "binomial":
            pvalue = stats.binom.sf(n_obs - 1, n_ref, np.clip(expected / n_ref, 0.0, 1.0))
        else:
            raise ValueError(f"test must be 'poisson' or 'binomial', got {test!r}")
        pvalue = np.clip(np.asarray(pvalue, dtype=np.float64), 0.0, 1.0)
        qvalue = stats.false_discovery_control(pvalue, method="bh")
        return EnrichmentResult(
            n_obs=n_obs, n_bg=n_bg, expected=expected, fold=n_obs / expected,
            pvalue=pvalue, qvalue=qvalue, radius=radius_out,
        )

    # --- abundance-aware weighted count (appendix §T.6 sec:dens-abund) ---
    if weight not in _WEIGHTS:
        raise ValueError(f"weight must be one of {sorted(_WEIGHTS)}, got {weight!r}")
    a = np.asarray(abundance, dtype=np.float64)
    if a.shape[0] != n_obs_total:
        raise ValueError(f"abundance length {a.shape[0]} != N_obs {n_obs_total}")
    g = _WEIGHTS[weight](a)
    idx = _lists_obs()  # neighbour index lists (exact BallTree or approximate ANN)
    # vectorised weighted mass: flatten the ragged neighbour lists and one bincount, instead of a
    # Python sum per point (the whole-repertoire hot path — ~400k points).
    counts = np.fromiter((ix.size for ix in idx), dtype=np.int64, count=n_obs_total)
    cols = np.concatenate(idx) if n_obs_total else np.empty(0, dtype=np.intp)
    rows = np.repeat(np.arange(n_obs_total), counts)
    n_obs = counts - 1                                            # exclude self
    S = np.bincount(rows, weights=g[cols], minlength=n_obs_total) - g  # weighted mass, excl self
    mean_g = float(g.mean())
    phi = float((g * g).mean() / mean_g)  # dispersion index E[g²]/E[g] (=1 for g≡1)

    if calibrate == "median":
        pos = expected > 0
        mu_raw = expected * mean_g
        c = np.median(S[pos]) / np.median(mu_raw[pos]) if pos.any() else 1.0
        expected = expected * max(float(c), 1.0)
    elif calibrate is not None:
        raise ValueError(f"calibrate must be 'median' or None, got {calibrate!r}")
    mu_S = expected * mean_g  # expected weighted neighbour mass under H0
    # compound-Poisson null: Var = mu_S·φ -> moment-matched Gamma(shape=mu_S/φ, scale=φ) upper tail
    p_breadth = np.clip(stats.gamma.sf(S, a=np.clip(mu_S / phi, 1e-6, None), scale=phi), 0.0, 1.0)

    if orphan:
        p_size = _emp_survival(a)  # depth channel: P(A ≥ a_j) under the empirical size law
        chi2 = -2.0 * (np.log(np.clip(p_breadth, 1e-300, 1.0)) + np.log(np.clip(p_size, 1e-300, 1.0)))
        pvalue = np.clip(stats.chi2.sf(chi2, df=4), 0.0, 1.0)
    else:
        p_size = None
        pvalue = p_breadth
    qvalue = stats.false_discovery_control(pvalue, method="bh")
    return EnrichmentResult(
        n_obs=n_obs, n_bg=n_bg, expected=mu_S, fold=S / np.where(mu_S > 0, mu_S, 1.0),
        pvalue=pvalue, qvalue=qvalue, radius=radius_out,
        score=S, pvalue_breadth=p_breadth, pvalue_size=p_size,
    )


def enriched_mask(
    res: EnrichmentResult, *, alpha: float = 0.05, min_fold: float = 1.0, min_neighbors: int = 2
) -> np.ndarray:
    """Boolean hit mask: ``q < alpha`` & ``fold > min_fold`` & ``≥ min_neighbors`` (self + neighbours).

    ``min_neighbors=2`` reproduces the legacy "self plus at least one neighbour" criterion
    (``n_obs`` excludes self, so the threshold is ``n_obs ≥ min_neighbors − 1``).

    For an abundance-aware result (``res.score`` set) the ``min_fold``/``min_neighbors`` breadth
    gates are dropped: the combined breadth×depth ``q`` already governs, so a hyperexpanded orphan
    (significant depth, no neighbours) is kept rather than filtered out.
    """
    if res.score is not None:
        return res.qvalue < alpha
    return (
        (res.qvalue < alpha)
        & (res.fold > min_fold)
        & (res.n_obs >= min_neighbors - 1)
    )


def denoise_and_cluster(
    obs_emb: np.ndarray,
    res: EnrichmentResult,
    *,
    alpha: float = 0.05,
    min_fold: float = 1.0,
    min_neighbors: int = 2,
    **cluster_kw,
) -> tuple[np.ndarray, np.ndarray]:
    """Noise-filter then cluster: DBSCAN the enriched subset, non-enriched → label ``-1``.

    Background subtraction (:func:`enriched_mask`) removes the naive-repertoire noise, then
    :func:`mir.bench.metrics.cluster` groups the surviving hits into convergent motifs.

    Returns:
        ``(labels, mask)`` — ``labels`` is length ``N_obs`` (``-1`` for non-enriched or
        DBSCAN-noise points); ``mask`` is the enriched-hit boolean.
    """
    from mir.bench.metrics import cluster

    mask = enriched_mask(res, alpha=alpha, min_fold=min_fold, min_neighbors=min_neighbors)
    labels = np.full(obs_emb.shape[0], -1, dtype=np.int64)
    if int(mask.sum()) >= 5:  # DBSCAN eps estimation needs a handful of points
        labels[mask] = cluster(np.ascontiguousarray(obs_emb[mask]), **cluster_kw)
    return labels, mask


def generate_background(
    locus: str, n: int, *, source: str = "learned", seed: int = 0, productive_only: bool = True
) -> pl.DataFrame:
    """Draw ``n`` synthetic background clonotypes from the bundled vdjtools P_gen model.

    Lazy ``vdjtools.model`` import (``vdjtools`` is a core dependency; imported here only to defer
    the cost). Returns a frame with
    ``junction_aa``, ``v_call``, ``j_call`` — ready for ``TCREmp.embed`` — sampled from the
    vdjtools VDJ-rearrangement model, i.e. a Monte-Carlo estimate of the
    ``f_gen = φ_# P_gen`` pushforward (the ALICE analog background).

    Args:
        locus: Bundled model locus (e.g. ``"TRB"``).
        n: Number of sequences to generate.
        source: Bundled model source — ``"learned"`` (vdjtools EM-inferred, the default) or
            ``"olga"`` (legacy OLGA-parameter bootstrap, retained only for comparison).
        seed: Sampling seed.
        productive_only: Reject out-of-frame / stop-codon rearrangements.
    """
    from vdjtools.model import generate, load_bundled

    model = load_bundled(locus, source)
    df = generate.generate(model, n, seed=seed, productive_only=productive_only)
    return df.select(list(_REQUIRED_COLS))


if __name__ == "__main__":
    # Self-check: an injected tight cluster in an otherwise background-like observed set
    # must come out enriched; background-scattered observed points must not.
    rng = np.random.default_rng(0)
    d = 10
    bg = rng.standard_normal((5000, d))
    obs = rng.standard_normal((1000, d))
    center = np.full(d, 2.5)                       # a low-background region
    obs[:50] = center + 0.05 * rng.standard_normal((50, d))  # 50-point convergent cluster
    for label, res in (("fixed", neighbor_enrichment(obs, bg, radius=0.5)),
                       ("balloon", neighbor_enrichment(obs, bg, lambda0=3.0))):
        mask = enriched_mask(res)
        injected, background = mask[:50], mask[50:]
        assert injected.mean() > 0.8, (label, injected.mean())
        assert background.mean() < 0.05, (label, background.mean())
        assert res.fold[:50].mean() > res.fold[50:].mean()
        print(f"mir.density [{label}] OK; injected hit-rate {injected.mean():.2f}, "
              f"background hit-rate {background.mean():.3f}, "
              f"median injected fold {np.median(res.fold[:50]):.1f}")

    # abundance-aware: sizes add the depth channel. The convergent cluster (breadth) stays enriched;
    # a hyperexpanded clone gets the smallest size p-value (the orphan side-channel, for individual
    # inspection — a lone orphan is BH-conservative among N clones); concavity bounds the Zipf tail.
    a = np.ones(1000)
    a[500] = 5000.0                                    # a hyperexpanded clone among background
    res = neighbor_enrichment(obs, bg, lambda0=3.0, abundance=a, weight="log1p")
    mask = enriched_mask(res)
    assert res.score is not None and res.pvalue_size is not None
    assert mask[:50].mean() > 0.8, mask[:50].mean()    # breadth channel keeps the convergent cluster
    assert res.pvalue_size[500] == res.pvalue_size.min()  # depth channel flags the big clone
    assert res.score.max() < 100                       # concavity: O(log) mass, not O(size)
    print(f"mir.density [abundance] OK; cluster hit-rate {mask[:50].mean():.2f}, "
          f"orphan size-p {res.pvalue_size[500]:.4f}, max weighted mass {res.score.max():.1f}")
