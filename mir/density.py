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
        ).astype(np.float32)  # float32 halves the whole-repertoire memory footprint
    return _slice(model.embed(df), space).astype(np.float32, copy=False)


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
    expected: np.ndarray # expected observed neighbours under the background null
    fold: np.ndarray     # n_obs / expected == the density ratio E(z)
    pvalue: np.ndarray   # one-sided enrichment p-value
    qvalue: np.ndarray   # Benjamini-Hochberg adjusted p-value
    radius: float


def fit_density_space(
    model,
    obs_df: pl.DataFrame,
    bg_df: pl.DataFrame,
    *,
    n_components: int,
    space: str = "full",
    seed: int = 0,
    pca_fit_cap: int | None = None,
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
            pooled rows (then transform *all* rows). Lets whole repertoires be embedded
            without a full-matrix PCA; ``None`` fits on everything.

    Returns:
        ``(density_space, obs_emb, bg_emb)`` — the fitted :class:`DensitySpace` and the
        two reduced ``float`` arrays, row-aligned to ``obs_df`` / ``bg_df``.
    """
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
    return float(np.quantile(drift, quantile))


def neighbor_enrichment(
    obs_emb: np.ndarray,
    bg_emb: np.ndarray,
    radius: float | None = None,
    *,
    lambda0: float = 3.0,
    test: str = "poisson",
    pseudocount: float = 1.0,
    calibrate: str | None = "median",
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

    Args:
        obs_emb: ``(N_obs, d)`` observed embedding.
        bg_emb: ``(N_bg, d)`` background embedding, **same basis** as ``obs_emb``
            (produce both with :func:`fit_density_space`). Use ``M ≥ 5N`` for a stable ratio.
        radius: Fixed neighbourhood radius; ``None`` selects the balloon estimator.
        lambda0: Target expected background occupancy for the balloon estimator (``[1, 5]``).
        test: ``"poisson"`` (ALICE analog) or ``"binomial"`` (TCRNET control analog).
        pseudocount: Added to the background count to stabilize ``p_bg``.
        calibrate: ``"median"`` empirical-null water-level calibration (default), or ``None``.

    Returns:
        An :class:`EnrichmentResult`; ``fold`` is the (calibrated) density ratio ``E(z)``. In
        balloon mode ``radius`` is the median adaptive radius used.
    """
    obs = np.ascontiguousarray(obs_emb, dtype=np.float64)
    bg = np.ascontiguousarray(bg_emb, dtype=np.float64)
    n_obs_total, n_bg_total = obs.shape[0], bg.shape[0]
    n_ref = max(n_obs_total - 1, 1)
    obs_tree, bg_tree = BallTree(obs), BallTree(bg)

    if radius is None:  # balloon: fix expected background occupancy at lambda0
        k = int(round(lambda0 * n_bg_total / n_ref))
        k = min(max(k, 1), n_bg_total)
        rad = bg_tree.query(obs, k=k)[0][:, -1]  # per-point radius = k-th bg-neighbour distance
        # count the *actual* background occupancy in that ball (>= k under ties/duplicate
        # embeddings), not a hardcoded k — hardcoding would understate dense regions and bias
        # the test anti-conservative.
        n_bg = bg_tree.query_radius(obs, rad, count_only=True).astype(np.int64)
        n_obs = (obs_tree.query_radius(obs, rad, count_only=True) - 1).astype(np.int64)
        radius_out = float(np.median(rad))
    else:  # fixed global radius
        n_obs = (obs_tree.query_radius(obs, radius, count_only=True) - 1).astype(np.int64)
        n_bg = bg_tree.query_radius(obs, radius, count_only=True).astype(np.int64)
        radius_out = float(radius)

    p_bg = (n_bg + pseudocount) / (n_bg_total + pseudocount)
    expected = n_ref * p_bg

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


def enriched_mask(
    res: EnrichmentResult, *, alpha: float = 0.05, min_fold: float = 1.0, min_neighbors: int = 2
) -> np.ndarray:
    """Boolean hit mask: ``q < alpha`` & ``fold > min_fold`` & ``≥ min_neighbors`` (self + neighbours).

    ``min_neighbors=2`` reproduces the legacy "self plus at least one neighbour" criterion
    (``n_obs`` excludes self, so the threshold is ``n_obs ≥ min_neighbors − 1``).
    """
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

    Lazy ``vdjtools`` dependency (``[rearrangement]``). Returns a frame with
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
