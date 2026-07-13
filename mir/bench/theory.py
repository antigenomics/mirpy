"""Theory-validation experiments (reproduce TCREMP supplementary S1–S3).

These numerically check the properties the embedding is claimed to have, using the
actual v3 pipeline distances (``seqtree.gapblock`` junction dissimilarity):

* **S2 / Theory T1** — Euclidean distance ``D_ij`` in embedding space tracks the
  pairwise junction dissimilarity ``d_ij`` (:func:`s2_dissimilarity_distance_correlation`;
  paper Pearson R ≈ 0.56). Using the sequences as their own prototypes, the embedding
  of sequence *i* is row *i* of the dissimilarity matrix and ``D_ij = ‖d_i − d_j‖₂``.
* **S1 / Theory T4** — ``d_ij`` follows a Gamma law (Gaussian fails); ``D_ij`` follows
  a Fréchet (GEV, shape ξ>0) law (:func:`fit_distributions`).
* **S3** — distances from *real* vs *model* prototypes agree (:func:`prototype_source_correlation`;
  paper Pearson R ≈ 0.96), i.e. the prototype source barely matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist

from mir.distances.junction import junction_distance_matrix


def junction_dissimilarity(cdr3s, threads: int = 0) -> np.ndarray:
    """Symmetric ``(n, n)`` junction dissimilarity ``d_ij`` (v3 gapblock metric)."""
    return junction_distance_matrix(cdr3s, cdr3s, threads=threads)


def junction_dissimilarity_sw(cdr3s) -> np.ndarray:
    """Paper-exact ``d_ij`` via Smith-Waterman BLOSUM62 (supplementary S1/S2).

    ``d_ij = s_ii + s_jj − 2·s_ij`` from a local BLOSUM62 alignment (linear gap).
    Requires BioPython (``[bench]``/``[build]`` extra). O(n²) alignments — use for
    the theory validation on a few thousand sequences, not at scale.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0  # linear gap penalty
    seqs = list(cdr3s)
    n = len(seqs)
    s = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        s[i, i] = aligner.score(seqs[i], seqs[i])
    for i in range(n):
        for j in range(i + 1, n):
            s[i, j] = s[j, i] = aligner.score(seqs[i], seqs[j])
    return s.diagonal()[:, None] + s.diagonal()[None, :] - 2.0 * s


@dataclass
class CorrelationResult:
    n: int
    n_pairs: int
    pearson: float
    d: np.ndarray  # flat i>j dissimilarities
    D: np.ndarray  # flat i>j embedding-space Euclidean distances


def s2_dissimilarity_distance_correlation(
    cdr3s, threads: int = 0, dissimilarity: str = "gapblock"
) -> CorrelationResult:
    """Correlate junction dissimilarity ``d_ij`` with embedding distance ``D_ij``.

    Uses the *self-prototype* construction from supplementary Fig. S2: the embedding
    of each sequence is its row of the dissimilarity matrix, and ``D_ij`` is the
    Euclidean distance between rows. Returns the flat i>j vectors and Pearson R.

    Args:
        dissimilarity: ``"gapblock"`` (the v3 pipeline metric) or ``"sw"``
            (paper-exact Smith-Waterman BLOSUM62).
    """
    if dissimilarity == "sw":
        d = junction_dissimilarity_sw(cdr3s).astype(np.float64)
    else:
        d = junction_dissimilarity(cdr3s, threads=threads).astype(np.float64)
    n = d.shape[0]
    iu = np.triu_indices(n, k=1)
    d_flat = d[iu]                       # dissimilarities, i>j order
    D_flat = pdist(d, metric="euclidean")  # same i>j order as triu_indices(n,1)
    r = float(np.corrcoef(d_flat, D_flat)[0, 1])
    return CorrelationResult(n=n, n_pairs=d_flat.size, pearson=r, d=d_flat, D=D_flat)


def _aic(logpdf_sum: float, k: int) -> float:
    return 2 * k - 2 * logpdf_sum


def _fit_one(data: np.ndarray, dist, floc: bool, init: tuple = ()) -> dict:
    if init:
        params = dist.fit(data, *init, loc=float(np.median(data)), scale=float(data.std()))
    elif floc:
        params = dist.fit(data, floc=0)
    else:
        params = dist.fit(data)
    logL = float(np.sum(dist.logpdf(data, *params)))
    ks = float(stats.kstest(data, dist.cdf, args=params).statistic)
    return {"params": params, "aic": _aic(logL, len(params)), "ks": ks}


def fit_distributions(
    d_flat: np.ndarray, D_flat: np.ndarray, sample: int = 200_000, seed: int = 0
) -> dict:
    """Fit S1 distributions: Gamma vs Normal for ``d``, GEV/Fréchet vs Normal for ``D``.

    Returns a nested dict of ``{ks, aic, params}`` per fit plus the GEV shape ``xi``
    (``xi > 0`` ⇒ Fréchet). Large inputs are subsampled to *sample* points.
    """
    rng = np.random.default_rng(seed)

    def _sub(x):
        return x if x.size <= sample else rng.choice(x, sample, replace=False)

    d, D = _sub(np.asarray(d_flat, float)), _sub(np.asarray(D_flat, float))
    d = d[d > 0]
    # genextreme MLE is unstable without a good shape init; c=-xi, seed near Frechet
    gev = _fit_one(D, stats.genextreme, floc=False, init=(-0.1,))
    return {
        # free location: pairwise d is concentrated far from 0, so forcing loc=0 misfits
        "d_gamma": _fit_one(d, stats.gamma, floc=False),
        "d_normal": _fit_one(d, stats.norm, floc=False),
        "D_gev": gev,
        "D_normal": _fit_one(D, stats.norm, floc=False),
        "D_gev_xi": float(-gev["params"][0]),  # scipy genextreme c = -xi
    }


_AA = "ACDEFGHIKLMNPQRSTVWY"


def _mutate(seq: str, k: int, rng) -> str:
    """Apply *k* random interior amino-acid substitutions (an SHM proxy)."""
    s = list(seq)
    interior = list(range(1, len(s) - 1))  # keep the conserved C…[FW] ends
    if not interior:
        return seq
    for p in rng.choice(interior, min(k, len(interior)), replace=False):
        s[p] = _AA[int(rng.integers(20))]
    return "".join(s)


def shm_embedding_drift(
    cdr3s, prototypes, max_mut: int = 8, n_rep: int = 3, seed: int = 0, threads: int = 0
) -> dict[int, tuple[float, float]]:
    """Embedding drift vs mutation load (Theory T5, the IGH/SHM case).

    Applies ``k`` random interior substitutions (a somatic-hypermutation proxy) to each
    CDR3 and measures the junction-embedding Euclidean distance to the original,
    ``D_k = ‖φ(mutated) − φ(original)‖``. Returns ``k -> (mean D_k, std)``. The claim: the
    drift is *bounded by the mutation load* — ``D_k`` grows ~linearly in ``k`` — so heavily
    hypermutated IGH sequences sit controllably far from germline in embedding space.
    """
    rng = np.random.default_rng(seed)
    seqs = list(cdr3s)
    base = junction_distance_matrix(seqs, prototypes, threads=threads).astype(np.float64)
    out: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    for k in range(1, max_mut + 1):
        drift: list[float] = []
        for _ in range(n_rep):
            emb = junction_distance_matrix([_mutate(s, k, rng) for s in seqs],
                                           prototypes, threads=threads).astype(np.float64)
            drift.extend(np.linalg.norm(emb - base, axis=1).tolist())
        out[k] = (float(np.mean(drift)), float(np.std(drift)))
    return out


def prototype_source_correlation(
    query_cdr3s, real_prototypes, model_prototypes, threads: int = 0
) -> dict:
    """Correlate embedding distances built from *real* vs *model* prototypes (S3).

    Embeds ``query_cdr3s`` against each prototype set (junction only), takes pairwise
    Euclidean distances under each embedding, and returns their Pearson R (paper ≈ 0.96).
    """
    Xr = junction_distance_matrix(query_cdr3s, real_prototypes, threads=threads).astype(np.float64)
    Xm = junction_distance_matrix(query_cdr3s, model_prototypes, threads=threads).astype(np.float64)
    Dr, Dm = pdist(Xr), pdist(Xm)
    return {"pearson": float(np.corrcoef(Dr, Dm)[0, 1]), "n_pairs": Dr.size}


def _hamming1_counts(seqs) -> np.ndarray:
    """Discrete Hamming-1 neighbour count per sequence (equal-length, exactly one mismatch)."""
    seqs = list(seqs)
    out = np.zeros(len(seqs), dtype=np.int64)
    for i, a in enumerate(seqs):
        la, c = len(a), 0
        for j, b in enumerate(seqs):
            if i == j or len(b) != la:
                continue
            m = 0
            for x, y in zip(a, b):
                if x != y:
                    m += 1
                    if m > 1:
                        break
            if m == 1:
                c += 1
        out[i] = c
    return out


def tcrnet_convergence(
    obs_cdr3s, bg_cdr3s, prototypes, *, n_components: int = 25,
    scales=(0.5, 1.0, 1.5, 2.0, 3.0), seed: int = 0, threads: int = 0
) -> dict:
    """T6: continuous embedding enrichment converges to discrete Hamming-1 enrichment.

    Embeds observed and background CDR3s against the prototype set (junction only), reduces
    with one shared ``StandardScaler → PCA``, and at radius ``scale × r₁`` (``r₁`` = the
    median one-substitution embedding drift) counts each observed clonotype's neighbours in
    embedding space. Returns the Spearman correlation between those continuous counts and the
    discrete Hamming-1 neighbour counts. The correlation is high at small radii and fades as
    the radius grows past one substitution — numerically confirming that graph
    neighbour-enrichment (TCRNET/ALICE) is the ``r→0`` limit of the density ratio ``E(z)``.

    Args:
        obs_cdr3s: Observed junction sequences.
        bg_cdr3s: Background junction sequences (P_gen sample / control).
        prototypes: Prototype junctions defining the embedding.
        n_components: PCA dimensionality of the shared coordinate system.
        scales: Radius multiples of the one-substitution scale ``r₁`` to evaluate.
        seed: RNG seed for the calibration mutations and PCA solver.

    Returns:
        ``{radius_1sub, hamming1_mean, spearman_by_scale, spearman_at_1sub}``.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import BallTree
    from sklearn.preprocessing import StandardScaler

    obs = junction_distance_matrix(obs_cdr3s, prototypes, threads=threads).astype(np.float64)
    bg = junction_distance_matrix(bg_cdr3s, prototypes, threads=threads).astype(np.float64)
    scaler = StandardScaler().fit(np.vstack([obs, bg]))
    k = min(n_components, obs.shape[0] + bg.shape[0], obs.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(scaler.transform(np.vstack([obs, bg])))
    obs_emb = pca.transform(scaler.transform(obs))

    rng = np.random.default_rng(seed)
    mut = [_mutate(s, 1, rng) for s in obs_cdr3s]
    mut_emb = pca.transform(scaler.transform(
        junction_distance_matrix(mut, prototypes, threads=threads).astype(np.float64)))
    r1 = float(np.median(np.linalg.norm(obs_emb - mut_emb, axis=1)))

    discrete = _hamming1_counts(obs_cdr3s)
    tree = BallTree(obs_emb)
    corr = {}
    for f in scales:
        n_obs = tree.query_radius(obs_emb, r1 * f, count_only=True) - 1
        corr[round(float(f), 2)] = float(stats.spearmanr(n_obs, discrete).statistic)
    return {
        "radius_1sub": r1,
        "hamming1_mean": float(discrete.mean()),
        "spearman_by_scale": corr,
        "spearman_at_1sub": corr[1.0],
    }
