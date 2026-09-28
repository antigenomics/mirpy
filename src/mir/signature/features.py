"""Raw rsig features for one sample: the prototype-sum measure and its functionals.

Everything rests on one object::

    Phi(S) = sum_sigma  w_sigma * z_sigma

``z_sigma`` is a clonotype's vector of distances to a fixed, bundled prototype panel (K receptors
per locus, embedded by germline V/J distance plus junction gapblock alignment), and ``w_sigma`` is
its normalised clone weight. ``TCREmp.embed`` interleaves the three components per prototype as
``[V, J, junction]``, so ``Phi[0::3]``, ``Phi[1::3]`` and ``Phi[2::3]`` are the exact V / J /
junction slots -- literal column strides, not an attribution model, which is what makes "how much of
this distance is V?" answerable without SHAP or a surrogate.

**No rotation and no standardisation here.** Both need a corpus, and both live in
:mod:`vdjtools.signature.corpus`, which this half shares with ``vsig``. That is the correction this
rewrite exists for: the previous rotation was fitted on 10,000 individual clonotypes from the
prototype panel while every column it produced was a repertoire statistic.

**There is no ``contrast`` group any more, and nothing was lost.** It used to be
``Psi = mass * (Phi - naive)``, with ``naive`` a separately drawn 20,000-sequence reference, because
the rotation was fit-free and needed an explicit subtraction point. The corpus centre now *is* that
point: rotating through the ``naive`` corpus subtracts the median ``Phi`` of unselected repertoires,
which is what the contrast measured. 231 columns, one frozen vector and one whole failure mode --
a ``naive`` drawn against a different release of the recombination models -- replaced by choosing a
corpus. ``mass`` remains a feature in its own right, so the rotation still sees it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from vdjtools.signature import transform as T
from vdjtools.signature.features import WEIGHTS

#: Rows embedded per pass. Bounds **memory**, not time: ``Phi`` and the Rao accumulator are both
#: running sums, so the full ``(n, 3K)`` matrix is never held.
CHUNK = 50_000

#: Prototypes per locus. The panel is a bundled resource and this is its size; changing it changes
#: the raw column set, which the corpus artifact's load-time check will refuse rather than reindex.
K = 256

#: Abundance compartments, as row predicates over the clone-size vector. A *partition* -- the
#: mixture identity below is only exact for a partition, and an NNLS over overlapping parts is not a
#: composition at all: its weights need not sum to one and one share can exceed it.
#:
#: **These are depth-fragile, deliberately uncorrected.** A compartment's share genuinely moves with
#: depth: the singleton fraction grows as rarer clones are sampled, and a 1% quantile selects 20
#: clonotypes in a 2,000-clonotype sample against 1,000 in a 100,000-clonotype one. Measured on one
#: repertoire across a 67x depth range, ``band:top`` spans about 6.9 in log-ratio coordinates.
#: Bounding the quantile to a clonotype count was tried and merely relocated the discontinuity.
#: The answer to a depth-fragile column is to carry the covariate, not to correct it, which is why
#: ``depth`` is in the rotation and ``cov:*:cstar`` is a channel.
BANDS: dict[str, "callable"] = {
    "singleton": lambda a: a == 1,
    "middle": lambda a: (a > 1) & (a < np.maximum(np.quantile(a, 0.99), 2)),
    "top": lambda a: a >= np.maximum(np.quantile(a, 0.99), 2),
}

#: IGH isotype compartments, by constant-gene call. ``IGHGP`` is a pseudogene and ``IGHC`` is
#: ambiguous, so neither is called; roughly two fifths of IGH reads carry no call at all and form
#: their own part rather than being folded into IgM.
ISOTYPE_BANDS: dict[str, tuple[str, ...]] = {
    "IgM": ("IGHM", "IGHD"),
    "IgG": ("IGHG1", "IGHG2", "IGHG3", "IGHG4"),
    "IgA": ("IGHA1", "IGHA2"),
}

#: Columns that must not reach the embedder. Their presence silently switches ``TCREmp.embed`` to
#: SHM-aware V distances, which is a different coordinate system from the one the corpus was fitted
#: in -- and it changes the numbers without changing any name.
_SHM_COLUMNS = ("v_identity", "v_mutations")


def weights(counts: np.ndarray, weight: str = "log2p1") -> np.ndarray:
    """Normalised clone weights ``w = g(a)/sum(g)``.

    Raises:
        ValueError: If ``weight`` is unknown, or no clonotype carries any weight.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"unknown weight {weight!r}; known: {sorted(WEIGHTS)}")
    g = WEIGHTS[weight](np.asarray(counts, dtype=float))
    s = g.sum()
    if s <= 0:
        raise ValueError("every clone weight is zero -- the sample carries no usable counts")
    return g / s


def prototype_sum(df: pl.DataFrame, model, w: np.ndarray, *, chunk: int = CHUNK):
    """``Phi = sum w_sigma z_sigma`` and its Rao dispersion, in one chunked pass.

    Both are running sums over the rows, so the full ``(n, 3K)`` matrix is never held: the
    accumulators are ``sum w z`` and ``sum w ||z||^2``, and Rao's ``Q`` telescopes out of the pair as
    ``2(sum w||z||^2 - ||Phi||^2)``.

    Returns:
        ``(phi, mean_sq_norm)`` -- ``phi`` is ``(3K,)``, ``mean_sq_norm`` is ``sum w ||z||^2``.
    """
    phi = np.zeros(3 * model.n_prototypes)
    mean_sq = 0.0
    for a in range(0, df.height, chunk):
        part = df.slice(a, chunk)
        z = np.asarray(model.embed(part), dtype=float)
        wp = w[a:a + part.height]
        phi += wp @ z
        mean_sq += float(wp @ (z * z).sum(axis=1))
    return phi, mean_sq


@dataclass
class Dispersion:
    """Everything the embedding-diversity family needs, from one chunked pass over a repertoire.

    Every field is a running sum, so the ``(n, 3K)`` clonotype matrix is never held. The one field
    that is not a vector -- ``second``, the ``(3K, 3K)`` weighted second moment -- is what the
    spectrum comes from, and accumulating it rather than the ``n x n`` Gram is what keeps the pass
    **linear in sequencing depth**: the Gram is quadratic, which on a deep repertoire is the whole
    cost of the signature rather than a few milliseconds of it.

    Attributes:
        phi: ``sum w z``, shape ``(3K,)``.
        mean_sq: ``sum w ||z||^2``.
        stride_sq: ``sum w ||z[s::3]||^2`` for ``s`` in V, J, junction order, shape ``(3,)``.
        second: ``sum w z z^T``, shape ``(3K, 3K)``.
        n_eff: ``1 / sum w^2``, the effective clonotype count under the weighting.
        n_rows: Clonotypes that entered the pass.
        uni_phi: ``phi`` again under uniform weights -- the reference for ``evenness``.
        uni_mean_sq: ``mean_sq`` under uniform weights.
        bands: ``{band: (mass, phi_b, mean_sq_b, sum_w2_b, n_b)}`` for each clone-size or isotype
            compartment, with sums taken over that compartment only and **not** renormalised; the
            caller divides by ``mass`` to get the within-compartment measure.
        rows: The embedded clonotypes and their weights, ``(z, w)``, kept **only** when there are
            no more of them than there are coordinates. That is the one case where the ``n x n``
            weighted Gram is cheaper to decompose than the ``p x p`` covariance, and both have the
            same non-zero eigenvalues. ``None`` above that, where keeping them would be the
            quadratic-in-depth term this whole design exists to avoid.
    """

    phi: np.ndarray
    mean_sq: float
    stride_sq: np.ndarray
    second: np.ndarray
    n_eff: float
    n_rows: int
    uni_phi: np.ndarray
    uni_mean_sq: float
    bands: dict
    rows: "tuple[np.ndarray, np.ndarray] | None" = None


def dispersion_pass(df: pl.DataFrame, model, w: np.ndarray, *,
                    band_masks: "dict[str, np.ndarray] | None" = None,
                    chunk: int = CHUNK) -> Dispersion:
    """One chunked pass yielding every accumulator :func:`diversity_channels` reads.

    A superset of :func:`prototype_sum`, and it re-embeds nothing: the same ``z`` that produces
    ``phi`` produces the strides, the uniform-weight sums, the compartment sums and the second
    moment. Adding a metric that reads these costs no extra embedding.

    Args:
        df: One locus's sanitised clonotype frame, SHM columns already dropped.
        model: The embedder for this locus.
        w: Normalised clone weights, ``sum w == 1``.
        band_masks: ``{name: boolean mask over df's rows}``; each becomes a ``bands`` entry.
        chunk: Rows embedded per pass; bounds memory, not time.
    """
    p = 3 * model.n_prototypes
    phi = np.zeros(p)
    uni_phi = np.zeros(p)
    second = np.zeros((p, p))
    stride_sq = np.zeros(3)
    mean_sq = uni_mean_sq = 0.0
    u = 1.0 / df.height if df.height else 0.0
    masks = band_masks or {}
    acc = {name: [0.0, np.zeros(p), 0.0, 0.0, int(m.sum())] for name, m in masks.items()}
    keep: "list[np.ndarray] | None" = [] if df.height <= p else None

    for a in range(0, df.height, chunk):
        part = df.slice(a, chunk)
        z = np.asarray(model.embed(part), dtype=float)
        wp = w[a:a + part.height]
        sq = (z * z)
        phi += wp @ z
        mean_sq += float(wp @ sq.sum(axis=1))
        for sidx in range(3):
            stride_sq[sidx] += float(wp @ sq[:, sidx::3].sum(axis=1))
        uni_phi += u * z.sum(axis=0)
        uni_mean_sq += u * float(sq.sum())
        second += z.T @ (wp[:, None] * z)
        if keep is not None:
            keep.append(z)
        for name, m in masks.items():
            mp = m[a:a + part.height]
            if not mp.any():
                continue
            wb = wp[mp]
            zb = z[mp]
            e = acc[name]
            e[0] += float(wb.sum())
            e[1] += wb @ zb
            e[2] += float(wb @ (zb * zb).sum(axis=1))
            e[3] += float(wb @ wb)

    n_eff = 1.0 / float(w @ w) if df.height else float("nan")
    return Dispersion(phi=phi, mean_sq=mean_sq, stride_sq=stride_sq, second=second, n_eff=n_eff,
                      n_rows=df.height, uni_phi=uni_phi, uni_mean_sq=uni_mean_sq,
                      bands={k: tuple(v) for k, v in acc.items()},
                      rows=(np.vstack(keep), w) if keep else None)


def _rao(mean_sq: float, phi_sq: float, n_eff: float) -> float:
    """Rao's ``Q`` from the two accumulators, self-pair corrected. See :func:`rao_of`."""
    q = 2.0 * (mean_sq - phi_sq)
    if n_eff > 1.0:
        q *= n_eff / (n_eff - 1.0)
    return max(q, 0.0)


def _eff_dim(acc: "Dispersion") -> tuple[float, float]:
    """Effective dimension of the weighted clonotype cloud, orders 1 and 2.

    ``exp(H(lambda))`` and ``(sum lambda)^2 / sum lambda^2`` over the eigenvalues of the weighted
    covariance ``sum w (z - phi)(z - phi)^T = second - phi phi^T``. This is richness with the
    geometry put back: a thousand clones inside one convergent cluster occupy few directions of
    receptor space and a thousand unrelated clones occupy many, and no clonotype count can tell
    those two apart.

    **The order-2 number needs no spectrum.** ``sum lambda`` is the trace and ``sum lambda^2`` is
    the squared Frobenius norm, both exactly and for free, so ``eff_dim_pr`` costs two reductions
    over a matrix we already hold. Only the Shannon term needs eigenvalues, and that is a cubic
    decomposition -- 25.7 ms at ``p = 768`` against 0.29 ms at 100 -- so it is taken on the
    **smaller** of the ``p x p`` covariance and the ``n x n`` weighted Gram, which have identical
    non-zero eigenvalues.
    """
    cov = acc.second - np.outer(acc.phi, acc.phi)
    cov = 0.5 * (cov + cov.T)                       # symmetrise: the accumulation is not exact
    total = float(np.trace(cov))
    sq = float((cov * cov).sum())
    if total <= 0 or sq <= 0:
        return float("nan"), float("nan")
    eff_dim_pr = (total * total) / sq

    if acc.rows is not None:
        z, w = acc.rows
        a = np.sqrt(w)[:, None] * (z - acc.phi)
        lam = np.clip(np.linalg.eigvalsh(a @ a.T), 0.0, None)
    else:
        lam = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s = lam.sum()
    if s <= 0:
        return float("nan"), eff_dim_pr
    q = lam / s
    nz = q[q > 0]
    return float(np.exp(-(nz * np.log(nz)).sum())), eff_dim_pr


def diversity_channels(acc: Dispersion, locus: str) -> dict[str, float]:
    """The embedding-diversity channel family for one locus, from one :func:`dispersion_pass`.

    Rao quadratic entropy is the metric-space analogue of Simpson diversity, and until now it was
    the only diversity read-out the geometry half gave up -- everything else it knew was inside a
    rotated component, which has no name a domain reader can use. These are the rest of the
    analogues, all from accumulators the Phi pass already forms:

    ================  =========================================================================
    ``rao``           Rao's ``Q`` over the whole embedding. Simpson, with distance.
    ``q_v/q_j/q_c``   The same restricted to the V, J and junction strides. The strides are
                      literal column offsets, so "how much of this is V-driven" needs no
                      attribution model.
    ``q_frac_*``      Each stride's share of the total dispersion, in clr coordinates -- a
                      composition of *where* the diversity sits. No counting index has this.
    ``evenness``      ``Q(w) / Q(uniform)``: the clone-size weighting's effect with the
                      composition held fixed. Bounded, and far less depth-fragile than richness.
    ``eff_dim``       ``exp(H(lambda))`` of the weighted covariance spectrum -- richness as
                      *directions of receptor space occupied*.
    ``eff_dim_pr``    ``(sum lambda)^2 / sum lambda^2``, the order-2 version, led by the
                      dominant directions.
    ``q_top``,        Rao **inside** a clone-size compartment, weights renormalised within it --
    ``q_singleton``   the diversity *of* the expanded compartment rather than its share, which
                      ``band`` already carries.
    ``q_ratio_top``   ``q_top / q_singleton``, the two compared directly.
    ================  =========================================================================

    Returns:
        ``{column: value}``, every value a ``rsig:div:<locus>:*`` channel. Holes are ``nan``.
    """
    pre = f"rsig:div:{locus}"
    out: dict[str, float] = {}
    phi_sq = float(acc.phi @ acc.phi)
    rao = _rao(acc.mean_sq, phi_sq, acc.n_eff)
    out[f"{pre}:rao"] = T.log1p(rao)

    # Per-stride Rao. phi[s::3] is the same stride of the mean, so each is a self-contained Rao in
    # its own subspace and the three sum to the total by orthogonality of the strides.
    qs = {}
    for name, sidx in (("v", 0), ("j", 1), ("c", 2)):
        qs[name] = _rao(float(acc.stride_sq[sidx]),
                        float(acc.phi[sidx::3] @ acc.phi[sidx::3]), acc.n_eff)
        out[f"{pre}:q_{name}"] = T.log1p(qs[name])
    total = sum(qs.values())
    frac = T.clr(qs, m=3) if total > 0 else {}
    for name in ("v", "j", "c"):
        out[f"{pre}:q_frac_{name}"] = float(frac.get(name, np.nan))

    uni_rao = _rao(acc.uni_mean_sq, float(acc.uni_phi @ acc.uni_phi),
                   float(acc.n_rows) if acc.n_rows else float("nan"))
    out[f"{pre}:evenness"] = float(rao / uni_rao) if uni_rao > 0 else float("nan")

    ed, edp = _eff_dim(acc)
    out[f"{pre}:eff_dim"] = T.log1p(ed) if np.isfinite(ed) else float("nan")
    out[f"{pre}:eff_dim_pr"] = T.log1p(edp) if np.isfinite(edp) else float("nan")

    band_q: dict[str, float] = {}
    for name in ("top", "singleton"):
        e = acc.bands.get(name)
        if e is None or e[0] <= 0:
            out[f"{pre}:q_{name}"] = float("nan")
            continue
        mass, phi_b, msq_b, sw2_b, _n = e
        # Renormalise INSIDE the compartment: w_b = w / mass, so phi_b/mass and msq_b/mass are the
        # compartment's own mean and second moment, and its n_eff is mass^2 / sum w^2.
        band_q[name] = _rao(msq_b / mass, float(phi_b @ phi_b) / (mass * mass),
                            (mass * mass) / sw2_b if sw2_b > 0 else float("nan"))
        out[f"{pre}:q_{name}"] = T.log1p(band_q[name])
    out[f"{pre}:q_ratio_top"] = (float(band_q["top"] / band_q["singleton"])
                                 if band_q.get("singleton", 0.0) > 0 and "top" in band_q
                                 else float("nan"))
    return out


def displacement_channels(acc: Dispersion, locus: str) -> dict[str, float]:
    """``disp`` channels: distances and cosines **between compartment centroids** of one locus.

    A displacement is a quantity no counting index has: where the expanded compartment sits
    relative to the singleton one, rather than how large either is.

    NOTE: only ever **within** one locus. Each locus has its own panel of ``K`` prototype receptors,
    so ``Phi(TRA)`` and ``Phi(TRB)`` are vectors in different spaces and a distance between them is
    arithmetic without a meaning. Cross-locus comparison belongs in ``vsig:pair``, which is a ratio
    of scalars.
    """
    pre = f"rsig:disp:{locus}"
    out: dict[str, float] = {}

    def centroid(name):
        e = acc.bands.get(name)
        return None if e is None or e[0] <= 0 else e[1] / e[0]

    pairs = [("top_singleton", "top", "singleton")]
    if locus == "IGH":
        pairs += [("IgG_IgM", "IgG", "IgM"), ("IgA_IgM", "IgA", "IgM")]
    for label, a_name, b_name in pairs:
        ca, cb = centroid(a_name), centroid(b_name)
        if ca is None or cb is None:
            out[f"{pre}:{label}"] = float("nan")
            out[f"{pre}:cos_{label}"] = float("nan")
            continue
        out[f"{pre}:{label}"] = T.log1p(float(np.linalg.norm(ca - cb)))
        na, nb = float(np.linalg.norm(ca)), float(np.linalg.norm(cb))
        out[f"{pre}:cos_{label}"] = float(ca @ cb / (na * nb)) if na > 0 and nb > 0 else float("nan")
    out[f"{pre}:norm"] = T.log1p(float(np.linalg.norm(acc.phi)))
    return out


def slots(phi: np.ndarray) -> dict[str, np.ndarray]:
    """Split ``Phi`` into its ``V`` / ``J`` / ``junction`` strides, exactly."""
    return {"phiv": phi[0::3], "phij": phi[1::3], "phic": phi[2::3]}


def rao_of(phi: np.ndarray, mean_sq: float, n_eff: float) -> float:
    """Rao quadratic entropy in embedding coordinates, self-pair corrected.

    Sees that two clonotypes are one substitution apart, which no Hill number can. The
    ``n_eff/(n_eff-1)`` correction removes the self-pair bias, which is ``O(1/n_eff)`` and therefore
    tracks depth -- exactly the confounder this column would otherwise smuggle in.
    """
    rao = 2.0 * (mean_sq - float(phi @ phi))
    if n_eff > 1.0:
        rao *= n_eff / (n_eff - 1.0)
    return max(rao, 0.0)


def band_shares(df: pl.DataFrame, w: np.ndarray, *, bands: "dict | None" = None,
                min_clonotypes: int = 5) -> dict[str, float]:
    """Compartment shares of ``Phi``, in closed form rather than by NNLS.

    ``Phi`` is linear in the clone-weight measure and the compartments partition the clonotypes, so
    compartment ``c``'s share of ``Phi`` is exactly its share of the weight::

        Phi(S) = sum_c pi_c Phi(c)        with   pi_c = sum_{sigma in c} w_sigma

    with no fitting. A compartment below ``min_clonotypes`` is recorded **absent** -- dropped from
    the composition -- rather than set to zero. Zero is a measurement; absent is not.

    Returns:
        ``{band: share}`` over the bands that cleared the floor, plus ``_residual``. Raw shares; the
        caller closes them into log-ratio coordinates.
    """
    a = df["duplicate_count"].to_numpy()
    out: dict[str, float] = {}
    claimed = 0.0
    for name, pred in (bands or BANDS).items():
        m = np.asarray(pred(a), dtype=bool)
        if int(m.sum()) < min_clonotypes:
            continue
        out[name] = float(w[m].sum())
        claimed += out[name]
    out["_residual"] = max(1.0 - claimed, 0.0)
    return out


def isotype_masks(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """``{isotype: boolean mask}`` over ``df``'s rows, for the IGH displacement channels.

    Allele-stripped for the same reason :func:`isotype_shares` strips: the classes are gene names
    matched by equality, so ``IGHG1*01`` would match nothing and every read would come back
    uncalled -- a composition rather than an error.
    """
    from vdjtools.io.schema import strip_allele

    if "c_call" not in df.columns:
        return {}
    calls = df.select(strip_allele(pl.col("c_call").cast(pl.Utf8)).alias("c"))["c"].to_list()
    return {name: np.array([c in names for c in calls], dtype=bool)
            for name, names in ISOTYPE_BANDS.items()}


def isotype_shares(df: pl.DataFrame, w: np.ndarray, *,
                   min_clonotypes: int = 5) -> dict[str, float]:
    """Isotype shares of ``Phi(IGH)``, by the same mixture identity as :func:`band_shares`.

    A *share of the geometry*, which is a different quantity from the read fraction the statistics
    half reports. The signature carries both rather than picking the flattering one.
    """
    from vdjtools.io.schema import strip_allele

    if "c_call" not in df.columns:
        return {}
    # Allele-stripped: the classes are gene names matched by equality, so ``IGHG1*01`` would match
    # nothing and the whole repertoire would come back uncalled -- a composition, not an error.
    calls = df.select(strip_allele(pl.col("c_call").cast(pl.Utf8)).alias("c"))["c"].to_list()
    out: dict[str, float] = {}
    claimed = 0.0
    for name, names in ISOTYPE_BANDS.items():
        m = np.array([c in names for c in calls], dtype=bool)
        if int(m.sum()) < min_clonotypes:
            continue
        out[name] = float(w[m].sum())
        claimed += out[name]
    out["_uncalled"] = max(1.0 - claimed, 0.0)
    return out


def _demo() -> None:
    """Self-check: the algebraic identities these functions rest on."""
    from mir.repertoire import rao_dispersion

    rng = np.random.default_rng(0)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    df = pl.DataFrame({
        "v_call": ["TRBV20-1"] * 200, "j_call": ["TRBJ2-2"] * 200, "c_call": [None] * 200,
        "junction_aa": ["C" + "".join(rng.choice(aa, 12)) + "F" for _ in range(200)],
        "duplicate_count": np.ceil(rng.zipf(1.5, 200).clip(1, 900)).astype(np.int64).tolist(),
    }, schema_overrides={"c_call": pl.Utf8})
    w = weights(df["duplicate_count"].to_numpy())

    # the weights close, and n_eff is a Hill number OF THEM
    assert abs(w.sum() - 1.0) < 1e-12
    n_eff = 1.0 / float(w @ w)
    assert 1.0 <= n_eff <= df.height

    # the mixture identity is exact: the shares of a partition close
    sh = band_shares(df, w, min_clonotypes=1)
    assert abs(sum(sh.values()) - 1.0) < 1e-12, sh

    # the slots are literal strides, so they reconstruct Phi
    phi = rng.normal(size=3 * K)
    s = slots(phi)
    assert np.array_equal(s["phiv"], phi[0::3]) and s["phic"].size == K

    # Rao telescopes out of the same chunked pass, and agrees with the independent accumulator
    # implementation in mir.repertoire -- two spellings of one identity, cross-checked rather than
    # trusted.
    from mir.embedding.tcremp import TCREmp
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=32)
    phi, mean_sq = prototype_sum(df, model, w)
    mine = rao_of(phi, mean_sq, n_eff)
    theirs = rao_dispersion(np.asarray(model.embed(df), dtype=float), w)
    assert abs(mine - float(theirs)) / max(mine, 1e-12) < 1e-9, (mine, theirs)
    print(f"features OK  n_eff={n_eff:.1f} rao={mine:.4f}")


if __name__ == "__main__":
    _demo()
