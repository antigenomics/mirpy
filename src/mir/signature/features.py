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

import numpy as np
import polars as pl
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
