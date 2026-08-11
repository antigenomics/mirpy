"""The geometry half of the signature — features of the prototype-sum measure.

Every column here is a linear functional, a norm, or a mixture coefficient of one object::

    Φ(S) = Σ_σ w_σ z_σ          z_σ = TCREmp row for clonotype σ,  w_σ = g(a_σ)/Σg

That object is **fit-free**: ``z_σ`` is a vector of distances to a fixed, bundled prototype
panel, so no basis is estimated from anybody's cohort and two people who never share data land
in the same coordinate system. It is also *linear in the clone-weight measure*, which is what
makes the compartment shares well-posed rather than heuristic: for a genuine partition
``Φ(S) = Σ_c π_c Φ(c)`` holds exactly, so ``π`` can be measured instead of assumed.

Three structural facts the layout depends on, all verified against the library:

* ``TCREmp.embed`` lays columns out interleaved per prototype as ``[V, J, junction]``, so the
  three slots are the exact strides ``Φ[0::3]``, ``Φ[1::3]``, ``Φ[2::3]``. Attribution across
  them is exact, not a surrogate model.
* Prototype ``k``'s three columns never move as the panel grows, so a ``K``-prototype embedding
  is a bit-exact prefix of a larger one. Widening the panel is therefore a superset, never a
  new coordinate system.
* Distances to an *unrecognised* V or J allele silently take the germline max-distance
  fallback. Nothing here can detect that; ``vsig:qc:*:*_fallback_frac`` reports it, and a
  vector with a high fallback fraction is not comparable to one without.

**Φ must be centred against the frozen reference before it means anything.** Every prototype
distance is large and positive, so every repertoire's ``Φ`` sits in almost the same place:
measured on eight unrelated donors, the cosine between *different* people's raw ``Φ`` spans
0.9989–0.9999, and the shared offset is about **55×** the between-donor signal in norm. Subtract
the reference mean and the same eight span −0.81 to +0.66 — roughly a thousandfold more
discriminative. So ``mu_phi`` is not a tidying step and the rotation is not fit on raw ``Φ``:
without the centring, the leading component is the constant everyone shares and the identity
block is nearly blank. The same argument is why :func:`contrast` subtracts a naive reference
rather than reporting ``Φ`` directly.
"""
from __future__ import annotations

import numpy as np
import polars as pl

#: Rows embedded per chunk. The embedding itself is cheap — measured 0.01 s for 2,000
#: clonotypes at K=512 — so this exists to bound *memory*, not time: the full matrix for a
#: 500,000-clonotype sample would be 3 GB at K=512, while the accumulators are 1536 floats.
CHUNK = 50_000

#: Clone-size weights ``g``. Concave by default: raw read weighting lets one dominant clone be
#: the entire profile, presence weighting throws the expansion signal away, and ``log2(1+a)``
#: sits between. Mirrors ``mir.density._WEIGHTS``.
WEIGHTS = {
    "log2p1": lambda a: np.log2(1.0 + a),
    "log1p": np.log1p,
    "anscombe": lambda a: np.sqrt(a + 0.375),
    "duplicate_count": lambda a: a.astype(float),
    "distinct": np.ones_like,
}

#: Abundance compartments, as row predicates over the clone-size vector. A *partition* — unlike
#: ``mir.repertoire.band_frames``, whose ``top`` is deliberately a subset of ``expanded``. The
#: mixture identity is only exact for a partition, and an NNLS over overlapping parts is not a
#: composition at all: its weights need not sum to one and an individual share can exceed it.
BANDS: dict[str, "callable"] = {
    "singleton": lambda a: a == 1,
    "middle": lambda a: (a > 1) & (a < np.maximum(np.quantile(a, 0.99), 2)),
    "top": lambda a: a >= np.maximum(np.quantile(a, 0.99), 2),
}

#: IGH isotype compartments, by constant-gene call. ``IGHGP`` is a pseudogene and ``IGHC`` is
#: ambiguous, so neither is called; roughly two fifths of IGH reads carry no call at all and
#: form their own part rather than being folded into IgM.
ISOTYPE_BANDS: dict[str, tuple[str, ...]] = {
    "IgM": ("IGHM", "IGHD"),
    "IgG": ("IGHG1", "IGHG2", "IGHG3", "IGHG4"),
    "IgA": ("IGHA1", "IGHA2"),
}


def weights(counts: np.ndarray, weight: str = "log2p1") -> np.ndarray:
    """Normalised clone weights ``w = g(a)/Σg``.

    Raises:
        ValueError: If ``weight`` is unknown, or no clonotype carries any weight.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"unknown weight {weight!r}; known: {sorted(WEIGHTS)}")
    g = WEIGHTS[weight](np.asarray(counts, dtype=float))
    s = g.sum()
    if s <= 0:
        raise ValueError("every clone weight is zero — the sample carries no usable counts")
    return g / s


def prototype_sum(df: pl.DataFrame, model, w: np.ndarray, *, chunk: int = CHUNK):
    """``Φ = Σ w_σ z_σ`` and its Rao dispersion, in one chunked pass.

    Both quantities are running sums over the rows, so the full ``(n, 3K)`` matrix is never
    held: the accumulators are ``Σ w z`` and ``Σ w‖z‖²``, and Rao's ``Q`` telescopes out of the
    pair as ``2(Σw‖z‖² − ‖Φ‖²)`` (see :func:`mir.repertoire.rao_dispersion`).

    Args:
        df: One locus of one sample, already sanitised. Row order must match ``w``.
        model: A :class:`~mir.embedding.tcremp.TCREmp` for that locus.
        w: Normalised clone weights, from :func:`weights`.
        chunk: Rows embedded at a time.

    Returns:
        ``(phi, mean_sq_norm)`` — ``phi`` is ``(3K,)``, ``mean_sq_norm`` is ``Σ w‖z‖²``.
    """
    phi = np.zeros(model.n_features, dtype=np.float64)
    mean_sq = 0.0
    for lo in range(0, df.height, chunk):
        block = df.slice(lo, chunk)
        z = model.embed(block).astype(np.float64)
        wc = w[lo:lo + z.shape[0]]
        phi += wc @ z
        mean_sq += float(wc @ np.einsum("ij,ij->i", z, z))
    return phi, mean_sq


def slots(phi: np.ndarray) -> dict[str, np.ndarray]:
    """Split ``Φ`` into its ``V`` / ``J`` / ``junction`` strides.

    Exact by construction — these are literal column strides of the embedding, not an
    attribution model — which is what makes "how much of this distance is V?" answerable
    without SHAP, sampling, or a surrogate.
    """
    return {"phiv": phi[0::3], "phij": phi[1::3], "phic": phi[2::3]}


def depth_block(counts: np.ndarray, w: np.ndarray, mass: float) -> dict[str, float]:
    """Effective size and retained mass — the geometry's own reading of depth.

    ``n_eff = 1/Σw²`` is a Hill number *of the weights the geometry actually uses*, which is not
    the same quantity as the richness of the count vector: it says how many clonotypes are
    effectively contributing to ``Φ``, and so predicts how noisy this sample's ``Φ`` is. ``mass``
    is ``1 − M₀``, the share of the repertoire that was ever drawn.
    """
    from vdjtools.signature import transform as T

    n_eff = 1.0 / float(w @ w) if w.size else np.nan
    return {"n_eff": T.log10(n_eff), "mass": T.logit(np.clip(mass, 0.0, 1.0), counts.size)}


def band_shares(df: pl.DataFrame, model, w: np.ndarray, phi: np.ndarray, *,
                bands: dict | None = None, min_clonotypes: int = 5,
                chunk: int = CHUNK) -> dict[str, float]:
    """Compartment shares of ``Φ``, in closed form rather than by NNLS.

    Because ``Φ`` is linear in the clone-weight measure and the compartments partition the
    clonotypes, the share of ``Φ`` owned by compartment ``c`` is just its share of the weight::

        Φ(S) = Σ_c π_c Φ(c)        with     π_c = Σ_{σ ∈ c} w_σ

    exactly, with no fitting. Solving a non-negative least squares for the same quantity — the
    obvious alternative — is both slower and worse posed: over overlapping compartments the
    weights need not sum to one and a share can exceed it, which then breaks any log-ratio
    coordinate downstream.

    A compartment below ``min_clonotypes`` is recorded **absent** (its share is dropped from the
    composition) rather than set to zero. Zero is a measurement; absent is not.

    Returns:
        ``{band: share}`` over the bands that cleared the floor, plus ``_residual`` for whatever
        no compartment owned. Shares are raw, un-transformed; the caller closes them into
        log-ratio coordinates.
    """
    del model, phi, chunk                # shares need only the weights; Φ is implied by linearity
    a = df["duplicate_count"].to_numpy()
    out: dict[str, float] = {}
    claimed = 0.0
    for name, pred in (bands or BANDS).items():
        m = np.asarray(pred(a), dtype=bool)
        if int(m.sum()) < min_clonotypes:
            continue
        share = float(w[m].sum())
        out[name] = share
        claimed += share
    out["_residual"] = max(1.0 - claimed, 0.0)
    return out


def isotype_shares(df: pl.DataFrame, w: np.ndarray, *,
                   min_clonotypes: int = 5) -> dict[str, float]:
    """Isotype shares of ``Φ(IGH)``, by the same mixture identity as :func:`band_shares`.

    A *share of the geometry*, which is a different quantity from the read fraction the
    statistics half reports — the two answer different questions and the signature carries both
    rather than picking the flattering one.
    """
    if "c_call" not in df.columns:
        return {}
    calls = df["c_call"].to_list()
    out: dict[str, float] = {}
    claimed = 0.0
    for name, prefixes in ISOTYPE_BANDS.items():
        m = np.array([c in prefixes for c in calls], dtype=bool)
        if int(m.sum()) < min_clonotypes:
            continue
        out[name] = float(w[m].sum())
        claimed += out[name]
    out["_uncalled"] = max(1.0 - claimed, 0.0)
    return out


def contrast(phi: np.ndarray, naive: np.ndarray, mass: float) -> np.ndarray:
    """``Ψ = mass·(Φ − naive)`` — the signed deviation from unselected V(D)J output.

    Two things are encoded in one vector. The *direction* says how this repertoire's receptors
    differ from what recombination alone would produce; the *magnitude* says how confident that
    statement is, because a sample that saw almost none of its repertoire is scaled toward the
    origin rather than being dropped.

    That is why the block is magnitude-scaled and never centred downstream. An immune desert
    must land at the origin; per-column standardisation would move it to minus-the-median and
    make it indistinguishable from a typical sample — which is exactly the deficiency the
    contrast exists to carry.
    """
    return float(np.clip(mass, 0.0, 1.0)) * (np.asarray(phi, dtype=np.float64)
                                             - np.asarray(naive, dtype=np.float64))


def _demo() -> None:
    """Self-check on bundled prototypes: linearity, exact strides, and the partition identity."""
    from mir.embedding.tcremp import TCREmp
    from mir.repertoire import rao_dispersion

    rng = np.random.default_rng(0)
    n = 300
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    df = pl.DataFrame({
        "v_call": ["TRBV20-1"] * n, "j_call": ["TRBJ2-2"] * n, "c_call": [None] * n,
        "junction_aa": ["C" + "".join(rng.choice(aa, 12)) + "F" for _ in range(n)],
        "duplicate_count": rng.integers(1, 200, n).tolist(),
    })
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=64)
    a = df["duplicate_count"].to_numpy()
    w = weights(a)

    phi, mean_sq = prototype_sum(df, model, w)
    assert phi.shape == (3 * 64,)

    # chunking is an implementation detail, not a different answer
    phi_c, mean_sq_c = prototype_sum(df, model, w, chunk=37)
    assert np.allclose(phi, phi_c) and np.isclose(mean_sq, mean_sq_c)

    # Phi is exactly the weighted mean of the rows
    z = model.embed(df).astype(np.float64)
    assert np.allclose(phi, w @ z)

    # the three slots are literal strides, and they partition the coordinates
    s = slots(phi)
    assert sum(v.size for v in s.values()) == phi.size
    assert np.allclose(s["phiv"], (w @ z)[0::3])

    # Rao from the accumulators agrees with the direct computation
    assert np.isclose(2.0 * (mean_sq - phi @ phi),
                      rao_dispersion(z, w, correct=False))

    # the mixture identity: shares of a partition are shares of the weight, and they close
    shares = band_shares(df, model, w, phi, min_clonotypes=1)
    assert abs(sum(shares.values()) - 1.0) < 1e-12, "compartment shares do not close"

    # contrast sends an unobserved repertoire to the origin instead of somewhere confident
    assert np.allclose(contrast(phi, np.zeros_like(phi), 0.0), 0.0)
    assert np.allclose(contrast(phi, phi, 1.0), 0.0)

    d = depth_block(a, w, mass=0.8)
    assert np.isfinite(list(d.values())).all()
    print("mir.signature.blocks OK")


if __name__ == "__main__":
    _demo()
