#!/usr/bin/env python
"""Build the fit-free half of the signature artifact. Uses **no samples at all**.

Everything this writes is derived from resources already bundled with the library — the
prototype panels and the recombination models — so it is reproducible by anyone who installs
mirpy, and re-running it on a different machine is not supposed to need anybody's cohort.

Per locus it emits:

``R_{V,J,C}``
    The rotation for each prototype slot: the top principal components of the **centred**
    prototype cloud — the *whole* bundled panel embedded against ``K`` of its own kind. Centred,
    because every prototype distance is large and positive and the leading component of the
    uncentred cloud is the constant everything shares. Whole rather than a draw, because a draw
    makes the rotation a random variable in ``n_cloud`` and the junction slot is measurably still
    moving at 4,000 (see ``N_CLOUD``).
``gap_{V,J,C}``
    Relative eigenvalue gap to the next component. Near zero, the pair spans a stable plane but
    the individual coordinates are exchangeable — fine for a model that spans the plane, not fine
    for a per-coordinate importance read-out.
``mu_phi`` / ``sd_phi``
    Location and scale of the cloud's own coordinates, used to standardise ``Φ`` before rotating.
    These are cloud statistics, not corpus statistics: they say what a *typical receptor* looks
    like, which is a fixed property of the panel.
``naive``
    ``Φ`` of a large synthetic repertoire drawn from the bundled recombination model — where an
    unselected repertoire sits. The ``contrast`` block measures deviation from this.

Sign convention: eigenvector signs are arbitrary and not reproducible across BLAS builds, so each
component is flipped to make its largest-magnitude coordinate positive. Deterministic given the
matrix, unlike a coordinate-sum rule, which flips whenever the sum sits near zero.

**Reproducibility: every array is bit-identical on rebuild**, verified 1-thread against
8-thread, with zero eigenvector sign flips.

That took fixing a real bug rather than tolerating it. ``naive`` was initially irreproducible,
because ``generate(model, n, seed=1)`` drew different sequences in every *process* — determinis-
tic within one, so a same-process round trip could not see it. The cause was upstream of the
generator: ``collapse_alleles`` aggregated the model's marginal tables with polars ``group_by``
calls that lacked ``maintain_order=True``, and polars leaves group order unspecified without it.
``tables["v_choice"]`` therefore came out in a different row order per process, the cumulative
distributions were built over those rows, and the same random stream selected different alleles.
Fixed in ``vdjtools.model.collapse``, with a cross-process regression test, so this file no
longer has to trade reproducibility for a tolerance.

The corpus-fitted half — per-column location and scale — is *not* here; see
``fit_signature_reference.py``. Keeping them apart is what makes the claim auditable: the
geometry never sees a sample, so re-fitting the reference cannot move a coordinate.

Run:  python build_rsig.py [--out rsig_v2.npz] [--loci TRB,IGH] [--n-cloud 0] [--n-naive 20000]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Prototype panel size the signature is defined against. Deliberately smaller than the library
#: preset (2000 for the diverse chains): the identity block keeps at most 48 coordinates per
#: slot, and 256 already exceeds the numerical rank of the germline slots.
K = 256

#: Prototypes embedded to define the rotation. ``None`` means **the whole bundled panel** (10,000
#: for the diverse chains), and that is deliberate: with a draw, the rotation is a random variable
#: in ``n_cloud``, and the junction slot's trailing coordinates are measurably still moving at
#: 4,000. Against the whole panel, components matched by ``|cos|``: V 23/24 stable at n=4,000 and
#: 24/24 at 5,000, J 12/12 throughout — but C only 5/48 at 4,000 and 14/48 at 5,000, with the
#: unstable ones exactly the near-degenerate pairs (relative eigenvalue gap < 2% at C15, C35, C38,
#: C41-43, C46), which *swap* rather than drift. Taking the whole panel removes the draw: the
#: rotation becomes a deterministic function of a fixed bundled resource, so there is nothing left
#: to converge to. It costs nothing — retention on two real cohorts is unchanged (V 0.9988/0.9983,
#: C 0.9873/0.9898 against 0.9988/0.9981, 0.9870/0.9899 at 4,000) and the embedding is instant.
N_CLOUD = None

#: Synthetic recombinations behind the naive reference.
N_NAIVE = 20_000

SEED = 20260811

#: Full-tier width per slot, from vdjtools.signature.layout.PC_DIMS. Stored at full width; the
#: narrower tiers are prefixes.
SLOTS = {"V": 0, "J": 1, "C": 2}


def _fix_signs(V: np.ndarray) -> np.ndarray:
    """Flip each component so its largest-magnitude coordinate is positive.

    Eigenvector signs are arbitrary and differ across LAPACK implementations and thread counts.
    Left alone, regenerating the artifact would invert whole coordinates and silently break every
    coefficient a collaborator had trained. Keying on the largest-magnitude entry is stable; a
    coordinate-sum rule is not, because the sum can sit arbitrarily close to zero.
    """
    idx = np.argmax(np.abs(V), axis=1)
    signs = np.sign(V[np.arange(V.shape[0]), idx])
    signs[signs == 0] = 1.0
    return V * signs[:, None]


def build_locus(locus: str, *, species: str = "human", n_cloud: int | None = N_CLOUD,
                n_naive: int = N_NAIVE, seed: int = SEED) -> dict:
    """The fit-free arrays for one locus."""
    from mir.embedding.prototypes import load_prototypes
    from mir.embedding.tcremp import TCREmp
    from mir.ml.bundle import prototype_hash
    from vdjtools.signature.layout import PC_DIMS

    model = TCREmp.from_defaults(species, locus, n_prototypes=K)
    cloud = model.embed(load_prototypes(species, locus, n_cloud)).astype(np.float64)

    mu = cloud.mean(0)
    sd = cloud.std(0)
    sd[sd <= 0] = 1.0                     # a constant coordinate carries nothing; do not divide
    Zs = (cloud - mu) / sd

    out: dict[str, np.ndarray] = {"mu_phi": mu, "sd_phi": sd, "n_cloud": np.array(cloud.shape[0])}
    widths = {"V": PC_DIMS["phiv"]["full"], "J": PC_DIMS["phij"]["full"],
              "C": PC_DIMS["phic"]["full"]}
    for name, off in SLOTS.items():
        S = Zs[:, off::3]
        _, sv, vt = np.linalg.svd(S - S.mean(0), full_matrices=False)
        out[f"R_{name}"] = _fix_signs(vt[: widths[name]]).T      # (p, k)
        # Relative eigenvalue gap to the next component. Where it is near zero the two components
        # span a stable plane but which of them is which is determined by nothing — a rebuild
        # under a different BLAS or a different panel size exchanges them. A linear model spanning
        # the plane is unaffected; a per-coordinate importance read-out is not, so the number ships
        # rather than being left for someone to rediscover.
        ev = sv ** 2 / (sv ** 2).sum()
        gap = np.diff(ev) / np.maximum(ev[:-1], 1e-300)
        out[f"gap_{name}"] = np.abs(gap[: widths[name]])

    # where an unselected repertoire sits, weighted uniformly: a generated multiset has no
    # abundances, and inventing some would put a shape into the reference that recombination
    # does not produce
    from vdjtools.model.bundled import load_bundled
    from vdjtools.model.generate import generate

    gen = generate(load_bundled(locus), n_naive, seed=seed, productive_only=True)
    gen = gen.filter(gen["junction_aa"].str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$"))
    Zn = model.embed(gen).astype(np.float64)
    out["naive"] = Zn.mean(0)
    # Standard error of that mean. The rebuild is bit-identical, so this is not a reproducibility
    # tolerance — it is how tightly n_naive draws pin the reference, which is what says whether
    # a contrast is a deviation from the naive repertoire or from sampling noise in the estimate.
    out["naive_sem"] = Zn.std(0, ddof=1) / np.sqrt(max(Zn.shape[0], 1))
    out["prototype_hash"] = np.array(prototype_hash(species, locus, K, 0))
    out["n_naive_used"] = np.array(Zn.shape[0])
    return out


def main(out_path: str, loci: list[str], n_cloud: int | None, n_naive: int, seed: int) -> int:
    import vdjtools

    t0 = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    # The vdjtools version is provenance, not decoration: `naive` is drawn from that release's
    # bundled recombination models, and retraining them moves it (0.1-1.6% per locus between 3.2
    # and 3.6). Nothing else in this file depends on it — the rotations are bit-identical across
    # the same change — but a reference whose origin cannot be named is not a reference.
    meta: dict[str, object] = {"K": K, "n_cloud": n_cloud or "all", "n_naive": n_naive,
                               "seed": seed, "loci": loci, "signature_version": "RSIG-v2",
                               "vdjtools_version": vdjtools.__version__}
    for locus in loci:
        t = time.perf_counter()
        try:
            d = build_locus(locus, n_cloud=n_cloud, n_naive=n_naive, seed=seed)
        except Exception as e:
            print(f"  {locus}: SKIP ({type(e).__name__}: {e})")
            continue
        for k, v in d.items():
            arrays[f"{locus}/{k}"] = v
        meta[f"{locus}/prototype_hash"] = str(d["prototype_hash"])
        print(f"  {locus}: R_V{d['R_V'].shape} R_J{d['R_J'].shape} R_C{d['R_C'].shape} "
              f"naive from {int(d['n_naive_used']):,} ({time.perf_counter() - t:.0f}s)")

    out = Path(out_path)
    np.savez_compressed(out, **arrays)
    meta["bytes"] = out.stat().st_size
    Path(out.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size / 1024:.0f} KB) + {out.with_suffix('.json').name}"
          f" — {len([k for k in arrays if k.endswith('/naive')])} loci, "
          f"{time.perf_counter() - t0:.0f}s")
    print("fit-free: no sample was read to produce any of this.")
    return 0


if __name__ == "__main__":
    from vdjtools.signature.layout import LOCI

    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(Path(__file__).parent / "rsig_v2.npz"))
    p.add_argument("--loci", default=",".join(LOCI))
    p.add_argument("--n-cloud", type=int, default=N_CLOUD,
                   help="0 or omitted = the whole bundled panel")
    p.add_argument("--n-naive", type=int, default=N_NAIVE)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    raise SystemExit(main(a.out, [x.strip() for x in a.loci.split(",") if x.strip()],
                          a.n_cloud or None, a.n_naive, a.seed))
