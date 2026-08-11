#!/usr/bin/env python
"""Build the fit-free half of the signature artifact. Uses **no samples at all**.

Everything this writes is derived from resources already bundled with the library — the
prototype panels and the recombination models — so it is reproducible by anyone who installs
mirpy, and re-running it on a different machine is not supposed to need anybody's cohort.

Per locus it emits:

``R_{V,J,C}``
    The rotation for each prototype slot: the top principal components of the **centred**
    prototype cloud, ``n_cloud`` bundled prototypes embedded against ``K`` of their own kind.
    Centred, because every prototype distance is large and positive and the leading component of
    the uncentred cloud is the constant everything shares.
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

**Reproducibility, measured, and one caveat.** Rebuilding under a different thread count leaves
``R_*``, ``mu_phi`` and ``sd_phi`` **bit-identical** — verified 1-thread against 8-thread, with
zero sign flips. ``naive`` is not bit-identical, because ``vdjtools.model.generate.generate``
does not reproduce across *processes* even with a fixed seed: it is deterministic within one
process, but two processes given ``seed=1`` draw different sequences, and neither
``PYTHONHASHSEED`` nor ``maintain_order=True`` on the grouped tables changes that (all its
randomness does go through the seeded generator, so the divergence is in the order of the
prepared probability tables). Filed as a vdjtools issue.

What that costs here is bounded and was measured rather than assumed: across independent draws
the naive vector moves by ~1e-4 relative, falling as ``1/sqrt(n)`` (max coordinate spread 3.11 at
n=5,000, 2.07 at 20,000, 1.25 at 80,000). Since the artifact **ships as a file**, every user gets
the identical vector and portability is unaffected; only a *rebuild* differs, and only at that
tolerance. ``verify()`` therefore checksums the fit-free arrays exactly and checks ``naive``
against a tolerance, rather than pretending to a bit-exactness that does not hold.

The corpus-fitted half — per-column location and scale — is *not* here; see
``fit_signature_reference.py``. Keeping them apart is what makes the claim auditable: the
geometry never sees a sample, so re-fitting the reference cannot move a coordinate.

Run:  python build_rsig.py [--out rsig_v1.npz] [--loci TRB,IGH] [--n-cloud 4000] [--n-naive 20000]
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

#: Prototypes embedded to define the rotation. Must exceed K comfortably so the cloud covariance
#: is well determined — 4000 against p=256 per slot is gamma ~ 0.06, far from the p>n regime that
#: makes a sample-fitted rotation unusable.
N_CLOUD = 4000

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


def build_locus(locus: str, *, species: str = "human", n_cloud: int = N_CLOUD,
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

    out: dict[str, np.ndarray] = {"mu_phi": mu, "sd_phi": sd}
    widths = {"V": PC_DIMS["phiv"]["full"], "J": PC_DIMS["phij"]["full"],
              "C": PC_DIMS["phic"]["full"]}
    for name, off in SLOTS.items():
        S = Zs[:, off::3]
        _, _, vt = np.linalg.svd(S - S.mean(0), full_matrices=False)
        out[f"R_{name}"] = _fix_signs(vt[: widths[name]]).T      # (p, k)

    # where an unselected repertoire sits, weighted uniformly: a generated multiset has no
    # abundances, and inventing some would put a shape into the reference that recombination
    # does not produce
    from vdjtools.model.bundled import load_bundled
    from vdjtools.model.generate import generate

    gen = generate(load_bundled(locus), n_naive, seed=seed, productive_only=True)
    gen = gen.filter(gen["junction_aa"].str.contains(r"^[ACDEFGHIKLMNPQRSTVWY]+$"))
    Zn = model.embed(gen).astype(np.float64)
    out["naive"] = Zn.mean(0)
    # The Monte Carlo error on that mean, so verify() can check a rebuild against a tolerance it
    # measured rather than one somebody guessed. Standard error of the mean per coordinate; the
    # observed between-draw spread runs a few multiples of its maximum.
    out["naive_sem"] = Zn.std(0, ddof=1) / np.sqrt(max(Zn.shape[0], 1))
    out["prototype_hash"] = np.array(prototype_hash(species, locus, K, 0))
    out["n_naive_used"] = np.array(Zn.shape[0])
    return out


def main(out_path: str, loci: list[str], n_cloud: int, n_naive: int, seed: int) -> int:
    t0 = time.perf_counter()
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {"K": K, "n_cloud": n_cloud, "n_naive": n_naive, "seed": seed,
                               "loci": loci, "signature_version": "RSIG-v1"}
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
    p.add_argument("--out", default=str(Path(__file__).parent / "rsig_v1.npz"))
    p.add_argument("--loci", default=",".join(LOCI))
    p.add_argument("--n-cloud", type=int, default=N_CLOUD)
    p.add_argument("--n-naive", type=int, default=N_NAIVE)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    raise SystemExit(main(a.out, [x.strip() for x in a.loci.split(",") if x.strip()],
                          a.n_cloud, a.n_naive, a.seed))
