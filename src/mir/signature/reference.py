"""The frozen reference: load it, check it is the one you think, and project through it.

A signature is only portable if everyone projects through the *same* basis. This module owns
that basis — the artifact built by ``mir/resources/signature/build_rsig.py`` — and the one
operation that turns a raw prototype-sum into signature coordinates::

    Φ  →  (Φ − naive)/sd_phi  →  split into V/J/junction slots  →  rotate by R_*

The centring is not cosmetic. Every prototype distance is large and positive, so all repertoires
sit in nearly the same place: across unrelated donors the raw between-donor cosine spans about
0.001, while the shared offset is ~55× the between-donor signal. Rotate without subtracting a
centre and the leading component is the constant everyone shares, so the identity block comes out
nearly blank. Centred, the same donors span 1.48. The centre is ``naive`` rather than the
prototype cloud's own ``mu_phi``, which is a measured distinction — see
:meth:`LocusReference.standardize`.

**Comparability is checked, not assumed.** The artifact records the prototype hash it was built
against, and :meth:`SignatureReference.verify` refuses a mismatch rather than silently producing
numbers in a different coordinate system that look perfectly reasonable. That is the same
contract ``RepertoireSpace`` and ``CodecBundle`` enforce.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Where the bundled artifact lives. Alongside the germline distance matrices and the prototype
#: panels, since it is the same kind of object: a versioned, baked resource.
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "resources" / "signature" / "rsig_v2.npz"

#: Slot name in the artifact -> the layout block it feeds, and its stride in ``Φ``.
SLOTS: dict[str, tuple[str, int]] = {"V": ("phiv", 0), "J": ("phij", 1), "C": ("phic", 2)}


@dataclass(frozen=True)
class LocusReference:
    """The frozen basis for one locus."""

    mu_phi: np.ndarray                 # (3K,) prototype-cloud mean
    sd_phi: np.ndarray                 # (3K,) prototype-cloud scale, never zero
    naive: np.ndarray                  # (3K,) Φ of an unselected repertoire
    naive_sem: np.ndarray              # (3K,) standard error of that mean
    rotations: dict[str, np.ndarray]   # slot -> (p, k)
    prototype_hash: str
    #: slot -> (k,) relative eigenvalue gap to the next component. See :meth:`exchangeable`.
    gaps: dict[str, np.ndarray] | None = None

    @property
    def n_prototypes(self) -> int:
        return self.mu_phi.size // 3

    def exchangeable(self, slot: str, *, tol: float = 0.02) -> np.ndarray:
        """Indices of components whose neighbour sits within ``tol`` relative eigenvalue.

        Such a pair spans a well-determined plane, but *which* of the two is the earlier
        coordinate is determined by nothing: rebuild the artifact against a different panel size
        or a different LAPACK and they exchange. Measured on the junction slot, components
        matched by ``|cos|`` between a 5,000-prototype rotation and the whole-panel one drop to
        0.01 exactly at the near-degenerate pairs while their neighbours stay above 0.95.

        The shipped artifact is frozen, so nothing exchanges in practice. It matters for what a
        *coordinate* is allowed to mean: a linear model spanning the plane is unaffected, a
        per-coordinate feature-importance read-out on one of the pair is not interpretable.

        Returns an empty array if the artifact predates the stored gaps.
        """
        if not self.gaps or slot not in self.gaps:
            return np.empty(0, dtype=int)
        return np.flatnonzero(self.gaps[slot] < tol)

    def standardize(self, phi: np.ndarray) -> np.ndarray:
        """``(Φ − naive)/sd_phi`` — centre on an unselected repertoire, then scale.

        The centre is ``naive``, **not** ``mu_phi``, and the difference is large. Both are
        fit-free, but they are means over different things: ``naive`` is a sample-level ``Φ``
        (an unselected repertoire, weighted over its clonotypes), while ``mu_phi`` averages the
        *prototype panel*, which is a differently-constituted set. Real repertoires live near the
        former. Measured on two cohorts, between-donor cosine spread after centring:

        ============================  ========  ========
        centre                        cohort A  cohort B
        ============================  ========  ========
        none (raw)                    0.0105    0.0010
        ``mu_phi`` (prototype cloud)  0.4008    0.0785
        ``naive`` (unselected)        1.2660    1.6148
        own sample mean (oracle)      1.9031    1.9088
        ============================  ========  ========

        ``naive`` recovers 67–85% of what an oracle centred on the cohort's own mean achieves;
        ``mu_phi`` recovers 4–21%. The residual offset explains it: ``‖mu_phi − sample mean‖`` is
        5–7× the between-donor spread, ``‖naive − sample mean‖`` well under it.

        ``mu_phi`` is still shipped, because it is the centre the rotation was fitted against and
        is needed to reproduce the artifact — it is simply not the right origin for a sample.
        """
        return (np.asarray(phi, dtype=np.float64) - self.naive) / self.sd_phi

    def project(self, phi: np.ndarray, slot: str, k: int | None = None) -> np.ndarray:
        """Standardise ``Φ``, take one slot's stride, and rotate into the frozen basis.

        Args:
            phi: One sample's raw ``(3K,)`` prototype-sum.
            slot: ``"V"``, ``"J"`` or ``"C"``.
            k: Keep the leading ``k`` components; ``None`` keeps every stored one. The narrower
                tiers are prefixes, so this is a slice rather than a different projection.

        Raises:
            ValueError: If ``slot`` is unknown, ``phi`` has the wrong width, or ``k`` exceeds
                what the artifact stores.
        """
        if slot not in SLOTS:
            raise ValueError(f"slot must be one of {sorted(SLOTS)}; got {slot!r}")
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape != self.mu_phi.shape:
            raise ValueError(
                f"Φ has width {phi.shape} but this reference was built for {self.mu_phi.shape} "
                f"({self.n_prototypes} prototypes) — embed with n_prototypes="
                f"{self.n_prototypes} or load a matching reference")
        R = self.rotations[slot]
        if k is not None and k > R.shape[1]:
            raise ValueError(f"slot {slot} stores {R.shape[1]} components; {k} were requested")
        z = self.standardize(phi)[SLOTS[slot][1]::3]
        out = z @ R
        return out[:k] if k is not None else out

    def contrast(self, phi: np.ndarray, mass: float) -> np.ndarray:
        """``Ψ = mass·(Φ − naive)`` in raw prototype coordinates.

        Deliberately *not* standardised by ``sd_phi``: this block carries its meaning in its
        magnitude, and a per-coordinate rescale would make a sample that deviates barely at all
        look like one that deviates a lot.
        """
        return float(np.clip(mass, 0.0, 1.0)) * (np.asarray(phi, dtype=np.float64) - self.naive)


@dataclass(frozen=True)
class SignatureReference:
    """The frozen bases for every locus the artifact covers."""

    loci: dict[str, LocusReference]
    meta: dict
    path: Path

    @property
    def version(self) -> str:
        return str(self.meta.get("signature_version", "unknown"))

    def __contains__(self, locus: str) -> bool:
        return locus in self.loci

    def __getitem__(self, locus: str) -> LocusReference:
        try:
            return self.loci[locus]
        except KeyError:
            raise KeyError(
                f"no frozen reference for locus {locus!r}; the artifact covers "
                f"{sorted(self.loci)}") from None

    def verify(self, *, species: str = "human") -> dict:
        """Check each locus's basis against the prototype panel currently installed.

        Returns:
            ``{locus: True}`` when every hash matches.

        Raises:
            ValueError: On the first mismatch. Loud, because the failure it prevents is silent —
                a mismatched panel still yields a full, plausible vector, in coordinates nobody
                else shares.
        """
        from mir.ml.bundle import prototype_hash

        out = {}
        for locus, ref in self.loci.items():
            current = prototype_hash(species, locus, ref.n_prototypes, 0)
            if current != ref.prototype_hash:
                raise ValueError(
                    f"prototype hash mismatch for {species} {locus}: the reference was built "
                    f"against {ref.prototype_hash} but the installed panel is {current}. Any "
                    "signature computed now would be in a different coordinate system while "
                    "looking entirely reasonable. Rebuild the artifact or reinstall the panel.")
            out[locus] = True
        return out


@lru_cache(maxsize=4)
def load_reference(path: "str | Path | None" = None) -> SignatureReference:
    """Load (and cache) the frozen reference.

    Raises:
        FileNotFoundError: If the artifact is missing — with the command that rebuilds it, since
            it is derived entirely from bundled resources and needs no data to regenerate.
    """
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"no signature reference at {p}. It is built from bundled resources only "
            f"(no cohort needed): python {p.parent / 'build_rsig.py'}")
    d = np.load(p, allow_pickle=False)
    meta_path = p.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    loci: dict[str, LocusReference] = {}
    for locus in sorted({k.split("/")[0] for k in d.files}):
        loci[locus] = LocusReference(
            mu_phi=d[f"{locus}/mu_phi"], sd_phi=d[f"{locus}/sd_phi"],
            naive=d[f"{locus}/naive"], naive_sem=d[f"{locus}/naive_sem"],
            rotations={s: d[f"{locus}/R_{s}"] for s in SLOTS},
            gaps=({s: d[f"{locus}/gap_{s}"] for s in SLOTS}
                  if f"{locus}/gap_V" in d.files else None),
            prototype_hash=str(d[f"{locus}/prototype_hash"]),
        )
    return SignatureReference(loci=loci, meta=meta, path=p)


def self_test(path: "str | Path | None" = None) -> dict:
    """Check the installed reference end to end, and return what it found.

    Verifies the prototype hashes, then embeds a fixed synthetic repertoire and confirms the
    projection is finite, the right width, and — the property that matters — that centring
    actually buys discrimination. Cheap enough to run on import in a notebook.
    """
    import polars as pl

    from mir.embedding.tcremp import TCREmp
    from mir.signature.blocks import prototype_sum, weights

    ref = load_reference(path)
    ref.verify()

    # Donors that genuinely differ. Drawing them all from one generator would make the centring
    # check vacuous: near-identical repertoires have no between-donor structure to recover, so
    # the assertion would pass on an arbitrarily broken reference. Each donor here gets its own
    # V gene and its own residue bias, which is real composition signal for centring to expose.
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    v_genes = ["TRBV20-1", "TRBV5-1", "TRBV19", "TRBV28", "TRBV7-9", "TRBV6-5"]
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=ref["TRB"].n_prototypes)
    phis = []
    for seed, v_gene in enumerate(v_genes):
        r = np.random.default_rng(seed)
        n = 300
        bias = np.full(20, 1.0)
        bias[seed % 20] = 8.0                     # one residue enriched, differently per donor
        p = bias / bias.sum()
        df = pl.DataFrame({
            "v_call": [v_gene] * n, "j_call": ["TRBJ2-2"] * n,
            "junction_aa": ["C" + "".join(r.choice(aa, 12, p=p)) + "F" for _ in range(n)],
            "duplicate_count": r.integers(1, 100, n).tolist(),
        })
        phi, _ = prototype_sum(df, model, weights(df["duplicate_count"].to_numpy()))
        phis.append(phi)

    P = np.array(phis)
    proj = np.array([ref["TRB"].project(p, "C", k=16) for p in P])
    assert np.isfinite(proj).all(), "projection produced non-finite coordinates"
    assert proj.shape[1] == 16

    def spread(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        c = Xn @ Xn.T
        iu = np.triu_indices(len(X), 1)
        return float(c[iu].max() - c[iu].min())

    raw, centred = spread(P), spread(P - ref["TRB"].naive)
    # A sanity floor, not the evidence for the centring choice. These synthetic donors differ by
    # whole V genes, which swamps the shared offset the centring exists to remove — measured, they
    # cannot tell `naive` from `mu_phi` at all (ratio 1.00). The claim that `naive` is the right
    # origin is a real-cohort result and lives in the benchmark record; what this checks is only
    # that the shipped reference is in the same coordinate system as the installed panel, which a
    # mismatch would break outright.
    assert centred > 5 * raw, (
        f"centring bought only {centred / raw:.1f}x separation — the naive reference does not "
        "appear to match this prototype panel")

    return {"version": ref.version, "loci": sorted(ref.loci),
            "n_prototypes": ref["TRB"].n_prototypes, "path": str(ref.path),
            "raw_cosine_spread": round(raw, 6), "centred_cosine_spread": round(centred, 4),
            "centring_gain": round(centred / raw, 1)}


def _demo() -> None:
    """Self-check: the artifact loads, verifies, projects, and refuses a mismatched width."""
    ref = load_reference()
    print(f"reference {ref.version} — {len(ref.loci)} loci, K={ref['TRB'].n_prototypes}")
    print(f"  verify: {len(ref.verify())} loci hash-matched")

    for slot, (block, _) in SLOTS.items():
        k = ref["TRB"].rotations[slot].shape[1]
        print(f"  {block:5s} slot {slot}: rotation {ref['TRB'].rotations[slot].shape} -> {k} coords")

    phi = np.zeros(3 * ref["TRB"].n_prototypes)
    assert ref["TRB"].project(phi, "C", k=8).shape == (8,)
    try:
        ref["TRB"].project(np.zeros(10), "C")
        raise AssertionError("accepted a Φ of the wrong width")
    except ValueError:
        pass
    try:
        ref["TRB"].project(phi, "C", k=999)
        raise AssertionError("accepted more components than are stored")
    except ValueError:
        pass
    assert np.allclose(ref["TRB"].contrast(ref["TRB"].naive, 1.0), 0.0)
    assert np.allclose(ref["TRB"].contrast(phi, 0.0), 0.0)

    print(f"  self_test: {self_test()}")


if __name__ == "__main__":
    _demo()
