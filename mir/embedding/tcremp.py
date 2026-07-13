"""TCREMP: distance-vector embeddings for TCR/BCR clonotypes.

TCREMP (T-Cell Receptor EMbedding with Prototypes) embeds each clonotype as a flat
vector of distances to a fixed set of *prototype* clonotypes. For clonotype *i* and
prototype *k*, three distances are emitted; the first two depend on the ``mode``:

* ``mode="vjcdr3"`` (default): **V-germline**, **J-germline**, **junction**.
* ``mode="cdr123"``: **CDR1**, **CDR2** (both germline V-gene-determined), **junction**.

The embedding vector (vjcdr3 shown) is::

    [v_i1, j_i1, junc_i1, v_i2, j_i2, junc_i2, ..., v_iK, j_iK, junc_iK]

for K prototypes; the output matrix is ``(n_clonotypes, 3*K)`` ``float32``.

Junction (CDR3) distances are computed with :func:`seqtree.gapblock.score_matrix`
(best of contiguous gap-block placements at ``(3, 4, -4, -3)``, BLOSUM62 Gram
penalty — a genuine metric, ``d(a,a)=0``). V/J and CDR1/CDR2 distances are looked up
from baked matrices (:class:`mir.distances.germline.GermlineDistances`). Input is a
polars DataFrame with the ``vdjtools.io.schema`` columns ``v_call``, ``j_call``,
``junction_aa``.

Typical usage::

    import polars as pl
    from mir.embedding.tcremp import TCREmp

    model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
    df = pl.DataFrame({"v_call": ["TRBV10-3*01"], "j_call": ["TRBJ2-7*01"],
                       "junction_aa": ["CASSIRSSYEQYF"]})
    X = model.embed(df)          # shape: (1, 3000)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from mir.aliases import normalize_locus_alias, normalize_species_alias
from mir.distances.germline import GermlineDistances, load_germline_distances
from mir.distances.junction import junction_distance_matrix
from mir.embedding.prototypes import N_PROTOTYPES, load_prototypes

MODES = ("vjcdr3", "cdr123")
DEFAULT_GAP_POSITIONS: tuple[int, ...] = (3, 4, -4, -3)
DEFAULT_N_PROTOTYPES = 3000

# mode -> ((component, query_gene_column, prototype_gene_attr), ...) for the two
# germline slots (0, 1). The junction is always slot 2.
_MODE_SPEC: dict[str, tuple[tuple[str, str, str], tuple[str, str, str]]] = {
    "vjcdr3": (("V", "v_call", "_proto_v"), ("J", "j_call", "_proto_j")),
    "cdr123": (("CDR1", "v_call", "_proto_v"), ("CDR2", "v_call", "_proto_v")),
}

_REQUIRED_COLS = ("v_call", "j_call", "junction_aa")


class TCREmp:
    """Prototype distance-vector embedding for one locus."""

    def __init__(
        self,
        species: str,
        locus: str,
        prototypes: pl.DataFrame,
        germline: GermlineDistances,
        mode: str = "vjcdr3",
        gap_positions: tuple[int, ...] = DEFAULT_GAP_POSITIONS,
        threads: int = 0,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        missing = set(_REQUIRED_COLS) - set(prototypes.columns)
        if missing:
            raise ValueError(f"prototypes missing columns: {sorted(missing)}")
        self.species = species
        self.locus = locus
        self.mode = mode
        self.threads = threads
        self._germline = germline
        self._proto_v = prototypes["v_call"].to_list()
        self._proto_j = prototypes["j_call"].to_list()
        self._proto_junction = prototypes["junction_aa"].to_list()
        self.n_prototypes = len(self._proto_junction)
        self._gap_positions = gap_positions
        # fail fast if the locus lacks a component this mode needs
        for comp, _, _ in _MODE_SPEC[mode]:
            if not germline.has(comp):
                raise ValueError(
                    f"germline distances for {locus} lack component {comp!r} "
                    f"required by mode {mode!r}"
                )

    @classmethod
    def from_defaults(
        cls,
        species: str,
        locus: str,
        n_prototypes: int = DEFAULT_N_PROTOTYPES,
        mode: str = "vjcdr3",
        **kwargs,
    ) -> "TCREmp":
        """Build from the bundled prototypes and baked germline distances."""
        species_c = normalize_species_alias(species)
        locus_c = normalize_locus_alias(locus)
        prototypes = load_prototypes(species_c, locus_c, n=n_prototypes)
        germline = load_germline_distances(species_c, locus_c)
        return cls(species_c, locus_c, prototypes, germline, mode=mode, **kwargs)

    @classmethod
    def from_file(
        cls,
        prototypes_path: str | Path,
        species: str,
        locus: str,
        mode: str = "vjcdr3",
        **kwargs,
    ) -> "TCREmp":
        """Build from a custom prototype TSV (``v_call``/``j_call``/``junction_aa``).

        Embeddings from a custom prototype set are **not** comparable to those from
        the bundled prototypes — the coordinate system differs.
        """
        species_c = normalize_species_alias(species)
        locus_c = normalize_locus_alias(locus)
        prototypes = pl.read_csv(prototypes_path, separator="\t", columns=list(_REQUIRED_COLS))
        germline = load_germline_distances(species_c, locus_c)
        return cls(species_c, locus_c, prototypes, germline, mode=mode, **kwargs)

    @property
    def n_features(self) -> int:
        return 3 * self.n_prototypes

    def _junction_distances(self, junctions: list[str]) -> np.ndarray:
        return junction_distance_matrix(
            junctions, self._proto_junction,
            gap_positions=self._gap_positions, threads=self.threads,
        )

    def embed(self, clonotypes: pl.DataFrame) -> np.ndarray:
        """Embed a clonotype frame into ``(n_clonotypes, 3 * n_prototypes)`` float32.

        Args:
            clonotypes: polars DataFrame with columns ``v_call``, ``j_call``,
                ``junction_aa`` (``vdjtools.io.schema`` names).

        Returns:
            ``float32`` array of shape ``(len(clonotypes), 3 * n_prototypes)``,
            interleaved per prototype as ``[slot0, slot1, junction]``.
        """
        missing = set(_REQUIRED_COLS) - set(clonotypes.columns)
        if missing:
            raise ValueError(f"clonotypes missing columns: {sorted(missing)}")

        n = clonotypes.height
        out = np.empty((n, 3 * self.n_prototypes), dtype=np.float32)

        for slot, (comp, gene_col, proto_attr) in enumerate(_MODE_SPEC[self.mode]):
            out[:, slot::3] = self._germline.matrix(
                comp, clonotypes[gene_col].to_list(), getattr(self, proto_attr)
            )
        out[:, 2::3] = self._junction_distances(clonotypes["junction_aa"].to_list())
        return out


class PairedTCREmp:
    """Paired-chain embedding: concatenate two per-chain :class:`TCREmp` embeddings."""

    def __init__(self, chains: dict[str, TCREmp]):
        # dict preserves insertion order; embedding = concat in that order
        self._chains = chains

    @classmethod
    def from_defaults(
        cls,
        species: str,
        loci: tuple[str, str] = ("TRA", "TRB"),
        n_prototypes: int = DEFAULT_N_PROTOTYPES,
        mode: str = "vjcdr3",
        **kwargs,
    ) -> "PairedTCREmp":
        chains = {
            locus: TCREmp.from_defaults(species, locus, n_prototypes, mode, **kwargs)
            for locus in loci
        }
        return cls(chains)

    @property
    def n_features(self) -> int:
        return sum(m.n_features for m in self._chains.values())

    def embed(self, frames: dict[str, pl.DataFrame]) -> np.ndarray:
        """Embed per-chain frames (keyed by locus, row-aligned) and concatenate.

        Args:
            frames: Mapping ``locus -> DataFrame``; must cover every locus in the
                model, all with the same row count (row *i* is the same clonotype).
        """
        missing = set(self._chains) - set(frames)
        if missing:
            raise ValueError(f"missing frames for loci: {sorted(missing)}")
        parts = [self._chains[locus].embed(frames[locus]) for locus in self._chains]
        heights = {p.shape[0] for p in parts}
        if len(heights) != 1:
            raise ValueError(f"per-chain frames must have equal row counts, got {heights}")
        return np.hstack(parts)


if __name__ == "__main__":
    df = pl.DataFrame(
        {
            "v_call": ["TRBV10-3*01", "TRBV20-1*01"],
            "j_call": ["TRBJ2-7*01", "TRBJ1-2*01"],
            "junction_aa": ["CASSIRSSYEQYF", "CSARVSGYYGYTF"],
        }
    )
    for mode in MODES:
        m = TCREmp.from_defaults("human", "TRB", n_prototypes=50, mode=mode)
        X = m.embed(df)
        assert X.shape == (2, 150), X.shape
        assert X.dtype == np.float32
        assert np.isfinite(X).all()
        print(f"  mode={mode}: X {X.shape} range [{X.min():.1f}, {X.max():.1f}]")
    print("mir.embedding.tcremp self-check OK")
