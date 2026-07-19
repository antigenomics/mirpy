"""Bake per-locus germline distance matrices for TCREMP embeddings.

Build-time only (needs the ``[build]`` extra: BioPython). Reads
``mir/resources/gene_library/region_annotations.txt`` and, for every
``(species, locus)``, computes pairwise germline distances for four components:

* ``V``    — full V segment AA (``fwr1+cdr1+fwr2+cdr2+fwr3``), used by ``vjcdr3`` mode.
* ``J``    — full J segment AA (``jcdr3+fwr4``), used by ``vjcdr3`` mode.
* ``CDR1`` — germline V-gene CDR1 AA, used by ``cdr123`` mode.
* ``CDR2`` — germline V-gene CDR2 AA, used by ``cdr123`` mode.

The distance is ``d(a,b) = s(a,a) + s(b,b) − 2·s(a,b)`` where ``s`` is a global
BLOSUM62 alignment score (BioPython ``blastp`` defaults: gap open −12, extend −1).
Output is one compressed ``.npz`` per ``(species, locus)`` under this directory,
consumed at runtime by :class:`mir.distances.germline.GermlineDistances` — so the
runtime never needs BioPython or the gene library.

Run::

    python mir/resources/germline_dist/build_germline_dist.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
from Bio.Align import PairwiseAligner, substitution_matrices

_HERE = Path(__file__).resolve().parent
_REGION_ANNOTATIONS = _HERE.parent / "gene_library" / "region_annotations.txt"

# component -> (gene_type, list of region columns concatenated to form the AA seq)
_COMPONENTS: dict[str, tuple[str, list[str]]] = {
    "V": ("V", ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa"]),
    "CDR1": ("V", ["cdr1_aa"]),
    "CDR2": ("V", ["cdr2_aa"]),
    "J": ("J", ["jcdr3_aa", "fwr4_aa"]),
}


def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -12.0
    aligner.extend_gap_score = -1.0
    aligner.mode = "global"
    return aligner


def _component_seqs(
    df: pl.DataFrame, species: str, locus: str, gene_type: str, cols: list[str]
) -> tuple[list[str], list[str]]:
    """Return ``(alleles, sequences)`` for one component, dropping empty rows."""
    sub = df.filter(
        (pl.col("species") == species)
        & (pl.col("locus") == locus)
        & (pl.col("gene") == gene_type)
    )
    alleles: list[str] = []
    seqs: list[str] = []
    for row in sub.iter_rows(named=True):
        seq = "".join((row[c] or "") for c in cols)
        if seq:
            alleles.append(row["allele"])
            seqs.append(seq)
    return alleles, seqs


def _distance_matrix(aligner: PairwiseAligner, seqs: list[str]) -> np.ndarray:
    """Pairwise ``d(a,b)=s(a,a)+s(b,b)−2s(a,b)`` over *seqs*, deduping identical seqs."""
    uniq = sorted(set(seqs))
    uidx = {s: i for i, s in enumerate(uniq)}
    n = len(uniq)
    self_score = np.array([aligner.score(s, s) for s in uniq], dtype=np.float64)
    cross = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        cross[i, i] = self_score[i]
        for j in range(i + 1, n):
            s = aligner.score(uniq[i], uniq[j])
            cross[i, j] = cross[j, i] = s
    dist_uniq = self_score[:, None] + self_score[None, :] - 2.0 * cross
    # expand unique matrix back to allele order
    order = np.array([uidx[s] for s in seqs])
    return dist_uniq[np.ix_(order, order)].astype(np.float32)


def build() -> None:
    df = pl.read_csv(_REGION_ANNOTATIONS, separator="\t")
    aligner = _make_aligner()
    pairs = (
        df.select("species", "locus").unique().sort(["species", "locus"]).iter_rows()
    )
    for species, locus in pairs:
        out: dict[str, np.ndarray] = {}
        t0 = time.perf_counter()
        for comp, (gene_type, cols) in _COMPONENTS.items():
            alleles, seqs = _component_seqs(df, species, locus, gene_type, cols)
            if not seqs:
                continue
            dist = _distance_matrix(aligner, seqs)
            out[f"{comp}_alleles"] = np.array(alleles)
            out[f"{comp}_dist"] = dist
            out[f"{comp}_fallback"] = np.float32(dist.max() if dist.size else 0.0)
        if not out:
            continue
        path = _HERE / f"{species}_{locus}.npz"
        np.savez_compressed(path, **out)
        comps = sorted({k.split("_")[0] for k in out})
        dt = time.perf_counter() - t0
        print(f"  {species}_{locus}: {comps} -> {path.name} ({dt:.1f}s)")


if __name__ == "__main__":
    print(f"Baking germline distances from {_REGION_ANNOTATIONS.name} ...")
    t = time.perf_counter()
    build()
    print(f"Done in {time.perf_counter() - t:.1f}s")
