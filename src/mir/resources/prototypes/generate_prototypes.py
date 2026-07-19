#!/usr/bin/env python
"""Generate TCREMP prototype reference files from **arda-annotated real repertoires**.

Populates ``mir/resources/prototypes/`` with one TSV per (species, locus): ``N_PROTOTYPES`` rows
of ``v_call, j_call, junction_aa`` sampled from **productive real reads annotated by arda**
(``vdjtools.model.data.prepare(..., "functional")`` over the ``isalgo/airr_model_read`` FASTQs →
``arda rnaseq map``). This gives arda IMGT allele names (one coordinate frame with arda-annotated
query data → no germline-distance fallback) **and** the real CDR3 junction manifold, which embeds
far better than synthetic P_gen junctions: EM-learned generative models produce degenerate
over-deleted CDR3 lengths whose self-prototype distance correlation (T1/S2) is negative, whereas
real repertoires give tight lengths and a strongly positive correlation.

Build-time only: the output TSVs are checked in and are the *fixed, versioned* reference for TCREMP
embeddings. Regenerating changes the embedding coordinate system.

Run (needs the ``[build]`` extra: ``arda-mapper`` + ``ARDA_HOME``; downloads reads from HF)::

    ARDA_HOME=/path/to/arda EM_WORK=/scratch/reads \
        python mir/resources/prototypes/generate_prototypes.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

_HERE = Path(__file__).resolve().parent
N_PROTOTYPES: int = 10_000
_SEED: int = 42
_COLS: list[str] = ["v_call", "j_call", "junction_aa"]
_CANON = r"^C[ACDEFGHIKLMNPQRSTVWY]*[FW]$"  # canonical CDR3: Cys … Phe/Trp, amino acids only
_READ_CAP = int(os.environ.get("READ_CAP", "300000"))
_WORK = os.environ.get("EM_WORK", str(_HERE / "_reads"))
# mirpy prototype combos (species, locus): 7 human loci + mouse TRA/TRB
_COMBOS: list[tuple[str, str]] = (
    [("human", loc) for loc in ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")]
    + [("mouse", "TRA"), ("mouse", "TRB")]
)


def _resolvable(uniq: pl.DataFrame, species: str, locus: str) -> pl.DataFrame:
    """Keep only rows whose v_call/j_call resolve in the baked germline distances (no fallback)."""
    from mir.distances.germline import load_germline_distances

    gd = load_germline_distances(species, locus)
    cv, cj = gd._components["V"], gd._components["J"]
    ok_v = [x for x in uniq["v_call"].unique().to_list() if cv.resolve(x) != cv.fallback_idx]
    ok_j = [x for x in uniq["j_call"].unique().to_list() if cj.resolve(x) != cj.fallback_idx]
    return uniq.filter(pl.col("v_call").is_in(ok_v) & pl.col("j_call").is_in(ok_j))


def _generate(species: str, locus: str, n: int, seed: int) -> pl.DataFrame:
    """Sample ``n`` unique germline-resolvable prototypes from arda-annotated real reads."""
    from vdjtools.model import data

    uniq = data.prepare(species, locus, "functional", out_dir=_WORK, cap=_READ_CAP)
    df = (
        uniq.select(_COLS)
        .filter(pl.col("junction_aa").str.contains(_CANON))
        .unique(subset=_COLS)
    )
    df = _resolvable(df, species, locus)
    if df.height <= n:
        return df.sort(_COLS)  # deterministic order; fewer than n unique clonotypes available
    return df.sample(n, seed=seed, shuffle=True)  # diverse uniform-over-clonotype sample


def generate_all_prototypes(output_dir: Path, n: int = N_PROTOTYPES, seed: int = _SEED,
                            overwrite: bool = False) -> None:
    """Write a prototype TSV + manifest entry for every (species, locus) combo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for species, locus in _COMBOS:
        out_path = output_dir / f"{species}_{locus}.tsv"
        if out_path.exists() and not overwrite:
            print(f"[skip] {species}_{locus}: exists")
            continue
        df = _generate(species, locus, n, seed)
        df.write_csv(out_path, separator="\t")
        manifest[f"{species}_{locus}"] = {
            "species": species, "locus": locus, "n": df.height, "source": "arda-real", "seed": seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"[generate] {species}_{locus}: {df.height} prototypes, "
              f"{df['v_call'].n_unique()} V / {df['j_call'].n_unique()} J genes")
    print(f"Done. Manifest: {manifest_path}")


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate TCREMP prototypes from arda-annotated reads.")
    p.add_argument("--output-dir", default=str(_HERE))
    p.add_argument("--n", type=int, default=N_PROTOTYPES)
    p.add_argument("--seed", type=int, default=_SEED)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    generate_all_prototypes(Path(args.output_dir), n=args.n, seed=args.seed, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
