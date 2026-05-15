#!/usr/bin/env python
"""Generate TCREMP prototype reference files.

Populates mir/resources/prototypes/ with one TSV file per species/locus
combination found in the bundled OLGA models.  Each file contains exactly
N_PROTOTYPES (10,000) rows with columns v_gene, j_gene, junction_aa,
sampled from a real repertoire control when one is available and falling
back to a synthetic OLGA-generated control otherwise.

Run once from the repository root with the venv activated:

    python mir/resources/prototypes/generate_prototypes.py

Options:
    --overwrite     Regenerate files even if they already exist.
    --n INT         Number of prototypes per file (default 10,000).
    --seed INT      RNG seed (default 42).
    --n-jobs INT    Worker processes for synthetic generation (default: all CPUs).
    --no-real       Skip real control download; always use synthetic.
    --output-dir    Directory to write TSV files (default: same directory as
                    this script, i.e. mir/resources/prototypes/).

Do NOT run this script at build or test time — the output files are checked
into the repository and serve as a fixed reference for TCREMP embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Editable install puts the repo root on sys.path; this is a fallback for
# running the script directly without the venv being the active interpreter.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import polars as pl

from mir.basic.aliases import normalize_locus_alias, normalize_species_alias
from mir.basic.pgen import OlgaModel
from mir.common.alleles import allele_with_default
from mir.common.control import ControlManager

N_PROTOTYPES: int = 10_000
_SEED: int = 42
_OVERSAMPLE: int = 3  # request 3× to absorb (v_gene, j_gene, junction_aa) collisions
_COLS: list[str] = ["v_gene", "j_gene", "junction_aa"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_allele(gene: str) -> str:
    return allele_with_default(gene)


def _unique_triples(df: pl.DataFrame) -> pl.DataFrame:
    """Return rows with unique (v_gene, j_gene, junction_aa) in original order."""
    return df.select(_COLS).unique(subset=_COLS, maintain_order=True)


def _from_synthetic(
    species: str,
    locus: str,
    n: int,
    seed: int,
) -> pl.DataFrame:
    """Generate prototypes from the OLGA synthetic model.

    Uses ``generate_sequences_with_meta(pgens=False)`` which skips Pgen
    computation — much faster than ``generate_pool`` for this purpose.

    Oversamples by _OVERSAMPLE to handle duplicate (v_gene, j_gene, junction_aa)
    tuples; extends with an extra pass if the first pass yields fewer than n
    unique triples (very unlikely in practice).
    """
    target = n * _OVERSAMPLE

    def _generate_batch(batch_seed: int) -> pl.DataFrame:
        model = OlgaModel(species=species, locus=locus, seed=batch_seed)
        records = model.generate_sequences_with_meta(n=target, pgens=False, seed=batch_seed)
        rows = [
            {
                "v_gene": _normalize_allele(rec["v_gene"]),
                "j_gene": _normalize_allele(rec["j_gene"]),
                "junction_aa": rec["junction_aa"],
            }
            for rec in records
        ]
        return pl.DataFrame(rows, schema=_COLS)

    unique = _unique_triples(_generate_batch(seed))

    if len(unique) < n:
        print(
            f"  First pass yielded only {len(unique)} unique triples for "
            f"{species}/{locus}; generating extension…"
        )
        extra = _unique_triples(_generate_batch(seed + n))
        unique = (
            pl.concat([unique, extra])
            .unique(subset=_COLS, maintain_order=True)
        )

    return unique.head(n)


def _from_real(
    manager: ControlManager,
    species: str,
    locus: str,
    n: int,
    seed: int,
) -> pl.DataFrame:
    """Sample prototypes from a real control repertoire.

    Downloads the real control from HuggingFace if not already cached.
    Raises FileNotFoundError / ImportError when the combo is unavailable.
    """
    manager.ensure_real_control(species, locus)
    real_df = manager.load_control_df("real", species, locus)
    oversample_n = min(len(real_df), n * _OVERSAMPLE)
    sampled = real_df.sample(n=oversample_n, seed=seed, shuffle=True)
    unique = _unique_triples(sampled)
    if len(unique) >= n:
        return unique.head(n)
    # Fewer unique triples than n — return everything we have.
    print(
        f"  Real control for {species}/{locus} has only "
        f"{len(unique)} unique triples (< {n}); using all of them."
    )
    return unique


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def generate_all_prototypes(
    output_dir: Path,
    n: int = N_PROTOTYPES,
    seed: int = _SEED,
    overwrite: bool = False,
    use_real: bool = True,
) -> None:
    """Generate prototype TSV files for every available OLGA model combination.

    Args:
        output_dir: Directory to write TSV files and manifest.json.
        n: Number of prototype rows per file.
        seed: RNG seed for reproducibility.
        overwrite: Regenerate files that already exist.
        use_real: Attempt real control download before falling back to synthetic.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = ControlManager.list_available_olga_models()
    if not combos:
        print("No OLGA models found — nothing to generate.")
        return

    print(f"Found {len(combos)} OLGA model(s): {combos}")

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

    manager = ControlManager()

    for species, locus in combos:
        out_path = output_dir / f"{species}_{locus}.tsv"

        if out_path.exists() and not overwrite:
            print(f"[skip] {species}/{locus}: {out_path.name} already exists")
            continue

        print(f"\n[generate] {species}/{locus} → {out_path.name}")
        source = "synthetic"
        df: pl.DataFrame | None = None

        if use_real:
            # Only use real control if already cached locally (no download attempt).
            rec = manager.get_record("real", species, locus)
            if rec is not None and Path(rec.path).exists():
                try:
                    df = _from_real(manager, species, locus, n, seed)
                    source = "real"
                    print(f"  Using cached real control: {len(df)} prototypes")
                except Exception as exc:
                    print(f"  Real control load failed ({type(exc).__name__}: {exc}); using synthetic")
            else:
                print(f"  No cached real control for {species}/{locus}; using synthetic")

        if df is None:
            df = _from_synthetic(species, locus, n, seed)
            source = "synthetic"
            print(f"  Generated synthetic prototypes: {len(df)} entries")

        df.write_csv(out_path, separator="\t")

        manifest[f"{species}_{locus}"] = {
            "species": species,
            "locus": locus,
            "n": len(df),
            "source": source,
            "seed": seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

        print(f"  Wrote {out_path}")

    print(f"\nDone. Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate TCREMP prototype reference files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=str(_HERE),
        help="Directory to write TSV files (default: mir/resources/prototypes/)",
    )
    parser.add_argument("--n", type=int, default=N_PROTOTYPES, help="Prototypes per file")
    parser.add_argument("--seed", type=int, default=_SEED, help="RNG seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--no-real",
        action="store_true",
        help="Skip real control download; always use synthetic",
    )
    args = parser.parse_args(argv)

    generate_all_prototypes(
        output_dir=Path(args.output_dir),
        n=args.n,
        seed=args.seed,
        overwrite=args.overwrite,
        use_real=not args.no_real,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
