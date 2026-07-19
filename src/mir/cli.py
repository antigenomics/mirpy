"""``mir`` command-line interface — turn receptor tables into embeddings.

Two commands cover the two scales mirpy embeds at:

* ``mir embed clonotypes SAMPLE``   — one repertoire's clonotype table → a per-clonotype
  TCREMP embedding table (``e0…``), the input to clustering / ML.
* ``mir embed repertoires SAMPLE…`` — a *dataset* of clonotype tables → one repertoire
  vector ``Φ(S)`` (``phi0…``) per sample, per chain, on one shared basis (so the rows are
  mutually comparable / MMD-able).

Inputs are any format ``vdjtools.io.read`` sniffs (AIRR TSV, vdjtools, MiXCR, immunoSEQ,
parquet, …). Output is TSV (default / ``.tsv``) or Parquet (``.parquet`` — recommended for
the wide raw embedding); ``-o -`` (or no ``-o``) writes TSV to stdout.

Run ``mir embed clonotypes -h`` / ``mir embed repertoires -h`` for the full flag list.
"""
from __future__ import annotations

import argparse
import sys

import polars as pl

import mir


# --- IO helpers ------------------------------------------------------------
def _read(path: str) -> pl.DataFrame:
    """Read a clonotype file into a normalized AIRR frame (any format vdjtools sniffs)."""
    from vdjtools import io

    return io.read(path)


def _with_locus(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure a ``locus`` column (derive from ``v_call`` when absent)."""
    if "locus" in df.columns and df["locus"].null_count() < df.height:
        return df
    from vdjtools import io

    try:
        return io.add_locus(df)
    except Exception:
        # Fallback: the IMGT locus is the v_call's first 3 characters (TRB/TRA/IGH/…).
        return df.with_columns(pl.col("v_call").str.slice(0, 3).alias("locus"))


def _emb_frame(X, prefix: str) -> pl.DataFrame:
    """(N, d) float matrix → a polars frame with columns ``{prefix}0…{prefix}{d-1}``."""
    return pl.from_numpy(X, schema=[f"{prefix}{i}" for i in range(X.shape[1])])


def _write(df: pl.DataFrame, path: str | None) -> None:
    if path is None or path == "-":
        sys.stdout.write(df.write_csv(separator="\t"))
    elif path.endswith(".parquet"):
        df.write_parquet(path)
    else:
        df.write_csv(path, separator="\t")


def _sample_id(path: str) -> str:
    """Sample id = filename up to the first dot (``P1.TRB.tsv.gz`` → ``P1``)."""
    import os

    return os.path.basename(path).split(".")[0]


def _pick_locus(df: pl.DataFrame, requested: str | None) -> str:
    loci = [x for x in df["locus"].unique().to_list() if x]
    if requested:
        return requested
    if len(loci) == 1:
        return loci[0]
    raise SystemExit(
        f"multiple loci present ({', '.join(sorted(loci))}); pass --locus to pick one"
    )


# --- commands --------------------------------------------------------------
def cmd_clonotypes(a: argparse.Namespace) -> None:
    from mir.embedding.pca import pca_denoise
    from mir.embedding.tcremp import TCREmp

    df = _with_locus(_read(a.input))
    locus = _pick_locus(df, a.locus)
    sub = df.filter(pl.col("locus") == locus)
    if sub.is_empty():
        raise SystemExit(f"no clonotypes for locus {locus!r} in {a.input}")

    model = TCREmp.from_defaults(a.species, locus, n_prototypes=a.n_prototypes,
                                 mode=a.mode, threads=a.threads)
    X = model.embed(sub)
    if a.pca:
        X = pca_denoise(X, n_components=a.pca)

    if (a.output is None or a.output == "-" or a.output.endswith(".tsv")) and X.shape[1] > 500:
        print(f"[mir] {X.shape[1]} embedding columns — consider --pca K or a .parquet output.",
              file=sys.stderr)

    id_cols = [c for c in ("junction_aa", "v_call", "j_call", "duplicate_count") if c in sub.columns]
    out = sub.select(id_cols).hstack(_emb_frame(X, "e"))
    _write(out, a.output)
    print(f"[mir] embedded {X.shape[0]} {locus} clonotypes → {X.shape[1]}-d "
          f"({'PCA ' if a.pca else ''}table)", file=sys.stderr)


def cmd_repertoires(a: argparse.Namespace) -> None:
    from collections import defaultdict

    from mir.embedding.tcremp import TCREmp
    from mir.repertoire import fit_repertoire_space, mmd_matrix, sample_embedding

    blocks = tuple(b.strip() for b in a.blocks.split(",") if b.strip())
    n_rff_second = a.n_rff_second if "second" in blocks else 0

    # Load every sample, split its clonotypes by locus.
    by_locus: dict[str, list] = defaultdict(list)
    for path in a.input:
        df = _with_locus(_read(path))
        sid = _sample_id(path)
        for locus in [x for x in df["locus"].unique().to_list() if x]:
            if a.locus and locus != a.locus:
                continue
            by_locus[locus].append((sid, df.filter(pl.col("locus") == locus)))

    if not by_locus:
        raise SystemExit("no samples/loci to embed (check inputs / --locus)")

    rows: list[dict] = []
    vectors: list = []
    for locus in sorted(by_locus):
        items = by_locus[locus]
        model = TCREmp.from_defaults(a.species, locus, n_prototypes=a.n_prototypes, threads=a.threads)
        pooled = pl.concat([sub for _, sub in items])
        space = fit_repertoire_space(model, pooled, n_rff=a.n_rff, n_rff_second=n_rff_second,
                                     n_components=a.n_components, seed=a.seed)
        embs = [sample_embedding(space, sub, weight=a.weight, blocks=blocks) for _, sub in items]
        for (sid, sub), se in zip(items, embs):
            rows.append({"sample_id": sid, "locus": locus, "n_clonotypes": sub.height})
            vectors.append(se.vector)
        if a.mmd:
            D = mmd_matrix(embs, unbiased=True)
            ids = [sid for sid, _ in items]
            mmd_df = pl.DataFrame({"sample_id": ids}).with_columns(
                [pl.Series(ids[j], D[:, j]) for j in range(len(ids))])
            out = a.mmd if len(by_locus) == 1 else a.mmd.replace(".", f".{locus}.", 1)
            _write(mmd_df, out)
        print(f"[mir] {locus}: {len(items)} samples → Φ dim {len(embs[0].vector)}", file=sys.stderr)

    import numpy as np

    meta = pl.DataFrame(rows)
    out = meta.hstack(_emb_frame(np.vstack(vectors), "phi"))
    _write(out, a.output)


# --- parser ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mir", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"mir {mir.__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    embed = sub.add_parser("embed", help="compute embeddings").add_subparsers(dest="what", required=True)

    c = embed.add_parser("clonotypes", help="repertoire → per-clonotype embedding table")
    c.add_argument("input", help="a clonotype table (AIRR/vdjtools/MiXCR/parquet/…)")
    c.add_argument("-o", "--output", help="output .tsv/.parquet (default: stdout TSV)")
    c.add_argument("--species", default="human")
    c.add_argument("--locus", help="chain to embed (inferred if the file has one locus)")
    c.add_argument("--n-prototypes", type=int, default=None,
                   help="prototype count (default: per-chain preset)")
    c.add_argument("--mode", default="vjcdr3", choices=("vjcdr3", "cdr123"))
    c.add_argument("--pca", type=int, default=None, metavar="K",
                   help="PCA-denoise the embedding to K dims (compact table)")
    c.add_argument("--threads", type=int, default=0, help="0 = all cores")
    c.set_defaults(func=cmd_clonotypes)

    r = embed.add_parser("repertoires", help="dataset of clonotype tables → per-sample Φ(S), by chain")
    r.add_argument("input", nargs="+", help="one clonotype file per repertoire (sample id = filename stem)")
    r.add_argument("-o", "--output", help="output .tsv/.parquet (default: stdout TSV)")
    r.add_argument("--species", default="human")
    r.add_argument("--locus", help="restrict to one chain (default: all loci present, one basis each)")
    r.add_argument("--n-prototypes", type=int, default=None)
    r.add_argument("--weight", default="log1p", choices=("log1p", "anscombe", "distinct"),
                   help="clone-size weight g (frequencies w = g(a)/Σg)")
    r.add_argument("--blocks", default="mean,diversity",
                   help="Φ blocks: mean,diversity[,second] (second = heavy HLA-interaction block)")
    r.add_argument("--n-rff", type=int, default=1024, help="mean-block RFF dimension")
    r.add_argument("--n-rff-second", type=int, default=128, help="second-moment RFF dimension (if used)")
    r.add_argument("--n-components", type=int, default=None,
                   help="clonotype-PCA dims for the shared basis (default: preset)")
    r.add_argument("--mmd", metavar="OUT", help="also write the per-chain pairwise unbiased-MMD matrix")
    r.add_argument("--threads", type=int, default=0, help="0 = all cores")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_repertoires)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
