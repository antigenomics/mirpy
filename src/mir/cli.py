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


def _per_locus_path(path: str, locus: str) -> str:
    """Insert ``locus`` before the extension: ``mmd.tsv`` → ``mmd.TRB.tsv``.

    A plain ``str.replace(".", …, 1)`` would split on the first dot *anywhere* — mangling
    ``./mmd.tsv`` into ``.TRB./mmd.tsv`` — and would silently no-op on an extensionless path,
    letting one locus overwrite another's matrix.
    """
    from pathlib import Path

    p = Path(path)
    return str(p.with_name(f"{p.stem}.{locus}{p.suffix}"))


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
    """Resolve ``--locus`` to a canonical IMGT locus, or infer it when the file has only one."""
    loci = [x for x in df["locus"].unique().to_list() if x]
    if requested:
        # Through the alias table, so `--locus beta` matches the data's `TRB` rows rather
        # than silently selecting nothing.
        from mir.aliases import normalize_locus_alias

        try:
            return normalize_locus_alias(requested)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if len(loci) == 1:
        return loci[0]
    raise SystemExit(
        f"multiple loci present ({', '.join(sorted(loci))}); pass --locus to pick one"
    )


def _apply_functional_filter(sub: pl.DataFrame, enabled: bool) -> pl.DataFrame:
    """Drop non-coding clonotypes (stop codon / out-of-frame ``junction_aa``) unless disabled.

    Defends against exactly what crashes or silently corrupts embedding otherwise: a stop codon
    (``*``) or legacy out-of-frame marker (e.g. ``_``) in ``junction_aa``.
    """
    if not enabled:
        return sub
    from vdjtools.preprocess import filter_functional

    n0 = sub.height
    sub = filter_functional(sub)
    if sub.height < n0:
        print(f"[mir] filtered {n0 - sub.height} non-coding clonotype(s) "
              "(stop codon / out-of-frame junction_aa)", file=sys.stderr)
    return sub


# --- commands --------------------------------------------------------------
def cmd_clonotypes(a: argparse.Namespace) -> None:
    from mir.embedding.pca import pca_denoise
    from mir.embedding.tcremp import TCREmp

    df = _with_locus(_read(a.input))
    locus = _pick_locus(df, a.locus)
    sub = df.filter(pl.col("locus") == locus)
    if sub.is_empty():
        raise SystemExit(f"no clonotypes for locus {locus!r} in {a.input}")
    sub = _apply_functional_filter(sub, a.filter_functional)
    if sub.is_empty():
        raise SystemExit(f"no coding clonotypes remain for locus {locus!r} after functional filtering")

    model = TCREmp.from_defaults(a.species, locus, n_prototypes=a.n_prototypes,
                                 mode=a.mode, replicate=a.replicate, threads=a.threads)
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
            sub = _apply_functional_filter(df.filter(pl.col("locus") == locus), a.filter_functional)
            if sub.is_empty():
                print(f"[mir] {sid}/{locus}: no coding clonotypes after functional filtering, "
                      "skipping this sample/locus", file=sys.stderr)
                continue
            by_locus[locus].append((sid, sub))

    if not by_locus:
        raise SystemExit("no samples/loci to embed (check inputs / --locus)")

    rows: list[dict] = []
    vectors: list = []
    for locus in sorted(by_locus):
        items = by_locus[locus]
        model = TCREmp.from_defaults(a.species, locus, n_prototypes=a.n_prototypes,
                                    replicate=a.replicate, threads=a.threads)
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
            out = a.mmd if len(by_locus) == 1 else _per_locus_path(a.mmd, locus)
            _write(mmd_df, out)
        print(f"[mir] {locus}: {len(items)} samples → Φ dim {len(embs[0].vector)}", file=sys.stderr)

    import numpy as np

    meta = pl.DataFrame(rows)
    out = meta.hstack(_emb_frame(np.vstack(vectors), "phi"))
    _write(out, a.output)


def cmd_signature(a: argparse.Namespace) -> None:
    from vdjtools.signature import presets as P

    from mir.signature import assemble, columns, describe

    # --preset picks BOTH the tier and the column subset. mirpy is where the two halves meet, so
    # unlike `vdjtools signature` every preset resolves here in full.
    keep = None
    if getattr(a, "preset", None):
        try:
            spec = P.get(a.preset)
        except KeyError as e:
            raise SystemExit(str(e)) from None
        a.tier, keep = spec.tier, spec.columns()
        print(f"[mir] preset {spec.name!r} [{spec.rank}]: {len(keep)} columns, "
              f"suggested scaling {spec.scaling}", file=sys.stderr)

    if a.describe:
        d = describe(a.tier)
        _write(d.filter(pl.col("column").is_in(keep)) if keep else d, a.output)
        return

    # A sample is one file, or several files sharing a sample id — a donor sequenced on TRA and
    # TRB is one signature with both loci filled, not two half-empty ones.
    from collections import defaultdict

    samples: dict[str, dict[str, pl.DataFrame]] = defaultdict(dict)
    for path in a.input:
        df = _with_locus(_read(path))
        sid = _sample_id(path)
        for locus in [x for x in df["locus"].unique().to_list() if x]:
            sub = df.filter(pl.col("locus") == locus)
            if sub.height:
                samples[sid][locus] = sub

    if not samples:
        raise SystemExit("no samples to sign (check inputs)")

    scale = None
    if a.standardize == "reference" and a.scale:
        from mir.signature.scale import load_scale
        scale = load_scale(a.scale)

    out = assemble.signature_cohort(samples, tier=a.tier, species=a.species, weight=a.weight,
                                    standardize=a.standardize, scale=scale,
                                    n_jobs=a.threads, columns=keep)
    n_cols = len(keep) if keep else len(columns(a.tier))
    print(f"[mir] {out.height} samples x {n_cols} columns "
          f"({a.preset or a.tier}, standardize={a.standardize})", file=sys.stderr)
    _write(out, a.output)


def cmd_presets(a: argparse.Namespace) -> None:
    """List the feature presets, or explain one.

    `recommended` — use unless you have a reason not to. `specific` — correct for a stated purpose
    and wrong outside it. `avoid` — a control or a measured dead end, named so that picking it is
    deliberate rather than accidental.
    """
    from vdjtools.signature import presets as P

    if not a.name:
        _write(P.table().select("preset", "rank", "columns", "halves", "scaling", "summary"),
               a.output)
        return
    try:
        spec = P.get(a.name)
    except KeyError as e:
        raise SystemExit(str(e)) from None
    print(f"{spec.name}  [{spec.rank}]  {spec.n_columns} columns  tier={spec.tier}  "
          f"halves={'+'.join(spec.sig)}  scaling={spec.scaling}\n")
    for label, text in (("summary", spec.summary), ("features", spec.features),
                        ("how it is computed", spec.how), ("use cases", spec.use_cases),
                        ("notes", spec.notes)):
        if text:
            print(f"{label}:\n  {text}\n")


# --- parser ----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Return the ``mir`` argument parser (``embed clonotypes`` / ``embed repertoires``)."""
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
    c.add_argument("--replicate", type=int, default=0, metavar="R",
                   help="prototype draw: 0 = the default set; r>0 = an independent disjoint draw of the same size, for prototype-sensitivity runs (embeddings across draws are NOT comparable)")
    c.add_argument("--pca", type=int, default=None, metavar="K",
                   help="PCA-denoise the embedding to K dims (compact table)")
    c.add_argument("--filter-functional", action=argparse.BooleanOptionalAction, default=True,
                   help="drop non-coding clonotypes (stop codon / out-of-frame junction_aa) "
                        "before embedding (default: on)")
    c.add_argument("--threads", type=int, default=0, help="0 = all cores")
    c.set_defaults(func=cmd_clonotypes)

    r = embed.add_parser("repertoires", help="dataset of clonotype tables → per-sample Φ(S), by chain")
    r.add_argument("input", nargs="+", help="one clonotype file per repertoire (sample id = filename stem)")
    r.add_argument("-o", "--output", help="output .tsv/.parquet (default: stdout TSV)")
    r.add_argument("--species", default="human")
    r.add_argument("--locus", help="restrict to one chain (default: all loci present, one basis each)")
    r.add_argument("--n-prototypes", type=int, default=None)
    r.add_argument("--replicate", type=int, default=0, metavar="R",
                   help="prototype draw: 0 = the default set; r>0 = an independent disjoint draw of the same size, for prototype-sensitivity runs (embeddings across draws are NOT comparable)")
    r.add_argument("--weight", default="log2p1",
                   choices=("log2p1", "duplicate_count", "distinct", "log1p", "anscombe"),
                   help="clone-size weight g (frequencies w = g(a)/Σg): log2p1 g=log2(1+a) "
                        "(default), duplicate_count g=a (linear), distinct g=1 (presence), "
                        "log1p g=ln(1+a), anscombe g=√(a+3/8)")
    r.add_argument("--blocks", default="mean,diversity",
                   help="Φ blocks: mean,diversity[,second] (second = heavy HLA-interaction block)")
    r.add_argument("--n-rff", type=int, default=1024, help="mean-block RFF dimension")
    r.add_argument("--n-rff-second", type=int, default=128, help="second-moment RFF dimension (if used)")
    r.add_argument("--n-components", type=int, default=None,
                   help="clonotype-PCA dims for the shared basis (default: preset)")
    r.add_argument("--mmd", metavar="OUT", help="also write the per-chain pairwise unbiased-MMD matrix")
    r.add_argument("--filter-functional", action=argparse.BooleanOptionalAction, default=True,
                   help="drop non-coding clonotypes (stop codon / out-of-frame junction_aa) "
                        "before embedding (default: on)")
    r.add_argument("--threads", type=int, default=0, help="0 = all cores")
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(func=cmd_repertoires)

    s = sub.add_parser("signature",
                       help="clonotype tables → the portable repertoire signature (one row/sample)")
    s.add_argument("input", nargs="*",
                   help="one or more clonotype files; files sharing a sample id (the name up to "
                        "the first dot) are joined into one multi-locus sample")
    s.add_argument("-o", "--output", help="output .tsv/.parquet (default: stdout TSV)")
    s.add_argument("--tier", default="standard", choices=("core", "standard", "full"),
                   help="column set; the narrower tiers are exact index subsets of the wider ones")
    s.add_argument("--species", default="human")
    s.add_argument("--weight", default="log2p1",
                   choices=("log2p1", "duplicate_count", "distinct", "log1p", "anscombe"),
                   help="clone-size weight g (default log2p1)")
    s.add_argument("--standardize", default="reference", choices=("reference", "none"),
                   help="'reference' rescales every column against the bundled reference so the "
                        "vector is comparable with anyone else's (default); 'none' emits raw "
                        "block values")
    s.add_argument("--scale", default=None,
                   help="path to an alternative scale artifact (default: the bundled one)")
    s.add_argument("--preset", default=None,
                   help="named feature set; overrides --tier (see `mir presets`)")
    s.add_argument("--threads", type=int, default=1,
                   help="worker processes over samples; 0 = every core (default 1)")
    s.add_argument("--describe", action="store_true",
                   help="print the column dictionary for --tier/--preset and exit; reads no input")
    s.set_defaults(func=cmd_signature)

    q = sub.add_parser("presets",
                       help="list the named feature sets with their rankings")
    q.add_argument("name", nargs="?", help="show one preset in full")
    q.add_argument("-o", "--output", help="output .tsv/.parquet (default: stdout TSV)")
    q.set_defaults(func=cmd_presets)

    return p


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv[1:]``) and run the requested ``mir`` subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
