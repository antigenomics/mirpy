"""``mir`` command-line interface — turn receptor tables into embeddings and signatures.

**Start with ``mir signature``** unless you know you want a raw embedding. It is the command you
send a collaborator: fixed-width, named columns, already standardised.

Four commands. Two embed at the two scales mirpy works at:

* ``mir embed clonotypes SAMPLE``   — one repertoire's clonotype table → a per-clonotype
  TCREMP embedding table (``e0…``), the input to clustering / ML.
* ``mir embed repertoires SAMPLE…`` — a *dataset* of clonotype tables → one repertoire
  vector ``Φ(S)`` (``phi0…``) per sample, per chain, on one shared basis (so the rows are
  mutually comparable / MMD-able).

Two produce the portable signature — the hand-off object, whose basis is frozen rather than fitted
on your cohort, so two people's vectors are comparable:

* ``mir signature SAMPLE…`` — → one fixed-width named vector per sample (528 columns at the
  ``standard`` tier), standardised against a frozen reference. Emits **this tool's half**:
  ``rsig``, the embedding geometry. The other half, ``vsig`` (statistics), comes from
  ``vdjtools signature``; run both and join on ``sample_id`` for the full 689-column vector.
* ``mir presets [NAME]``    — the named column subsets and their ranking, so a subset is chosen by
  intent rather than by reading a 1,403-row column dictionary.

Inputs are any format ``vdjtools.io.read`` sniffs (AIRR TSV, vdjtools, MiXCR, immunoSEQ,
parquet, …). Output is TSV (default / ``.tsv``) or Parquet (``.parquet`` — recommended for
the wide raw embedding); ``-o -`` (or no ``-o``) writes TSV to stdout.

Run ``mir <command> -h`` for the full flag list; ``mir signature --describe`` prints the column
dictionary and ``mir signature --channels`` the channel vocabulary --- the named groups those
columns fall into, which is the level a finding is usually stated at. Both read no input.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import polars as pl

import mir


# --- IO helpers ------------------------------------------------------------
def _read(path: str, *, productive_only: bool = True,
          recompute_frequencies: bool = True) -> pl.DataFrame:
    """Read a clonotype file into a normalized AIRR frame, **productive rearrangements only**.

    ``v_identity`` is kept by name: it is the one field the signature needs that the canonical
    schema does not carry, so without it ``vsig:shm:IGH:mean_v_identity`` is not merely absent
    but uncomputable, and ships as a nan column on files that do have it.

    **Non-productive rearrangements are removed on every read, and that cannot be turned off.**
    This is where mirpy deliberately differs from vdjtools, where the same filter is optional. The
    reason is specific: a stop codon is *in* seqtree's amino-acid alphabet, so an unfiltered frame
    does not crash the distance code -- it returns a finite, meaningless number and contaminates
    the geometry silently. mirpy cannot embed a non-productive sequence *meaningfully*, which is a
    stronger and more useful statement than cannot embed it at all.

    Use ``vdjtools`` if you want the non-productive fraction: ``vdjtools filter --nonproductive``,
    or :func:`vdjtools.preprocess.filter_productive` in Python.

    Args:
        path: Clonotype file, any format vdjtools sniffs.
        productive_only: Must stay ``True``. Present so that passing ``False`` raises with an
            explanation rather than silently doing the wrong thing.
        recompute_frequencies: Renormalise ``frequency`` over the surviving rearrangements.

    Raises:
        ValueError: If ``productive_only`` is ``False``.
    """
    from vdjtools import io
    from vdjtools.preprocess import filter_productive, productive_mask

    if not productive_only:
        raise ValueError(
            "mirpy cannot embed non-productive rearrangements meaningfully: a stop codon is in "
            "seqtree's alphabet, so it does not raise in the distance code -- it returns a finite, "
            "meaningless value and contaminates the geometry silently. productive_only=False is "
            "therefore refused here, unlike in vdjtools where the same filter is optional. Use "
            "`vdjtools filter --nonproductive` or vdjtools.preprocess.filter_productive() if the "
            "non-productive fraction is what you want.")
    df = io.read(path, keep=("v_identity",))
    n0 = df.height
    _, src = productive_mask(df)
    df = filter_productive(df, recompute_frequencies=recompute_frequencies)
    if df.height < n0:
        print(f"[mir] dropped {n0 - df.height} non-productive rearrangement(s) of {n0} "
              f"(productivity read from {src}); mirpy always does this", file=sys.stderr)
    return df


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


def _read_sample(paths: "list[str]") -> "dict[str, pl.DataFrame]":
    """Read one sample's file(s) into ``{locus: frame}``. Module-level so it can be pickled.

    ``signature_cohort`` takes a zero-argument callable per sample, and this is the one the CLI
    hands it via ``functools.partial``. Deferring the read is what keeps a big cohort inside a
    small machine: the parent process never holds a clonotype frame at all, each worker reads its
    own slice one sample at a time, and nothing but a list of paths crosses the pipe.
    """
    out: dict[str, pl.DataFrame] = {}
    for path in paths:
        df = _with_locus(_read(path))
        for locus in [x for x in df["locus"].unique().to_list() if x]:
            sub = df.filter(pl.col("locus") == locus)
            if sub.height:
                out[locus] = pl.concat([out[locus], sub]) if locus in out else sub
    return out


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


def _require_productive(enabled: bool) -> None:
    """mirpy has no --no-filter-functional. Refuse it, and say where to go instead."""
    if not enabled:
        raise SystemExit(
            "[mir] --no-filter-functional is not available in mirpy. A stop codon is in seqtree's "
            "alphabet, so an unfiltered frame does not crash -- it embeds to a finite, meaningless "
            "distance and contaminates the geometry silently. Use `vdjtools filter "
            "--nonproductive` if the non-productive fraction is what you want.")


def _apply_functional_filter(sub: pl.DataFrame, enabled: bool) -> pl.DataFrame:
    """Drop unusable clonotypes unless disabled.

    Uses ``vdjtools.signature.features.sanitise``, NOT ``preprocess.filter_functional``. The two are
    not the same predicate and the difference matters here: ``filter_functional`` is a denylist
    (``[*atgc#~_?]``) and keeps the ambiguity codes ``X``/``B``/``Z``, while ``sanitise`` is an
    anchored allowlist of the 20 standard amino acids and drops them. ``TCREmp.embed`` refuses
    exactly what ``sanitise`` drops, so filtering with the weaker predicate here would leave rows
    the embedder then rejects -- the default path has to clear its own guard.

    Neither predicate filters on LENGTH. A two-residue junction and a sixty-residue one both
    survive; the question asked is only whether the string is a plain amino-acid string.
    """
    _require_productive(enabled)
    from vdjtools.signature.features import sanitise

    n0 = sub.height
    sub, dropped = sanitise(sub)
    if sub.height < n0:
        print(f"[mir] filtered {n0 - sub.height} unusable clonotype(s) "
              f"({100 * dropped:.2f}% of reads: stop codon, out-of-frame marker, ambiguity code "
              "or non-positive count)", file=sys.stderr)
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


def _sample_items(a: argparse.Namespace) -> list:
    """``[(sample_id, deferred_read)]``, grouped by id **without reading anything**.

    A sample is one file, or several files sharing a sample id -- a donor sequenced on TRA and TRB is
    one signature with both loci filled, not two half-empty ones. The read is deferred into the
    worker that will use it: on 1,000 samples of 10,000 clonotypes that is 1.1 GB resident instead
    of 6.6 GB, because the alternative is for this process to materialise the whole cohort and then
    copy it down a pipe.
    """
    import functools
    from collections import defaultdict

    by_id: dict[str, list[str]] = defaultdict(list)
    for path in a.input:
        by_id[_sample_id(path)].append(str(path))
    if not by_id:
        raise SystemExit("no samples to sign (check inputs)")
    return [(sid, functools.partial(_read_sample, paths)) for sid, paths in by_id.items()]


def cmd_signature(a: argparse.Namespace) -> None:
    from vdjtools.signature import layout as L

    from mir.signature import rsig_cohort

    art = _resolve_corpus(a.corpus)
    ncomp = _parse_components(a.components)
    want = [c for c in pathlib.Path(a.columns).read_text().split() if c] if a.columns else None
    blocks = ()
    if a.named is not None:
        blocks = True if a.named.strip().lower() == "all" else tuple(
            b for b in (x.strip() for x in a.named.split(",")) if b)
        try:
            L.resolve_named(art.sig, blocks)
        except ValueError as e:
            sys.exit(f"mir signature: {e}")

    if a.describe:
        cols = art.columns(ncomp, named=blocks)
        if want is not None:
            cols = [c for c in cols if c in set(want)]
        named_set = set(L.named_columns(art.sig, blocks, art.vocab) if blocks else ())
        spec = {(r["block"], r["feature"]): r for r in L.channel_table(art.sig)}
        rows = []
        for c in cols:
            _sig, block, locus, feature = L.parse(c)
            kind = ("rotated" if block == L.PC_BLOCK
                    else "named" if c in named_set else "channel")
            rows.append({"column": c, "block": block, "locus": locus, "feature": feature,
                         "kind": kind,
                         "transform": spec.get((block, feature), {}).get("transform", "none"),
                         "support": L.support_of(c)})
        _write(pl.DataFrame(rows), a.output)
        return

    items = _sample_items(a)
    jobs = a.jobs
    print(f"[mir] {len(items)} samples | corpus {art.name} "
          f"({art.meta.get('content_sha256', '?')[:12]}) | winsorize={a.winsorize} | "
          f"k={art.resolve_k(ncomp)} | jobs={jobs}", file=sys.stderr)
    out = rsig_cohort(items, art, n_jobs=jobs, mode=a.winsorize, winsor_p=a.winsor_p,
                      n_components=ncomp, species=a.species, weight=a.weight,
                      on_duplicate=a.on_duplicate, named=blocks, columns=want,
                      min_clonotypes=a.min_clonotypes)
    _write(out, a.output)


def _parse_components(spec):
    if spec is None:
        return None
    try:
        return int(spec) if "." not in spec else float(spec)
    except ValueError:
        raise SystemExit(f"--components must be an integer count or a fraction in (0, 1); "
                         f"got {spec!r}") from None


def _resolve_corpus(name: str):
    """A bundled corpus name, or a path to an artifact. There is no default."""
    from vdjtools.signature.corpus import Corpus

    from mir.signature.signature import bundled_names, bundled_path

    path = bundled_path(name)
    if path is None:
        have = bundled_names()
        raise SystemExit(
            f"no corpus named {name!r}, and no artifact at that path. "
            + (f"Installed: {', '.join(have)}. " if have else
               "No corpus ships with this version yet. ")
            + "Build one with: mir corpus --corpus naive --smoke -o naive.npz")
    try:
        return Corpus.load(path)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from None


def cmd_corpus(a: argparse.Namespace) -> None:
    import time

    from mir.signature import synthesize

    if a.fetch:
        from vdjtools.signature.corpus import corpora_index, corpus_cache_dir, fetch_artifact

        from mir.signature.signature import CORPUS_REPO, _RES, bundled_names
        idx = corpora_index(_RES)
        if not idx:
            raise SystemExit("this version ships no corpus index, so there is nothing to fetch; "
                             f"the corpora it knows are {', '.join(bundled_names()) or '(none)'}")
        for nm in (sorted(idx) if a.fetch == "all" else [a.fetch]):
            if nm not in idx:
                raise SystemExit(f"no published corpus named {nm!r}; have {', '.join(sorted(idx))}")
            p = fetch_artifact(nm, sig="rsig", res_dir=_RES, repo=CORPUS_REPO)
            print(f"[mir] {p}  {p.stat().st_size / 1e6:.2f} MB", file=sys.stderr)
        print(f"[mir] cache: {corpus_cache_dir()}", file=sys.stderr)
        return
    if not a.output:
        raise SystemExit("-o/--output is required when building a corpus (--fetch does not use it)")

    size = None if a.size == "auto" else a.size
    if a.size not in ("auto", "n_eff", "p05", "p95"):
        try:
            size = int(a.size)
        except ValueError:
            raise SystemExit(f"--size must be an integer or one of auto, n_eff, p05, p95; "
                             f"got {a.size!r}") from None
    n_samples = a.samples
    if a.smoke:
        n_samples, size = 200, 1000
    ks = _parse_components(a.components)
    kw = {} if not a.loci else {"loci": tuple(a.loci.split(","))}

    t0 = time.time()
    art, _rows = synthesize(a.corpus, n_samples=n_samples, size=size, seed=a.seed,
                            n_components=ks, mode=a.winsorize, winsor_p=a.winsor_p,
                            source=a.source, species=a.species, n_jobs=a.jobs,
                            depth_spread=a.depth_spread,
                            progress=lambda loc, d, t: print(
                                f"  {loc:4s} {d}/{t}  {time.time() - t0:5.0f}s",
                                file=sys.stderr, flush=True), **kw)
    path = art.save(a.output)
    print(f"[mir] {path}  {path.stat().st_size / 1e6:.2f} MB  k={art.k}  "
          + "variance@k=" + str({loc: round(f.variance_at(f.k), 3)
                                 for loc, f in art.fits.items()})
          + f"  {time.time() - t0:.0f}s", file=sys.stderr)



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

    s = sub.add_parser(
        "signature",
        help="clonotype tables -> the geometry half of the portable signature (one row/sample)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Emits the `rsig` half of the portable repertoire signature: the clone-weighted\n"
            "prototype-sum measure Phi, rotated through a named corpus. The statistics half\n"
            "(`vsig`) comes from `vdjtools signature`; run both against the SAME corpus name\n"
            "and join on `sample_id`.\n"),
        epilog=(
            "A CORPUS IS REQUIRED. There is no default, because a signature is comparable to\n"
            "another one only if both were rotated through the same corpus -- and nothing about\n"
            "the numbers would say otherwise.\n"
            "\n"
            "  mir corpus --corpus naive --smoke -o naive.npz      # build one (no cohort needed)\n"
            "  mir signature --corpus naive.npz cohort/*.tsv.gz -o sig.parquet\n"
            "  mir signature --corpus naive.npz --components 32 --describe\n"
            "\n"
            "READ THESE BEFORE TRUSTING A ROW:\n"
            "  rsig:qc:-:winsor_frac    how much of the row the corpus's bounds clamped\n"
            "  rsig:mask:<locus>:*      present / estimable -- a FEATURE block, not diagnostics\n"
            "  rsig:div:<locus>:*       embedding diversity, carried in its own units\n"
            "A hole is nan, never 0. A locus below --min-clonotypes holes its whole div/disp\n"
            "family rather than reporting a plausible 0.0.\n"),
    )
    s.add_argument("input", nargs="*",
                   help="clonotype tables; several files of one sample_id are joined by locus")
    s.add_argument("-o", "--output", help="output .tsv/.parquet (default: stdout TSV)")
    s.add_argument("--corpus", required=True, metavar="NAME|PATH",
                   help="corpus to rotate through. REQUIRED -- no default")
    s.add_argument("--winsorize", default="features", choices=("features", "pcs", "none"),
                   help="features: clamp raw features then rotate; pcs: rotate then clamp PC "
                        "scores; none. Whatever is clamped is reported in rsig:qc:-:winsor_frac")
    s.add_argument("--winsor-p", type=float, default=None, metavar="P",
                   help="stored percentile to clamp at (0.01 or 0.05); default: as fitted")
    s.add_argument("--components", default=None, metavar="N",
                   help="truncate the rotation: an integer count, or a variance fraction. Exact; "
                        "asking for more than was fitted refuses")
    s.add_argument("--species", default="human")
    s.add_argument("--weight", default="log2p1",
                   choices=("log2p1", "log1p", "anscombe", "duplicate_count", "distinct"),
                   help="clone-size weight g")
    s.add_argument("--columns", default=None, metavar="FILE",
                   help="file of column names (one per line) to restrict the output to")
    s.add_argument("--jobs", "-j", type=int, default=1,
                   help="worker processes over samples (default: 1, 0 = all available cores). "
                        "Each spawned worker uses one kernel thread; explicit counts are "
                        "limited only by the number of samples")
    s.add_argument("--on-duplicate", choices=("error", "sum"), default="error",
                   help="a frame with no junction_nt repeating an amino-acid clonotype key cannot "
                        "say whether those rows are one clonotype or two")
    s.add_argument("--min-clonotypes", type=int, default=5, metavar="N",
                   help="a locus with fewer distinct clonotypes than this has its whole div/disp "
                        "family holed (nan), because a dispersion measured on three clonotypes is "
                        "not a measurement. The geometry itself is still computed")
    s.add_argument("--named", default=None, metavar="BLOCKS",
                   help="also emit the reportable raw blocks in their own right: 'all', or a "
                        "comma-separated list (depth,band,band_igh). Values carry their declared "
                        "transform, not a natural scale")
    s.add_argument("--describe", action="store_true",
                   help="print the columns THIS invocation emits, and exit")
    s.set_defaults(func=cmd_signature)

    c2 = sub.add_parser(
        "corpus",
        help="build a synthetic corpus and fit its rotation, bounds and scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Uses NO samples from anybody's cohort: every receptor is drawn from vdjtools'\n"
            "bundled recombination models, so the artifact is reproducible by anyone who\n"
            "installs the library.\n"),
        epilog=(
            "  mir corpus --corpus naive  -o naive.npz\n"
            "  mir corpus --corpus synthetic-blood  -o synthetic-blood.npz\n"
            "  mir corpus --corpus memory --size n_eff --components 0.95 -o memory.npz\n"
            "  mir corpus --smoke -o /tmp/smoke.npz          # minutes, not hours\n"
            "\n"
            "The build is a deterministic function of (corpus, loci, samples, size, seed,\n"
            "source) and those models, and must be byte-identical across processes and thread\n"
            "counts -- run it twice and `cmp` the output.\n"
            "\n"
            "Use the SAME name and seed as `vdjtools corpus` so the two halves describe the\n"
            "same repertoires; that is what makes joining them meaningful.\n"),
    )
    c2.add_argument("--corpus", default="naive",
                    choices=("naive", "memory", "synthetic-blood", "synthetic-tissue"),
                    help="naive: every clone size 1; memory: Zipf rank-abundance clone sizes; "
                         "synthetic-blood / synthetic-tissue: the mixture of the two, drawn across "
                         "that real cohort's measured per-locus richness, reads per clonotype and "
                         "singleton fraction")
    c2.add_argument("-o", "--output", help="artifact path; writes .npz and .json. Required "
                                          "unless --fetch, which downloads into the cache instead")
    c2.add_argument("--fetch", metavar="NAME|all",
                    help="download a published corpus instead of building one, into the cache "
                         "($VDJTOOLS_CORPUS_DIR or ~/.cache/vdjtools/signature); `all` pre-warms "
                         "every corpus, which is the install-time step since an artifact is "
                         "otherwise fetched on first use")
    c2.add_argument("--samples", type=int, default=10_000, help="repertoires in the corpus")
    c2.add_argument("--size", default="auto",
                    help="receptors per repertoire: an integer, or n_eff / p05 / p95. Default auto "
                         "is the corpus's own -- 10000 for naive/memory, the centre of the cohort's "
                         "measured richness band for a synthetic-* one")
    c2.add_argument("--components", default="128",
                    help="components per locus: an integer count, or a variance fraction")
    c2.add_argument("--winsorize", default="features", choices=("features", "pcs", "none"))
    c2.add_argument("--winsor-p", type=float, default=0.01)
    c2.add_argument("--seed", type=int, default=20260927)
    c2.add_argument("--loci", default=None, help="comma-separated subset (default: all seven)")
    c2.add_argument("--source", default="olga", choices=("olga", "learned", "arda"))
    c2.add_argument("--species", default="human")
    c2.add_argument("--depth-spread", type=float, default=None,
                    help="multiplicative depth range each repertoire's size is drawn log-uniformly "
                         "across, around --size; default is the corpus's own (2.4x-11.0x for "
                         "naive/memory, 43x on blood TRB and 259x on tissue IGH for the "
                         "synthetic-* ones). --size 3162 --depth-spread 1000 spans 100 to 100000 "
                         "receptors. Must match the vsig half to describe the same repertoires")
    c2.add_argument("--smoke", action="store_true",
                    help="reduced build (200 samples of 1000) for tests and the cmp check")
    c2.add_argument("-j", "--jobs", type=int, default=0,
                    help="worker PROCESSES across samples (not kernel threads): 0 = every core, "
                         "1 = in-process. The artifact is identical at any value")
    c2.set_defaults(func=cmd_corpus)


    return p


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default ``sys.argv[1:]``) and run the requested ``mir`` subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
