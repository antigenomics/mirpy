# mirpy -- from a folder of AIRR TSV files to one table you can join with your metadata.
#
# 2026-09-24. The whole stakeholder pipeline and nothing else: point `mir signature` at a
# directory, get one row per sample, join it to your own sheet on `sample_id`, done. Every
# other notebook here is about the method; this one is about the four commands.
#
# Data: `isalgo/airr_benchmark`, folder `sra/` -- 1,764 per-sample AIRR TSVs plus `meta.tsv`
# (PMID, Run, BioProject, Sample). Auto-downloads and caches; a local `./data_dump/` copy wins.
#
# Run with:  marimo edit examples/signature_pipeline.py
#       or:  python examples/signature_pipeline.py     (plain script, prints, no UI)
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # A folder of AIRR files to a joinable table

        You have a directory of per-sample AIRR TSVs and a metadata sheet. You want one row per
        sample, with named feature columns, that you can join to that sheet and analyse.

        That is one command:

        ```bash
        mir signature --preset classify samples/*.tsv -o sig.tsv
        ```

        `sample_id` is the file name up to the first dot, so `samples/SRR8364167.tsv` becomes
        `SRR8364167`. Name the files after the key your metadata already uses and the join needs
        no mapping table. The rest of this notebook runs exactly that on a real cohort.
        """
    )
    return


@app.cell
def _():
    import subprocess
    import sys
    import tarfile
    from pathlib import Path

    import polars as pl

    REPO = "isalgo/airr_benchmark"
    HF_FOLDER = "sra"

    def local_base(nb_dir):
        """`./data_dump/airr_benchmark/sra/` if present, else None (then we fetch)."""
        for root in (Path.cwd(), nb_dir, nb_dir.parent):
            cand = root / "data_dump" / "airr_benchmark" / HF_FOLDER
            if (cand / "meta.tsv").exists():
                return cand
        return None

    return HF_FOLDER, Path, REPO, local_base, pl, subprocess, sys, tarfile


@app.cell
def _(HF_FOLDER, Path, REPO, local_base, mo, tarfile):
    # --- fetch the cohort once, cache it ---------------------------------------------------
    _nb = mo.notebook_dir() or Path.cwd()
    work = _nb / ".data" / "signature_pipeline"
    work.mkdir(parents=True, exist_ok=True)
    samples_dir = work / "samples"

    _base = local_base(_nb)
    if _base is not None:
        meta_path = _base / "meta.tsv"
        _tar = _base / "samples.tar.gz"
    else:
        import huggingface_hub as _hub
        meta_path = Path(_hub.hf_hub_download(REPO, f"{HF_FOLDER}/meta.tsv", repo_type="dataset"))
        _tar = Path(_hub.hf_hub_download(REPO, f"{HF_FOLDER}/samples.tar.gz", repo_type="dataset"))

    if not samples_dir.exists():
        samples_dir.mkdir(parents=True)
        with tarfile.open(_tar) as t:
            t.extractall(samples_dir, filter="data")

    files = sorted(p for p in samples_dir.rglob("*.tsv"))
    mo.md(f"**{len(files)} AIRR files** in `{samples_dir.name}/`, metadata at `{meta_path.name}`.")
    return files, meta_path, samples_dir, work


@app.cell
def _(mo):
    n_samples = mo.ui.slider(10, 200, value=20, step=10,
                            label="samples to run (roughly 0.6 s each on eight cores)")
    n_samples
    return (n_samples,)


@app.cell
def _(files, mo, n_samples, subprocess, sys, work):
    # --- THE COMMAND ------------------------------------------------------------------------
    # Exactly what a stakeholder types, run through subprocess so the notebook shows the real
    # CLI rather than a Python re-implementation of it.
    subset = files[: n_samples.value]
    sig_path = work / f"sig_{len(subset)}.tsv"

    if not sig_path.exists():
        _cmd = [sys.executable, "-m", "mir.cli", "signature", "--preset", "classify",
                *[str(p) for p in subset], "-o", str(sig_path)]
        _r = subprocess.run(_cmd, capture_output=True, text=True)
        _tail = _r.stderr.strip().splitlines()[-1:] or [""]
        _msg = _tail[0]
    else:
        _msg = "cached"

    mo.md(f"""
    ```bash
    mir signature --preset classify samples/*.tsv -o sig.tsv
    ```
    {len(subset)} samples -> `{sig_path.name}`. {_msg}
    """)
    return sig_path, subset


@app.cell
def _(meta_path, mo, pl, sig_path):
    # --- THE JOIN ---------------------------------------------------------------------------
    sig = pl.read_csv(sig_path, separator="\t")
    meta = pl.read_csv(meta_path, separator="\t")
    full = sig.join(meta, left_on="sample_id", right_on="Run", how="left")

    unmatched = full["PMID"].null_count()
    mo.md(f"""
    `sig.tsv` is **{sig.height} rows x {sig.width} columns**; joined to metadata on
    `sample_id == Run` it is **{full.width} columns**, with **{unmatched} unmatched**.

    That table is the deliverable. Everything after this is your analysis.
    """)
    return full, meta, sig


@app.cell
def _(full, mo, pl):
    mo.ui.table(full.select("sample_id", "PMID", "BioProject",
                            *[c for c in full.columns if c.startswith("vsig:div:TRB")][:3]
                            ).head(10).with_columns(pl.selectors.float().round(3)))
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## What the `nan` columns mean

        A hole is never filled with zero -- `nan` means *not estimable*. Most holes are a property
        of the sample (the locus is absent, or too shallow for that estimator) and you separate
        those with the `vsig:mask:<locus>:present` / `:estimable` columns.

        One class of hole is a property of the **shipped reference**, not of your data, and it is
        worth seeing once so it is never mistaken for a bug.
        """
    )
    return


@app.cell
def _(mo, pl, sig):
    _cols = [c for c in sig.columns if c != "sample_id"]

    def _all_nan(c):
        s = sig[c]
        n = s.null_count() + (s.is_nan().sum() if s.dtype.is_float() else 0)
        return n == sig.height

    _dead = [c for c in _cols if _all_nan(c)]
    _rows = [{"block": ":".join([c.split(":")[0], c.split(":")[1], c.split(":")[3]]),
              "locus": c.split(":")[2]} for c in _dead]
    _tab = (pl.DataFrame(_rows).group_by("block").agg(pl.col("locus").sort().str.join(", "),
                                                      pl.len().alias("n"))
            .sort("n", descending=True)) if _rows else pl.DataFrame()

    mo.md(f"""
    **{len(_dead)} of {len(_cols)} columns are `nan` for every sample.** Per block:

    {mo.as_html(_tab) if _rows else "none"}

    A locus with no productive clonotype is a hole everywhere, and `vsig:mask:<locus>:present`
    says so. A locus that is present but too shallow to reach the coverage target loses its
    diversity columns only, and `vsig:mask:<locus>:estimable` says *that* -- two different facts
    that used to render as the same `nan`.

    The coverage target is a **runtime argument**, not a corpus constant: `--cstar-target` defaults
    to this cohort's own per-locus minimum attained coverage, and `vsig:cov:<locus>:cstar` reports
    what each sample actually reached. Until 4.0 the target lived in the scaling artifact, which is
    how one shipped reference came to standardise tissue samples to a level measured on blood --
    identical to 17 significant digits across all seven loci.
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Reading the result: channels

        A signature is rotated coordinates plus **channels**. The channels are the columns carried
        in their own units and never rotated -- because a provenance number mixed with the
        measurements it was supposed to qualify is no longer provenance. Read them first.
        """
    )
    return


@app.cell
def _(mo, pl, sig):
    from mir.signature import channel_columns

    _chan = [c for c in sig.columns if c in set(channel_columns("rsig"))
             or c in set(__import__("vdjtools.signature", fromlist=["channel_columns"])
                         .channel_columns("vsig"))]
    _pcs = [c for c in sig.columns if ":pc:" in c]
    _tab = pl.DataFrame({"column": _chan, "value": [sig[c][0] for c in _chan]})

    mo.md(f"""
    **{len(_pcs)} rotated columns and {len(_chan)} channels.**

    {mo.as_html(_tab.head(12))}

    The three to read before trusting any rotated value of a row:

    | channel | what it says |
    |---|---|
    | `qc:-:winsor_frac` | how much of the row the corpus's bounds clamped. Near 1.0 means this
      sample does not belong to this corpus -- not that it is unusual |
    | `cov:<locus>:cstar` | the coverage this sample actually attained, always emitted |
    | `mask:<locus>:estimable` | whether the diversity columns rest on a real estimate |

    Hand the rotated columns to `mir.explain.channel_report` with a scorer of your own to find
    which block carries your signal:

    ```python
    from mir.explain import channel_report
    rep = channel_report(X, spec, lambda B: cv_auc(B, y), base=0.5, mode="both")
    rep.best
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## The two halves

        Each tool emits its own half, against its own artifact for the **same corpus name and
        seed**. There is no joined entry point any more: each half has its own artifact, so the
        wrapper's only real job -- one scale reference over both -- no longer exists, and two calls
        plus a polars join is the whole story.

        ```bash
        vdjtools corpus    --corpus naive --smoke -o vsig_naive.npz
        mir      corpus    --corpus naive --smoke -o rsig_naive.npz
        vdjtools signature --corpus vsig_naive.npz samples/*.tsv -o vsig.tsv
        mir      signature --corpus rsig_naive.npz samples/*.tsv -o rsig.tsv
        ```

        ```python
        full = vsig_frame.join(rsig_frame, on="sample_id", how="inner")
        ```

        Column names are `<sig>:<block>:<locus>:<feature>`, with `-` for cross-locus columns. The
        two halves are disjoint by construction, so the join cannot collide -- and a matrix is
        comparable to another one only if both were rotated through the same corpus.
        """
    )
    return


if __name__ == "__main__":
    app.run()
