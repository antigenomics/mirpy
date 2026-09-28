# mirpy -- from a folder of AIRR TSV files to one table you can join with your metadata.
#
# 2026-09-28. The whole pipeline and nothing else: point `mir signature` at a directory, get one
# row per sample, join it to your own sheet on `sample_id`, done. Every other notebook here is
# about the method; this one is about the commands.
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

        You have a directory of per-sample AIRR TSVs and a metadata sheet. You want **one row per
        sample**, with named feature columns, that you can join to that sheet and hand to a
        classifier or a regression.

        That is one command per half of the signature:

        ```bash
        mir      signature --corpus blood samples/*.tsv -o rsig.tsv   # geometry  (this library)
        vdjtools signature --corpus blood samples/*.tsv -o vsig.tsv   # statistics (vdjtools)
        ```

        and one join. `sample_id` is the file name up to the first dot, so `samples/SRR8364167.tsv`
        becomes `SRR8364167`. Name the files after the key your metadata already uses and the join
        needs no mapping table.

        ## The two things to understand before running it

        **What the halves measure.** The *statistics* half (`vsig`) is the classical repertoire
        summary: how diverse, how clonal, which V and J genes, what junction lengths, which
        isotypes. The *geometry* half (`rsig`) is where the receptors sit in sequence space --
        each clonotype is placed by its similarity to a fixed panel of reference receptors, and
        the sample is summarised by the clone-size-weighted average of those placements. Two
        donors can have identical diversity and still occupy different regions of sequence space,
        which is what the second half is for.

        **What `--corpus` does, and why it is required.** Raw repertoire features live on wildly
        different scales (reads in the millions, a frequency in `[0, 1]`) and are strongly
        correlated with each other and with sequencing depth. A *corpus* is a large published
        reference collection of repertoires; the signature is expressed relative to it. That makes
        your matrix and a collaborator's directly comparable, with neither of you fitting a scaler.
        There is no default corpus, because two matrices rotated through different corpora are not
        comparable and nothing about the numbers would say so. `blood` is fitted on 11,117 real
        blood samples; pick the one that matches how your samples were produced.

        The rest of this notebook runs exactly those commands on a real cohort.
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
    CORPUS = "blood"       # bulk blood samples -> the blood corpus; fetched and cached on first use
    COMPONENTS = 32        # how many rotated coordinates per locus to keep

    def local_base(nb_dir):
        """`./data_dump/airr_benchmark/sra/` if present, else None (then we fetch)."""
        for root in (Path.cwd(), nb_dir, nb_dir.parent):
            cand = root / "data_dump" / "airr_benchmark" / HF_FOLDER
            if (cand / "meta.tsv").exists():
                return cand
        return None

    return (COMPONENTS, CORPUS, HF_FOLDER, Path, REPO, local_base, pl,
            subprocess, sys, tarfile)


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
                            label="samples to run (roughly 0.4 s each)")
    n_samples
    return (n_samples,)


@app.cell
def _(COMPONENTS, CORPUS, files, mo, n_samples, subprocess, sys, work):
    # --- THE COMMAND ------------------------------------------------------------------------
    # What you would type, run through subprocess so the notebook shows the real CLI rather than
    # a Python re-implementation of it. A failure raises: a command that did not run must not
    # render as a status line.
    subset = files[: n_samples.value]
    sig_path = work / f"rsig_{len(subset)}_{CORPUS}.tsv"

    if not sig_path.exists():
        _cmd = [sys.executable, "-m", "mir.cli", "signature",
                "--corpus", CORPUS, "--components", str(COMPONENTS),
                *[str(p) for p in subset], "-o", str(sig_path)]
        _r = subprocess.run(_cmd, capture_output=True, text=True)
        if _r.returncode != 0:
            raise RuntimeError(f"mir signature failed ({_r.returncode}):\n{_r.stderr}")
        _msg = (_r.stderr.strip().splitlines() or [""])[0]
    else:
        _msg = "cached"

    mo.md(f"""
    ```bash
    mir signature --corpus {CORPUS} --components {COMPONENTS} samples/*.tsv -o rsig.tsv
    ```
    {len(subset)} samples -> `{sig_path.name}`

    ```
    {_msg}
    ```

    The stderr line is the provenance of the matrix: which corpus, that corpus's content hash,
    what was clamped, how wide the result is. Keep it with the file.
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
    `rsig.tsv` is **{sig.height} rows x {sig.width} columns**; joined to metadata on
    `sample_id == Run` it is **{full.width} columns**, with **{unmatched} unmatched**.

    That table is the deliverable. Everything after this is your analysis.
    """)
    return full, meta, sig


@app.cell
def _(full, mo, pl):
    mo.ui.table(full.select("sample_id", "PMID", "BioProject",
                            *[c for c in full.columns if c.startswith("rsig:pc:TRB")][:3]
                            ).head(10).with_columns(pl.selectors.float().round(3)))
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## What the `nan` columns mean

        A missing value is never filled with zero -- `nan` means **not estimable**, and a zero
        would be a number a model would happily learn from. Most holes are a property of the
        sample rather than of the software, and two different facts are kept apart:

        | column | says |
        |---|---|
        | `rsig:mask:<locus>:present` | the locus has at least one productive clonotype |
        | `rsig:mask:<locus>:estimable` | it has enough clonotypes for the dispersion estimators |

        A blood sample with no gamma-delta T cells has no TRG locus to measure; a sample with
        three TRG clonotypes has a locus but cannot support a variance. The first is
        `present = 0`, the second `present = 1, estimable = 0`. Both render as `nan` in the
        feature columns, and the masks are what tell them apart -- which is why the masks are
        **features in their own right**, not diagnostics: which donors sit below the floor tracks
        real lymphocyte content.
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
    **{len(_dead)} of {len(_cols)} columns are `nan` for every sample in this subset.** Per block:

    {mo.as_html(_tab) if _rows else "none"}

    Read that as a description of the cohort, not of the library. These are amplicon libraries
    with one locus amplified per run, so every other locus is legitimately absent -- and a locus
    that no sample in the subset observed is a hole for all of them.
    """)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Reading the result: channels

        A signature is **rotated coordinates plus channels**.

        The rotated coordinates (`rsig:pc:<locus>:PC01`, …) are the signature proper: correlated
        raw features re-expressed as uncorrelated axes ordered by how much variation each one
        carries across the corpus. They are the columns a model consumes, and they are not
        individually interpretable -- `PC01` is a direction, not a quantity.

        The **channels** are carried in their own units and never rotated, because a number whose
        job is to qualify a measurement must not be mixed into the measurement. Read them first.
        """
    )
    return


@app.cell
def _(mo, pl, sig):
    from mir.signature import channel_columns

    _chan = [c for c in sig.columns if c in set(channel_columns("rsig"))]
    _pcs = [c for c in sig.columns if ":pc:" in c]
    _tab = pl.DataFrame({"column": _chan, "value": [sig[c][0] for c in _chan]})

    mo.md(f"""
    **{len(_pcs)} rotated columns and {len(_chan)} channels.**

    {mo.as_html(_tab.head(12))}

    The ones to read before trusting any rotated value of a row:

    | channel | what it says |
    |---|---|
    | `rsig:qc:-:winsor_frac` | the fraction of the row the corpus's bounds clamped. A value near
      1.0 means this sample lies outside the corpus's range, so the standardisation is
      extrapolating -- pick a closer corpus rather than proceeding |
    | `rsig:mask:<locus>:present`, `:estimable` | whether the locus was there, and whether it had
      enough clonotypes to measure |
    | `rsig:div:<locus>:*` | how spread out the sample's receptors are in the embedding, in the
      embedding's own units |

    To find *which* block carries your signal, hand the matrix to `mir.explain.channel_report`
    with a scorer of your own:

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
        ## Both halves

        Each tool emits its own half against its own artifact for the **same corpus name**. There
        is no joined entry point: two commands and a polars join is the whole story.

        ```bash
        pip install vdjtools mirpy-lib

        vdjtools signature --corpus blood --components 32 samples/*.tsv -o vsig.tsv
        mir      signature --corpus blood --components 32 samples/*.tsv -o rsig.tsv
        ```

        ```python
        full = vsig_frame.join(rsig_frame, on="sample_id", how="inner")
        ```

        Column names are `<half>:<block>:<locus>:<feature>`, with `-` in the locus position for
        columns that span loci. The two halves are disjoint by construction, so the join cannot
        collide. The next cell runs both and joins them.
        """
    )
    return


@app.cell
def _(COMPONENTS, CORPUS, mo, pl, sig, subprocess, subset, sys, work):
    # --- BOTH HALVES, for real --------------------------------------------------------------
    _vsig_path = work / f"vsig_{len(subset)}_{CORPUS}.tsv"
    if not _vsig_path.exists():
        _cmd = [sys.executable, "-m", "vdjtools.cli", "signature",
                "--corpus", CORPUS, "--components", str(COMPONENTS),
                *[str(p) for p in subset], "-o", str(_vsig_path)]
        _r = subprocess.run(_cmd, capture_output=True, text=True)
        if _r.returncode != 0:
            raise RuntimeError(f"vdjtools signature failed ({_r.returncode}):\n{_r.stderr}")

    vsig = pl.read_csv(_vsig_path, separator="\t")
    both = vsig.join(sig, on="sample_id", how="inner")

    mo.md(f"""
    `vsig` is **{vsig.width} columns**, `rsig` is **{sig.width}**, and the join is
    **{both.width}** on **{both.height} samples** -- so the two halves share exactly the
    `sample_id` key and nothing else.

    That joined table is the model-ready matrix.
    """)
    return both, vsig


if __name__ == "__main__":
    app.run()
