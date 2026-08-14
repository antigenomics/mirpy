# mirpy — repertoire → feature vector, end to end, using named presets.
#
# The notebook to hand a collaborator who asks "how do I turn my AIRR files into a table I can
# model?". It answers, in order: which preset, what is in it, how to run it over a whole dataset
# in parallel, and how to check the result is not measuring the sequencer.
#
# Self-contained: repertoires are sampled from the bundled recombination models, so there is no
# download and no cohort. Everything shown works identically on real AIRR files.
#
# Run with:  marimo edit examples/feature_vectors.py
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import polars as pl

    from vdjtools.model import load_bundled
    from vdjtools.model.generate import generate
    from vdjtools.signature import presets
    return generate, load_bundled, np, pl, presets


@app.cell
def _(mo):
    mo.md(
        """
        # Repertoire → feature vector

        One AIRR repertoire in, one **fixed, named, positional** row out. Same columns in the same
        order for everyone, so your matrix and a collaborator's concatenate.

        Two halves, one vector: `vsig:` are repertoire **statistics** (vdjtools), `rsig:` are
        embedding **geometry** (mirpy). `mir signature` emits both; `vdjtools signature` emits the
        statistics alone if you have not installed mirpy.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. Pick a preset, not columns

        There are over 1,400 contract columns. Presets name the useful subsets and **rank** them, so
        the choice is by intent:

        * **recommended** — use unless you have a reason not to
        * *specific* — right for a stated purpose, wrong outside it
        * `avoid` — a control or a measured dead end, named so picking it is deliberate
        """
    )
    return


@app.cell
def _(presets):
    presets.table().select("preset", "rank", "columns", "halves", "scaling", "summary")
    return


@app.cell
def _(mo, presets):
    pick = mo.ui.dropdown(sorted(presets.PRESETS), value="compact", label="preset")
    pick
    return (pick,)


@app.cell
def _(mo, pick, presets):
    _p = presets.get(pick.value)
    mo.md(
        f"""
        ### `{_p.name}` — {_p.rank}, {_p.n_columns} columns

        **What it is.** {_p.summary}

        **Features.** {_p.features}

        **How they are computed.** {_p.how}

        **Use it for.** {_p.use_cases}

        **Notes.** {_p.notes or "—"}

        Suggested scaling: `{_p.scaling}`.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 2. A dataset

        Sampled from the bundled models: three donors, three loci each. Real usage points at files
        instead — `mir signature cohort/*.tsv.gz`.
        """
    )
    return


@app.cell
def _(generate, load_bundled, pl):
    def clones(locus, n, seed):
        """`generate` emits one rearrangement per row; a clonotype frame counts the duplicates."""
        g = generate(load_bundled(locus, source="olga"), n, seed=seed)
        return (g.group_by(["junction_aa", "v_call", "j_call"]).len()
                .rename({"len": "duplicate_count"})
                .with_columns((pl.col("duplicate_count")
                               / pl.col("duplicate_count").sum()).alias("frequency")))

    cohort = {f"donor{d}": {loc: clones(loc, 400, 100 * d + i)
                            for i, loc in enumerate(("TRA", "TRB", "IGH"))}
              for d in range(3)}
    {k: list(v) for k, v in cohort.items()}
    return clones, cohort


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. One call over the whole dataset

        `n_jobs=0` uses every core. Samples are independent and the frozen artifacts are read-only,
        so this is embarrassingly parallel — which matters, because per-sample cost runs from
        milliseconds on shallow blood to minutes on a deep tissue biopsy.

        `standardize="none"` here because the bundled reference is fitted for real repertoires;
        on your own data leave it at the default `"reference"`, which is what makes your numbers
        comparable with someone else's.
        """
    )
    return


@app.cell
def _(cohort, pick, presets):
    from mir.signature import assemble

    spec = presets.get(pick.value)
    X = assemble.signature_cohort(cohort, tier=spec.tier, columns=spec.columns(),
                                  standardize="none", n_jobs=0)
    X.select(X.columns[:6])
    return X, assemble, spec


@app.cell
def _(X, mo, spec):
    mo.md(
        f"""
        `{X.height} samples × {X.width - 1} columns` — the shape the preset promised.

        From the shell, the same thing:

        ```bash
        mir presets                                     # the table above
        mir presets {spec.name}                         # this preset in full
        mir signature cohort/*.tsv.gz --preset {spec.name} --threads 0 -o features.parquet
        ```

        And without mirpy installed, the statistics half only:

        ```bash
        vdjtools signature cohort/*.tsv --preset statistics --threads 0 --out vsig.parquet
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Check it is not measuring the sequencer

        The `nuisance` preset is depth, presence masks and call-quality fractions — sequencing
        protocol and nothing immunological. It is ranked `avoid` as a *feature set* precisely so it
        can be used as a **control**.

        Train your model on it. If that scores as well as your real model, your real model is
        reading library prep. This is the single most useful check in the whole package, and it
        costs one extra run.
        """
    )
    return


@app.cell
def _(cohort, presets):
    from mir.signature import assemble as _a

    floor = presets.get("nuisance")
    F = _a.signature_cohort(cohort, tier=floor.tier, columns=floor.columns(),
                            standardize="none", n_jobs=0)
    F.select(F.columns[:5])
    return F, floor


@app.cell
def _(mo):
    mo.md(
        """
        ## 5. Holes are `nan`, never 0

        A locus that was not sequenced, or a statistic the sample is too shallow to support, is
        `nan` with a `mask:` column beside it. A model that reads "absent" as "zero" reads an
        unsequenced chain as a biological finding. Most learners take `nan` natively; those that do
        not should impute **and keep the mask** as its own feature.

        ## Where the rankings come from

        A benchmark over a public multi-study AIRR corpus, scored with **study-disjoint folds** —
        fit on some studies, predict on studies the fit never saw. Under that split a column that
        merely encodes sequencing protocol scores at chance. Anyone with a comparable SRA/AIRR
        corpus can reproduce it.
        """
    )
    return


if __name__ == "__main__":
    app.run()
