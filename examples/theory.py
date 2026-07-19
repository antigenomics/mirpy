# mirpy — TCREMP theory validation (supplementary S1-S3).
# Reactive marimo app reproducing the embedding's claimed properties on the bundled
# human_TRB prototypes: D_ij vs d_ij correlation (T1), the Gamma / Frechet distribution
# laws (T4), and real-vs-model prototype robustness (S3).
# Run with:  marimo edit notebooks/theory.py
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from mir.bench.theory import (
        fit_distributions,
        prototype_source_correlation,
        s2_dissimilarity_distance_correlation,
    )
    from mir.embedding.prototypes import load_prototypes
    from vdjtools.model import generate, load_bundled
    return (fit_distributions, generate, load_bundled, load_prototypes, np, plt,
            prototype_source_correlation, s2_dissimilarity_distance_correlation, stats)


@app.cell
def _(mo):
    mo.md(
        """
        # TCREMP theory validation
        The prototype embedding is built so that **Euclidean distance ≈ pairwise alignment
        distance**. This notebook reproduces supplementary figures **S1–S3** on the bundled
        `human_TRB` prototypes (sequences act as their own prototypes, as in the paper).
        """
    )
    return


@app.cell
def _(mo):
    n_seq = mo.ui.slider(500, 3000, value=2000, step=250, label="# CDR3β sequences")
    n_seq
    return (n_seq,)


@app.cell
def _(load_prototypes, n_seq, s2_dissimilarity_distance_correlation):
    cdr3 = load_prototypes("human", "TRB", n=int(n_seq.value))["junction_aa"].to_list()
    res = s2_dissimilarity_distance_correlation(cdr3)  # gapblock (v3 pipeline)
    return cdr3, res


@app.cell
def _(np, plt, res):
    _rng = np.random.default_rng(0)
    _sel = _rng.choice(res.d.size, min(20000, res.d.size), replace=False)
    _fig, _ax = plt.subplots(figsize=(5.6, 5))
    _ax.scatter(res.d[_sel], res.D[_sel], s=3, alpha=0.15)
    _ax.set_xlabel("dissimilarity  $d_{ij}$")
    _ax.set_ylabel("embedding distance  $D_{ij}$")
    _ax.set_title(f"S2 / T1:  Pearson$(D, d)$ = {res.pearson:.3f}   (paper 0.56)")
    _fig
    return


@app.cell
def _(fit_distributions, np, plt, res, stats):
    _fits = fit_distributions(res.d, res.D)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4))
    _d = res.d[res.d > 0]
    _a1.hist(_d, bins=80, density=True, alpha=0.5, color="steelblue")
    _xs = np.linspace(_d.min(), _d.max(), 300)
    _a1.plot(_xs, stats.gamma.pdf(_xs, *_fits["d_gamma"]["params"]), "r", label="Gamma")
    _a1.plot(_xs, stats.norm.pdf(_xs, *_fits["d_normal"]["params"]), "k--", label="Normal")
    _a1.set_title(f"S1 / T4: $d_{{ij}}$ ~ Gamma  (KS {_fits['d_gamma']['ks']:.3f})")
    _a1.set_xlabel("$d_{ij}$"); _a1.legend()

    _D = res.D
    _a2.hist(_D, bins=80, density=True, alpha=0.5, color="darkseagreen")
    _xs2 = np.linspace(_D.min(), _D.max(), 300)
    _a2.plot(_xs2, stats.genextreme.pdf(_xs2, *_fits["D_gev"]["params"]), "r",
             label=f"GEV ξ={_fits['D_gev_xi']:+.2f}")
    _a2.plot(_xs2, stats.norm.pdf(_xs2, *_fits["D_normal"]["params"]), "k--", label="Normal")
    _a2.set_title(f"S1 / T4: $D_{{ij}}$ ~ GEV/Fréchet  (KS {_fits['D_gev']['ks']:.3f})")
    _a2.set_xlabel("$D_{ij}$"); _a2.legend()
    _fig
    return


@app.cell
def _(cdr3, generate, load_bundled, load_prototypes, mo, prototype_source_correlation):
    _N = len(cdr3)
    _model_p = generate.generate(load_bundled("TRB", source="learned"), 2 * _N, seed=42,
                                 productive_only=True)["junction_aa"] \
        .unique(maintain_order=True).to_list()[:_N]
    _query = load_prototypes("human", "TRB", n=_N + 800)["junction_aa"].to_list()[_N:_N + 600]
    _s3 = prototype_source_correlation(_query, cdr3, _model_p)
    mo.md(
        f"""
        ### S3 — prototype source robustness
        Embedding distances from **real** (Britanova-like) vs **model** (vdjtools P_gen) prototypes
        over {len(_query)} query TCRs:

        **Pearson$(D_{{real}}, D_{{model}})$ = {_s3['pearson']:.3f}**  (paper 0.96) —
        the prototype *source* barely matters.
        """
    )
    return
