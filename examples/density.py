# mirpy — continuous-density TCRNET/ALICE (Theory T6).
# Reactive marimo app demonstrating graph-free neighbour enrichment in embedding space:
# background subtraction, per-clonotype significance, noise filtering, and clustering.
# Self-contained on the bundled human_TRB prototypes (an injected convergent family plays
# the role of an antigen-specific cluster) — no downloads.
# Run with:  marimo edit notebooks/density.py
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
    import polars as pl

    from mir.density import (
        denoise_and_cluster,
        enriched_mask,
        fit_density_space,
        neighbor_enrichment,
    )
    from mir.embedding.prototypes import load_prototypes
    from mir.embedding.tcremp import TCREmp
    return (TCREmp, denoise_and_cluster, enriched_mask, fit_density_space,
            load_prototypes, neighbor_enrichment, np, pl, plt)


@app.cell
def _(mo):
    mo.md(
        """
        # Continuous-density TCRNET / ALICE (Theory T6)

        TCRNET/ALICE flag antigen-driven convergent clones by counting near-identical (Hamming-1)
        neighbours and comparing to a background. `mir.density` does the same test with
        neighbour-counting in the **TCREMP embedding space** — the graph-free density ratio
        `E(z) = f_obs(z) / f_gen(z)`. Below, a synthetic convergent family injected into a diverse
        prototype pool stands in for an antigen-specific cluster; everything else is the background.
        """
    )
    return


@app.cell
def _(load_prototypes, np, pl):
    # observed = a diverse base (background-like) + an injected convergent family (1-substitution
    # variants of one seed) that plays the antigen-specific cluster; background = a disjoint pool.
    _AA = "ACDEFGHIKLMNPQRSTVWY"
    protos = load_prototypes("human", "TRB", n=3000)
    rng = np.random.default_rng(0)

    def _mut1(s):
        p = int(rng.integers(1, len(s) - 1))
        return s[:p] + _AA[int(rng.integers(20))] + s[p + 1:]

    base = protos.slice(0, 800)
    seed_row = base.row(0, named=True)
    family = [_mut1(seed_row["junction_aa"]) for _ in range(40)]
    fam_df = pl.DataFrame({"junction_aa": [seed_row["junction_aa"], *family],
                           "v_call": [seed_row["v_call"]] * 41,
                           "j_call": [seed_row["j_call"]] * 41})
    obs_df = pl.concat([base, fam_df.select(base.columns)])
    bg_df = protos.slice(800, 1600)
    is_family = np.array([False] * base.height + [True] * fam_df.height)
    return bg_df, is_family, obs_df


@app.cell
def _(TCREmp, bg_df, fit_density_space, neighbor_enrichment, obs_df):
    # one shared TCREMP -> PCA coordinate system for observed + background, then the balloon test
    model = TCREmp.from_defaults("human", "TRB", n_prototypes=1000)
    space, obs_emb, bg_emb = fit_density_space(
        model, obs_df, bg_df, n_components=20, space="full")
    res = neighbor_enrichment(obs_emb, bg_emb, lambda0=3.0)  # adaptive-bandwidth Poisson test
    return obs_emb, res, space


@app.cell
def _(is_family, mo, np, res):
    _fam_fold = float(np.median(res.fold[is_family]))
    _bg_fold = float(np.median(res.fold[~is_family]))
    _fam_q = float((res.qvalue[is_family] < 0.05).mean())
    _bg_q = float((res.qvalue[~is_family] < 0.05).mean())
    mo.md(
        f"""
        **Enrichment `E(z)` = fold** — the injected family vs the background pool:

        | group | median fold `E(z)` | q < 0.05 rate |
        |---|--:|--:|
        | injected family | {_fam_fold:.1f} | {_fam_q:.0%} |
        | background | {_bg_fold:.2f} | {_bg_q:.0%} |
        """
    )
    return


@app.cell
def _(is_family, plt, res):
    # fold enrichment per clonotype: the family sits far above the background bulk
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(range(len(res.fold)), res.fold, s=6, c=["#d62728" if f else "#bbbbbb" for f in is_family])
    ax.axhline(1.0, color="k", lw=0.7, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("clonotype")
    ax.set_ylabel("fold enrichment  E(z)")
    ax.set_title("balloon enrichment — injected family (red) vs background (grey)")
    fig.tight_layout()
    fig
    return


@app.cell
def _(denoise_and_cluster, is_family, mo, np, obs_emb, res):
    # noise filtering + clustering: keep the enriched hits, DBSCAN them into convergent groups
    labels, mask = denoise_and_cluster(obs_emb, res, alpha=0.05)
    _fam_clustered = float((labels[is_family] >= 0).mean())
    _bg_clustered = float((labels[~is_family] >= 0).mean())
    _n_clusters = len({int(x) for x in labels if x >= 0})
    mo.md(
        f"""
        **Noise filtering + clustering** (`denoise_and_cluster`): {int(mask.sum())} enriched hits →
        {_n_clusters} cluster(s). Fraction landing in a cluster: **family {_fam_clustered:.0%}**,
        background {_bg_clustered:.0%}. Background subtraction removes the naive noise; the surviving
        convergent group is the antigen-specific cluster.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## On real repertoires

        Real repertoires are *pervasively* convergent, so a generative `P_gen` background flags a
        large fraction of clones. Prefer a **biological control** (pre- vs post-vaccination, patient
        vs healthy, stimulated vs baseline) — differential enrichment cancels generic public
        convergence and isolates the antigen-specific response:

        ```python
        space, obs_emb, bg_emb = fit_density_space(model, day15_df, day0_df, n_components=20)
        res  = neighbor_enrichment(obs_emb, bg_emb, test="binomial")   # TCRNET, real control
        hits = day15_df.filter(enriched_mask(res))
        ```

        See `experiments/benchmark_density_{yfv,ankspond,tcrnet}.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
